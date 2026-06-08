# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_model import GPTModel


class _LossProbeLanguageModule(LanguageModule):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(cross_entropy_loss_fusion=False)
        self.pg_collection = SimpleNamespace(tp=None)

    def compute_language_model_loss(self, labels, logits):
        del labels
        return logits.transpose(0, 1).sum(dim=-1)


class _OutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.bias = torch.nn.Parameter(torch.randn(vocab_size))

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        output_weight = self.weight if weight is None else weight
        logits = torch.matmul(hidden, output_weight.t()) + self.bias
        return logits, None


@pytest.mark.parametrize("use_external_weight", [False, True])
def test_output_layer_loss_trains_output_weight_by_default(use_external_weight):
    language_module = _LossProbeLanguageModule()
    output_layer = _OutputLayer(hidden_size=4, vocab_size=7)
    hidden = torch.randn(3, 2, 4, requires_grad=True)
    labels = torch.zeros(2, 3, dtype=torch.long)

    external_weight = None
    col_linear_kwargs = {"runtime_gather_output": False}
    if use_external_weight:
        external_weight = torch.nn.Parameter(torch.randn(7, 4))
        col_linear_kwargs["weight"] = external_weight

    loss = language_module.compute_output_layer_and_language_model_loss(
        hidden,
        labels=labels,
        weight=external_weight,
        column_parallel_linear=output_layer,
        col_linear_kwargs=col_linear_kwargs,
    )
    loss.sum().backward()

    trained_weight = external_weight if use_external_weight else output_layer.weight
    assert trained_weight.grad is not None
    assert torch.count_nonzero(trained_weight.grad).item() > 0
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad).item() > 0


class _MTPProbeOutputLayer(torch.nn.Module):
    sequence_parallel = False

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        output_weight = self.weight if weight is None else weight
        return torch.matmul(hidden, output_weight.t()), None


class _MTPProbeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        hidden_states = kwargs["hidden_states"]
        return torch.cat([hidden_states, hidden_states + 1], dim=0)


class _MTPProbeGPTModel(GPTModel):
    def __init__(self, hidden_size=4, vocab_size=7):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=1.0,
            calculate_per_token_loss=True,
            cross_entropy_loss_fusion=False,
        )
        self.cp_group = None
        self.embedding = torch.nn.Module()
        self.mtp = _MTPProbeBlock()
        self.output_layer = _MTPProbeOutputLayer(hidden_size, vocab_size)
        self.post_process = True
        self.share_embeddings_and_output_weights = True
        self.loss_calls = []
        self.eval()

    def shared_embedding_or_output_weight(self):
        return self.output_layer.weight

    def compute_output_layer_and_language_model_loss(
        self,
        hidden,
        labels,
        weight=None,
        sequence_parallel_enabled=False,
        column_parallel_linear=None,
        col_linear_kwargs=None,
        reduction="none",
        ignore_index=-100,
    ):
        del sequence_parallel_enabled, column_parallel_linear, reduction, ignore_index
        self.loss_calls.append(
            {
                "hidden": hidden,
                "labels": labels,
                "weight": weight,
                "col_linear_kwargs": col_linear_kwargs,
            }
        )
        return torch.ones_like(labels, dtype=hidden.dtype)


def _roll_labels_for_single_mtp_layer(labels):
    labels = torch.roll(labels, shifts=-1, dims=-1)
    labels[..., -1] = 0
    labels = torch.roll(labels, shifts=-1, dims=-1)
    labels[..., -1] = 0
    return labels


def test_gpt_model_mtp_uses_native_labels_by_default():
    model = _MTPProbeGPTModel()
    hidden = torch.randn(3, 2, 4)
    labels = torch.tensor([[10, 11, 12], [20, 21, 22]])

    model._postprocess(
        hidden_states=hidden,
        input_ids=torch.zeros(2, 3, dtype=torch.long),
        position_ids=torch.zeros(2, 3, dtype=torch.long),
        labels=labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=True,
    )

    assert len(model.mtp.calls) == 1
    assert len(model.loss_calls) == 2
    assert torch.equal(model.loss_calls[0]["labels"], _roll_labels_for_single_mtp_layer(labels))
    assert torch.equal(model.loss_calls[1]["labels"], labels)
    assert model.loss_calls[0]["col_linear_kwargs"]["weight"] is model.output_layer.weight
    assert model.loss_calls[0]["col_linear_kwargs"]["weight"].requires_grad


def test_gpt_model_mtp_accepts_explicit_labels_without_native_main_loss():
    model = _MTPProbeGPTModel()
    hidden = torch.randn(3, 2, 4)
    mtp_labels = torch.tensor([[30, 31, 32], [40, 41, 42]])

    logits = model._postprocess(
        hidden_states=hidden,
        input_ids=torch.zeros(2, 3, dtype=torch.long),
        position_ids=torch.zeros(2, 3, dtype=torch.long),
        labels=None,
        mtp_labels=mtp_labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=True,
    )

    assert logits.shape == (2, 3, 7)
    assert len(model.mtp.calls) == 1
    assert len(model.loss_calls) == 1
    assert torch.equal(model.loss_calls[0]["labels"], _roll_labels_for_single_mtp_layer(mtp_labels))
    assert model.loss_calls[0]["col_linear_kwargs"]["weight"] is model.output_layer.weight
    assert model.loss_calls[0]["col_linear_kwargs"]["weight"].requires_grad
