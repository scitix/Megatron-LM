import importlib.util
from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import MCoreLoadPlanner
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel.p2p_communication import _batched_p2p_ops
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer


def test_checkpoint_shape_validation_fails_on_missing_key():
    metadata = SimpleNamespace(state_dict_metadata={})
    sharded_tensor = SimpleNamespace(key="missing.weight")

    with pytest.raises(KeyError, match="missing.weight"):
        MCoreLoadPlanner._validate_global_shapes(None, metadata, [sharded_tensor])


def test_huggingface_tokenizer_honors_trust_remote_code(monkeypatch):
    calls = []

    class _Tokenizer:
        eos_token_id = 0

        def get_vocab(self):
            return {"<eos>": 0}

        def __len__(self):
            return 1

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return _Tokenizer()

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    _HuggingFaceTokenizer("model-a", trust_remote_code=False)
    _HuggingFaceTokenizer("model-b", trust_remote_code=True)

    assert calls[0][1]["trust_remote_code"] is False
    assert calls[1][1]["trust_remote_code"] is True


def test_batched_p2p_ops_passes_pipeline_group(monkeypatch):
    captured = []

    class _P2POp:
        def __init__(self, op, tensor, peer, group):
            captured.append((op, tensor, peer, group))

    monkeypatch.setattr(torch.distributed, "P2POp", _P2POp)
    monkeypatch.setattr(torch.distributed, "isend", object())
    monkeypatch.setattr(torch.distributed, "irecv", object())
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", lambda ops: ops)

    group = object()
    tensor = torch.zeros(1)

    _batched_p2p_ops(
        tensor_send_prev=tensor,
        tensor_recv_prev=tensor,
        tensor_send_next=tensor,
        tensor_recv_next=tensor,
        group=group,
        prev_pipeline_rank=1,
        next_pipeline_rank=2,
    )

    assert [item[3] for item in captured] == [group, group, group, group]


def test_moe_submodules_exposes_generic_router_extension_point():
    class CustomRouter(TopKRouter):
        pass

    assert MoESubmodules().router is TopKRouter

    legacy_positional = MoESubmodules("experts", "shared_experts")

    assert legacy_positional.experts == "experts"
    assert legacy_positional.shared_experts == "shared_experts"
    assert legacy_positional.router is TopKRouter

    submodules = MoESubmodules(router=ModuleSpec(module=CustomRouter))

    assert submodules.router.module is CustomRouter


def test_dp_reshardable_optimizer_state_keeps_step_local():
    class _Group:
        def rank(self):
            return 0

        def size(self):
            return 1

    class _Bucket:
        numel_unpadded = 4
        grad_data = torch.zeros(4)

    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.data_parallel_group = _Group()
    optimizer.data_parallel_group_idx = 0
    optimizer.distributed_optimizer_instance_id = 0
    optimizer.gbuf_ranges = [None]
    optimizer.buffers = [SimpleNamespace(buckets=[_Bucket()])]
    bucket_state = [
        {
            "gbuf_local_start": 0,
            "gbuf_local_end": 4,
            "exp_avg": torch.zeros(4),
            "step": torch.tensor(3.0),
        }
    ]
    optimizer.get_parameter_state_dp_reshardable = lambda: {
        "per_bucket_numel": [4],
        "per_bucket_numel_unpadded": [4],
        0: {torch.float32: [bucket_state]},
    }

    state = DistributedOptimizer.sharded_param_state_dp_reshardable(
        optimizer, model_sharded_state_dict={}
    )

    tensor_state = state[0][torch.float32][0][0]
    assert isinstance(tensor_state["exp_avg"], ShardedTensor)
    assert isinstance(tensor_state["step"], LocalNonpersistentObject)
    assert tensor_state["step"].obj.shape == torch.Size([])


def test_megatron_stable_has_no_trainer_owned_routing_replay_module():
    assert importlib.util.find_spec("megatron.core.transformer.moe.routing_replay") is None
