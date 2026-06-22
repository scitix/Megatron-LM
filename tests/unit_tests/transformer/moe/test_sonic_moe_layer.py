# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import os
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_default_pg_collection,
    get_moe_layer_wise_logging_tracker,
    router_gating_linear,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.sonic_moe_layer import (
    SonicMoELayer,
    replace_moe_layer_specs_with_sonic_moe,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

EXACT_RTOL = 0.0
EXACT_ATOL = 0.0
ROUTER_KERNEL_RTOL = 1.0e-4
ROUTER_KERNEL_ATOL = 3.0e-5
FORWARD_RTOL = 1.0e-2
FORWARD_ATOL = 1.0e-5
EP2_DCP_PREFIX = "decoder.layers.0.mlp."


def _require_sonicmoe():
    pytest.importorskip("sonicmoe")


def _ep2_dcp_expected_tensors(config, device):
    router = (
        torch.arange(
            config.num_moe_experts * config.hidden_size,
            dtype=torch.float32,
            device=device,
        ).view(config.num_moe_experts, config.hidden_size)
        / 100.0
    )
    fc1 = torch.arange(
        config.num_moe_experts * 2 * config.moe_ffn_hidden_size * config.hidden_size,
        dtype=torch.float32,
        device=device,
    ).view(config.num_moe_experts, 2 * config.moe_ffn_hidden_size, config.hidden_size)
    fc2 = torch.arange(
        config.num_moe_experts * config.hidden_size * config.moe_ffn_hidden_size,
        dtype=torch.float32,
        device=device,
    ).view(config.num_moe_experts, config.hidden_size, config.moe_ffn_hidden_size)
    return router, fc1.to(dtype=config.params_dtype), fc2.to(dtype=config.params_dtype)


def _ep2_dcp_sharded_state(config, rank, world_size, device, with_data):
    router, fc1, fc2 = _ep2_dcp_expected_tensors(config, device)
    if not with_data:
        router = torch.empty_like(router)
        fc1 = torch.empty_like(fc1)
        fc2 = torch.empty_like(fc2)

    local_experts = config.num_moe_experts // world_size
    start = rank * local_experts
    end = start + local_experts
    fc1_gate, fc1_up = torch.chunk(fc1[start:end], 2, dim=1)

    return {
        f"{EP2_DCP_PREFIX}router.weight": ShardedTensor.from_rank_offsets(
            f"{EP2_DCP_PREFIX}router.weight",
            router,
            replica_id=(0, 0, rank),
        ),
        f"{EP2_DCP_PREFIX}experts.experts.linear_fc1.weight.gate": (
            ShardedTensor.from_rank_offsets(
                f"{EP2_DCP_PREFIX}experts.experts.linear_fc1.weight",
                fc1_gate.contiguous(),
                (0, rank, world_size),
                (1, 0, 2),
                replica_id=(0, 0, 0),
            )
        ),
        f"{EP2_DCP_PREFIX}experts.experts.linear_fc1.weight.up": (
            ShardedTensor.from_rank_offsets(
                f"{EP2_DCP_PREFIX}experts.experts.linear_fc1.weight",
                fc1_up.contiguous(),
                (0, rank, world_size),
                (1, 1, 2),
                replica_id=(0, 0, 0),
            )
        ),
        f"{EP2_DCP_PREFIX}experts.experts.linear_fc2.weight": ShardedTensor.from_rank_offsets(
            f"{EP2_DCP_PREFIX}experts.experts.linear_fc2.weight",
            fc2[start:end].contiguous(),
            (0, rank, world_size),
            (2, 0, 1),
            replica_id=(0, 0, 0),
        ),
    }


class TestSonicMoELayerRouterLoss:
    def setup_method(self, method):
        _require_sonicmoe()
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        clear_aux_losses_tracker()

        self.default_config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            num_moe_experts=8,
            moe_ffn_hidden_size=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=0.7,
            moe_z_loss_coeff=0.3,
            moe_router_score_function="softmax",
            moe_router_dtype="fp32",
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

    def teardown_method(self, method):
        clear_aux_losses_tracker()
        Utils.destroy_model_parallel()

    def _new_topk_router(self, config):
        router = TopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda()
        router.set_layer_number(0)
        router.weight.data = router.weight.data.float()
        return router

    def _new_sonic_layer(self, config):
        layer = SonicMoELayer(config=config, pg_collection=get_default_pg_collection()).cuda()
        layer.set_layer_number(0)
        return layer

    def _new_original_sequential_moe_layer(self, config):
        spec = get_moe_module_spec(
            use_te=False,
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=False,
        )
        layer = MoELayer(
            config=config,
            submodules=spec.submodules,
            pg_collection=get_default_pg_collection(),
        ).cuda()
        layer.set_layer_number(0)
        return layer

    def test_replace_moe_layer_specs_with_sonic_moe(self):
        config = dataclasses.replace(
            self.default_config,
            num_layers=2,
            moe_layer_freq=[1, 0],
        )

        block_spec = get_gpt_decoder_block_spec(
            config=config,
            use_transformer_engine=False,
            normalization=config.normalization,
        )

        assert block_spec.layer_specs[0].submodules.mlp.module is MoELayer
        assert block_spec.layer_specs[1].submodules.mlp.module is MLP
        assert replace_moe_layer_specs_with_sonic_moe(block_spec) == 1
        assert block_spec.layer_specs[0].submodules.mlp.module is SonicMoELayer
        assert block_spec.layer_specs[1].submodules.mlp.module is MLP
        assert replace_moe_layer_specs_with_sonic_moe(block_spec) == 1

    def test_sonic_param_and_main_grad_dtypes(self):
        layer = self._new_sonic_layer(self.default_config)

        assert type(layer.sonic_moe).__name__ == "_MegatronMoE"
        assert (
            type(layer.sonic_moe).__module__
            == "megatron.core.transformer.moe.sonic_moe_layer"
        )
        assert (
            layer.sonic_moe.accumulate_wgrad_into_main_grad
            is self.default_config.gradient_accumulation_fusion
        )
        assert layer.sonic_moe.router.weight.dtype is torch.float32
        assert layer.sonic_moe.c_fc.weight.dtype is torch.bfloat16
        assert layer.sonic_moe.c_proj.weight.dtype is torch.bfloat16
        if layer.sonic_moe.c_fc.bias is not None:
            assert layer.sonic_moe.c_fc.bias.dtype is torch.bfloat16
        if layer.sonic_moe.c_proj.bias is not None:
            assert layer.sonic_moe.c_proj.bias.dtype is torch.bfloat16

        wrapped_layer = Float16Module(self.default_config, self._new_sonic_layer(self.default_config))
        assert wrapped_layer.module.sonic_moe.router.weight.dtype is torch.float32
        assert wrapped_layer.module.sonic_moe.c_fc.weight.dtype is torch.bfloat16
        assert wrapped_layer.module.sonic_moe.c_proj.weight.dtype is torch.bfloat16

        ddp = DistributedDataParallel(
            self.default_config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            module=layer,
            disable_bucketing=True,
        )
        for name, param in ddp.module.named_parameters():
            assert param.main_grad.dtype is torch.float32, name

    def test_megatron_moe_accumulates_expert_wgrad_into_main_grad(self):
        config = dataclasses.replace(
            self.default_config,
            moe_z_loss_coeff=None,
            moe_aux_loss_coeff=0.0,
            gradient_accumulation_fusion=True,
        )
        layer = self._new_sonic_layer(config)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            module=layer,
            disable_bucketing=True,
        )
        ddp.zero_grad_buffer()
        hidden_state = torch.randn(
            (8192, 1, config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        output, bias = ddp(hidden_state)
        assert bias is None
        output.float().square().mean().backward()

        for name, param in (
            ("c_fc", ddp.module.sonic_moe.c_fc.weight),
            ("c_proj", ddp.module.sonic_moe.c_proj.weight),
        ):
            assert param.grad_added_to_main_grad is True, name
            assert param.grad is None, name
            assert param.main_grad.dtype is torch.float32, name
            assert torch.count_nonzero(param.main_grad).item() > 0, name

    @pytest.mark.parametrize("moe_router_pre_softmax", [False, True])
    def test_softmax_router_flags_follow_megatron_config(self, moe_router_pre_softmax):
        config = dataclasses.replace(
            self.default_config,
            moe_router_pre_softmax=moe_router_pre_softmax,
            moe_router_topk_scaling_factor=1.5 if moe_router_pre_softmax else None,
            moe_z_loss_coeff=None,
            moe_aux_loss_coeff=0.0,
        )
        layer = self._new_sonic_layer(config)

        assert layer._score_over_topk() is (not moe_router_pre_softmax)
        assert layer.sonic_moe.router_score_function == "softmax"
        assert layer.sonic_moe.router_score_over_topk is (not moe_router_pre_softmax)

    @pytest.mark.parametrize("moe_router_pre_softmax", [False, True])
    def test_sigmoid_router_flags_follow_megatron_config(self, moe_router_pre_softmax):
        config = dataclasses.replace(
            self.default_config,
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=moe_router_pre_softmax,
            moe_router_topk_scaling_factor=1.5,
            moe_z_loss_coeff=None,
            moe_aux_loss_coeff=0.0,
        )
        layer = self._new_sonic_layer(config)

        assert layer.sonic_moe.router_score_function == "sigmoid"
        assert layer.sonic_moe.router_score_over_topk is (not moe_router_pre_softmax)
        assert layer._score_over_topk() is (not moe_router_pre_softmax)

    @pytest.mark.parametrize(
        "score_over_topk",
        [False, True],
        ids=["topk_over_sigmoid", "sigmoid_over_topk"],
    )
    @pytest.mark.parametrize("num_experts, topk", [(8, 2), (128, 8)])
    def test_sigmoid_router_kernel_matches_reference_pre_and_post_topk(
        self, score_over_topk, num_experts, topk
    ):
        from sonicmoe.functional import TC_Softmax_Topk_Router_Function

        config = dataclasses.replace(
            self.default_config,
            num_moe_experts=num_experts,
            moe_router_topk=topk,
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=not score_over_topk,
            moe_router_topk_scaling_factor=None,
        )
        router_logits = self._make_margin_separated_router_logits(
            8192, config.num_moe_experts
        )

        topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
            router_logits,
            config.num_moe_experts,
            config.moe_router_topk,
            score_over_topk,
            False,
            "sigmoid",
        )

        ref_logits = router_logits.detach().clone().requires_grad_(True)
        if score_over_topk:
            ref_topk = ref_logits.topk(config.moe_router_topk, dim=-1)
            ref_scores = ref_topk.values.sigmoid()
        else:
            ref_topk = ref_logits.sigmoid().topk(config.moe_router_topk, dim=-1)
            ref_scores = ref_topk.values

        assert torch.equal(topk_indices, ref_topk.indices.to(dtype=topk_indices.dtype))
        torch.testing.assert_close(
            topk_scores, ref_scores, rtol=ROUTER_KERNEL_RTOL, atol=ROUTER_KERNEL_ATOL
        )

        grad = torch.randn_like(topk_scores)
        topk_scores.backward(grad)
        ref_scores.backward(grad)
        torch.testing.assert_close(
            router_logits.grad,
            ref_logits.grad,
            rtol=ROUTER_KERNEL_RTOL,
            atol=ROUTER_KERNEL_ATOL,
        )

    @staticmethod
    def _make_margin_separated_router_logits(num_tokens: int, num_experts: int):
        expert_scores = torch.linspace(-2.0, 2.0, num_experts, device="cuda")
        row_offsets = torch.arange(num_tokens, device="cuda").unsqueeze(1)
        expert_offsets = torch.arange(num_experts, device="cuda").unsqueeze(0)
        rotated_experts = (expert_offsets + row_offsets) % num_experts
        logits = expert_scores[rotated_experts]
        logits = logits + 1.0e-4 * torch.randn_like(logits)
        return logits.requires_grad_(True)

    @pytest.mark.parametrize("moe_router_score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("moe_router_pre_softmax", [False, True])
    @pytest.mark.parametrize("num_accumulation_steps", [1, 2, 4, 8])
    def test_z_loss_and_global_aux_loss_match_original_router(
        self, num_accumulation_steps, moe_router_pre_softmax, moe_router_score_function
    ):
        config = dataclasses.replace(
            self.default_config,
            moe_router_score_function=moe_router_score_function,
            moe_router_pre_softmax=moe_router_pre_softmax,
            moe_router_topk_scaling_factor=1.5 if moe_router_pre_softmax else None,
        )
        router = self._new_topk_router(config)
        sonic_layer = self._new_sonic_layer(config)
        with torch.no_grad():
            sonic_layer.sonic_moe.router.weight.copy_(router.weight)

        hidden_states = [
            torch.randn((8192, 1, config.hidden_size), device="cuda", dtype=torch.bfloat16)
            for _ in range(num_accumulation_steps)
        ]

        for hidden_state in hidden_states:
            ref = self._run_original_router(router, hidden_state)
            sonic = self._run_sonic_loss_path(sonic_layer, hidden_state)

            torch.testing.assert_close(
                sonic["z_loss"], ref["z_loss"], rtol=EXACT_RTOL, atol=EXACT_ATOL
            )
            torch.testing.assert_close(
                sonic["global_aux_loss"],
                ref["global_aux_loss"],
                rtol=EXACT_RTOL,
                atol=EXACT_ATOL,
            )
            assert sonic["router_weight_grad"].dtype is torch.float32
            torch.testing.assert_close(
                sonic["router_weight_grad"],
                ref["router_weight_grad"],
                rtol=EXACT_RTOL,
                atol=EXACT_ATOL,
            )
            torch.testing.assert_close(
                sonic["hidden_state_grad"],
                ref["hidden_state_grad"],
                rtol=EXACT_RTOL,
                atol=EXACT_ATOL,
            )
            torch.testing.assert_close(
                sonic_layer.global_tokens_per_expert,
                router.global_tokens_per_expert,
                rtol=EXACT_RTOL,
                atol=EXACT_ATOL,
            )
            torch.testing.assert_close(
                sonic_layer.ga_steps, router.ga_steps, rtol=EXACT_RTOL, atol=EXACT_ATOL
            )

        sonic_layer.reset_global_aux_loss_tracker()
        router.reset_global_aux_loss_tracker()
        torch.testing.assert_close(
            sonic_layer.global_tokens_per_expert,
            router.global_tokens_per_expert,
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )
        torch.testing.assert_close(
            sonic_layer.ga_steps, router.ga_steps, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )

    def test_sigmoid_end_to_end_forward_backward(self):
        config = dataclasses.replace(
            self.default_config,
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=False,
            moe_router_topk_scaling_factor=None,
        )
        layer = self._new_sonic_layer(config)
        hidden_state = torch.randn(
            (8192, 1, config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        output, bias = layer(hidden_state)
        assert bias is None
        assert output.shape == hidden_state.shape
        assert output.dtype is torch.bfloat16
        assert torch.isfinite(output).all()

        output.float().square().mean().backward()

        tracker = get_moe_layer_wise_logging_tracker()
        assert "z_loss" in tracker
        assert "global_load_balancing_loss" in tracker
        assert hidden_state.grad.dtype is torch.bfloat16
        assert torch.isfinite(hidden_state.grad).all()
        assert layer.sonic_moe.router.weight.grad.dtype is torch.float32
        assert torch.isfinite(layer.sonic_moe.router.weight.grad).all()
        assert layer.sonic_moe.c_fc.weight.grad.dtype is torch.bfloat16
        assert torch.isfinite(layer.sonic_moe.c_fc.weight.grad).all()
        assert layer.sonic_moe.c_proj.weight.grad.dtype is torch.bfloat16
        assert torch.isfinite(layer.sonic_moe.c_proj.weight.grad).all()

    @pytest.mark.parametrize("moe_router_score_function", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("moe_router_pre_softmax", [False, True])
    def test_forward_matches_original_moe_layer(
        self, moe_router_pre_softmax, moe_router_score_function
    ):
        config = dataclasses.replace(
            self.default_config,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_z_loss_coeff=None,
            moe_router_score_function=moe_router_score_function,
            moe_router_pre_softmax=moe_router_pre_softmax,
            moe_router_topk_scaling_factor=1.5 if moe_router_pre_softmax else None,
            moe_token_dispatcher_type="allgather",
            bias_activation_fusion=False,
        )
        original_layer = self._new_original_sequential_moe_layer(config)
        sonic_layer = self._new_sonic_layer(config)
        origin_state = self._origin_grouped_moe_state_dict(
            config, expert_weight_scale=config.init_method_std
        )
        self._set_margin_separated_router_weight(origin_state["router.weight"], config)
        self._load_origin_grouped_state_into_sequential_moe(
            original_layer, origin_state, config
        )
        load_result = sonic_layer.load_state_dict(origin_state, strict=True)
        assert load_result.missing_keys == []
        assert load_result.unexpected_keys == []

        original_layer.eval()
        sonic_layer.eval()
        hidden_state = self._make_forward_hidden_state(config)

        with torch.no_grad():
            original_output, original_bias = original_layer(hidden_state)
            sonic_output, sonic_bias = sonic_layer(hidden_state)

        assert original_bias is None
        assert sonic_bias is None
        assert sonic_output.shape == original_output.shape == hidden_state.shape
        assert sonic_output.dtype is torch.bfloat16
        torch.testing.assert_close(
            sonic_output,
            original_output,
            rtol=FORWARD_RTOL,
            atol=FORWARD_ATOL,
        )

    def test_load_origin_grouped_moe_state_and_save_origin_format(self):
        config = dataclasses.replace(
            self.default_config,
            moe_z_loss_coeff=None,
            moe_aux_loss_coeff=0.0,
        )
        origin_state = self._origin_grouped_moe_state_dict(config)
        layer = self._new_sonic_layer(config)

        load_result = layer.load_state_dict(origin_state, strict=True)
        assert load_result.missing_keys == []
        assert load_result.unexpected_keys == []

        expected_fc1 = origin_state["experts.weight1"].view(
            config.num_moe_experts,
            config.hidden_size,
            2 * config.moe_ffn_hidden_size,
        ).transpose(1, 2).contiguous()
        expected_fc2 = origin_state["experts.weight2"].view(
            config.num_moe_experts,
            config.moe_ffn_hidden_size,
            config.hidden_size,
        ).transpose(1, 2).contiguous()
        torch.testing.assert_close(
            layer.sonic_moe.router.weight,
            origin_state["router.weight"],
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )
        torch.testing.assert_close(
            layer.sonic_moe.c_fc.weight, expected_fc1, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        torch.testing.assert_close(
            layer.sonic_moe.c_proj.weight, expected_fc2, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )

        saved_state = layer.state_dict()
        assert set(saved_state.keys()) == {
            "router.weight",
            "experts.weight1",
            "experts.weight2",
        }
        assert not any(key.startswith("sonic_moe.") for key in saved_state)
        assert not any(key.startswith("sonic_param_sync.") for key in saved_state)
        torch.testing.assert_close(
            saved_state["router.weight"],
            origin_state["router.weight"],
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )
        torch.testing.assert_close(
            saved_state["experts.weight1"],
            origin_state["experts.weight1"],
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )
        torch.testing.assert_close(
            saved_state["experts.weight2"],
            origin_state["experts.weight2"],
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )

        prefixed_state = layer.state_dict(prefix="mlp.")
        assert set(prefixed_state.keys()) == {
            "mlp.router.weight",
            "mlp.experts.weight1",
            "mlp.experts.weight2",
        }
        torch.testing.assert_close(
            prefixed_state["mlp.experts.weight1"],
            origin_state["experts.weight1"],
            rtol=EXACT_RTOL,
            atol=EXACT_ATOL,
        )

        sharded_state = layer.sharded_state_dict(prefix="mlp.")
        assert set(sharded_state.keys()) == {
            "mlp.router.weight",
            "mlp.experts.weight1",
            "mlp.experts.weight2",
        }

        roundtrip_layer = self._new_sonic_layer(config)
        roundtrip_load_result = roundtrip_layer.load_state_dict(saved_state, strict=True)
        assert roundtrip_load_result.missing_keys == []
        assert roundtrip_load_result.unexpected_keys == []
        roundtrip_state = roundtrip_layer.state_dict()
        assert set(roundtrip_state.keys()) == set(origin_state.keys())
        for key, tensor in saved_state.items():
            assert roundtrip_state[key].shape == tensor.shape
            assert roundtrip_state[key].dtype == tensor.dtype
            torch.testing.assert_close(
                roundtrip_state[key], tensor, rtol=EXACT_RTOL, atol=EXACT_ATOL
            )

    def test_load_ep2_distributed_checkpoint(self, tmp_path):
        if Utils.world_size != 2:
            pytest.skip("Run with torchrun --nproc_per_node=2 to exercise EP=2 DCP.")
        if torch.cuda.device_count() < 2:
            pytest.skip("EP=2 checkpoint test requires at least 2 CUDA devices.")

        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            ffn_hidden_size=16,
            num_moe_experts=4,
            moe_ffn_hidden_size=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_z_loss_coeff=None,
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        rank = Utils.rank
        world_size = Utils.world_size
        device = torch.device("cuda", torch.cuda.current_device())
        shared_tmp_path = [str(tmp_path) if rank == 0 else None]
        dist.broadcast_object_list(shared_tmp_path, src=0)
        src_ckpt_dir = os.path.join(shared_tmp_path[0], "source_ep2_dcp")
        sonic_ckpt_dir = os.path.join(shared_tmp_path[0], "sonic_saved_dcp")

        if rank == 0:
            for ckpt_dir in (src_ckpt_dir, sonic_ckpt_dir):
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                os.makedirs(ckpt_dir, exist_ok=True)
        dist.barrier()

        src_ep2_state = _ep2_dcp_sharded_state(
            config, rank, world_size, device, with_data=True
        )
        dist_checkpointing.save(
            src_ep2_state, src_ckpt_dir, validate_access_integrity=False
        )
        dist.barrier()

        layer = SonicMoELayer(config=config, pg_collection=get_default_pg_collection()).cuda()
        layer.set_layer_number(0)
        loaded = dist_checkpointing.load(
            layer.sharded_state_dict(prefix=EP2_DCP_PREFIX),
            src_ckpt_dir,
            validate_access_integrity=False,
            strict="raise_all",
        )
        load_result = layer.load_state_dict(
            {
                key.removeprefix(EP2_DCP_PREFIX): value
                for key, value in loaded.items()
                if key.startswith(EP2_DCP_PREFIX)
            },
            strict=True,
        )
        assert load_result.missing_keys == []
        assert load_result.unexpected_keys == []

        router, fc1, fc2 = _ep2_dcp_expected_tensors(
            config, torch.device("cuda", torch.cuda.current_device())
        )
        torch.testing.assert_close(
            layer.sonic_moe.router.weight, router, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        torch.testing.assert_close(
            layer.sonic_moe.c_fc.weight, fc1, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        torch.testing.assert_close(
            layer.sonic_moe.c_proj.weight, fc2, rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        assert layer.sonic_moe.router.weight.dtype is torch.float32

        dist_checkpointing.save(
            layer.sharded_state_dict(prefix=EP2_DCP_PREFIX),
            sonic_ckpt_dir,
            validate_access_integrity=False,
        )
        dist.barrier()

        src_ep2_loaded = dist_checkpointing.load(
            _ep2_dcp_sharded_state(config, rank, world_size, device, with_data=False),
            src_ckpt_dir,
            validate_access_integrity=False,
            strict="raise_all",
        )
        sonic_ep2_loaded = dist_checkpointing.load(
            _ep2_dcp_sharded_state(config, rank, world_size, device, with_data=False),
            sonic_ckpt_dir,
            validate_access_integrity=False,
            strict="raise_all",
        )
        assert set(src_ep2_loaded.keys()) == set(sonic_ep2_loaded.keys())
        for key, tensor in src_ep2_loaded.items():
            torch.testing.assert_close(
                sonic_ep2_loaded[key], tensor, rtol=EXACT_RTOL, atol=EXACT_ATOL
            )

    @staticmethod
    def _origin_grouped_moe_state_dict(config, expert_weight_scale: float = 1.0):
        weight1_shape = (
            config.hidden_size,
            config.num_moe_experts * 2 * config.moe_ffn_hidden_size,
        )
        weight2_shape = (
            config.num_moe_experts * config.moe_ffn_hidden_size,
            config.hidden_size,
        )
        return {
            "router.weight": torch.randn(
                config.num_moe_experts,
                config.hidden_size,
                device="cuda",
                dtype=torch.float32,
            ),
            "experts.weight1": torch.randn(
                weight1_shape, device="cuda", dtype=config.params_dtype
            )
            * expert_weight_scale,
            "experts.weight2": torch.randn(
                weight2_shape, device="cuda", dtype=config.params_dtype
            )
            * expert_weight_scale,
        }

    @staticmethod
    def _set_margin_separated_router_weight(router_weight, config):
        assert config.hidden_size >= config.num_moe_experts
        router_weight.zero_()
        router_weight[:, : config.num_moe_experts].copy_(
            torch.eye(
                config.num_moe_experts,
                device=router_weight.device,
                dtype=router_weight.dtype,
            )
        )

    @staticmethod
    def _load_origin_grouped_state_into_sequential_moe(layer, origin_state, config):
        with torch.no_grad():
            layer.router.weight.data = origin_state["router.weight"].detach().clone()
            fc1_by_expert = origin_state["experts.weight1"].view(
                config.num_moe_experts,
                config.hidden_size,
                2 * config.moe_ffn_hidden_size,
            )
            fc2_by_expert = origin_state["experts.weight2"].view(
                config.num_moe_experts,
                config.moe_ffn_hidden_size,
                config.hidden_size,
            )
            for expert_idx, expert in enumerate(layer.experts.local_experts):
                expert.linear_fc1.weight.copy_(
                    fc1_by_expert[expert_idx].transpose(0, 1).contiguous()
                )
                expert.linear_fc2.weight.copy_(
                    fc2_by_expert[expert_idx].transpose(0, 1).contiguous()
                )

    def _make_forward_hidden_state(self, config):
        hidden_state = torch.randn(
            (8192, 1, config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.1
        router_logits = self._make_margin_separated_router_logits(
            hidden_state.shape[0], config.num_moe_experts
        ).detach()
        hidden_state[:, 0, : config.num_moe_experts] = router_logits.to(
            dtype=hidden_state.dtype
        )
        return hidden_state

    def _run_original_router(self, router, hidden_state):
        clear_aux_losses_tracker()
        router.weight.grad = None
        hidden_state = hidden_state.detach().clone().requires_grad_(True)

        scores, _ = router(hidden_state)
        scores.backward(torch.zeros_like(scores))

        tracker = get_moe_layer_wise_logging_tracker()
        return {
            "z_loss": tracker["z_loss"]["values"][0].detach().clone(),
            "global_aux_loss": tracker["global_load_balancing_loss"]["values"][0]
            .detach()
            .clone(),
            "router_weight_grad": router.weight.grad.detach().clone(),
            "hidden_state_grad": hidden_state.grad.detach().clone(),
        }

    def _run_sonic_loss_path(self, sonic_layer, hidden_state):
        clear_aux_losses_tracker()
        sonic_layer.sonic_moe.router.weight.grad = None
        hidden_state = hidden_state.detach().clone().requires_grad_(True)

        router_logits = router_gating_linear(
            hidden_state,
            sonic_layer.sonic_moe.router.weight,
            sonic_layer.sonic_moe.router.bias,
            sonic_layer._router_dtype(hidden_state),
        ).view(
            -1,
            sonic_layer.config.num_moe_experts,
        )
        tokens_per_expert = self._tokens_per_expert(router_logits, sonic_layer.config)
        output = torch.zeros_like(hidden_state, requires_grad=True)

        output = sonic_layer._apply_router_losses(
            output, hidden_state, router_logits, tokens_per_expert
        )
        output.backward(torch.zeros_like(output))

        tracker = get_moe_layer_wise_logging_tracker()
        return {
            "z_loss": tracker["z_loss"]["values"][0].detach().clone(),
            "global_aux_loss": tracker["global_load_balancing_loss"]["values"][0]
            .detach()
            .clone(),
            "router_weight_grad": sonic_layer.sonic_moe.router.weight.grad.detach().clone(),
            "hidden_state_grad": hidden_state.grad.detach().clone(),
        }

    @staticmethod
    def _tokens_per_expert(router_logits, config):
        if config.moe_router_score_function == "sigmoid":
            routing_scores = torch.sigmoid(router_logits.float())
        elif config.moe_router_pre_softmax:
            routing_scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        else:
            routing_scores = router_logits
        selected_experts = routing_scores.topk(config.moe_router_topk, dim=-1).indices
        return selected_experts.flatten().bincount(
            minlength=config.num_moe_experts
        ).to(dtype=torch.int32)
