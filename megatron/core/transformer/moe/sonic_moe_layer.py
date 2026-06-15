# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure SonicMoE module adapter for Megatron-Core model specs.

This module wraps ``sonicmoe.MoE`` directly. SonicMoE owns the router, routing
metadata generation, and fused expert computation. The adapter owns Megatron
auxiliary-loss logging/scaling and the MLP/MoE call signature.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
    get_default_pg_collection,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
    z_loss_func,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)


try:
    from sonicmoe import KernelBackendMoE, MoE as _SonicMoE
    from sonicmoe.enums import ActivationType
except ImportError as exc:
    KernelBackendMoE = None
    ActivationType = None
    _SonicMoE = torch.nn.Module
    _SONICMOE_IMPORT_ERROR = exc
else:
    _SONICMOE_IMPORT_ERROR = None


def _require_sonicmoe() -> None:
    if _SONICMOE_IMPORT_ERROR is not None:
        raise ImportError(
            "SonicMoELayer requires the optional sonic-moe package. "
            "Install it on the training machine"
        ) from _SONICMOE_IMPORT_ERROR


class _MegatronMoE(_SonicMoE):
    """Sonic MoE variant that follows Megatron gradient-accumulation fusion."""

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function,
        add_bias: bool,
        std: float,
        router_score_function: str = "softmax",
        router_score_over_topk: bool = True,
        accumulate_wgrad_into_main_grad: bool = False,
    ) -> None:
        _require_sonicmoe()
        super().__init__(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            add_bias=add_bias,
            std=std,
            router_score_function=router_score_function,
            router_score_over_topk=router_score_over_topk,
        )
        self.accumulate_wgrad_into_main_grad = accumulate_wgrad_into_main_grad


def _import_sonicmoe_functional():
    try:
        from sonicmoe.enums import ActivationType, is_glu
        from sonicmoe.functional import (
            TC_Softmax_Topk_Router_Function,
            TC_topk_router_metadata_triton,
            _DownProjection,
            _UpProjection,
        )
    except ImportError as exc:
        raise ImportError(
            "SonicMoELayer requires sonic-moe functional kernels from the optional "
            "sonic-moe package."
        ) from exc
    return (
        ActivationType,
        is_glu,
        TC_Softmax_Topk_Router_Function,
        TC_topk_router_metadata_triton,
        _DownProjection,
        _UpProjection,
    )


def _sonic_activation_type(config: TransformerConfig):
    _require_sonicmoe()
    if not config.gated_linear_unit:
        raise ValueError("SonicMoELayer only supports gated MoE MLPs.")
    activation_name = getattr(config.activation_func, "__name__", None)
    if config.activation_func is F.silu or activation_name == "silu":
        return ActivationType.SWIGLU
    if config.activation_func is F.gelu or activation_name == "gelu":
        return ActivationType.GEGLU
    raise ValueError(
        "SonicMoELayer only supports SonicMoE GLU activations SwiGLU and GEGLU. "
        f"Got activation_func={config.activation_func}."
    )


def _as_list(value: Union[str, list]) -> list:
    return value if isinstance(value, list) else [value]


def _loss_coeff(config: TransformerConfig, loss_type: str) -> float:
    routing_types = _as_list(config.moe_router_load_balancing_type)
    aux_coeffs = _as_list(config.moe_aux_loss_coeff)
    if len(aux_coeffs) == 1 and len(routing_types) > 1:
        aux_coeffs = aux_coeffs * len(routing_types)
    for routing_type, coeff in zip(routing_types, aux_coeffs):
        if routing_type == loss_type:
            return float(coeff)
    return 0.0


def _has_positive_unsupported_aux_loss(config: TransformerConfig) -> bool:
    routing_types = _as_list(config.moe_router_load_balancing_type)
    aux_coeffs = _as_list(config.moe_aux_loss_coeff)
    if len(aux_coeffs) == 1 and len(routing_types) > 1:
        aux_coeffs = aux_coeffs * len(routing_types)
    for routing_type, coeff in zip(routing_types, aux_coeffs):
        if routing_type not in ("aux_loss", "seq_aux_loss", "global_aux_loss", "none") and float(coeff) > 0.0:
            return True
    return False


def _get_tokens_per_expert_and_token_count(
    routing_map: torch.Tensor,
    reduce_group: torch.distributed.ProcessGroup,
    topk: int = None,
    with_padding_mask: bool = False,
):
    """Target-local copy of the newer MoE aux-loss token-count helper."""
    local_tokens_per_expert = routing_map.sum(dim=0)
    global_tokens_per_expert = local_tokens_per_expert
    group_size = reduce_group.size() if reduce_group is not None else 1
    if group_size > 1:
        global_tokens_per_expert = local_tokens_per_expert.clone()
        torch.distributed.all_reduce(global_tokens_per_expert, group=reduce_group)

    if with_padding_mask:
        local_num_tokens = local_tokens_per_expert.sum() / topk
        total_num_tokens = global_tokens_per_expert.sum() / topk
    else:
        local_num_tokens = routing_map.shape[0]
        total_num_tokens = local_num_tokens * group_size
    return global_tokens_per_expert, local_num_tokens, total_num_tokens


def _check_supported_config(config: TransformerConfig) -> None:
    if config.num_moe_experts is None:
        raise ValueError("SonicMoELayer requires config.num_moe_experts.")
    if config.tensor_model_parallel_size != 1:
        raise ValueError("SonicMoELayer does not integrate tensor parallelism yet.")
    if config.expert_model_parallel_size != 1:
        raise ValueError("SonicMoELayer does not integrate expert parallelism yet.")
    if config.expert_tensor_parallel_size != 1:
        raise ValueError("SonicMoELayer does not integrate expert tensor parallelism yet.")
    if getattr(config, "moe_latent_size", None) is not None:
        raise ValueError("SonicMoELayer does not support MoE latent projections.")
    if config.moe_shared_expert_intermediate_size is not None:
        raise ValueError("SonicMoELayer does not support shared experts.")
    if config.overlap_moe_expert_parallel_comm:
        raise ValueError("SonicMoELayer does not support EP communication overlap.")
    if config.fp8 or config.fp4:
        raise ValueError("SonicMoELayer does not support fp8/fp4 expert compute.")
    if config.moe_expert_capacity_factor is not None:
        raise ValueError("SonicMoELayer does not support token dropping or expert capacity.")
    if config.moe_router_padding_for_quantization:
        raise ValueError("SonicMoELayer does not support router padding for quantization.")
    if config.moe_router_score_function not in ("softmax", "sigmoid"):
        raise ValueError("SonicMoELayer pure Sonic router only supports softmax/sigmoid routing.")
    if config.moe_router_num_groups is not None or config.moe_router_group_topk is not None:
        raise ValueError("SonicMoELayer does not support group-limited routing.")
    if config.moe_router_enable_expert_bias:
        raise ValueError("SonicMoELayer does not support Megatron expert-bias routing.")
    if config.moe_router_force_load_balancing or getattr(
        config, "moe_router_force_biased", None
    ) is not None:
        raise ValueError("SonicMoELayer does not support forced benchmark routing.")
    if config.moe_input_jitter_eps is not None:
        raise ValueError("SonicMoELayer does not support Megatron router input jitter.")
    if _has_positive_unsupported_aux_loss(config):
        raise ValueError(
            "SonicMoELayer only supports aux_loss, seq_aux_loss, global_aux_loss, or no aux loss."
        )
    if getattr(config, "glu_linear_offset", 0.0) != 0.0:
        raise ValueError("SonicMoELayer does not support nonzero glu_linear_offset.")
    if getattr(config, "activation_func_clamp_value", None) is not None:
        raise ValueError("SonicMoELayer does not support activation_func_clamp_value.")
    _sonic_activation_type(config)


def _set_sonic_param_dtypes(module: torch.nn.Module, config: TransformerConfig) -> None:
    module.router.to(dtype=torch.float32)
    module.c_fc.to(dtype=config.params_dtype)
    module.c_proj.to(dtype=config.params_dtype)


def _maybe_move_to_runtime_device(module: torch.nn.Module, config: TransformerConfig) -> None:
    if not config.use_cpu_initialization and torch.cuda.is_available():
        module.to(device=torch.cuda.current_device())
    _set_sonic_param_dtypes(module, config)


class _SonicParamSync(torch.nn.Module):
    """Expose SonicMoE's directly-read params to Megatron DDP pre-hooks.

    SonicMoE's fast path does not call the ``router``, ``c_fc``, or ``c_proj``
    modules; it reads their parameters directly. Megatron's overlapped
    distributed optimizer waits for param all-gathers in module forward
    pre-hooks, so this no-op module is called immediately before SonicMoE.
    """

    def __init__(self, sonic_moe: torch.nn.Module) -> None:
        super().__init__()
        self.register_parameter("router_weight", sonic_moe.router.weight)
        self.register_parameter("c_fc_weight", sonic_moe.c_fc.weight)
        self.register_parameter("c_proj_weight", sonic_moe.c_proj.weight)
        if sonic_moe.c_fc.bias is not None:
            self.register_parameter("c_fc_bias", sonic_moe.c_fc.bias)
        if sonic_moe.c_proj.bias is not None:
            self.register_parameter("c_proj_bias", sonic_moe.c_proj.bias)

    def forward(self) -> None:
        return None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        del destination, prefix, keep_vars

    # pylint: disable=arguments-differ
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


class SonicMoELayer(MoELayer):
    """Megatron-compatible wrapper around pure ``sonicmoe.MoE``.

    This class subclasses ``MoELayer`` so ``TransformerLayer`` treats it as a
    MoE block and forwards MoE-only kwargs. It deliberately bypasses Megatron
    token dispatchers and therefore requires TP/EP/ETP sizes to be 1.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        del submodules
        _check_supported_config(config)
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        BaseMoELayer.__init__(
            self, config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.tp_group = pg_collection.tp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp
        self.layer_number = layer_number
        self.is_mtp = False
        self._aux_loss_coeff = _loss_coeff(config, "aux_loss")
        self._seq_aux_loss_coeff = _loss_coeff(config, "seq_aux_loss")
        self._global_aux_loss_coeff = _loss_coeff(config, "global_aux_loss")
        self._z_loss_coeff = config.moe_z_loss_coeff

        if self._global_aux_loss_coeff > 0.0:
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            self.register_buffer(
                "global_tokens_per_expert",
                torch.zeros(config.num_moe_experts, dtype=torch.float32, device=device),
                persistent=False,
            )
            self.register_buffer(
                "ga_steps",
                torch.tensor(0, dtype=torch.float32, device=device),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        _require_sonicmoe()
        self.kernel_backend_moe = KernelBackendMoE.sonicmoe
        self.sonic_moe = _MegatronMoE(
            num_experts=config.num_moe_experts,
            num_experts_per_tok=config.moe_router_topk,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            activation_function=_sonic_activation_type(config),
            add_bias=config.add_bias_linear,
            std=config.init_method_std,
            router_score_function=config.moe_router_score_function,
            router_score_over_topk=self._score_over_topk(),
            accumulate_wgrad_into_main_grad=config.gradient_accumulation_fusion,
        )
        _maybe_move_to_runtime_device(self.sonic_moe, config)
        self.sonic_param_sync = _SonicParamSync(self.sonic_moe)

        for param in self.sonic_moe.parameters():
            setattr(param, "allreduce", True)

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number

    def set_is_mtp(self):
        self.is_mtp = True

    def reset_global_aux_loss_tracker(self):
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def _num_layers_for_loss_tracker(self) -> int:
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        return num_layers

    def _attach_scaled_loss(
        self,
        output: torch.Tensor,
        loss: torch.Tensor,
        coeff: float,
        name: str,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_group_has_dp: bool = False,
        valid_token_count: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if coeff == 0.0:
            return output

        save_to_aux_losses_tracker(
            name,
            loss / coeff,
            self.layer_number,
            self._num_layers_for_loss_tracker(),
            reduce_group=reduce_group,
        )
        if self.config.calculate_per_token_loss:
            num_tokens = valid_token_count if valid_token_count is not None else output.shape[0]
            loss = loss * num_tokens
        return MoEAuxLossAutoScaler.apply(output, loss)

    def _needs_router_losses(self) -> bool:
        return (
            self.training
            and torch.is_grad_enabled()
            and (
                self._aux_loss_coeff > 0.0
                or self._seq_aux_loss_coeff > 0.0
                or self._global_aux_loss_coeff > 0.0
                or (self._z_loss_coeff is not None and self._z_loss_coeff != 0.0)
            )
        )

    def _tokens_per_expert_and_count(
        self, local_tokens_per_expert: torch.Tensor, reduce_group: torch.distributed.ProcessGroup
    ):
        global_tokens_per_expert = local_tokens_per_expert.float()
        if reduce_group.size() > 1:
            global_tokens_per_expert = global_tokens_per_expert.clone()
            torch.distributed.all_reduce(global_tokens_per_expert, group=reduce_group)
        local_num_tokens = local_tokens_per_expert.sum() / self.config.moe_router_topk
        total_num_tokens = global_tokens_per_expert.sum() / self.config.moe_router_topk
        return global_tokens_per_expert, local_num_tokens, total_num_tokens

    def _score_over_topk(self) -> bool:
        return not self.config.moe_router_pre_softmax

    def _apply_sonic_router_topk(
        self,
        router_func,
        router_logits: torch.Tensor,
        num_experts: int,
    ):
        args = (
            router_logits,
            num_experts,
            self.config.moe_router_topk,
            self._score_over_topk(),
            False,
        )
        if self.config.moe_router_score_function == "sigmoid":
            return router_func.apply(*args, "sigmoid")
        try:
            return router_func.apply(*args, "softmax")
        except TypeError:
            return router_func.apply(*args)

    def _normalize_topk_scores(self, topk_scores: torch.Tensor) -> torch.Tensor:
        if (
            self.config.moe_router_score_function == "sigmoid"
            and self.config.moe_router_topk > 1
        ):
            topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-20)
        if self.config.moe_router_topk_scaling_factor is not None:
            topk_scores = topk_scores * self.config.moe_router_topk_scaling_factor
        return topk_scores

    def _scores_for_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.config.moe_router_score_function == "softmax":
            return F.softmax(router_logits, dim=-1, dtype=torch.float32)
        if self.config.moe_router_score_function == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            return scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        raise ValueError(
            f"Invalid score_function: {self.config.moe_router_score_function}"
        )

    def _sonic_tc_forward(
        self,
        hidden_states: torch.Tensor,
        is_inference_mode_enabled: bool = False,
    ):
        (
            ActivationType,
            is_glu,
            TC_Softmax_Topk_Router_Function,
            TC_topk_router_metadata_triton,
            _DownProjection,
            _UpProjection,
        ) = _import_sonicmoe_functional()

        original_shape = hidden_states.shape
        x = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(x.float(), self.sonic_moe.router.weight)
        num_experts = self.sonic_moe.router.weight.size(0)
        topk_scores, topk_indices = self._apply_sonic_router_topk(
            TC_Softmax_Topk_Router_Function,
            router_logits,
            num_experts,
        )
        topk_scores = self._normalize_topk_scores(topk_scores)

        num_tokens, topk = topk_indices.size()
        num_routed_tokens = num_tokens * topk
        device = topk_indices.device

        s_scatter_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
        expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)

        TC_topk_router_metadata_triton(
            topk_indices,
            num_experts,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        activation_type = self.sonic_moe.activation_function
        if type(activation_type) == str:
            activation_type = ActivationType(activation_type)

        assert not torch.compiler.is_compiling()
        assert is_glu(activation_type), "SonicMoELayer only supports GLU Sonic kernels."

        w1 = self.sonic_moe.c_fc.weight.permute(1, 2, 0)
        w2 = self.sonic_moe.c_proj.weight.permute(1, 2, 0)
        accumulate_wgrad_into_main_grad = getattr(
            self.sonic_moe, "accumulate_wgrad_into_main_grad", False
        )
        a, h = _UpProjection.apply(
            x,
            w1,
            self.sonic_moe.c_fc.bias,
            expert_frequency_offset,
            num_routed_tokens,
            topk,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,
            activation_type,
            is_inference_mode_enabled,
            True,
            accumulate_wgrad_into_main_grad,
        )

        output = _DownProjection.apply(
            a,
            h,
            w2,
            self.sonic_moe.c_proj.bias,
            topk_scores,
            expert_frequency_offset,
            num_tokens,
            topk,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,
            activation_type,
            accumulate_wgrad_into_main_grad,
        )

        return output.view(original_shape), router_logits, expert_frequency

    def _compute_seq_aux_inputs(self, router_logits: torch.Tensor):
        routing_map, scores = compute_routing_scores_for_aux_loss(
            router_logits,
            self.config.moe_router_topk,
            self.config.moe_router_score_function,
            fused=False,
        )
        return routing_map, scores

    def _apply_router_losses(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not torch.is_grad_enabled():
            return output
        if (
            self._aux_loss_coeff == 0.0
            and self._seq_aux_loss_coeff == 0.0
            and self._global_aux_loss_coeff == 0.0
            and self._z_loss_coeff is None
        ):
            return output

        if self._z_loss_coeff is not None and self._z_loss_coeff != 0.0:
            moe_z_loss_coeff = self._z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(router_logits, moe_z_loss_coeff)
            output = self._attach_scaled_loss(
                output,
                z_loss,
                moe_z_loss_coeff,
                "z_loss",
                valid_token_count=router_logits.shape[0],
            )

        scores = None
        if self._aux_loss_coeff > 0.0:
            scores = self._scores_for_aux_loss(router_logits)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                self._tokens_per_expert_and_count(tokens_per_expert, self.tp_cp_group)
            )
            aux_loss = switch_load_balancing_loss_func(
                probs=scores,
                tokens_per_expert=global_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.config.moe_router_topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=self._aux_loss_coeff,
                fused=False,
            )
            output = self._attach_scaled_loss(
                output,
                aux_loss,
                self._aux_loss_coeff,
                "load_balancing_loss",
                reduce_group=self.tp_cp_group,
                valid_token_count=local_num_tokens,
            )

        if self._seq_aux_loss_coeff > 0.0:
            if scores is None:
                scores = self._scores_for_aux_loss(router_logits)
            routing_map, _ = self._compute_seq_aux_inputs(router_logits)
            if hidden_states.dim() >= 3:
                seq_length, bsz = hidden_states.shape[0], hidden_states.shape[1]
            else:
                seq_length, bsz = hidden_states.shape[0], 1
            seq_scores = scores.reshape(seq_length, -1)
            seq_routing_map = routing_map.reshape(seq_length, -1)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                _get_tokens_per_expert_and_token_count(
                    routing_map=seq_routing_map,
                    reduce_group=self.tp_cp_group,
                    topk=self.config.moe_router_topk * bsz,
                )
            )
            seq_aux_loss = (
                switch_load_balancing_loss_func(
                    probs=seq_scores,
                    tokens_per_expert=global_tokens_per_expert,
                    total_num_tokens=total_num_tokens,
                    topk=self.config.moe_router_topk,
                    num_experts=self.config.num_moe_experts,
                    moe_aux_loss_coeff=self._seq_aux_loss_coeff,
                    fused=False,
                )
                / bsz
            )
            output = self._attach_scaled_loss(
                output,
                seq_aux_loss,
                self._seq_aux_loss_coeff,
                "seq_load_balancing_loss",
                reduce_group=self.tp_cp_group,
                valid_token_count=local_num_tokens,
            )

        if self._global_aux_loss_coeff > 0.0:
            if scores is None:
                scores = self._scores_for_aux_loss(router_logits)
            global_tokens_per_expert, local_num_tokens, total_num_tokens = (
                self._tokens_per_expert_and_count(tokens_per_expert, self.tp_dp_cp_group)
            )
            self.global_tokens_per_expert += global_tokens_per_expert
            self.ga_steps += 1
            averaged_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps
            global_aux_loss = switch_load_balancing_loss_func(
                probs=scores,
                tokens_per_expert=averaged_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.config.moe_router_topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=self._global_aux_loss_coeff,
                fused=False,
            )
            output = self._attach_scaled_loss(
                output,
                global_aux_loss,
                self._global_aux_loss_coeff,
                "global_load_balancing_loss",
                reduce_group=self.tp_dp_cp_group,
                reduce_group_has_dp=True,
                valid_token_count=local_num_tokens,
            )

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        del input_ids
        if intermediate_tensors is not None:
            raise ValueError("SonicMoELayer does not support partial MoE CUDA graph replay.")
        if padding_mask is not None and torch.any(padding_mask):
            raise ValueError("Pure SonicMoE routing does not support padding masks.")

        self.sonic_param_sync()
        need_router_losses = self._needs_router_losses()
        output, router_logits, tokens_per_expert = self._sonic_tc_forward(
            hidden_states,
            is_inference_mode_enabled=(not self.training),
        )
        if need_router_losses:
            output = self._apply_router_losses(
                output, hidden_states, router_logits, tokens_per_expert
            )
        return output, None

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        del routed_experts, shared_experts

    def set_for_recompute_pre_mlp_layernorm(self):
        raise ValueError(
            "SonicMoELayer does not support fp8/fp4 pre-MLP layernorm recompute."
        )

    def _state_tensor(self, tensor: torch.Tensor, keep_vars: bool) -> torch.Tensor:
        return tensor if keep_vars else tensor.detach()

    def _origin_grouped_fc1_weight(self, keep_vars: bool) -> torch.Tensor:
        weight = self._state_tensor(self.sonic_moe.c_fc.weight, keep_vars)
        return (
            weight.transpose(1, 2)
            .contiguous()
            .view(
                self.config.hidden_size,
                self.config.num_moe_experts * 2 * self.config.moe_ffn_hidden_size,
            )
        )

    def _origin_grouped_fc2_weight(self, keep_vars: bool) -> torch.Tensor:
        weight = self._state_tensor(self.sonic_moe.c_proj.weight, keep_vars)
        return (
            weight.transpose(1, 2)
            .contiguous()
            .view(
                self.config.num_moe_experts * self.config.moe_ffn_hidden_size,
                self.config.hidden_size,
            )
        )

    def _origin_state_dict_entries(self, prefix: str, keep_vars: bool) -> dict:
        entries = {
            f"{prefix}router.weight": self._state_tensor(
                self.sonic_moe.router.weight, keep_vars
            ),
            f"{prefix}experts.weight1": self._origin_grouped_fc1_weight(keep_vars),
            f"{prefix}experts.weight2": self._origin_grouped_fc2_weight(keep_vars),
        }
        if self.sonic_moe.c_fc.bias is not None:
            entries[f"{prefix}experts.linear_fc1.bias"] = self._state_tensor(
                self.sonic_moe.c_fc.bias, keep_vars
            )
        if self.sonic_moe.c_proj.bias is not None:
            entries[f"{prefix}experts.linear_fc2.bias"] = self._state_tensor(
                self.sonic_moe.c_proj.bias, keep_vars
            )
        return entries

    def _remove_sonic_state_dict_entries(self, state_dict, prefix: str) -> None:
        for key in (
            "sonic_moe.router.weight",
            "sonic_moe.c_fc.weight",
            "sonic_moe.c_fc.bias",
            "sonic_moe.c_proj.weight",
            "sonic_moe.c_proj.bias",
            "sonic_param_sync.router_weight",
            "sonic_param_sync.c_fc_weight",
            "sonic_param_sync.c_fc_bias",
            "sonic_param_sync.c_proj_weight",
            "sonic_param_sync.c_proj_bias",
        ):
            state_dict.pop(f"{prefix}{key}", None)

    # pylint: disable=arguments-differ
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        self._remove_sonic_state_dict_entries(state_dict, prefix)
        state_dict.update(self._origin_state_dict_entries(prefix, keep_vars))
        return state_dict

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self._origin_state_dict_entries("", keep_vars=True),
            prefix,
            {},
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def _move_if_present(self, state_dict, src_key: str, dst_key: str) -> None:
        if dst_key not in state_dict and src_key in state_dict:
            state_dict[dst_key] = state_dict.pop(src_key)

    def _stack_per_expert_if_present(
        self, state_dict, prefix: str, src_prefix: str, dst_key: str
    ) -> None:
        if dst_key in state_dict:
            return
        keys = [
            f"{prefix}{src_prefix}{expert_idx}"
            for expert_idx in range(self.config.num_moe_experts)
        ]
        if all(key in state_dict for key in keys):
            state_dict[dst_key] = torch.stack([state_dict.pop(key) for key in keys], dim=0)

    def _merge_legacy_grouped_mlp_state_dict(self, state_dict, prefix: str) -> None:
        weight1_key = f"{prefix}experts.weight1"
        weight2_key = f"{prefix}experts.weight2"
        if f"{prefix}sonic_moe.c_fc.weight" not in state_dict and weight1_key in state_dict:
            weight1 = state_dict.pop(weight1_key)
            state_dict[f"{prefix}sonic_moe.c_fc.weight"] = (
                weight1.view(
                    self.config.num_moe_experts,
                    self.config.hidden_size,
                    2 * self.config.moe_ffn_hidden_size,
                )
                .transpose(1, 2)
                .contiguous()
            )
        if f"{prefix}sonic_moe.c_proj.weight" not in state_dict and weight2_key in state_dict:
            weight2 = state_dict.pop(weight2_key)
            state_dict[f"{prefix}sonic_moe.c_proj.weight"] = (
                weight2.view(
                    self.config.num_moe_experts,
                    self.config.moe_ffn_hidden_size,
                    self.config.hidden_size,
                )
                .transpose(1, 2)
                .contiguous()
            )

    # pylint: disable=arguments-differ
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        self._move_if_present(
            state_dict, f"{prefix}router.weight", f"{prefix}sonic_moe.router.weight"
        )
        state_dict.pop(f"{prefix}router.bias", None)

        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc1.weight", f"{prefix}sonic_moe.c_fc.weight"
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc1.bias", f"{prefix}sonic_moe.c_fc.bias"
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc2.weight", f"{prefix}sonic_moe.c_proj.weight"
        )
        self._move_if_present(
            state_dict, f"{prefix}experts.linear_fc2.bias", f"{prefix}sonic_moe.c_proj.bias"
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.linear_fc1.weight",
            f"{prefix}sonic_moe.c_fc.weight",
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.linear_fc1.bias",
            f"{prefix}sonic_moe.c_fc.bias",
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.linear_fc2.weight",
            f"{prefix}sonic_moe.c_proj.weight",
        )
        self._stack_per_expert_if_present(
            state_dict,
            prefix,
            "experts.linear_fc2.bias",
            f"{prefix}sonic_moe.c_proj.bias",
        )
        self._merge_legacy_grouped_mlp_state_dict(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def get_sonic_moe_module_spec() -> ModuleSpec:
    """Return an MLP/MoE ModuleSpec for use in a TransformerLayer spec."""
    return ModuleSpec(
        module=SonicMoELayer,
        metainfo={"fuse_pre_mlp_layernorm": False},
    )


def replace_moe_layer_specs_with_sonic_moe(transformer_layer_spec) -> int:
    """Replace Megatron MoELayer MLP specs with SonicMoELayer specs in-place.

    Accepts a TransformerBlockSubmodules object, a TransformerLayer ModuleSpec,
    or a list/tuple of layer specs. Returns the number of MoE MLP specs that
    use SonicMoELayer after the replacement.
    """
    if transformer_layer_spec is None:
        return 0
    if isinstance(transformer_layer_spec, (list, tuple)):
        return sum(
            replace_moe_layer_specs_with_sonic_moe(spec) for spec in transformer_layer_spec
        )

    layer_specs = getattr(transformer_layer_spec, "layer_specs", None)
    if layer_specs is not None:
        return replace_moe_layer_specs_with_sonic_moe(layer_specs)

    submodules = getattr(transformer_layer_spec, "submodules", None)
    if submodules is None:
        return 0

    mlp_spec = getattr(submodules, "mlp", None)
    module = getattr(mlp_spec, "module", None)
    if module is SonicMoELayer:
        return 1
    if module is not MoELayer:
        return 0

    submodules.mlp = get_sonic_moe_module_spec()
    return 1
