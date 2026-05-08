# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

try:
    import transformer_engine as te  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # type: ignore[import-not-found]  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False


def is_linear_attention_variant(experimental_attention_variant: str) -> bool:
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants


def get_gated_delta_net_module_spec_for_backend(
    backend: BackendSpecProvider, normalization: Optional[str] = None
) -> ModuleSpec:
    """Helper function to get module spec for Linear Attention"""
    rms_norm = normalization == "RMSNorm"
    attention = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
    return attention


def get_dsa_module_spec_for_backend(
    backend: BackendSpecProvider,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert multi_latent_attention, "Currently only MLA supports sparse attention."
    assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
    assert fallback_to_eager_attn is False, "Fallback to eager attention is not supported with DSA."

    linear_q_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_kv_down_proj = (
        backend.column_parallel_linear() if mla_down_proj_use_column_parallel else backend.linear()
    )
    linear_q_up_proj = backend.column_parallel_linear()
    linear_kv_up_proj = backend.column_parallel_linear()

    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )

    # Adjust for RMS norm.
    rms_norm = normalization == "RMSNorm"
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp

    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=linear_q_down_proj,
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=linear_kv_down_proj,
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )

    return attention


def get_experimental_attention_variant_module_spec_for_backend(
    backend: BackendSpecProvider,
    sharded_state_dict_keys_map: dict,
    experimental_attention_variant: Optional[str] = None,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    mla_down_proj_use_column_parallel: Optional[bool] = False,
    normalization: Optional[str] = None,
    fallback_to_eager_attn: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for Attention"""
    if experimental_attention_variant == "gated_delta_net":
        return get_gated_delta_net_module_spec_for_backend(
            backend=backend, normalization=normalization
        )
    elif experimental_attention_variant == "dsa":
        return get_dsa_module_spec_for_backend(
            backend=backend,
            qk_layernorm=qk_layernorm,
            qk_l2_norm=qk_l2_norm,
            multi_latent_attention=multi_latent_attention,
            mla_down_proj_use_column_parallel=mla_down_proj_use_column_parallel,
            normalization=normalization,
            fallback_to_eager_attn=fallback_to_eager_attn,
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {experimental_attention_variant}"
        )


def get_experimental_attention_variant_module_spec(
    config: TransformerConfig, backend: Optional[BackendSpecProvider] = None
) -> ModuleSpec:
    """Build an attention module spec from ``config.experimental_attention_variant``.

    This companion API is used by custom model specs that need to patch the
    experimental attention spec while reusing Megatron's block construction.
    """
    if backend is None:
        backend = _get_backend_spec_provider(config)

    return get_experimental_attention_variant_module_spec_for_backend(
        backend=backend,
        sharded_state_dict_keys_map={},
        experimental_attention_variant=config.experimental_attention_variant,
        qk_layernorm=config.qk_layernorm,
        qk_l2_norm=config.qk_l2_norm,
        multi_latent_attention=config.multi_latent_attention,
        mla_down_proj_use_column_parallel=False,
        normalization=config.normalization,
        fallback_to_eager_attn=config.fallback_to_eager_attn,
    )


def get_transformer_layer_with_experimental_attention_variant_spec(
    config: TransformerConfig, backend: Optional[BackendSpecProvider] = None
) -> List[ModuleSpec]:
    """Build per-layer Transformer specs for experimental-attention GPT blocks."""
    if backend is None:
        backend = _get_backend_spec_provider(config)

    experimental_attention_pattern = [0] * config.num_layers
    if is_linear_attention_variant(config.experimental_attention_variant):
        experimental_attention_pattern = _get_linear_attention_pattern(config)
    elif config.experimental_attention_variant is not None:
        experimental_attention_pattern = [1] * config.num_layers

    experimental_attention_spec = (
        get_experimental_attention_variant_module_spec(config=config, backend=backend)
        if 1 in experimental_attention_pattern
        else None
    )
    standard_attention_spec = (
        _get_self_attention_module_spec(config=config, backend=backend)
        if 0 in experimental_attention_pattern
        else None
    )

    moe_layer_pattern = (
        _get_moe_layer_pattern(config)
        if config.num_moe_experts is not None
        else [0] * config.num_layers
    )

    moe_layer_spec = (
        _get_moe_module_spec(config=config, backend=backend)
        if 1 in moe_layer_pattern
        else None
    )
    dense_mlp_layer_spec = (
        _get_dense_mlp_module_spec(config=config, backend=backend)
        if 0 in moe_layer_pattern
        else None
    )

    rms_norm = config.normalization == "RMSNorm"
    layer_specs = []
    for layer_number in range(config.num_layers):
        attention = (
            experimental_attention_spec
            if experimental_attention_pattern[layer_number] == 1
            else standard_attention_spec
        )
        mlp = moe_layer_spec if moe_layer_pattern[layer_number] == 1 else dense_mlp_layer_spec
        input_layernorm = (
            IdentityOp
            if attention.metainfo["fuse_input_layernorm"]
            else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        )
        pre_mlp_layernorm = (
            IdentityOp
            if mlp.metainfo["fuse_pre_mlp_layernorm"]
            else backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        )

        layer_specs.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=input_layernorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=pre_mlp_layernorm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            )
        )

    return layer_specs


def get_transformer_block_with_experimental_attention_variant_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Build a TransformerBlock spec from experimental-attention layer specs."""
    backend = _get_backend_spec_provider(config)
    layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(
        config=config,
        backend=backend,
    )

    if config.pipeline_model_parallel_layout is not None:
        local_layer_ids = config.pipeline_model_parallel_layout.get_layer_id_list(
            layer_type=LayerType.decoder,
            vp_stage=vp_stage,
            pp_rank=pp_rank,
        )
    else:
        offset = get_transformer_layer_offset(
            config,
            vp_stage=vp_stage,
            pp_rank=pp_rank,
        )
        num_layers_to_build = get_num_layers_to_build(
            config,
            vp_stage=vp_stage,
            pp_rank=pp_rank,
        )
        local_layer_ids = range(offset, offset + num_layers_to_build)

    rms_norm = config.normalization == "RMSNorm"
    return TransformerBlockSubmodules(
        layer_specs=[layer_specs[layer_id] for layer_id in local_layer_ids],
        layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
    )


def _get_backend_spec_provider(config: TransformerConfig) -> BackendSpecProvider:
    assert config.transformer_impl == "transformer_engine", (
        "Experimental GPT decoder block spec only supports transformer_engine."
    )
    if config.use_kitchen:
        assert HAVE_KITCHEN
        return KitchenSpecProvider(
            fallback=TESpecProvider(fallback_to_eager_attn=config.fallback_to_eager_attn)
        )
    assert HAVE_TE
    return TESpecProvider(fallback_to_eager_attn=config.fallback_to_eager_attn)


def _get_moe_layer_pattern(config: TransformerConfig) -> List[int]:
    if isinstance(config.moe_layer_freq, int):
        return [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]
    if isinstance(config.moe_layer_freq, list):
        assert len(config.moe_layer_freq) == config.num_layers, (
            f"Invalid length of moe_layer_freq: {len(config.moe_layer_freq)}, "
            f"expected {config.num_layers}."
        )
        return config.moe_layer_freq
    raise ValueError(f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}")


def _get_linear_attention_pattern(config: TransformerConfig) -> List[int]:
    if isinstance(config.linear_attention_freq, int):
        return [
            0 if ((i + 1) % config.linear_attention_freq == 0) else 1
            for i in range(config.num_layers)
        ]
    if isinstance(config.linear_attention_freq, list):
        assert len(config.linear_attention_freq) == config.num_layers, (
            f"Invalid length of linear_attention_freq: {len(config.linear_attention_freq)}, "
            f"expected {config.num_layers}."
        )
        return config.linear_attention_freq
    if config.linear_attention_freq is None:
        if is_linear_attention_variant(config.experimental_attention_variant):
            return [1] * config.num_layers
        return [0] * config.num_layers
    raise ValueError(
        f"Invalid linear_attention_freq: {type(config.linear_attention_freq)}, "
        f"{config.linear_attention_freq}"
    )


def _get_self_attention_module_spec(
    config: TransformerConfig,
    backend: Optional[BackendSpecProvider] = None,
) -> ModuleSpec:
    if backend is None:
        backend = _get_backend_spec_provider(config)

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=config.qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        fallback_to_eager_attn=config.fallback_to_eager_attn,
    )
    attn_spec = layer_spec.submodules.self_attention
    if config.multi_latent_attention:
        attn_spec.metainfo["fuse_input_layernorm"] = False
    else:
        attn_spec.metainfo["fuse_input_layernorm"] = backend.fuse_layernorm_and_linear()
    return attn_spec


def _get_dense_mlp_module_spec(
    config: TransformerConfig,
    backend: Optional[BackendSpecProvider] = None,
) -> ModuleSpec:
    if backend is None:
        backend = _get_backend_spec_provider(config)

    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend

    mlp_spec = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=None,
        use_te_activation_func=config.use_te_activation_func,
    )
    mlp_spec.metainfo["fuse_pre_mlp_layernorm"] = backend.fuse_layernorm_and_linear()
    return mlp_spec


def _get_moe_module_spec(
    config: TransformerConfig,
    backend: Optional[BackendSpecProvider] = None,
) -> ModuleSpec:
    if backend is None:
        backend = _get_backend_spec_provider(config)

    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend

    mlp_spec = get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    mlp_spec.metainfo["fuse_pre_mlp_layernorm"] = False
    return mlp_spec
