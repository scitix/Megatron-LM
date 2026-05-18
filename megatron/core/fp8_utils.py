# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions related to FP8 that are used throughout Megatron core"""

import importlib
import json
import math
import os
import weakref
from contextlib import nullcontext
from functools import wraps
from typing import List, Optional, Union

import torch

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_te_version, is_te_min_version

# Check if Transformer Engine is installed
HAVE_TE = False
try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False

# Check if Transformer Engine has class for fp8 tensors.
HAVE_TE_FP8_TENSOR_CLASS = False
if HAVE_TE:
    if is_te_min_version("2.0"):
        # In TE2.x, QuantizedTensor is the base class for all different type of fp8 tensors,
        # including fp8 tensor for delayed scaling, current scaling and mxfp8, etc.
        from transformer_engine.pytorch.tensor import QuantizedTensor as FP8_TENSOR_CLASS
    else:
        from transformer_engine.pytorch.float8_tensor import Float8Tensor as FP8_TENSOR_CLASS

    HAVE_TE_FP8_TENSOR_CLASS = True
else:
    HAVE_TE_FP8_TENSOR_CLASS = False
    FP8_TENSOR_CLASS = None

# Check if Transformer Engine has MXFP8Tensor class

try:
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_TE_MXFP8TENSOR = True
except (ImportError, ModuleNotFoundError):
    # MXFP8Tensor not found
    HAVE_TE_MXFP8TENSOR = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )

    TE_LINEAR_TYPES = (
        TELinear,
        TEColumnParallelLinear,
        TERowParallelLinear,
        TELayerNormColumnParallelLinear,
    )
else:
    TE_LINEAR_TYPES = ()

try:
    from megatron.core.extensions.transformer_engine import Fp8Padding, Fp8Unpadding
except ImportError:
    Fp8Padding = None
    Fp8Unpadding = None

try:
    from transformer_engine.pytorch.tensor.utils import (
        post_all_gather_processing as te_post_all_gather_processing,
    )
except ImportError:
    te_post_all_gather_processing = None


def _maybe_fake_mxfp4_expert_qat_main_param_shard(
    model_param: torch.Tensor,
    main_param: Optional[torch.Tensor],
    start_offset: Optional[int],
) -> Optional[torch.Tensor]:
    if main_param is None:
        return None
    if os.getenv("OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG", "0") != "1":
        return main_param
    if not getattr(model_param, "mcore_mxfp4_expert_qat_weight", False):
        return main_param

    block_size = int(os.getenv("OPEN_TRAINING_MXFP4_BLOCK_SIZE", "32"))
    if block_size <= 0:
        raise RuntimeError(
            "OPEN_TRAINING_MXFP4_BLOCK_SIZE must be a positive integer when "
            "OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG=1."
        )

    weight_shape = getattr(
        model_param, "mcore_mxfp4_expert_qat_shape", tuple(model_param.shape)
    )
    if len(weight_shape) != 2 or weight_shape[-1] % block_size != 0:
        raise RuntimeError(
            "MXFP4 expert QAT with fp8_param_gather requires a 2D expert weight "
            f"whose K dimension is divisible by block size {block_size}; got {weight_shape}."
        )
    if start_offset is None:
        raise RuntimeError(
            "MXFP4 expert QAT with fp8_param_gather requires shard start offsets."
        )
    if start_offset % block_size != 0 or main_param.numel() % block_size != 0:
        raise RuntimeError(
            "MXFP4 expert QAT with fp8_param_gather requires optimizer shards "
            f"aligned to group-{block_size} boundaries; got start_offset={start_offset}, "
            f"numel={main_param.numel()}."
        )

    from megatron.core.extensions.transformer_engine import fake_mxfp4_quantization_ste

    fake_main_param = fake_mxfp4_quantization_ste(
        main_param.view(-1, block_size), block_size
    ).view_as(main_param)
    if hasattr(main_param, "main_grad"):
        fake_main_param.main_grad = main_param.main_grad
    return fake_main_param


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)


def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 tensor to a higher precision tensor."""
    if is_te_min_version("2.0"):
        return fp8_tensor.dequantize()
    else:
        return fp8_tensor.from_float8()


_MXFP4_QAT_FP8_COPY_TRACE_REMAINING = None
_MXFP4_QAT_FP8_COPY_TRACE_CALL_INDEX = 0
_MXFP4_QAT_FP8_COPY_TRACE_RECORD_INDEX = 0
_MXFP4_QAT_FP8_COPY_TRACE_COUNTS_BY_CALL: dict[int, int] = {}
_MXFP4_E2M1_POS_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _mxfp4_qat_fp8_copy_trace_remaining() -> int:
    global _MXFP4_QAT_FP8_COPY_TRACE_REMAINING
    if _MXFP4_QAT_FP8_COPY_TRACE_REMAINING is None:
        raw_limit = os.getenv("SIRL_DSV4_QAT_FP8_COPY_TRACE_LIMIT", "")
        _MXFP4_QAT_FP8_COPY_TRACE_REMAINING = int(raw_limit) if raw_limit else 0
    return _MXFP4_QAT_FP8_COPY_TRACE_REMAINING


def _decrement_mxfp4_qat_fp8_copy_trace_remaining() -> None:
    global _MXFP4_QAT_FP8_COPY_TRACE_REMAINING
    _MXFP4_QAT_FP8_COPY_TRACE_REMAINING = max(
        0, _mxfp4_qat_fp8_copy_trace_remaining() - 1
    )


def _next_mxfp4_qat_fp8_copy_trace_call_index() -> int:
    global _MXFP4_QAT_FP8_COPY_TRACE_CALL_INDEX
    _MXFP4_QAT_FP8_COPY_TRACE_CALL_INDEX += 1
    return _MXFP4_QAT_FP8_COPY_TRACE_CALL_INDEX


def _mxfp4_qat_fp8_copy_trace_limit_per_call() -> Optional[int]:
    raw_limit = os.getenv("SIRL_DSV4_QAT_FP8_COPY_TRACE_LIMIT_PER_CALL", "")
    return int(raw_limit) if raw_limit else None


def _mxfp4_qat_fp8_copy_trace_sample_elems() -> int:
    raw_limit = os.getenv("SIRL_DSV4_QAT_FP8_COPY_TRACE_SAMPLE_ELEMS", "")
    return int(raw_limit) if raw_limit else 0


def _evenly_spaced_indices(numel: int, limit: int) -> List[int]:
    if numel <= 0 or limit <= 0:
        return []
    if limit >= numel:
        return list(range(numel))
    if limit == 1:
        return [0]
    return sorted({round(index * (numel - 1) / (limit - 1)) for index in range(limit)})


def _mxfp4_e2m1_code(value: float) -> int:
    mag = abs(value)
    code = 0
    for boundary in _MXFP4_E2M1_BOUNDARIES:
        if mag > boundary:
            code += 1
        elif mag == boundary and code % 2 == 1:
            code += 1
        else:
            break
    return min(code, len(_MXFP4_E2M1_POS_GRID) - 1)


def _nearest_mxfp4_boundary_distance(value: float) -> dict[str, float]:
    mag = abs(value)
    boundary = min(_MXFP4_E2M1_BOUNDARIES, key=lambda candidate: abs(mag - candidate))
    return {
        "nearest_e2m1_boundary": boundary,
        "nearest_e2m1_boundary_distance": abs(mag - boundary),
    }


def _sample_mxfp4_qat_fp8_copy_boundary(
    *,
    raw_main_param: torch.Tensor,
    fake_main_param: torch.Tensor,
    actual: torch.Tensor,
    block_size: int,
    sample_elems: int,
) -> List[dict[str, object]]:
    raw_flat = raw_main_param.detach().float().reshape(-1)
    fake_flat = fake_main_param.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if raw_flat.numel() != fake_flat.numel() or raw_flat.numel() != actual_flat.numel():
        raise RuntimeError(
            "MXFP4 QAT FP8-copy sample trace requires raw/fake/actual numel equality: "
            f"raw={raw_flat.numel()} fake={fake_flat.numel()} actual={actual_flat.numel()}"
        )
    if raw_flat.numel() % block_size != 0:
        raise RuntimeError(
            "MXFP4 QAT FP8-copy sample trace requires shard numel aligned to block size: "
            f"numel={raw_flat.numel()} block_size={block_size}"
        )

    samples: List[dict[str, object]] = []
    for flat_index in _evenly_spaced_indices(raw_flat.numel(), sample_elems):
        group_start = (flat_index // block_size) * block_size
        group_end = group_start + block_size
        raw_value = float(raw_flat[flat_index].cpu())
        fake_value = float(fake_flat[flat_index].cpu())
        actual_value = float(actual_flat[flat_index].cpu())
        group_absmax = float(raw_flat[group_start:group_end].abs().max().cpu())
        scale_exponent = math.ceil(math.log2(max(group_absmax / 6.0, 1e-8)))
        scale_exponent = min(127, max(-127, scale_exponent))
        scale_value = float(2.0 ** scale_exponent)
        raw_over_scale = raw_value / scale_value
        fake_over_scale = fake_value / scale_value
        actual_over_scale = actual_value / scale_value
        samples.append(
            {
                "flat_index": int(flat_index),
                "group_start": int(group_start),
                "group_end": int(group_end),
                "block_size": int(block_size),
                "scale_exponent": int(scale_exponent),
                "scale_value": scale_value,
                "group_absmax": group_absmax,
                "raw_value": raw_value,
                "fake_value": fake_value,
                "actual_value": actual_value,
                "raw_over_scale": raw_over_scale,
                "fake_over_scale": fake_over_scale,
                "actual_over_scale": actual_over_scale,
                "raw_e2m1_code": _mxfp4_e2m1_code(raw_over_scale),
                "fake_e2m1_code": _mxfp4_e2m1_code(fake_over_scale),
                **_nearest_mxfp4_boundary_distance(raw_over_scale),
            }
        )
    return samples


def _should_record_mxfp4_qat_fp8_copy_trace(copy_call_index: int) -> bool:
    per_call_limit = _mxfp4_qat_fp8_copy_trace_limit_per_call()
    if per_call_limit is None:
        return _mxfp4_qat_fp8_copy_trace_remaining() > 0

    count = _MXFP4_QAT_FP8_COPY_TRACE_COUNTS_BY_CALL.get(copy_call_index, 0)
    return count < per_call_limit


def _mark_recorded_mxfp4_qat_fp8_copy_trace(copy_call_index: int) -> int:
    global _MXFP4_QAT_FP8_COPY_TRACE_RECORD_INDEX
    record_index = _MXFP4_QAT_FP8_COPY_TRACE_RECORD_INDEX
    _MXFP4_QAT_FP8_COPY_TRACE_RECORD_INDEX += 1

    per_call_limit = _mxfp4_qat_fp8_copy_trace_limit_per_call()
    if per_call_limit is None:
        _decrement_mxfp4_qat_fp8_copy_trace_remaining()
    else:
        _MXFP4_QAT_FP8_COPY_TRACE_COUNTS_BY_CALL[copy_call_index] = (
            _MXFP4_QAT_FP8_COPY_TRACE_COUNTS_BY_CALL.get(copy_call_index, 0) + 1
        )
    return record_index


class _TraceFormatDict(dict):
    def __missing__(self, key):
        return "unknown"


def _trace_rank() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except RuntimeError:
        pass
    return os.getenv("RANK", "unknown")


def _format_mxfp4_qat_fp8_copy_trace_path(
    path_template: str,
    *,
    copy_call_index: int,
    record_index: int,
) -> str:
    return path_template.format_map(
        _TraceFormatDict(
            rank=_trace_rank(),
            pid=str(os.getpid()),
            copy_call=str(copy_call_index),
            trace_index=str(record_index),
        )
    )


def _maybe_record_mxfp4_qat_fp8_copy_trace(
    model_param: torch.Tensor,
    fake_main_param: Optional[torch.Tensor],
    raw_main_param: Optional[torch.Tensor],
    start_offset: Optional[int],
    copy_call_index: int,
) -> None:
    """Trace fake-MXFP4 main-param -> TE FP8 compute-param loss.

    This is disabled by default. It is intentionally train-side: rollout export
    tracing proves TE-FP8-export -> native-FP4 pack/dequant, while this hook
    measures the earlier copy boundary where QAT feeds fake FP4 weights into TE
    blockwise FP8 compute params.
    """
    path_template = os.getenv("SIRL_DSV4_QAT_FP8_COPY_TRACE_PATH", "")
    if not path_template or not _should_record_mxfp4_qat_fp8_copy_trace(copy_call_index):
        return
    if fake_main_param is None:
        return
    if os.getenv("OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG", "0") != "1":
        return
    if not getattr(model_param, "mcore_mxfp4_expert_qat_weight", False):
        return
    if start_offset is None:
        return
    if raw_main_param is None:
        return

    with torch.no_grad():
        expected = fake_main_param.detach().float().reshape(-1)
        raw = raw_main_param.detach().float().reshape(-1)
        if raw.numel() != expected.numel():
            raise RuntimeError(
                "MXFP4 QAT FP8-copy trace requires raw and fake main-param shards "
                f"with matching numel, got raw={raw.numel()} fake={expected.numel()}."
            )
        dequantized_model = dequantize_fp8_tensor(model_param).detach().float().reshape(-1)
        actual = dequantized_model.narrow(0, int(start_offset), expected.numel())
        diff = actual - expected
        max_abs_error = diff.abs().max()
        rmse = diff.pow(2).mean().sqrt()
        expected_rmse = expected.pow(2).mean().sqrt().clamp_min(1e-12)
        actual_rmse = actual.pow(2).mean().sqrt()
        record_index = _mark_recorded_mxfp4_qat_fp8_copy_trace(copy_call_index)
        record = {
            "schema": "dsv4_mxfp4_qat_fp8_copy_trace.v1",
            "trace_index": record_index,
            "copy_call_index": copy_call_index,
            "rank": _trace_rank(),
            "pid": os.getpid(),
            "param_name": getattr(model_param, "mcore_mxfp4_expert_qat_name", None),
            "start_offset": int(start_offset),
            "numel": int(expected.numel()),
            "model_shape": list(getattr(model_param, "shape", ())),
            "main_param_shape": list(fake_main_param.shape),
            "model_param_type": f"{type(model_param).__module__}.{type(model_param).__qualname__}",
            "model_param_dtype": str(getattr(model_param, "dtype", None)),
            "fake_main_param_dtype": str(fake_main_param.dtype),
            "max_abs_error": float(max_abs_error.cpu()),
            "rmse": float(rmse.cpu()),
            "relative_rmse": float((rmse / expected_rmse).cpu()),
            "expected_absmax": float(expected.abs().max().cpu()),
            "actual_absmax": float(actual.abs().max().cpu()),
            "expected_mean": float(expected.mean().cpu()),
            "actual_mean": float(actual.mean().cpu()),
            "expected_rms": float(expected_rmse.cpu()),
            "actual_rms": float(actual_rmse.cpu()),
        }
        sample_elems = _mxfp4_qat_fp8_copy_trace_sample_elems()
        if sample_elems:
            block_size = int(os.getenv("OPEN_TRAINING_MXFP4_BLOCK_SIZE", "32"))
            record["e2m1_positive_grid"] = list(_MXFP4_E2M1_POS_GRID)
            record["e2m1_boundaries"] = list(_MXFP4_E2M1_BOUNDARIES)
            record["sampled_main_param_values"] = _sample_mxfp4_qat_fp8_copy_boundary(
                raw_main_param=raw,
                fake_main_param=expected,
                actual=actual,
                block_size=block_size,
                sample_elems=sample_elems,
            )

    path = _format_mxfp4_qat_fp8_copy_trace_path(
        path_template,
        copy_call_index=copy_call_index,
        record_index=record_index,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_callable_from_python_import_path(dotted_path: str):
    """Resolve a Python import path like 'pkg.mod.func' to a callable.

    Raises ValueError with clear message on failure.
    """
    if not isinstance(dotted_path, str) or not dotted_path:
        raise ValueError(
            "fp8_quantizer_factory must be a non-empty string with format 'pkg.mod.func'."
        )

    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Invalid fp8_quantizer_factory '{dotted_path}'. Expected 'pkg.mod.func'.")
    module_path, attr = parts[0], parts[1]

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to import module '{module_path}' for fp8_quantizer_factory: {exc}"
        ) from exc

    fn = getattr(mod, attr, None)
    if fn is None:
        raise ValueError(
            f"Attribute '{attr}' not found in module '{module_path}' for fp8_quantizer_factory."
        )
    if not callable(fn):
        raise ValueError(
            f"Resolved attribute '{module_path}.{attr}' is not callable for fp8_quantizer_factory."
        )
    return fn


def _get_custom_recipe(quantizer_factory_python_path: str) -> Union[Fp8Recipe, Fp4Recipe]:
    quantizer_factory = _resolve_callable_from_python_import_path(quantizer_factory_python_path)
    try:
        custom_recipe = transformer_engine.common.recipe.CustomRecipe(qfactory=quantizer_factory)
    except AttributeError:
        raise ValueError(
            """CustomRecipe recipe is not available in this version of 
            Transformer Engine. Please make sure you are using TE version 
            >= 2.9.0.dev0."""
        )
    return custom_recipe


def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32
    else:
        return 16


def is_column_parallel_linear(module):
    """Returns whether the given module is a ColumnParallelLinear layer."""
    if HAVE_TE and (
        isinstance(module, TEColumnParallelLinear)
        or isinstance(module, TELayerNormColumnParallelLinear)
    ):
        return True
    elif isinstance(module, ColumnParallelLinear):
        return True
    return False


def is_row_parallel_linear(module):
    """Returns whether the given module is a RowParallelLinear layer."""
    if HAVE_TE and isinstance(module, TERowParallelLinear):
        return True
    elif isinstance(module, RowParallelLinear):
        return True
    return False


"""
The code below abstracts the functionalities needed for implementing "--fp8-param-gather" into
several functions. It provides different implementations for each function based on different
versions of TE, ensuring compatibility across various TE versions.

Currently, there are three functions:
    - modify_underlying_storage
        This function is used in DDP to place all parameters into a contiguous buffer. For
        non-fp8 tensors, replacing their data is simple, just using code like
        "tensor.data = new_data". However, for fp8 tensors, their raw data is not stored in the
        ".data" attribute, and it varies with different TE versions and different recipes. This
        function provides a unified interface to replace the underlying storage of a fp8 tensor.
    - quantize_param_shard
        This function is used in dist-opt to cast fp32 main params to fp8 params. For non-fp8
        params, this casting is as simple as "bf16_params.copy_(fp32_main_params)"; but for fp8
        params, the casting logic varies with different TE versions and different recipes. This
        function provides a unified interface to cast fp32 main params to fp8 params, and also
        updates the necessary attributes (like amax, scale, scale_inv or transpose cache) of the
        fp8 model params.
    - correct_amax_history_if_needed
        This function is used to correct the amax history of fp8 tensors. In TE1.x, some inplace
        copy operations will write unwanted values to the amax_history of fp8 tensors. This function
        corrects the amax_history back. For TE2.x, it's an empty function.
        Only useful for delayed scaling.
"""
if HAVE_TE and is_te_min_version("2.2"):
    # Supported TE versions: 2.2+
    from transformer_engine.pytorch.tensor import QuantizedTensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        from transformer_engine.pytorch.tensor.utils import replace_raw_data

        replace_raw_data(fp8_tensor, new_raw_data)

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if len(model_params) == 0:
            return

        from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8
        copy_call_index = _next_mxfp4_qat_fp8_copy_trace_call_index()

        if fsdp_shard_model_params is not None:
            if not HAVE_PACKAGING:
                raise ImportError(
                    "packaging not found, please install it with `pip install packaging`"
                )
            if not (
                get_te_version() == PkgVersion("2.3.0.dev0+5fdd7bb")
                or is_te_min_version("2.3.0")
            ):
                raise NotImplementedError(
                    f"FSDP with --fp8-param-gather is not supported in TE v{get_te_version()}"
                )

        # For newer TE versions (i.e., have post_all_gather_processing function), we keep the
        # columnwise data and manually call post_all_gather_processing after all-gather, this
        # makes fp8 params compatible with CUDA graph.
        kwargs = {}
        if te_post_all_gather_processing is not None:
            kwargs["manual_post_all_gather_processing"] = True

        raw_main_params = main_params
        main_params = [
            _maybe_fake_mxfp4_expert_qat_main_param_shard(
                model_param, main_param, start_offset
            )
            for model_param, main_param, start_offset in zip(
                model_params, main_params, start_offsets
            )
        ]
        args = [model_params, main_params, start_offsets, data_parallel_group]
        if fsdp_shard_model_params is not None:
            args.append(fsdp_shard_model_params)
        cast_master_weights_to_fp8(*args, **kwargs)
        for model_param, main_param, raw_main_param, start_offset in zip(
            model_params, main_params, raw_main_params, start_offsets
        ):
            _maybe_record_mxfp4_qat_fp8_copy_trace(
                model_param, main_param, raw_main_param, start_offset, copy_call_index
            )

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("2.0"):
    # Supported TE versions: 2.0
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        old_raw_data = fp8_tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        fp8_tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        copy_call_index = _next_mxfp4_qat_fp8_copy_trace_call_index()
        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue
            raw_main_param = main_param

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            quantizer = model_param._quantizer
            main_param = _maybe_fake_mxfp4_expert_qat_main_param_shard(
                model_param, main_param, start_offset
            )
            trace_main_param = main_param
            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            out = Float8Tensor(
                shape=main_param.size(),
                dtype=model_param.dtype,
                requires_grad=False,
                data=shard_model_param,
                fp8_scale_inv=model_param._scale_inv,
                fp8_dtype=model_param._fp8_dtype,
                quantizer=quantizer,
            )
            quantizer.update_quantized(main_param, out)
            _maybe_record_mxfp4_qat_fp8_copy_trace(
                model_param, trace_main_param, raw_main_param, start_offset, copy_call_index
            )

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            quantizer = model_param._quantizer
            amaxes.append(quantizer.amax.view(1))
            scales.append(quantizer.scale.view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        pass

elif HAVE_TE and is_te_min_version("1.0"):
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(tensor: Float8Tensor, new_raw_data: torch.Tensor) -> None:
        old_raw_data = tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[Float8Tensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # Avoid circular import
        from megatron.core.optimizer.optimizer import _multi_tensor_copy_this_to_that

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        copy_call_index = _next_mxfp4_qat_fp8_copy_trace_call_index()
        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue
            raw_main_param = main_param

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = _maybe_fake_mxfp4_expert_qat_main_param_shard(
                model_param, main_param, start_offset
            )
            trace_main_param = main_param
            main_param = main_param.to(model_param.dtype)
            cast_to_fp8(
                main_param.view(1, -1),
                model_param._fp8_meta["scaling_fwd"],
                model_param._fp8_meta_index,
                model_param._fp8_dtype,
                out=shard_model_param.view(1, -1),
            )
            _maybe_record_mxfp4_qat_fp8_copy_trace(
                model_param, trace_main_param, raw_main_param, start_offset, copy_call_index
            )

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            fp8_meta = model_param._fp8_meta["scaling_fwd"]
            fp8_meta_index = model_param._fp8_meta_index
            amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
            scales.append(fp8_meta.scale[fp8_meta_index].view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(packed_scale_views, scale_invs, dummy_overflow_buf)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

    def _correct_amax_history_if_needed_impl(model: List[torch.nn.Module]) -> None:
        for model_module in model:
            for param in model_module.parameters():
                if is_float8tensor(param) and param._fp8_meta is not None:
                    fp8_meta = param._fp8_meta["scaling_fwd"]
                    fp8_meta_index = param._fp8_meta_index
                    if hasattr(param, "get_high_precision_init_val"):
                        fp8_meta.amax_history[0][fp8_meta_index].copy_(
                            param.get_high_precision_init_val().abs().max()
                        )
                    else:
                        fp8_meta.amax_history[0][fp8_meta_index] = 0

else:
    # Fallback impl if TE version is invalid or TE is not installed.
    def _modify_underlying_storage_impl(*args, **kwargs):
        raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _quantize_param_shard_impl(model_params, *args, **kwargs):
        if len(model_params) == 0:
            return
        else:
            # If TE is not installed, there shouldn't be any fp8 params.
            raise RuntimeError("Invalid Transformer Engine version for FP8 distributed optimizer")

    def _correct_amax_history_if_needed_impl(*args, **kwargs):
        # If TE is not installed, we are definitely not using fp8 for training, so no correction
        # is needed.
        pass


# Interface Function
def modify_underlying_storage(tensor: torch.Tensor, new_raw_data: torch.Tensor):
    """Replace the underlying raw data of a tensor with new data."""
    _modify_underlying_storage_impl(tensor, new_raw_data)


# Interface Function
def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
    """Cast shard fp32 main params to fp8 model params."""
    _quantize_param_shard_impl(
        model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
    )


# Interface Function
def correct_amax_history_if_needed(model: List[torch.nn.Module]):
    """Correct the amax history of fp8 tensors when it's necessary (i.e., in TE1.x)."""
    _correct_amax_history_if_needed_impl(model)


def post_all_gather_processing(model_params):
    """
    Post-processing after all-gather for weights in distributed optimizer.
    - tensorwise: may need to create a transposed view to match backend GEMM.
    - blockwise: create column-wise storage.
    """
    if te_post_all_gather_processing is not None:
        te_post_all_gather_processing(model_params)
    else:
        # If the TE version is old and does not have post_all_gather_processing function, this is
        # a no-op, and the transpose/columnwise data will be created in the next forward pass.
        pass


def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):
    """Check if the layer is in bf16."""
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )
    # Since layer_no is a global layer index, additional checks on whether
    # we are in the first or last pipeline-parallel rank are not needed.
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    if layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
        return True
    else:
        return False


if HAVE_TE:
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    def get_fp8_recipe(config: TransformerConfig):
        """Return fp8 recipe.

        Arguments:
            config (TransformerConfig): Configuration object.

        Returns:
            FP8 recipe.
        """
        if config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        # Select fp8 recipe (TE version >= 2.1.0).
        fp8_recipe = None
        if is_te_min_version("2.1.0"):
            if config.fp8_recipe == Fp8Recipe.delayed:
                fp8_recipe = TEDelayedScaling(
                    config=config,
                    fp8_format=fp8_format,
                    override_linear_precision=(False, False, not config.fp8_wgrad),
                )
            elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                    fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
                )
            elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
                fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                    fp8_format=fp8_format
                )
            elif config.fp8_recipe == Fp8Recipe.mxfp8:
                fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                    fp8_format=fp8_format
                )
            elif config.fp8_recipe == Fp8Recipe.custom:
                fp8_recipe = _get_custom_recipe(config.fp8_quantizer_factory)
            else:
                raise ValueError(
                    "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                    "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                    "you are using a compatible TE version."
                )
        else:
            # Assert that the user is using delayed scaling.
            assert config.fp8_recipe == Fp8Recipe.delayed, (
                "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
                "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
                "Float8BlockScaling."
            )
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        return fp8_recipe

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Return fp8 context manager.

        Arguments:
            config (TransformerConfig): Configuration object.
            layer_no (int): *Global* layer index (including layers on other
                pipeline-parallel ranks).
            is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

        Returns:
            FP8 context.
            If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
            We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
            that needs to be trained in bf16.
        """

        need_fp8_context = config.fp8 if not is_init else config.fp8_param

        if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
            # bf16 training or bf16 layer in fp8 training
            fp8_context = nullcontext()
        else:
            # fp8 training and this layer_no is in fp8
            fp8_recipe = get_fp8_recipe(config)

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
                )

            if not is_init:
                fp8_context = transformer_engine.pytorch.fp8_autocast(
                    enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
                )
            else:
                import inspect

                context_args = {"enabled": True}
                # Check if fp8_model_init supports setting recipe
                if "recipe" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["recipe"] = fp8_recipe
                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in (
                    inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
                ):
                    context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
                fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

            # First / last layer in bf16 isn't supported with delayed scaling since it
            # requires entering/exiting fp8 context per layer, causing incorrect amax
            # reduction behavior.
            assert not (
                config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
            ), "Delayed scaling does not support first / last layer in BF16."

        return fp8_context

else:

    def get_fp8_recipe(config: TransformerConfig):
        """Returns None since TE is not available."""
        return None

    def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
        """Returns dummy fp8 context manager since TE is not available."""
        return nullcontext()


if HAVE_TE:
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # Modules that have been wrapped for inference for fp8
    _fp8_inference_wrapped_modules = weakref.WeakSet()

    def _wrap_te_linear_for_padding(module: torch.nn.Module):
        """Wrap a TE linear module to automatically pad sequences for FP8 inference.

        Modifies the module's forward method to:
        1. Pad input sequences to FP8 alignment requirements
        2. Run the original forward pass
        3. Unpad outputs to original sequence length

        Args:
            module: A Transformer Engine linear layer (TELinear, TEColumnParallelLinear, etc.)
        """
        if module in _fp8_inference_wrapped_modules:
            return
        _pad_func = Fp8Padding(1)
        _unpad_func = Fp8Unpadding(1)

        original_forward = module.forward

        @wraps(original_forward)
        def padded_forward(input_tensor, *args, **kwargs):
            # Only do padding for fp8 if we are in fp8 context
            if not FP8GlobalStateManager.is_fp8_enabled():
                return original_forward(input_tensor, *args, **kwargs)

            # With sequence parallelism we need to all-gather before padding
            # and reduce-scatter after unpadding
            if is_sequence_parallel := getattr(module, "sequence_parallel", False):
                if is_column_parallel_linear(module):
                    input_tensor = gather_from_sequence_parallel_region(
                        input_tensor, group=module.tp_group
                    )

                # Disable sequence parallelism on the module because we are handling the
                # all-gather and reduce-scatter externally
                module.sequence_parallel = False

            seq_len, batch_size, hidden_size = input_tensor.shape
            # Reshape to (S, B*H) to pad sequence dimension
            input_2d = input_tensor.reshape(seq_len, -1)
            # Pad the sequence dimension
            padded_input_2d, _ = _pad_func(input_2d, [seq_len])
            padded_seq_len = padded_input_2d.shape[0]

            # Reshape back to (padded_S, B, H)
            padded_input_3d = padded_input_2d.view(padded_seq_len, batch_size, hidden_size)
            output = original_forward(padded_input_3d, *args, **kwargs)

            # Handle output
            if isinstance(output, tuple):
                output_tensor = output[0]
                other_outputs = output[1:]
            else:
                output_tensor = output
                other_outputs = ()

            # Unpad output - reshape to 2D, unpad, reshape back
            _, _, output_hidden_size = output_tensor.shape
            output_2d = output_tensor.reshape(padded_seq_len, -1)
            unpadded_output_2d = _unpad_func(output_2d, [seq_len])
            unpadded_output = unpadded_output_2d.reshape(seq_len, batch_size, output_hidden_size)

            if is_sequence_parallel:
                # Reduce-scatter after unpadding
                if is_row_parallel_linear(module):
                    unpadded_output = reduce_scatter_to_sequence_parallel_region(
                        unpadded_output, group=module.tp_group
                    )

                # Reset sequence parallelism flag on the module
                module.sequence_parallel = True

            if other_outputs:
                return (unpadded_output,) + other_outputs
            else:
                return unpadded_output

        module.forward = padded_forward
        _fp8_inference_wrapped_modules.add(module)

    def prepare_model_for_fp8_inference(model):
        """Prepare a model for FP8 inference by wrapping TE linear layers with padding support.

        FP8 TE Gemms have specific shape requirements. This function wraps all Transformer
        Engine linear layers in the model to automatically pad/unpad sequences during inference.

        Args:
            model (model (GPTModel): Model containing TE linear layers.

        Returns:
            GPTModel: The same model with wrapped linear layers (modified in-place).

        """
        assert Fp8Padding and Fp8Unpadding, "TE version does not have FP8 padding functions"
        # Find and wrap all TE linear layers
        for module in model.modules():
            if isinstance(module, TE_LINEAR_TYPES):
                _wrap_te_linear_for_padding(module)

        return model

else:

    def prepare_model_for_fp8_inference(model):
        """If trys using prepare_model_for_fp8_inference without TE we error"""
        raise RuntimeError(
            "prepare_model_for_fp8_inference requires Transformer Engine to be installed. "
            "Please install transformer-engine to use FP8 inference."
        )
