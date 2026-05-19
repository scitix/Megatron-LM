# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch.distributed as dist
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.transformer.tree_metadata import TreeMetadata


@dataclass
class PackedSeqParams:
    '''
    parameters to TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    # Tree-attention metadata for the native tree-training path. ``None`` for
    # all existing callers; populated by downstream tree-data iterators when
    # tree training is enabled. TEDotProductAttention passes this through to TE,
    # which dispatches to TreeFlashAttention.
    # See megatron.core.transformer.tree_metadata.
    tree_metadata: Optional["TreeMetadata"] = None
