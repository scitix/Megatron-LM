# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tree-attention metadata carried alongside packed THD sequences.

Used by the tree-training native path to flow tree topology + precomputed FA3
attention buffers + per-token positional ids from the data iterator through
``PackedSeqParams.tree_metadata`` to ``TEDotProductAttention`` (tree fast
path) and ``RotaryEmbedding`` (tree-position gather).

The dataclass is deliberately lightweight: torch is imported only for
``Tensor`` typing, no CUDA assumptions are made at definition time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor


@dataclass
class TreeMetadata:
    """Tree topology + per-token positional ids for one packed micro-batch.

    Fields
    ------
    cu_node_lens
        ``int32`` tensor of shape ``[num_nodes + 1]`` — cumulative token counts
        per trie node. Identical contract to the FA3 tree kernel input.
    node_parent
        ``int32`` tensor of shape ``[num_nodes]`` — parent index per trie node;
        ``-1`` marks roots.
    tree_position_ids
        ``int64`` tensor of shape ``[total_tokens]`` — per-token positional id
        used by ``RotaryEmbedding.forward`` to gather rope freqs.
    padded_size
        Padded packed length (``cu_node_lens[-1]`` after padding).
    num_nodes
        Number of real trie nodes (``len(node_parent)``); convenience accessor
        so consumers do not have to ``shape[0]`` the parent tensor.
    precomputed
        Opaque ``dict`` returned by ``flash_attn_interface.precompute_tree_metadata``.
        Reused across all transformer layers in a forward / backward pass.
    """

    cu_node_lens: Tensor
    node_parent: Tensor
    tree_position_ids: Tensor
    padded_size: int
    num_nodes: int
    precomputed: Optional[Dict[str, Any]] = field(default=None)
