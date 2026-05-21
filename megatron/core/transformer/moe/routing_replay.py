"""MoE routing replay buffer for deterministic expert routing."""
import logging
import os

import torch


ROUTING_REPLAY = None

logger = logging.getLogger(__name__)


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


def _active_routing_replay():
    if ROUTING_REPLAY is None:
        raise RuntimeError("Routing replay stage is active but no RoutingReplay module is bound to this forward pass")
    return ROUTING_REPLAY


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self):
        if self.forward_index >= len(self.top_indices_list):
            raise RuntimeError(
                "Routing replay forward buffer exhausted: "
                f"forward_index={self.forward_index} recorded_batches={len(self.top_indices_list)}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self):
        if self.backward_index >= len(self.top_indices_list):
            raise RuntimeError(
                "Routing replay backward buffer exhausted: "
                f"backward_index={self.backward_index} recorded_batches={len(self.top_indices_list)}"
            )
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def get_routing_replay_compute_topk(old_compute_topk):
    def validate_replayed_top_indices(top_indices, scores, topk):
        if not isinstance(top_indices, torch.Tensor):
            raise TypeError(f"routing replay top_indices must be a tensor, got {type(top_indices).__name__}")
        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"routing replay scores must be a tensor, got {type(scores).__name__}")
        if top_indices.ndim != 2:
            raise ValueError(f"routing replay top_indices must be rank-2, got shape={tuple(top_indices.shape)}")
        if scores.ndim != 2:
            raise ValueError(f"routing replay scores must be rank-2, got shape={tuple(scores.shape)}")
        expected_shape = (scores.shape[0], topk)
        if tuple(top_indices.shape) != expected_shape:
            raise ValueError(
                "routing replay top_indices shape must match scores batch and topk: "
                f"top_indices={tuple(top_indices.shape)} expected={expected_shape}"
            )

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ.get("ROUTING_REPLAY_STAGE")
            if not routing_replay_stage:
                raise RuntimeError("ENABLE_ROUTING_REPLAY=1 requires ROUTING_REPLAY_STAGE to be set")
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                _active_routing_replay().record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = _active_routing_replay().pop_forward()
                validate_replayed_top_indices(top_indices, scores, topk)
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = _active_routing_replay().pop_backward()
                validate_replayed_top_indices(top_indices, scores, topk)
                probs = scores.gather(1, top_indices)
            else:
                raise ValueError(f"Unsupported ROUTING_REPLAY_STAGE={routing_replay_stage!r}")
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module, skip_mtp=None):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        if skip_mtp is True:
            return
        if os.environ.get("SKIP_MTP_ROUTING_REPLAY", "0") == "1":
            return

        module.routing_replay = RoutingReplay()
        import sys
        n = len(RoutingReplay.all_routing_replays)
        sys.stderr.write(f"[routing_replay] Registered RoutingReplay #{n} on {type(module).__name__}\n")
        sys.stderr.flush()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
