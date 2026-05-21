import pytest
import torch

from megatron.core.transformer.moe.routing_replay import (
    RoutingReplay,
    get_routing_replay_compute_topk,
    set_routing_replay,
)


def teardown_function():
    RoutingReplay.all_routing_replays.clear()
    set_routing_replay(None)


def test_routing_replay_disabled_uses_original_topk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    monkeypatch.delenv("ROUTING_REPLAY_STAGE", raising=False)
    scores = torch.tensor([[1.0, 3.0, 2.0]])

    def _topk(scores, topk, num_groups=None, group_topk=None):
        return torch.topk(scores, k=topk, dim=1)

    wrapped_topk = get_routing_replay_compute_topk(_topk)

    probs, top_indices = wrapped_topk(scores, topk=2)

    assert torch.equal(probs, torch.tensor([[3.0, 2.0]]))
    assert torch.equal(top_indices, torch.tensor([[1, 2]]))


def test_routing_replay_enabled_requires_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.delenv("ROUTING_REPLAY_STAGE", raising=False)

    def _topk(*_args, **_kwargs):
        raise AssertionError("missing ROUTING_REPLAY_STAGE must fail before fresh topk")

    wrapped_topk = get_routing_replay_compute_topk(_topk)

    with pytest.raises(RuntimeError, match="requires ROUTING_REPLAY_STAGE"):
        wrapped_topk(torch.zeros((2, 4)), topk=1)


def test_routing_replay_forward_exhaustion_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = RoutingReplay()
    set_routing_replay(replay)
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")

    def _fallback_topk(*_args, **_kwargs):
        raise AssertionError("routing replay must not fallback to fresh topk during replay")

    wrapped_topk = get_routing_replay_compute_topk(_fallback_topk)

    with pytest.raises(RuntimeError, match="forward buffer exhausted"):
        wrapped_topk(torch.zeros((2, 4)), topk=1)


def test_routing_replay_rejects_shape_mismatch_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = RoutingReplay()
    replay.pop_forward = lambda: torch.zeros((1, 1), dtype=torch.long)
    set_routing_replay(replay)
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")

    def _fallback_topk(*_args, **_kwargs):
        raise AssertionError("routing replay must not fallback to fresh topk during replay")

    wrapped_topk = get_routing_replay_compute_topk(_fallback_topk)

    with pytest.raises(ValueError, match="top_indices shape"):
        wrapped_topk(torch.zeros((2, 4)), topk=1)
