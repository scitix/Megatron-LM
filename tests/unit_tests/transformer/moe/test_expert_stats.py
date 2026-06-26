# Copyright (c) 2026, Scitix. All rights reserved.
"""Unit tests for the shared MoE expert-routing stats module.

`TestExpertStatsLogic` is pure-Python (config validation, 1-based layer guard,
scalar math, output-dir resolution) and needs no GPU/model-parallel.
`TestExpertStatsDistributed` exercises the collective reduce + the real
`TopKRouter.routing()` capture hook and is skipped without CUDA. The strong
multi-topology single-count / CP-no-double-count check stays in the external GPU
smoke (it needs a controlled world size); here we assert the structural
invariants that hold at any world size.
"""
import numpy as np
import pytest
import torch

from megatron.core.transformer.moe import expert_stats as es


def _reset_tracker():
    if hasattr(torch, "_expert_stats_tracker"):
        torch._expert_stats_tracker.clear()
    torch._expert_stats_step_count = 0


class TestExpertStatsLogic:
    """CPU-only: configure / guards / scalars / path resolution."""

    def setup_method(self, method):
        _reset_tracker()
        es.configure(enabled=True, log_interval=2, heatmap_interval=2, output_dir=None, per_layer=False)

    def teardown_method(self, method):
        _reset_tracker()
        es.configure(enabled=False, log_interval=0)

    @pytest.mark.internal
    def test_configure_negative_interval_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            es.configure(enabled=True, log_interval=-1)

    @pytest.mark.internal
    def test_configure_zero_is_disabled(self):
        es.configure(enabled=False, log_interval=0)
        assert es.is_enabled() is False

    @pytest.mark.internal
    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("MEGATRON_DISABLE_EXPERT_STATS", "1")
        es.configure(enabled=True, log_interval=100)
        assert es.is_enabled() is False

    @pytest.mark.internal
    def test_should_render_heatmap_cadence(self):
        es.configure(enabled=True, log_interval=100, heatmap_interval=10)
        assert es.should_render_heatmap(9) is True       # (9+1) % 10 == 0
        assert es.should_render_heatmap(8) is False
        es.configure(enabled=True, log_interval=100, heatmap_interval=0)
        assert es.should_render_heatmap(9) is False       # <=0 disables

    @pytest.mark.internal
    def test_resolve_output_dir(self, monkeypatch):
        monkeypatch.delenv("RUN_LOG_DIR", raising=False)
        # explicit relative stays relative when RUN_LOG_DIR unset
        assert es.resolve_output_dir("foo", save_dir="/ckpt") == "foo"
        # default None -> {save}/expert_stats
        assert es.resolve_output_dir(None, save_dir="/ckpt") == "/ckpt/expert_stats"
        # absolute explicit -> as-is
        assert es.resolve_output_dir("/abs/x") == "/abs/x"
        # RUN_LOG_DIR set -> under it
        monkeypatch.setenv("RUN_LOG_DIR", "/run")
        assert es.resolve_output_dir("foo") == "/run/foo"
        assert es.resolve_output_dir(None, save_dir="/ckpt") == "/run/expert_stats"

    @pytest.mark.internal
    def test_save_one_based_layer_guard(self):
        # layer_number is 1-based; rows written at layer_number - 1.
        rmap = torch.zeros(5, 4)
        rmap[:, 0] = 1.0
        probs = torch.zeros(5, 4)
        es.save_expert_stats(rmap, probs, 0, 3, 4)            # invalid 0 -> skip
        es.save_expert_stats(rmap, probs, 4, 3, 4)            # > num_layers -> skip (e.g. MTP)
        assert "tokens_per_expert" not in torch._expert_stats_tracker
        es.save_expert_stats(rmap, probs, 3, 3, 4)            # last valid layer -> row index 2
        t = torch._expert_stats_tracker["tokens_per_expert"]
        assert t.shape == (3, 4)
        assert t[2, 0].item() == 5.0
        assert t[:2].sum().item() == 0.0

    @pytest.mark.internal
    def test_save_accumulates(self):
        rmap = torch.zeros(6, 4)
        rmap[:, 2] = 1.0
        es.save_expert_stats(rmap, torch.ones(6, 4), 1, 4, 4)
        es.save_expert_stats(rmap, torch.ones(6, 4), 1, 4, 4)
        assert torch._expert_stats_tracker["tokens_per_expert"][0, 2].item() == 12.0

    @pytest.mark.internal
    def test_save_noop_when_disabled(self):
        es.configure(enabled=False, log_interval=100)
        es.save_expert_stats(torch.ones(8, 4), torch.ones(8, 4), 1, 2, 4)
        assert "tokens_per_expert" not in torch._expert_stats_tracker

    @pytest.mark.internal
    def test_collect_scalar_metrics(self):
        # layer 0 active, layer 1 all-zero (non-MoE) -> skipped.
        tokens = np.array([[10.0, 30.0], [0.0, 0.0]], dtype=np.float32)
        weights = np.array([[0.25, 0.75], [0.0, 0.0]], dtype=np.float32)
        snap = {"tokens": tokens, "weights": weights, "num_layers": 2, "num_experts": 2}
        m = es.collect_scalar_metrics(snap, step=99)
        assert "moe_expert_token_count_cv/layer_1" not in m
        assert m["moe_expert_token_count_max/layer_0"] == 30.0
        assert m["moe_expert_token_count_min/layer_0"] == 10.0
        assert abs(m["moe_expert_token_count_cv/layer_0"] - 0.5) < 1e-6   # std50/mean20... mean20 std10 -> 0.5
        assert abs(m["moe_expert_token_count_cv/global_avg"] - 0.5) < 1e-6
        p = np.array([0.25, 0.75])
        exp_h = float(-np.sum(p * np.log(p)))
        assert abs(m["moe_expert_weight_entropy/layer_0"] - exp_h) < 1e-6


def _moe_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        ffn_hidden_size=32,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",   # skip aux loss -> no extra collectives
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=False,
        add_bias_linear=False,
        params_dtype=torch.float32,
        perform_initialization=False,
        bf16=False,
        fp16=False,
        sequence_parallel=False,
    )


class TestExpertStatsDistributed:
    """GPU + model-parallel: collective reduce + real TopKRouter capture hook."""

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1, 1)
        _reset_tracker()

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        _reset_tracker()
        es.configure(enabled=False, log_interval=0)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reduce_clears_and_normalizes(self):
        es.configure(enabled=True, log_interval=1, output_dir=None)
        dev = torch.cuda.current_device()
        rmap = torch.zeros(10, 4, device=dev)
        rmap[:, 1] = 1.0
        es.save_expert_stats(rmap, torch.ones(10, 4, device=dev), 1, 2, 4)
        es.increment_step_count()
        es.increment_step_count()  # step_count = 2 -> normalize divides by 2

        snap = es.reduce_snapshot_and_clear(num_layers=2, num_experts=4, is_logging_rank=True)

        # cleared on every rank
        assert torch._expert_stats_step_count == 0
        assert torch._expert_stats_tracker["tokens_per_expert"].abs().sum().item() == 0.0
        # logging rank: structural invariants that hold at ANY world size
        assert snap is not None
        assert snap["tokens"].shape == (2, 4)
        assert np.isfinite(snap["tokens"]).all()
        assert (snap["tokens"] >= 0).all()
        # single rank (collective no-op): 10 tokens / step_count 2 = 5.0 at expert 1
        # (>1 ranks sum, so only assert exact value when world size is 1)
        from megatron.core import parallel_state as ps

        if ps.get_data_parallel_world_size(with_context_parallel=True) == 1:
            assert abs(float(snap["tokens"][0, 1]) - 5.0) < 1e-4
        metrics = es.collect_scalar_metrics(snap, step=0)
        assert all(np.isfinite(v) for v in metrics.values())

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_base_hook_fires_only_on_grad_training_forward(self):
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.moe.router import TopKRouter

        cfg = _moe_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        dev = torch.cuda.current_device()

        def fresh_router():
            r = TopKRouter(cfg, pg_collection=pg).to(dev)
            r.set_layer_number(1)
            return r

        def logits(grad=True):
            return torch.randn(4, 3, cfg.num_moe_experts, device=dev, requires_grad=grad)

        es.configure(enabled=True, log_interval=1, output_dir=None)

        # train + grad -> FIRES
        _reset_tracker()
        r = fresh_router()
        r.train()
        r.routing(logits())
        assert torch._expert_stats_tracker.get("tokens_per_expert") is not None
        assert torch._expert_stats_tracker["tokens_per_expert"].sum().item() > 0

        # no_grad -> EXCLUDED (ref/old_actor path)
        _reset_tracker()
        r = fresh_router()
        r.train()
        with torch.no_grad():
            r.routing(logits(grad=False))
        assert "tokens_per_expert" not in torch._expert_stats_tracker

        # eval -> EXCLUDED
        _reset_tracker()
        r = fresh_router()
        r.eval()
        r.routing(logits())
        assert "tokens_per_expert" not in torch._expert_stats_tracker

        # disabled -> no capture even in train+grad
        _reset_tracker()
        es.configure(enabled=False, log_interval=1)
        r = fresh_router()
        r.train()
        r.routing(logits())
        assert "tokens_per_expert" not in torch._expert_stats_tracker
