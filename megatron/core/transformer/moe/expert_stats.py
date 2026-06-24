# Copyright (c) 2026, Scitix. All rights reserved.
"""Shared MoE expert-routing statistics for Scitix trainers.

Framework-agnostic core consumed by BOTH mcore-trainer (SFT/CPT/DPO) and
slime-trainer (RL). It captures per-layer per-expert token counts + routing
weights inside the router, reduces them across model-parallel groups, and emits
summary scalars (+ optional local heatmap PNG / JSONL). It owns NO logging sink:
each trainer feeds the returned scalar dict into its own logger.

Design split (see the Strategy-3 port plan):
  - capture / reduce / scalars / heatmap render -> here (depends only on
    ``parallel_state`` + tensor shapes, which ARE Megatron).
  - enable decision (role/provider checks), step cadence, output-dir convention,
    and the logging sink -> the trainer ("engine") layer.

Capture wiring:
  - Built-in / stock ``TopKRouter`` callers get capture automatically via the
    hook in ``TopKRouter.routing()`` (it calls :func:`save_expert_stats` when
    :func:`is_enabled`).
  - Subclasses that fully override ``routing()`` without calling ``super()``
    (e.g. slime's ``SiRLTopKRouter``) must call :func:`save_expert_stats`
    themselves; the base-class hook cannot reach them.

Metrics (global, v1 — per-source/CPT distribution is a future ``register_extra_observer`` extension):
  moe_expert_token_count_{cv,std,max,min}/layer_{i}, moe_expert_weight_entropy/layer_{i},
  moe_expert_token_count_cv/global_avg, moe_expert_weight_entropy/global_avg,
  optional moe_expert_detail/layer_{i}/expert_{j}/{tokens,weight,weight_ratio},
  heatmap PNG + expert_stats.jsonl under the configured output dir.

Reduction semantics — SUM, each parallel axis counted EXACTLY once:
  PP (combine layers across pipeline stages) + TP&CP (combine token subsets from
  tensor/context parallel) + DP-without-CP (combine data-parallel shards).
  NOTE: ``get_tensor_and_context_parallel_group()`` already covers CP, so DP must
  use ``with_context_parallel=False`` — otherwise CP is summed twice.
  Divide by the number of accumulated training steps for a per-global-batch mean.
"""

import json
import os
import queue
import sys
import threading
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

# ======================== Module state ========================

# Carriers live on ``torch`` so all import instances (PYTHONPATH dual-module,
# fork inheritance) share one tracker + step counter.
if not hasattr(torch, "_expert_stats_tracker"):
    torch._expert_stats_tracker = {}
    torch._expert_stats_step_count = 0

# Hard kill-switch independent of config (set by ops to force-disable).
_ENV_DISABLE = "MEGATRON_DISABLE_EXPERT_STATS"

# Single source of truth for the enable decision + cadence + output. Set ONCE by
# the engine via :func:`configure` before the first forward; read everywhere
# (router hook, reduce, render).
_CONFIG = {
    "enabled": False,
    "log_interval": 0,
    "heatmap_interval": 10,
    "output_dir": None,
    "per_layer": False,
}

# Optional per-forward observers (v2 extension point, e.g. mcore-CPT per-source).
_EXTRA_OBSERVERS: list = []

# Background logging — bounded queue so a slow consumer cannot grow memory.
_LOG_QUEUE: "queue.Queue" = queue.Queue(maxsize=4)
_LOG_THREAD: Optional[threading.Thread] = None


# ======================== Configuration ========================

def configure(
    *,
    enabled: bool,
    log_interval: int,
    heatmap_interval: int = 10,
    output_dir: Optional[str] = None,
    per_layer: bool = False,
) -> None:
    """Set the single source-of-truth config. Call ONCE before the first forward.

    The ``enabled`` decision (interval > 0, role/provider/MoE checks) is computed
    by the engine; this only stores plain values. A negative ``log_interval`` is a
    config error (argparse ``type=int`` does not reject it), not "disabled".
    """
    if log_interval < 0:
        raise ValueError(
            f"expert-stats log_interval must be >= 0 (0 = disabled); got {log_interval}."
        )
    env_ok = os.environ.get(_ENV_DISABLE, "0") != "1"
    _CONFIG["enabled"] = bool(enabled) and env_ok
    _CONFIG["log_interval"] = int(log_interval)
    _CONFIG["heatmap_interval"] = int(heatmap_interval)
    _CONFIG["output_dir"] = output_dir
    _CONFIG["per_layer"] = bool(per_layer)


def is_enabled() -> bool:
    """True only after :func:`configure` enabled it (and env did not disable)."""
    return _CONFIG["enabled"]


def register_extra_observer(fn: Callable) -> None:
    """Register ``fn(routing_map, probs, layer_number)`` called on every capture.

    Extension point for engine-specific breakdowns (e.g. mcore-CPT per-source)
    without teaching this module about that concept. Inert in v1.
    """
    if fn not in _EXTRA_OBSERVERS:
        _EXTRA_OBSERVERS.append(fn)


def resolve_output_dir(
    explicit: Optional[str],
    *,
    save_dir: Optional[str] = None,
    run_log_dir_env: str = "RUN_LOG_DIR",
) -> Optional[str]:
    """Resolve an output dir, honoring an explicit value first (engine helper).

    Absolute explicit -> as-is. Relative -> under ``$RUN_LOG_DIR`` if set; else an
    explicit relative path stays relative; else default ``{save_dir}/expert_stats``.
    """
    from pathlib import Path

    is_explicit = explicit is not None
    p = Path(explicit if is_explicit else "expert_stats").expanduser()
    if p.is_absolute():
        return str(p)
    run_log_dir = os.environ.get(run_log_dir_env)
    if run_log_dir:
        return str(Path(run_log_dir).expanduser() / p)
    if is_explicit:
        return str(p)
    if save_dir:
        return str(Path(save_dir) / "expert_stats")
    return None


# ======================== Capture (GPU, no sync) ========================

def increment_step_count() -> None:
    """Bump the accumulation-window counter. Call on EVERY training step.

    Normalization divides the accumulated sum by this count; gating it to log
    steps would divide by 1.
    """
    torch._expert_stats_step_count += 1


def save_expert_stats(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    layer_number: int,
    num_layers: int,
    num_experts: int,
) -> None:
    """Accumulate global per-layer per-expert token counts + routing weights.

    GPU-only, no collectives. Cheap: one ``sum(dim=0)`` per tensor + in-place add.

    ``layer_number`` is Megatron's 1-based router layer index; rows are written at
    ``layer_number - 1``. Out-of-range layers are skipped — in particular MTP MoE
    layers, whose ``layer_number`` is offset past ``num_layers`` and which v1 does
    not collect (the tracker is sized to ``num_layers`` only).
    """
    if not _CONFIG["enabled"]:
        return
    if layer_number is None or not (1 <= layer_number <= num_layers):
        return

    tracker = torch._expert_stats_tracker
    if "tokens_per_expert" not in tracker:
        device = routing_map.device
        tracker["tokens_per_expert"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )
        tracker["weight_per_expert"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )
        tracker["num_layers"] = num_layers
        tracker["num_experts"] = num_experts

    with torch.no_grad():
        idx = layer_number - 1
        tracker["tokens_per_expert"][idx] += routing_map.float().sum(dim=0)  # [E]
        tracker["weight_per_expert"][idx] += probs.detach().sum(dim=0)       # [E]

    for fn in _EXTRA_OBSERVERS:
        try:
            fn(routing_map, probs, layer_number)
        except Exception as e:  # never let an observer break training
            print(f"[expert_stats] extra observer error: {e}", file=sys.stderr)


# ======================== Reduce (COLLECTIVE — all ranks) ========================

def should_render_heatmap(step: int) -> bool:
    """Heatmap cadence on the dedicated interval (0-based ``step``)."""
    hi = _CONFIG["heatmap_interval"]
    return hi > 0 and ((step + 1) % hi == 0)


def reduce_snapshot_and_clear(*, num_layers: int, num_experts: int, is_logging_rank: bool):
    """Reduce accumulated stats across model-parallel groups and clear.

    COLLECTIVE: must be called on ALL ranks at the same cadence, OUTSIDE any
    rank guard, or it deadlocks. All ranks zero-init an identically-shaped tracker
    so the all-reduce is well-formed even on a rank that never ran an MoE forward
    (e.g. a PP stage holding no MoE layer).

    Returns a small CPU snapshot dict on the logging rank, else ``None``. The
    tracker + step counter are cleared on EVERY rank in ``finally`` so a logging-
    rank exception cannot leave a half-cleared tracker that double-counts next
    interval.
    """
    tracker = torch._expert_stats_tracker
    if "tokens_per_expert" not in tracker:
        device = torch.cuda.current_device()
        tracker["tokens_per_expert"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )
        tracker["weight_per_expert"] = torch.zeros(
            num_layers, num_experts, device=device, dtype=torch.float32
        )

    try:
        from megatron.core import parallel_state

        for key in ("tokens_per_expert", "weight_per_expert"):
            values = tracker[key]
            # PP: different stages hold different layers (idle stages contribute 0).
            dist.all_reduce(values, group=parallel_state.get_pipeline_model_parallel_group())
            # TP + CP: tensor/context parallel split tokens across ranks.
            dist.all_reduce(values, group=parallel_state.get_tensor_and_context_parallel_group())
            # DP only (NO CP — CP already summed above; double-counting it inflates counts).
            dist.all_reduce(
                values, group=parallel_state.get_data_parallel_group(with_context_parallel=False)
            )

        if not is_logging_rank:
            return None

        steps = max(int(torch._expert_stats_step_count), 1)
        tokens = (tracker["tokens_per_expert"] / steps).detach().to("cpu").numpy().copy()
        weights = (tracker["weight_per_expert"] / steps).detach().to("cpu").numpy().copy()
        return {
            "tokens": tokens,
            "weights": weights,
            "num_layers": num_layers,
            "num_experts": num_experts,
        }
    finally:
        tracker["tokens_per_expert"].zero_()
        tracker["weight_per_expert"].zero_()
        torch._expert_stats_step_count = 0


# ======================== Scalars (logging rank, sync) ========================

def _moe_layer_indices(tokens: np.ndarray) -> list:
    """Active MoE layers = rows with any routed tokens (robust to interleaved MoE)."""
    return [i for i in range(tokens.shape[0]) if tokens[i].sum() > 0]


def collect_scalar_metrics(snapshot: dict, step: int) -> dict:
    """Build per-layer + global summary scalars from a CPU snapshot. Sync, cheap.

    Returns scalars ONLY (no images). When ``per_layer`` is configured, also emits
    per-expert detail scalars. The engine merges this into its own log dict.
    """
    tokens = snapshot["tokens"]
    weights = snapshot["weights"]
    num_experts = snapshot["num_experts"]

    metrics: dict = {}
    cvs: list = []
    entropies: list = []

    moe_idx = _moe_layer_indices(tokens)
    for i in moe_idx:
        layer_tokens = tokens[i]
        layer_weights = weights[i]

        mean_t = layer_tokens.mean()
        std_t = layer_tokens.std()
        cv = float(std_t / mean_t) if mean_t > 0 else 0.0
        cvs.append(cv)
        metrics[f"moe_expert_token_count_cv/layer_{i}"] = cv
        metrics[f"moe_expert_token_count_std/layer_{i}"] = float(std_t)
        metrics[f"moe_expert_token_count_max/layer_{i}"] = float(layer_tokens.max())
        metrics[f"moe_expert_token_count_min/layer_{i}"] = float(layer_tokens.min())

        weight_sum = layer_weights.sum()
        if weight_sum > 0:
            p = np.clip(layer_weights / weight_sum, 1e-10, None)
            entropy = float(-np.sum(p * np.log(p)))
        else:
            entropy = 0.0
        entropies.append(entropy)
        metrics[f"moe_expert_weight_entropy/layer_{i}"] = entropy

    if cvs:
        metrics["moe_expert_token_count_cv/global_avg"] = float(np.mean(cvs))
    if entropies:
        metrics["moe_expert_weight_entropy/global_avg"] = float(np.mean(entropies))

    if _CONFIG["per_layer"]:
        for i in moe_idx:
            layer_tokens = tokens[i]
            layer_weights = weights[i]
            weight_sum = layer_weights.sum()
            for j in range(num_experts):
                metrics[f"moe_expert_detail/layer_{i}/expert_{j}/tokens"] = float(layer_tokens[j])
                metrics[f"moe_expert_detail/layer_{i}/expert_{j}/weight"] = float(layer_weights[j])
                if weight_sum > 0:
                    metrics[f"moe_expert_detail/layer_{i}/expert_{j}/weight_ratio"] = float(
                        layer_weights[j] / weight_sum
                    )

    return metrics


# ======================== Heatmap + JSONL (async) ========================

def _log_worker() -> None:
    while True:
        item = _LOG_QUEUE.get()
        if item is None:
            break
        try:
            item()
        except Exception as e:
            print(f"[expert_stats] logging error: {e}", file=sys.stderr)
        finally:
            _LOG_QUEUE.task_done()


def _ensure_log_thread() -> None:
    global _LOG_THREAD
    if _LOG_THREAD is None or not _LOG_THREAD.is_alive():
        _LOG_THREAD = threading.Thread(target=_log_worker, daemon=True, name="expert-stats-logger")
        _LOG_THREAD.start()


def _render_heatmap(data: np.ndarray, title: str, xlabel: str = "Expert", ylabel: str = "Layer"):
    """Render an [L, E] matrix to an RGB uint8 array. Lazy matplotlib import.

    Returns ``None`` (warn) if matplotlib is unavailable — matplotlib is NOT a hard
    Megatron dependency. Both trainers ship it, so this never misses in practice.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import io

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception:
        print("[expert_stats] matplotlib unavailable; skipping heatmap", file=sys.stderr)
        return None

    num_layers, num_experts = data.shape
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=100)

    vmin = data[data > 0].min() if (data > 0).any() else 1.0
    vmax = data.max()
    norm = LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, 1.0)) if vmax / max(vmin, 1e-10) > 100 else None

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ystep = max(1, num_layers // 12)
    ax.set_yticks(range(0, num_layers, ystep))
    ax.set_yticklabels([str(i) for i in range(0, num_layers, ystep)])
    xstep = max(1, num_experts // 12)
    ax.set_xticks(range(0, num_experts, xstep))
    ax.set_xticklabels([str(i) for i in range(0, num_experts, xstep)])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    try:
        from PIL import Image as PILImage

        return np.array(PILImage.open(buf).convert("RGB"), dtype=np.uint8)
    except Exception:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name  # file path fallback


def _save_heatmap_png(img_data, directory: str, filename: str) -> None:
    if img_data is None:
        return
    filepath = os.path.join(directory, filename)
    try:
        if isinstance(img_data, str):
            import shutil

            shutil.copy2(img_data, filepath)
        elif isinstance(img_data, np.ndarray):
            from PIL import Image as PILImage

            PILImage.fromarray(img_data).save(filepath)
        print(f"[expert_stats] saved heatmap: {filepath}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[expert_stats] failed to save heatmap {filepath}: {e}", file=sys.stderr)


def render_and_dump(snapshot: dict, step: int, *, heatmap: bool, jsonl: bool = True) -> None:
    """Async: render a local heatmap PNG (if ``heatmap``) + append JSONL (if ``jsonl``).

    No logging sink. Uses the configured ``output_dir``; a no-op when it is unset.
    Safe to call from the logging rank only.
    """
    output_dir = _CONFIG["output_dir"]
    if output_dir is None or snapshot is None:
        return

    tokens = snapshot["tokens"]
    weights = snapshot["weights"]
    moe_idx = _moe_layer_indices(tokens)

    def _job():
        if heatmap and moe_idx:
            try:
                img = _render_heatmap(
                    tokens[moe_idx],
                    title=f"Expert Token Count (step {step})",
                    ylabel=f"MoE Layer (0-{len(moe_idx) - 1})",
                )
                if img is not None:
                    hm_dir = os.path.join(output_dir, "heatmaps")
                    os.makedirs(hm_dir, exist_ok=True)
                    _save_heatmap_png(img, hm_dir, f"token_count_iter{step:07d}.png")
            except Exception as e:
                print(f"[expert_stats] heatmap error: {e}", file=sys.stderr)

        if jsonl:
            try:
                os.makedirs(output_dir, exist_ok=True)
                record = {
                    "step": step,
                    "tokens_per_expert": tokens.tolist(),
                    "weight_per_expert": weights.tolist(),
                }
                with open(os.path.join(output_dir, "expert_stats.jsonl"), "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                print(f"[expert_stats] jsonl error: {e}", file=sys.stderr)

    _ensure_log_thread()
    try:
        _LOG_QUEUE.put_nowait(_job)
    except queue.Full:
        print(f"[expert_stats] log queue full at step {step}, dropping", file=sys.stderr)
