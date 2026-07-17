"""Step 3 — Sliding-window feature generation.

Each window of shape ``(n_features, window_size)`` is one training/inference
sample. Alongside the input tensor we emit per-window targets:

* ``cls``    — workload behavior class (mode of in-window labels)
* ``demand`` — next-step [gpu_utilization, memory_utilization, power_consumption]
* ``ttf``    — log1p(hours until node failure), capped at the TTF horizon
* ``meta``   — node/timestamps/ground-truth incident flag for evaluation
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEMAND_TARGETS = ("gpu_utilization", "memory_utilization", "power_consumption")


def with_deltas(X: np.ndarray) -> np.ndarray:
    """Append first-difference channels: (N, F, T) → (N, 2F, T).

    Raw channels encode *level*; delta channels encode *trend*. A cooling
    failure ("temperature rising while power is flat") is invisible in levels
    until the value leaves the training range, but it is an immediate,
    unambiguous pattern in the trend channels — which is what lets the model
    fire early instead of at the threshold.
    """
    d = np.diff(X, axis=-1, prepend=X[..., :1])
    return np.concatenate([X, d], axis=1).astype(np.float32)


def channel_names(feature_names: list[str], use_deltas: bool) -> list[str]:
    """Model input channel names (physical features, then trend channels)."""
    if use_deltas:
        return list(feature_names) + [f"{n} (trend)" for n in feature_names]
    return list(feature_names)


@dataclass
class WindowBatch:
    X: np.ndarray                 # (N, F, T) float32
    cls: np.ndarray               # (N,) int64
    demand: np.ndarray            # (N, 3) float32
    ttf: np.ndarray               # (N, 1) float32  — log1p(hours)
    anomaly_gt: np.ndarray        # (N,) float32  — 1 if an incident overlaps the window tail
    meta: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.X)


def _mode_label(labels: np.ndarray, classes: list[str]) -> int:
    ids = np.array([classes.index(x) if x in classes else -1 for x in labels])
    ids = ids[ids >= 0]
    if len(ids) == 0:
        return 0
    return int(np.bincount(ids).argmax())


def build_windows(
    node_id: str,
    df: pd.DataFrame,
    feature_names: list[str],
    classes: list[str],
    size: int,
    stride: int,
    interval_seconds: int,
    ttf_horizon_hours: float,
    failure_step: int | None = None,
) -> WindowBatch:
    """Slice one node's frame into windows with targets."""
    values = df[feature_names].to_numpy(dtype=np.float32)  # (steps, F)
    n_steps = len(df)
    demand_idx = [feature_names.index(c) for c in DEMAND_TARGETS]

    starts = np.arange(0, n_steps - size, stride)  # last window needs end+1 target
    X, cls, demand, ttf, gt, meta = [], [], [], [], [], []
    ttf_horizon_steps = ttf_horizon_hours * 3600 / interval_seconds

    labels = df["label"].to_numpy() if "label" in df else np.array(["idle"] * n_steps)
    incidents = df["incident"].to_numpy() if "incident" in df else np.array([""] * n_steps)

    for s in starts:
        e = s + size - 1
        X.append(values[s:e + 1].T)  # (F, T)
        cls.append(_mode_label(labels[s:e + 1], classes))
        demand.append(values[e + 1, demand_idx])

        if failure_step is not None and failure_step > e:
            hours = min((failure_step - e) * interval_seconds / 3600.0, ttf_horizon_hours)
        else:
            hours = ttf_horizon_hours
        ttf.append(np.log1p(hours))

        # Ground truth: incident present in the last third of the window —
        # this is the region the detector is expected to react to.
        tail = incidents[e - size // 3:e + 1]
        gt.append(float(any(x not in ("", "ghost_job") for x in tail)))
        meta.append({
            "node_id": node_id,
            "start_ts": str(df["timestamp"].iloc[s]),
            "end_ts": str(df["timestamp"].iloc[e]),
            "end_step": int(e),
        })

    return WindowBatch(
        X=np.asarray(X, dtype=np.float32),
        cls=np.asarray(cls, dtype=np.int64),
        demand=np.asarray(demand, dtype=np.float32),
        ttf=np.asarray(ttf, dtype=np.float32).reshape(-1, 1),
        anomaly_gt=np.asarray(gt, dtype=np.float32),
        meta=meta,
    )


def concat_batches(batches: list[WindowBatch]) -> WindowBatch:
    """Concatenate per-node batches into one training batch."""
    meta = [m for b in batches for m in b.meta]
    return WindowBatch(
        X=np.concatenate([b.X for b in batches]),
        cls=np.concatenate([b.cls for b in batches]),
        demand=np.concatenate([b.demand for b in batches]),
        ttf=np.concatenate([b.ttf for b in batches]),
        anomaly_gt=np.concatenate([b.anomaly_gt for b in batches]),
        meta=meta,
    )
