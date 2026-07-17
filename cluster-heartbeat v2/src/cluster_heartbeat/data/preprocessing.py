"""Step 2 — Cleaning and preprocessing.

* enforce a regular time grid per node,
* interpolate short sensor gaps (<= 3 samples), forward/back-fill the rest,
* clip values to physical validity ranges from the feature registry,
* de-spike analog sensors (temperature, power) with a rolling-median filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureSpec

_ANALOG_DESPIKE = {"gpu_temperature", "power_consumption"}


def clean_node_frame(
    df: pd.DataFrame,
    features: list[FeatureSpec],
    interval_seconds: int,
    max_interp_gap: int = 3,
) -> pd.DataFrame:
    """Clean one node's telemetry frame in place-safe fashion."""
    names = [f.name for f in features]
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Regular time grid.
    full_index = pd.date_range(
        df["timestamp"].iloc[0], df["timestamp"].iloc[-1],
        freq=f"{interval_seconds}s",
    )
    annotations = df[["timestamp", "label", "incident"]].copy() if "label" in df else None
    df = df.set_index("timestamp")[names].reindex(full_index)
    df.index.name = "timestamp"

    # Gap filling: interpolate short gaps, then ffill/bfill.
    df = df.interpolate(limit=max_interp_gap, limit_area="inside")
    df = df.ffill().bfill()

    # Physical validity clipping.
    for spec in features:
        lo = -np.inf if spec.min is None else spec.min
        hi = np.inf if spec.max is None else spec.max
        df[spec.name] = df[spec.name].clip(lo, hi)

    # De-spike analog sensors: replace points > 4*MAD from rolling median.
    for col in _ANALOG_DESPIKE & set(names):
        med = df[col].rolling(5, center=True, min_periods=1).median()
        mad = (df[col] - med).abs().rolling(5, center=True, min_periods=1).median()
        spike = (df[col] - med).abs() > np.maximum(4 * mad, 2.0)
        df.loc[spike, col] = med[spike]

    df = df.reset_index()
    if annotations is not None:
        annotations = annotations.set_index("timestamp").reindex(full_index)
        annotations["label"] = annotations["label"].ffill().bfill().fillna("unknown")
        annotations["incident"] = annotations["incident"].fillna("")
        df["label"] = annotations["label"].to_numpy()
        df["incident"] = annotations["incident"].to_numpy()
    return df


def clean_all(
    frames: dict[str, pd.DataFrame],
    features: list[FeatureSpec],
    interval_seconds: int,
) -> dict[str, pd.DataFrame]:
    return {
        node: clean_node_frame(df, features, interval_seconds)
        for node, df in frames.items()
    }
