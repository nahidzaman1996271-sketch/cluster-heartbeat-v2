"""Explainable predictions.

Every fingerprint ships with human-readable evidence:

* which features drive the anomaly score (per-feature reconstruction z),
* observed vs. model-expected values in physical units,
* why the classifier picked a workload class,
* plain-language reason strings for scores and alerts.
"""
from __future__ import annotations

import numpy as np


def top_anomaly_features(
    per_feature_z: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> list[dict]:
    """Rank features by reconstruction-error z-score."""
    order = np.argsort(per_feature_z)[::-1][:k]
    return [
        {"feature": feature_names[i], "z": round(float(per_feature_z[i]), 2)}
        for i in order
    ]


def observed_vs_expected(
    x_scaled: np.ndarray,
    recon_scaled: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> list[dict]:
    """Compare the last timestep of a window against its reconstruction.

    Values are returned in physical units so operators can sanity-check them.
    """
    last_obs = x_scaled[:, -1] * std + mean
    last_exp = recon_scaled[:, -1] * std + mean
    denom = np.maximum(np.abs(last_exp), 1e-3)
    delta_pct = (last_obs - last_exp) / denom * 100.0
    order = np.argsort(np.abs(delta_pct))[::-1][:k]
    return [
        {
            "feature": feature_names[i],
            "observed": round(float(last_obs[i]), 2),
            "expected": round(float(last_exp[i]), 2),
            "delta_pct": round(float(delta_pct[i]), 1),
        }
        for i in order
    ]


def classification_reasons(window_means: dict[str, float], label: str) -> list[str]:
    """Why the classifier chose ``label`` — key in-window means."""
    reasons = []
    gpu = window_means.get("gpu_utilization", 0)
    mem = window_means.get("memory_utilization", 0)
    net = window_means.get("network_throughput", 0)
    disk = window_means.get("disk_io", 0)
    if label == "compute_bound":
        reasons.append(f"GPU utilization high (mean {gpu:.0f}%)")
    elif label == "memory_bound":
        reasons.append(f"GPU memory utilization dominant (mean {mem:.0f}%)")
    elif label == "network_bound":
        reasons.append(f"network throughput dominant (mean {net:.0f} MB/s)")
    elif label == "io_bound":
        reasons.append(f"disk I/O dominant (mean {disk:.0f} MB/s)")
    elif label == "idle":
        reasons.append(f"GPU utilization near zero (mean {gpu:.1f}%)")
    else:
        reasons.append(f"balanced load (GPU {gpu:.0f}%, mem {mem:.0f}%)")
    return reasons


def build_reasons(
    anomaly_score: float,
    top_features: list[dict],
    failure_risk_value: float,
    ttf_hours: float,
) -> list[str]:
    """Plain-language summary attached to every fingerprint."""
    reasons: list[str] = []
    if anomaly_score >= 0.7 and top_features:
        drivers = ", ".join(f["feature"] for f in top_features[:3] if f["z"] > 1)
        reasons.append(
            f"Anomalous telemetry shape driven by: {drivers or 'multi-metric drift'}."
        )
    elif anomaly_score >= 0.35:
        reasons.append("Mild drift from healthy baseline — monitoring.")
    else:
        reasons.append("Telemetry shape matches the healthy baseline.")
    if failure_risk_value >= 0.6:
        reasons.append(
            f"Failure risk {failure_risk_value:.2f}; model estimates ~{ttf_hours:.1f} h to failure."
        )
    return reasons
