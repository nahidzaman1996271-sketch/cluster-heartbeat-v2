"""Transparent, explainable scoring rules.

Every score shipped to the dashboard is a documented function of observable
quantities — no hidden magic. Each returns both the value and its component
breakdown so the UI can show *why*.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def anomaly_score_from_z(z: float, z_full_scale: float = 6.0) -> float:
    """Map a reconstruction-error z-score to [0, 1].

    ``z_full_scale`` is the z at which we consider the window fully anomalous
    (default 6σ). Negative z (better than average reconstruction) clamps to 0.
    """
    return float(min(1.0, max(0.0, z) / z_full_scale))


def aggregate_z(
    per_feature_err,
    per_feature_mean,
    per_feature_std,
    topk: int = 2,
) -> float:
    """Aggregate per-channel reconstruction errors into one z-score.

    Plain MSE averaging dilutes a single deviating sensor across all channels
    (1/26th of the signal). Z-scoring each channel against the healthy
    training distribution and averaging the **top-k** keeps a lone
    misbehaving sensor (an ECC burst, a temperature/load correlation break)
    visible while staying more robust to one-off spikes than a pure max.
    """
    import numpy as np

    err = np.asarray(per_feature_err, dtype=np.float64)
    mean = np.asarray(per_feature_mean, dtype=np.float64)
    std = np.maximum(np.asarray(per_feature_std, dtype=np.float64), 1e-8)
    z = np.maximum(0.0, (err - mean) / std)
    k = max(1, min(topk, len(z)))
    return float(np.sort(z)[::-1][:k].mean())


@dataclass
class HealthReport:
    value: float                      # 0-100
    penalties: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def gpu_health(
    latest: dict[str, float],
    anomaly_score: float,
    ecc_rate: float,
    xid_present: bool,
    temp_warn: float = 80.0,
    temp_crit: float = 90.0,
) -> HealthReport:
    """GPU Health Score: 100 minus weighted, bounded penalty contributions."""
    penalties: dict[str, float] = {}
    reasons: list[str] = []

    temp = latest.get("gpu_temperature", 0.0)
    if temp >= temp_crit:
        penalties["temperature"] = 25.0
        reasons.append(f"temperature {temp:.0f}°C ≥ critical {temp_crit:.0f}°C")
    elif temp >= temp_warn:
        penalties["temperature"] = 10.0 + 15.0 * (temp - temp_warn) / max(temp_crit - temp_warn, 1)
        reasons.append(f"temperature {temp:.0f}°C above warning {temp_warn:.0f}°C")

    if anomaly_score > 0:
        penalties["behavior_anomaly"] = round(45.0 * anomaly_score, 2)
        if anomaly_score >= 0.5:
            reasons.append(f"telemetry shape deviates from healthy baseline (anomaly {anomaly_score:.2f})")

    if ecc_rate > 0:
        penalties["ecc_errors"] = min(15.0, 1.5 * ecc_rate)
        reasons.append(f"ECC error rate elevated ({ecc_rate:.1f}/interval)")

    if xid_present:
        penalties["xid_errors"] = 15.0
        reasons.append("XID error(s) observed in the recent window")

    if latest.get("memory_utilization", 0) >= 97:
        penalties["memory_pressure"] = 5.0
        reasons.append("GPU memory pressure ≥ 97%")

    value = max(0.0, 100.0 - sum(penalties.values()))
    return HealthReport(value=round(value, 1), penalties=penalties, reasons=reasons)


def failure_risk(
    anomaly_score: float,
    ecc_rate: float,
    xid_present: bool,
    ttf_hours: float,
    ttf_horizon_hours: float = 48.0,
) -> tuple[float, dict[str, float]]:
    """Failure Risk Score in [0, 1] with its component breakdown."""
    components = {
        "behavior_anomaly": 0.55 * anomaly_score,
        "ecc_trend": 0.15 * min(1.0, ecc_rate / 20.0),
        "xid_signal": 0.15 * (1.0 if xid_present else 0.0),
        "ttf_estimate": 0.15 * max(0.0, 1.0 - ttf_hours / ttf_horizon_hours),
    }
    risk = min(1.0, sum(components.values()))
    return round(risk, 3), {k: round(v, 3) for k, v in components.items()}


def severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def cluster_health(node_healths: list[float], n_critical_alerts: int) -> float:
    """Cluster Health Score: mean node health, minus 2 points per critical alert."""
    if not node_healths:
        return 0.0
    base = sum(node_healths) / len(node_healths)
    return round(max(0.0, base - 2.0 * n_critical_alerts), 1)
