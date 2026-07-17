"""Pillar 1 — Predictive failure detection.

Consumes the latest fingerprint per node and emits operator-ready alerts
*before* thresholds are crossed, with estimated time to failure and a
recommended action (drain / migrate / inspect).
"""
from __future__ import annotations

from ..inference.scores import severity
from ..utils.helpers import now_iso


def detect_alerts(
    latest: dict[str, dict],
    risk_alert_threshold: float = 0.6,
    temp_warn: float = 80.0,
    temp_crit: float = 90.0,
) -> list[dict]:
    """Build the alert list from the latest per-node fingerprints."""
    alerts: list[dict] = []

    for node, fp in sorted(latest.items()):
        risk = fp["failure_risk"]
        raw = fp["raw_latest"]
        temp = raw.get("gpu_temperature", 0.0)

        if risk >= risk_alert_threshold:
            drivers = ", ".join(f["feature"] for f in fp["anomaly_features"][:3])
            alerts.append({
                "id": f"{node}:predictive_failure",
                "node_id": node,
                "type": "predictive_failure",
                "severity": severity(risk),
                "message": (
                    f"Failure risk {risk:.2f} on {node}; telemetry drift driven by "
                    f"{drivers}. Estimated time to failure: "
                    f"{fp['ttf_hours']:.1f} h."
                ),
                "evidence": {
                    "failure_risk": risk,
                    "risk_components": fp["risk_components"],
                    "anomaly_score": fp["anomaly_score"],
                    "top_anomaly_features": fp["anomaly_features"],
                    "observed_vs_expected": fp["observed_vs_expected"],
                },
                "recommended_action": (
                    "Cordon and drain the node; migrate GPU workloads to a "
                    "healthy node (see scheduling recommendations)."
                ),
                "ts": fp["window_end_ts"],
            })

        if temp >= temp_crit:
            alerts.append({
                "id": f"{node}:thermal_critical",
                "node_id": node,
                "type": "thermal_critical",
                "severity": "CRITICAL",
                "message": f"GPU temperature {temp:.0f}°C ≥ {temp_crit:.0f}°C on {node}.",
                "evidence": {"gpu_temperature": temp},
                "recommended_action": "Throttle or evacuate workloads immediately; check cooling.",
                "ts": fp["window_end_ts"],
            })
        elif temp >= temp_warn:
            alerts.append({
                "id": f"{node}:thermal_warning",
                "node_id": node,
                "type": "thermal_warning",
                "severity": "MEDIUM",
                "message": f"GPU temperature {temp:.0f}°C above warning level on {node}.",
                "evidence": {"gpu_temperature": temp},
                "recommended_action": "Watch closely; verify airflow and fan curves.",
                "ts": fp["window_end_ts"],
            })

        if raw.get("xid_errors", 0) > 0:
            alerts.append({
                "id": f"{node}:xid_error",
                "node_id": node,
                "type": "xid_error",
                "severity": "HIGH",
                "message": f"XID error code {int(raw['xid_errors'])} reported on {node}.",
                "evidence": {"xid_errors": raw["xid_errors"], "ecc_errors": raw.get("ecc_errors", 0)},
                "recommended_action": "Check dmesg/DCGM diagnostics; schedule node for inspection.",
                "ts": fp["window_end_ts"],
            })

    alerts.sort(key=lambda a: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[a["severity"]])
    return alerts
