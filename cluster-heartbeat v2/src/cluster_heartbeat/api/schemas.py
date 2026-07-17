"""Pydantic schemas for the Cluster Heartbeat API.

Response payloads are plain JSON shaped for Grafana (JSON API / Infinity
datasource) and React dashboards: time-series arrays of ``{ts, value}``,
stat-panel scalars, alert tables and recommendation objects.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TimeSeriesPoint(BaseModel):
    ts: str
    value: float


class InferenceRequest(BaseModel):
    """One telemetry window for ad-hoc inference.

    ``records`` must contain exactly ``window_size`` rows; each row carries
    the 13 canonical features (missing features are rejected with 422).
    """

    node_id: str = Field(examples=["gpu-node-7"])
    records: list[dict] = Field(
        description="window_size telemetry rows with the 13 canonical features"
    )


class FingerprintResponse(BaseModel):
    node_id: str
    window_end_ts: str
    embedding: list[float]
    anomaly_score: float
    anomaly_features: list[dict]
    observed_vs_expected: list[dict]
    classification: dict
    demand: dict
    ttf_hours: float
    failure_risk: float
    risk_components: dict
    gpu_health: float
    health_penalties: dict
    reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str | None = None
    report_loaded: bool = False
    version: str
