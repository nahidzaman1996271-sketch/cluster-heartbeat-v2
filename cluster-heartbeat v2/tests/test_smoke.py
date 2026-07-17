"""Smoke tests: shapes, scoring bounds, checkpoint round-trip and the API.

Run from the repository root:  pytest
API tests require a trained checkpoint at checkpoints/best (run
scripts/train.py first); they are skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cluster_heartbeat.data.windows import build_windows, channel_names, with_deltas
from cluster_heartbeat.inference.scores import (
    aggregate_z,
    anomaly_score_from_z,
    cluster_health,
    failure_risk,
    gpu_health,
)
from cluster_heartbeat.models.fingerprint import FingerprintNet, reconstruction_errors

FEATURES = [
    "gpu_utilization", "memory_utilization", "gpu_temperature", "power_consumption",
    "ecc_errors", "xid_errors", "cpu_usage", "ram_usage", "network_throughput",
    "disk_io", "job_runtime", "queue_length", "active_processes",
]
CLASSES = ["idle", "compute_bound", "memory_bound", "io_bound", "network_bound", "mixed"]


def _toy_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=n, freq="60s")
    data = {
        "gpu_utilization": rng.uniform(0, 100, n),
        "memory_utilization": rng.uniform(0, 100, n),
        "gpu_temperature": rng.uniform(40, 90, n),
        "power_consumption": rng.uniform(60, 400, n),
        "ecc_errors": np.zeros(n),
        "xid_errors": np.zeros(n),
        "cpu_usage": rng.uniform(0, 100, n),
        "ram_usage": rng.uniform(0, 100, n),
        "network_throughput": rng.uniform(0, 500, n),
        "disk_io": rng.uniform(0, 200, n),
        "job_runtime": np.arange(n, dtype=float),
        "queue_length": np.zeros(n),
        "active_processes": np.ones(n),
        "label": ["compute_bound"] * n,
        "incident": [""] * n,
    }
    return pd.DataFrame({"timestamp": ts, **data})


def test_window_shapes():
    batch = build_windows(
        "node_t", _toy_frame(), FEATURES, CLASSES,
        size=30, stride=5, interval_seconds=60,
        ttf_horizon_hours=12, failure_step=100,
    )
    n_expected = len(range(0, 120 - 30, 5))
    assert batch.X.shape == (n_expected, 13, 30)
    assert batch.demand.shape == (n_expected, 3)
    assert batch.ttf.shape == (n_expected, 1)
    assert len(batch.meta) == n_expected
    # TTF target is log1p(hours), capped at the horizon.
    assert batch.ttf.max() <= np.log1p(12) + 1e-6


def test_delta_channels():
    X = np.random.rand(4, 13, 30).astype(np.float32)
    aug = with_deltas(X)
    assert aug.shape == (4, 26, 30)
    assert channel_names(FEATURES, True)[13] == "gpu_utilization (trend)"


def test_model_forward():
    model = FingerprintNet(n_features=26, window_size=30, latent_dim=48,
                           hidden_dims=[512, 256], num_classes=6)
    x = torch.randn(8, 26, 30)
    out = model(x)
    assert out["recon"].shape == (8, 26, 30)
    assert out["latent"].shape == (8, 48)
    assert out["cls_logits"].shape == (8, 6)
    assert out["demand"].shape == (8, 3)
    assert out["ttf"].shape == (8, 1)
    pf, agg = reconstruction_errors(out["recon"], x)
    assert pf.shape == (8, 26) and agg.shape == (8,)


def test_score_bounds_and_aggregation():
    z = aggregate_z(np.array([4.0, 0.0]), np.zeros(2), np.ones(2), topk=1)
    assert z == 4.0
    assert 0.0 <= anomaly_score_from_z(z) <= 1.0

    health = gpu_health({"gpu_temperature": 95.0, "memory_utilization": 50.0},
                        anomaly_score=0.5, ecc_rate=2.0, xid_present=True)
    assert 0.0 <= health.value < 60.0  # heavy penalties applied
    assert set(health.penalties) >= {"temperature", "behavior_anomaly", "ecc_errors", "xid_errors"}

    risk, comp = failure_risk(0.8, ecc_rate=10, xid_present=True, ttf_hours=1.0,
                              ttf_horizon_hours=12.0)
    assert 0.0 <= risk <= 1.0
    assert pytest.approx(sum(comp.values()), rel=1e-2) == risk

    assert cluster_health([90, 80, 70], n_critical_alerts=2) == 76.0


# ---------------------------------------------------------------------------
# API tests (need a trained checkpoint)
# ---------------------------------------------------------------------------
CKPT = Path("checkpoints/best")
DATA = Path("data/synthetic")
needs_ckpt = pytest.mark.skipif(
    not (CKPT / "fingerprint_net.pt").exists() or not DATA.exists(),
    reason="train a model and generate data first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from cluster_heartbeat.api.main import create_app

    return TestClient(create_app(str(CKPT), str(DATA), "configs/default.yaml"))


@needs_ckpt
def test_api_health_and_report(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["report_loaded"]

    summary = client.get("/api/v1/cluster/summary").json()
    assert 0 <= summary["health_score"] <= 100
    assert summary["nodes_total"] > 0

    ts = client.get("/api/v1/cluster/timeseries").json()
    assert {"cluster_health", "avg_gpu_utilization", "max_failure_risk"} <= set(ts)
    assert len(ts["cluster_health"]) > 10


@needs_ckpt
def test_api_recommendations_and_fingerprint(client):
    sched = client.get("/api/v1/recommendations/scheduling",
                       params={"gpu_request": 50}).json()
    assert "recommended" in sched and "avoid" in sched

    cost = client.get("/api/v1/recommendations/cost").json()
    assert "idle_or_wasteful_nodes" in cost

    nodes = client.get("/api/v1/nodes").json()
    node = sorted(nodes)[0]
    fps = client.get(f"/api/v1/nodes/{node}/fingerprint", params={"tail": 1}).json()
    fp = fps[0]
    assert len(fp["embedding"]) > 0
    assert 0 <= fp["anomaly_score"] <= 1
    assert 0 <= fp["failure_risk"] <= 1
    assert 0 <= fp["gpu_health"] <= 100


@needs_ckpt
def test_api_inference_validation(client):
    resp = client.post("/api/v1/inference", json={"node_id": "x", "records": []})
    assert resp.status_code == 422


needs_shap = pytest.mark.skipif(
    pytest.importorskip("shap", reason="shap not installed") is None,
    reason="shap not installed",
)


@needs_ckpt
def test_api_shap_explain(client):
    """Model-agnostic SHAP attribution endpoint (merged in from the second
    prototype's explainability approach, adapted to the trained model)."""
    pytest.importorskip("shap")
    nodes = client.get("/api/v1/nodes").json()
    node = sorted(nodes)[0]
    resp = client.get(f"/api/v1/nodes/{node}/shap_explain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == node
    assert len(body["contributions"]) == 13
    assert all("feature" in c and "shap_value" in c for c in body["contributions"])
