"""FastAPI application exposing fingerprints, scores and recommendations.

Run::

    python scripts/serve.py --ckpt checkpoints/best --data data/synthetic

or::

    CH_CKPT_DIR=checkpoints/best CH_DATA_DIR=data/synthetic \
        uvicorn cluster_heartbeat.api.main:app --port 8000
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from .. import __version__
from ..config import load_config
from ..data.ingestion import IngestResult
from ..data.preprocessing import clean_node_frame
from ..inference.pipeline import HeartbeatPipeline
from ..logging_utils import get_logger
from .schemas import HealthResponse, InferenceRequest

logger = get_logger(__name__)

REQUESTS = Counter("cluster_heartbeat_requests_total",
                   "API requests", ["endpoint"])


def create_app(
    ckpt_dir: str = "checkpoints/best",
    data_dir: str | None = "data/synthetic",
    config_path: str = "configs/default.yaml",
) -> FastAPI:
    app = FastAPI(
        title="Cluster Heartbeat API",
        version=__version__,
        description=(
            "Unified GPU-cluster workload fingerprints powering predictive "
            "failure detection, smart scheduling and cost optimization. "
            "All payloads are dashboard-ready JSON."
        ),
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # -- state --------------------------------------------------------------
    app.state.pipeline = None
    app.state.report = None
    app.state.histories = {}
    app.state.feature_specs = None
    app.state.frames = {}

    if Path(ckpt_dir).exists():
        try:
            app.state.pipeline = HeartbeatPipeline(ckpt_dir)
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("failed to load checkpoint at %s", ckpt_dir)

    if app.state.pipeline is not None and data_dir and Path(data_dir).exists():
        try:
            result = IngestResult.load(data_dir)
            cfg = load_config(config_path)
            from ..config import load_feature_registry as _lfr
            specs, _ = _lfr("configs/features.yaml")
            frames = {
                n: clean_node_frame(df, specs, cfg.data.synthetic.interval_seconds)
                for n, df in result.frames.items()
            }
            started = time.perf_counter()
            app.state.report = app.state.pipeline.process_cluster(frames)
            app.state.histories = {
                n: app.state.pipeline.process_node(n, df) for n, df in frames.items()
            }
            app.state.frames = frames
            logger.info("cluster report built in %.1fs", time.perf_counter() - started)
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("failed to build cluster report from %s", data_dir)

    # -- helpers -------------------------------------------------------------
    def pipeline() -> HeartbeatPipeline:
        if app.state.pipeline is None:
            raise HTTPException(503, "model checkpoint not loaded — train first")
        return app.state.pipeline

    def report() -> dict:
        if app.state.report is None:
            raise HTTPException(503, "cluster report not built — provide --data")
        return app.state.report

    # -- meta ----------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        REQUESTS.labels("/health").inc()
        return HealthResponse(
            status="ok",
            model_loaded=app.state.pipeline is not None,
            backend=app.state.pipeline.backend if app.state.pipeline else None,
            report_loaded=app.state.report is not None,
            version=__version__,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # -- cluster --------------------------------------------------------------
    @app.get("/api/v1/cluster/summary")
    def cluster_summary():
        REQUESTS.labels("summary").inc()
        r = report()
        return {**r["cluster"], "generated_at": r["generated_at"], "model": r["model"]}

    @app.get("/api/v1/cluster/timeseries")
    def cluster_timeseries():
        REQUESTS.labels("timeseries").inc()
        return report()["timeseries"]

    @app.get("/api/v1/dashboard")
    def dashboard():
        """Everything a dashboard needs in one round trip."""
        REQUESTS.labels("dashboard").inc()
        return report()

    # -- nodes -----------------------------------------------------------------
    @app.get("/api/v1/nodes")
    def nodes():
        REQUESTS.labels("nodes").inc()
        return report()["nodes"]

    @app.get("/api/v1/nodes/{node_id}")
    def node_detail(node_id: str):
        REQUESTS.labels("node_detail").inc()
        r = report()
        if node_id not in r["nodes"]:
            raise HTTPException(404, f"unknown node {node_id!r}")
        node_alerts = [a for a in r["alerts"] if a["node_id"] == node_id]
        return {**r["nodes"][node_id], "node_id": node_id, "alerts": node_alerts}

    @app.get("/api/v1/nodes/{node_id}/fingerprint")
    def node_fingerprint(
        node_id: str,
        tail: int = Query(1, ge=1, le=200,
                          description="how many recent fingerprints to return"),
    ):
        REQUESTS.labels("fingerprint").inc()
        history = app.state.histories.get(node_id)
        if not history:
            raise HTTPException(404, f"no fingerprint history for {node_id!r}")
        return [fp.to_dict() for fp in history[-tail:]]

    @app.get("/api/v1/nodes/{node_id}/shap_explain")
    def node_shap_explain(node_id: str):
        """Second-opinion, model-agnostic SHAP attribution for a node's
        current anomaly score. Slower than the default explanation shipped
        on every fingerprint — computed on demand, not cached in history.
        """
        REQUESTS.labels("shap_explain").inc()
        pipe = pipeline()
        frames = app.state.frames
        if node_id not in frames:
            raise HTTPException(404, f"no telemetry frame for {node_id!r}")

        from ..inference.shap_explain import explain_with_shap

        target_df = frames[node_id].iloc[-pipe.window_size:]
        if len(target_df) < pipe.window_size:
            raise HTTPException(422, f"not enough history for {node_id!r}")

        background = [
            df.iloc[-pipe.window_size:]
            for n, df in frames.items()
            if n != node_id and len(df) >= pipe.window_size
        ]
        try:
            result = explain_with_shap(pipe, node_id, target_df, background)
        except RuntimeError as e:
            raise HTTPException(501, str(e)) from e
        return result.to_dict()

    # -- alerts & recommendations ------------------------------------------------
    @app.get("/api/v1/alerts")
    def alerts(severity: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")):
        REQUESTS.labels("alerts").inc()
        items = report()["alerts"]
        if severity:
            items = [a for a in items if a["severity"] == severity]
        return items

    @app.get("/api/v1/recommendations/scheduling")
    def scheduling_recommendation(
        gpu_request: float = Query(0.0, ge=0, le=100),
        mem_request: float = Query(0.0, ge=0, le=100),
    ):
        """Rank nodes for a new workload (Scheduler Framework Score input)."""
        REQUESTS.labels("scheduling").inc()
        from ..services import scheduling

        latest = {n: fps[-1].to_dict() for n, fps in app.state.histories.items() if fps}
        if not latest:
            raise HTTPException(503, "no fingerprints available")
        cfg = load_config(config_path)
        return scheduling.recommend(
            latest, gpu_request=gpu_request, mem_request=mem_request,
            avoid_risk=cfg.scoring.failure["risk_alert_threshold"],
        )

    @app.get("/api/v1/recommendations/cost")
    def cost_recommendation():
        REQUESTS.labels("cost").inc()
        return report()["recommendations"]["cost"]

    # -- ad-hoc inference ----------------------------------------------------------
    @app.post("/api/v1/inference")
    def inference(req: InferenceRequest):
        """Score one telemetry window submitted by a caller (e.g. a collector)."""
        REQUESTS.labels("inference").inc()
        pipe = pipeline()
        df = pd.DataFrame(req.records)
        missing = [f for f in pipe.feature_names if f not in df.columns]
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="60s")
        if len(df) != pipe.window_size:
            raise HTTPException(
                422, f"expected exactly {pipe.window_size} rows, got {len(df)}")
        fp = pipe.process_window(req.node_id, df)
        return fp.to_dict()

    return app


# Module-level app for `uvicorn cluster_heartbeat.api.main:app`.
app = create_app(
    ckpt_dir=os.environ.get("CH_CKPT_DIR", "checkpoints/best"),
    data_dir=os.environ.get("CH_DATA_DIR", "data/synthetic"),
    config_path=os.environ.get("CH_CONFIG", "configs/default.yaml"),
)
