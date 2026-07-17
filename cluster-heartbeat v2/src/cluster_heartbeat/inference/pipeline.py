"""Steps 6-11 — end-to-end inference pipeline.

Window in → unified workload fingerprint out, plus fleet-level orchestration
that feeds all three services and produces dashboard-ready JSON.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.windows import DEMAND_TARGETS
from ..logging_utils import get_logger
from ..training.checkpoint import CheckpointBundle, load_checkpoint
from ..utils.helpers import get_device, now_iso
from . import explain
from .scores import (
    aggregate_z,
    anomaly_score_from_z,
    cluster_health,
    failure_risk,
    gpu_health,
)

logger = get_logger(__name__)


@dataclass
class Fingerprint:
    """The unified workload fingerprint for one node, one window."""

    node_id: str
    window_end_ts: str
    end_step: int
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
    raw_latest: dict
    raw_window_means: dict

    def to_dict(self) -> dict:
        return asdict(self)


class HeartbeatPipeline:
    """Loads a checkpoint once and turns telemetry windows into fingerprints."""

    def __init__(self, ckpt_dir: str | Path, device: str = "auto"):
        self.device = get_device(device)
        self.bundle: CheckpointBundle = load_checkpoint(ckpt_dir, self.device)
        p = self.bundle.payload
        self.backend = self.bundle.backend
        self.feature_names: list[str] = p["feature_names"]
        self.class_names: list[str] = p.get(
            "class_names", ["idle", "compute_bound", "memory_bound",
                            "io_bound", "network_bound", "mixed"])
        self.window_size: int = int(p["window_size"])
        self.use_deltas: bool = bool(p.get("use_deltas", False))
        self.channel_names: list[str] = p.get("channel_names", self.feature_names)
        self.stats: dict = p["stats"]
        self.cfg: dict = p.get("config", {})
        self.scoring = self.cfg.get("scoring", {})
        self.demand_idx = [self.feature_names.index(c) for c in DEMAND_TARGETS]
        logger.info("pipeline ready: backend=%s ckpt=%s device=%s",
                    self.backend, ckpt_dir, self.device)

    # ------------------------------------------------------------------
    def process_window(self, node_id: str, window_df: pd.DataFrame,
                       end_step: int | None = None) -> Fingerprint:
        """One cleaned window (window_size rows, raw units) → fingerprint."""
        if len(window_df) != self.window_size:
            raise ValueError(
                f"expected {self.window_size} rows, got {len(window_df)}"
            )
        normalizer = self.bundle.normalizer
        raw = window_df[self.feature_names].to_numpy(dtype=np.float32)  # (T, F)
        x_scaled = ((raw - normalizer.mean_) / normalizer.std_).T        # (F, T)

        if self.backend == "pca":
            result = self._pca_forward(x_scaled)
        else:
            from ..data.windows import with_deltas

            x_model = with_deltas(x_scaled[None]) if self.use_deltas else x_scaled[None]
            result = self._nn_forward(x_model)

        if "per_feature_mean" in self.stats:
            z_agg = aggregate_z(result["per_feature_err"],
                                self.stats["per_feature_mean"],
                                self.stats["per_feature_std"])
            if "agg_z_mean" in self.stats:  # calibrate against healthy baseline
                z_agg = (z_agg - self.stats["agg_z_mean"]) / self.stats["agg_z_std"]
        else:  # PCA baseline ships aggregate stats only
            z_agg = (result["agg_err"] - self.stats["err_mean"]) / self.stats["err_std"]
        anomaly = anomaly_score_from_z(
            float(z_agg), self.scoring.get("anomaly", {}).get("z_full_scale", 6.0))

        n_channels = len(result["per_feature_err"])
        per_feat_z = (
            (result["per_feature_err"] - np.array(self.stats.get("per_feature_mean", np.zeros(n_channels))))
            / np.array(self.stats.get("per_feature_std", np.ones(n_channels)))
        )
        top_feats = explain.top_anomaly_features(per_feat_z, self.channel_names)
        ovs_exp = explain.observed_vs_expected(
            x_scaled, result["recon"][:len(self.feature_names)],
            normalizer.mean_, normalizer.std_, self.feature_names)

        # --- raw context from the window itself --------------------------
        raw_latest = {f: float(raw[-1, i]) for i, f in enumerate(self.feature_names)}
        means = {f: float(raw[:, i].mean()) for i, f in enumerate(self.feature_names)}
        tail = raw[-5:]
        ecc_rate = float(tail[:, self.feature_names.index("ecc_errors")].mean())
        xid_present = bool((tail[:, self.feature_names.index("xid_errors")] > 0).any())

        ttf_hours = result["ttf_hours"]
        risk, risk_comp = failure_risk(
            anomaly, ecc_rate, xid_present, ttf_hours,
            self.cfg.get("window", {}).get("ttf_horizon_hours", 48.0))
        health = gpu_health(
            raw_latest, anomaly, ecc_rate, xid_present,
            temp_warn=self.scoring.get("temperature", {}).get("warn_celsius", 80.0),
            temp_crit=self.scoring.get("temperature", {}).get("crit_celsius", 90.0))

        reasons = (explain.build_reasons(anomaly, top_feats, risk, ttf_hours)
                   + explain.classification_reasons(means, result["cls_label"])
                   + health.reasons)

        return Fingerprint(
            node_id=node_id,
            window_end_ts=str(window_df["timestamp"].iloc[-1]),
            end_step=int(end_step if end_step is not None else len(window_df) - 1),
            embedding=[round(v, 4) for v in result["latent"].tolist()],
            anomaly_score=round(anomaly, 3),
            anomaly_features=top_feats,
            observed_vs_expected=ovs_exp,
            classification={
                "label": result["cls_label"],
                "confidence": round(float(result["cls_conf"]), 3),
                "probabilities": {
                    c: round(float(p), 3)
                    for c, p in zip(self.class_names, result["cls_probs"])
                },
            },
            demand={k: round(float(v), 2) for k, v in result["demand"].items()},
            ttf_hours=round(ttf_hours, 2),
            failure_risk=risk,
            risk_components=risk_comp,
            gpu_health=health.value,
            health_penalties=health.penalties,
            reasons=reasons,
            raw_latest=raw_latest,
            raw_window_means=means,
        )

    # ------------------------------------------------------------------
    def _nn_forward(self, x_model: np.ndarray) -> dict:
        """x_model: (1, C, T) — normalized and (optionally) delta-augmented."""
        from ..models.fingerprint import reconstruction_errors

        x = torch.from_numpy(x_model).to(self.device)
        out = self.bundle.model.fingerprint(x)
        per_feat, agg = reconstruction_errors(out["recon"], x)
        demand_scaled = out["demand"][0].cpu().numpy()
        demand = self.bundle.normalizer.inverse_demand(demand_scaled, self.demand_idx)
        probs = out["cls_probs"][0].cpu().numpy()
        horizon = self.cfg.get("window", {}).get("ttf_horizon_hours", 48.0)
        return {
            "latent": out["latent"][0].cpu().numpy(),
            "recon": out["recon"][0].cpu().numpy(),
            "per_feature_err": per_feat[0].cpu().numpy(),
            "agg_err": float(agg[0].cpu()),
            "cls_label": self.class_names[int(probs.argmax())],
            "cls_conf": float(probs.max()),
            "cls_probs": probs,
            "demand": dict(zip(DEMAND_TARGETS, demand)),
            "ttf_hours": float(np.clip(np.expm1(float(out["ttf"][0, 0].cpu())), 0.0, horizon)),
        }

    def _pca_forward(self, x_scaled: np.ndarray) -> dict:
        """PCA backend: embedding + anomaly; heads use documented fallbacks."""
        flat = x_scaled.flatten()[None]
        recon_flat = self.bundle.model.reconstruct(flat)[0]
        per_col = (recon_flat - flat[0]) ** 2
        per_feat = per_col.reshape(len(self.feature_names), self.window_size).mean(axis=1)
        means = {f: float(x_scaled[i].mean()) for i, f in enumerate(self.feature_names)}
        # Fallback behavior classification from window means.
        if means["gpu_utilization"] < 5:
            label = "idle"
        elif means["gpu_utilization"] > 75:
            label = "compute_bound"
        elif means["memory_utilization"] > 75:
            label = "memory_bound"
        else:
            label = "mixed"
        probs = np.array([1.0 if c == label else 0.0 for c in self.class_names])
        horizon = self.cfg.get("window", {}).get("ttf_horizon_hours", 48.0)
        return {
            "latent": self.bundle.model.embed(flat)[0],
            "recon": recon_flat.reshape(len(self.feature_names), self.window_size),
            "per_feature_err": per_feat,
            "agg_err": float(per_col.mean()),
            "cls_label": label,
            "cls_conf": 1.0,
            "cls_probs": probs,
            # Demand fallback: persistence forecast (next step = last step).
            "demand": {k: float(x_scaled[idx, -1] * self.bundle.normalizer.std_[idx]
                                + self.bundle.normalizer.mean_[idx])
                       for k, idx in zip(DEMAND_TARGETS, self.demand_idx)},
            "ttf_hours": horizon,
        }

    # ------------------------------------------------------------------
    def process_node(self, node_id: str, df: pd.DataFrame,
                     stride: int | None = None) -> list[Fingerprint]:
        """Slide across a node's full frame → chronologically ordered fingerprints."""
        stride = stride or int(self.cfg.get("window", {}).get("stride", 5))
        fps = []
        for start in range(0, len(df) - self.window_size + 1, stride):
            window = df.iloc[start:start + self.window_size]
            fps.append(self.process_window(node_id, window, end_step=start + self.window_size - 1))
        return fps

    # ------------------------------------------------------------------
    def process_cluster(self, frames: dict[str, pd.DataFrame]) -> dict:
        """Fleet orchestration → dashboard-ready JSON (steps 9-11)."""
        from ..services import cost_optimization, failure_detection, scheduling

        histories: dict[str, list[dict]] = {}
        for node, df in sorted(frames.items()):
            histories[node] = [fp.to_dict() for fp in self.process_node(node, df)]

        latest = {n: fps[-1] for n, fps in histories.items() if fps}
        if not latest:
            raise ValueError("no fingerprints produced — check input data length")

        scoring = self.scoring
        alerts = failure_detection.detect_alerts(
            latest,
            risk_alert_threshold=scoring.get("failure", {}).get("risk_alert_threshold", 0.6),
            temp_warn=scoring.get("temperature", {}).get("warn_celsius", 80.0),
            temp_crit=scoring.get("temperature", {}).get("crit_celsius", 90.0),
        )
        schedule = scheduling.recommend(
            latest,
            avoid_risk=scoring.get("failure", {}).get("risk_alert_threshold", 0.6),
        )
        interval = self.cfg.get("data", {}).get("synthetic", {}).get("interval_seconds", 60)
        stride = int(self.cfg.get("window", {}).get("stride", 5))
        cost = cost_optimization.idle_report(
            histories, window_stride_seconds=stride * interval,
            idle_cfg=scoring.get("idle", {}), cost_cfg=scoring.get("cost", {}),
        )

        n_crit = sum(1 for a in alerts if a["severity"] == "CRITICAL")
        ch = cluster_health([fp["gpu_health"] for fp in latest.values()], n_crit)

        # Cluster time series, aligned on the shared window-end timestamps.
        ts_buckets: dict[str, list[dict]] = {}
        for fps in histories.values():
            for fp in fps:
                ts_buckets.setdefault(fp["window_end_ts"], []).append(fp)
        series_health, series_util, series_risk = [], [], []
        for ts in sorted(ts_buckets):
            bucket = ts_buckets[ts]
            series_health.append({
                "ts": ts,
                "value": round(cluster_health(
                    [f["gpu_health"] for f in bucket],
                    sum(1 for f in bucket if f["failure_risk"] >= 0.85)), 1),
            })
            series_util.append({
                "ts": ts,
                "value": round(float(np.mean(
                    [f["raw_window_means"]["gpu_utilization"] for f in bucket])), 1),
            })
            series_risk.append({
                "ts": ts,
                "value": round(max(f["failure_risk"] for f in bucket), 3),
            })

        return {
            "generated_at": now_iso(),
            "model": {
                "backend": self.backend,
                "version": self.bundle.payload.get("version"),
                "embedding_dim": len(next(iter(latest.values()))["embedding"]),
            },
            "cluster": {
                "health_score": ch,
                "nodes_total": len(latest),
                "nodes_healthy": sum(1 for fp in latest.values() if fp["gpu_health"] >= 80),
                "nodes_at_risk": sum(
                    1 for fp in latest.values()
                    if fp["failure_risk"] >= scoring.get("failure", {}).get("risk_alert_threshold", 0.6)),
                "active_alerts": len(alerts),
                "critical_alerts": n_crit,
                "avg_gpu_utilization_pct": round(float(np.mean(
                    [fp["raw_window_means"]["gpu_utilization"] for fp in latest.values()])), 1),
                "estimated_reclaimable_usd": cost["estimated_reclaimable_usd"],
            },
            "timeseries": {
                "cluster_health": series_health,
                "avg_gpu_utilization": series_util,
                "max_failure_risk": series_risk,
            },
            "nodes": {
                n: {
                    "gpu_health": fp["gpu_health"],
                    "failure_risk": fp["failure_risk"],
                    "ttf_hours": fp["ttf_hours"],
                    "anomaly_score": fp["anomaly_score"],
                    "classification": fp["classification"],
                    "demand_prediction": fp["demand"],
                    "embedding": fp["embedding"],
                    "reasons": fp["reasons"],
                }
                for n, fp in latest.items()
            },
            "alerts": alerts,
            "recommendations": {"scheduling": schedule, "cost": cost},
        }
