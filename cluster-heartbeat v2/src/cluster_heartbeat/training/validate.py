"""Validation / evaluation for a trained FingerprintNet.

Computes, on the held-out *nodes*:

* classification accuracy / macro-F1 (workload behavior)
* demand regression MAE/RMSE in physical units (%, watts)
* TTF regression error in hours
* anomaly-detection AUROC and best-F1 operating point
* mean predictive lead time before injected node failures — the headline
  metric for the "catch failures before they happen" claim
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import Config
from ..data.normalization import FeatureNormalizer
from ..data.windows import DEMAND_TARGETS, WindowBatch
from ..inference.scores import aggregate_z, anomaly_score_from_z
from ..models.fingerprint import FingerprintNet, reconstruction_errors
from .metrics import (
    anomaly_metrics,
    classification_metrics,
    failure_lead_times,
    per_target_regression,
    regression_metrics,
)


@torch.no_grad()
def evaluate_model(
    model: FingerprintNet,
    batch: WindowBatch,
    normalizer: FeatureNormalizer,
    stats: dict,
    cfg: Config,
    classes: list[str],
    failures: dict,
    device: torch.device,
) -> dict:
    X_np = normalizer.transform_X(batch.X)
    if cfg.model.use_deltas:
        from ..data.windows import with_deltas

        X_np = with_deltas(X_np)
    X = torch.from_numpy(X_np)
    demand_idx = [normalizer.feature_names.index(c) for c in DEMAND_TARGETS]
    loader = DataLoader(TensorDataset(X), batch_size=512)

    model.eval()
    per_feat, cls_p, demand_p, ttf_p = [], [], [], []
    for (xb,) in loader:
        out = model(xb.to(device))
        pf, _ = reconstruction_errors(out["recon"], xb.to(device))
        per_feat.append(pf.cpu().numpy())
        cls_p.append(out["cls_logits"].argmax(dim=-1).cpu().numpy())
        demand_p.append(out["demand"].cpu().numpy())
        ttf_p.append(out["ttf"].cpu().numpy())

    per_feat = np.concatenate(per_feat)
    cls_pred = np.concatenate(cls_p)
    demand_pred = normalizer.inverse_demand(np.concatenate(demand_p), demand_idx)
    ttf_pred_h = np.clip(np.expm1(np.concatenate(ttf_p).ravel()), 0, cfg.window.ttf_horizon_hours)
    ttf_true_h = np.clip(np.expm1(batch.ttf.ravel()), 0, cfg.window.ttf_horizon_hours)

    z_full = cfg.scoring.anomaly["z_full_scale"]
    scores = np.array([
        anomaly_score_from_z(
            (aggregate_z(pf, stats["per_feature_mean"], stats["per_feature_std"])
             - stats["agg_z_mean"]) / stats["agg_z_std"],
            z_full)
        for pf in per_feat
    ])

    # Lead time per failing validation node.
    interval = cfg.data.synthetic.interval_seconds
    threshold = cfg.scoring.anomaly["alert_threshold"]
    end_steps = np.array([m["end_step"] for m in batch.meta])
    node_ids = np.array([m["node_id"] for m in batch.meta])
    lead_times = {}
    for node, failure in failures.items():
        mask = node_ids == node
        if mask.sum() == 0:
            continue
        lt = failure_lead_times(
            scores[mask], end_steps[mask], failure["failure_step"], threshold, interval
        )
        if lt is not None:
            lead_times[node] = lt

    return {
        "backend": "autoencoder",
        "n_windows": int(len(batch)),
        "n_val_nodes": int(len(set(node_ids))),
        "classification": classification_metrics(batch.cls, cls_pred, classes),
        "demand": per_target_regression(batch.demand, demand_pred, list(DEMAND_TARGETS)),
        "ttf_hours": regression_metrics(ttf_true_h, ttf_pred_h),
        "anomaly": anomaly_metrics(scores, batch.anomaly_gt),
        "predictive_lead_time_hours": {
            "mean": float(np.mean(list(lead_times.values()))) if lead_times else None,
            "per_node": lead_times,
        },
    }
