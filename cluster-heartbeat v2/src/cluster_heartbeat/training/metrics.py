"""Evaluation metrics for all four model tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
    }


def per_target_regression(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    return {
        name: regression_metrics(y_true[:, i], y_pred[:, i])
        for i, name in enumerate(names)
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {c: float(v) for c, v in zip(classes, per_class_f1)},
    }


def anomaly_metrics(scores: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Threshold-independent AUROC plus best-F1 operating point."""
    out: dict = {}
    if ground_truth.sum() in (0, len(ground_truth)):
        out["note"] = "ground truth has a single class; detection metrics skipped"
        return out
    out["auroc"] = float(roc_auc_score(ground_truth, scores))

    best = {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for thr in np.linspace(0.05, 0.95, 19):
        p, r, f1, _ = precision_recall_fscore_support(
            ground_truth, scores >= thr, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(thr),
                    "precision": float(p), "recall": float(r)}
    out["best"] = best
    return out


def failure_lead_times(
    scores: np.ndarray,
    end_steps: np.ndarray,
    failure_step: int,
    threshold: float,
    interval_seconds: int,
    sustain: int = 2,
    max_lead_hours: float = 6.0,
) -> float | None:
    """Hours between the first *sustained* alert and the actual failure.

    A sustained alert means ``sustain`` consecutive windows above threshold —
    this filters one-off spikes, matching how an operator would page on it.
    Detections earlier than ``max_lead_hours`` before the failure are stale
    false positives, not predictions, and are not counted.
    """
    order = np.argsort(end_steps)
    scores, ends = scores[order], end_steps[order]
    max_steps = max_lead_hours * 3600 / interval_seconds
    pre = (ends < failure_step) & (ends >= failure_step - max_steps)
    scores, ends = scores[pre], ends[pre]
    if len(scores) < sustain:
        return None
    for i in range(len(scores) - sustain + 1):
        if np.all(scores[i:i + sustain] >= threshold):
            return float((failure_step - ends[i]) * interval_seconds / 3600.0)
    return None
