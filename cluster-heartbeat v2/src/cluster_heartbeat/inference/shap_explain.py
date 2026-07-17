"""Model-agnostic SHAP explainability layer.

The pipeline's default explanations (``inference.explain``) are exact and
free: they read reconstruction error directly off the trained model, so
there's no approximation involved. That's the right default for every
fingerprint the API returns.

This module adds a second, complementary explanation path merged in from
an earlier prototype that used SHAP over a classical (PCA + IsolationForest)
anomaly detector. Here it's adapted to explain the *trained* FingerprintNet
instead: it treats ``window -> anomaly_score`` as a black box and uses
SHAP's KernelExplainer to attribute the score to each of the 13 raw
features (averaged over the window), against a background of healthy
windows.

Why keep both:
  * ``inference.explain`` — instant, exact, always computed, ships on every
    fingerprint.
  * ``shap_explain`` — slower (a few seconds), model-agnostic, useful as a
    second opinion when presenting to someone who wants attribution that
    doesn't depend on trusting the reconstruction-error mechanism itself.

Requires the optional ``shap`` dependency (see requirements.txt). Import is
lazy so the rest of the package works without it installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShapAttribution:
    node_id: str
    window_end_ts: str
    base_value: float
    anomaly_score: float
    contributions: list[dict]  # [{feature, shap_value}], sorted by |value| desc

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "window_end_ts": self.window_end_ts,
            "base_value": round(float(self.base_value), 4),
            "anomaly_score": round(float(self.anomaly_score), 4),
            "contributions": self.contributions,
        }


def _feature_means(window_df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Collapse a (window_size, n_features) frame to one mean-per-feature row."""
    return window_df[feature_names].to_numpy(dtype=np.float32).mean(axis=0)


def _make_score_fn(pipeline, node_id: str, window_df: pd.DataFrame, feature_names: list[str]):
    """Build f(X) -> anomaly_score where X is (n_samples, n_features) of
    per-feature *window means*. Perturbed means are broadcast back across
    the original window's timesteps (last-timestep detail is preserved so
    the model still sees a plausible trajectory, not a flat line).
    """
    base_window = window_df[feature_names].to_numpy(dtype=np.float32)  # (T, F)
    base_means = base_window.mean(axis=0)  # (F,)

    def score_fn(X: np.ndarray) -> np.ndarray:
        scores = np.empty(len(X), dtype=np.float32)
        for i, row in enumerate(X):
            shift = row - base_means  # (F,)
            perturbed = base_window + shift[None, :]  # broadcast shift across T
            df = window_df.copy()
            df[feature_names] = perturbed
            fp = pipeline.process_window(node_id, df)
            scores[i] = fp.anomaly_score
        return scores

    return score_fn, base_means


def explain_with_shap(
    pipeline,
    node_id: str,
    window_df: pd.DataFrame,
    background_windows: list[pd.DataFrame],
    n_background: int = 20,
    nsamples: int = 100,
) -> ShapAttribution:
    """SHAP attribution for one node's current anomaly score.

    Args:
        pipeline: a loaded HeartbeatPipeline.
        node_id: node being explained.
        window_df: the exact window (window_size rows) to explain.
        background_windows: healthy reference windows (same shape) used to
            build the SHAP background distribution.
        n_background: how many background windows to sample (kept small —
            KernelExplainer cost scales with this).
        nsamples: SHAP's internal perturbation sample count.
    """
    try:
        import shap
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "shap is not installed. `pip install shap` to use explain_with_shap()."
        ) from e

    feature_names = pipeline.feature_names
    score_fn, _ = _make_score_fn(pipeline, node_id, window_df, feature_names)

    rng = np.random.default_rng(0)
    bg_pool = background_windows if len(background_windows) <= n_background else [
        background_windows[i] for i in rng.choice(len(background_windows), n_background, replace=False)
    ]
    background = np.stack([_feature_means(w, feature_names) for w in bg_pool])

    explainer = shap.KernelExplainer(score_fn, background)
    target = _feature_means(window_df, feature_names)[None, :]
    shap_values = explainer.shap_values(target, nsamples=nsamples, silent=True)
    shap_values = np.asarray(shap_values).reshape(-1)

    order = np.argsort(np.abs(shap_values))[::-1]
    contributions = [
        {"feature": feature_names[i], "shap_value": round(float(shap_values[i]), 4)}
        for i in order
    ]

    anomaly_score = float(score_fn(target)[0])
    return ShapAttribution(
        node_id=node_id,
        window_end_ts=str(window_df["timestamp"].iloc[-1]),
        base_value=float(explainer.expected_value),
        anomaly_score=anomaly_score,
        contributions=contributions,
    )
