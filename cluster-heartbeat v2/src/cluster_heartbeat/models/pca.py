"""PCA heartbeat — the lightweight Phase-1 baseline from the project plan.

Implements the same embed/reconstruct contract as :class:`FingerprintNet` so
the inference pipeline and API work unchanged with either backend. Behavior
classification and demand/TTF prediction are neural-only; with the PCA backend
the services fall back to documented heuristics.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAHeartbeat:
    backend = "pca"

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    # -- training ----------------------------------------------------------
    def fit(self, X_flat: np.ndarray) -> "PCAHeartbeat":
        self.pca.fit(X_flat)
        return self

    # -- inference ----------------------------------------------------------
    def embed(self, X_flat: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_flat).astype(np.float32)

    def reconstruct(self, X_flat: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(self.pca.transform(X_flat)).astype(np.float32)

    def errors(self, X_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature and aggregate reconstruction error."""
        n_features = X_flat.shape[1] // 1  # flat layout: (F * T), F from reshape below
        recon = self.reconstruct(X_flat)
        per_feature = (recon - X_flat) ** 2
        return per_feature, per_feature.mean(axis=1)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        joblib.dump({"n_components": self.n_components, "pca": self.pca}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PCAHeartbeat":
        payload = joblib.load(path)
        obj = cls(payload["n_components"])
        obj.pca = payload["pca"]
        return obj
