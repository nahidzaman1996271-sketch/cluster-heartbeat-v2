"""Step 4 — Feature normalization.

Per-feature z-score normalization computed on the *training* windows only.
Parameters persist next to the model checkpoint so inference applies the
identical transform (and can invert predictions back to physical units).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class FeatureNormalizer:
    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.mean_: np.ndarray | None = None  # (F,)
        self.std_: np.ndarray | None = None

    # -- fitting ---------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        """Fit on windows of shape (N, F, T)."""
        flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])  # (N*T, F)
        self.mean_ = flat.mean(axis=0)
        self.std_ = np.maximum(flat.std(axis=0), 1e-6)
        return self

    # -- transforms ------------------------------------------------------
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return ((X - self.mean_[:, None]) / self.std_[:, None]).astype(np.float32)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check()
        out = df.copy()
        out[self.feature_names] = (
            (df[self.feature_names] - self.mean_) / self.std_
        ).astype(np.float32)
        return out

    def scale_demand(self, y: np.ndarray, demand_idx: list[int]) -> np.ndarray:
        self._check()
        return ((y - self.mean_[demand_idx]) / self.std_[demand_idx]).astype(np.float32)

    def inverse_demand(self, y: np.ndarray, demand_idx: list[int]) -> np.ndarray:
        self._check()
        return y * self.std_[demand_idx] + self.mean_[demand_idx]

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"feature_names": self.feature_names, "mean": self.mean_, "std": self.std_},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        payload = joblib.load(path)
        obj = cls(payload["feature_names"])
        obj.mean_, obj.std_ = payload["mean"], payload["std"]
        return obj

    def _check(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("FeatureNormalizer is not fitted.")
