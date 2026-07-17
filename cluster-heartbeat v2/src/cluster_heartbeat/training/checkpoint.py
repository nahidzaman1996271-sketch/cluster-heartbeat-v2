"""Model checkpoint persistence.

A checkpoint directory contains:

* ``fingerprint_net.pt`` — model weights + full reconstruction payload, or
* ``pca_heartbeat.joblib`` — the PCA baseline backend,
* ``normalizer.joblib``  — fitted :class:`FeatureNormalizer`,
* ``metrics.json``       — validation metrics at save time,
* ``history.json``       — per-epoch training history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .. import __version__
from ..config import Config, asdict
from ..data.normalization import FeatureNormalizer
from ..models.fingerprint import FingerprintNet
from ..models.pca import PCAHeartbeat
from ..utils.helpers import ensure_dir, write_json

NN_FILENAME = "fingerprint_net.pt"
PCA_FILENAME = "pca_heartbeat.joblib"
NORMALIZER_FILENAME = "normalizer.joblib"


@dataclass
class CheckpointBundle:
    model: Any                       # FingerprintNet | PCAHeartbeat
    normalizer: FeatureNormalizer
    payload: dict                    # stats, metrics, feature/class names, config
    device: torch.device

    @property
    def backend(self) -> str:
        return self.payload.get("model_type", "autoencoder")

    @property
    def stats(self) -> dict:
        return self.payload["stats"]


def save_checkpoint(
    model: Any,
    normalizer: FeatureNormalizer,
    cfg: Config,
    feature_names: list[str],
    class_names: list[str],
    stats: dict,
    metrics: dict,
    ckpt_dir: str | Path,
) -> Path:
    """Persist everything inference needs into ``ckpt_dir``."""
    out = ensure_dir(ckpt_dir)
    normalizer.save(out / NORMALIZER_FILENAME)

    from ..data.windows import channel_names

    payload: dict[str, Any] = {
        "version": __version__,
        "model_type": cfg.model.type,
        "feature_names": feature_names,
        "channel_names": channel_names(
            feature_names, cfg.model.use_deltas and cfg.model.type != "pca"),
        "use_deltas": bool(cfg.model.use_deltas and cfg.model.type != "pca"),
        "class_names": class_names,
        "window_size": cfg.window.size,
        "stats": stats,
        "metrics": metrics,
        "config": asdict(cfg),
    }

    if cfg.model.type == "pca":
        model.save(out / PCA_FILENAME)
    else:
        payload["model_kwargs"] = {
            "n_features": len(payload["channel_names"]),
            "window_size": cfg.window.size,
            "latent_dim": cfg.model.latent_dim,
            "hidden_dims": list(cfg.model.hidden_dims),
            "num_classes": len(class_names),
            "dropout": cfg.model.dropout,
        }
        payload["model_state"] = model.state_dict()
        torch.save(payload, out / NN_FILENAME)
        # A weight-free descriptor for quick inspection without torch.load.
        write_json({k: v for k, v in payload.items() if k != "model_state"},
                   out / "fingerprint_net.meta.json")

    write_json(metrics, out / "metrics.json")
    return out


def load_checkpoint(ckpt_dir: str | Path, device: torch.device) -> CheckpointBundle:
    """Load a checkpoint directory saved by :func:`save_checkpoint`."""
    ckpt_dir = Path(ckpt_dir)
    normalizer = FeatureNormalizer.load(ckpt_dir / NORMALIZER_FILENAME)

    if (ckpt_dir / PCA_FILENAME).exists():
        model = PCAHeartbeat.load(ckpt_dir / PCA_FILENAME)
        meta_path = ckpt_dir / "metrics.json"
        payload = {
            "model_type": "pca",
            "feature_names": normalizer.feature_names,
            "metrics": json.loads(meta_path.read_text()) if meta_path.exists() else {},
        }
        stats_path = ckpt_dir / "pca_stats.json"
        payload["stats"] = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        return CheckpointBundle(model, normalizer, payload, device)

    payload = torch.load(ckpt_dir / NN_FILENAME, map_location=device, weights_only=False)
    model = FingerprintNet(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return CheckpointBundle(model, normalizer, payload, device)
