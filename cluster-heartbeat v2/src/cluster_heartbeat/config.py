"""Typed configuration for Cluster Heartbeat.

All configuration lives in YAML under ``configs/`` and is loaded into the
dataclasses below. Any leaf can be overridden from the CLI with dotted paths,
e.g. ``--set train.epochs=30``.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """One canonical telemetry signal (see configs/features.yaml)."""

    name: str
    source: str
    prometheus_metric: str
    unit: str
    min: Optional[float]
    max: Optional[float]
    description: str = ""


def load_feature_registry(path: str | Path) -> tuple[list[FeatureSpec], list[str]]:
    """Load the feature registry YAML.

    Returns:
        (features, class_names) — features in canonical tensor order.
    """
    raw = yaml.safe_load(Path(path).read_text())
    features = [
        FeatureSpec(
            name=f["name"],
            source=f.get("source", "custom"),
            prometheus_metric=str(f.get("prometheus_metric", "")),
            unit=f.get("unit", ""),
            min=f.get("min"),
            max=f.get("max"),
            description=f.get("description", ""),
        )
        for f in raw["features"]
    ]
    return features, list(raw.get("classes", []))


# ---------------------------------------------------------------------------
# Config tree
# ---------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    nodes: int = 12
    steps: int = 720
    interval_seconds: int = 60
    out_dir: str = "data/synthetic"
    incident_probability: float = 0.5
    seed: int = 42


@dataclass
class PrometheusConfig:
    url: str = "http://localhost:9090"
    lookback_hours: int = 12
    step_seconds: int = 60
    timeout_seconds: int = 30
    verify_tls: bool = False


@dataclass
class AlibabaConfig:
    raw_dir: str = "data/alibaba"
    machine_usage_file: str = "machine_usage.csv"
    max_machines: int = 50


@dataclass
class DataConfig:
    source: str = "synthetic"  # synthetic | prometheus | alibaba
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    alibaba: AlibabaConfig = field(default_factory=AlibabaConfig)


@dataclass
class WindowConfig:
    size: int = 30
    stride: int = 5
    forecast_horizon_steps: int = 1
    ttf_horizon_hours: float = 48.0


@dataclass
class ModelConfig:
    type: str = "autoencoder"  # autoencoder | pca
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    pca_components: int = 16
    use_deltas: bool = True    # append first-difference (trend) channels


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.2
    patience: int = 5
    device: str = "auto"  # auto | cpu | cuda
    mask_prob: float = 0.15  # channel-masking (denoising) during training
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "reconstruction": 1.0,
            "classification": 0.5,
            "demand": 0.5,
            "ttf": 0.3,
        }
    )


@dataclass
class ScoringConfig:
    temperature: dict[str, float] = field(
        default_factory=lambda: {"warn_celsius": 80.0, "crit_celsius": 90.0}
    )
    anomaly: dict[str, float] = field(
        default_factory=lambda: {"z_full_scale": 6.0, "alert_threshold": 0.7}
    )
    failure: dict[str, float] = field(
        default_factory=lambda: {"risk_alert_threshold": 0.6}
    )
    idle: dict[str, float] = field(
        default_factory=lambda: {
            "gpu_util_max": 5.0,
            "mem_util_max": 10.0,
            "min_consecutive_windows": 3,
        }
    )
    cost: dict[str, float] = field(
        default_factory=lambda: {
            "electricity_usd_per_kwh": 0.12,
            "gpu_hourly_cost_usd": 2.50,
        }
    )


@dataclass
class PathConfig:
    checkpoints: str = "checkpoints"
    logs: str = "logs"
    reports: str = "reports"


@dataclass
class Config:
    project: str = "cluster-heartbeat"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# ---------------------------------------------------------------------------
# Loading & overrides
# ---------------------------------------------------------------------------
def _apply(dc: Any, mapping: dict[str, Any]) -> None:
    """Recursively apply a dict onto a dataclass instance (in place)."""
    for key, value in mapping.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key: {key!r} in {type(dc).__name__}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(dc, key, copy.deepcopy(value))


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    leaf = parts[-1]
    current = getattr(node, leaf, None)
    if isinstance(current, bool):
        value = str(value).lower() in {"1", "true", "yes", "on"}
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(node, leaf, value)


def load_config(
    path: str | Path = "configs/default.yaml",
    overrides: Optional[list[str]] = None,
) -> Config:
    """Load a YAML config into a :class:`Config`.

    Args:
        path: YAML file path.
        overrides: optional list of ``"a.b.c=value"`` strings.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = Config()
    _apply(cfg, raw)
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def asdict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
