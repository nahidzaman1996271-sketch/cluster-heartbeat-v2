"""Small shared utilities."""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def get_device(pref: str = "auto"):
    """Resolve a torch device from a preference string."""
    import torch

    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    return p


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


@contextmanager
def timer(label: str, logger=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        msg = f"{label} took {time.perf_counter() - start:.2f}s"
        (logger.info if logger else print)(msg)
