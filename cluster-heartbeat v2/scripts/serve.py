#!/usr/bin/env python
"""Serve the Cluster Heartbeat REST API."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default="checkpoints/best")
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import os

    os.environ.update(CH_CKPT_DIR=args.ckpt, CH_DATA_DIR=args.data,
                      CH_CONFIG=args.config)

    import uvicorn

    uvicorn.run("cluster_heartbeat.api.main:app",
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
