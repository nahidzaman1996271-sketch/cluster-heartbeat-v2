#!/usr/bin/env python
"""Generate synthetic Prometheus/DCGM telemetry (Phase-1 dataset)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_heartbeat.config import load_config, load_feature_registry
from cluster_heartbeat.data.ingestion import SyntheticDCGMGenerator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", default="configs/features.yaml")
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.nodes:
        cfg.data.synthetic.nodes = args.nodes
    if args.steps:
        cfg.data.synthetic.steps = args.steps
    if args.out:
        cfg.data.synthetic.out_dir = args.out

    specs, _ = load_feature_registry(args.features)
    result = SyntheticDCGMGenerator(cfg.data.synthetic, specs).generate()
    result.save(cfg.data.synthetic.out_dir)
    print(f"Wrote {len(result.frames)} node frames to {cfg.data.synthetic.out_dir}")
    print(f"Injected failures: {len(result.failures)} "
          f"({sorted(result.failures) or 'none'})")


if __name__ == "__main__":
    main()
