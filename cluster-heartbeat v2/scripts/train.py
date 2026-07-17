#!/usr/bin/env python
"""Train the heartbeat model (autoencoder or PCA baseline)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_heartbeat.config import load_config, load_feature_registry
from cluster_heartbeat.training.train import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", default="configs/features.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted config override, e.g. --set train.epochs=30")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    specs, classes = load_feature_registry(args.features)
    metrics = Trainer(cfg, specs, classes).run()
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
