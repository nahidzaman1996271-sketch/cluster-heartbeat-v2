#!/usr/bin/env python
"""Run fleet inference and write the dashboard-ready JSON report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_heartbeat.config import load_config, load_feature_registry
from cluster_heartbeat.data.ingestion import IngestResult
from cluster_heartbeat.data.preprocessing import clean_all
from cluster_heartbeat.inference.pipeline import HeartbeatPipeline
from cluster_heartbeat.utils.helpers import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", default="configs/features.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best")
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/latest_report.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs, _ = load_feature_registry(args.features)
    result = IngestResult.load(args.data)
    frames = clean_all(result.frames, specs, cfg.data.synthetic.interval_seconds)

    pipeline = HeartbeatPipeline(args.ckpt, device=cfg.train.device)
    report = pipeline.process_cluster(frames)
    write_json(report, args.out)

    c = report["cluster"]
    print(json.dumps(c, indent=2))
    print(f"\nFull dashboard JSON written to {args.out}")
    if report["alerts"]:
        print(f"\nTop alert: {report['alerts'][0]['message']}")
    sched = report["recommendations"]["scheduling"]
    print(f"Best placement node: {sched['best_node']}")
    cost = report["recommendations"]["cost"]
    print(f"Reclaimable spend: ${cost['estimated_reclaimable_usd']} "
          f"across {cost['n_flagged']} node(s)")


if __name__ == "__main__":
    main()
