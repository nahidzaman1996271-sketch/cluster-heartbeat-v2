# Datasets

Cluster Heartbeat consumes the same canonical per-node schema from any of
three sources (`data.source` in `configs/default.yaml`):

```
timestamp | gpu_utilization | memory_utilization | gpu_temperature |
power_consumption | ecc_errors | xid_errors | cpu_usage | ram_usage |
network_throughput | disk_io | job_runtime | queue_length | active_processes
[| label | incident]        # annotations exist for the synthetic source only
```

## 1. Synthetic Prometheus/DCGM telemetry (default, Phase 1)

```bash
python scripts/generate_synthetic_data.py
```

Physics-plausible simulation of a 16-node A100-class cluster over 12 h:

* power is linear in GPU utilization (60 W idle → 400 W max);
* temperature follows power through a first-order thermal lag;
* workload classes (`compute_bound`, `memory_bound`, `io_bound`,
  `network_bound`, `mixed`, `idle`) with labeled job blocks;
* injected, gradually evolving incidents per node (~50% probability):
  * **thermal_runaway** — cooling failure: temperature climbs while power is
    flat, then the GPU thermal-throttles and dies;
  * **ecc_burst** — exponential ECC error growth, then XID errors, then death;
  * **memory_leak** — RAM/FB growth until OOM-restart (anomaly, no node death);
  * **ghost_job** — zero compute with memory still allocated (pure waste;
    used by the cost pillar, not a failure label).

Output: `data/synthetic/node_XXX.csv` + `failures.json` (ground truth).

## 2. Prometheus (production)

Set `data.source: prometheus` and `data.prometheus.url`. The ingester issues
`query_range` calls for every metric in `configs/features.yaml` — DCGM
exporter (`DCGM_FI_DEV_*`), node_exporter and scheduler metrics — and pivots
them to per-node frames on the canonical schema.

## 3. Alibaba Cluster Trace (cluster-trace-v2018)

The trace is public but requires manual acceptance of the download terms:

1. Get `machine_usage.csv` from
   https://github.com/alibaba/clusterdata (cluster-trace-v2018).
2. Place it at `data/alibaba/machine_usage.csv`.
3. Set `data.source: alibaba`.

The adapter (`data/ingestion.py::AlibabaTraceAdapter`) maps
`cpu_util_percent / mem_util_percent / disk_io_percent` onto the canonical
schema. **The trace contains no GPU telemetry**, so — following the Phase-1
plan ("Alibaba traces → Synthetic DCGM metrics") — DCGM-like signals are
synthesized correlated with CPU load. Failure/lead-time metrics are therefore
only reported for the synthetic source, which ships ground truth.

## 4. Google Borg trace (optional)

The Borg `machine_events`/`machine_usage` tables
(https://github.com/google/cluster-data) can be adapted the same way: export
per-machine CPU/mem time series to CSV and drop them in `data/alibaba/` using
the Alibaba column layout, or add a small adapter next to
`AlibabaTraceAdapter`.
