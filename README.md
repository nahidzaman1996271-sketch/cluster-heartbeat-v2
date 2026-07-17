# Cluster Heartbeat

> One heartbeat driving up reliability, efficiency, and cost savings.

**Cluster Heartbeat** is an AI system that continuously monitors GPU clusters
using Prometheus + NVIDIA DCGM telemetry and compresses every node's behavior
into a single **workload fingerprint** — a compact, explainable embedding that
powers three intelligent services from the same operational picture:

| Pillar | Service | What it does |
|---|---|---|
| 1 | **Predictive Failure Detection** | Flags multi-metric drift *before* thresholds are crossed, with estimated time-to-failure and per-feature evidence. |
| 2 | **Smart GPU Scheduling** | Ranks nodes by *predicted* free capacity, health and failure risk — behavior-aware placement input for a Kubernetes Scheduler Framework plugin. |
| 3 | **GPU Cost Optimization** | Finds idle nodes and "ghost jobs" (memory allocated, zero compute) and estimates reclaimable spend. |

```
Prometheus metrics ──┐
                     ├──► Heartbeat engine ──► unified workload fingerprint ──┬──► Predictive ops
DCGM GPU telemetry ──┘   (PyTorch multi-head                                  ├──► Smart scheduling
                          autoencoder + PCA baseline)                          └──► Cost optimization
```

---

## Why one fingerprint?

Healthy workloads have a characteristic multi-metric *shape*: utilization,
memory, temperature, power and error counters move together in a predictable
way. Three separate tools (monitoring, scheduler, cost analyzer) each see a
slice of that shape. Cluster Heartbeat learns the whole shape once — the
fingerprint — and every service reads from it, the way a hospital monitor
gives a physician one integrated picture of patient health.

## The 13 input features

Defined in [`configs/features.yaml`](configs/features.yaml) (canonical order,
source exporter, Prometheus expression, validity range):

GPU utilization · GPU memory utilization · GPU temperature · Power consumption ·
ECC errors · XID errors · CPU usage · RAM usage · Network throughput · Disk IO ·
Job runtime · Queue length · Active processes

## Architecture

```
src/cluster_heartbeat/
├── config.py               # typed YAML config (dataclasses) + CLI overrides
├── logging_utils.py        # console + file logging
├── data/
│   ├── ingestion.py        # ① synthetic DCGM generator | Prometheus | Alibaba trace
│   ├── preprocessing.py    # ② regular grid, gap filling, physical clipping, de-spiking
│   ├── windows.py          # ③ sliding windows + trend (Δ) channels + targets
│   └── normalization.py    # ④ per-feature z-score, persisted with the checkpoint
├── models/
│   ├── fingerprint.py      # ⑤-⑧ FingerprintNet: shared encoder → 4 heads
│   └── pca.py              #    PCA heartbeat (Phase-1 baseline)
├── training/
│   ├── train.py            #    multi-task training, denoising masking, early stop
│   ├── validate.py         #    AUROC/F1, lead-time, demand & TTF regression
│   ├── metrics.py          #    metric implementations
│   └── checkpoint.py       #    versioned checkpoint dirs (weights + scaler + stats)
├── inference/
│   ├── pipeline.py         # ⑥-⑪ window → fingerprint → cluster report
│   ├── scores.py           #    health/risk scores (transparent, decomposed)
│   └── explain.py          #    per-feature attribution, observed-vs-expected
├── services/
│   ├── failure_detection.py  # Pillar 1: alerts with actions
│   ├── scheduling.py         # Pillar 2: placement scoring
│   └── cost_optimization.py  # Pillar 3: idle/ghost detection, savings
└── api/
    ├── main.py             # FastAPI app + Prometheus /metrics
    └── schemas.py          # dashboard-ready response models
```

### FingerprintNet

A window of telemetry `(13 features × 30 steps)` plus first-difference
**trend channels** `(26 × 30)` is compressed by a shared MLP encoder into the
heartbeat embedding `z` (48-d). Four heads consume `z`:

* **Decoder** — reconstructs the window; per-channel reconstruction error
  (z-scored against the healthy fleet, top-k aggregated) is the anomaly
  signal. Training uses **channel masking** (denoising), which forces
  cross-feature reconstruction — so a correlation break like *"temperature
  rising while power is flat"* fires early instead of at the threshold.
* **Classifier** — workload behavior: `idle | compute_bound | memory_bound |
  io_bound | network_bound | mixed`.
* **Demand head** — next-step GPU %, memory %, watts (placement & capacity).
* **TTF head** — `log1p(hours to failure)` within a 12 h honesty horizon.

The **unified fingerprint** = `[embedding | anomaly | class probs | demand |
ttf]` plus health/failure scores and human-readable evidence — one JSON
object per node per window that powers all three services.

## Quickstart

```bash
pip install -e .                      # or: pip install -r requirements.txt

# 1. generate the Phase-1 synthetic DCGM dataset (16 nodes × 12 h, injected incidents)
python scripts/generate_synthetic_data.py

# 2. train the heartbeat model  (autoencoder default; --set model.type=pca for baseline)
python scripts/train.py

# 3. fleet inference → dashboard-ready JSON
python scripts/run_inference.py       # writes reports/latest_report.json

# 4. serve the API
python scripts/serve.py --port 8000   # docs at http://localhost:8000/docs
```

Any config leaf can be overridden: `python scripts/train.py --set train.epochs=50 --set model.latent_dim=64`

Against a real cluster, set `data.source: prometheus` and point
`data.prometheus.url` at your server — the ingester pulls every metric in
`configs/features.yaml`. For the Alibaba Cluster Trace, see
[`data/README.md`](data/README.md).

## REST API (dashboard-ready JSON)

| Endpoint | Payload |
|---|---|
| `GET /health` | service/model status |
| `GET /api/v1/dashboard` | everything below in one round trip |
| `GET /api/v1/cluster/summary` | health score, nodes at risk, alerts, reclaimable $ |
| `GET /api/v1/cluster/timeseries` | `{ts, value}` series: cluster health, avg GPU util, max failure risk |
| `GET /api/v1/nodes` / `GET /api/v1/nodes/{id}` | per-node scores, class, demand prediction, embedding, reasons |
| `GET /api/v1/nodes/{id}/fingerprint?tail=N` | recent fingerprints for a node |
| `GET /api/v1/alerts?severity=HIGH` | operator alerts with evidence & recommended action |
| `GET /api/v1/recommendations/scheduling?gpu_request=50&mem_request=30` | ranked nodes, score breakdown, avoid list |
| `GET /api/v1/recommendations/cost` | idle/ghost nodes, $/day reclaimable |
| `POST /api/v1/inference` | ad-hoc window → fingerprint (collector integration) |
| `GET /api/v1/nodes/{id}/shap_explain` | second-opinion SHAP attribution (see below) |
| `GET /metrics` | Prometheus counters |

### Example: node fingerprint (excerpt)

```json
{
  "node_id": "node_011",
  "embedding": [0.112, -0.874, "..."],
  "anomaly_score": 1.0,
  "anomaly_features": [
    {"feature": "ecc_errors", "z": 41.2},
    {"feature": "ecc_errors (trend)", "z": 22.8},
    {"feature": "xid_errors", "z": 13.6}
  ],
  "classification": {"label": "idle", "confidence": 0.52},
  "demand": {"gpu_utilization": 5.1, "memory_utilization": 22.5, "power_consumption": 66.9},
  "ttf_hours": 0.0,
  "failure_risk": 1.0,
  "gpu_health": 25.0,
  "reasons": ["Anomalous telemetry shape driven by: ecc_errors, ecc_errors (trend), xid_errors."]
}
```

## Outputs (per node per window + fleet rollups)

* **Cluster Health Score** — mean node health, −2 per critical alert
* **GPU Health Score** — 100 minus decomposed penalties (temperature, anomaly, ECC, XID, memory pressure)
* **Failure Risk Score** — weighted components: behavior anomaly, ECC trend, XID, TTF estimate
* **Estimated Time to Failure** — TTF head (hours, ≤ 12 h horizon)
* **Scheduling Recommendation** — ranked nodes with score breakdown + avoid list + complementary-packing hint
* **Cost Saving Recommendation** — idle/ghost nodes, $/day reclaimable, recommended action
* **Resource Utilization Prediction** — next-step GPU/mem/power per node

## Evaluation (synthetic Phase-1, held-out nodes)

| Metric | Value |
|---|---|
| Anomaly detection AUROC | 0.78 |
| Anomaly F1 (best threshold) | 0.67 (precision 0.96) |
| **Mean predictive lead time before node failure** | **≈ 36 min** (ECC burst: 1.1 h) |
| Workload classification accuracy | 0.97 |
| Demand prediction MAE (GPU util) | 6.6 pct pts |
| TTF regression MAE | 5.0 h |

Numbers come from `checkpoints/best/metrics.json` after `scripts/train.py`;
they recompute deterministically (`seed: 42`).

## Dashboards

* `GET /api/v1/dashboard` is shaped for Grafana's JSON API / Infinity
  datasource or a React frontend (stat scalars + `{ts, value}` series +
  alert table + recommendation objects).
* [`dashboards/grafana_dashboard.json`](dashboards/grafana_dashboard.json) is
  an importable starter dashboard (cluster health stat, health/util/risk
  time series, alerts table, scheduling & cost panels).
* [`reports/latest_report.json`](reports/latest_report.json) is a real sample
  produced by `scripts/run_inference.py`.

## Docker

```bash
docker build -t cluster-heartbeat .
docker run -p 8000:8000 -v $PWD/checkpoints:/app/checkpoints cluster-heartbeat
# or the full demo stack (api + prometheus + grafana):
docker compose up
```

## Explainability: two complementary layers

Every fingerprint ships with a **fast, exact** explanation by default —
`anomaly_features`, `observed_vs_expected` and `reasons` come straight from
the model's own per-feature reconstruction error, so there's no
approximation and no extra cost.

`GET /api/v1/nodes/{id}/shap_explain` adds a **second opinion**: a
model-agnostic SHAP (`KernelExplainer`) attribution over the trained model's
anomaly score, treating the model as a black box and perturbing feature
means against a background of other nodes' windows. It's slower (a few
seconds to ~15s depending on background size) and isn't cached in the node
history — it's meant for on-demand use, e.g. answering "why do you trust
this score?" in a live demo, independent of whether the audience trusts the
reconstruction-error mechanism itself. This was merged in from an earlier
prototype that used SHAP over a classical PCA + IsolationForest detector;
here it explains the actual trained `FingerprintNet` instead. Requires
`pip install shap` (optional dependency, not required for the rest of the
service).

## Tests

```bash
pytest          # shapes, scoring bounds, model forward, API end-to-end, SHAP explain
```

## Design notes & honest limitations

* **Phase 1 runs on synthetic DCGM telemetry** (physics-plausible: power ∝
  utilization, first-order thermal lag, gradually evolving incidents) per the
  project plan; the Prometheus ingester and Alibaba adapter implement the same
  canonical schema for real data.
* TTF is only predicted within a 12 h horizon — beyond that, one 30-minute
  window honestly cannot say.
* The PCA backend (`--set model.type=pca`) reproduces the Phase-1 baseline:
  embedding + anomaly only; classification/demand fall back to documented
  heuristics.
* Lead-time evaluation only counts detections within 6 h before a failure —
  earlier crossings are stale false positives, not predictions.

Team Slow Walker · AI for Cluster Intelligence Hackathon · Team Leader: Nahid Ibn Zaman,  Mentor: Md Adib Hossain
