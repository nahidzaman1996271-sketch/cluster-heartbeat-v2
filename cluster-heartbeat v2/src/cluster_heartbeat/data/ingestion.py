"""Step 1 — Data ingestion.

Three interchangeable sources produce the same canonical per-node DataFrame:

    timestamp | <13 registry features> | label | incident

* :class:`SyntheticDCGMGenerator` — physics-plausible Prometheus/DCGM telemetry
  with labeled workload classes, injected incidents and node failures.
  Used for Phase-1 development and tests.
* :class:`PrometheusIngester` — pulls the real metric set from a Prometheus
  server (dcgm-exporter + node_exporter + scheduler metrics).
* :class:`AlibabaTraceAdapter` — maps the public Alibaba Cluster Trace onto the
  canonical schema. The trace carries CPU/memory only, so GPU telemetry is
  synthesized correlated with CPU load (documented in ``data/README.md``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import AlibabaConfig, FeatureSpec, PrometheusConfig, SyntheticConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Canonical column order for the ground-truth / annotation columns.
ANNOTATION_COLUMNS = ["label", "incident"]

#: Physical constants approximating an A100-class node.
_GPU_IDLE_W = 60.0
_GPU_MAX_W = 400.0
_AMBIENT_C = 30.0

#: Per-class workload profiles: (gpu%, mem%, cpu%, ram%, net MB/s, disk MB/s)
_CLASS_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "idle":           {"gpu": (0, 3),    "mem": (5, 15),  "cpu": (2, 8),   "ram": (20, 35), "net": (1, 10),    "disk": (0, 5)},
    "compute_bound":  {"gpu": (85, 98),  "mem": (40, 60), "cpu": (40, 70), "ram": (50, 70), "net": (50, 150),  "disk": (20, 80)},
    "memory_bound":   {"gpu": (30, 50),  "mem": (85, 97), "cpu": (30, 50), "ram": (60, 80), "net": (40, 100),  "disk": (30, 90)},
    "io_bound":       {"gpu": (15, 30),  "mem": (30, 50), "cpu": (20, 40), "ram": (40, 60), "net": (30, 80),   "disk": (200, 500)},
    "network_bound":  {"gpu": (20, 40),  "mem": (30, 50), "cpu": (25, 45), "ram": (40, 60), "net": (400, 900), "disk": (20, 60)},
    "mixed":          {"gpu": (60, 80),  "mem": (60, 80), "cpu": (45, 65), "ram": (55, 75), "net": (150, 400), "disk": (80, 200)},
}

_FAILURE_XID_CODES = [13, 31, 43, 48, 63, 74, 79]  # graphics, DBE, page retirement...


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
@dataclass
class IngestResult:
    """Canonical ingestion output."""

    frames: dict[str, pd.DataFrame]          # node_id -> telemetry frame
    failures: dict[str, dict]                # node_id -> failure metadata

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for node, df in self.frames.items():
            df.to_csv(out / f"{node}.csv", index=False)
        (out / "failures.json").write_text(json.dumps(self.failures, indent=2))
        logger.info("Saved %d node frames to %s", len(self.frames), out)

    @classmethod
    def load(cls, out_dir: str | Path) -> "IngestResult":
        out = Path(out_dir)
        frames = {
            p.stem: pd.read_csv(p, parse_dates=["timestamp"])
            for p in sorted(out.glob("node_*.csv"))
        }
        failures_path = out / "failures.json"
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
        if not frames:
            raise FileNotFoundError(f"No node_*.csv found in {out}")
        return cls(frames=frames, failures=failures)


class SyntheticDCGMGenerator:
    """Generate correlated, physics-plausible DCGM/node telemetry.

    Realism notes:
    * Power is a linear function of GPU utilization (idle + dynamic power).
    * Temperature follows power through a first-order thermal lag (EMA).
    * Incidents evolve *gradually* — thermal runaway, ECC bursts and memory
      leaks are visible in the telemetry well before the node fails, which is
      exactly what predictive detection should catch.
    * ``ghost_job`` incidents emulate hidden waste: memory stays allocated
      while utilization collapses to near zero.
    """

    def __init__(self, cfg: SyntheticConfig, features: list[FeatureSpec]):
        self.cfg = cfg
        self.feature_names = [f.name for f in features]
        self.rng = np.random.default_rng(cfg.seed)

    # -- public ---------------------------------------------------------
    def generate(self) -> IngestResult:
        frames, failures = {}, {}
        for i in range(self.cfg.nodes):
            node = f"node_{i:03d}"
            rng = np.random.default_rng(self.cfg.seed + 1000 + i)
            df, failure = self._generate_node(node, rng)
            frames[node] = df
            if failure:
                failures[node] = failure
        logger.info(
            "Generated synthetic telemetry: %d nodes x %d steps, %d failures",
            len(frames), self.cfg.steps, len(failures),
        )
        return IngestResult(frames=frames, failures=failures)

    # -- node simulation -------------------------------------------------
    def _generate_node(self, node: str, rng: np.random.Generator):
        n = self.cfg.steps
        ts = pd.date_range("2026-01-05", periods=n, freq=f"{self.cfg.interval_seconds}s")

        # Job schedule: 2-5 blocks with a workload class each.
        labels = np.array(["idle"] * n, dtype=object)
        jobs: list[tuple[int, int, str]] = []
        cursor = int(rng.integers(0, max(1, n // 12)))
        while cursor < n - 20:
            cls = rng.choice(
                ["compute_bound", "memory_bound", "io_bound", "network_bound", "mixed"],
                p=[0.34, 0.18, 0.14, 0.14, 0.20],
            )
            dur = int(rng.integers(n // 12, n // 3))
            end = min(cursor + dur, n)
            labels[cursor:end] = cls
            jobs.append((cursor, end, cls))
            cursor = end + int(rng.integers(5, max(6, n // 10)))

        data: dict[str, np.ndarray] = {}
        profile_cache: dict[str, dict[str, float]] = {}
        for cls in _CLASS_PROFILES:
            profile_cache[cls] = {
                k: rng.uniform(*v) for k, v in _CLASS_PROFILES[cls].items()
            }

        def ar1(target: np.ndarray, sigma: float, rho: float = 0.85) -> np.ndarray:
            """Smooth AR(1) noise around a target trajectory."""
            noise = np.zeros(n)
            eps = rng.normal(0, sigma, n)
            for t in range(1, n):
                noise[t] = rho * noise[t - 1] + eps[t]
            return target + noise

        target = {k: np.zeros(n) for k in ["gpu", "mem", "cpu", "ram", "net", "disk"]}
        runtime = np.zeros(n)
        procs = np.zeros(n)
        for t in range(n):
            cls = labels[t]
            prof = profile_cache[cls]
            for k in target:
                target[k][t] = prof[k]
        for start, end, _ in jobs:
            runtime[start:end] = np.arange(end - start) * self.cfg.interval_seconds / 60.0
            procs[start:end] = rng.integers(1, 4)

        gpu = np.clip(ar1(target["gpu"], 2.5), 0, 100)
        mem = np.clip(ar1(target["mem"], 2.0), 0, 100)
        cpu = np.clip(ar1(target["cpu"], 3.0), 0, 100)
        ram = np.clip(ar1(target["ram"], 2.0), 0, 100)
        net = np.clip(ar1(target["net"], 15.0), 0, None)
        disk = np.clip(ar1(target["disk"], 10.0), 0, None)

        # Power follows utilization; temperature follows power with lag.
        power = np.clip(_GPU_IDLE_W + (_GPU_MAX_W - _GPU_IDLE_W) * gpu / 100.0
                        + rng.normal(0, 5, n), 50, 410)
        temp = np.zeros(n)
        temp[0] = _AMBIENT_C + 8
        temp_noise = rng.normal(0, 0.25, n)  # DCGM temperature is stable —
        for t in range(1, n):                # a tight healthy band makes
            steady = _AMBIENT_C + (power[t] / _GPU_MAX_W) * 50.0
            temp[t] = temp[t - 1] + 0.12 * (steady - temp[t - 1]) + temp_noise[t]
        temp = np.clip(temp, 25, 100)

        ecc = rng.poisson(0.01, n).astype(float)     # near-zero background
        xid = np.zeros(n)
        queue = np.clip(rng.poisson(2.0, n) + (labels != "idle") * 2, 0, None).astype(float)

        data.update(
            gpu_utilization=gpu, memory_utilization=mem, gpu_temperature=temp,
            power_consumption=power, ecc_errors=ecc, xid_errors=xid,
            cpu_usage=cpu, ram_usage=ram, network_throughput=net, disk_io=disk,
            job_runtime=runtime, queue_length=queue, active_processes=procs,
        )

        incident = np.array([""] * n, dtype=object)
        failure = None
        if rng.random() < self.cfg.incident_probability:
            failure = self._inject_incident(
                rng, data, labels, incident, n
            )

        df = pd.DataFrame({"timestamp": ts, **data, "label": labels, "incident": incident})
        if failure:
            df = df.iloc[: failure["failure_step"] + 1].reset_index(drop=True)
        return df, failure

    # -- incidents --------------------------------------------------------
    def _inject_incident(self, rng, data, labels, incident, n):
        kind = str(rng.choice(
            ["thermal_runaway", "ecc_burst", "memory_leak", "ghost_job"],
            p=[0.30, 0.30, 0.20, 0.20],
        ))
        start = int(rng.integers(int(n * 0.35), int(n * 0.75)))
        failure = None

        if kind == "thermal_runaway":
            # Cooling failure: temperature climbs while load (and therefore
            # power) stays flat — the correlation break IS the early signal.
            # Near the end the GPU thermal-throttles (power/util sag) and dies.
            dur = min(int(rng.integers(55, 95)), n - start)
            ramp = np.linspace(0, 1, dur) ** 1.2
            data["gpu_temperature"][start:start + dur] += 36 * ramp
            throttle = np.linspace(0, 1, dur) ** 4          # only near the end
            data["power_consumption"][start:start + dur] -= 60 * throttle
            data["gpu_utilization"][start:start + dur] -= 15 * throttle
            incident[start:start + dur] = kind
            fstep = start + dur - 1
            data["xid_errors"][max(start, fstep - 15):fstep + 1] = rng.choice(
                _FAILURE_XID_CODES, size=min(16, fstep - start + 1))
            failure = {"failure_step": fstep, "type": kind}

        elif kind == "ecc_burst":
            dur = min(int(rng.integers(50, 90)), n - start)
            data["ecc_errors"][start:start + dur] += np.linspace(0.5, 45, dur) ** 1.3
            data["xid_errors"][start + int(dur * 0.7):start + dur] = rng.choice(
                _FAILURE_XID_CODES, size=dur - int(dur * 0.7))
            incident[start:start + dur] = kind
            failure = {"failure_step": start + dur - 1, "type": kind}

        elif kind == "memory_leak":
            dur = min(int(rng.integers(80, 150)), n - start)
            leak = np.minimum(np.arange(dur) * 0.5, 55)
            data["ram_usage"][start:start + dur] = np.clip(
                data["ram_usage"][start:start + dur] + leak, 0, 99)
            data["memory_utilization"][start:start + dur] = np.clip(
                data["memory_utilization"][start:start + dur] + leak * 0.4, 0, 99)
            incident[start:start + dur] = kind
            # job OOM-restart: runtime resets (no node failure)
            data["job_runtime"][start + dur - 1:] = 0

        elif kind == "ghost_job":
            dur = n - start
            data["gpu_utilization"][start:] = rng.uniform(0.5, 3.0, dur)
            data["power_consumption"][start:] = _GPU_IDLE_W + rng.uniform(5, 20, dur)
            data["gpu_temperature"][start:] = _AMBIENT_C + rng.uniform(4, 9, dur)
            data["memory_utilization"][start:] = rng.uniform(35, 55, dur)  # still allocated!
            data["active_processes"][start:] = 1
            labels[start:] = "idle"
            incident[start:] = kind

        return failure


# ---------------------------------------------------------------------------
# Prometheus ingestion
# ---------------------------------------------------------------------------
class PrometheusIngester:
    """Fetch the canonical metric set from a Prometheus HTTP API."""

    def __init__(self, cfg: PrometheusConfig, features: list[FeatureSpec]):
        self.cfg = cfg
        self.features = features
        self.session = requests.Session()
        self.session.verify = cfg.verify_tls

    def _query_range(self, expr: str, start: float, end: float) -> list[dict]:
        resp = self.session.get(
            f"{self.cfg.url}/api/v1/query_range",
            params={"query": expr, "start": start, "end": end,
                    "step": self.cfg.step_seconds},
            timeout=self.cfg.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus error for {expr!r}: {payload}")
        return payload["data"]["result"]

    def fetch(self) -> IngestResult:
        """Pull all registry metrics and pivot to per-node frames."""
        import time

        end = time.time()
        start = end - self.cfg.lookback_hours * 3600
        per_node: dict[str, pd.DataFrame] = {}

        for spec in self.features:
            for series in self._query_range(spec.prometheus_metric, start, end):
                node = (series["metric"].get("instance")
                        or series["metric"].get("Hostname") or "unknown").split(":")[0]
                df = per_node.setdefault(node, pd.DataFrame())
                s = pd.DataFrame(series["values"], columns=["ts", spec.name])
                s["timestamp"] = pd.to_datetime(s["ts"], unit="s", utc=True).dt.tz_localize(None)
                s[spec.name] = pd.to_numeric(s[spec.name], errors="coerce")
                s = s[["timestamp", spec.name]]
                per_node[node] = s if df.empty else df.merge(s, on="timestamp", how="outer")

        for node, df in per_node.items():
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["label"] = "unknown"
            df["incident"] = ""
            per_node[node] = df

        logger.info("Prometheus ingestion: %d nodes", len(per_node))
        return IngestResult(frames=per_node, failures={})


# ---------------------------------------------------------------------------
# Alibaba Cluster Trace adapter
# ---------------------------------------------------------------------------
class AlibabaTraceAdapter:
    """Map Alibaba Cluster Trace ``machine_usage.csv`` to the canonical schema.

    Expected raw columns (cluster-trace-v2018): machine_id, time_stamp,
    cpu_util_percent, mem_util_percent, ... GPU telemetry does not exist in
    the trace, so DCGM-like signals are synthesized correlated with CPU load;
    see ``data/README.md``.
    """

    def __init__(self, cfg: AlibabaConfig, features: list[FeatureSpec]):
        self.cfg = cfg
        self.features = features

    def fetch(self) -> IngestResult:
        path = Path(self.cfg.raw_dir) / self.cfg.machine_usage_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download cluster-trace-v2018 machine_usage.csv "
                f"into {self.cfg.raw_dir}/ — see data/README.md for instructions."
            )
        raw = pd.read_csv(
            path,
            names=["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
                   "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"],
            usecols=["machine_id", "time_stamp", "cpu_util_percent",
                     "mem_util_percent", "disk_io_percent"],
        )
        frames: dict[str, pd.DataFrame] = {}
        rng = np.random.default_rng(7)
        machines = raw["machine_id"].unique()[: self.cfg.max_machines]
        for machine in machines:
            sub = raw[raw["machine_id"] == machine].sort_values("time_stamp")
            node = f"ali_{int(machine):04d}"
            n = len(sub)
            cpu = sub["cpu_util_percent"].to_numpy(float)
            gpu = np.clip(cpu * rng.uniform(0.6, 1.1) + rng.normal(0, 3, n), 0, 100)
            power = np.clip(_GPU_IDLE_W + (_GPU_MAX_W - _GPU_IDLE_W) * gpu / 100, 50, 410)
            temp = np.clip(_AMBIENT_C + power / _GPU_MAX_W * 45 + rng.normal(0, 1, n), 25, 95)
            frames[node] = pd.DataFrame({
                "timestamp": pd.to_datetime(sub["time_stamp"], unit="s"),
                "gpu_utilization": gpu,
                "memory_utilization": np.clip(sub["mem_util_percent"].to_numpy(float) * 0.8, 0, 100),
                "gpu_temperature": temp,
                "power_consumption": power,
                "ecc_errors": rng.poisson(0.01, n).astype(float),
                "xid_errors": np.zeros(n),
                "cpu_usage": cpu,
                "ram_usage": sub["mem_util_percent"].to_numpy(float),
                "network_throughput": np.clip(cpu * 2 + rng.normal(0, 10, n), 0, None),
                "disk_io": np.clip(sub["disk_io_percent"].to_numpy(float) * 4, 0, None),
                "job_runtime": np.minimum(np.arange(n) * 60 / 60.0, 600),
                "queue_length": rng.poisson(2.0, n).astype(float),
                "active_processes": (gpu > 5).astype(float) * 2,
                "label": np.where(gpu > 60, "compute_bound",
                           np.where(gpu > 5, "mixed", "idle")),
                "incident": "",
            })
        logger.info("Alibaba adapter: %d machines mapped", len(frames))
        return IngestResult(frames=frames, failures={})


def ingest(source: str, cfg, features: list[FeatureSpec]) -> IngestResult:
    """Factory: build the ingester selected by ``data.source``."""
    if source == "synthetic":
        gen = SyntheticDCGMGenerator(cfg.data.synthetic, features)
        result = gen.generate()
        result.save(cfg.data.synthetic.out_dir)
        return result
    if source == "prometheus":
        return PrometheusIngester(cfg.data.prometheus, features).fetch()
    if source == "alibaba":
        return AlibabaTraceAdapter(cfg.data.alibaba, features).fetch()
    raise ValueError(f"Unknown data source: {source!r}")
