"""Pillar 3 — GPU cost optimization.

Finds the spend nobody is watching:

* **idle nodes** — near-zero GPU utilization with no active processes;
* **ghost jobs** — GPU memory still allocated but zero compute (allocation
  leak: the node looks busy to the scheduler but does no work);
* **rightsizing** — nodes whose predicted demand is far below their load
  profile, candidates for MIG slicing or consolidation.

Savings combine direct power cost and the opportunity cost of the reserved
GPU capacity.
"""
from __future__ import annotations


def idle_report(
    histories: dict[str, list[dict]],
    window_stride_seconds: int,
    idle_cfg: dict,
    cost_cfg: dict,
) -> dict:
    """Analyze per-node fingerprint histories for reclaimable spend.

    Args:
        histories: node_id → chronologically ordered fingerprint dicts.
        window_stride_seconds: seconds between consecutive windows.
        idle_cfg: thresholds (gpu_util_max, mem_util_max, min_consecutive_windows).
        cost_cfg: electricity_usd_per_kwh, gpu_hourly_cost_usd.
    """
    gpu_max = idle_cfg["gpu_util_max"]
    mem_max = idle_cfg["mem_util_max"]
    min_streak = int(idle_cfg["min_consecutive_windows"])
    price_kwh = cost_cfg["electricity_usd_per_kwh"]
    gpu_hourly = cost_cfg["gpu_hourly_cost_usd"]
    hours_per_window = window_stride_seconds / 3600.0

    nodes: list[dict] = []
    total_power_usd = total_opportunity_usd = 0.0

    for node, fps in sorted(histories.items()):
        if not fps:
            continue
        # Walk backwards from the most recent window through the idle streak.
        streak, ghost = 0, False
        powers = []
        for fp in reversed(fps):
            m = fp["raw_window_means"]
            if m["gpu_utilization"] <= gpu_max and m["active_processes"] <= 1:
                streak += 1
                powers.append(m["power_consumption"])
                ghost = ghost or m["memory_utilization"] > mem_max
            else:
                break
        if streak < min_streak:
            continue

        hours = streak * hours_per_window
        mean_power_w = sum(powers) / len(powers) if powers else 0.0
        power_usd = mean_power_w / 1000.0 * hours * price_kwh
        opportunity_usd = gpu_hourly * hours
        total_power_usd += power_usd
        total_opportunity_usd += opportunity_usd

        kind = "ghost_job" if ghost else "idle_node"
        action = (
            "Reclaim the leaked allocation (kill ghost process / fix the job's "
            "cleanup path), then return the node to the schedulable pool."
            if ghost else
            "Power off or hibernate the node, or consolidate remaining work and "
            "add it to a spot/preemptible pool."
        )
        nodes.append({
            "node_id": node,
            "kind": kind,
            "idle_windows": streak,
            "idle_hours": round(hours, 2),
            "mean_power_w": round(mean_power_w, 1),
            "mean_gpu_util_pct": round(
                sum(fp["raw_window_means"]["gpu_utilization"] for fp in fps[-streak:]) / streak, 2
            ),
            "estimated_savings_usd_per_day": round(
                (power_usd + opportunity_usd) / max(hours, 1e-6) * 24.0, 2
            ),
            "recommended_action": action,
        })

    nodes.sort(key=lambda n: n["estimated_savings_usd_per_day"], reverse=True)
    daily = (total_power_usd + total_opportunity_usd)
    return {
        "idle_or_wasteful_nodes": nodes,
        "n_flagged": len(nodes),
        "estimated_reclaimable_usd": round(daily, 2),
        "assumptions": {
            "electricity_usd_per_kwh": price_kwh,
            "gpu_hourly_cost_usd": gpu_hourly,
            "note": "power cost + opportunity cost of reserved GPU capacity",
        },
    }
