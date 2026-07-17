"""Pillar 2 — Smart GPU scheduling.

Behavior-aware placement: rank nodes by *predicted* free capacity (from the
demand head), health and failure risk instead of static resource requests.
Designed to back a Kubernetes Scheduler Framework plugin: the API response is
a direct scoring input to a Filter+Score cycle.
"""
from __future__ import annotations

#: Complementary classes that bin-pack well together.
_COMPLEMENTS = {
    "compute_bound": "memory_bound",
    "memory_bound": "compute_bound",
    "network_bound": "io_bound",
    "io_bound": "network_bound",
    "mixed": "idle",
    "idle": "compute_bound",
}


def recommend(
    latest: dict[str, dict],
    gpu_request: float = 0.0,
    mem_request: float = 0.0,
    avoid_risk: float = 0.6,
    min_health: float = 50.0,
) -> dict:
    """Rank nodes for placing a new workload.

    Args:
        latest: node_id → latest fingerprint dict.
        gpu_request: required GPU % (0-100) of the incoming workload.
        mem_request: required GPU memory % (0-100).
        avoid_risk: failure-risk above which a node is excluded.
        min_health: GPU health below which a node is excluded.
    """
    ranked, avoided = [], []

    for node, fp in sorted(latest.items()):
        pred = fp["demand"]
        free_gpu = max(0.0, 100.0 - pred["gpu_utilization"])
        free_mem = max(0.0, 100.0 - pred["memory_utilization"])
        risk, health = fp["failure_risk"], fp["gpu_health"]

        fits = free_gpu >= gpu_request and free_mem >= mem_request
        safe = risk < avoid_risk and health >= min_health

        entry = {
            "node_id": node,
            "predicted_free_gpu_pct": round(free_gpu, 1),
            "predicted_free_mem_pct": round(free_mem, 1),
            "gpu_health": health,
            "failure_risk": risk,
            "current_class": fp["classification"]["label"],
        }
        if not (fits and safe):
            reasons = []
            if not fits:
                reasons.append(
                    f"insufficient predicted capacity (need {gpu_request:.0f}% GPU/"
                    f"{mem_request:.0f}% mem)"
                )
            if risk >= avoid_risk:
                reasons.append(f"failure risk {risk:.2f} ≥ {avoid_risk}")
            if health < min_health:
                reasons.append(f"GPU health {health:.0f} < {min_health}")
            avoided.append({**entry, "reasons": reasons})
            continue

        capacity_term = 0.5 * free_gpu / 100.0 + 0.3 * free_mem / 100.0
        score = (0.7 * capacity_term + 0.3 * health / 100.0) * (1.0 - risk)
        ranked.append({
            **entry,
            "score": round(score, 3),
            "score_breakdown": {
                "capacity": round(0.7 * capacity_term, 3),
                "health": round(0.3 * health / 100.0, 3),
                "risk_multiplier": round(1.0 - risk, 3),
            },
            "complementary_placement": (
                f"good host for a {_COMPLEMENTS.get(fp['classification']['label'], 'mixed')} "
                f"co-tenant (current load is {fp['classification']['label']})"
            ),
        })

    ranked.sort(key=lambda e: e["score"], reverse=True)
    return {
        "request": {"gpu_pct": gpu_request, "mem_pct": mem_request},
        "recommended": ranked,
        "avoid": sorted(avoided, key=lambda e: e["failure_risk"], reverse=True),
        "best_node": ranked[0]["node_id"] if ranked else None,
        "note": (
            "Scores combine predicted free capacity (demand head), GPU health and "
            "(1 - failure risk). Feed directly into a Scheduler Framework Score plugin."
        ),
    }
