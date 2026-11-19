"""Objective functions computed from a run summary (see summarize()).

Every function takes the KPI dictionary of a run and returns a scalar to
MINIMISE (maximisation objectives are negated). This makes any optimiser —
grid search over policy weights, a genetic algorithm, an RL reward, ... —
interchangeable: it only needs ``objective(summary) -> float``.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping


def _val(summary: Mapping, key: str) -> float:
    v = summary.get(key)
    return math.inf if v is None else float(v)


def makespan(s: Mapping) -> float:
    return _val(s, "makespan")


def average_flow_time(s: Mapping) -> float:
    return _val(s, "avg_flow_time")


def average_waiting_time(s: Mapping) -> float:
    return _val(s, "avg_waiting_time")


def total_laser_travel(s: Mapping) -> float:
    return _val(s, "total_laser_travel_distance")


def total_conveyor_stop_time(s: Mapping) -> float:
    return _val(s, "total_conveyor_stop_time")


def negative_throughput(s: Mapping) -> float:
    """Maximise throughput == minimise its negative."""
    return -_val(s, "throughput_per_min")


def maximum_lateness(s: Mapping) -> float:
    return _val(s, "max_lateness")


#: Name -> single objective. Keys are the names used in ``objective.weights``.
OBJECTIVES: dict[str, Callable[[Mapping], float]] = {
    "makespan": makespan,
    "avg_flow_time": average_flow_time,
    "avg_waiting_time": average_waiting_time,
    "total_laser_travel": total_laser_travel,
    "total_conveyor_stop_time": total_conveyor_stop_time,
    "neg_throughput": negative_throughput,
    "max_lateness": maximum_lateness,
}


def composite_objective(summary: Mapping, weights: Mapping[str, float]) -> float:
    """Weighted sum  sum_k w_k * f_k(summary)  (only non-zero weights are evaluated).

    Example (config ``objective.weights``)::

        alpha*makespan + beta*total_laser_travel
        + gamma*total_conveyor_stop_time + delta*avg_waiting_time
    """
    total = 0.0
    for name, w in weights.items():
        if w == 0:
            continue
        if name not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{name}'. Available: {sorted(OBJECTIVES)}")
        total += w * OBJECTIVES[name](summary)
    return total


def pareto_front(points: list[tuple[float, ...]]) -> list[int]:
    """Indices of non-dominated points (all objectives minimised).

    Small O(n^2) helper for future multi-objective comparisons of policies.
    """
    front = []
    for i, p in enumerate(points):
        dominated = any(all(q[k] <= p[k] for k in range(len(p))) and q != p
                        for j, q in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front
