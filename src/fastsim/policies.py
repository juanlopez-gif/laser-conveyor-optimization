"""Reference dispatching rules on the fast simulator.

Same rules and tie-breaks as :mod:`policies.nearest_job` and
:mod:`policies.nearest_head` of the fixed-step engine.
"""

from __future__ import annotations


def nearest_job(sim, cands: list[int]) -> int:
    """Smallest interception time; ties: earliest arrival, then part number."""
    return min(cands, key=lambda j: (round(sim.travel_time(j), 9), sim.pr[j], j))


def nearest_head(sim, cands: list[int]) -> int:
    """Nearest conveyor head: only the most downstream candidate of each conveyor."""
    heads: dict[int, int] = {}
    for j in cands:
        k = sim.pk[j]
        h = heads.get(k)
        if h is None or sim.x(j) > sim.x(h) or (sim.x(j) == sim.x(h) and j < h):
            heads[k] = j
    return min(heads.values(), key=lambda j: (round(sim.travel_time(j), 9), sim.pr[j], j))


POLICIES = {"nearest_job": nearest_job, "nearest_head": nearest_head}
