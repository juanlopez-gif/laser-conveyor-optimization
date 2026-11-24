"""Reference dispatching rules on the fast simulator.

Same rules and tie-breaks as :mod:`policies.nearest_job` and
:mod:`policies.nearest_head` of the fixed-step engine.

Tie-break (0.7.26): smallest interception time (rounded to 1e-9 s), then the instant
from which the part became **observable** (:meth:`FastSim.observed_since`), then the
part number. Up to 0.7.25 the second key was the real arrival time ``sim.pr[j]``, which
under the ``belts`` and ``zone`` information models the controller does not observe
(a part that waited in the hidden upstream queue shows no trace of it). The legacy key
is kept as :data:`KEY_ARRIVAL` for the audit of ``experiments/tiebreak_audit.py``.
"""

from __future__ import annotations


def KEY_OBSERVED(sim, j: int):
    return (round(sim.travel_time(j), 9), sim.observed_since(j), j)


def KEY_ARRIVAL(sim, j: int):
    """Legacy tie-break of versions <= 0.7.25 (unobservable arrival time)."""
    return (round(sim.travel_time(j), 9), sim.pr[j], j)


def nearest_job(sim, cands: list[int], key=KEY_OBSERVED) -> int:
    """Smallest interception time; ties: earliest observation, then part number."""
    return min(cands, key=lambda j: key(sim, j))


def nearest_head(sim, cands: list[int], key=KEY_OBSERVED) -> int:
    """Nearest conveyor head: only the most downstream candidate of each conveyor."""
    heads: dict[int, int] = {}
    for j in cands:
        k = sim.pk[j]
        h = heads.get(k)
        if h is None or sim.x(j) > sim.x(h) or (sim.x(j) == sim.x(h) and j < h):
            heads[k] = j
    return min(heads.values(), key=lambda j: key(sim, j))


POLICIES = {"nearest_job": nearest_job, "nearest_head": nearest_head}
