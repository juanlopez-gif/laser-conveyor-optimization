"""The rules break ties on observable quantities only (0.7.26).

Up to 0.7.25 ``nearest_job``, ``nearest_head`` and the candidate ranking of the search
used the real arrival time ``r_j`` as the second tie-break key. Under the ``belts`` and
``zone`` information models that instant is not observed (a part that waited in the
hidden upstream queue shows no trace of it). These tests change the arrival times of the
*candidates* without touching anything the controller observes (positions, interception
times, statuses, observation instants) and check that no decision changes, on the fast
simulator and on the fixed-step engine.
"""

from __future__ import annotations

import random

import pytest

from fastsim import FastSim
from fastsim.policies import KEY_ARRIVAL, nearest_head, nearest_job
from rl2.expert_iteration import val_config
from rl2.search import SearchTeacher

LEVELS = ("belts", "zone")


def _cfg(seed, obs, mode, extra=()):
    return val_config("real", seed, "none",
                      extra=(f"simulation.observation={obs}", f"laser.prepositioning={mode}") + tuple(extra))


def _perturb(sim, cands, rng):
    """Shift the candidates' arrival times (positions depend on the entry coordinate,
    not on ``pr``, so nothing observable changes); returns an undo function."""
    old = {j: sim.pr[j] for j in cands}
    for j in cands:
        sim.pr[j] = old[j] - rng.uniform(5.0, 40.0)
    rng.shuffle(cands)

    def undo():
        for j, v in old.items():
            sim.pr[j] = v
    return undo


@pytest.mark.parametrize("obs", LEVELS)
@pytest.mark.parametrize("mode", ("none", "when_idle"))
def test_fastsim_rules_ignore_arrival_times(obs, mode):
    if obs == "zone" and mode == "when_idle":
        pytest.skip("zone forces none")
    rng = random.Random(0)
    teacher = SearchTeacher(None, k=6)
    sim = FastSim(_cfg(2000, obs, mode))
    decisions = ties = 0
    while sim.run_until_decision():
        cands = sim.candidates()
        if len(cands) > 1:
            a = nearest_job(sim, cands), nearest_head(sim, cands), teacher.preference(sim, cands)
            keys = {round(sim.travel_time(j), 9) for j in cands}
            ties += len(keys) < len(cands)
            undo = _perturb(sim, list(cands), rng)
            b = nearest_job(sim, cands), nearest_head(sim, cands), teacher.preference(sim, cands)
            undo()
            assert a == b
            # the legacy key does depend on the arrival time when there are ties
            decisions += 1
        sim.apply(nearest_job(sim, cands))
    assert decisions > 20
    assert ties > 0, "the episode must contain exact ties for the test to be informative"


def test_legacy_key_is_sensitive_to_arrival_times():
    """Sanity check of the test itself: the 0.7.25 key changes with the arrival time."""
    rng = random.Random(1)
    sim = FastSim(_cfg(2000, "belts", "none"))
    changed = 0
    while sim.run_until_decision():
        cands = sim.candidates()
        if len(cands) > 1:
            a = nearest_job(sim, cands, key=KEY_ARRIVAL)
            undo = _perturb(sim, list(cands), rng)
            b = nearest_job(sim, cands, key=KEY_ARRIVAL)
            undo()
            changed += a != b
        sim.apply(nearest_job(sim, cands))
    assert changed > 0


def test_observed_since_matches_level():
    sim = FastSim(_cfg(2000, "belts", "none"))
    while sim.run_until_decision():
        for j in sim.candidates():
            assert sim.observed_since(j) == sim.pvis[j]
            assert sim.observed_since(j) >= sim.pr[j]
        sim.apply(sim.candidates()[0])


class _Checked:
    """Engine policy wrapper: selects, then selects again with the candidates' arrival
    times shifted, and requires the same choice."""

    def __init__(self, inner):
        self.inner, self.rng, self.checked = inner, random.Random(2), 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def select(self, candidates, ctx):
        a = self.inner.select(candidates, ctx)
        old = {p.id: p.arrival_time for p in candidates}
        for p in candidates:
            p.arrival_time -= self.rng.uniform(5.0, 40.0)
        b = self.inner.select(list(reversed(candidates)), ctx)
        for p in candidates:
            p.arrival_time = old[p.id]
        assert a is b
        self.checked += len(candidates) > 1
        return a


@pytest.mark.parametrize("name", ("nearest_job", "nearest_head"))
def test_engine_rules_ignore_arrival_times(name):
    from simulation.engine import Simulation
    cfg = _cfg(2000, "belts", "when_idle", extra=(f"policy.name={name}", "simulation.dt=0.05",
                                                  "simulation.arrival_horizon=300"))
    sim = Simulation(cfg, record_trace=False)
    sim.policy = _Checked(sim.policy)
    sim.run()
    assert sim.policy.checked > 15
    for c in sim.conveyors:
        for p in c.parts:
            assert p.observed_time is None or p.observed_time >= p.entry_time
