"""Visible length of the entry zone and bursty arrivals (0.7.10).

* ``simulation.visible_length``: at level ``belts`` the camera sees only the last
  ``visible_length`` cm of the entry zone. A MOVING part upstream of that stretch is
  not observable: no token, no pre-positioning candidate, no trace of its time there.
* ``arrivals.stochastic.distribution: burst``: on/off Poisson source with the same mean
  rate as the Poisson process of ``exponential``.
"""

from __future__ import annotations

import statistics

import pytest

from fastsim import POLICIES, FastSim
from optimization.schedule_check import check_schedule
from policies.base_policy import BasePolicy
from policies import get_policy
from simulation.arrivals import generate_arrivals
from simulation.config import load_config
from simulation.engine import Simulation
from simulation.part import PartStatus

HEAVY = "config/examples/heavy_load.yaml"
EPS = 1e-7
MOVING, READY, PROCESSING = 1, 2, 3


def cfg(path=None, seed=4, mode="none", obs="auto", **extra):
    sets = ["arrivals.mode=stochastic", f"simulation.seed={seed}",
            f"laser.prepositioning={mode}", f"simulation.observation={obs}"]
    return load_config(path, set_items=sets + [f"{k}={v}" for k, v in extra.items()])


def burst_cfg(seed=1, on_fraction=0.3, on_mean=20.0, **extra):
    return cfg(seed=seed, **{"arrivals.stochastic.distribution": "burst",
                             "arrivals.stochastic.burst.on_fraction": on_fraction,
                             "arrivals.stochastic.burst.on_mean": on_mean, **extra})


# ------------------------------------------------------------------ config
def test_visible_length_resolution():
    c = load_config()
    assert c.simulation.visible_length == c.geometry.entry_length      # null = whole zone
    assert cfg(**{"simulation.visible_length": 3}).simulation.visible_length == 3.0
    assert cfg(**{"simulation.visible_length": 0}).simulation.visible_length == 0.0
    with pytest.raises(ValueError):
        cfg(**{"simulation.visible_length": 10.5})
    with pytest.raises(ValueError):
        cfg(**{"simulation.visible_length": -1})


def test_burst_config_validation():
    c = cfg()
    assert c.arrivals.distribution == "exponential"
    assert c.arrivals.burst_on_fraction == (1.0, 1.0, 1.0)
    b = burst_cfg()
    assert b.arrivals.distribution == "burst"
    assert b.arrivals.burst_on_fraction == (0.3, 0.3, 0.3) and b.arrivals.burst_on_mean == (20.0,) * 3
    with pytest.raises(ValueError):
        burst_cfg(on_fraction=0.0)
    with pytest.raises(ValueError):
        burst_cfg(on_fraction=1.2)
    with pytest.raises(ValueError):
        burst_cfg(on_mean=0.0)
    with pytest.raises(ValueError):
        cfg(**{"arrivals.stochastic.distribution": "gamma"})


# ---------------------------------------------------------------- arrivals
def _dispersion(times, horizon, window=10.0):
    n = int(horizon / window)
    counts = [0] * n
    for t in times:
        counts[min(int(t // window), n - 1)] += 1
    return statistics.variance(counts) / statistics.mean(counts)


def test_burst_arrivals_keep_the_mean_rate_and_are_bursty():
    horizon = 20000.0
    exp = generate_arrivals(cfg(seed=1, **{"simulation.arrival_horizon": horizon}))
    bur = generate_arrivals(burst_cfg(seed=1, **{"simulation.arrival_horizon": horizon}))
    means = cfg().arrivals.mean_interarrival
    for k in range(3):
        te = [r.time for r in exp if r.conveyor_index == k]
        tb = [r.time for r in bur if r.conveyor_index == k]
        assert len(tb) / horizon == pytest.approx(1.0 / means[k], rel=0.06)
        assert len(te) / horizon == pytest.approx(1.0 / means[k], rel=0.06)
        assert 0.85 < _dispersion(te, horizon) < 1.15          # Poisson: variance = mean
        assert _dispersion(tb, horizon) > 1.8                  # bursts: over-dispersed
        assert tb == sorted(tb) and all(0.0 < t <= horizon for t in tb)
        assert min(b - a for a, b in zip(tb, tb[1:])) > 0.0


def test_burst_arrivals_are_reproducible_and_on_fraction_one_is_poisson():
    assert generate_arrivals(burst_cfg(seed=7)) == generate_arrivals(burst_cfg(seed=7))
    assert generate_arrivals(burst_cfg(seed=7)) != generate_arrivals(burst_cfg(seed=8))
    assert generate_arrivals(burst_cfg(seed=7)) != generate_arrivals(cfg(seed=7))
    # on_fraction 1 reproduces the exponential stream exactly (same random draws)
    assert generate_arrivals(burst_cfg(seed=7, on_fraction=1.0)) == generate_arrivals(cfg(seed=7))


def test_fastsim_runs_bursty_instances():
    sim = FastSim(burst_cfg(seed=3, mode="when_idle", **{"simulation.arrival_horizon": 200}))
    sim.run(POLICIES["nearest_job"])
    assert sim.n_completed == len(sim.pk) > 0
    assert check_schedule(sim.result(), time_tol=1e-6, dist_tol=1e-6) == []


# ------------------------------------------------------------ fastsim view
def _decisions(c, n=40):
    sim = FastSim(c)
    rule = POLICIES["nearest_job"]
    seen = 0
    while sim.run_until_decision() and seen < n:
        cands = sim.candidates()
        yield sim, cands
        seen += 1
        sim.apply(rule(sim, cands))


@pytest.mark.parametrize("mode", ["none", "when_idle"])
def test_fastsim_visibility_follows_the_visible_stretch(mode):
    lv = 3.0
    c = cfg(HEAVY, seed=5, mode=mode, **{"simulation.visible_length": lv})
    hidden_seen, cand_moving = 0, 0
    for sim, cands in _decisions(c):
        avis = sim.a - lv
        for j in range(len(sim.pk)):
            if sim.pst[j] != MOVING:
                continue
            if sim.pvis[j] is None:
                assert sim.x(j) < avis + EPS and not sim.observable(j)
                assert j not in cands
                hidden_seen += 1
            else:
                assert sim.x(j) >= avis - EPS and sim.observable(j)
                assert sim.pvis[j] >= sim.pentry[j] - EPS
        for j in cands:
            if sim.pst[j] == MOVING:
                cand_moving += 1
                assert sim.x(j) >= avis - EPS
    assert hidden_seen > 0
    assert (cand_moving > 0) == (mode == "when_idle")


def test_visible_length_zero_sees_what_zone_sees():
    """``belts`` with ``visible_length: 0`` observes READY/PROCESSING parts only, like
    ``zone``; the full entry zone (default) observes every MOVING part."""
    seen_moving = 0
    for sim, cands in _decisions(cfg(HEAVY, seed=6, **{"simulation.visible_length": 0})):
        for j in range(len(sim.pk)):
            if sim.pst[j] == MOVING:
                assert not sim.observable(j)
                seen_moving += 1
            elif sim.pst[j] in (READY, PROCESSING):
                assert sim.observable(j)
    assert seen_moving > 0
    for sim, cands in _decisions(cfg(HEAVY, seed=6)):
        for j in range(len(sim.pk)):
            if sim.pst[j] == MOVING:
                assert sim.observable(j) and sim.pvis[j] == sim.pentry[j]


def test_rl2_tokens_and_teacher_ignore_parts_upstream_of_the_visible_stretch():
    """Changing a part that is on the belt but upstream of the visible stretch changes
    neither the tokens nor the teacher's costs; with the whole entry zone visible the
    same change is seen."""
    pytest.importorskip("torch")
    import numpy as np
    from rl2.features import encode, observed_since
    from rl2.search import SearchTeacher

    def changes(c, lv):
        restricted = c.simulation.visible_length == lv
        out_tok, out_cost = [], []
        for sim, cands in _decisions(c):
            hidden = [j for j in range(len(sim.pk)) if sim.pst[j] == MOVING
                      and sim.x(j) < sim.a - lv - 1e-6]
            if not hidden or len(cands) < 2:
                continue
            obs = encode(sim, cands)
            if restricted:
                assert set(obs.part_ids).isdisjoint(hidden)
            else:
                assert set(hidden) <= set(obs.part_ids)
            for j in obs.part_ids:
                assert observed_since(sim, j) >= sim.pentry[j] - EPS
            alt = sim.clone()
            for j in hidden:
                alt.pp[j] += 5.0
            alt_obs = encode(alt, alt.candidates())
            out_tok.append(not (alt_obs.part_ids == obs.part_ids
                                and np.allclose(alt_obs.part, obs.part)
                                and np.allclose(alt_obs.conv, obs.conv)))
            t = SearchTeacher(None, k=6)
            out_cost.append(t.costs(alt, cands) != t.costs(sim, cands))
        assert out_tok
        return sum(out_tok), sum(out_cost)
    lv = 3.0
    assert changes(cfg(HEAVY, seed=8, **{"simulation.visible_length": lv}), lv) == (0, 0)
    tok, cost = changes(cfg(HEAVY, seed=8), lv)
    assert tok > 0 and cost > 0


# --------------------------------------------------------------- engine
class _Probe(BasePolicy):
    name = "probe"

    def __init__(self, lv):
        super().__init__()
        self.lv, self.hidden, self.cand_moving = lv, 0, 0
        self.base = get_policy("nearest_job")

    def select(self, candidates, ctx):
        sim = ctx.simulation
        avis = sim.geometry.active_start - self.lv
        for p in sim.parts:
            if p.status == PartStatus.MOVING and p.x < avis - 1e-6:
                assert not sim.observable(p)
                assert p not in candidates
                self.hidden += 1
        for p in candidates:
            if p.status == PartStatus.MOVING:
                assert p.x >= avis - EPS and sim.observable(p)
                self.cand_moving += 1
        return self.base.select(candidates, ctx)


def test_engine_candidates_respect_the_visible_stretch():
    lv = 3.0
    c = cfg(HEAVY, seed=5, mode="when_idle", **{"simulation.visible_length": lv,
                                                 "simulation.arrival_horizon": 120})
    sim = Simulation(c, record_trace=False)
    probe = _Probe(lv)
    sim.policy = probe
    sim.run()
    assert probe.hidden > 0 and probe.cand_moving > 0


@pytest.mark.parametrize("path,rule,seed,lv", [
    (None, "nearest_job", 5004, 3.0),
    (HEAVY, "nearest_head", 5005, 2.0),
])
def test_fastsim_agrees_with_the_engine_under_a_visible_length(path, rule, seed, lv):
    c = cfg(path, seed=seed, mode="when_idle", **{"simulation.visible_length": lv,
                                                  "simulation.arrival_horizon": 120,
                                                  "simulation.dt": 0.001,
                                                  "policy.name": rule})
    res = Simulation(c, record_trace=False).run()
    sim = FastSim(c)
    sim.run(POLICIES[rule])
    order = [p.number - 1 for p in sorted(
        (p for p in res.parts if p.start_processing_time is not None),
        key=lambda p: p.start_processing_time)]
    assert order == sim.order
    for p in res.parts:
        assert sim.pS[p.number - 1] == pytest.approx(p.start_processing_time, abs=0.05)
