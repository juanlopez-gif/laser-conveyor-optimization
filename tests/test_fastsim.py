"""Event-driven fast simulator (strong-RL plan, phase 1).

The large-scale agreement study with the fixed-step engine is
``experiments/fastsim_validation.py`` (docs/methods/fastsim.md); these tests keep the
essential properties cheap enough for every run of the suite.
"""

import pytest

from fastsim import POLICIES, FastSim
from optimization.objective_functions import OBJECTIVES
from optimization.schedule_check import check_schedule
from simulation.config import load_config
from simulation.engine import Simulation

HEAVY = "config/examples/heavy_load.yaml"


def cfg(path=None, seed=7, mode="none", rule="nearest_job", **extra):
    sets = ["arrivals.mode=stochastic", f"simulation.seed={seed}", f"policy.name={rule}",
            f"laser.prepositioning={mode}"] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(path, set_items=sets)


@pytest.mark.parametrize("path", [None, HEAVY])
@pytest.mark.parametrize("mode", ["none", "when_idle", "always"])
@pytest.mark.parametrize("rule", ["nearest_job", "nearest_head"])
def test_schedule_satisfies_the_formal_model(path, mode, rule):
    sim = FastSim(cfg(path, seed=11, mode=mode, rule=rule))
    sim.run(POLICIES[rule])
    assert sim.n_completed == len(sim.pk) > 0
    assert check_schedule(sim.result(), time_tol=1e-6, dist_tol=1e-6) == []


@pytest.mark.parametrize("path", [None, HEAVY])
def test_cost_identities(path):
    """WIP integral = sum of flow times; blocked-belt integral = sum (S_j - B_j)."""
    sim = FastSim(cfg(path, seed=3))
    sim.run(POLICIES["nearest_job"])
    flows = sum(sim.pC[j] - sim.pr[j] for j in range(len(sim.pk)))
    assert sim.wip_area == pytest.approx(flows, abs=1e-6)
    tb = sum(sim.pS[j] - sim.pB[j] for j in range(len(sim.pk)) if sim.pB[j] is not None)
    assert sim.blocked_area == pytest.approx(tb, abs=1e-6)
    if path == HEAVY:
        assert sim.n_block_events > 0          # the heavy case does exercise blocking


@pytest.mark.parametrize("path,mode,rule,seed", [
    (None, "none", "nearest_head", 5001),
    (None, "when_idle", "nearest_job", 5002),
    (HEAVY, "always", "nearest_head", 5003),
])
def test_agrees_with_the_engine_in_its_continuous_limit(path, mode, rule, seed):
    """Against the fixed-step engine at dt = 0.001 (short horizon to keep it fast):
    same processing order, start times within a few engine steps, same KPI."""
    c = cfg(path, seed=seed, mode=mode, rule=rule, **{"simulation.arrival_horizon": 120,
                                                      "simulation.dt": 0.001})
    res = Simulation(c, record_trace=False).run()
    sim = FastSim(c)
    s = sim.run(POLICIES[rule])
    order = [p.number - 1 for p in sorted(
        (p for p in res.parts if p.start_processing_time is not None),
        key=lambda p: p.start_processing_time)]
    assert order == sim.order
    for p in res.parts:
        assert sim.pS[p.number - 1] == pytest.approx(p.start_processing_time, abs=0.05)
    k = OBJECTIVES["penalized_flow_time"](res.summary)
    assert s["penalized_flow_time"] == pytest.approx(k, rel=0.005)


def test_decision_api_matches_run():
    a = FastSim(cfg(HEAVY, seed=9, mode="when_idle"))
    a.run(POLICIES["nearest_head"])
    b = FastSim(cfg(HEAVY, seed=9, mode="when_idle"))
    while b.run_until_decision():
        b.apply(POLICIES["nearest_head"](b, b.candidates()))
    assert a.order == b.order and a.summary() == b.summary()


def test_only_candidates_can_be_applied():
    sim = FastSim(cfg(seed=4))
    assert sim.run_until_decision()
    cands = sim.candidates()
    bad = next(j for j in range(len(sim.pk) + 1) if j not in cands)
    with pytest.raises(ValueError):
        sim.apply(bad)


def test_clone_is_independent():
    sim = FastSim(cfg(HEAVY, seed=8, mode="when_idle"))
    for _ in range(15):
        sim.run_until_decision()
        sim.apply(POLICIES["nearest_job"](sim, sim.candidates()))
    sim.run_until_decision()
    snapshot = (sim.t, list(sim.order), list(sim.D), [list(q) for q in sim.belt],
                sim.wip_area, sim.candidates())
    twin = sim.clone()
    twin.run(POLICIES["nearest_head"])                     # play the clone to the end
    assert twin.done and not sim.done
    assert (sim.t, list(sim.order), list(sim.D), [list(q) for q in sim.belt],
            sim.wip_area, sim.candidates()) == snapshot
    # the original continues exactly as a fresh run would
    ref = FastSim(cfg(HEAVY, seed=8, mode="when_idle"))
    for _ in range(15):
        ref.run_until_decision()
        ref.apply(POLICIES["nearest_job"](ref, ref.candidates()))
    sim.run(POLICIES["nearest_job"])
    ref.run(POLICIES["nearest_job"])
    assert sim.order == ref.order and sim.summary() == ref.summary()


def test_pass_mode_is_rejected():
    with pytest.raises(NotImplementedError):
        FastSim(cfg(**{"simulation.unprocessed_at_zone_end": "pass"}))
