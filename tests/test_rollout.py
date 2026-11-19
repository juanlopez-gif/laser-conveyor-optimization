"""Tests of the rollout (one-step look-ahead) policy: fairness, isolation, improvement."""

from __future__ import annotations

import copy
import random

import pytest

from optimization.evaluate import evaluate_policy
from optimization.instances import Instance, instance_config, make_instance
from optimization.schedule_check import check_schedule
from optimization.solvers import discover
from policies import available_policies, get_policy
from policies.base_policy import BasePolicy
from policies.rollout import RolloutPolicy, _ForcedFirstPolicy, _NullEventLog
from simulation.config import load_config
from simulation.engine import Simulation
from simulation.events import EventKind


# ----------------------------------------------------------------- helpers
class _ClairvoyantRollout(RolloutPolicy):
    """FORBIDDEN variant (tests only): the clone keeps the real future arrivals.

    Used to show that the instance below is discriminating, i.e. that knowing
    the future WOULD change the decision.
    """

    def _strip_future(self, clone, original, scenario):
        clone.pending_arrivals = copy.deepcopy(original.pending_arrivals)


def _run(inst: Instance, policy: BasePolicy, dt: float = 0.01):
    cfg = instance_config(inst, "nearest_job", None, dt)
    return Simulation(cfg, policy=policy, record_trace=False).run()


def _assignments(result, before: float | None = None) -> list[tuple[float, str]]:
    return [(e.time, e.part_id) for e in result.events.of_kind(EventKind.ASSIGN)
            if before is None or e.time < before]


def _fingerprint(sim: Simulation) -> tuple:
    """Everything observable of a running simulation (for 'unchanged' checks)."""
    parts = tuple(tuple(sorted(p.as_record().items())) + (p.assigned, p.x) for p in sim.parts)
    convs = tuple((c.displacement, tuple(sorted(c.stop_reasons)), tuple(p.id for p in c.parts),
                   tuple(p.id for p in c.buffer)) for c in sim.conveyors)
    las = sim.laser
    laser = (las.x, las.y, las.state, las.target_x, las.target_y, las.distance,
             tuple(las.history), las.current_part.id if las.current_part else None)
    m = sim.metrics
    metrics = (tuple(sorted(m.laser_time.items())), tuple(m.stopped_time), tuple(m.queue_area),
               len(m.queue_series))
    return (sim.step_index, sim.dt, sim.time, sim.finished, parts, convs, laser, metrics,
            len(sim.log), tuple(sim.pending_arrivals), sim.n_completed, sim.n_exited,
            tuple(sim._was_running))


class _Probe(BasePolicy):
    """nearest_job that runs a callback on the live simulation at decision ``k``."""

    name = "probe"

    def __init__(self, k: int, callback) -> None:
        super().__init__()
        self.k, self.n, self.callback = k, 0, callback
        self.base = get_policy("nearest_job")

    def select(self, candidates, ctx):
        choice = self.base.select(candidates, ctx)
        if self.n == self.k:
            self.callback(candidates, ctx, choice)
        self.n += 1
        return choice


# Two parts known at t=1.2; one more part arrives on conveyor 2 at t=11.4.
# With the real future visible, the very first decision (t=4.54) changes.
FAIR_JOBS = [(1.2, 1, 2.5), (1.2, 2, 2.5)]
FUTURE_JOB = (11.4, 2, 2.5)


# ----------------------------------------------------------------- tests
def test_registered_policy_and_method():
    assert "rollout" in available_policies()
    methods = discover()
    assert "policy:rollout" in methods
    sol = methods["policy:rollout"](make_instance(FAIR_JOBS), "makespan", 1.0, 0)
    assert sol.policy == "rollout" and sol.policy_params["objective"] == "makespan"
    sol = methods["policy:rollout"](make_instance(FAIR_JOBS), "neg_throughput", 1.0, 0)
    assert sol.policy_params["objective"] == "avg_flow_time"     # unsupported -> default


def test_invalid_parameters():
    with pytest.raises(ValueError):
        RolloutPolicy(base_policy="rollout")
    with pytest.raises(ValueError):
        RolloutPolicy(objective="not_an_objective")
    with pytest.raises(ValueError):
        RolloutPolicy(max_candidates=0)


def test_no_future_information_is_used():
    without = make_instance(FAIR_JOBS)
    with_future = make_instance(FAIR_JOBS + [FUTURE_JOB])
    t_future = FUTURE_JOB[0]

    # The instance is discriminating: a clairvoyant look-ahead reacts to the future part.
    clair_a = _assignments(_run(without, _ClairvoyantRollout()), t_future)
    clair_b = _assignments(_run(with_future, _ClairvoyantRollout()), t_future)
    assert clair_a != clair_b

    # The real rollout makes exactly the same decisions before the part arrives.
    for params in ({}, {"dt_lookahead": None, "max_candidates": None},
                   {"future_samples": 2, "future_window_s": 20.0}):
        fair_a = _assignments(_run(without, RolloutPolicy(**params)), t_future)
        fair_b = _assignments(_run(with_future, RolloutPolicy(**params)), t_future)
        assert fair_a and fair_a == fair_b, params


def test_clone_is_isolated_and_has_no_future():
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=3",
                                 "simulation.arrival_horizon=80"])
    seen = {}

    def callback(candidates, ctx, choice):
        sim = ctx.simulation
        ro = RolloutPolicy()
        clone = ro.clone(sim, _ForcedFirstPolicy(choice.id, get_policy("nearest_job")))
        assert sim.pending_arrivals and not clone.pending_arrivals
        assert isinstance(clone.log, _NullEventLog) and clone.trace is None
        assert clone.policy is not sim.policy and not isinstance(clone.policy, RolloutPolicy)
        assert clone.cfg is sim.cfg                            # immutable, shared
        own = [sim.laser, sim.metrics, sim.parts, *sim.parts, *sim.conveyors,
               sim.metrics.queue_series, sim.metrics.stopped_time]
        cloned = [clone.laser, clone.metrics, clone.parts, *clone.parts, *clone.conveyors,
                  clone.metrics.queue_series, clone.metrics.stopped_time]
        assert all(a is not b for a, b in zip(own, cloned))
        seen["ok"] = True

    Simulation(cfg, policy=_Probe(6, callback), record_trace=False).run()
    assert seen.get("ok")


@pytest.mark.parametrize("k", [0, 4])
def test_lookahead_leaves_original_untouched(k):
    """Running every look-ahead at a decision does not change the original state,
    and the run continues exactly as the run without look-ahead."""
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=11",
                                 "simulation.arrival_horizon=60"])
    reference = Simulation(cfg, policy=get_policy("nearest_job"), record_trace=True).run()

    def callback(candidates, ctx, choice):
        before = _fingerprint(ctx.simulation)
        ro = RolloutPolicy(dt_lookahead=None, max_candidates=None)
        for cand in candidates:
            ro.lookahead_cost(ctx.simulation, cand.id)
        assert _fingerprint(ctx.simulation) == before

    probed = Simulation(cfg, policy=_Probe(k, callback), record_trace=True).run()
    assert [(p.id, p.start_processing_time, p.completion_time) for p in probed.parts] == \
           [(p.id, p.start_processing_time, p.completion_time) for p in reference.parts]
    assert len(probed.events) == len(reference.events)


def test_forced_clone_reproduces_base_run():
    """Clone with the real future + same dt, forced to the base choice and continued
    with the base policy == the base-policy run, bit for bit (validates the
    'resume inside step 5' mechanism of the clone)."""
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=7",
                                 "simulation.arrival_horizon=80"])
    base = Simulation(cfg, policy=get_policy("nearest_job"), record_trace=False).run()
    clones = {}

    def callback(candidates, ctx, choice):
        ro = RolloutPolicy(dt_lookahead=None)
        clone = ro.clone(ctx.simulation, _ForcedFirstPolicy(choice.id, get_policy("nearest_job")))
        clone.pending_arrivals = copy.deepcopy(ctx.simulation.pending_arrivals)   # test only
        while not clone.finished:
            clone.step()
        clones["c"] = clone

    for k in (0, 3, 9):
        Simulation(cfg, policy=_Probe(k, callback), record_trace=False).run()
        c = clones.pop("c")
        rec = lambda parts: [(p.id, p.start_processing_time, p.completion_time, p.processing_x,
                              p.exit_time) for p in parts]
        assert rec(c.parts) == rec(base.parts)
        assert c.time == base.summary["total_simulation_time"]
        assert c.laser.distance == base.summary["total_laser_travel_distance"]


def test_determinism():
    cfg_items = ["arrivals.mode=stochastic", "simulation.seed=5", "simulation.arrival_horizon=90",
                 "policy.name=rollout"]
    r1 = Simulation(load_config(set_items=cfg_items), record_trace=False).run()
    r2 = Simulation(load_config(set_items=cfg_items), record_trace=False).run()
    assert [p.as_record() for p in r1.parts] == [p.as_record() for p in r2.parts]
    assert r1.events.lines() == r2.events.lines()
    assert r1.summary == r2.summary


def _simultaneous_instance(seed: int, n: int) -> Instance:
    rng = random.Random(seed)
    return make_instance([(0.0, rng.randint(1, 3), round(rng.uniform(1.5, 3.5), 2))
                          for _ in range(n)], name=f"t0_{seed}")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_never_worse_than_base_when_lookahead_is_exact(seed):
    """All parts arrive at t=0 -> every part is known at every decision, so with the
    same dt and no horizon the look-ahead is exact and rollout's cost-improvement
    property applies: avg flow time(rollout) <= avg flow time(nearest_job)."""
    inst = _simultaneous_instance(seed, 6)
    base = evaluate_policy(inst, "nearest_job", objective="avg_flow_time")
    ro = evaluate_policy(inst, "rollout", {"dt_lookahead": None, "max_candidates": None},
                         objective="avg_flow_time")
    assert base.feasible and ro.feasible
    assert ro.objective <= base.objective + 1e-9


@pytest.mark.parametrize("objective", ["makespan", "avg_waiting_time"])
def test_never_worse_other_objectives(objective):
    inst = _simultaneous_instance(7, 6)
    base = evaluate_policy(inst, "nearest_job", objective=objective)
    ro = evaluate_policy(inst, "rollout", {"dt_lookahead": None, "max_candidates": None,
                                           "objective": objective}, objective=objective)
    assert ro.objective <= base.objective + 1e-9


def test_schedule_passes_formal_check():
    inst = make_instance([(0.0, 1, 2.5), (1.0, 2, 2.5), (2.0, 3, 2.5), (6.0, 1, 2.5),
                          (7.5, 3, 2.0), (9.0, 2, 3.0), (12.0, 1, 2.5), (13.0, 3, 2.5)])
    cfg = instance_config(inst, "rollout", {}, 0.01)
    result = Simulation(cfg, record_trace=False).run()
    assert result.summary["parts_completed"] == inst.n
    assert check_schedule(result) == []
    ev = evaluate_policy(inst, "rollout", objective="avg_flow_time")
    assert ev.feasible and not ev.violations


def test_decision_statistics_are_collected():
    pol = RolloutPolicy()
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=2",
                                 "simulation.arrival_horizon=60"])
    Simulation(cfg, policy=pol, record_trace=False).run()
    st = pol.stats()
    assert st["decisions"] > 0 and st["decision_time_total_s"] > 0
