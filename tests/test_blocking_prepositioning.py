"""Tests of version 0.4.0: blocking penalty, nearest_head policy, laser pre-positioning.

See docs/CHANGELOG.md (0.4.0) and docs/formal_formulation.md §1.4-1.5.
"""

from __future__ import annotations

import math

import pytest

from optimization.evaluate import evaluate_policy, evaluate_sequence
from optimization.instances import make_instance
from optimization.objective_functions import OBJECTIVES
from optimization.schedule_check import check_schedule
from policies import available_policies, get_policy
from policies.nearest_head import NearestHeadPolicy, conveyor_heads
from simulation.config import load_config
from simulation.engine import Simulation
from simulation.events import EventKind
from simulation.motion import find_intercept
from simulation.part import Part, PartStatus

ONE_PART = ["arrivals.mode=explicit",
            "arrivals.explicit=[{time: 0, conveyor: 1, processing_time: 2.5}]"]
# Two long jobs at t=0 on the outer conveyors: while the laser processes the first,
# the second reaches the end of the laser zone and blocks its conveyor.
BLOCKING = ["arrivals.mode=explicit",
            "arrivals.explicit=[{time: 0, conveyor: 1, processing_time: 15}, "
            "{time: 0, conveyor: 3, processing_time: 15}]"]


def run(set_items, policy=None):
    cfg = load_config(set_items=set_items)
    return Simulation(cfg, policy=policy, record_trace=False).run()


# --------------------------------------------------------------- blocking penalty
def test_blocking_is_measured_and_penalized():
    r = run(BLOCKING + ["objective.blocking_penalty.per_event=7",
                        "objective.blocking_penalty.per_second=2"])
    s = r.summary
    blocked = [p for p in r.parts if p.caused_block]
    assert s["blocking_events"] == 1 and len(blocked) == 1
    p = blocked[0]
    assert p.processing_x == pytest.approx(r.config.geometry.active_end)   # held at b
    assert p.blocked_duration > 0
    # T_B = sum_j (S_j - B_j): the conveyor-time integral equals the per-part sum
    assert s["total_blocked_time"] == pytest.approx(
        sum(q.blocked_duration for q in blocked), abs=1e-6)
    assert s["blocking_penalty"] == pytest.approx(7 * 1 + 2 * s["total_blocked_time"])
    assert OBJECTIVES["penalized_flow_time"](s) == pytest.approx(
        s["avg_flow_time"] + s["blocking_penalty"] / s["parts_completed"])
    assert OBJECTIVES["penalized_makespan"](s) == pytest.approx(
        s["makespan"] + s["blocking_penalty"])
    assert check_schedule(r) == []


def test_no_blocking_no_penalty():
    s = run(ONE_PART).summary
    assert s["blocking_events"] == 0
    assert s["total_blocked_time"] == 0.0 and s["blocking_penalty"] == 0.0
    assert OBJECTIVES["penalized_flow_time"](s) == pytest.approx(s["avg_flow_time"])


def test_blocked_time_identity_on_a_loaded_run():
    """T_B = sum_j (S_j - B_j) also on a long stochastic run with many blocks."""
    r = run(["arrivals.mode=stochastic", "simulation.seed=3", "policy.name=round_robin"])
    blocked = [p for p in r.parts if p.caused_block]
    assert r.summary["blocking_events"] > 0
    assert r.summary["total_blocked_time"] == pytest.approx(
        sum(p.blocked_duration for p in blocked), abs=1e-6)


def test_default_objective_is_penalized_flow_time():
    cfg = load_config()
    assert cfg.objective_weights["penalized_flow_time"] == 1.0
    assert cfg.blocking_penalty_per_event > 0 and cfg.blocking_penalty_per_second > 0


def test_negative_penalty_rejected():
    with pytest.raises(ValueError):
        load_config(set_items=["objective.blocking_penalty.per_event=-1"])


# ------------------------------------------------------------------ nearest_head
def _part(pid, conv, x):
    p = Part(id=pid, conveyor_id=conv, arrival_time=0.0, processing_time=1.0,
             due_date=10.0, x=x)
    p.status = PartStatus.READY
    return p


def test_conveyor_heads_are_most_downstream():
    parts = [_part("P001", 1, 30.0), _part("P002", 1, 20.0), _part("P003", 2, 12.0)]
    assert {p.id for p in conveyor_heads(parts)} == {"P001", "P003"}


class _Spy(NearestHeadPolicy):
    def __init__(self):
        super().__init__()
        self.violations = 0
        self.decisions = 0

    def select(self, candidates, ctx):
        choice = super().select(candidates, ctx)
        self.decisions += 1
        assert choice is not None                     # never idles with candidates
        own = [p for c in ctx.conveyors if c.id == choice.conveyor_id
               for p in c.parts if p.needs_processing]
        if any(p.x > choice.x + 1e-9 for p in own):   # an unprocessed part ahead of it
            self.violations += 1
        return choice


@pytest.mark.parametrize("prepositioning", ["none", "when_idle", "always"])
def test_nearest_head_keeps_belt_order(prepositioning):
    spy = _Spy()
    r = run(["arrivals.mode=stochastic", "simulation.seed=5", "simulation.arrival_horizon=150",
             f"laser.prepositioning={prepositioning}"], policy=spy)
    assert spy.decisions > 10 and spy.violations == 0
    assert check_schedule(r) == []
    # belt order inside every conveyor
    for k in (1, 2, 3):
        starts = [p.start_processing_time for p in sorted(r.parts, key=lambda p: p.number)
                  if p.conveyor_id == k and p.start_processing_time is not None]
        assert starts == sorted(starts)


def test_nearest_head_registered():
    assert "nearest_head" in available_policies()
    assert get_policy("nearest_head").name == "nearest_head"


# --------------------------------------------------------------- pre-positioning
def test_intercept_upstream_part():
    # part at x=0 moving at 3 cm/s, zone starts at 10: meeting not before 10/3 s, x >= 10
    icp = find_intercept(27.5, 12.0, 0.0, 0.0, 3.0, 45.0, True, 10.0, 10.0,
                         "simultaneous", x_min=10.0)
    assert icp is not None and icp.x >= 10.0 - 1e-9 and icp.time >= 10.0 / 3 - 1e-9
    # laser already waiting at (10, 0): meets exactly when the part enters
    icp = find_intercept(10.0, 0.0, 0.0, 0.0, 3.0, 45.0, True, 10.0, 10.0,
                         "simultaneous", x_min=10.0)
    assert icp.time == pytest.approx(10.0 / 3) and icp.x == pytest.approx(10.0)
    # upstream part on a stopped belt: unreachable for now
    assert find_intercept(27.5, 12.0, 0.0, 0.0, 0.0, 45.0, True, 10.0, 10.0,
                          "simultaneous", x_min=10.0) is None
    # x_min has no effect on a part already inside the zone
    a = find_intercept(27.5, 12.0, 20.0, 0.0, 3.0, 45.0, True, 10.0, 10.0, "simultaneous")
    b = find_intercept(27.5, 12.0, 20.0, 0.0, 3.0, 45.0, True, 10.0, 10.0, "simultaneous",
                       x_min=10.0)
    assert a == b


@pytest.mark.parametrize("value,mode", [(False, "none"), (True, "always"),
                                        ("when_idle", "when_idle")])
def test_prepositioning_config_aliases(value, mode):
    v = str(value).lower() if isinstance(value, bool) else value
    assert load_config(set_items=[f"laser.prepositioning={v}"]).laser.prepositioning == mode


def test_prepositioning_value_single_part():
    """Counterexample D1 (docs/methods/theory.md): 7.19 s in P^sim, 5.83 s in P."""
    none = run(ONE_PART + ["laser.prepositioning=none"]).parts[0]
    pre = run(ONE_PART + ["laser.prepositioning=always"]).parts[0]
    assert none.completion_time == pytest.approx(7.19, abs=0.02)
    assert pre.completion_time == pytest.approx(5.83, abs=0.02)
    assert pre.processing_x == pytest.approx(10.0, abs=0.05)     # met at the zone start


@pytest.mark.parametrize("mode", ["when_idle", "always"])
@pytest.mark.parametrize("policy", ["fifo", "nearest_job", "nearest_head", "weighted_score"])
def test_prepositioning_schedules_are_feasible(mode, policy):
    r = run(["arrivals.mode=stochastic", "simulation.seed=11", "simulation.arrival_horizon=150",
             f"policy.name={policy}", f"laser.prepositioning={mode}"])
    assert r.summary["parts_completed"] == r.summary["parts_generated"]
    assert check_schedule(r) == []


def test_when_idle_never_holds_a_preposition_while_a_part_is_ready():
    """Invariant of 'when_idle': if after a step the laser is heading to an entry-zone
    part while another part is READY, the next step revokes that pre-positioning."""
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=2",
                                 "simulation.arrival_horizon=200",
                                 "laser.prepositioning=when_idle"])
    sim = Simulation(cfg, record_trace=False)
    prepositioned = must_revoke = 0
    while not sim.finished:
        cur = sim.laser.current_part
        pending = (cur is not None and cur.status == PartStatus.MOVING
                   and sim._ready_candidates_exist())
        sim.step()
        if pending:
            must_revoke += 1
            assert sim.laser.current_part is not cur
        cur = sim.laser.current_part
        if cur is not None and cur.status == PartStatus.MOVING:
            prepositioned += 1
    assert prepositioned > 0
    revoked = [e for e in sim.log.events
               if e.kind == EventKind.UNASSIGN and "revoked" in e.message]
    assert len(revoked) == must_revoke


def test_when_idle_revocation_happens():
    """Laser pre-positions to a part on conveyor 3; meanwhile a part on conveyor 1
    enters the laser zone -> the pre-positioning is revoked."""
    r = run(["arrivals.mode=explicit",
             "arrivals.explicit=[{time: 0, conveyor: 3, processing_time: 2}, "
             "{time: 0.2, conveyor: 1, processing_time: 2}]",
             "laser.home={x: 10, y: 24}", "laser.prepositioning=when_idle",
             "conveyors.speed=[1.0, 3.0, 0.5]"])
    msgs = [e.message for e in r.events.events]
    assert any("Pre-positioning" in m and "revoked" in m for m in msgs)
    assert r.summary["parts_completed"] == 2 and check_schedule(r) == []


def test_fixed_sequence_with_prepositioning_is_decoder_of_P():
    """With pre-positioning 'always' the decoder realises P: never later than P^sim."""
    inst = make_instance([(0.0, 1, 2.5), (1.0, 3, 2.5), (2.0, 2, 2.0), (6.0, 1, 2.5)])
    seq = ["P001", "P002", "P003", "P004"]
    sim_val = evaluate_sequence(inst, seq, "makespan", dt=0.01, check=True)
    inst_p = make_instance([(j.time, j.conveyor, j.processing_time) for j in inst.jobs],
                           overrides={"laser.prepositioning": "always"})
    p_val = evaluate_sequence(inst_p, seq, "makespan", dt=0.01, check=True)
    assert sim_val.feasible and p_val.feasible
    assert p_val.objective <= sim_val.objective + 0.02


def test_rollout_with_penalized_objective_runs():
    inst = make_instance([(0.0, 1, 2.5), (1.0, 2, 2.5), (2.0, 3, 2.5), (6.0, 1, 2.5),
                          (7.5, 3, 2.0), (9.0, 2, 3.0)])
    ev = evaluate_policy(inst, "rollout", {"objective": "penalized_flow_time"},
                         objective="penalized_flow_time")
    assert ev.feasible and math.isfinite(ev.objective)


def test_intercept_when_belt_outruns_the_laser():
    """External audit 2026-11-19: laser at x = 26, part at x = 20 on a 6 cm/s belt, laser
    X axis at 2 cm/s. They can meet at t = 0.75 s, x = 24.5 (the part comes towards the
    laser); the old search assumed a decreasing gap and returned the zone end (9.5 s,
    ``block``) or no meeting (``pass``)."""
    for hold in (True, False):
        icp = find_intercept(26.0, 10.0, 20.0, 10.0, 6.0, 45.0, hold, 2.0, 2.0,
                             "simultaneous")
        assert icp is not None
        assert icp.time == pytest.approx(0.75) and icp.x == pytest.approx(24.5)


def test_intercept_matches_brute_force():
    """The exact piecewise-linear solution equals a fine time scan on random cases,
    including belts faster than the laser and both motion models."""
    import random

    from simulation.motion import travel_time
    rng = random.Random(7)
    for _ in range(300):
        lx, ly, px, py = (rng.uniform(0, 45) for _ in range(4))
        v, vx, vy = rng.uniform(0.5, 8), rng.uniform(1, 12), rng.uniform(1, 12)
        model = rng.choice(["simultaneous", "sequential"])
        icp = find_intercept(lx, ly, px, py, v, 45.0, False, vx, vy, model)
        t_lim = (45.0 - px) / v
        grid = [t_lim * k / 2000 for k in range(2001)]
        first = next((t for t in grid
                      if travel_time(lx, ly, px + v * t, py, vx, vy, model) - t <= 1e-9), None)
        if first is None:
            # no grid point meets; an exact meeting may still lie between grid points
            assert icp is None or icp.time <= t_lim + 1e-9
            continue
        assert icp is not None and icp.time <= first + 1e-9
        assert icp.time >= first - t_lim / 2000 - 1e-9
        assert travel_time(lx, ly, icp.x, py, vx, vy, model) <= icp.time + 1e-6
