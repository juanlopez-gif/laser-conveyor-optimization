"""Validity of the lower bounds of optimization/lower_bounds.py and reproduction of
the counterexamples of docs/methods/theory.md.

A bound is valid if it is <= the objective of EVERY feasible schedule. We check it
against every schedule we can simulate: the four online policies, random
permutations through the decoder, and the exhaustive optimum of the decoder on
tiny instances. The simulator integrates with a step dt, so simulated values are
exact only up to O(dt); comparisons use the tolerance TOL = 2 * dt.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from optimization.benchmark import trivial_lower_bound
from optimization.evaluate import evaluate_policy, evaluate_sequence
from optimization.instances import generate_instance, instance_config, make_instance
from optimization.lower_bounds import (_Relaxation, all_lower_bounds, best_lower_bound,
                                       bound_data, earliest_starts, sequence_lower_bound,
                                       srpt_sum_completion)
from optimization.solvers.baselines import exhaustive
from policies import get_policy
from simulation.engine import Simulation

DT = 0.05
TOL = 2 * DT
OBJECTIVES = ("makespan", "avg_flow_time", "avg_waiting_time")
POLICIES = ("fifo", "nearest_job", "round_robin", "weighted_score")
SINGLE = {"conveyors.count": 1, "conveyors.y_positions": [0.0]}


def _deterministic_instances():
    return [
        make_instance([(0, 1, 2.5), (0, 1, 2.5), (0, 1, 2.5), (0.5, 2, 2.5), (1, 3, 2.5), (1, 1, 2.5)],
                      "burst"),
        # delta = 9 cm: a part 4 places behind on the belt forces its predecessors (blocking)
        make_instance([(0, 1, 1.0)] * 5, "forced_predecessors", {"simulation.min_part_spacing": 9.0}),
        generate_instance(6, 4, overrides={"simulation.conveyor_stop_policy": "all_conveyors"}),
        generate_instance(5, 5, overrides={"laser.motion_model": "sequential"}),
        generate_instance(6, 7, overrides={"processing.distribution": "uniform"}),
    ]


def _check_against(inst, summaries):
    lbs = {o: all_lower_bounds(inst, o) for o in OBJECTIVES}
    assert summaries
    for s in summaries:
        for o in OBJECTIVES:
            for name, val in lbs[o].items():
                assert val <= s[o] + TOL, (inst.name, o, name, val, s[o])


def _policy_summaries(inst):
    out = []
    for pol in POLICIES:
        ev = evaluate_policy(inst, pol, objective="makespan", dt=DT, check=False)
        assert ev.feasible
        out.append(ev.summary)
    return out


RANDOM_CASES = [(4, 1), (4, 2), (5, 3), (6, 1), (7, 2), (8, 3)]


@pytest.mark.parametrize("n,seed", RANDOM_CASES)
def test_bounds_below_policies_and_random_permutations(n, seed):
    inst = generate_instance(n, seed)
    sums = _policy_summaries(inst)
    rng = random.Random(seed)
    for _ in range(6):
        ev = evaluate_sequence(inst, rng.sample(inst.ids, n), "makespan", dt=DT)
        if ev.feasible:
            sums.append(ev.summary)
    _check_against(inst, sums)


@pytest.mark.parametrize("inst", _deterministic_instances(), ids=lambda i: i.name)
def test_bounds_below_schedules_on_deterministic_instances(inst):
    sums = _policy_summaries(inst)
    rng = random.Random(0)
    for _ in range(4):
        ev = evaluate_sequence(inst, rng.sample(inst.ids, inst.n), "makespan", dt=DT)
        if ev.feasible:
            sums.append(ev.summary)
    _check_against(inst, sums)


def test_bounds_below_every_permutation_and_exhaustive_optimum():
    inst = generate_instance(4, 3)
    sums = []
    for perm in itertools.permutations(inst.ids):
        ev = evaluate_sequence(inst, list(perm), "makespan", dt=DT)
        if ev.feasible:
            sums.append(ev.summary)
    _check_against(inst, sums)
    for obj in ("makespan", "avg_flow_time"):
        sol = exhaustive(inst, obj, 60.0, 0)
        best = evaluate_sequence(inst, sol.sequence, obj, dt=DT).objective
        assert best_lower_bound(inst, obj) <= best + TOL


@pytest.mark.parametrize("n", [4, 6, 8, 12, 25])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_makespan_bound_at_least_trivial(n, seed):
    inst = generate_instance(n, seed)
    assert best_lower_bound(inst, "makespan") >= trivial_lower_bound(inst) - 1e-9
    f = best_lower_bound(inst, "avg_flow_time")
    w = best_lower_bound(inst, "avg_waiting_time")
    mean_p = sum(j.processing_time for j in inst.jobs) / n
    assert f == pytest.approx(w + mean_p)
    assert best_lower_bound(inst, {"makespan": 1.0}) is None


def test_earliest_start_includes_queue_and_home():
    # three parts released together on conveyor 1: m-th part needs (a + (m-1) delta)/v
    inst = make_instance([(0, 1, 2.5)] * 3, "queue")
    cfg = instance_config(inst)
    a, v, delta = cfg.geometry.active_start, cfg.conveyors[0].speed, cfg.simulation.min_part_spacing
    e = earliest_starts(bound_data(inst))
    assert e == pytest.approx([(a + m * delta) / v for m in range(3)])


def test_srpt_small_cases():
    assert srpt_sum_completion([0, 0], [2, 1]) == pytest.approx(1 + 3)
    # job 1 preempted by the short job 2 released at t = 1
    assert srpt_sum_completion([0, 1], [4, 1]) == pytest.approx(2 + 5)


def test_dp_equals_min_over_orders_of_sequence_bound():
    inst = generate_instance(5, 2)
    d = bound_data(inst)
    rel = _Relaxation(d, list(range(d.n)))
    cmax, sumc = rel.dp()
    seqs = [rel.sequence(list(p)) for p in itertools.permutations(range(d.n))]
    # makespan: the DP is exact for the relaxation; sum C: the two-label DP is a
    # relaxation of the relaxation (never above the brute-force minimum)
    assert cmax == pytest.approx(min(max(c) for c in seqs))
    assert sumc <= min(sum(c) for c in seqs) + 1e-9


def test_window_bounds_large_instance():
    inst = generate_instance(14, 1)
    _check_against(inst, _policy_summaries(inst))


# --------------------------------------------------------------------------- #
# Special cases of docs/methods/theory.md                                     #
# --------------------------------------------------------------------------- #
def test_counterexample_single_conveyor_fifo_not_optimal_for_flow_time():
    """Counterexample A: one conveyor, p = (10, 1). Every schedule of P that
    processes in belt order has mean flow >= 13.75 s; the reverse order achieves
    about 11.77 s (simulated, checked against the formal model)."""
    inst = make_instance([(0.0, 1, 10.0), (1.5, 1, 1.0)], "ce_single_flow", SINGLE)
    fifo_lb = sequence_lower_bound(inst, ["P001", "P002"])["avg_flow_time"]
    rev = evaluate_sequence(inst, ["P002", "P001"], "avg_flow_time", dt=0.01, check=True)
    assert rev.feasible and not rev.violations
    assert fifo_lb == pytest.approx(13.75)
    assert rev.objective < fifo_lb - 1.5
    # makespan: belt order is better here
    fifo = evaluate_sequence(inst, ["P001", "P002"], "makespan", dt=0.01, check=True)
    assert fifo.objective < evaluate_sequence(inst, ["P002", "P001"], "makespan", dt=0.01).objective


def test_counterexample_decoder_gap_anticipation():
    """Counterexample D: one part. P can pre-position the laser and start at
    r + a/v; the decoder / every policy waits for the part to enter the zone."""
    inst = make_instance([(0.0, 1, 2.5)], "ce_phi_gap")
    cfg = instance_config(inst)
    a, v = cfg.geometry.active_start, cfg.conveyors[0].speed
    lb = best_lower_bound(inst, "makespan")
    assert lb == pytest.approx(a / v + 2.5)                 # attained by a P-schedule (proof in doc)
    for pol in POLICIES:
        ev = evaluate_policy(inst, pol, objective="makespan")
        assert ev.objective > lb + 1.2                      # strictly worse than P's optimum


def test_light_traffic_all_policies_coincide_and_bound_is_exact():
    """Theorem LT: the instance satisfies the light-traffic conditions."""
    inst = make_instance([(6.0 * i, i % 3 + 1, 2.5) for i in range(6)], "light_traffic")
    d = bound_data(inst)
    e0 = [d.r[j] + d.a / d.v[j] for j in range(d.n)]
    lb_m, lb_f = best_lower_bound(inst, "makespan"), best_lower_bound(inst, "avg_flow_time")
    assert lb_m == pytest.approx(max(e0) + 2.5)
    assert lb_f == pytest.approx(d.a / d.v[0] + 2.5)
    imax = max((d.b - d.a) / (d.vx + max(d.v)), (max(d.y) - min(d.y)) / d.vy)
    starts = []
    for pol in POLICIES:
        cfg = instance_config(inst, pol, dt=0.01)
        res = Simulation(cfg, policy=get_policy(pol, cfg.policy_params), record_trace=False).run()
        s = [p.start_processing_time for p in res.parts]
        assert all(0 <= s[j] - e0[j] <= imax + 0.02 for j in range(d.n))
        starts.append(s)
    assert all(s == starts[0] for s in starts)


def test_counterexample_nearest_and_fifo_not_optimal():
    """Counterexample C2: generate_instance(5, 3). Every policy serves P004 as soon as
    it enters the zone; waiting 0.2 s for P005 (on the laser's conveyor) is better
    for both makespan and mean flow time."""
    inst = generate_instance(5, 3)
    better = evaluate_sequence(inst, ["P001", "P002", "P003", "P005", "P004"], "makespan",
                               dt=0.01, check=True)
    assert better.feasible and not better.violations
    for pol in POLICIES:
        ev = evaluate_policy(inst, pol, objective="makespan")
        assert ev.sequence == ["P001", "P002", "P003", "P004", "P005"]
        assert ev.summary["makespan"] > better.summary["makespan"] + 0.5
        assert ev.summary["avg_flow_time"] > better.summary["avg_flow_time"] + 0.2
    assert best_lower_bound(inst, "makespan") <= better.summary["makespan"]


def test_bounds_finite_without_blocking():
    inst = generate_instance(5, 9, overrides={"simulation.unprocessed_at_zone_end": "pass"})
    assert math.isfinite(best_lower_bound(inst, "makespan"))
