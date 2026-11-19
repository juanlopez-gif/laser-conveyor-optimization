"""Tests of the permutation metaheuristics (SA, ILS) and their incremental evaluator."""

from __future__ import annotations

import math
import random

import pytest

from optimization.evaluate import evaluate_policy, evaluate_sequence
from optimization.instances import generate_instance
from optimization.solvers import discover
from optimization.solvers.baselines import exhaustive
from optimization.solvers.metaheuristics import (WARM_START_POLICIES, SequenceEvaluator,
                                                 iterated_local_search, neighbourhood,
                                                 random_move, simulated_annealing)

METHODS = {"sa": simulated_annealing, "ils": iterated_local_search}


def test_methods_registered():
    registry = discover()
    assert "sa" in registry and "ils" in registry


def test_incremental_evaluation_matches_full_simulation():
    """Resuming from snapshots + repair gives exactly evaluate_sequence's value."""
    inst = generate_instance(12, 2)
    ev = SequenceEvaluator(inst, "avg_flow_time")
    base = evaluate_policy(inst, "nearest_job").sequence
    ev.set_reference(base)
    rng = random.Random(3)
    for _ in range(12):
        res = ev.evaluate(random_move(base, rng, 4))
        full = evaluate_sequence(inst, list(res.sequence), "avg_flow_time")
        assert full.objective == res.value or (math.isinf(res.value) and not full.feasible)
    assert ev.stats.steps_saved > 0


def test_repair_turns_deadlock_into_feasible_order():
    """Processing a late part first deadlocks its blocked conveyor; repair fixes it."""
    inst = generate_instance(10, 1)
    bad = [inst.ids[-1]] + inst.ids[:-1]          # last arrival first
    assert not evaluate_sequence(inst, bad).feasible
    res = SequenceEvaluator(inst, "makespan", max_repairs=20).evaluate(bad)
    assert math.isfinite(res.value) and res.repairs > 0
    assert evaluate_sequence(inst, list(res.sequence), check=True).feasible


def test_neighbourhood_is_restricted():
    seq = [f"P{i:03d}" for i in range(1, 11)]
    for cand in neighbourhood(seq, 2, random.Random(0)):
        moved = [abs(cand.index(p) - seq.index(p)) for p in seq]
        assert max(moved) <= 2 and sorted(cand) == seq


@pytest.mark.parametrize("name", ["sa", "ils"])
def test_deterministic_permutation_feasible_and_not_worse(name):
    inst = generate_instance(10, 2)
    a = METHODS[name](inst, "avg_flow_time", 60, seed=7, max_evals=40)
    b = METHODS[name](inst, "avg_flow_time", 60, seed=7, max_evals=40)
    assert a.sequence == b.sequence and a.info["value"] == b.info["value"]
    assert sorted(a.sequence) == sorted(inst.ids)
    ev = evaluate_sequence(inst, a.sequence, "avg_flow_time", dt=0.01, check=True)
    assert ev.feasible and not ev.violations
    best_policy = min(evaluate_policy(inst, p, objective="avg_flow_time").objective
                      for p in WARM_START_POLICIES)
    assert ev.objective <= best_policy + 1e-9


@pytest.mark.parametrize("seed,objective", [(3, "avg_flow_time"), (1, "makespan")])
def test_ils_never_worse_than_policies_and_accepts_weights(seed, objective):
    inst = generate_instance(15, seed)
    sol = iterated_local_search(inst, objective, 60, seed=0, max_evals=30)
    ev = evaluate_sequence(inst, sol.sequence, objective, dt=0.01, check=True)
    assert ev.feasible
    assert ev.objective <= min(sol.info["policy_values"].values()) + 1e-9
    weights = {"makespan": 1.0, "avg_flow_time": 2.0}
    sol_w = simulated_annealing(inst, weights, 60, seed=0, max_evals=20)
    assert evaluate_sequence(inst, sol_w.sequence, weights, dt=0.01, check=True).feasible


# n=5, seed 3: the exhaustive optimum is strictly better than every policy for both
# objectives (9.32 -> 8.94 avg flow, 30.45 -> 29.5 makespan at the search dt).
@pytest.mark.parametrize("n,seed,objective", [(5, 3, "avg_flow_time"), (5, 3, "makespan")])
def test_matches_exhaustive_optimum_on_small_instances(n, seed, objective):
    inst = generate_instance(n, seed)
    opt_seq = exhaustive(inst, objective, 120, 0).sequence
    opt = evaluate_sequence(inst, opt_seq, objective, dt=0.01).objective
    sol = iterated_local_search(inst, objective, 60, seed=0, max_evals=120)
    val = evaluate_sequence(inst, sol.sequence, objective, dt=0.01, check=True).objective
    assert val <= opt + 0.06          # equal up to the dt discretisation of the search


def test_time_limit_respected():
    import time
    inst = generate_instance(20, 1)
    t0 = time.time()
    sol = simulated_annealing(inst, "avg_flow_time", 4.0, seed=1)
    assert time.time() - t0 < 4.0 + 0.5
    assert sorted(sol.sequence) == sorted(inst.ids)
