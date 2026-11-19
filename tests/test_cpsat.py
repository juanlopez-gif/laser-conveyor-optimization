"""Tests of the CP-SAT model of P_Delta (src/optimization/solvers/cpsat.py).

* tiny hand-made instances whose optimum follows from reasoning;
* agreement with exhaustive search over the simulator decoder on n = 5 instances;
* the solver bound never exceeds the replayed objective (up to discretisation);
* every returned sequence replays feasibly in the simulator (formulation checker).

Expected values are derived from the configuration, never hard-coded.
"""

from __future__ import annotations

import itertools
import math

import pytest

pytest.importorskip("ortools")

from optimization.evaluate import evaluate_sequence                     # noqa: E402
from optimization.instances import generate_instance, instance_config, make_instance  # noqa: E402
from optimization.solvers import discover                               # noqa: E402
from optimization.solvers.baselines import exhaustive                   # noqa: E402
from optimization.solvers.cpsat import (DEFAULT_STEP, decode_order,     # noqa: E402
                                        scale_instance, solve_cpsat)

STEP = DEFAULT_STEP
#: discretisation tolerance [s]: release rounded up (< Delta), every travel rounded up
#: to whole steps (< Delta per move) and grid start times; a few steps in total
TOL = 5 * STEP


def _first_reach(cfg) -> tuple[float, float]:
    """Time for a part to reach the zone (a / v) and laser time home -> (a, y_1)."""
    c1 = cfg.conveyors[0]
    las = cfg.laser
    travel = max(abs(las.home_x - cfg.geometry.active_start) / las.speed_x,
                 abs(las.home_y - c1.y) / las.speed_y)
    return cfg.geometry.active_start / c1.speed, travel


def _grid(t: float) -> float:
    return math.ceil(t / STEP - 1e-9) * STEP


# ---------------------------------------------------------------- hand-made instances
def test_single_part_optimum_by_reasoning():
    """One part: the laser waits at the zone entrance, S = a/v on the grid."""
    inst = make_instance([(0.0, 1, 2.5)], "single")
    cfg = instance_config(inst)
    reach, travel = _first_reach(cfg)
    assert travel <= reach                      # the laser is there before the part
    expected = _grid(reach) + 2.5
    for obj in ("makespan", "avg_flow_time"):
        sol = solve_cpsat(inst, obj, time_limit=10, seed=0)
        assert sol.proven_optimal
        assert sol.sequence == ["P001"]
        assert sol.info["model_objective"] == pytest.approx(expected, abs=1e-6)
        assert sol.lower_bound == pytest.approx(expected, abs=1e-6)
        assert sol.info["replay_feasible"]


def test_two_independent_parts_same_position():
    """Two parts far apart in time on the same belt: no interaction, the laser stays at
    the processing point of the first, so both start at r_j + a/v (on the grid)."""
    gap = 30.0
    inst = make_instance([(0.0, 1, 2.5), (gap, 1, 2.5)], "independent")
    reach, _ = _first_reach(instance_config(inst))
    s1, s2 = _grid(reach), _grid(gap + reach)
    sol = solve_cpsat(inst, "makespan", time_limit=10, seed=0)
    assert sol.proven_optimal and sol.sequence == ["P001", "P002"]
    assert sol.info["model_objective"] == pytest.approx(s2 + 2.5, abs=1e-6)
    sol = solve_cpsat(inst, "avg_flow_time", time_limit=10, seed=0)
    assert sol.proven_optimal
    assert sol.info["model_objective"] == pytest.approx((s1 + 2.5 + s2 + 2.5 - gap) / 2, abs=1e-6)


def test_ready_dispatch_matches_simulator_on_single_part():
    """With the simulator's dispatch rule the model value equals the replay up to TOL."""
    inst = make_instance([(0.0, 1, 2.5)], "single")
    sol = solve_cpsat(inst, "makespan", time_limit=10, seed=0, ready_dispatch=True)
    assert sol.proven_optimal
    assert abs(sol.info["replay_objective"] - sol.info["model_objective"]) <= TOL


def test_blocking_and_all_conveyors_stop_rule():
    """A burst on one belt forces blocking; the all-conveyors stop rule is honoured."""
    jobs = [(0.0, 1, 2.5), (0.5, 1, 2.5), (1.0, 1, 2.5), (1.0, 3, 2.5), (1.5, 2, 2.5)]
    for overrides in ({}, {"simulation.conveyor_stop_policy": "all_conveyors"}):
        inst = make_instance(jobs, "burst", overrides)
        sol = solve_cpsat(inst, "makespan", time_limit=15, seed=0)
        assert sol.sequence is not None and sol.info["replay_feasible"]
        assert sol.info["replay_violations"] == 0
        assert sol.lower_bound <= sol.info["replay_objective"] + TOL


def test_discrete_decoder_respects_window():
    """The decoder (warm start) never processes outside [a, b]."""
    inst = generate_instance(6, 2)
    d = scale_instance(inst, instance_config(inst), STEP)
    sched = decode_order(d, list(range(inst.n)), 10_000)
    assert sched is not None
    assert all(d.a <= x <= d.b for x in sched.X)


def test_registered():
    methods = discover()
    assert "cpsat" in methods and "cpsat_ready" in methods


# ---------------------------------------------------------------- exhaustive agreement
OBJECTIVES = ("makespan", "avg_flow_time")


@pytest.fixture(scope="module")
def enumerated():
    """Exhaustive search over the decoder, as ``baselines.exhaustive`` does (same
    evaluation, search dt), but scoring both objectives in a single enumeration."""
    cache: dict[int, dict[str, tuple[float, list[str]]]] = {}

    def get(seed: int) -> dict[str, tuple[float, list[str]]]:
        if seed not in cache:
            inst = generate_instance(5, seed)
            best = {obj: (math.inf, []) for obj in OBJECTIVES}
            for perm in itertools.permutations(inst.ids):
                summary = evaluate_sequence(inst, list(perm)).summary
                for obj in OBJECTIVES:
                    if summary.get(obj) is not None and summary[obj] < best[obj][0]:
                        best[obj] = (summary[obj], list(perm))
            cache[seed] = best
        return cache[seed]
    return get


def test_exhaustive_baseline_agrees_on_n4():
    """Direct comparison with ``baselines.exhaustive`` on a 4-part instance."""
    inst = generate_instance(4, 2)
    for objective in OBJECTIVES:
        ex = exhaustive(inst, objective, 60, 0)
        ex_val = evaluate_sequence(inst, ex.sequence, objective, dt=0.01, check=True).objective
        ready = solve_cpsat(inst, objective, time_limit=20, seed=0, ready_dispatch=True)
        assert ready.proven_optimal and ready.info["replay_feasible"]
        assert ready.info["replay_objective"] <= ex_val + TOL
        sol = solve_cpsat(inst, objective, time_limit=20, seed=0)
        assert sol.info["model_objective"] <= ex_val + TOL


@pytest.mark.parametrize("seed", [1, 3])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_agreement_with_exhaustive(enumerated, seed, objective):
    inst = generate_instance(5, seed)
    _, ex_seq = enumerated(seed)[objective]
    ex_val = evaluate_sequence(inst, ex_seq, objective, dt=0.01, check=True).objective

    # formulation P_Delta: a relaxation of the simulator's decoder (the laser may
    # anticipate), so its optimum cannot be worse than the exhaustive optimum
    sol = solve_cpsat(inst, objective, time_limit=20, seed=0)
    assert sol.proven_optimal
    assert sol.info["model_objective"] <= ex_val + TOL
    assert sol.lower_bound <= sol.info["replay_objective"] + TOL
    assert sol.info["replay_feasible"] and sol.info["replay_violations"] == 0

    # simulator semantics: the replayed CP-SAT order matches the exhaustive optimum
    ready = solve_cpsat(inst, objective, time_limit=20, seed=0, ready_dispatch=True)
    assert ready.proven_optimal and ready.info["replay_feasible"]
    assert abs(ready.info["replay_objective"] - ex_val) <= TOL
    assert abs(ready.info["replay_objective"] - ready.info["model_objective"]) <= TOL
