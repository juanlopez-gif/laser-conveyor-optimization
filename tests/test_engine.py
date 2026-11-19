"""Tests 4, 9, 10 + policies, configuration and N conveyors."""

from __future__ import annotations

import math

import pytest

from optimization.objective_functions import composite_objective, pareto_front
from policies import available_policies, get_policy
from simulation.config import load_config
from simulation.engine import Simulation
from simulation.laser import Laser, LaserBusyError, LaserState
from simulation.part import Part, PartStatus

DT = 0.01


# 4 -------------------------------------------------------------------------
def test_laser_refuses_second_job(config):
    laser = Laser.from_spec(config.laser)
    ws = config.laser.workspace
    a = Part("P001", 1, 0.0, 3.0, 20.0, x=ws.x_min + 1, y=ws.y_min)
    b = Part("P002", 2, 0.0, 3.0, 20.0, x=ws.x_min + 2, y=ws.y_min)
    laser.start_processing(a, 0.0)
    with pytest.raises(LaserBusyError):
        laser.start_processing(b, 0.5)
    with pytest.raises(LaserBusyError):
        laser.assign(b, b.x, b.y, 0.0)


@pytest.mark.parametrize("policy", ["fifo", "nearest_job", "round_robin", "weighted_score"])
def test_never_two_parts_processing_at_once(config, policy):
    sim = Simulation(config, policy=get_policy(policy, {}), record_trace=False)
    while not sim.finished:
        sim.step()
        processing = [p for p in sim.parts if p.status == PartStatus.PROCESSING]
        assert len(processing) <= 1
    # processing intervals do not overlap
    done = sorted((p for p in sim.parts if p.is_processed), key=lambda p: p.start_processing_time)
    for prev, nxt in zip(done, done[1:]):
        assert nxt.start_processing_time >= prev.completion_time - 1e-9


# 9 -------------------------------------------------------------------------
def test_metrics_single_part_analytic(explicit_config):
    """One part on C1, laser at the workspace centre (27.5, 12), all speeds 5 cm/s.

    * part reaches the laser zone (x=10) at t=2.0
    * interception: x(t)=10+5t, laser needs max(|x-27.5|, 12)/5 -> meets at x=22 after 2.4 s
    * processing 4.4 -> 7.4, conveyor 1 stopped 3 s, exit at 7.4 + (55-22)/5 = 14.0
    """
    cfg = explicit_config([{"time": 0.0, "conveyor": 1, "processing_time": 3.0}],
                          conveyors__speed=5.0, laser__speed_x=5.0, laser__speed_y=5.0)
    s = Simulation(cfg, record_trace=False).run().summary
    tol = 2 * DT + 1e-6
    assert s["parts_generated"] == 1 and s["parts_completed"] == 1 and s["parts_exited"] == 1
    assert s["avg_waiting_time"] == pytest.approx(4.4, abs=tol)
    assert s["max_waiting_time"] == pytest.approx(4.4, abs=tol)
    assert s["avg_flow_time"] == pytest.approx(7.4, abs=tol)
    assert s["makespan"] == pytest.approx(7.4, abs=tol)
    assert s["total_simulation_time"] == pytest.approx(14.0, abs=tol)
    assert s["conveyor_1_stopped_time"] == pytest.approx(3.0, abs=tol)
    assert s["conveyor_2_stopped_time"] == pytest.approx(0.0, abs=tol)
    assert s["total_laser_travel_time"] == pytest.approx(2.4, abs=tol)
    assert s["laser_processing_time"] == pytest.approx(3.0, abs=tol)
    expected_dist = math.hypot(5.5, 5.5) + (12.0 - 5.5)          # diagonal, then pure Y
    assert s["total_laser_travel_distance"] == pytest.approx(expected_dist, abs=0.1)
    T = s["total_simulation_time"]
    assert s["laser_utilization"] == pytest.approx((2.4 + 3.0) / T, abs=0.01)
    assert s["throughput_per_min"] == pytest.approx(60.0 / T, rel=1e-6)
    total = s["total_laser_travel_time"] + s["laser_processing_time"] + s["laser_idle_time"]
    assert total == pytest.approx(T, abs=1e-6)


def test_summary_is_consistent(config):
    s = Simulation(config, record_trace=False).run().summary
    assert s["parts_completed"] <= s["parts_generated"]
    assert s["parts_exited"] <= s["parts_generated"]
    assert 0.0 <= s["laser_utilization"] <= 1.0 + 1e-9
    assert s["laser_utilization"] == pytest.approx(
        s["laser_processing_utilization"] + s["laser_movement_utilization"])
    assert s["total_conveyor_stop_time"] == pytest.approx(
        sum(s[f"conveyor_{i}_stopped_time"] for i in (1, 2, 3)))


# 10 ------------------------------------------------------------------------
def test_same_seed_same_experiment():
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=123",
                                 "simulation.arrival_horizon=120",
                                 "processing.distribution=uniform"])
    r1 = Simulation(cfg, record_trace=False).run()
    r2 = Simulation(cfg, record_trace=False).run()
    assert r1.summary == r2.summary
    assert r1.events.lines() == r2.events.lines()
    other = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=124",
                                   "simulation.arrival_horizon=120",
                                   "processing.distribution=uniform"])
    r3 = Simulation(other, record_trace=False).run()
    assert [p.arrival_time for p in r3.parts] != [p.arrival_time for p in r1.parts]


# policies / architecture ----------------------------------------------------
def test_all_policies_registered_and_complete_runs(config):
    assert {"fifo", "nearest_job", "round_robin", "weighted_score"} <= set(available_policies())
    for name in available_policies():
        s = Simulation(config, policy=get_policy(name), record_trace=False).run().summary
        assert s["parts_completed"] == s["parts_generated"], name   # block mode: nothing lost


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        get_policy("does_not_exist")


def test_round_robin_cycles_through_conveyors(explicit_config):
    arrivals = [{"time": 0.0, "conveyor": c} for c in (1, 2, 3)] * 2
    cfg = explicit_config(arrivals, simulation__min_part_spacing=0, policy__name="round_robin")
    result = Simulation(cfg, record_trace=False).run()
    order = [p.conveyor_id for p in sorted(result.parts, key=lambda p: p.start_processing_time)]
    assert order[:3] == [1, 2, 3]


def test_pass_mode_can_lose_parts():
    cfg = load_config(set_items=["simulation.unprocessed_at_zone_end=pass"])
    s = Simulation(cfg, record_trace=False).run().summary
    assert s["parts_exited"] == s["parts_generated"]
    assert s["parts_exited_unprocessed"] == s["parts_generated"] - s["parts_completed"]


def test_n_conveyors_from_config():
    cfg = load_config(set_items=["conveyors.count=5", "conveyors.y_positions=null",
                                 "conveyors.spacing=10",
                                 "arrivals.deterministic.intervals=20",
                                 "arrivals.deterministic.first_arrival=[0,2,4,6,8]",
                                 "simulation.arrival_horizon=100"])
    assert cfg.n_conveyors == 5
    assert [c.y for c in cfg.conveyors] == [0, 10, 20, 30, 40]
    assert cfg.laser.workspace.y_max == 40
    s = Simulation(cfg, record_trace=False).run().summary
    assert "conveyor_5_stopped_time" in s and s["parts_completed"] == s["parts_generated"]


def test_mismatched_per_conveyor_list_is_an_error():
    with pytest.raises(ValueError):
        load_config(set_items=["conveyors.count=4"])          # y_positions still has 3 values


def test_composite_objective_and_pareto():
    s = {"makespan": 100.0, "total_laser_travel_distance": 50.0,
         "total_conveyor_stop_time": 20.0, "avg_waiting_time": 5.0}
    w = {"makespan": 1.0, "total_laser_travel": 0.5, "total_conveyor_stop_time": 0.0,
         "avg_waiting_time": 2.0}
    assert composite_objective(s, w) == pytest.approx(100 + 25 + 10)
    assert pareto_front([(1, 5), (2, 2), (3, 3), (5, 1)]) == [0, 1, 3]


def test_trace_is_recorded(config):
    result = Simulation(config, record_trace=True).run()
    tr = result.trace
    assert tr["frames"] and tr["events"]
    assert len(tr["geometry"]["conveyors"]) == 3
    assert tr["frames"][-1]["K"]["completed"] == result.summary["parts_completed"]
    assert LaserState.IDLE.value == "IDLE"
