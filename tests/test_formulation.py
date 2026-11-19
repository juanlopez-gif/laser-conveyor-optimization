"""Simulated schedules must be feasible for the formal model (docs/formal_formulation.md)."""

from __future__ import annotations

import pytest

from optimization.schedule_check import check_schedule
from policies import available_policies, get_policy
from simulation.config import PROJECT_ROOT, load_config
from simulation.engine import Simulation


@pytest.mark.parametrize("policy", available_policies())
def test_default_instance_is_feasible_for_every_policy(policy):
    cfg = load_config()
    result = Simulation(cfg, policy=get_policy(policy), record_trace=False).run()
    assert check_schedule(result) == []


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stochastic_instances_are_feasible(seed):
    cfg = load_config(set_items=["arrivals.mode=stochastic", f"simulation.seed={seed}",
                                 "processing.distribution=uniform",
                                 "simulation.arrival_horizon=200"])
    assert check_schedule(Simulation(cfg, record_trace=False).run()) == []


def test_heavy_load_with_blocking_is_feasible():
    cfg = load_config(PROJECT_ROOT / "config" / "examples" / "heavy_load.yaml")
    result = Simulation(cfg, record_trace=False).run()
    assert result.summary["conveyor_1_stopped_blocked_time"] > 0     # blocking really occurs
    assert check_schedule(result) == []


def test_all_conveyors_stop_rule_is_feasible():
    cfg = load_config(set_items=["simulation.conveyor_stop_policy=all_conveyors",
                                 "simulation.arrival_horizon=150"])
    assert check_schedule(Simulation(cfg, record_trace=False).run()) == []


def test_checker_detects_a_violation():
    cfg = load_config(set_items=["simulation.arrival_horizon=60"])
    result = Simulation(cfg, record_trace=False).run()
    part = next(p for p in result.parts if p.processing_x is not None)
    part.processing_x = cfg.geometry.active_end + 5.0           # outside the laser zone
    errors = check_schedule(result)
    assert any(part.id in e for e in errors)
