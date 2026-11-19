"""Tests 1-3, 5-8: conveyor motion, stopping rules, laser kinematics and limits."""

from __future__ import annotations

import math

import pytest

from simulation.config import ConveyorSpec, Geometry
from simulation.conveyor import STOP_PROCESSING, Conveyor
from simulation.engine import Simulation
from simulation.events import EventKind
from simulation.laser import Laser, LaserState
from simulation.motion import find_intercept, travel_time
from simulation.part import Part, PartStatus

GEOM = Geometry(10.0, 35.0, 10.0)


def _conveyor_with_part(x: float = 0.0, speed: float = 5.0) -> tuple[Conveyor, Part]:
    conv = Conveyor(ConveyorSpec(1, 0.0, speed), GEOM)
    part = Part("P001", 1, 0.0, 3.0, 20.0, x=x, status=PartStatus.MOVING)
    conv.parts.append(part)
    return conv, part


# 1 -------------------------------------------------------------------------
def test_part_moves_with_running_conveyor():
    conv, part = _conveyor_with_part()
    moved = conv.advance(1.0, block_unprocessed=True)
    assert moved == pytest.approx(5.0)
    assert part.x == pytest.approx(5.0)


# 2 -------------------------------------------------------------------------
def test_stopped_conveyor_freezes_its_parts():
    conv, part = _conveyor_with_part(x=20.0)
    conv.stop(STOP_PROCESSING)
    assert not conv.is_running
    assert conv.advance(1.0, block_unprocessed=True) == 0.0
    assert part.x == pytest.approx(20.0)


def test_block_mode_stops_unprocessed_part_at_zone_end():
    conv, part = _conveyor_with_part(x=44.0)
    conv.advance(1.0, block_unprocessed=True)          # would go to 49
    assert part.x == pytest.approx(GEOM.active_end)
    assert conv.blocking_part(True) is part


# 3 -------------------------------------------------------------------------
def test_only_the_processing_conveyor_stops(explicit_config):
    cfg = explicit_config([{"time": 0.0, "conveyor": 1}, {"time": 0.0, "conveyor": 2},
                           {"time": 0.0, "conveyor": 3}],
                          simulation__min_part_spacing=0)
    sim = Simulation(cfg, record_trace=False)
    while sim.laser.state != LaserState.PROCESSING:
        sim.step()
    busy = sim.laser.current_part.conveyor_id
    others = [c for c in sim.conveyors if c.id != busy]
    before = {c.id: [p.x for p in c.parts] for c in sim.conveyors}
    for _ in range(20):                                 # 0.2 s, processing lasts 3 s
        sim.step()
    assert sim.laser.state == LaserState.PROCESSING
    assert not sim.conveyors[busy - 1].is_running
    assert [p.x for p in sim.conveyors[busy - 1].parts] == before[busy]
    for c in others:
        assert c.is_running
        assert all(after > b for after, b in zip([p.x for p in c.parts], before[c.id]))


def test_all_conveyors_stop_policy(explicit_config):
    cfg = explicit_config([{"time": 0.0, "conveyor": 1}, {"time": 0.0, "conveyor": 2}],
                          simulation__conveyor_stop_policy="all_conveyors")
    sim = Simulation(cfg, record_trace=False)
    while sim.laser.state != LaserState.PROCESSING:
        sim.step()
    assert all(not c.is_running for c in sim.conveyors)


# 5 -------------------------------------------------------------------------
def test_laser_respects_workspace(config):
    laser = Laser.from_spec(config.laser)
    ws = config.laser.workspace
    for _ in range(2000):
        laser.move_towards(1000.0, -1000.0, 0.01)       # far outside
    assert ws.contains(laser.x, laser.y)
    assert laser.x == pytest.approx(ws.x_max) and laser.y == pytest.approx(ws.y_min)
    with pytest.raises(ValueError):
        laser.set_position(ws.x_max + 1.0, ws.y_min)


def test_laser_stays_in_workspace_during_a_run(config):
    sim = Simulation(config, record_trace=False)
    ws = config.laser.workspace
    for _ in range(20000):
        sim.step()
        assert ws.contains(sim.laser.x, sim.laser.y)


# 6 -------------------------------------------------------------------------
def test_travel_time_models():
    assert travel_time(0, 0, 10, 5, 5, 5, "simultaneous") == pytest.approx(2.0)
    assert travel_time(0, 0, 10, 5, 5, 5, "sequential") == pytest.approx(3.0)
    assert travel_time(0, 0, 3, 24, 5, 10, "simultaneous") == pytest.approx(2.4)
    with pytest.raises(ValueError):
        travel_time(0, 0, 1, 1, 5, 5, "teleport")


@pytest.mark.parametrize("model", ["simultaneous", "sequential"])
def test_laser_motion_matches_travel_time(config, model):
    from dataclasses import replace
    spec = replace(config.laser, motion_model=model)
    laser = Laser.from_spec(spec)
    laser.set_position(10.0, 0.0)
    expected = travel_time(10.0, 0.0, 40.0, 24.0, spec.speed_x, spec.speed_y, model)
    dt, steps = 0.01, 0
    while not (abs(laser.x - 40.0) < 1e-9 and abs(laser.y - 24.0) < 1e-9):
        laser.move_towards(40.0, 24.0, dt)
        steps += 1
    assert steps * dt == pytest.approx(expected, abs=dt)


def test_intercept_of_moving_part():
    # laser at (27.5, 12), part at (10, 0) moving at 5 cm/s: meets at x=22 after 2.4 s
    icp = find_intercept(27.5, 12.0, 10.0, 0.0, 5.0, 45.0, True, 5.0, 5.0, "simultaneous")
    assert icp.time == pytest.approx(2.4, abs=1e-6)
    assert icp.x == pytest.approx(22.0, abs=1e-5)
    # part ahead and as fast as the laser: only reachable because it stops at the zone end
    icp = find_intercept(10.0, 0.0, 30.0, 0.0, 5.0, 45.0, True, 5.0, 5.0, "simultaneous")
    assert icp.x == pytest.approx(45.0) and icp.time == pytest.approx(7.0)
    assert find_intercept(10.0, 0.0, 30.0, 0.0, 5.0, 45.0, False, 5.0, 5.0, "simultaneous") is None


# 7 -------------------------------------------------------------------------
def test_no_processing_before_reaching_laser_zone(config):
    result = Simulation(config, record_trace=False).run()
    started = [p for p in result.parts if p.start_processing_time is not None]
    assert started
    for p in started:
        assert p.ready_time is not None
        assert p.arrival_time <= p.ready_time <= p.start_processing_time + 1e-9
    # positions at the start of processing (logged) are inside the zone
    g = config.geometry
    for e in result.events.of_kind(EventKind.LASER_ARRIVED):
        x = float(e.message.split("(")[1].split(",")[0])
        assert g.active_start - 1e-6 <= x <= g.active_end + 1e-6


def test_part_in_entry_zone_is_not_a_candidate(explicit_config):
    cfg = explicit_config([{"time": 0.0, "conveyor": 1}])
    sim = Simulation(cfg, record_trace=False)
    for _ in range(150):                                # 1.5 s: part at x = 7.5 < 10
        sim.step()
        assert sim.laser.state == LaserState.IDLE
    assert sim.parts[0].status == PartStatus.MOVING


# 8 -------------------------------------------------------------------------
def test_conveyor_restarts_after_processing(explicit_config):
    cfg = explicit_config([{"time": 0.0, "conveyor": 2}])
    result = Simulation(cfg, record_trace=False).run()
    part = result.parts[0]
    stops = [e for e in result.events.of_kind(EventKind.CONVEYOR_STOP) if e.conveyor_id == 2]
    starts = [e for e in result.events.of_kind(EventKind.CONVEYOR_START) if e.conveyor_id == 2]
    assert len(stops) == 1 and len(starts) == 1
    assert stops[0].time == pytest.approx(part.start_processing_time)
    assert starts[0].time == pytest.approx(part.completion_time)
    assert part.status == PartStatus.EXITED and part.exit_time > part.completion_time
    assert math.isclose(part.completion_time - part.start_processing_time, 3.0, abs_tol=0.011)
