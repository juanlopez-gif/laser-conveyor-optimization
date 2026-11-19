"""Check a simulated schedule against the formal model (docs/formal_formulation.md).

Given a finished :class:`~simulation.engine.SimulationResult`, verify that the
schedule satisfies the identities and constraints of the continuous-time model
(P), up to tolerances of the order of the simulator step ``dt``:

* §1.1  entry coordinates   E_j = max(D_k(r_j), E_pred(j) + delta)
* §1.2  processing position X_j = D_k(S_j) - E_j
* (C1)  a <= X_j <= b                                  (includes blocking, C4)
* (C2)  S_next >= C_prev + tau(P_prev, P_next), first part from P_0
* (C3)  the belt does not move while its part is processed: D_k(C_j) = D_k(S_j)
* belt kinematics: D_k(S_j) - D_k(S_i) <= v_k * (S_j - S_i - stopped-for-processing time)

The function returns a list of human-readable violations (empty = consistent).
It is used by tests/test_formulation.py and can be applied to any future solver
output that is replayed through the simulator.
"""

from __future__ import annotations

from simulation.engine import SimulationResult
from simulation.motion import travel_time


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def check_schedule(result: SimulationResult, time_tol: float | None = None,
                   dist_tol: float | None = None) -> list[str]:
    cfg = result.config
    dt = cfg.simulation.dt
    vmax = max(c.speed for c in cfg.conveyors)
    t_tol = time_tol if time_tol is not None else 3 * dt + 1e-9
    d_tol = dist_tol if dist_tol is not None else 3 * vmax * dt + 1e-9
    g, laser = cfg.geometry, cfg.laser
    speed = {c.id: c.speed for c in cfg.conveyors}
    y_of = {c.id: c.y for c in cfg.conveyors}
    stop_all = cfg.simulation.conveyor_stop_policy == "all_conveyors"
    errors: list[str] = []

    done = [p for p in result.parts if p.start_processing_time is not None]
    if any(p.completion_time is None for p in done):
        errors.append("a part started processing but never completed")
        return errors

    # §1.1 entry coordinates (FIFO per conveyor, minimum spacing)
    for cid in speed:
        on_belt = sorted((p for p in result.parts if p.conveyor_id == cid
                          and p.entry_coordinate is not None), key=lambda p: p.number)
        prev = None
        for p in on_belt:
            expected = p.arrival_displacement
            if prev is not None:
                expected = max(expected, prev.entry_coordinate + cfg.simulation.min_part_spacing)
            if not (expected - 1e-9 <= p.entry_coordinate <= expected + d_tol):
                errors.append(f"{p.id}: E_j={p.entry_coordinate:.3f}, model gives {expected:.3f}")
            prev = p

    for p in done:
        # §1.2 X_j = D_k(S_j) - E_j
        x_model = p.start_displacement - p.entry_coordinate
        if abs(x_model - p.processing_x) > 1e-6:
            errors.append(f"{p.id}: X_j={p.processing_x:.4f} but D(S)-E={x_model:.4f}")
        # (C1) window
        if not (g.active_start - d_tol <= p.processing_x <= g.active_end + 1e-9):
            errors.append(f"{p.id}: X_j={p.processing_x:.3f} outside [{g.active_start}, {g.active_end}]")
        # (C3) belt frozen during processing
        if abs(p.completion_displacement - p.start_displacement) > 1e-9:
            errors.append(f"{p.id}: belt moved {p.completion_displacement - p.start_displacement:.4f} "
                          "cm during processing")

    # (C2) sequence with travel / interception
    seq = sorted(done, key=lambda p: p.start_processing_time)
    prev_point, prev_c = (laser.home_x, laser.home_y), 0.0
    for p in seq:
        tau = travel_time(prev_point[0], prev_point[1], p.processing_x, y_of[p.conveyor_id],
                          laser.speed_x, laser.speed_y, laser.motion_model)
        if p.start_processing_time < prev_c + tau - t_tol:
            errors.append(f"{p.id}: S_j={p.start_processing_time:.3f} < C_prev + tau = "
                          f"{prev_c + tau:.3f}")
        prev_point, prev_c = (p.processing_x, y_of[p.conveyor_id]), p.completion_time

    # Belt kinematics between consecutive starts on the same conveyor
    for cid, v in speed.items():
        own = sorted((p for p in done if p.conveyor_id == cid), key=lambda p: p.start_processing_time)
        stoppers = done if stop_all else own
        for i, j in zip(own, own[1:]):
            si, sj = i.start_processing_time, j.start_processing_time
            stopped = sum(_overlap(si, sj, q.start_processing_time, q.completion_time)
                          for q in stoppers)
            moved = j.start_displacement - i.start_displacement
            if moved < -1e-9 or moved > v * (sj - si - stopped) + d_tol:
                errors.append(f"conveyor {cid}: belt moved {moved:.3f} cm between {i.id} and "
                              f"{j.id}, max allowed {v * (sj - si - stopped):.3f}")
    return errors
