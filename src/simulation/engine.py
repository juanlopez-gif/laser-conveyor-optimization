"""Simulation engine (no GUI dependency).

Time advance: fixed-step hybrid simulation
------------------------------------------
Positions of parts and of the laser change *continuously* and the conveyors
start/stop depending on what the laser does, so the state is advanced with a
small fixed step ``dt`` (default 0.01 s). Inside each step the discrete
logic (arrivals, dispatching decisions, start/end of processing, stops) is
resolved in a fixed, documented order, so a run is fully deterministic.
Event times are therefore exact up to ``dt``. See docs/assumptions.md for the
rationale of not using SimPy.

Order of operations in one step [t, t + dt)
-------------------------------------------
1. finish processing if its end time has been reached (conveyor released)
2. generate arrivals (parts join the upstream queue: WAITING)
3. recompute blocking stops, log conveyor STOP/START transitions
4. admit WAITING parts onto running belts if the entry is free
5. if the laser is IDLE, ask the policy for the next part (ASSIGN). Candidates are
   the READY parts plus, with pre-positioning, the parts still in the entry zone of
   a running belt: the laser then goes to the meeting point at x >= a and waits
   there (model P instead of P^sim, docs/formal_formulation.md §1.5).
   ``always``: entry-zone parts are always candidates. ``when_idle``: only when no
   READY part exists, and the pre-positioning is revoked (UNASSIGN) as soon as
   another part becomes READY, so the policy decides again.
6. drive the laser towards the meeting point; start processing on contact
   (the stop policy stops the corresponding conveyor(s))
7. accumulate metrics / record a trace frame
8. move the running belts by ``speed * dt``
9. update part statuses (READY when entering the laser zone, EXITED at the end)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .arrivals import generate_arrivals
from .config import Config
from .conveyor import STOP_BLOCKED, STOP_PROCESSING, Conveyor
from .events import EventKind, EventLog
from .laser import Laser, LaserState
from .metrics import MetricsCollector, summarize
from .motion import Intercept, find_intercept
from .part import Part, PartStatus
from .stop_policy import get_stop_policy
from .trace import TraceRecorder

EPS = 1e-7


@dataclass
class SimulationResult:
    """Everything produced by one run."""

    config: Config
    policy_name: str
    summary: dict
    parts: list[Part]
    events: EventLog
    queue_series: list[tuple[float, list[int]]]
    trace: dict | None


class Simulation:
    """One simulation run. Create, then call :meth:`run` (or :meth:`step`)."""

    def __init__(self, config: Config, policy=None, record_trace: bool = True) -> None:
        from policies import get_policy   # local import keeps the layers decoupled

        self.cfg = config
        self.dt = config.simulation.dt
        self.geometry = config.geometry
        self.policy = policy if policy is not None else get_policy(
            config.policy_name, config.policy_params)
        self.policy.reset()
        self.stop_policy = get_stop_policy(config.simulation.conveyor_stop_policy)
        self.block = config.simulation.unprocessed_at_zone_end == "block"
        self.prepositioning = config.laser.prepositioning      # none | when_idle | always
        self.observation = config.simulation.observation        # zone | belts | full (0.7.3)

        self.conveyors = [Conveyor(spec, config.geometry) for spec in config.conveyors]
        self.laser = Laser.from_spec(config.laser)
        self.pending_arrivals = deque(generate_arrivals(config))
        self.parts: list[Part] = []
        self.log = EventLog()
        self.metrics = MetricsCollector(len(self.conveyors))
        self.trace = TraceRecorder(config) if record_trace else None

        self.step_index = 0
        self.trace_every = max(1, round(config.simulation.trace_interval / self.dt))
        self._was_running = [True] * len(self.conveyors)
        self.finished = False
        self.n_completed = self.n_exited = self.n_missed = 0
        self.n_block_events = 0          # parts with S_j > B_j (counted at S_j)
        self.wip_area = 0.0              # integral of the number of unfinished parts [part-s]

    # ------------------------------------------------------------------ utils
    @property
    def time(self) -> float:
        return self.step_index * self.dt

    def conveyor_of(self, part: Part) -> Conveyor:
        return self.conveyors[part.conveyor_id - 1]

    def intercept(self, part: Part) -> Intercept | None:
        """Meeting point between the laser and ``part`` given the current state."""
        conv = self.conveyor_of(part)
        spec = self.cfg.laser
        return find_intercept(self.laser.x, self.laser.y, part.x, part.y,
                              conv.current_speed, self.geometry.active_end, self.block,
                              spec.speed_x, spec.speed_y, spec.motion_model,
                              x_min=self.geometry.active_start)

    def _ready_candidates_exist(self) -> bool:
        return any(p.status == PartStatus.READY and not p.assigned
                   for c in self.conveyors for p in c.parts)

    def _revoke_prepositioning(self, t: float) -> None:
        """``when_idle``: drop a pre-positioning as soon as some part is READY."""
        part = self.laser.current_part
        if (self.prepositioning != "when_idle" or part is None
                or part.status != PartStatus.MOVING or not self._ready_candidates_exist()):
            return
        self.log.log(t, EventKind.UNASSIGN,
                     f"Pre-positioning to {part.id} revoked (a part is READY)",
                     part.id, part.conveyor_id)
        self.laser.clear()

    # ------------------------------------------------------------------- run
    def run(self) -> SimulationResult:
        while not self.finished:
            self.step()
        return self.result()

    def step(self) -> None:
        """Advance the simulation by one time step."""
        if self.finished:
            return
        self.begin_step()
        self.end_step()

    # Split of one step around the dispatching decision (step 5). ``step()`` is
    # exactly ``begin_step(); end_step()``. External controllers (reinforcement
    # learning, src/rl/env.py) call ``begin_step()``, read ``decision_candidates()``
    # and pass their choice to ``end_step(choice)``.
    def begin_step(self) -> None:
        """Steps 1-4 of the current time step (everything before dispatching)."""
        t = self.time
        if self.step_index == 0:
            self.log.log(t, EventKind.SIM_START,
                         f"Simulation started (policy={self.policy.name}, "
                         f"{len(self.conveyors)} conveyors, {len(self.pending_arrivals)} arrivals planned)")
        self._finish_processing(t)                      # 1
        self._generate_arrivals(t)                      # 2
        self._update_blocking(t)                        # 3
        self._log_conveyor_transitions(t)
        self._admit_parts(t)                            # 4
        if self.laser.state in (LaserState.MOVING, LaserState.WAITING_FOR_PART):
            self._revoke_prepositioning(t)

    def decision_candidates(self) -> list[Part]:
        """Parts the laser may be assigned to now (empty unless the laser is IDLE).
        Valid between ``begin_step()`` and ``end_step()``."""
        if self.laser.state != LaserState.IDLE:
            return []
        return self._candidates(self._intercept_memo())

    def end_step(self, choice: Part | None = None) -> None:
        """Steps 5-9. ``choice`` (a part of ``decision_candidates()``) replaces the
        policy's decision; ``None`` asks the policy as usual."""
        t, dt = self.time, self.dt
        if self.laser.state == LaserState.IDLE:         # 5
            self._dispatch(t, choice)
        if self.laser.state in (LaserState.MOVING, LaserState.WAITING_FOR_PART):
            self._drive_laser(t)                        # 6
        self._log_conveyor_transitions(t)
        # work-in-progress integral: sum_j F_j = integral of (arrived, not completed)
        self.wip_area += (len(self.parts) - self.n_completed - self.n_missed) * dt

        self.metrics.record_step(self.laser, self.conveyors, dt)   # 7
        if self.step_index % self.trace_every == 0:
            self._sample(t)

        moved = [c.advance(dt, self.block) > 0 for c in self.conveyors]   # 8
        self._update_statuses(t + dt, moved)            # 9

        self.step_index += 1
        if self._should_stop(self.time):
            self._finalize()

    # ---------------------------------------------------------------- phases
    def _finish_processing(self, t: float) -> None:
        laser = self.laser
        if laser.state != LaserState.PROCESSING or t < laser.processing_end - EPS:
            return
        part = laser.finish_processing()
        part.completion_displacement = self.conveyor_of(part).displacement
        part.status = PartStatus.COMPLETED
        part.completion_time = t
        self.n_completed += 1
        self.log.log(t, EventKind.PROCESS_END, f"Processing {part.id} completed",
                     part.id, part.conveyor_id)
        for c in self.conveyors:
            c.release(STOP_PROCESSING)

    def _generate_arrivals(self, t: float) -> None:
        while self.pending_arrivals and self.pending_arrivals[0].time <= t + EPS:
            rec = self.pending_arrivals.popleft()
            conv = self.conveyors[rec.conveyor_index]
            part = Part(id=f"P{len(self.parts) + 1:03d}", conveyor_id=conv.id,
                        arrival_time=rec.time, processing_time=rec.processing_time,
                        due_date=rec.time + self.cfg.simulation.due_date_allowance,
                        x=0.0, y=conv.y)
            part.arrival_displacement = conv.displacement
            self.parts.append(part)
            conv.buffer.append(part)
            self.log.log(t, EventKind.ARRIVAL, f"Part {part.id} arrived at Conveyor {conv.id}",
                         part.id, conv.id)

    def _update_blocking(self, t: float) -> None:
        for c in self.conveyors:
            bp = c.blocking_part(self.block)
            if bp is not None:
                c.stop(STOP_BLOCKED)
                if bp.block_time is None:          # B_j: first time at the zone end
                    bp.block_time = t
            else:
                c.release(STOP_BLOCKED)

    def _admit_parts(self, t: float) -> None:
        spacing = self.cfg.simulation.min_part_spacing
        for c in self.conveyors:
            while c.can_admit(spacing):
                p = c.admit(t)
                self.log.log(t, EventKind.ENTER, f"Part {p.id} entered Conveyor {c.id}", p.id, c.id)
                if spacing > 0:
                    break

    def _intercept_memo(self):
        memo: dict[str, Intercept | None] = {}

        def icp(part: Part) -> Intercept | None:
            if part.id not in memo:
                memo[part.id] = self.intercept(part)
            return memo[part.id]
        return icp

    def _candidates(self, icp) -> list[Part]:
        mode = self.prepositioning
        upcoming = mode == "always" or (mode == "when_idle" and not self._ready_candidates_exist())
        return [p for c in self.conveyors for p in c.parts
                if not p.assigned and (p.status == PartStatus.READY or (
                    upcoming and p.status == PartStatus.MOVING)) and icp(p) is not None]

    def _dispatch(self, t: float, forced: Part | None = None) -> None:
        icp = self._intercept_memo()
        candidates = self._candidates(icp)
        if not candidates:
            return
        if forced is not None:
            if forced not in candidates:
                raise ValueError(f"{forced.id} is not a dispatch candidate at t={t:.2f}")
            choice = forced
        else:
            from policies.base_policy import DecisionContext
            ctx = DecisionContext(t, self.laser, self.conveyors, self.geometry, icp,
                                  simulation=self)
            choice = self.policy.select(candidates, ctx)
        if choice is None:
            return
        plan = icp(choice)
        self.laser.assign(choice, plan.x, plan.y, self.conveyor_of(choice).current_speed)
        self.log.log(t, EventKind.ASSIGN,
                     f"Laser assigned to {choice.id} (Conveyor {choice.conveyor_id}, "
                     f"meet at x={plan.x:.1f} in {plan.time:.2f} s)", choice.id, choice.conveyor_id)

    def _drive_laser(self, t: float) -> None:
        laser, part = self.laser, self.laser.current_part
        conv = self.conveyor_of(part)
        tol = conv.nominal_speed * self.dt + EPS

        reachable = part.status == PartStatus.READY or (
            self.prepositioning != "none" and part.status == PartStatus.MOVING)
        if not reachable or part.x > self.geometry.active_end + EPS:
            self.log.log(t, EventKind.UNASSIGN, f"Assignment to {part.id} dropped (unreachable)",
                         part.id, part.conveyor_id)
            laser.clear()
            return

        def in_contact() -> bool:
            return laser.at_target and laser.target_x - EPS <= part.x <= laser.target_x + tol

        if not in_contact():
            if conv.current_speed != laser.planned_speed or part.x > laser.target_x + tol:
                plan = self.intercept(part)          # the part changed behaviour: re-plan
                if plan is None:
                    self.log.log(t, EventKind.UNASSIGN,
                                 f"Assignment to {part.id} dropped (unreachable)",
                                 part.id, part.conveyor_id)
                    laser.clear()
                    return
                laser.set_target(plan.x, plan.y, conv.current_speed)
            if not laser.at_target:
                laser.move_towards(laser.target_x, laser.target_y, self.dt)

        if in_contact():
            self._start_processing(part, t)
        else:
            laser.state = LaserState.MOVING if not laser.at_target else LaserState.WAITING_FOR_PART

    def _start_processing(self, part: Part, t: float) -> None:
        self.laser.start_processing(part, t)
        part.status = PartStatus.PROCESSING
        part.start_processing_time = t
        part.processing_x = part.x
        conv = self.conveyor_of(part)
        if part.block_time is not None and t - part.block_time > 1e-9:
            self.n_block_events += 1
        part.start_displacement = conv.displacement
        if conv.blocking_part(self.block) is None:
            conv.release(STOP_BLOCKED)    # the block ends at S_j (belt now held by processing)
        self.log.log(t, EventKind.LASER_ARRIVED,
                     f"Laser reached {part.id} at ({part.x:.1f}, {part.y:.1f})",
                     part.id, part.conveyor_id)
        for c in self.stop_policy.conveyors_to_stop(part, self.conveyors):
            c.stop(STOP_PROCESSING)
        self.log.log(t, EventKind.PROCESS_START,
                     f"Processing {part.id} started ({part.processing_time:.2f} s)",
                     part.id, part.conveyor_id)

    def _log_conveyor_transitions(self, t: float) -> None:
        for i, c in enumerate(self.conveyors):
            if c.is_running == self._was_running[i]:
                continue
            if c.is_running:
                self.log.log(t, EventKind.CONVEYOR_START, f"Conveyor {c.id} STARTED", conveyor_id=c.id)
            else:
                reasons = []
                if STOP_PROCESSING in c.stop_reasons and self.laser.current_part:
                    reasons.append(f"laser processing {self.laser.current_part.id}")
                if STOP_BLOCKED in c.stop_reasons:
                    bp = c.blocking_part(self.block)
                    reasons.append(f"blocked: {bp.id if bp else '?'} unprocessed at end of laser zone")
                self.log.log(t, EventKind.CONVEYOR_STOP,
                             f"Conveyor {c.id} STOPPED ({'; '.join(reasons)})", conveyor_id=c.id)
            self._was_running[i] = c.is_running

    def _update_statuses(self, t: float, moved: list[bool]) -> None:
        start = self.geometry.active_start - EPS
        end = self.geometry.total_length - EPS
        for c, has_moved in zip(self.conveyors, moved):
            if not has_moved:            # positions unchanged -> statuses unchanged
                continue
            for p in list(c.parts):
                if p.status == PartStatus.MOVING and p.x >= start:
                    p.status = PartStatus.READY
                    p.ready_time = t
                    self.log.log(t, EventKind.READY,
                                 f"Part {p.id} entered laser zone of Conveyor {c.id}", p.id, c.id)
                if p.x >= end:
                    c.parts.remove(p)
                    p.status = PartStatus.EXITED
                    p.exit_time = t
                    self.n_exited += 1
                    if p.is_processed:
                        msg = f"Part {p.id} exited Conveyor {c.id}"
                    else:
                        self.n_missed += 1
                        msg = f"Part {p.id} exited Conveyor {c.id} UNPROCESSED"
                    self.log.log(t, EventKind.EXIT, msg, p.id, c.id)

    def _sample(self, t: float) -> None:
        self.metrics.sample_queues(t, self.conveyors)
        if self.trace is None:
            return
        lt = self.metrics.laser_time
        elapsed = max(t + self.dt, self.dt)
        kpis = {
            "completed": self.n_completed, "exited": self.n_exited,
            "generated": len(self.parts), "missed": self.n_missed,
            "util": round((lt["MOVING"] + lt["PROCESSING"]) / elapsed, 4),
            "thr": round(60.0 * self.n_completed / elapsed, 3),
            "queues": [c.pending_count() for c in self.conveyors],
            "dist": round(self.laser.distance, 2),
            "blocks": sum(1 for p in self.parts if p.caused_block),
            "blocked_s": round(sum(self.metrics.stopped_blocked), 2),
        }
        self.trace.record(t, self.laser, self.conveyors, kpis)

    def _should_stop(self, t: float) -> bool:
        s = self.cfg.simulation
        if t >= s.max_time - EPS:
            return True
        if t < s.arrival_horizon - EPS or self.pending_arrivals:
            return False
        if not s.drain:
            return True
        empty = all(not c.parts and not c.buffer for c in self.conveyors)
        return empty and self.laser.state == LaserState.IDLE

    def _finalize(self) -> None:
        t = self.time
        self._sample(t)
        self.log.log(t, EventKind.SIM_END,
                     f"Simulation finished: {self.n_completed} completed, {self.n_exited} exited, "
                     f"{len(self.parts) - self.n_exited} still in system")
        self.finished = True

    def observable(self, part) -> bool:
        """True if a policy may observe ``part`` (information model, 0.7.3;
        simulation/observation.py). The physics never depends on it."""
        from .observation import visible_status
        return visible_status(part.status.value if hasattr(part.status, "value")
                              else str(part.status), self.observation)

    # ---------------------------------------------------------------- output
    def result(self) -> SimulationResult:
        from optimization.objective_functions import composite_objective

        summary = summarize(self.parts, self.metrics, self.laser, self.time)
        summary["policy"] = self.policy.name
        summary["seed"] = self.cfg.simulation.seed
        summary["blocking_penalty"] = (
            self.cfg.blocking_penalty_per_event * summary["blocking_events"]
            + self.cfg.blocking_penalty_per_second * summary["total_blocked_time"])
        summary["objective"] = composite_objective(summary, self.cfg.objective_weights)
        trace = None
        if self.trace is not None:
            events = [{"t": e.time, "k": e.kind.value, "m": e.message} for e in self.log.events]
            trace = self.trace.to_dict(events, summary, self.policy.name)
        return SimulationResult(self.cfg, self.policy.name, summary, self.parts, self.log,
                                self.metrics.queue_series, trace)
