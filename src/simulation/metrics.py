"""Performance metrics.

:class:`MetricsCollector` integrates time-based quantities step by step
(laser state times, conveyor stopped times, queue lengths); :func:`summarize`
turns the collector + the list of parts into a flat dictionary of KPIs that is
saved to JSON/CSV and consumed by the objective functions.

Definitions (also in docs/mathematical_model.md):

* waiting time   = start_processing - arrival
* flow time      = completion - arrival
* system time    = exit - arrival
* makespan       = completion time of the last processed part
* throughput     = completed parts / total simulation time (reported per minute)
* lateness       = completion - due_date,  due_date = arrival + allowance
* queue length   = parts of a conveyor that arrived and have not started processing
* laser utilization = (moving + processing) / total time
* blocking events N_B = parts that reached the end of the laser zone unprocessed and
  held their conveyor stopped for a positive time (S_j > B_j)
* blocked time   T_B = total conveyor time stopped because of blocking [conveyor-s]
                     = sum_j (S_j - B_j)  (exact identity, checked by the tests)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Sequence

from .conveyor import STOP_BLOCKED, STOP_PROCESSING, Conveyor
from .laser import Laser, LaserState
from .part import Part


@dataclass
class MetricsCollector:
    n_conveyors: int
    laser_time: dict[str, float] = field(default_factory=lambda: {s.value: 0.0 for s in LaserState})
    running_time: list[float] = field(init=False)
    stopped_time: list[float] = field(init=False)
    stopped_processing: list[float] = field(init=False)
    stopped_blocked: list[float] = field(init=False)
    queue_area: list[float] = field(init=False)
    queue_max: list[int] = field(init=False)
    queue_series: list[tuple[float, list[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.n_conveyors
        self.running_time = [0.0] * n
        self.stopped_time = [0.0] * n
        self.stopped_processing = [0.0] * n
        self.stopped_blocked = [0.0] * n
        self.queue_area = [0.0] * n
        self.queue_max = [0] * n

    def record_step(self, laser: Laser, conveyors: Sequence[Conveyor], dt: float) -> None:
        """Accumulate the state held during one time step of length ``dt``."""
        self.laser_time[laser.state.value] += dt
        for i, c in enumerate(conveyors):
            if c.is_running:
                self.running_time[i] += dt
            else:
                self.stopped_time[i] += dt
                if STOP_PROCESSING in c.stop_reasons:
                    self.stopped_processing[i] += dt
                if STOP_BLOCKED in c.stop_reasons:
                    self.stopped_blocked[i] += dt
            q = c.pending_count()
            self.queue_area[i] += q * dt
            if q > self.queue_max[i]:
                self.queue_max[i] = q

    def sample_queues(self, time: float, conveyors: Sequence[Conveyor]) -> None:
        self.queue_series.append((round(time, 6), [c.pending_count() for c in conveyors]))


def _avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize(parts: Sequence[Part], collector: MetricsCollector, laser: Laser,
              total_time: float) -> dict:
    """Return a flat dict with every KPI of the run."""
    T = max(total_time, 1e-12)
    completed = [p for p in parts if p.is_processed]
    exited = [p for p in parts if p.exit_time is not None]
    waits = [p.waiting_time for p in parts if p.waiting_time is not None]
    flows = [p.flow_time for p in completed]
    systems = [p.exit_time - p.arrival_time for p in exited]
    lateness = [p.lateness for p in completed]
    lt = collector.laser_time
    moving, processing = lt["MOVING"], lt["PROCESSING"]
    idle = lt["IDLE"] + lt["WAITING_FOR_PART"]

    s: dict = {
        "total_simulation_time": total_time,
        "parts_generated": len(parts),
        "parts_completed": len(completed),
        "parts_exited": len(exited),
        "parts_exited_unprocessed": sum(1 for p in exited if not p.is_processed),
        "parts_in_system_at_end": len(parts) - len(exited),
        "throughput_per_min": 60.0 * len(completed) / T,
        "avg_waiting_time": _avg(waits),
        "max_waiting_time": max(waits) if waits else None,
        "avg_flow_time": _avg(flows),
        "max_flow_time": max(flows) if flows else None,
        "avg_system_time": _avg(systems),
        "makespan": max((p.completion_time for p in completed), default=0.0),
        "avg_lateness": _avg(lateness),
        "max_lateness": max(lateness) if lateness else None,
        "tardy_parts": sum(1 for v in lateness if v > 0),
        "total_laser_travel_distance": laser.distance,
        "laser_travel_distance_x": laser.distance_x,
        "laser_travel_distance_y": laser.distance_y,
        "total_laser_travel_time": moving,
        "laser_processing_time": processing,
        "laser_waiting_for_part_time": lt["WAITING_FOR_PART"],
        "laser_idle_time": idle,
        "laser_utilization": (moving + processing) / T,
        "laser_processing_utilization": processing / T,
        "laser_movement_utilization": moving / T,
        "total_conveyor_stop_time": sum(collector.stopped_time),
        "blocking_events": sum(1 for p in parts if p.caused_block and (
            p.blocked_duration is None or p.blocked_duration > 1e-9)),
        "total_blocked_time": sum(collector.stopped_blocked),
    }
    for i in range(collector.n_conveyors):
        k = f"conveyor_{i + 1}"
        s[f"{k}_stopped_time"] = collector.stopped_time[i]
        s[f"{k}_stopped_processing_time"] = collector.stopped_processing[i]
        s[f"{k}_stopped_blocked_time"] = collector.stopped_blocked[i]
        s[f"{k}_running_pct"] = 100.0 * collector.running_time[i] / T
        s[f"{k}_avg_queue_length"] = collector.queue_area[i] / T
        s[f"{k}_max_queue_length"] = collector.queue_max[i]
    return s
