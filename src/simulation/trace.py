"""Compact state trace for the (decoupled) visualization.

The engine pushes a snapshot every ``trace_interval`` seconds. The browser
only *replays* this trace: changing the visual speed cannot change any
result, because the results were computed before the page was opened.

Frame layout (kept short so long runs stay small)::

    {"t": 12.3,
     "L": [x, y, laser_state, target_part_number | null],
     "C": [[running 0/1, stop_reason], ...],            # one per conveyor
     "P": [[part_number, conveyor_id, x, status_code], ...],
     "K": {completed, exited, generated, missed, util, queues: [...]}}
"""

from __future__ import annotations

from typing import Sequence

from .config import Config
from .conveyor import Conveyor
from .laser import Laser
from .part import STATUS_CODE


class TraceRecorder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.frames: list[dict] = []

    def record(self, time: float, laser: Laser, conveyors: Sequence[Conveyor],
               kpis: dict) -> None:
        parts = []
        for c in conveyors:
            for p in c.buffer:
                parts.append([p.number, c.id, 0.0, "W"])
            for p in c.parts:
                parts.append([p.number, c.id, round(p.x, 2), STATUS_CODE[p.status]])
        self.frames.append({
            "t": round(time, 3),
            "L": [round(laser.x, 2), round(laser.y, 2), laser.state.value,
                  laser.current_part.number if laser.current_part else None],
            "C": [[1 if c.is_running else 0, ",".join(sorted(c.stop_reasons))] for c in conveyors],
            "P": parts,
            "K": kpis,
        })

    def to_dict(self, events: list[dict], summary: dict, policy_name: str) -> dict:
        cfg = self.config
        g, ws = cfg.geometry, cfg.laser.workspace
        return {
            "meta": {
                "policy": policy_name,
                "seed": cfg.simulation.seed,
                "arrival_mode": cfg.arrivals.mode,
                "dt": cfg.simulation.dt,
                "trace_interval": cfg.simulation.trace_interval,
                "unprocessed_at_zone_end": cfg.simulation.unprocessed_at_zone_end,
                "conveyor_stop_policy": cfg.simulation.conveyor_stop_policy,
                "motion_model": cfg.laser.motion_model,
            },
            "geometry": {
                "entry_length": g.entry_length, "active_length": g.active_length,
                "exit_length": g.exit_length, "total_length": g.total_length,
                "workspace": {"x_min": ws.x_min, "x_max": ws.x_max,
                              "y_min": ws.y_min, "y_max": ws.y_max},
                "conveyors": [{"id": c.id, "y": c.y, "speed": c.speed} for c in cfg.conveyors],
                "min_part_spacing": cfg.simulation.min_part_spacing,
            },
            "frames": self.frames,
            "events": events,
            "summary": summary,
        }
