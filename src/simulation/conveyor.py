"""Conveyor belt model.

A conveyor is a 1-D belt along +X. Every part on the belt moves with it, so
when the belt stops *all* its parts stop and the distance between them is
preserved. Whether the belt runs is decided by a set of *stop reasons*: the
belt runs only when that set is empty. This keeps the stopping rules
(processing, blocking, future policies) explicit and independent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import ConveyorSpec, Geometry
from .part import Part, PartStatus

STOP_PROCESSING = "processing"   # the laser is processing a part (stop policy)
STOP_BLOCKED = "blocked"         # an unprocessed part reached the end of the laser zone


@dataclass
class Conveyor:
    spec: ConveyorSpec
    geometry: Geometry
    parts: list[Part] = field(default_factory=list)       # on the belt, upstream -> ordered by entry
    buffer: deque[Part] = field(default_factory=deque)    # WAITING parts before the entry
    stop_reasons: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ state
    @property
    def id(self) -> int:
        return self.spec.id

    @property
    def y(self) -> float:
        return self.spec.y

    @property
    def nominal_speed(self) -> float:
        return self.spec.speed

    @property
    def is_running(self) -> bool:
        return not self.stop_reasons

    @property
    def current_speed(self) -> float:
        return self.spec.speed if self.is_running else 0.0

    def stop(self, reason: str) -> None:
        self.stop_reasons.add(reason)

    def release(self, reason: str) -> None:
        self.stop_reasons.discard(reason)

    def pending_count(self) -> int:
        """Queue length: parts that arrived and whose processing has not started."""
        return len(self.buffer) + sum(1 for p in self.parts if p.start_processing_time is None)

    # --------------------------------------------------------------- entry
    def can_admit(self, min_spacing: float) -> bool:
        """A WAITING part may enter if the belt runs and the entry is free."""
        if not self.is_running or not self.buffer:
            return False
        return not self.parts or self.parts[-1].x >= min_spacing - 1e-9

    def admit(self, time: float) -> Part:
        part = self.buffer.popleft()
        part.x, part.y = 0.0, self.y
        part.status = PartStatus.MOVING
        part.entry_time = time
        self.parts.append(part)
        return part

    # ------------------------------------------------------------- motion
    def blocking_part(self, block_unprocessed: bool) -> Part | None:
        """Unprocessed part sitting at the end of the laser zone, if any."""
        if not block_unprocessed:
            return None
        end = self.geometry.active_end
        for p in self.parts:
            if p.needs_processing and p.x >= end - 1e-9:
                return p
        return None

    def advance(self, dt: float, block_unprocessed: bool) -> float:
        """Move the belt for ``dt`` seconds. Returns the distance travelled.

        In ``block`` mode the movement is clamped so that no unprocessed part
        crosses the end of the laser zone (it stops exactly on the boundary).
        """
        if not self.is_running:
            return 0.0
        distance = self.spec.speed * dt
        if block_unprocessed:
            end = self.geometry.active_end
            for p in self.parts:
                if p.needs_processing and p.x < end:
                    distance = min(distance, end - p.x)
        for p in self.parts:
            p.x += distance
        return distance
