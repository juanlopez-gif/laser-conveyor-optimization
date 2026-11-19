"""Cartesian XY laser head (Z fixed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import LaserSpec, Workspace
from .motion import path_length, travel_time
from .part import Part


class LaserState(str, Enum):
    IDLE = "IDLE"                          # no job assigned
    MOVING = "MOVING"                      # travelling to the meeting point
    WAITING_FOR_PART = "WAITING_FOR_PART"  # at the meeting point, part not there yet
    PROCESSING = "PROCESSING"              # working on a part


class LaserBusyError(RuntimeError):
    """Raised when trying to start a second job while the laser is processing."""


@dataclass
class Laser:
    spec: LaserSpec
    x: float = 0.0
    y: float = 0.0
    state: LaserState = LaserState.IDLE
    current_part: Part | None = None
    target_x: float | None = None
    target_y: float | None = None
    planned_speed: float | None = None     # conveyor speed assumed by the current plan
    processing_end: float | None = None
    distance_x: float = 0.0
    distance_y: float = 0.0
    distance: float = 0.0
    history: list[str] = field(default_factory=list)   # ids of processed parts, in order

    @classmethod
    def from_spec(cls, spec: LaserSpec) -> "Laser":
        laser = cls(spec=spec)
        laser.set_position(spec.home_x, spec.home_y)
        return laser

    # ---------------------------------------------------------------- geometry
    @property
    def workspace(self) -> Workspace:
        return self.spec.workspace

    def set_position(self, x: float, y: float) -> None:
        """Teleport (initialisation / tests). Refuses positions outside the workspace."""
        if not self.workspace.contains(x, y):
            raise ValueError(f"({x}, {y}) is outside the laser workspace {self.workspace}")
        self.x, self.y = x, y

    def travel_time_to(self, x: float, y: float) -> float:
        return travel_time(self.x, self.y, x, y, self.spec.speed_x,
                           self.spec.speed_y, self.spec.motion_model)

    @property
    def at_target(self) -> bool:
        return (self.target_x is not None and abs(self.x - self.target_x) < 1e-9
                and abs(self.y - self.target_y) < 1e-9)

    def move_towards(self, tx: float, ty: float, dt: float) -> float:
        """Advance the head for ``dt`` seconds towards (tx, ty).

        The target is clamped to the workspace, so the head can never leave it.
        Returns the distance travelled in this step.
        """
        tx, ty = self.workspace.clamp(tx, ty)
        vx, vy = self.spec.speed_x, self.spec.speed_y
        x0, y0 = self.x, self.y
        if self.spec.motion_model == "simultaneous":
            self.x += _step(tx - self.x, vx * dt)
            self.y += _step(ty - self.y, vy * dt)
        else:  # sequential: X first, the remaining time of the step goes to Y
            dx = _step(tx - self.x, vx * dt)
            self.x += dx
            remaining = dt - abs(dx) / vx
            if abs(tx - self.x) < 1e-12 and remaining > 0:
                self.y += _step(ty - self.y, vy * remaining)
        mx, my = abs(self.x - x0), abs(self.y - y0)
        self.distance_x += mx
        self.distance_y += my
        d = path_length(mx, my)
        self.distance += d
        return d

    # ---------------------------------------------------------------- jobs
    def assign(self, part: Part, x: float, y: float, planned_speed: float) -> None:
        if self.state == LaserState.PROCESSING:
            raise LaserBusyError("Laser is processing; cannot accept a new assignment")
        self.current_part = part
        part.assigned = True
        self.set_target(x, y, planned_speed)
        self.state = LaserState.MOVING

    def set_target(self, x: float, y: float, planned_speed: float) -> None:
        self.target_x, self.target_y = self.workspace.clamp(x, y)
        self.planned_speed = planned_speed

    def start_processing(self, part: Part, now: float) -> None:
        """Begin processing ``part``. Only one part at a time is allowed."""
        if self.state == LaserState.PROCESSING:
            raise LaserBusyError(
                f"Laser already processing {self.current_part.id if self.current_part else '?'}")
        if not self.workspace.contains(part.x, part.y):
            raise ValueError(f"Part {part.id} at ({part.x:.2f}, {part.y:.2f}) is out of reach")
        # Final alignment (at most one belt step, < speed*dt).
        self.move_towards(part.x, part.y, dt=1e9)
        self.current_part = part
        self.state = LaserState.PROCESSING
        self.processing_end = now + part.processing_time

    def finish_processing(self) -> Part:
        part = self.current_part
        assert part is not None
        self.history.append(part.id)
        self.clear()
        return part

    def clear(self) -> None:
        if self.current_part is not None:
            self.current_part.assigned = False
        self.current_part = None
        self.target_x = self.target_y = self.planned_speed = None
        self.processing_end = None
        self.state = LaserState.IDLE


def _step(delta: float, max_step: float) -> float:
    """Signed displacement towards ``delta`` limited to ``max_step``."""
    if abs(delta) <= max_step:
        return delta
    return max_step if delta > 0 else -max_step
