"""Interface shared by every scheduling (dispatching) policy.

The engine calls :meth:`BasePolicy.select` every time the laser becomes free
and at least one part is available. The policy only *chooses*; it never moves
anything. Everything it may need is exposed through :class:`DecisionContext`
(read-only view of the system + helper estimates), so policies stay
independent from the engine internals.

To add a policy: create ``src/policies/my_policy.py`` with a subclass
decorated by ``@register_policy("my_policy")`` and set ``policy.name:
my_policy`` in the config. The engine does not need to change.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from simulation.config import Geometry
from simulation.conveyor import Conveyor
from simulation.laser import Laser
from simulation.motion import Intercept
from simulation.part import Part


@dataclass
class DecisionContext:
    """Read-only snapshot handed to a policy at decision time."""

    time: float
    laser: Laser
    conveyors: Sequence[Conveyor]
    geometry: Geometry
    _intercept: Callable[[Part], Intercept | None]
    #: The running Simulation (read-only use!). Lets look-ahead policies such as
    #: rollout clone the current state; ordinary policies must not touch it.
    simulation: Any = None

    def conveyor(self, conveyor_id: int) -> Conveyor:
        return self.conveyors[conveyor_id - 1]

    def intercept(self, part: Part) -> Intercept | None:
        """Planned meeting point with ``part`` (None if unreachable)."""
        return self._intercept(part)

    def travel_time(self, part: Part) -> float:
        """Estimated time for the laser to reach ``part`` (inf if unreachable)."""
        icp = self._intercept(part)
        return math.inf if icp is None else icp.time

    def travel_key(self, part: Part) -> float:
        """``travel_time`` rounded to 1e-9 s, for comparisons with tie-breaks (0.7.0).
        Two parts with the same physical travel time can differ by round-off
        (e.g. a part stopped at b = 45 - 1e-15 cm); rounding lets the documented
        tie-break (earliest arrival, then id) decide instead."""
        return round(self.travel_time(part), 9)

    def waiting_time(self, part: Part) -> float:
        return self.time - part.arrival_time

    def observed_key(self, part: Part) -> float:
        """Instant from which the part has been observable at the cell's information
        level (the engine records it; falls back to the arrival time). The rules break
        ties on it, not on the arrival time, which is not observed (0.7.26)."""
        return part.arrival_time if part.observed_time is None else part.observed_time

    def time_to_zone_end(self, part: Part) -> float:
        """Time until the part reaches the end of the laser zone at nominal speed."""
        c = self.conveyor(part.conveyor_id)
        return max(0.0, self.geometry.active_end - part.x) / c.nominal_speed

    def urgency(self, part: Part) -> float:
        """In (0, 1]: 1 when the part is at the end of the laser zone."""
        return 1.0 / (1.0 + self.time_to_zone_end(part))


class BasePolicy(ABC):
    """Abstract dispatching rule."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def reset(self) -> None:
        """Clear internal state between runs (stateless policies do nothing)."""

    @abstractmethod
    def select(self, candidates: Sequence[Part], ctx: DecisionContext) -> Part | None:
        """Return the part the laser should go to next (or None to stay idle).

        ``candidates`` are READY, unassigned, reachable parts (never empty).
        """
