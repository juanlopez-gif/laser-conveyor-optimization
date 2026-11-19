"""Laser kinematics: travel-time models and interception of moving parts.

Travel-time models (documented decision, see docs/mathematical_model.md)
-------------------------------------------------------------------------
* ``simultaneous`` (default): X and Y axes are driven by independent motors
  that move at the same time, so the slower axis dominates::

      t = max(dx / vx, dy / vy)

* ``sequential``: the axes move one after the other (X first, then Y)::

      t = dx / vx + dy / vy

Accelerations are neglected (constant speed per axis) in this V1.

Interception
------------
Parts keep moving while the laser travels towards them (their conveyor only
stops when processing starts). The laser therefore aims at the *meeting
point*: the earliest time ``t`` such that the laser can be at the part's
predicted position ``x(t)`` no later than the part itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MOTION_MODELS = ("simultaneous", "sequential")


def travel_time(x1: float, y1: float, x2: float, y2: float,
                vx: float, vy: float, model: str = "simultaneous") -> float:
    """Time for the laser to go from (x1, y1) to (x2, y2)."""
    if vx <= 0 or vy <= 0:
        raise ValueError("Laser speeds must be positive")
    tx = abs(x2 - x1) / vx
    ty = abs(y2 - y1) / vy
    if model == "simultaneous":
        return max(tx, ty)
    if model == "sequential":
        return tx + ty
    raise ValueError(f"Unknown motion model '{model}'. Use one of {MOTION_MODELS}")


@dataclass(frozen=True)
class Intercept:
    """Planned meeting point between the laser and a (possibly moving) part."""

    time: float      # seconds from now until the laser and the part meet
    x: float
    y: float


def find_intercept(lx: float, ly: float, px: float, py: float, part_speed: float,
                   x_limit: float, hold_at_limit: bool,
                   vx: float, vy: float, model: str,
                   samples: int = 64, iterations: int = 40) -> Intercept | None:
    """Earliest meeting point between the laser and a part moving along +X.

    Parameters
    ----------
    lx, ly : current laser position.
    px, py : current part position.
    part_speed : current speed of the part's conveyor (0 if stopped).
    x_limit : last X where the laser can meet the part (end of laser zone).
    hold_at_limit : True if the part stops at ``x_limit`` (``block`` mode);
        False if it keeps going and becomes unreachable (``pass`` mode).

    Returns ``None`` if the laser cannot reach the part inside the zone.
    """
    def tt(x: float) -> float:
        return travel_time(lx, ly, x, py, vx, vy, model)

    if px > x_limit + 1e-9:
        return None
    if part_speed <= 0:
        return Intercept(tt(px), px, py)

    t_lim = (x_limit - px) / part_speed           # time until the part reaches the limit

    def gap(t: float) -> float:                    # >0: laser still late at time t
        return tt(px + part_speed * t) - t

    if gap(t_lim) > 0:                             # cannot meet before the limit
        if hold_at_limit:                          # part waits at the limit for the laser
            return Intercept(tt(x_limit), x_limit, py)
        return None
    if gap(0.0) <= 0:
        return Intercept(0.0, px, py)

    # gap() is continuous; scan for the first sign change, then bisect.
    lo, hi = 0.0, t_lim
    prev = 0.0
    for k in range(1, samples + 1):
        t = t_lim * k / samples
        if gap(t) <= 0:
            lo, hi = prev, t
            break
        prev = t
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if gap(mid) <= 0:
            hi = mid
        else:
            lo = mid
    x = px + part_speed * hi
    return Intercept(hi, x, py)


def path_length(dx: float, dy: float) -> float:
    """Euclidean length of a straight move (used for travel-distance metrics)."""
    return math.hypot(dx, dy)
