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

Pre-positioning (``laser.prepositioning: true``): the part may still be upstream of
the laser zone (x < a). The meeting point is then searched only for x >= a, i.e.
from the instant the part enters the zone; if the laser gets there first it waits
at (a, y_k). See docs/formal_formulation.md §1.5.
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
                   samples: int = 64, iterations: int = 40,
                   x_min: float | None = None) -> Intercept | None:
    """Earliest meeting point between the laser and a part moving along +X.

    Parameters
    ----------
    lx, ly : current laser position.
    px, py : current part position.
    part_speed : current speed of the part's conveyor (0 if stopped).
    x_limit : last X where the laser can meet the part (end of laser zone).
    hold_at_limit : True if the part stops at ``x_limit`` (``block`` mode);
        False if it keeps going and becomes unreachable (``pass`` mode).
    x_min : first X where the laser can meet the part (start of the laser zone).
        Only matters for parts still upstream of it (pre-positioning). A part
        upstream of ``x_min`` on a stopped belt is unreachable for now (None).

    Returns ``None`` if the laser cannot reach the part inside the zone.
    """
    def tt(x: float) -> float:
        return travel_time(lx, ly, x, py, vx, vy, model)

    if px > x_limit + 1e-9:
        return None
    t0, x0 = 0.0, px                               # earliest meeting time / position
    if x_min is not None and px < x_min - 1e-9:    # upstream of the zone (pre-positioning)
        if part_speed <= 0:
            return None
        t0, x0 = (x_min - px) / part_speed, x_min
    if part_speed <= 0:
        return Intercept(tt(px), px, py)

    t_lim = t0 + (x_limit - x0) / part_speed      # time until the part reaches the limit

    def gap(t: float) -> float:                    # >0: laser still late at time t
        return tt(x0 + part_speed * (t - t0)) - t

    # Exact earliest root of gap() on [t0, t_lim] (fixed in 0.6.0 after the external
    # audit of 2026-11-19). Both travel-time models are piecewise linear in the target
    # x, with kinks at x = lx and x = lx +- vx * |dy| / vy, so gap() is piecewise linear
    # in t: check the segments between kinks in order and solve the first one that
    # reaches gap <= 0 by linear interpolation. The previous version assumed gap() to be
    # decreasing and returned early when gap(t_lim) > 0; that holds while the belt is
    # slower than the laser's X axis (always the case in the default cell and in every
    # study so far), but misses an earlier meeting when a faster belt carries the part
    # towards the laser. ``samples`` and ``iterations`` are kept for backward
    # compatibility (unused).
    if gap(t0) <= 0:
        return Intercept(t0, x0, py)
    dy_t = abs(py - ly) / vy
    kinks = {t0, t_lim}
    for xk in (lx, lx - vx * dy_t, lx + vx * dy_t):
        tk = t0 + (xk - x0) / part_speed
        if t0 < tk < t_lim:
            kinks.add(tk)
    pts = sorted(kinks)
    for a, b in zip(pts, pts[1:]):
        ga, gb = gap(a), gap(b)
        if gb <= 0:
            t = a + (b - a) * ga / (ga - gb) if ga > gb else b
            t = min(max(t, a), b)
            if gap(t) > 1e-9:                      # guard against round-off
                t = b
            return Intercept(t, x0 + part_speed * (t - t0), py)
    # cannot meet before the limit
    if hold_at_limit:                              # part waits at the limit for the laser
        return Intercept(max(tt(x_limit), t_lim), x_limit, py)
    return None


def path_length(dx: float, dy: float) -> float:
    """Euclidean length of a straight move (used for travel-distance metrics)."""
    return math.hypot(dx, dy)
