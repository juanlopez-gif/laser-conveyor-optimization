"""Information model: which parts the controller can observe (version 0.7.3).

The physical model is unchanged; this only restricts what a *policy* may use.

Levels (``simulation.observation``):

* ``belts`` — every part on a belt, in the entry zone or in the laser zone, with its
  position (status MOVING, READY or PROCESSING). **This is the real cell** (owner,
  2026-11-20: an overhead camera sees all belts and their whole working area). Parts
  waiting upstream of the belt entry (status WAITING) are not seen.
* ``zone``  — only parts inside the laser zone (READY or PROCESSING): a what-if for a
  cell without the overhead camera. Incompatible with pre-positioning.
* ``full``  — every arrived part, also those waiting upstream: the information upper
  bound, for what-if studies only (the behaviour of every version before 0.7.3).
* ``auto``  — ``belts``.

Visible length (0.7.10, ``simulation.visible_length``): at level ``belts`` the camera
sees the laser zone and only the last ``visible_length`` cm of the entry zone (the
stretch ``[a - visible_length, a)``). ``null`` (the default) is the whole entry zone,
i.e. the level ``belts`` of 0.7.3–0.7.9; ``0`` makes ``belts`` see what ``zone`` sees
(parts become observable on entering the laser zone) while still allowing
pre-positioning to parts already visible. The parameter is the "preceding stretch of
defined length" of treatment B in ``docs/plans/papers.md``; it is ignored by ``zone``
and ``full``. A MOVING part upstream of the visible stretch is not observable: it is
neither a token nor a pre-positioning candidate, and its time on the belt is counted
from the instant it becomes visible (``rl2.features.observed_since``).

Every policy must decide from observable parts only. The rule policies already do:
their candidates are READY parts (``none``). Look-ahead (``rollout`` and hybrids),
the RL agents and the strong-RL network are restricted explicitly; the tests check it
(``tests/test_observation.py``).

Statistical knowledge of the cell (arrival rates, processing-time distribution, the
cell parameters) is not an observation and stays available.
"""

from __future__ import annotations

OBSERVATION_LEVELS = ("zone", "belts", "full")

#: status names (simulation.part.PartStatus values) visible at each level
_VISIBLE = {
    "zone": frozenset({"READY", "PROCESSING"}),
    "belts": frozenset({"MOVING", "READY", "PROCESSING"}),
    "full": frozenset({"WAITING", "MOVING", "READY", "PROCESSING"}),
}


def resolve_observation(value, prepositioning: str) -> str:
    """Resolve ``auto`` and check consistency with the pre-positioning mode."""
    level = "auto" if value is None else str(value)
    if level == "auto":
        return "belts"
    if level not in OBSERVATION_LEVELS:
        raise ValueError(f"simulation.observation must be auto or one of {OBSERVATION_LEVELS}")
    if level == "zone" and prepositioning != "none":
        raise ValueError("laser.prepositioning '%s' needs to see the entry zone: "
                         "use simulation.observation belts or full" % prepositioning)
    return level


def visible_status(status_name: str, level: str) -> bool:
    """True if a part with this status (name, e.g. 'READY') may be observable
    (a MOVING part at level ``belts`` also needs :func:`visible_moving`)."""
    return status_name in _VISIBLE[level]


def resolve_visible_length(value, entry_length: float) -> float:
    """``simulation.visible_length``: null -> the whole entry zone; else a length in
    ``[0, entry_length]`` cm."""
    if value is None:
        return float(entry_length)
    length = float(value)
    if not 0.0 <= length <= entry_length + 1e-9:
        raise ValueError("simulation.visible_length must be between 0 and "
                         f"geometry.entry_length = {entry_length} cm, not {length}")
    return min(length, float(entry_length))


def visible_moving(x: float, active_start: float, visible_length: float,
                   eps: float = 1e-7) -> bool:
    """True if a MOVING part at position ``x`` is inside the visible stretch of the
    entry zone, ``[active_start - visible_length, active_start)``, at level ``belts``."""
    return x >= active_start - visible_length - eps
