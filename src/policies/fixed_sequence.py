"""Fixed-sequence policy: the decoder Phi(pi) of docs/formal_formulation.md (Section 3).

Replays a given processing order ``sequence`` (list of part ids such as
``["P002", "P001", ...]``). When the laser is free it goes to the next part of
the sequence as soon as that part is in the laser zone; it never skips ahead.

A sequence can be infeasible under blocking: if the next part is still upstream
of the laser zone while its conveyor is blocked by another unprocessed part, the
system deadlocks. This is detected and reported with :class:`InfeasibleSequence`.
"""

from __future__ import annotations

from simulation.conveyor import STOP_BLOCKED

from . import register_policy
from .base_policy import BasePolicy


class InfeasibleSequence(RuntimeError):
    """The sequence cannot be executed (deadlock caused by blocking)."""


@register_policy("fixed_sequence")
class FixedSequencePolicy(BasePolicy):
    parametric = True      # needs a sequence: not listed among the stand-alone policies
    def __init__(self, sequence: list[str] | None = None, **params):
        super().__init__(sequence=list(sequence or []), **params)
        self.sequence = list(sequence or [])
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def next_id(self) -> str | None:
        return self.sequence[self.position] if self.position < len(self.sequence) else None

    def select(self, candidates, ctx):
        target = self.next_id()
        if target is None:
            return None
        for p in candidates:
            if p.id == target:
                self.position += 1
                return p
        self._check_deadlock(target, ctx)
        return None                     # wait (laser idle) for the next part

    def _check_deadlock(self, target: str, ctx) -> None:
        for conv in ctx.conveyors:
            waiting = list(conv.buffer) + [p for p in conv.parts if p.needs_processing]
            for p in waiting:
                if p.id == target and STOP_BLOCKED in conv.stop_reasons \
                        and p.x < ctx.geometry.active_start:
                    raise InfeasibleSequence(
                        f"{target} cannot reach the laser zone: conveyor {conv.id} is blocked "
                        "by a part that comes later in the sequence")
