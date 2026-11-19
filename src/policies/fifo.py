"""Policy 1 — FIFO global: serve the part that has been waiting the longest."""

from __future__ import annotations

from . import register_policy
from .base_policy import BasePolicy


@register_policy("fifo")
class FifoPolicy(BasePolicy):
    def select(self, candidates, ctx):
        # Earliest arrival first; ties broken by conveyor id for determinism.
        return min(candidates, key=lambda p: (p.arrival_time, p.conveyor_id, p.number))
