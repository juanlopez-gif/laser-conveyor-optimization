"""Policy 3 — Round robin by conveyor: 1 -> 2 -> 3 -> 1 ... skipping empty ones.

Within the chosen conveyor the most advanced part (largest X, i.e. the one
closest to leaving the laser zone) is served first — FIFO within the belt.
"""

from __future__ import annotations

from . import register_policy
from .base_policy import BasePolicy


@register_policy("round_robin")
class RoundRobinPolicy(BasePolicy):
    def __init__(self, **params):
        super().__init__(**params)
        self.last_served = 0           # conveyor id served last (0 = none yet)

    def reset(self) -> None:
        self.last_served = 0

    def select(self, candidates, ctx):
        n = len(ctx.conveyors)
        by_conveyor: dict[int, list] = {}
        for p in candidates:
            by_conveyor.setdefault(p.conveyor_id, []).append(p)
        for k in range(1, n + 1):
            cid = (self.last_served + k - 1) % n + 1
            if cid in by_conveyor:
                self.last_served = cid
                return max(by_conveyor[cid], key=lambda p: (p.x, -p.number))
        return None
