"""Policy 2 — Nearest Job First: smallest laser travel (interception) time."""

from __future__ import annotations

from . import register_policy
from .base_policy import BasePolicy


@register_policy("nearest_job")
class NearestJobPolicy(BasePolicy):
    def select(self, candidates, ctx):
        return min(candidates, key=lambda p: (ctx.travel_key(p), p.arrival_time, p.number))
