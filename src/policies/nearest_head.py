"""Policy — Nearest Head: the nearest conveyor, but always its FIRST unprocessed part.

Motivation (docs/CHANGELOG.md 0.4.0, docs/methods/nearest_head.md): the good
sequences found by the metaheuristics keep the belt order inside every conveyor
and only interleave conveyors. ``nearest_job`` violates this when it serves a
later part of a conveyor first: the earlier (downstream) part keeps travelling,
reaches the end of the laser zone unprocessed and stops the whole conveyor.

Rule: on every conveyor only the *head* is eligible, i.e. the unprocessed part
that is furthest downstream (largest x). Among the heads, pick the one with the
smallest laser interception time (ties: earliest arrival, then id).

Because the parts of a conveyor keep their order and spacing on the belt, the
head is always the first of its conveyor to enter the laser zone, so whenever a
conveyor has a candidate its head is a candidate too (with or without
pre-positioning). The rule therefore never idles the laser when ``nearest_job``
would not.
"""

from __future__ import annotations

from . import register_policy
from .base_policy import BasePolicy


def conveyor_heads(candidates):
    """The most downstream candidate of every conveyor."""
    heads = {}
    for p in candidates:
        h = heads.get(p.conveyor_id)
        if h is None or p.x > h.x or (p.x == h.x and p.number < h.number):
            heads[p.conveyor_id] = p
    return list(heads.values())


@register_policy("nearest_head")
class NearestHeadPolicy(BasePolicy):
    def select(self, candidates, ctx):
        heads = conveyor_heads(candidates)
        return min(heads, key=lambda p: (ctx.travel_key(p), p.arrival_time, p.number))
