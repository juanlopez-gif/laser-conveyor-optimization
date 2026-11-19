"""Policy 4 — Weighted score heuristic.

    score = w_wait * waiting_time  -  w_travel * travel_time  +  w_urgency * urgency

and the part with the HIGHEST score is chosen.

* waiting_time : seconds since the part arrived (rewards old parts),
* travel_time  : estimated interception time of the laser (penalises far parts),
* urgency      : 1 / (1 + time to reach the end of the laser zone) in (0, 1];
                 equals 1 when the part is blocking its conveyor.

Weights come from ``policy.params.weighted_score`` in the config.
"""

from __future__ import annotations

from . import register_policy
from .base_policy import BasePolicy


@register_policy("weighted_score")
class WeightedScorePolicy(BasePolicy):
    def __init__(self, w_wait: float = 1.0, w_travel: float = 1.0,
                 w_urgency: float = 10.0, **params):
        super().__init__(w_wait=w_wait, w_travel=w_travel, w_urgency=w_urgency, **params)
        self.w_wait, self.w_travel, self.w_urgency = w_wait, w_travel, w_urgency

    def score(self, part, ctx) -> float:
        return (self.w_wait * ctx.waiting_time(part)
                - self.w_travel * ctx.travel_time(part)
                + self.w_urgency * ctx.urgency(part))

    def select(self, candidates, ctx):
        return max(candidates, key=lambda p: (self.score(p, ctx), -p.number))
