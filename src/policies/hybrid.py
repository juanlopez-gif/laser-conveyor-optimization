"""Hybrid policies: learned dispatcher + look-ahead (version 0.6.0, docs/methods/hybrid.md).

All four use the RL agent of :mod:`policies.rl_policy` (action = serve the head of a
conveyor) and/or the rollout of :mod:`policies.rollout`. They are ``parametric``
(they need a trained model), so they are only run when asked for explicitly.

``rl_rollout``  (1) rollout whose **base policy is the RL agent**: every candidate is
                evaluated by simulating the known parts with the agent instead of
                ``nearest_job``. Rollout improves on its base policy, so a better base
                should give a better rollout (Bertsekas). Parameters: ``model`` plus
                every rollout parameter.
``rl_filter``   (2) the agent **proposes** its ``top_k`` best conveyors (default 2);
                the rollout simulates only their heads (base ``nearest_job`` by
                default; ``base_policy=rl`` uses the agent also inside). About
                6 / top_k times cheaper than the plain rollout.
``rl_gated``    (3) **switch**: use the agent directly when it is confident (largest
                action probability >= ``confidence``, default 0.8) or when there is no
                time to think (slack < ``min_slack_s``, default 0 = never); otherwise
                call the rollout. The slack is the smallest, over the conveyor heads,
                of (time until the head reaches the zone end b) - (laser travel time to
                it): how long the decision can wait before some part blocks.
``rl_ensemble`` (4) **majority vote** of several agents (``models``: comma-separated
                paths, e.g. three training seeds); ties broken by the sum of the
                action probabilities.

Forced decisions (only one conveyor has a candidate) are taken directly, as in the
RL environment.
"""

from __future__ import annotations

import math
import time as _time
from pathlib import Path
from typing import Sequence

import numpy as np

from simulation.config import PROJECT_ROOT

from . import get_policy, register_policy
from .base_policy import BasePolicy
from .rl_policy import DEFAULT_MODEL, load_model
from .rollout import RolloutPolicy


def _resolve(path) -> Path:
    p = Path(path) if path else DEFAULT_MODEL
    return p if p.is_absolute() else PROJECT_ROOT / p


def rl_probs(model, sim, candidates) -> tuple[np.ndarray, np.ndarray]:
    """(action probabilities, action mask) of a trained agent at a decision epoch."""
    import torch

    from rl.features import action_mask, encode
    n = len(sim.conveyors)
    mask = action_mask(candidates, n)
    obs = torch.as_tensor(encode(sim, candidates)[None])
    with torch.no_grad():
        dist = model.policy.get_distribution(obs, action_masks=mask[None])
        probs = dist.distribution.probs.numpy()[0]
    return probs * mask, mask


def decision_slack(candidates, ctx) -> float:
    """Smallest (time to reach the zone end) - (laser travel time) over the heads."""
    from rl.features import heads_by_conveyor
    g = ctx.geometry
    slack = math.inf
    for p in heads_by_conveyor(candidates, len(ctx.conveyors)):
        if p is None:
            continue
        c = ctx.conveyors[p.conveyor_id - 1]
        to_end = max(0.0, g.active_end - p.x) / c.nominal_speed
        slack = min(slack, to_end - ctx.travel_time(p))
    return slack


@register_policy("rl_rollout")
class RLRolloutPolicy(RolloutPolicy):
    parametric = True

    def __init__(self, model: str | None = None, **params):
        params.setdefault("base_policy", "rl")
        if params["base_policy"] == "rl":
            params.setdefault("base_params", {"model": str(_resolve(model))})
        super().__init__(**params)
        self.params["model"] = model


@register_policy("rl_filter")
class RLFilterPolicy(RolloutPolicy):
    parametric = True

    def __init__(self, model: str | None = None, top_k: int = 2,
                 base_policy: str = "nearest_job", **params):
        if base_policy == "rl":
            params.setdefault("base_params", {"model": str(_resolve(model))})
        params["max_candidates"] = None
        super().__init__(base_policy=base_policy, **params)
        self.params.update(model=model, top_k=top_k)
        self.model_path = _resolve(model)
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    def _select(self, candidates, ctx):
        from rl.features import heads_by_conveyor
        heads = heads_by_conveyor(candidates, len(ctx.conveyors))
        valid = [i for i, h in enumerate(heads) if h is not None]
        if len(valid) <= 1:
            return heads[valid[0]] if valid else None
        probs, _ = rl_probs(load_model(self.model_path), ctx.simulation, candidates)
        best = sorted(valid, key=lambda i: -probs[i])[:self.top_k]
        return super()._select([heads[i] for i in best], ctx)


@register_policy("rl_gated")
class RLGatedPolicy(BasePolicy):
    parametric = True

    def __init__(self, model: str | None = None, confidence: float = 0.8,
                 min_slack_s: float = 0.0, rollout_params: dict | None = None, **params):
        super().__init__(model=model, confidence=confidence, min_slack_s=min_slack_s,
                         rollout_params=rollout_params, **params)
        self.model_path = _resolve(model)
        self.confidence = float(confidence)
        self.min_slack_s = float(min_slack_s)
        self.rollout = get_policy("rollout", dict(rollout_params or {}))
        self.reset()

    def reset(self) -> None:
        self.n_decisions = 0
        self.n_rollout = 0
        self.decision_times: list[float] = []
        if hasattr(self, "rollout"):
            self.rollout.reset()

    def stats(self) -> dict[str, float]:
        d = self.decision_times
        return {"decisions": self.n_decisions, "rollout_calls": self.n_rollout,
                "rollout_share": self.n_rollout / self.n_decisions if self.n_decisions else 0.0,
                "decision_time_mean_ms": 1000 * sum(d) / len(d) if d else 0.0}

    def select(self, candidates, ctx):
        from rl.features import part_for_action
        start = _time.perf_counter()
        try:
            n = len(ctx.conveyors)
            probs, mask = rl_probs(load_model(self.model_path), ctx.simulation, candidates)
            if mask.sum() <= 1:
                return part_for_action(candidates, int(mask.argmax()), n)
            self.n_decisions += 1
            sure = probs.max() >= self.confidence
            urgent = decision_slack(candidates, ctx) < self.min_slack_s
            if sure or urgent:
                return part_for_action(candidates, int(probs.argmax()), n)
            self.n_rollout += 1
            return self.rollout.select(candidates, ctx)
        finally:
            self.decision_times.append(_time.perf_counter() - start)


@register_policy("rl_ensemble")
class RLEnsemblePolicy(BasePolicy):
    parametric = True

    def __init__(self, models: str | Sequence[str] = "", **params):
        super().__init__(models=models, **params)
        paths = models.split(",") if isinstance(models, str) else list(models)
        self.model_paths = [_resolve(p.strip()) for p in paths if str(p).strip()]
        if not self.model_paths:
            raise ValueError("rl_ensemble needs models=path1,path2,...")

    def select(self, candidates, ctx):
        from rl.features import part_for_action
        n = len(ctx.conveyors)
        votes = np.zeros(n)
        mass = np.zeros(n)
        mask = None
        for path in self.model_paths:
            probs, mask = rl_probs(load_model(path), ctx.simulation, candidates)
            votes[int(probs.argmax())] += 1
            mass += probs
        if mask.sum() <= 1:
            return part_for_action(candidates, int(mask.argmax()), n)
        top = votes.max()
        choice = max((i for i in range(n) if votes[i] == top and mask[i]),
                     key=lambda i: mass[i])
        return part_for_action(candidates, choice, n)
