"""Online look-ahead methods for the benchmark: the rollout policy.

These methods do not compute a sequence offline; they return the name of an
online policy (plus its parameters) which the benchmark then runs on the
instance with ``evaluate_policy``. The policy only uses information available
at each decision epoch (see ``policies/rollout.py``), so its result is an
*online* value comparable with the dispatching baselines, not with the
clairvoyant optimum.

Registered methods
------------------
``policy:rollout``        rollout over ``nearest_job``; the look-ahead minimises
                          the benchmark objective when the rollout supports it
                          (otherwise its default, average flow time).
``policy:rollout_exact``  same, but the look-ahead uses the simulation ``dt``
                          and evaluates every candidate (exact one-step rollout
                          for the parts present; slower).
"""

from __future__ import annotations

from optimization.solvers import Solution, register_method
from policies.rollout import LOOKAHEAD_OBJECTIVES

DEFAULT_ROLLOUT_PARAMS: dict = {"base_policy": "nearest_job", "max_candidates": 6,
                                "horizon_s": None, "dt_lookahead": 0.05,
                                "objective": "avg_flow_time"}


def _objective_param(objective: str | dict) -> str | dict:
    """Translate the benchmark objective into a look-ahead objective (if supported)."""
    if isinstance(objective, dict):
        weights = {k: v for k, v in objective.items() if v}
        if weights and all(k in LOOKAHEAD_OBJECTIVES for k in weights):
            return weights
        return DEFAULT_ROLLOUT_PARAMS["objective"]
    return objective if objective in LOOKAHEAD_OBJECTIVES else DEFAULT_ROLLOUT_PARAMS["objective"]


def rollout_params(objective: str | dict = "avg_flow_time", **overrides) -> dict:
    """Parameters of the rollout policy for a given benchmark objective."""
    params = dict(DEFAULT_ROLLOUT_PARAMS, objective=_objective_param(objective))
    params.update(overrides)
    return params


@register_method("policy:rollout")
def policy_rollout(instance, objective, time_limit, seed) -> Solution:
    """Rollout (one-step look-ahead) over nearest_job, default speed settings."""
    params = rollout_params(objective)
    return Solution(policy="rollout", policy_params=params,
                    info={"online": True, "lookahead_objective": params["objective"]})


@register_method("policy:rollout_exact")
def policy_rollout_exact(instance, objective, time_limit, seed) -> Solution:
    """Rollout with an exact look-ahead (simulation dt, every candidate)."""
    params = rollout_params(objective, dt_lookahead=None, max_candidates=None)
    return Solution(policy="rollout", policy_params=params,
                    info={"online": True, "lookahead_objective": params["objective"]})
