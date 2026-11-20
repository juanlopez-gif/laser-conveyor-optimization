"""Hybrid RL + look-ahead policies (version 0.6.0, docs/methods/hybrid.md)."""

from __future__ import annotations

import pytest

pytest.importorskip("sb3_contrib")

from optimization.schedule_check import check_schedule  # noqa: E402
from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402

MODEL = "models/rl_bc.zip"


def _run(policy: str, params: dict, seed: int = 1000, horizon: float = 120.0):
    sets = ["arrivals.mode=stochastic", f"simulation.seed={seed}",
            f"simulation.arrival_horizon={horizon}", f"policy.name={policy}"]
    sets += [f"policy.params.{policy}.{k}={v}" for k, v in params.items()]
    return Simulation(load_config(set_items=sets), record_trace=False).run()


def _key(r):
    return [(p.id, p.start_processing_time) for p in r.parts]


def test_ensemble_of_one_model_equals_the_agent():
    a = _run("rl", {"model": MODEL})
    b = _run("rl_ensemble", {"models": f"{MODEL},{MODEL},{MODEL}"})
    assert _key(a) == _key(b)


def test_gate_extremes():
    # confidence 0: always the agent; confidence > 1: always the rollout
    assert _key(_run("rl_gated", {"model": MODEL, "confidence": 0.0})) == \
        _key(_run("rl", {"model": MODEL}))
    assert _key(_run("rl_gated", {"model": MODEL, "confidence": 1.01})) == \
        _key(_run("rollout", {}))


@pytest.mark.parametrize("policy,params", [
    ("rl_rollout", {"model": MODEL}),
    ("rl_filter", {"model": MODEL, "top_k": 2}),
    ("rl_filter", {"model": MODEL, "top_k": 2, "base_policy": "rl"}),
    ("rl_gated", {"model": MODEL, "confidence": 0.9, "min_slack_s": 2.0}),
])
def test_hybrids_are_feasible(policy, params):
    r = _run(policy, params)
    assert r.summary["parts_completed"] == r.summary["parts_generated"]
    assert check_schedule(r) == []
