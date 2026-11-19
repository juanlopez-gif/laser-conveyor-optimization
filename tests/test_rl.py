"""Tests of the RL formulation (version 0.5.0, docs/methods/rl.md)."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from rl.env import LaserConveyorEnv  # noqa: E402
from rl.features import (action_mask, heads_by_conveyor, observation_size,  # noqa: E402
                         part_for_action)
from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402

HEAVY = "config/examples/heavy_load.yaml"


def test_split_step_is_bit_identical():
    cfg = load_config(set_items=["arrivals.mode=stochastic", "simulation.seed=4"])
    a = Simulation(cfg, record_trace=False).run().summary
    sim = Simulation(cfg, record_trace=False)
    while not sim.finished:
        sim.begin_step()
        sim.end_step()
    assert sim.result().summary == a


def _greedy_head(env):
    heads = heads_by_conveyor(env.candidates, env.n)
    return min((i for i, h in enumerate(heads) if h is not None),
               key=lambda i: (env.sim.intercept(heads[i]).time, heads[i].arrival_time,
                              heads[i].number))


@pytest.mark.parametrize("load_index", [0, 1])
def test_env_reproduces_nearest_head_and_return_equals_cost(load_index):
    env = LaserConveyorEnv(dt=0.05, seed=1)
    obs, info = env.reset(options={"load_index": load_index, "sim_seed": 7})
    assert obs.shape == (observation_size(3),) and env.observation_space.contains(obs)
    total, done = 0.0, False
    while not done:
        mask = env.action_masks()
        assert mask.sum() >= 2                       # forced decisions are not shown
        a = _greedy_head(env)
        assert mask[a]
        obs, r, done, _, inf = env.step(a)
        total += r
    s = inf["summary"]
    path = None if load_index == 0 else HEAVY
    ref = Simulation(load_config(path, set_items=[
        "arrivals.mode=stochastic", "simulation.seed=7", "simulation.dt=0.05",
        "policy.name=nearest_head"]), record_trace=False).run().summary
    assert s["avg_flow_time"] == pytest.approx(ref["avg_flow_time"], abs=1e-9)
    cost = s["avg_flow_time"] * s["parts_completed"] + s["blocking_penalty"]
    # return = -(sum F_j + Pi)/scale, up to one step dt per part (arrivals off-grid)
    assert -total * env.reward_scale == pytest.approx(cost, abs=s["parts_completed"] * 0.05)


def test_masks_and_actions():
    env = LaserConveyorEnv(dt=0.05, seed=3)
    env.reset(options={"load_index": 1})
    for _ in range(20):
        mask = env.action_masks()
        assert (mask == action_mask(env.candidates, env.n)).all()
        for a in range(env.n):
            if mask[a]:
                p = part_for_action(env.candidates, a, env.n)
                assert p.conveyor_id == a + 1 and p in env.candidates
            else:
                with pytest.raises(ValueError):
                    part_for_action(env.candidates, a, env.n)
        _, _, done, _, _ = env.step(int(mask.argmax()))
        if done:
            break


def test_trained_policy_runs_and_is_feasible(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    from optimization.schedule_check import check_schedule
    from rl.train import behaviour_cloning, collect
    data = collect("nearest_head", {}, 2, 0.99, {"dt": 0.05}, seed=0)
    env = LaserConveyorEnv(dt=0.05, seed=0)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, device="cpu")
    bc = behaviour_cloning(model, data, epochs=30, lr=1e-3)
    assert bc["accuracy"] > 0.6
    path = tmp_path / "m.zip"
    model.save(path)
    r = Simulation(load_config(set_items=[
        "policy.name=rl", f"policy.params.rl.model={path.as_posix()}",
        "arrivals.mode=stochastic", "simulation.seed=1000",
        "simulation.arrival_horizon=120"]), record_trace=False).run()
    assert r.summary["parts_completed"] == r.summary["parts_generated"]
    assert check_schedule(r) == []
