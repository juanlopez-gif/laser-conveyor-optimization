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


def test_dagger_collection_and_validation(tmp_path):
    """0.6.0: learner-driven roll-outs are labelled by the teacher; validation is paired."""
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    from rl.train import collect, evaluate, run_episodes, score, validation_set
    kw = {"dt": 0.05, "prepositioning": "when_idle", "reward_scale": 1000.0}
    env = LaserConveyorEnv(seed=0, **kw)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, device="cpu")
    path = tmp_path / "m.zip"
    model.save(path)
    # learner acts (beta = 0) but every stored label is the teacher's (nearest_head)
    d = collect("nearest_head", {}, 2, 1.0, kw, seed=0, learner=str(path), beta=0.0)
    assert len(d["actions"]) == len(d["obs"]) == len(d["masks"]) == len(d["returns"]) > 0
    assert all(m[a] for m, a in zip(d["masks"], d["actions"]))
    # gamma = 1: the first return of an episode is minus its whole cost / scale
    first = run_episodes({"teacher": "nearest_head", "teacher_params": {}, "env_kwargs": kw,
                          "seed": 3, "load_indices": [0], "gamma": 1.0, "learner": None,
                          "sim_seeds": [2000]})
    assert first["returns"][0] * 1000.0 == pytest.approx(-first["costs"][0]["episode_cost"],
                                                         rel=1e-4)
    # validation: fixed seeds, the reference policy scores exactly 1
    loads, seeds = validation_set(2)
    assert loads == [0, 0, 1, 1] and seeds == [2000, 2001, 2000, 2001]
    ref_costs = evaluate(kw, 2, workers=1)
    ref = {(c["load"], c["sim_seed"]): c["episode_cost"] for c in ref_costs}
    assert score(ref_costs, ref) == pytest.approx(1.0)


def test_decision_free_episode_reports_its_cost():
    """0.6.0 (external audit 2026-11-19): when every decision of an episode is forced,
    the whole run happens inside reset(); its cost must still be reported (reset info,
    same keys as a terminal step) and kept by the episode collector."""
    from rl.env import episode_cost_of
    from rl.train import run_episodes
    kw = {"dt": 0.05, "set_items": ["simulation.arrival_horizon=12"]}
    env = LaserConveyorEnv(seed=0, **kw)
    found = None
    for s in range(50):
        _, info = env.reset(options={"load_index": 0, "sim_seed": 5000 + s})
        if not info["alive"] and info["summary"]["parts_completed"] >= 1:
            found = 5000 + s
            break
    assert found is not None, "no decision-free episode found"
    assert info["episode_cost"] > 0
    # the WIP integral is accumulated per step: within one dt per part of sum F_j + Pi
    n = info["summary"]["parts_completed"]
    assert info["episode_cost"] == pytest.approx(episode_cost_of(env.sim), abs=0.05 * n + 1e-9)
    out = run_episodes({"teacher": "nearest_head", "teacher_params": {}, "env_kwargs": kw,
                        "seed": 0, "load_indices": [0], "gamma": 1.0, "learner": None,
                        "sim_seeds": [found]})
    assert len(out["costs"]) == 1 and out["costs"][0]["episode_cost"] == pytest.approx(
        info["episode_cost"])
