"""Train the RL dispatcher: behaviour cloning from a teacher, then PPO (docs/methods/rl.md).

    PYTHONPATH=src python -m rl.train --teacher rollout --episodes 60 --ppo-steps 150000

1. **Teacher data.** The teacher policy (default ``rollout``) drives the environment;
   at every decision epoch we store (observation, mask, teacher action). The teacher
   may pick any candidate; its action is the *conveyor* of the chosen part (the env
   then serves that conveyor's head). Discounted returns are stored for the critic.
2. **Behaviour cloning.** The actor of a MaskablePPO policy is trained by
   cross-entropy on the teacher actions (invalid actions masked); the critic by
   regression on the teacher's returns, so that PPO starts with a consistent
   value function and does not wash out the cloned policy.
3. **PPO fine-tuning** with action masking (sb3-contrib ``MaskablePPO``) on the same
   training distribution (mixed loads, training seeds >= 100000).

Outputs: ``models/rl_bc.zip`` (after step 2), ``models/rl_ppo.zip`` (after step 3) and
``models/rl_training.json`` (settings, BC accuracy, learning curve).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from policies import get_policy
from policies.base_policy import DecisionContext
from rl.env import LaserConveyorEnv
from simulation.config import PROJECT_ROOT

MODELS = PROJECT_ROOT / "models"


def collect(teacher: str, teacher_params: dict, episodes: int, gamma: float,
            env_kwargs: dict, seed: int) -> dict:
    env = LaserConveyorEnv(seed=seed, **env_kwargs)
    obs_l, act_l, mask_l, ret_l, costs = [], [], [], [], []
    for ep in range(episodes):
        pol = get_policy(teacher, teacher_params)
        pol.reset()
        obs, info = env.reset(options={"load_index": ep % len(env.loads)})
        rewards, done = [], not info["alive"]
        while not done:
            sim = env.sim
            ctx = DecisionContext(sim.time, sim.laser, sim.conveyors, sim.geometry,
                                  sim._intercept_memo(), simulation=sim)
            choice = pol.select(env.candidates, ctx)
            mask = env.action_masks()
            a = choice.conveyor_id - 1
            obs_l.append(obs)
            mask_l.append(mask)
            act_l.append(a)
            obs, r, done, _, inf = env.step(a)
            rewards.append(r)
        g, rets = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            rets.append(g)
        ret_l += rets[::-1]
        if inf:
            costs.append({"load": env.load_name,
                          "penalized_flow_time": inf["summary"]["objective"]})
    return {"obs": np.array(obs_l, np.float32), "actions": np.array(act_l, np.int64),
            "masks": np.array(mask_l, bool), "returns": np.array(ret_l, np.float32),
            "teacher_costs": costs}


def behaviour_cloning(model, data: dict, epochs: int, lr: float, batch: int = 256,
                      seed: int = 0) -> dict:
    policy = model.policy
    policy.set_training_mode(True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs = torch.as_tensor(data["obs"])
    act = torch.as_tensor(data["actions"])
    masks = data["masks"]
    ret = torch.as_tensor(data["returns"])
    n = len(act)
    gen = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        perm = gen.permutation(n)
        tot_pi = tot_v = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            dist = policy.get_distribution(obs[idx], action_masks=masks[idx])
            loss_pi = -dist.log_prob(act[idx]).mean()
            values = policy.predict_values(obs[idx]).flatten()
            loss_v = torch.nn.functional.mse_loss(values, ret[idx])
            loss = loss_pi + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_pi += loss_pi.item() * len(idx)
            tot_v += loss_v.item() * len(idx)
        history.append({"epoch": epoch, "loss_pi": tot_pi / n, "loss_v": tot_v / n})
    policy.set_training_mode(False)
    with torch.no_grad():
        dist = policy.get_distribution(obs, action_masks=masks)
        pred = dist.distribution.probs.argmax(dim=1).numpy()
    return {"accuracy": float((pred == data["actions"]).mean()), "history": history}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="rollout")
    ap.add_argument("--teacher-objective", default=None,
                    help="look-ahead objective of a rollout teacher (default: its own)")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--ppo-steps", type=int, default=150_000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel simulators for PPO (one process each; use ~ number of CPU "
                         "cores). The simulator, not the network, is the bottleneck: more CPU "
                         "cores help, a GPU does not (docs/methods/rl.md)")
    args = ap.parse_args(argv)

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.manual_seed(args.seed)
    MODELS.mkdir(exist_ok=True)
    env_kwargs = {"dt": args.dt}
    teacher_params = {"objective": args.teacher_objective} if args.teacher_objective else {}
    log: dict = {"args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    t0 = time.time()
    data = collect(args.teacher, teacher_params, args.episodes, args.gamma, env_kwargs,
                   args.seed)
    log["teacher"] = {"decisions": int(len(data["actions"])), "seconds": time.time() - t0,
                      "episodes": data["teacher_costs"]}
    print(f"teacher data: {len(data['actions'])} decisions in {time.time() - t0:.0f} s")

    if args.n_envs > 1:
        from functools import partial

        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([partial(LaserConveyorEnv, seed=args.seed + 1 + i, **env_kwargs)
                             for i in range(args.n_envs)])
    else:
        env = LaserConveyorEnv(seed=args.seed + 1, **env_kwargs)
    model = MaskablePPO("MlpPolicy", env, learning_rate=1e-4,
                        n_steps=max(64, 1024 // args.n_envs), batch_size=256,
                        n_epochs=5, gamma=args.gamma, gae_lambda=0.95, clip_range=0.1,
                        ent_coef=0.003, seed=args.seed, device="cpu", verbose=0,
                        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    bc = behaviour_cloning(model, data, args.bc_epochs, 1e-3, seed=args.seed)
    log["bc"] = bc
    print(f"behaviour cloning: accuracy {bc['accuracy']:.3f}")
    model.save(MODELS / f"{args.tag}_bc.zip")

    curve: list[dict] = []

    class Curve(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "summary" in info:
                    curve.append({"timesteps": int(self.num_timesteps),
                                  "load": info.get("load"),
                                  "penalized_flow_time": info["summary"]["objective"]})
            return True

    t1 = time.time()
    if args.ppo_steps > 0:
        model.learn(total_timesteps=args.ppo_steps, callback=Curve())
    log["ppo"] = {"seconds": time.time() - t1, "curve": curve}
    model.save(MODELS / f"{args.tag}_ppo.zip")
    (MODELS / f"{args.tag}_training.json").write_text(json.dumps(log, indent=1),
                                                      encoding="utf-8")
    print(f"PPO: {args.ppo_steps} steps in {time.time() - t1:.0f} s -> {MODELS / (args.tag + '_ppo.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
