"""Train the RL dispatcher (docs/methods/rl.md, version 0.6.0).

    PYTHONPATH=src python -m rl.train --teacher rollout --prepositioning when_idle \
        --episodes 40 --dagger-iters 3 --dagger-episodes 24 --ppo-steps 60000 \
        --workers 2 --seed 0 --tag rl_wi_s0

Pipeline
--------
1. **Teacher data.** The teacher policy (default ``rollout``) drives the environment;
   at every decision epoch we store (observation, mask, teacher action). The teacher
   may pick any candidate; its label is the *conveyor* of the chosen part (the env
   then serves that conveyor's head). Returns (discount ``--gamma``) are stored for
   the critic.
2. **Behaviour cloning (BC).** Actor by masked cross-entropy on the teacher labels,
   critic by regression on the returns.
3. **DAgger** (0.6.0; Ross, Gordon & Bagnell 2011). For ``--dagger-iters`` iterations
   the *learner* drives the environment (with probability ``beta_i = beta0**i`` the
   teacher acts instead), the teacher labels every visited decision state, the data
   are aggregated and the actor/critic retrained. This addresses the distribution
   shift of plain BC (the learner visits states the teacher never visits).
4. **PPO fine-tuning** (MaskablePPO) from the best checkpoint so far, with
   ``--gamma 1`` by default (0.6.0): episodes are finite and drained, so the
   undiscounted return is exactly ``-(sum F_j + Pi)/reward_scale`` and PPO optimizes
   the evaluated objective (0.5.0 used gamma = 0.99, a discounted proxy).
5. **Checkpoint selection on validation** (0.6.0). After BC, after every DAgger
   iteration and every ``--val-every`` PPO steps the deterministic policy is run on a
   fixed validation set (simulation seeds 2000+, both loads; disjoint from the
   training seeds >= 100000 and from the evaluation seeds 1000-1029). Score = mean over
   episodes of cost / cost of ``nearest_head`` on the same episode. The delivered model
   is the best checkpoint, so fine-tuning cannot make it worse *on validation* (in
   0.5.0 PPO degraded the cloned policy and nothing caught it).

Outputs: ``models/<tag>.zip`` (selected model) and ``models/<tag>_training.json``
(settings, accuracies, validation scores, learning curve). Intermediate checkpoints
go to ``results/rl/<tag>/`` (not versioned).

Teacher collection and validation run in ``--workers`` processes (the simulator, not
the network, is the bottleneck: CPU cores help, a GPU does not).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from policies import get_policy
from policies.base_policy import DecisionContext
from rl.env import LaserConveyorEnv
from simulation.config import PROJECT_ROOT

MODELS = PROJECT_ROOT / "models"
WORK = PROJECT_ROOT / "results" / "rl"
VAL_SEED_BASE = 2000

_LEARNERS: dict[str, object] = {}


def _load_learner(path: str):
    if path not in _LEARNERS:
        from sb3_contrib import MaskablePPO
        _LEARNERS[path] = MaskablePPO.load(path, device="cpu")
    return _LEARNERS[path]


def _teacher_action(pol, env) -> int:
    sim = env.sim
    ctx = DecisionContext(sim.time, sim.laser, sim.conveyors, sim.geometry,
                          sim._intercept_memo(), simulation=sim)
    return pol.select(env.candidates, ctx).conveyor_id - 1


def run_episodes(job: dict) -> dict:
    """Episodes driven by the teacher or (DAgger) by a learner, labelled by the teacher.

    job keys: teacher, teacher_params, env_kwargs, seed, load_indices, gamma,
    learner (model path or None), beta (probability that the teacher acts when a
    learner is given), label (query the teacher and store the data; False = plain
    evaluation run of the learner, or of the teacher if no learner), sim_seeds
    (optional fixed simulation seeds, one per episode).
    """
    env = LaserConveyorEnv(seed=job["seed"], **job["env_kwargs"])
    learner = _load_learner(job["learner"]) if job.get("learner") else None
    rng = random.Random(job["seed"])
    label = job.get("label", True)
    obs_l, act_l, mask_l, ret_l, costs = [], [], [], [], []
    sim_seeds = job.get("sim_seeds") or [None] * len(job["load_indices"])
    for li, ss in zip(job["load_indices"], sim_seeds):
        pol = None
        if label or learner is None:
            pol = get_policy(job["teacher"], job["teacher_params"])
            pol.reset()
        opts = {"load_index": li}
        if ss is not None:
            opts["sim_seed"] = ss
        obs, info = env.reset(options=opts)
        rewards, done = [], not info["alive"]
        inf = info if done else {}          # decision-free episode: cost is in reset's info
        while not done:
            mask = env.action_masks()
            a_t = _teacher_action(pol, env) if pol is not None else None
            if learner is not None and (a_t is None or rng.random() >= job.get("beta", 0.0)):
                a, _ = learner.predict(obs, action_masks=mask, deterministic=True)
                a = int(a)
            else:
                a = a_t
            if label:
                obs_l.append(obs)
                mask_l.append(mask)
                act_l.append(a_t)
            obs, r, done, _, inf = env.step(a)
            rewards.append(r)
        if label:
            g, rets = 0.0, []
            for r in reversed(rewards):
                g = r + job["gamma"] * g
                rets.append(g)
            ret_l += rets[::-1]
        if inf:
            costs.append({"load": env.load_name, "sim_seed": info["sim_seed"],
                          "penalized_flow_time": inf["summary"]["objective"],
                          "episode_cost": inf["episode_cost"]})
    n_obs = int(env.observation_space.shape[0])
    return {"obs": np.array(obs_l, np.float32).reshape(-1, n_obs),
            "actions": np.array(act_l, np.int64),
            "masks": np.array(mask_l, bool).reshape(-1, env.n),
            "returns": np.array(ret_l, np.float32), "costs": costs}


def _merge(parts: list[dict]) -> dict:
    return {"obs": np.concatenate([p["obs"] for p in parts]),
            "actions": np.concatenate([p["actions"] for p in parts]),
            "masks": np.concatenate([p["masks"] for p in parts]),
            "returns": np.concatenate([p["returns"] for p in parts]),
            "costs": [c for p in parts for c in p["costs"]]}


def run_parallel(jobs: list[dict], workers: int) -> dict:
    if workers <= 1 or len(jobs) == 1:
        return _merge([run_episodes(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return _merge(list(ex.map(run_episodes, jobs)))


def _split_jobs(base: dict, load_indices: list[int], workers: int, seed: int,
                sim_seeds: list | None = None) -> list[dict]:
    k = max(1, min(workers, len(load_indices)))
    jobs = []
    for w in range(k):
        j = dict(base, seed=seed * 1000 + w, load_indices=load_indices[w::k])
        if sim_seeds is not None:
            j["sim_seeds"] = sim_seeds[w::k]
        jobs.append(j)
    return jobs


def collect(teacher: str, teacher_params: dict, episodes: int, gamma: float,
            env_kwargs: dict, seed: int, learner: str | None = None, beta: float = 1.0,
            workers: int = 1, n_loads: int = 2) -> dict:
    """Teacher-labelled decision data (with ``learner``: DAgger roll-outs)."""
    base = {"teacher": teacher, "teacher_params": teacher_params, "env_kwargs": env_kwargs,
            "gamma": gamma, "learner": learner, "beta": beta, "label": True}
    return run_parallel(_split_jobs(base, [e % n_loads for e in range(episodes)], workers,
                                    seed), workers)


def validation_set(n_per_load: int, n_loads: int = 2) -> tuple[list[int], list[int]]:
    loads = [li for li in range(n_loads) for _ in range(n_per_load)]
    seeds = [VAL_SEED_BASE + k for _ in range(n_loads) for k in range(n_per_load)]
    return loads, seeds


def evaluate(env_kwargs: dict, n_per_load: int, workers: int, learner: str | None = None,
             policy: str = "nearest_head", policy_params: dict | None = None) -> list[dict]:
    """Validation episode costs of a learner (model path) or, if None, of ``policy``."""
    loads, seeds = validation_set(n_per_load)
    base = {"teacher": policy, "teacher_params": policy_params or {},
            "env_kwargs": env_kwargs, "gamma": 1.0, "learner": learner, "beta": 0.0,
            "label": False}
    return run_parallel(_split_jobs(base, loads, workers, 7, seeds), workers)["costs"]


def score(costs: list[dict], ref: dict) -> float:
    """Mean ratio to the reference (nearest_head) on the same validation episode."""
    return float(np.mean([c["episode_cost"] / ref[(c["load"], c["sim_seed"])] for c in costs]))


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
    ap.add_argument("--prepositioning", default="none",
                    help="laser pre-positioning mode of training and validation "
                         "(none / when_idle / always)")
    ap.add_argument("--episodes", type=int, default=40, help="initial teacher episodes")
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--dagger-episodes", type=int, default=24)
    ap.add_argument("--dagger-epochs", type=int, default=30,
                    help="retraining epochs on the aggregated data per DAgger iteration")
    ap.add_argument("--beta0", type=float, default=0.5,
                    help="teacher-acts probability in DAgger iteration i is beta0**i")
    ap.add_argument("--ppo-steps", type=int, default=60_000)
    ap.add_argument("--val-every", type=int, default=15_000)
    ap.add_argument("--val-episodes", type=int, default=10, help="validation episodes per load")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--reward-scale", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--workers", type=int, default=2,
                    help="processes for teacher collection and validation")
    ap.add_argument("--n-envs", type=int, default=2, help="parallel simulators for PPO")
    args = ap.parse_args(argv)

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.manual_seed(args.seed)
    MODELS.mkdir(exist_ok=True)
    work = WORK / args.tag
    work.mkdir(parents=True, exist_ok=True)
    env_kwargs = {"dt": args.dt, "prepositioning": args.prepositioning,
                  "reward_scale": args.reward_scale}
    teacher_params = {"objective": args.teacher_objective} if args.teacher_objective else {}
    log: dict = {"args": vars(args), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "validation": []}
    t0 = time.time()

    def save_log():
        (MODELS / f"{args.tag}_training.json").write_text(json.dumps(log, indent=1),
                                                          encoding="utf-8")

    ref = {(c["load"], c["sim_seed"]): c["episode_cost"]
           for c in evaluate(env_kwargs, args.val_episodes, args.workers)}
    best = {"score": float("inf"), "stage": None}

    def checkpoint(model, stage: str, extra: dict | None = None) -> float:
        path = WORK / args.tag / f"{stage}.zip"
        model.save(path)
        sc = score(evaluate(env_kwargs, args.val_episodes, args.workers,
                            learner=str(path)), ref)
        log["validation"].append({"stage": stage, "score": sc, "time": time.time() - t0,
                                  **(extra or {})})
        if sc < best["score"]:
            best.update(score=sc, stage=stage)
            shutil.copy(path, MODELS / f"{args.tag}.zip")
        print(f"[{time.time() - t0:6.0f} s] {stage}: validation {sc:.4f} "
              f"(best {best['stage']} {best['score']:.4f})", flush=True)
        save_log()
        return sc

    data = collect(args.teacher, teacher_params, args.episodes, args.gamma, env_kwargs,
                   args.seed, workers=args.workers)
    log["teacher"] = {"decisions": int(len(data["actions"])), "seconds": time.time() - t0,
                      "episodes": data["costs"]}
    print(f"teacher data: {len(data['actions'])} decisions in {time.time() - t0:.0f} s",
          flush=True)

    if args.n_envs > 1:
        from functools import partial

        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([partial(LaserConveyorEnv, seed=args.seed * 100 + 1 + i,
                                     **env_kwargs) for i in range(args.n_envs)])
    else:
        env = LaserConveyorEnv(seed=args.seed * 100 + 1, **env_kwargs)
    model = MaskablePPO("MlpPolicy", env, learning_rate=1e-4,
                        n_steps=max(64, 1024 // args.n_envs), batch_size=256,
                        n_epochs=5, gamma=args.gamma, gae_lambda=0.95, clip_range=0.1,
                        ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    bc = behaviour_cloning(model, data, args.bc_epochs, 1e-3, seed=args.seed)
    log["bc"] = {"accuracy": bc["accuracy"], "decisions": int(len(data["actions"]))}
    checkpoint(model, "bc", {"accuracy": bc["accuracy"]})

    log["dagger"] = []
    prev = "bc"
    for it in range(1, args.dagger_iters + 1):
        beta = args.beta0 ** it
        t1 = time.time()
        new = collect(args.teacher, teacher_params, args.dagger_episodes, args.gamma,
                      env_kwargs, args.seed * 10 + it, learner=str(WORK / args.tag /
                                                                 f"{prev}.zip"),
                      beta=beta, workers=args.workers)
        data = _merge([data, new])
        res = behaviour_cloning(model, data, args.dagger_epochs, 5e-4, seed=args.seed + it)
        log["dagger"].append({"iter": it, "beta": beta,
                              "new_decisions": int(len(new["actions"])),
                              "total_decisions": int(len(data["actions"])),
                              "accuracy": res["accuracy"], "seconds": time.time() - t1,
                              "rollout_costs": new["costs"]})
        prev = f"dagger{it}"
        checkpoint(model, prev, {"accuracy": res["accuracy"]})

    if args.ppo_steps > 0:
        ppo_start = best["stage"]
        start = MaskablePPO.load(WORK / args.tag / f"{ppo_start}.zip", device="cpu")
        model.policy.load_state_dict(start.policy.state_dict())
        curve: list[dict] = []

        class Monitor(BaseCallback):
            next_val = args.val_every

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "summary" in info:
                        curve.append({"timesteps": int(self.num_timesteps),
                                      "load": info.get("load"),
                                      "penalized_flow_time": info["summary"]["objective"]})
                if self.num_timesteps >= self.next_val:
                    self.next_val += args.val_every
                    checkpoint(self.model, f"ppo{self.num_timesteps}")
                return True

        t1 = time.time()
        model.learn(total_timesteps=args.ppo_steps, callback=Monitor())
        log["ppo"] = {"start": ppo_start, "seconds": time.time() - t1, "curve": curve}
        checkpoint(model, "ppo_final")
    log["selected"] = dict(best)
    log["seconds"] = time.time() - t0
    save_log()
    print(f"selected {best['stage']} (validation {best['score']:.4f}) -> "
          f"{MODELS / (args.tag + '.zip')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
