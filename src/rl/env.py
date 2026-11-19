"""Gymnasium environment: the laser dispatching SMDP (docs/methods/rl.md).

One ``step(action)`` = one dispatching decision. The simulator runs until the
next *decision epoch*: the laser is idle and at least two conveyors have a
dispatch candidate (when only one conveyor has candidates the decision is forced
and taken automatically, it is not shown to the agent).

Reward
------
Between two decision epochs t and t' the agent receives

    r = -( W(t, t') + lambda_T * T_B(t, t') + lambda_N * N_B(t, t') ) / reward_scale

with W = integral of the number of arrived-but-unfinished parts (Little: its total
over the episode is sum_j F_j up to one step dt per part), T_B the conveyor time
blocked and N_B the blocking events (counted at S_j). The undiscounted return of an
episode is therefore  -(sum_j F_j + Pi) / reward_scale = -n * penalized_flow_time /
reward_scale: the agent optimizes exactly the default objective of the code.

Episodes: stochastic arrivals for ``arrival_horizon`` seconds, then the system is
drained (as every experiment). Each reset samples a load (``loads``) and a seed
from the environment's own random stream; evaluation seeds (1000-1029) are never
used for training (``train_seed_base``).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from policies import get_policy
from rl.features import action_mask, encode, observation_size, part_for_action
from simulation.config import PROJECT_ROOT, load_config
from simulation.engine import Simulation

DEFAULT_LOADS = (("default", None), ("heavy", str(PROJECT_ROOT / "config" / "examples" /
                                                  "heavy_load.yaml")))


class LaserConveyorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, loads: Sequence[tuple[str, str | None]] = DEFAULT_LOADS,
                 set_items: Sequence[str] = (), dt: float = 0.05,
                 prepositioning: str = "none", reward_scale: float = 100.0,
                 train_seed_base: int = 100_000, seed: int | None = None,
                 fixed_seed: int | None = None) -> None:
        super().__init__()
        self.loads = list(loads)
        self.set_items = list(set_items)
        self.dt = dt
        self.prepositioning = prepositioning
        self.reward_scale = reward_scale
        self.train_seed_base = train_seed_base
        self.fixed_seed = fixed_seed
        self._rng = random.Random(seed)
        cfg = self._config(self.loads[0][1], 0)
        self.n = cfg.n_conveyors
        self.observation_space = spaces.Box(0.0, 1.0, (observation_size(self.n),), np.float32)
        self.action_space = spaces.Discrete(self.n)
        self.sim: Simulation | None = None
        self.candidates: list = []
        self.load_name = self.loads[0][0]

    # ----------------------------------------------------------------- helpers
    def _config(self, path, sim_seed: int):
        return load_config(path, set_items=[
            "arrivals.mode=stochastic", f"simulation.seed={sim_seed}",
            f"simulation.dt={self.dt}", f"laser.prepositioning={self.prepositioning}",
            "policy.name=nearest_head"] + self.set_items)

    def _cost_state(self) -> tuple[float, float, int]:
        s = self.sim
        return s.wip_area, sum(s.metrics.stopped_blocked), s.n_block_events

    def _advance(self) -> bool:
        """Run until the next decision epoch. Returns False when the episode ended."""
        sim = self.sim
        n = self.n
        while not sim.finished:
            sim.begin_step()
            cands = sim.decision_candidates()
            if cands:
                mask = action_mask(cands, n)
                if mask.sum() > 1:
                    self.candidates = cands
                    return True
                sim.end_step(part_for_action(cands, int(np.argmax(mask)), n))
            else:
                sim.end_step(None)
        self.candidates = []
        return False

    def action_masks(self) -> np.ndarray:
        """Used by sb3-contrib's MaskablePPO."""
        if not self.candidates:
            return np.ones(self.n, dtype=bool)
        return action_mask(self.candidates, self.n)

    def _obs(self) -> np.ndarray:
        if not self.candidates:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return encode(self.sim, self.candidates)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        options = options or {}
        idx = options.get("load_index", self._rng.randrange(len(self.loads)))
        self.load_name, path = self.loads[idx]
        sim_seed = options.get("sim_seed", self.fixed_seed if self.fixed_seed is not None
                               else self.train_seed_base + self._rng.randrange(10**6))
        cfg = self._config(path, sim_seed)
        self.sim = Simulation(cfg, policy=get_policy("nearest_head"), record_trace=False)
        self._lambda_n = cfg.blocking_penalty_per_event
        self._lambda_t = cfg.blocking_penalty_per_second
        self._last = self._cost_state()
        self.episode_cost = 0.0
        alive = self._advance()
        # cost accrued before the first decision belongs to the first transition
        return self._obs(), {"load": self.load_name, "sim_seed": sim_seed, "alive": alive}

    def step(self, action):
        if not self.candidates:                     # episode already over
            return self._obs(), 0.0, True, False, {}
        part = part_for_action(self.candidates, int(action), self.n)
        self.sim.end_step(part)
        alive = self._advance()
        w, tb, nb = self._cost_state()
        w0, tb0, nb0 = self._last
        cost = (w - w0) + self._lambda_t * (tb - tb0) + self._lambda_n * (nb - nb0)
        self._last = (w, tb, nb)
        self.episode_cost += cost
        reward = -cost / self.reward_scale
        terminated = not alive
        info = {}
        if terminated:
            s = self.sim.result().summary
            info = {"summary": s, "episode_cost": self.episode_cost, "load": self.load_name}
        return self._obs(), float(reward), terminated, False, info


def episode_cost_of(sim: Simulation) -> float:
    """sum_j F_j + Pi of a finished run (the quantity the return measures)."""
    s = sim.result().summary
    return s["avg_flow_time"] * s["parts_completed"] + s["blocking_penalty"] \
        if s["parts_completed"] else math.inf
