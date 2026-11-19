"""Observation, action space and action mask of the RL formulation (docs/methods/rl.md).

Action space
------------
``Discrete(N)``: action k = "serve the head of conveyor k+1", i.e. the most
downstream dispatch candidate of that conveyor. Restricting the agent to heads uses
the structural result of phase 2 (good sequences keep belt order inside every
conveyor; ``nearest_head``) and keeps the action space small and fixed. Action k is
valid (mask = 1) iff conveyor k+1 has at least one candidate.

Observation (all features roughly in [0, 1])
--------------------------------------------
* laser: x, y (normalized to the workspace)
* per conveyor k (N blocks):
    - running, blocked, upstream buffer length / 5, has candidate
    - K = 3 slots for its first K unstarted parts on the belt, head first. Each slot:
      present, x / L, inside laser zone, time to the zone end / 20 s, interception
      time / 20 s (1 if unreachable), waiting time / 60 s, processing time / 5 s,
      blocking now (sits at b)
* global: unfinished parts / 20

Size = 2 + N * (4 + 8 K) + 1  (87 for N = 3).
"""

from __future__ import annotations

import math

import numpy as np

from simulation.conveyor import STOP_BLOCKED

K_SLOTS = 3
SLOT_FEATURES = 8
CONVEYOR_FEATURES = 4


def observation_size(n_conveyors: int) -> int:
    return 2 + n_conveyors * (CONVEYOR_FEATURES + K_SLOTS * SLOT_FEATURES) + 1


def heads_by_conveyor(candidates, n_conveyors: int) -> list:
    """Most downstream candidate of every conveyor (None if it has no candidate)."""
    heads = [None] * n_conveyors
    for p in candidates:
        i = p.conveyor_id - 1
        h = heads[i]
        if h is None or p.x > h.x or (p.x == h.x and p.number < h.number):
            heads[i] = p
    return heads


def action_mask(candidates, n_conveyors: int) -> np.ndarray:
    return np.array([h is not None for h in heads_by_conveyor(candidates, n_conveyors)],
                    dtype=bool)


def part_for_action(candidates, action: int, n_conveyors: int):
    head = heads_by_conveyor(candidates, n_conveyors)[int(action)]
    if head is None:
        raise ValueError(f"action {action}: conveyor {action + 1} has no candidate")
    return head


def _clip01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def encode(sim, candidates) -> np.ndarray:
    """Observation vector of the simulation ``sim`` at a decision epoch."""
    cfg, g = sim.cfg, sim.geometry
    ws = cfg.laser.workspace
    n = len(sim.conveyors)
    t = sim.time
    cand_ids = {p.id for p in candidates}
    obs = np.zeros(observation_size(n), dtype=np.float32)
    obs[0] = _clip01((sim.laser.x - ws.x_min) / max(ws.x_max - ws.x_min, 1e-9))
    obs[1] = _clip01((sim.laser.y - ws.y_min) / max(ws.y_max - ws.y_min, 1e-9))
    i = 2
    unfinished = 0
    for c in sim.conveyors:
        unstarted = sorted((p for p in c.parts if p.needs_processing),
                           key=lambda p: -p.x)
        unfinished += len(c.buffer) + sum(1 for p in c.parts if p.completion_time is None)
        obs[i] = 1.0 if c.is_running else 0.0
        obs[i + 1] = 1.0 if STOP_BLOCKED in c.stop_reasons else 0.0
        obs[i + 2] = _clip01(len(c.buffer) / 5.0)
        obs[i + 3] = 1.0 if any(p.id in cand_ids for p in unstarted) else 0.0
        i += CONVEYOR_FEATURES
        for s in range(K_SLOTS):
            if s < len(unstarted):
                p = unstarted[s]
                icp = sim.intercept(p)
                obs[i] = 1.0
                obs[i + 1] = _clip01(p.x / g.total_length)
                obs[i + 2] = 1.0 if p.x >= g.active_start - 1e-9 else 0.0
                obs[i + 3] = _clip01(max(0.0, g.active_end - p.x) / c.nominal_speed / 20.0)
                obs[i + 4] = 1.0 if icp is None or not math.isfinite(icp.time) \
                    else _clip01(icp.time / 20.0)
                obs[i + 5] = _clip01((t - p.arrival_time) / 60.0)
                obs[i + 6] = _clip01(p.processing_time / 5.0)
                obs[i + 7] = 1.0 if p.x >= g.active_end - 1e-9 else 0.0
            i += SLOT_FEATURES
    obs[i] = _clip01(unfinished / 20.0)
    return obs
