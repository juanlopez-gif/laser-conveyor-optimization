"""Set-of-tokens observation of a decision state (strong-RL plan, phase 2).

The state is encoded as an **unordered set of tokens** of four types. No token carries
a position in a list or a conveyor index, so a model without positional encoding is
invariant to the order of the parts and to the labels of the conveyors, and the same
model accepts any number of parts and conveyors.

Token types and features (``docs/methods/rl2.md`` §2 has the full table)
----------------------------------------------------------------------
* **part** (one per arrived, unfinished part: waiting in the buffer, on the belt,
  or being processed), ``PART_F`` features: status one-hot (waiting, moving, ready,
  processing), position, distance and time to the zone end, interception time from
  the laser and a reachable flag, waiting time, processing time, remaining processing
  time, blocking flag, rank on its belt and head flag, candidate flag, committed
  flag, offset from the laser in x and y, and the state of its belt (speed, running,
  blocked).
* **conveyor** (one per belt), ``CONV_F``: offset in y from the laser, nominal and
  current speed, running, stopped for processing, blocked, buffer length, parts on
  the belt, arrival rate and offered load (arrival rate × mean processing time).
* **laser** (one), ``LASER_F``: position, state one-hot (idle, heading, busy),
  remaining processing time, committed target offset.
* **context** (one), ``CTX_F``: laser speeds and motion model, mean processing time,
  total offered load, geometry, pre-positioning mode one-hot, blocking-penalty
  weights λ_N and λ_T, and whether arrivals are still coming.

Scales: lengths are divided by the belt length L, speeds by ``V_REF`` = 3 cm/s (the
real belt), and times go through ``slog(t) = sign(t) log(1 + |t| / T_REF)`` with
``T_REF`` = 10 s, so long waits at heavy load stay bounded but distinguishable.
Features of the real cell lie roughly in [-3, 3].

The encoder reads a :class:`fastsim.FastSim` at an open decision. The fixed-step
engine will need an adapter with the same features (phase 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastsim.core import (BUSY, HEADING, IDLE, MOVING, PROCESSING, READY, WAITING,
                          FastSim)

V_REF = 3.0         # cm/s, belt speed of the real cell
T_REF = 10.0        # s, time scale of slog
UNREACHABLE = 5.0   # slog value used for "no meeting point"

PART_F = 23
CONV_F = 11
LASER_F = 9
CTX_F = 17

MODES = ("none", "when_idle", "always")


def slog(t: float) -> float:
    return math.copysign(math.log1p(abs(t) / T_REF), t)


def mean_processing_time(cfg) -> float:
    p = cfg.processing
    if p.distribution == "uniform":
        return 0.5 * (p.uniform_low + p.uniform_high)
    if p.distribution == "normal":
        return p.normal_mean
    return p.time


def arrival_rates(cfg) -> list[float]:
    """Mean arrivals per second per conveyor (0 if unknown, e.g. explicit lists)."""
    a, n = cfg.arrivals, len(cfg.conveyors)
    if a.mode == "stochastic":
        return [1.0 / m if m > 0 else 0.0 for m in a.mean_interarrival]
    if a.mode == "deterministic" and a.intervals:
        return [1.0 / m if m > 0 else 0.0 for m in a.intervals]
    return [0.0] * n


@dataclass
class Observation:
    """Tokens of one decision state. ``part_ids[i]`` is the simulator index of part
    token i; ``cand`` marks the candidates of the open decision."""

    part: np.ndarray      # (P, PART_F) float32
    conv: np.ndarray      # (N, CONV_F)
    laser: np.ndarray     # (LASER_F,)
    ctx: np.ndarray       # (CTX_F,)
    cand: np.ndarray      # (P,) bool
    part_ids: list[int]

    def candidate_ids(self) -> list[int]:
        return [j for j, c in zip(self.part_ids, self.cand) if c]


def encode(sim: FastSim, candidates=None) -> Observation:
    """Tokens of the current state of ``sim`` (normally at an open decision)."""
    cands = set(sim.candidates() if candidates is None else candidates)
    cfg, L, t = sim.cfg, sim.L, sim.t
    ws = sim.ws
    H = max(ws.y_max - ws.y_min, 1e-9)
    lx, ly = sim.lx, sim.ly
    rem_proc = max(sim.proc_end - t, 0.0) if sim.lstate == BUSY else 0.0
    p_mean = mean_processing_time(cfg)
    rates = arrival_rates(cfg)

    rows, ids, cflags = [], [], []
    for k in range(sim.n):
        vk = sim.v[k]
        cur = sim.speed(k)
        running = 1.0 if sim.running(k) else 0.0
        blk = 1.0 if sim.s_blk[k] else 0.0
        dy = (sim.yk[k] - ly) / H
        rank = 0
        # belt parts downstream first, then the buffer (FIFO): the natural belt order
        for j in list(sim.belt[k]) + list(sim.buf[k]):
            st = sim.pst[j]
            if st not in (WAITING, MOVING, READY, PROCESSING):
                continue
            x = sim.x(j)
            to_b = max(sim.b - x, 0.0)
            if st in (MOVING, READY):
                icp = sim.intercept(j)
            else:
                icp = None
            reach = icp is not None
            itime = slog(icp.time) if reach else UNREACHABLE
            blocking = 1.0 if (sim.pB[j] is not None and sim.pS[j] is None) else 0.0
            unstarted = st != PROCESSING
            is_cand = j in cands
            row = [
                1.0 if st == WAITING else 0.0,
                1.0 if st == MOVING else 0.0,
                1.0 if st == READY else 0.0,
                1.0 if st == PROCESSING else 0.0,
                x / L,
                to_b / L,
                slog(to_b / vk) if vk > 0 else UNREACHABLE,
                itime,
                1.0 if reach else 0.0,
                slog(t - sim.pr[j]),
                sim.pp[j] / p_mean if p_mean > 0 else 0.0,
                slog(rem_proc) if st == PROCESSING else 0.0,
                blocking,
                rank / 5.0 if unstarted else 0.0,
                1.0 if (unstarted and rank == 0) else 0.0,
                1.0 if is_cand else 0.0,
                1.0 if j == sim.assigned else 0.0,
                (x - lx) / L,
                dy,
                vk / V_REF,
                cur / V_REF,
                running,
                blk,
            ]
            if unstarted:
                rank += 1
            rows.append(row)
            ids.append(j)
            cflags.append(is_cand)

    conv = []
    for k in range(sim.n):
        on_belt = sum(1 for j in sim.belt[k] if sim.pst[j] in (MOVING, READY, PROCESSING))
        conv.append([
            (sim.yk[k] - ly) / H,
            sim.v[k] / V_REF,
            sim.speed(k) / V_REF,
            1.0 if sim.running(k) else 0.0,
            1.0 if sim.s_proc[k] else 0.0,
            1.0 if sim.s_blk[k] else 0.0,
            len(sim.buf[k]) / 5.0,
            on_belt / 5.0,
            rates[k] * T_REF,
            rates[k] * p_mean,
            1.0 if any(sim.pk[j] == k for j in cands) else 0.0,
        ])

    tgt = sim.assigned
    has_t = tgt >= 0 and sim.lstate != IDLE
    laser = [
        lx / L,
        (ly - ws.y_min) / H,
        1.0 if sim.lstate == IDLE else 0.0,
        1.0 if sim.lstate == HEADING else 0.0,
        1.0 if sim.lstate == BUSY else 0.0,
        slog(rem_proc),
        1.0 if has_t else 0.0,
        (sim.tx - lx) / L if has_t else 0.0,
        (sim.ty - ly) / H if has_t else 0.0,
    ]

    lam_n = cfg.blocking_penalty_per_event
    lam_t = cfg.blocking_penalty_per_second
    arrivals_left = 1.0 if sim.next_arr < len(sim.arr) else 0.0
    ctx = [
        sim.vx / (10 * V_REF),
        sim.vy / (10 * V_REF),
        1.0 if sim.motion == "simultaneous" else 0.0,
        p_mean / T_REF,
        sum(rates) * p_mean,
        sim.a / L,
        sim.b / L,
        sim.spacing / L,
        H / L,
        *[1.0 if sim.mode == m else 0.0 for m in MODES],
        slog(lam_n),
        lam_t,
        arrivals_left,
        len(rows) / 10.0,
        len(cands) / 5.0,
    ]

    part = np.asarray(rows, dtype=np.float32).reshape(-1, PART_F)
    return Observation(part=part,
                       conv=np.asarray(conv, dtype=np.float32).reshape(-1, CONV_F),
                       laser=np.asarray(laser, dtype=np.float32),
                       ctx=np.asarray(ctx, dtype=np.float32),
                       cand=np.asarray(cflags, dtype=bool),
                       part_ids=ids)
