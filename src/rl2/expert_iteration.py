"""Expert iteration for the Transformer dispatcher (strong-RL plan, phase 3).

Approximate policy iteration with a search teacher (``rl2.search``):

* **Iteration 0:** the teacher is the rollout over ``nearest_job``, and it also acts
  (ε-random for coverage). The network ``net_0`` is cloned from it, and its value
  head is trained on the realized cost-to-go.
* **Iteration k ≥ 1:** the teacher is the rollout over ``net_{k-1}``, i.e. the
  ``rl_rollout`` hybrid. The acting policy is ``net_{k-1}`` (the teacher acts with
  probability β, DAgger-style). Every decision with ≥ 2 candidates is labelled by
  the teacher. ``net_k`` starts from ``net_{k-1}`` and is trained on a replay buffer
  of the last ``--replay`` iterations.
* **Losses:** cross-entropy to the softmin of the teacher's look-ahead costs
  (temperature ``--temp`` seconds), plus ``--value-coef`` × MSE of the normalized
  cost-to-go, where cost-to-go = remaining Σ F + λ_N ΔN_B + λ_T ΔT_B.
* **Gating:** validation on four groups (``--val-mode``): the real cell and the heavy
  example (seeds 2000+), 2n general cells (seeds 2500+) and, since 0.7.11, the real
  cell with bursty arrivals (``real_burst``, seeds 2000+). Per group, the mean
  paired delta of penalized flow time vs ``nearest_job``; the score is the mean of
  the groups.
  ``net_k`` is kept as the best model only if its score improves. The teacher of
  each iteration is scored too: it must beat the network it is built on.
* **Domain randomization** per training episode (seeds 100000+): the general problem
  (N belts with their own speeds and arrival rates, processing time, laser speeds,
  geometry, penalties, stop rule), plus the real cell and the heavy example; see
  :data:`GENERAL` and :func:`sample_config`. Since 0.7.11 half the training cells
  get bursty arrivals and some general cells a slow head (:func:`draw_regime`).

Usage::

    python experiments/rl2_train.py --tag rl2_gen_s --iters 6 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEAVY = str(ROOT / "config/examples/heavy_load.yaml")
TRAIN_SEED_BASE = 100_000
VAL_SEED_BASE = 2000


# ------------------------------------------------------------- scenarios
#: Ranges of the general cell (agreed with the user, 2026-11-20). Factors multiply the
#: real-cell value. Every draw is independent.
GENERAL = {
    "n_conveyors": (1, 6),               # integer, uniform
    "belt_speed": (1.5, 6.0),            # cm/s, independent per belt
    "utilization": (0.3, 0.95),          # sum_k lambda_k * mean p  (processing load)
    "rate_weight": (0.5, 1.5),           # relative share of each belt in the arrivals
    "processing_mean": (1.5, 4.0),       # s
    "processing_dists": ("constant", "uniform", "normal"),
    "laser_speed": (5.0, 20.0),          # cm/s, vx and vy independent
    "p_sequential": 0.3,                 # axes one after the other
    "geometry_factor": (0.5, 1.5),       # entry, active, exit, belt spacing, part spacing
    "lambda_n": (0.0, 20.0),             # s per blocking event
    "lambda_t": (0.0, 2.0),              # s per blocked conveyor-second
    "p_all_conveyors_stop": 0.3,
    # observation level of a training cell (0.7.9): the design camera (belts), the
    # laser zone only (zone; forces pre-positioning none) or the hidden upstream queue
    # (full), so one network serves the treatments A/B/C of docs/plans/papers.md
    "p_observation": {"belts": 0.7, "zone": 0.15, "full": 0.15},
    # arrival regime and head speed of a training cell (0.7.11, owner's decision after
    # the information pilot): half the cells get bursty arrivals (on/off Poisson source
    # with the same mean rate, simulation/arrivals.py) and 30 % of the general cells a
    # slow head below the 0.7.2 range. Drawn from a separate random stream
    # (draw_regime), so every other draw of a cell, and the validation cells (no draw),
    # are unchanged.
    "p_burst": 0.5,
    "burst_on_fraction": (0.2, 1.0),     # fraction of time the source is on
    "burst_on_mean": (10.0, 40.0),       # s, mean length of an on-period
    "p_slow": 0.3,
    "slow_laser_speed": (2.5, 5.0),      # cm/s, vx and vy independent (general cells)
}

#: bursty setting of the validation group ``real_burst`` and of the information bench
BURST_VAL = ("arrivals.stochastic.distribution=burst",
             "arrivals.stochastic.burst.on_fraction=0.3",
             "arrivals.stochastic.burst.on_mean=20")


def draw_regime(seed: int, G: dict | None = None) -> dict:
    """Arrival regime and head speed of training cell ``seed`` (0.7.11), from its own
    random stream: ``burst`` = (on_fraction, on_mean) or None, ``slow`` = (vx, vy) or
    None. Every draw is made so the stream is the same whatever the outcome."""
    G = GENERAL if G is None else G
    rng = random.Random(f"rl2regime/{seed}")
    u_b, f, tau = rng.random(), rng.uniform(*G["burst_on_fraction"]), rng.uniform(*G["burst_on_mean"])
    u_s, vx, vy = rng.random(), rng.uniform(*G["slow_laser_speed"]), rng.uniform(*G["slow_laser_speed"])
    return {"burst": (round(f, 4), round(tau, 4)) if u_b < G["p_burst"] else None,
            "slow": (round(vx, 4), round(vy, 4)) if u_s < G["p_slow"] else None}


def regime_name(regime: dict | None) -> str:
    if not regime or not (regime.get("burst") or regime.get("slow")):
        return "regular"
    return "+".join(k for k in ("burst", "slow") if regime.get(k))


def _burst_items(burst) -> list[str]:
    f, tau = burst
    return ["arrivals.stochastic.distribution=burst",
            f"arrivals.stochastic.burst.on_fraction={f}",
            f"arrivals.stochastic.burst.on_mean={tau}"]


def draw_observation(seed: int, p_obs: dict | None) -> str | None:
    """Observation level of training cell ``seed`` from its own random stream (so the
    other draws of the cell are unchanged), or None when ``p_obs`` is None."""
    if not p_obs:
        return None
    rng = random.Random(f"rl2obs/{seed}")
    u, acc = rng.random(), 0.0
    for level, prob in p_obs.items():
        acc += prob
        if u < acc:
            return level
    return next(iter(p_obs))


def general_config(seed: int, p_none: float = 0.5, mode: str | None = None,
                   extra: tuple = (), p_obs: dict | None = None,
                   regime: dict | None = None):
    """A random cell of the general problem (ranges in :data:`GENERAL`).

    * N conveyors at evenly spaced y; speed v_k per belt.
    * Poisson arrivals with rate lambda_k per belt: the total processing load
      rho = sum_k lambda_k * mean(p) is drawn first, then split between belts with
      random weights, so cells are neither empty nor hopelessly overloaded.
    * Processing time: constant, uniform (+-30 %) or normal (sigma = 15 %, min 50 %).
    * Laser speeds vx, vy; motion model; geometry; blocking-penalty weights; conveyor
      stop rule; pre-positioning ``none`` (probability ``p_none``) or ``when_idle``.
    * Observation level (0.7.9): drawn from ``p_obs`` (``GENERAL["p_observation"]`` for
      training cells) in a separate random stream; ``zone`` forces ``none``. None (the
      validation cells) keeps the default ``belts``.
    * Regime (0.7.11): ``regime`` from :func:`draw_regime` (training cells) may give
      bursty arrivals and/or a slow head (vx, vy replaced); None (the validation
      cells) keeps Poisson arrivals and the laser speeds of the main draw.

    ``extra`` are further ``key=value`` overrides. Returns ``(config, description)``.
    """
    from simulation.config import load_config
    G = GENERAL
    rng = random.Random(f"rl2gen/{seed}")
    u = rng.uniform
    n = rng.randint(*G["n_conveyors"])
    v = [round(u(*G["belt_speed"]), 4) for _ in range(n)]
    pm = u(*G["processing_mean"])
    rho = u(*G["utilization"])
    w = [u(*G["rate_weight"]) for _ in range(n)]
    lam = [rho / pm * wk / sum(w) for wk in w]
    means = [round(1.0 / x, 4) for x in lam]
    dist = rng.choice(G["processing_dists"])
    vx, vy = u(*G["laser_speed"]), u(*G["laser_speed"])
    motion = "sequential" if rng.random() < G["p_sequential"] else "simultaneous"
    gf = [u(*G["geometry_factor"]) for _ in range(5)]
    ln, lt = u(*G["lambda_n"]), u(*G["lambda_t"])
    stop = ("all_conveyors" if rng.random() < G["p_all_conveyors_stop"]
            else "processing_conveyor_only")
    if mode is None:
        mode = "none" if rng.random() < p_none else "when_idle"
    obs = draw_observation(seed, p_obs)
    if obs == "zone":
        mode = "none"                     # the zone level cannot pre-position
    burst = slow = None
    if regime:
        burst, slow = regime.get("burst"), regime.get("slow")
        if slow:
            vx, vy = slow
    items = ([f"simulation.observation={obs}"] if obs else []) + (
        _burst_items(burst) if burst else []) + [
        "arrivals.mode=stochastic", f"simulation.seed={seed}", f"laser.prepositioning={mode}",
        f"conveyors.count={n}", "conveyors.y_positions=null",
        f"conveyors.spacing={12.0 * gf[3]:.4f}",
        f"conveyors.speed=[{','.join(map(str, v))}]",
        f"arrivals.stochastic.mean_interarrival=[{','.join(map(str, means))}]",
        f"processing.distribution={dist}", f"processing.time={pm:.4f}",
        f"processing.uniform.low={0.7 * pm:.4f}", f"processing.uniform.high={1.3 * pm:.4f}",
        f"processing.normal.mean={pm:.4f}", f"processing.normal.std={0.15 * pm:.4f}",
        f"processing.normal.min={0.5 * pm:.4f}",
        f"laser.speed_x={vx:.4f}", f"laser.speed_y={vy:.4f}", f"laser.motion_model={motion}",
        f"geometry.entry_length={10.0 * gf[0]:.4f}", f"geometry.active_length={35.0 * gf[1]:.4f}",
        f"geometry.exit_length={10.0 * gf[2]:.4f}",
        f"simulation.min_part_spacing={4.0 * gf[4]:.4f}",
        f"objective.blocking_penalty.per_event={ln:.4f}",
        f"objective.blocking_penalty.per_second={lt:.4f}",
        f"simulation.conveyor_stop_policy={stop}",
    ] + list(extra)
    desc = {"n": n, "v": v, "rho": rho, "p_mean": pm, "dist": dist, "vx": vx, "vy": vy,
            "motion": motion, "mode": mode, "stop": stop, "lambda_n": ln, "lambda_t": lt,
            "obs": obs or "belts", "regime": regime_name(regime)}
    return load_config(None, set_items=items), desc


def sample_config(seed: int, p_real: float = 0.25, p_heavy: float = 0.10,
                  p_none: float = 0.5, extra: tuple = ()):
    """Training cell for one episode.

    * With probability ``p_real``: the real cell (default config), so the general model
      keeps its quality there.
    * With probability ``p_heavy``: the heavy-load example (overloaded 3-belt cell).
    * Otherwise: a general cell, :func:`general_config`.
    * Mode: ``none`` with probability ``p_none``, else ``when_idle`` (0.7.7: 50/50;
      ``when_idle`` is the design mode of the cell, ``none`` the variant without entry
      detection).
    * Observation level (0.7.9): ``GENERAL["p_observation"]`` (belts 70 %, zone 15 %,
      full 15 %; ``zone`` forces ``none``), from a separate random stream.
    * Regime (0.7.11): :func:`draw_regime`; general cells may get bursty arrivals
      and/or a slow head, the real and heavy cells bursty arrivals only (their head
      is the cell's). The kind string is ``group/mode/observation/regime``.
    """
    from simulation.config import load_config
    rng = random.Random(f"rl2cfg/{seed}")
    u = rng.random()
    p_obs = GENERAL["p_observation"]
    regime = draw_regime(seed)
    if u >= p_real + p_heavy:
        cfg, d = general_config(seed, p_none, extra=extra, p_obs=p_obs, regime=regime)
        return cfg, f"general/{d['mode']}/{d['obs']}/{d['regime']}"
    mode = "none" if rng.random() < p_none else "when_idle"
    obs = draw_observation(seed, p_obs)
    if obs == "zone":
        mode = "none"
    regime = {"burst": regime["burst"]}
    items = [f"simulation.observation={obs}", "arrivals.mode=stochastic",
             f"simulation.seed={seed}", f"laser.prepositioning={mode}"] + (
             _burst_items(regime["burst"]) if regime["burst"] else []) + list(extra)
    kind = f"{mode}/{obs}/{regime_name(regime)}"
    if u < p_real:
        return load_config(None, set_items=items), f"real/{kind}"
    return load_config(HEAVY, set_items=items), f"heavy/{kind}"


VAL_GENERAL_SEED_BASE = 2500
#: validation groups: the real cell, the heavy example, 2n general cells and (0.7.11)
#: the real cell with bursty arrivals (``BURST_VAL``), seeds 2000+ like the real cell
VAL_GROUPS = ("real", "heavy", "general", "real_burst")
#: validation modes (0.7.7): ``design`` = the real cell and the heavy example with
#: ``when_idle`` (the design mode, docs/assumptions.md C4) and general cells with the
#: mode drawn 50/50; ``none`` = every cell with ``none`` (the validation of 0.7.2–0.7.6)
VAL_MODES = ("design", "none")


def val_config(group: str, seed: int, mode: str = "design", extra: tuple = ()):
    """Validation cell: the real cell, the heavy example, a general cell (seeds
    2500+, drawn by :func:`general_config`, Poisson arrivals, main laser draw) or the
    real cell with bursty arrivals (``real_burst``, 0.7.11). ``mode`` is one of
    :data:`VAL_MODES`."""
    from simulation.config import load_config
    if mode not in VAL_MODES:
        raise ValueError(f"validation mode must be one of {VAL_MODES}, not {mode!r}")
    if group == "general":
        return general_config(seed, mode=None if mode == "design" else "none",
                              extra=extra)[0]
    if group not in VAL_GROUPS:
        raise ValueError(f"validation group must be one of {VAL_GROUPS}, not {group!r}")
    mode = "when_idle" if mode == "design" else "none"
    burst = list(BURST_VAL) if group == "real_burst" else []
    return load_config(HEAVY if group == "heavy" else None,
                       set_items=["arrivals.mode=stochastic", f"simulation.seed={seed}",
                                  f"laser.prepositioning={mode}"] + burst + list(extra))


def _extra(job: dict) -> tuple:
    """Config overrides carried by a job (``horizon``: shorter episodes for quick runs)."""
    h = job.get("horizon")
    return (f"simulation.arrival_horizon={h}",) if h else ()


# ---------------------------------------------------------------- workers
_NETS: dict = {}


def _net(path):
    if path is None:
        return None
    if path not in _NETS:
        from rl2.model import load
        _NETS.clear()
        _NETS[path] = load(path)
    return _NETS[path]


_SCALES: dict = {}


def _scale(path):
    """Value-head scale saved with the model at ``path`` (cached per path)."""
    if path is None:
        return None
    if path not in _SCALES:
        from rl2.model import load_scale
        _SCALES.clear()
        _SCALES[path] = load_scale(path)
    return _SCALES[path]


def _teacher(job: dict, net, key: str):
    """The search teacher of a job: known-parts look-ahead, or (``la`` =
    (horizon_s, scenarios), 0.7.6) the value-truncated look-ahead over the network."""
    from rl2.search import SearchTeacher
    la = tuple(job.get("la", ()))
    horizon, scen, cut, window = la + (0.0, 0, "time", 30.0)[len(la):]
    return SearchTeacher(net, k=job["k"], w_pen=job["w_pen"], key=key, horizon_s=horizon,
                         scenarios=scen, scale=_scale(job["net"]), cut=cut, window_s=window)


def _init_worker():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "src"))


def play_episode(job: dict) -> dict:
    """One labelled training episode."""
    from fastsim import FastSim
    from rl2.features import encode
    from rl2.model import NetPolicy

    seed = job["seed"]
    cfg, kind = sample_config(seed, **job.get("mix", {}), extra=_extra(job))
    net = _net(job["net"])
    teacher = _teacher(job, net, str(seed))
    actor = NetPolicy(net) if net is not None else None
    rng = random.Random(f"rl2act/{seed}")
    sim = FastSim(cfg)
    samples, marks = [], []
    t0 = time.perf_counter()
    while sim.run_until_decision():
        cands = sim.candidates()
        if len(cands) == 1:
            sim.apply(cands[0])
            continue
        obs = encode(sim, cands)
        costs = teacher.costs(sim, cands)
        cvec = np.full(len(obs.part_ids), np.nan, np.float32)
        best = teacher.choose(costs)
        for i, j in enumerate(obs.part_ids):
            if j in costs:
                # >= -1e-9 by the tie tolerance of choose(); clip the round-off
                cvec[i] = max(costs[j] - costs[best], 0.0)
        samples.append([obs, cvec])
        marks.append((sim.wip_area, sim.n_block_events, sim.blocked_area))
        if actor is None or rng.random() < job["beta"]:
            choice = rng.choice(cands) if rng.random() < job["eps"] else best
        else:
            choice = actor(sim, cands)
        sim.apply(choice)
    s = sim.summary()
    ln, lt = cfg.blocking_penalty_per_event, cfg.blocking_penalty_per_second
    for smp, (w, nb, tb) in zip(samples, marks):
        smp.append((sim.wip_area - w) + ln * (sim.n_block_events - nb)
                   + lt * (sim.blocked_area - tb))
    return {"samples": samples, "kind": kind, "seed": seed,
            "cost": s["penalized_flow_time"], "secs": time.perf_counter() - t0,
            "lookaheads": teacher.lookaheads}


def val_episode(job: dict) -> dict:
    from fastsim import POLICIES, FastSim
    from rl2.model import NetPolicy
    cfg = val_config(job["load"], job["seed"], job.get("mode", "design"), _extra(job))
    net = _net(job["net"])
    if job["who"] == "net":
        pol = NetPolicy(net)
    elif job["who"] == "teacher":
        pol = _teacher(job, net, str(job["seed"]))
    else:
        pol = POLICIES[job["who"]]
    t0 = time.perf_counter()
    s = FastSim(cfg).run(pol)
    return {**{k: job[k] for k in ("load", "seed", "who")},
            "cost": s["penalized_flow_time"], "secs": time.perf_counter() - t0}


# --------------------------------------------------------------- training
def _targets(cvecs, temp: float, P: int):
    """Soft targets (B, P) from relative look-ahead costs (NaN = not searched)."""
    import torch
    tgt = torch.zeros(len(cvecs), P)
    for i, c in enumerate(cvecs):
        ok = ~np.isnan(c)
        z = np.zeros(len(c), np.float64)
        if temp > 0:
            z[ok] = np.exp(-c[ok].astype(np.float64) / temp)
        else:
            z[ok] = (c[ok] <= 1e-9)
        z /= z.sum()
        tgt[i, :len(c)] = torch.from_numpy(z.astype(np.float32))
    return tgt


def train(net, data: list, scale, epochs: int, lr: float, batch: int, temp: float,
          value_coef: float, seed: int, threads: int, device: str = "cpu") -> dict:
    """Supervised update of ``net`` on (tokens, relative costs, cost-to-go) samples.
    Runs on ``device`` (``cpu``, ``cuda``, ``mps`` or ``auto``); the network is moved
    back to the CPU at the end (workers and saving use the CPU)."""
    import torch
    from rl2.model import collate, resolve_device
    dev = resolve_device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(data) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    hist = []
    net.train()
    for ep in range(epochs):
        order = rng.permutation(len(data))
        tot = {"policy": 0.0, "value": 0.0, "acc": 0.0, "n": 0}
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            b = collate([data[i][0] for i in idx])
            P = b.part.shape[1]
            tgt = _targets([data[i][1] for i in idx], temp, P).to(dev)
            vt = torch.tensor([scale.normalize(data[i][2]) for i in idx],
                              dtype=torch.float32, device=dev)
            b = b.to(dev)
            logits, value = net(b)
            logp = torch.log_softmax(logits, -1)
            logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
            lp = -(tgt * logp).sum(-1).mean()
            lv = torch.nn.functional.mse_loss(value, vt)
            loss = lp + value_coef * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            n = len(idx)
            best = tgt.argmax(-1)
            tot["policy"] += lp.item() * n
            tot["value"] += lv.item() * n
            tot["acc"] += (logits.argmax(-1) == best).float().sum().item()
            tot["n"] += n
        hist.append({k: tot[k] / tot["n"] for k in ("policy", "value", "acc")})
        print(f"    epoch {ep + 1}/{epochs}: policy {hist[-1]['policy']:.4f} "
              f"value {hist[-1]['value']:.4f} agree {hist[-1]['acc']:.3f}", flush=True)
    net.cpu().eval()
    return {"epochs": hist, "device": str(dev)}


# ------------------------------------------------------------- validation
def validate(ex, net_path, n: int, who: list[str], k: int, w_pen: float,
             horizon: float | None = None, la: tuple = (0.0, 0),
             val_mode: str = "design") -> dict:
    """Paired delta vs ``nearest_job`` per group (real, heavy, general: 2n cells,
    real_burst); score = mean of the group means. ``la`` = (horizon_s, scenarios, cut,
    window_s) of the cost-to-go teacher (0.7.6; (0, 0) = known-parts look-ahead);
    ``val_mode`` is one of :data:`VAL_MODES` (0.7.7)."""
    seeds = {g: VAL_GENERAL_SEED_BASE if g == "general" else VAL_SEED_BASE for g in VAL_GROUPS}
    jobs = [{"load": g, "seed": seeds[g] + i, "who": w, "net": net_path, "k": k,
             "w_pen": w_pen, "horizon": horizon, "la": la, "mode": val_mode}
            for w in who + ["nearest_job"] for g in VAL_GROUPS
            for i in range(n * (2 if g == "general" else 1))]
    res = list(ex.map(val_episode, jobs, chunksize=1))
    ref = {(r["load"], r["seed"]): r["cost"] for r in res if r["who"] == "nearest_job"}
    out = {}
    for w in who:
        for g in VAL_GROUPS:
            rs = [r["cost"] / ref[(g, r["seed"])] - 1 for r in res
                  if r["who"] == w and r["load"] == g]
            mu = float(np.mean(rs))
            ci = 1.96 * float(np.std(rs, ddof=1)) / math.sqrt(len(rs))
            secs = float(np.mean([r["secs"] for r in res if r["who"] == w and r["load"] == g]))
            out[f"{w}/{g}"] = {"delta": mu, "ci": ci, "secs": secs, "n": len(rs),
                               "wins": int(sum(x < 0 for x in rs))}
        out[f"{w}/score"] = float(np.mean([out[f"{w}/{g}"]["delta"] for g in VAL_GROUPS]))
    return out


def _fmt(v: dict, w: str) -> str:
    return f"{w}: " + ", ".join(
        f"{g} {100 * v[w + '/' + g]['delta']:+.2f} +-{100 * v[w + '/' + g]['ci']:.2f} %"
        for g in VAL_GROUPS)


# ------------------------------------------------------------------- main
#: Arguments that may change when a run is resumed (machine and stopping rule). All
#: others are taken from the saved run, so a resumed run is the same experiment.
RESUMABLE_ARGS = ("iters", "workers", "threads", "device", "patience", "resume")


def _rel(path) -> str | None:
    """Path relative to the repository (portable between machines)."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _abs(path) -> str | None:
    return None if path is None else str((ROOT / path) if not Path(path).is_absolute() else path)


def _save_state(path: Path, state: dict) -> None:
    """Atomic write (a crash during the write never leaves a broken state file)."""
    import pickle
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main(argv=None) -> int:
    import pickle

    import torch
    from rl2.model import SIZES, DispatchNet, RunningScale, load, resolve_device, save

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="rl2_gen_s")
    ap.add_argument("--size", default="S", choices=sorted(SIZES))
    ap.add_argument("--iters", type=int, default=6, help="iterations after iteration 0")
    ap.add_argument("--episodes0", type=int, default=800)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--epochs0", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0, help="softmin temperature [s]")
    ap.add_argument("--value-coef", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--w-pen", type=float, default=0.0,
                    help="weight of the blocking penalty in the look-ahead cost")
    ap.add_argument("--la-horizon", type=float, default=0.0,
                    help="cost-to-go look-ahead over the network (rl2.search): cut after "
                         "this many seconds and add the value head; inf = run until the "
                         "synthetic window drains, no value head; 0 = known-parts "
                         "look-ahead")
    ap.add_argument("--scenarios", type=int, default=0,
                    help="sampled future-arrival scenarios per look-ahead (0 = none)")
    ap.add_argument("--la-cut", default="time", choices=["time", "decision"],
                    help="where a finite --la-horizon cuts: at that instant, or at the "
                         "first decision after it")
    ap.add_argument("--la-window", type=float, default=30.0,
                    help="synthetic-arrival window [s] of --la-horizon inf")
    ap.add_argument("--beta", type=float, default=0.3, help="teacher-acts probability (k >= 1)")
    ap.add_argument("--eps", type=float, default=0.1, help="random action when the teacher acts")
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--val", type=int, default=20, help="validation seeds per load")
    ap.add_argument("--val-mode", default="design", choices=list(VAL_MODES),
                    help="design: real and heavy cells with when_idle, general cells "
                         "with the mode drawn; none: every cell with none (0.7.2-0.7.6)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="simulation worker processes (default: all logical CPUs)")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads for training")
    ap.add_argument("--device", default="auto",
                    help="training device: auto (CUDA > MPS > CPU), cuda, cuda:1, mps, cpu")
    ap.add_argument("--horizon", type=float, default=None,
                    help="override the arrival horizon of every episode [s] (quick runs/tests)")
    ap.add_argument("--init", default=None, help="start from this model (skips iteration 0)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run --tag from its last completed iteration")
    args = ap.parse_args(argv)

    mdir = ROOT / "models"
    rdir = ROOT / "results" / "experiments" / args.tag
    rdir.mkdir(parents=True, exist_ok=True)
    state_path = rdir / "state.pkl"
    logp = mdir / f"{args.tag}_training.json"

    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"--resume: no saved state at {state_path}")
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        saved = st["log"]["args"]
        for k, v in saved.items():                       # same experiment
            if k not in RESUMABLE_ARGS:
                setattr(args, k, v)
        for k, v in (("la_horizon", 0.0), ("scenarios", 0), ("la_cut", "time"),
                     ("la_window", 30.0), ("val_mode", "none")):  # runs before 0.7.6/0.7.7
            if k not in saved:
                setattr(args, k, v)
        log = st["log"]
        log["args"].update({k: getattr(args, k) for k in RESUMABLE_ARGS})
        log.setdefault("resumed", []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
        scale = RunningScale()
        scale.n, scale.mean, scale.m2 = st["scale"]
        buffers, seed_next = st["buffers"], st["seed_next"]
        best_path, best_score, stall = _abs(st["best"]), st["best_score"], st["stall"]
        prev, start = _abs(st["prev"]), st["next_it"]
        print(f"resuming {args.tag} at iteration {start} "
              f"(buffer {sum(len(b) for b in buffers)} samples, best {st['best']})", flush=True)
    else:
        log = {"tag": args.tag, "args": vars(args), "python": platform.python_version(),
               "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "iterations": []}
        scale = RunningScale()
        buffers, seed_next = [], TRAIN_SEED_BASE
        best_path, best_score, stall = None, math.inf, 0
        prev, start = _abs(args.init), (1 if args.init else 0)
    dev = resolve_device(args.device)
    log.setdefault("machine", []).append({
        "host": platform.node(), "cpus": os.cpu_count(), "workers": args.workers,
        "device": str(dev), "cuda": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version()})
    print(f"device {dev}, workers {args.workers}", flush=True)

    def save_log():
        logp.write_text(json.dumps(log, indent=2))

    def checkpoint(next_it):
        _save_state(state_path, {
            "log": log, "scale": (scale.n, scale.mean, scale.m2), "buffers": buffers,
            "seed_next": seed_next, "best": _rel(best_path), "best_score": best_score,
            "stall": stall, "prev": _rel(prev), "next_it": next_it})

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        if "reference" not in log:
            ref_val = validate(ex, None, args.val, ["nearest_head"], args.k, args.w_pen,
                               args.horizon, val_mode=args.val_mode)
            teacher0 = validate(ex, None, args.val, ["teacher"], args.k, args.w_pen,
                                args.horizon, val_mode=args.val_mode)
            log["reference"] = {**ref_val, **teacher0}
            print("reference", _fmt(ref_val, "nearest_head"), "|", _fmt(teacher0, "teacher"),
                  flush=True)
            save_log()
            checkpoint(start)

        for it in range(start, args.iters + 1):
            if stall >= args.patience:
                break
            t_it = time.perf_counter()
            n_ep = args.episodes0 if it == 0 else args.episodes
            la = (args.la_horizon, args.scenarios, args.la_cut, args.la_window)
            jobs = [{"seed": seed_next + e, "net": prev, "k": args.k, "w_pen": args.w_pen,
                     "beta": 1.0 if prev is None else args.beta, "eps": args.eps,
                     "horizon": args.horizon, "la": la}
                    for e in range(n_ep)]
            seed_next += n_ep
            eps = list(ex.map(play_episode, jobs, chunksize=2))
            data = [s for e in eps for s in e["samples"]]
            gen_s = time.perf_counter() - t_it
            scale.update([d[2] for d in data])      # new data only (the buffer repeats)
            buffers.append(data)
            buffers = buffers[-args.replay:]
            train_data = [s for b in buffers for s in b]
            print(f"iter {it}: {len(eps)} episodes, {len(data)} decisions "
                  f"({gen_s:.0f} s), buffer {len(train_data)}", flush=True)

            if prev is None:
                net = DispatchNet(SIZES[args.size])
                lr, epochs = args.lr, args.epochs0
            else:
                net = load(prev)
                lr, epochs = args.lr_ft, args.epochs
            t_tr = time.perf_counter()
            hist = train(net, train_data, scale, epochs, lr, args.batch, args.temp,
                         args.value_coef, seed=it, threads=args.threads, device=args.device)
            tr_s = time.perf_counter() - t_tr
            path = str(mdir / f"{args.tag}_it{it}.pt")
            save(net, path, {"iteration": it, "scale": [scale.n, scale.mean, scale.m2]})

            t_v = time.perf_counter()
            v = validate(ex, path, args.val, ["net", "teacher"], args.k, args.w_pen,
                         args.horizon, la, val_mode=args.val_mode)
            val_s = time.perf_counter() - t_v
            score = v["net/score"]
            accepted = score < best_score - 1e-6
            print(f"  {_fmt(v, 'net')} | {_fmt(v, 'teacher')} | "
                  f"{'ACCEPTED' if accepted else 'rejected'}  "
                  f"(train {tr_s:.0f} s, val {val_s:.0f} s)", flush=True)
            log["iterations"].append({
                "iteration": it, "episodes": len(eps), "decisions": len(data),
                "buffer": len(train_data), "seeds": [jobs[0]["seed"], jobs[-1]["seed"]],
                "kinds": {k: sum(1 for e in eps if e["kind"] == k)
                          for k in sorted({e["kind"] for e in eps})},
                "acting_cost_mean": float(np.mean([e["cost"] for e in eps])),
                "gen_s": gen_s, "train_s": tr_s, "val_s": val_s, "train": hist,
                "val": v, "accepted": accepted, "model": _rel(path)})
            if accepted:
                best_path, best_score, stall = path, score, 0
                save(net, mdir / f"{args.tag}.pt",
                     {"iteration": it, "scale": [scale.n, scale.mean, scale.m2]})
            else:
                stall += 1
            prev = best_path
            log["best"] = {"model": _rel(best_path), "score": best_score}
            save_log()
            checkpoint(it + 1)
            if stall >= args.patience:
                print("stopping: no validation improvement", flush=True)
                break
    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_log()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
