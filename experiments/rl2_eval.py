"""Phase 4: confirmatory evaluation on the reserved seeds (frozen protocol).

The protocol is the table "Phase 4 protocol" of ``docs/plans/papers.md`` (owner,
2026-11-22). One invocation runs, on **seeds 7000–7029** (``--seed-base 7000``,
``--seeds 30``) and with the frozen network ``models/rl2_gen_s_b.pt``:

* **Paper 2** (``info``): treatments A/B/C/D and the coverage grid
  (``simulation.visible_length`` in ``--visible-lengths`` with ``none`` and
  ``when_idle`` at every length) for ``nearest_job``, ``nearest_head``, the network and
  the 0.7.9 reference network, on the eight regimes of the pilot (load real / heavy ×
  arrivals Poisson / burst × head nominal / slow);
* **Paper 1** (``search``): in B and in C, the known-parts search over
  ``nearest_job``, the sampled-future teacher over the network (8 futures, 30 s,
  w_pen 0) and the network alone, each driving its own episode; plus, on the
  decisions of the network's episode, the per-decision latency of every policy
  (``nearest_job``, ``nearest_head``, search, teacher, network) on the **same
  states**;
* **engine check** (``engine``): the fixed-step engine at ``--engine-dt`` with the
  rules in B and C on every regime, against the fast simulator (the network has no
  engine adapter: it is validated through the fast simulator only).

Every configuration of a regime runs on the same seeds (common random numbers): every
contrast is paired. Outputs: ``results/experiments/<tag>/bench.json`` (every run) and
``tables.md``. The reserved seeds are used once: with ``--seed-base 7000`` the script
refuses to overwrite an existing ``bench.json`` (``--force`` to override). Smoke tests
use ``--seed-base 2000`` (development seeds) and a small ``--seeds``.

    python experiments/rl2_eval.py --tag phase4
    python experiments/rl2_eval.py --tag _smoke --seed-base 2000 --seeds 2 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl2.expert_iteration import BURST_VAL, HEAVY, _init_worker, val_config  # noqa: E402

TREATMENTS = {"A": ("zone", "none"), "B": ("belts", "none"),
              "C": ("belts", "when_idle"), "D": ("full", "none")}
LOADS = ("real", "heavy")
ARRIVALS = ("regular", "burst")
TRAVEL = ("nominal", "slow")
TEACHER = dict(k=6, horizon_s=math.inf, scenarios=8, window_s=30.0, w_pen=0.0)
SEARCH = dict(k=6)

_NET: dict = {}
_SCALE: dict = {}


def _net(path: str):
    if path not in _NET:
        from rl2.model import load, load_scale
        _NET[path] = load(path)
        _SCALE[path] = load_scale(path)
    return _NET[path]


def _laser_speed(load: str) -> tuple[float, float]:
    from simulation.config import load_config
    c = load_config(HEAVY if load == "heavy" else None)
    return c.laser.speed_x, c.laser.speed_y


def _extra(job: dict) -> list[str]:
    extra = [f"simulation.observation={job['obs']}", f"laser.prepositioning={job['mode']}"]
    if job.get("visible") is not None:
        extra.append(f"simulation.visible_length={job['visible']}")
    if job["arrivals"] == "burst":
        extra += list(BURST_VAL)
    if job["travel"] == "slow":
        vx, vy = _laser_speed(job["load"])
        extra += [f"laser.speed_x={vx * job['slow']:.4f}", f"laser.speed_y={vy * job['slow']:.4f}"]
    return extra


def _policies(job: dict) -> dict:
    """Named policies ``f(sim, cands) -> part`` of a job (built in the worker)."""
    from fastsim import POLICIES
    from rl2.model import NetPolicy
    from rl2.search import SearchTeacher
    key = str(job["seed"])
    pol = {"nearest_job": POLICIES["nearest_job"], "nearest_head": POLICIES["nearest_head"]}
    if job.get("net"):
        net = _net(job["net"])
        pol["net"] = NetPolicy(net)
        pol["teacher"] = SearchTeacher(net, key=key, scale=_SCALE[job["net"]], **TEACHER)
    if job.get("ref"):
        pol["net_ref"] = NetPolicy(_net(job["ref"]))
    pol["search"] = SearchTeacher(None, key=key, **SEARCH)
    return pol


def _summary(sim, s: dict) -> dict:
    return {"cost": s["penalized_flow_time"], "flow": s.get("avg_flow_time"),
            "blocks": s["blocking_events"], "blocked": s["total_blocked_time"],
            "parts": len(sim.arr), "done": sim.n_completed}


def run(job: dict) -> dict:
    """One episode. ``who``: the policy that drives it; ``probe``: also time every
    policy on the decisions of this episode (latency on the same states)."""
    from fastsim import FastSim
    cfg = val_config(job["load"], job["seed"], "none", extra=tuple(_extra(job)))
    pol = _policies(job)
    sim = FastSim(cfg)
    t0 = time.perf_counter()
    if not job.get("probe"):
        s = sim.run(pol[job["who"]])
        out = job | _summary(sim, s) | {"secs": time.perf_counter() - t0}
        if job["who"] in ("teacher", "search"):
            out["lookaheads"] = pol[job["who"]].lookaheads
        return out
    drive = pol[job["who"]]
    lat = {name: [] for name in job["probe"]}
    while sim.run_until_decision():
        cands = sim.candidates()
        if len(cands) > 1:
            for name in job["probe"]:
                t1 = time.perf_counter()
                pol[name](sim, cands)
                lat[name].append(round(1e3 * (time.perf_counter() - t1), 4))
        sim.apply(drive(sim, cands))
    s = sim.summary()
    return job | _summary(sim, s) | {"secs": time.perf_counter() - t0, "latency_ms": lat}


def run_engine(job: dict) -> dict:
    """The same instance on the fixed-step engine and on the fast simulator."""
    from fastsim import FastSim, POLICIES
    from optimization.objective_functions import OBJECTIVES
    from simulation.engine import Simulation
    extra = _extra(job) + [f"simulation.dt={job['dt']}", f"policy.name={job['who']}"]
    cfg = val_config(job["load"], job["seed"], "none", extra=tuple(extra))
    t0 = time.perf_counter()
    res = Simulation(cfg, record_trace=False).run()
    te = time.perf_counter() - t0
    order = [p.number - 1 for p in sorted((p for p in res.parts if p.start_processing_time is not None),
                                          key=lambda p: p.start_processing_time)]
    sim = FastSim(cfg)
    s = sim.run(POLICIES[job["who"]])
    return job | {"cost_engine": OBJECTIVES["penalized_flow_time"](res.summary),
                  "cost_fast": s["penalized_flow_time"], "same_order": order == sim.order,
                  "secs_engine": te, "parts": len(res.parts)}


# ------------------------------------------------------------------ statistics
def _ci(d):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return 100 * float(d.mean()) if len(d) else math.nan, math.nan
    return 100 * d.mean(), 100 * 1.96 * d.std(ddof=1) / math.sqrt(len(d))


def _cell(mu, ci):
    return f"{mu:+.2f} ±{ci:.2f}"


def _paired(a: dict, b: dict) -> tuple[float, float, int, int]:
    """Paired relative Δ of b vs a over the common seeds [%], with the count of seeds
    where b is lower."""
    seeds = sorted(set(a) & set(b))
    d = [b[s] / a[s] - 1 for s in seeds]
    mu, ci = _ci(d)
    return mu, ci, sum(x < 0 for x in d), len(seeds)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else math.nan


def _regime_name(g, a, tr):
    return f"{g} / {a} arrivals / {tr} head"


# ------------------------------------------------------------------ tables
def tables(res: dict, args) -> str:
    info, search, engine = res["info"], res["search"], res["engine"]
    lengths = list(args.visible_lengths)
    configs = list(TREATMENTS) + [f"L{lv:g}/{m}" for lv in lengths for m in ("none", "when_idle")]
    regimes = [(g, a, tr) for g in LOADS for a in ARRIVALS for tr in TRAVEL]
    L = [f"# Phase 4 confirmatory evaluation `{args.tag}`", "",
         f"Network `{args.net}` (reference `{args.ref}`), seeds {args.seed_base}–"
         f"{args.seed_base + args.seeds - 1} ({args.seeds} paired realizations per configuration), "
         f"slow head = speeds × {args.slow}, bursts `{' '.join(BURST_VAL)}`. Paired Δ of the "
         "penalized flow time, 95 % CI, negative = better; \"n/N\" = seeds where the second "
         "policy is better.", ""]

    def costs(rows, **sel):
        out = {}
        for r in rows:
            if all(r.get(k) == v for k, v in sel.items()):
                out[r["seed"]] = r["cost"]
        return out

    # ---- overview across regimes
    L += ["## Overview", "",
          "| regime | `nearest_job` B [s/part] | net B vs NJ | net C vs NJ | B vs A (net) | "
          "C vs B (net) | D vs B (net) | teacher vs net (B) | teacher vs net (C) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g, a, tr in regimes:
        sel = dict(load=g, arrivals=a, travel=tr)
        nj = costs(info, who="nearest_job", t="B", **sel)
        net = {t: costs(info, who="net", t=t, **sel) for t in TREATMENTS}
        tb = {t: costs(search, who="teacher", t=t, **sel) for t in ("B", "C")}
        cells = [f"{np.mean(list(nj.values())):.1f}" if nj else "–",
                 _cell(*_paired(nj, net["B"])[:2]), _cell(*_paired(nj, net["C"])[:2]),
                 _cell(*_paired(net["A"], net["B"])[:2]), _cell(*_paired(net["B"], net["C"])[:2]),
                 _cell(*_paired(net["B"], net["D"])[:2]),
                 _cell(*_paired(net["B"], tb["B"])[:2]), _cell(*_paired(net["C"], tb["C"])[:2])]
        L.append(f"| {_regime_name(g, a, tr)} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- per regime
    for g, a, tr in regimes:
        sel = dict(load=g, arrivals=a, travel=tr)
        rows = [r for r in info if all(r.get(k) == v for k, v in sel.items())]
        if not rows:
            continue
        ref = costs(rows, who="nearest_job", t="B")
        base = [r for r in rows if r["who"] == "nearest_job" and r["t"] == "B"]
        belt = base[0]["belt_speed"] if base else math.nan
        head = base[0]["head_span_s"] if base else math.nan
        L += [f"## {_regime_name(g, a, tr)}", "",
              f"`nearest_job` in B: {np.mean([r['cost'] for r in base]):.2f} s/part "
              f"(flow {np.mean([r['flow'] for r in base]):.2f} s, "
              f"{np.mean([r['blocks'] for r in base]):.1f} blocking events, "
              f"{np.mean([r['blocked'] for r in base]):.1f} blocked conveyor-s, "
              f"{np.mean([r['parts'] for r in base]):.0f} parts per episode). Belt speed "
              f"{belt:.2f} cm/s; head travel across the conveyors {head:.2f} s.", "",
              "### Paper 2: treatments and coverage grid (paired Δ vs `nearest_job` in B [%])", "",
              "| policy | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
        pols = [w for w in ("nearest_job", "nearest_head", "net", "net_ref")
                if any(r["who"] == w for r in rows)]
        for w in pols:
            cells = [_cell(*_paired(ref, costs(rows, who=w, t=t))[:2]) for t in configs]
            L.append(f"| `{w}` | " + " | ".join(cells) + " |")
        L += ["", "Contrasts between treatments (paired, same policy):", "",
              "| policy | B vs A | C vs B | D vs B | C vs A |", "|---|---|---|---|---|"]
        for w in pols:
            byt = {t: costs(rows, who=w, t=t) for t in TREATMENTS}
            cells = []
            for x, y in (("A", "B"), ("B", "C"), ("B", "D"), ("A", "C")):
                mu, ci, nb, n = _paired(byt[x], byt[y])
                cells.append(f"{_cell(mu, ci)} ({nb}/{n})")
            L.append(f"| `{w}` | " + " | ".join(cells) + " |")
        L += ["", "Coverage curve: `when_idle` vs `none` at the same visible length "
              "(paired); share = gain relative to the whole entry zone "
              f"(L{lengths[-1]:g}); seconds = length / belt speed:", "",
              "| policy | " + " | ".join(f"L{lv:g} ({lv / belt:.2f} s)" for lv in lengths) + " |",
              "|---|" + "---|" * len(lengths)]
        for w in pols:
            full = _paired(costs(rows, who=w, t=f"L{lengths[-1]:g}/none"),
                           costs(rows, who=w, t=f"L{lengths[-1]:g}/when_idle"))[0]
            cells = []
            for lv in lengths:
                mu, ci, nb, n = _paired(costs(rows, who=w, t=f"L{lv:g}/none"),
                                        costs(rows, who=w, t=f"L{lv:g}/when_idle"))
                share = mu / full if full else math.nan
                cells.append(f"{_cell(mu, ci)} ({100 * share:.0f} %)")
            L.append(f"| `{w}` | " + " | ".join(cells) + " |")
        L.append("")
        # ---- Paper 1
        srows = [r for r in search if all(r.get(k) == v for k, v in sel.items())]
        if srows:
            L += ["### Paper 1: search, teacher and network (paired Δ vs `nearest_job` [%]; "
                  "s / episode)", "",
                  "| policy | B | C | B vs net | C vs net | s / episode (B / C) |",
                  "|---|---|---|---|---|---|"]
            who = [w for w in ("nearest_head", "net_ref", "search", "teacher", "net")
                   if any(r["who"] == w for r in rows + srows)]
            for w in who:
                cells, vs, secs = [], [], []
                for t in ("B", "C"):
                    src = srows if w in ("search", "teacher") else rows
                    c = costs(src, who=w, t=t)
                    cells.append(_cell(*_paired(costs(rows, who="nearest_job", t=t), c)[:2]))
                    mu, ci, nb, n = _paired(costs(rows, who="net", t=t), c)
                    vs.append(f"{_cell(mu, ci)} ({nb}/{n})")
                    secs.append(f"{np.mean([r['secs'] for r in src if r['who'] == w and r['t'] == t]):.1f}")
                L.append(f"| `{w}` | " + " | ".join(cells + vs) + f" | {' / '.join(secs)} |")
            L += ["", "Per-decision latency on the decisions of the network's episode "
                  "(≥ 2 candidates, one thread), pooled over the seeds [ms]:", "",
                  "| policy | B p50 | B p95 | B max | C p50 | C p95 | C max | decisions (B / C) |",
                  "|---|---|---|---|---|---|---|---|"]
            probes = [r for r in srows if r.get("probe")]
            names = probes[0]["probe"] if probes else []
            for name in names:
                cells, nd = [], []
                for t in ("B", "C"):
                    xs = [x for r in probes if r["t"] == t for x in r["latency_ms"][name]]
                    cells += [f"{_pct(xs, .5):.2f}", f"{_pct(xs, .95):.2f}",
                              f"{max(xs) if xs else math.nan:.1f}"]
                    nd.append(str(len(xs)))
                L.append(f"| `{name}` | " + " | ".join(cells) + f" | {' / '.join(nd)} |")
            L.append("")
    # ---- engine check
    if engine:
        L += [f"## Engine check (fixed-step engine at dt = {args.engine_dt} vs the fast simulator)", "",
              "| regime | treatment | rule | J engine − J fast [%] (mean ± CI) | max |Δ| [%] | "
              "same order | s / episode (engine) |", "|---|---|---|---|---|---|---|"]
        for g, a, tr in regimes:
            for t in ("B", "C"):
                for w in ("nearest_job", "nearest_head"):
                    er = [r for r in engine if (r["load"], r["arrivals"], r["travel"], r["t"], r["who"])
                          == (g, a, tr, t, w)]
                    if not er:
                        continue
                    d = [r["cost_engine"] / r["cost_fast"] - 1 for r in er]
                    L.append(f"| {_regime_name(g, a, tr)} | {t} | `{w}` | {_cell(*_ci(d))} | "
                             f"{100 * max(abs(x) for x in d):.2f} | "
                             f"{sum(r['same_order'] for r in er)}/{len(er)} | "
                             f"{np.mean([r['secs_engine'] for r in er]):.1f} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--net", default="models/rl2_gen_s_b.pt")
    ap.add_argument("--ref", default="models/rl2_gen_s_d.pt", help="0.7.9 reference network ('' = none)")
    ap.add_argument("--tag", default="phase4")
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--slow", type=float, default=0.5)
    ap.add_argument("--visible-lengths", nargs="*", type=float, default=[0, 1, 2.5, 5, 7.5, 10])
    ap.add_argument("--parts", nargs="+", default=["info", "search", "engine"],
                    choices=["info", "search", "engine"])
    ap.add_argument("--engine-dt", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=47)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "results/experiments" / args.tag
    if args.seed_base == 7000 and (out_dir / "bench.json").exists() and not args.force:
        sys.exit(f"{out_dir / 'bench.json'} exists: the reserved seeds are used once (--force to override)")
    out_dir.mkdir(parents=True, exist_ok=True)
    from simulation.config import load_config
    geo = {}
    for g in LOADS:
        c = load_config(HEAVY if g == "heavy" else None)
        ys = [cv.y for cv in c.conveyors]
        geo[g] = {"belt_speed": float(np.mean([cv.speed for cv in c.conveyors])),
                  "head_span_s": (max(ys) - min(ys)) / c.laser.speed_y}
    regimes = [(g, a, tr) for g in LOADS for a in ARRIVALS for tr in TRAVEL]
    seeds = [args.seed_base + i for i in range(args.seeds)]
    net, ref = str(ROOT / args.net), (str(ROOT / args.ref) if args.ref else None)

    def base(g, a, tr, seed):
        geo_r = dict(geo[g])
        if tr == "slow":
            geo_r["head_span_s"] /= args.slow
        return {"load": g, "arrivals": a, "travel": tr, "slow": args.slow, "seed": seed,
                "net": net, "ref": ref} | geo_r

    configs = [{"t": t, "obs": o, "mode": m, "visible": None} for t, (o, m) in TREATMENTS.items()]
    for lv in args.visible_lengths:
        for m in ("none", "when_idle"):
            configs.append({"t": f"L{lv:g}/{m}", "obs": "belts", "mode": m, "visible": lv})
    info_pols = ["nearest_job", "nearest_head", "net"] + (["net_ref"] if ref else [])
    jobs_info = [base(g, a, tr, s) | c | {"who": w}
                 for g, a, tr in regimes for s in seeds for c in configs for w in info_pols] \
        if "info" in args.parts else []
    probe = ["nearest_job", "nearest_head", "search", "teacher", "net"]
    jobs_search = []
    if "search" in args.parts:
        for g, a, tr in regimes:
            for s in seeds:
                for t in ("B", "C"):
                    c = {"t": t, "obs": TREATMENTS[t][0], "mode": TREATMENTS[t][1], "visible": None}
                    jobs_search += [base(g, a, tr, s) | c | {"who": "teacher"},
                                    base(g, a, tr, s) | c | {"who": "search"},
                                    base(g, a, tr, s) | c | {"who": "net", "probe": probe}]
    jobs_engine = [base(g, a, tr, s) | {"t": t, "obs": TREATMENTS[t][0], "mode": TREATMENTS[t][1],
                                        "visible": None, "who": w, "dt": args.engine_dt}
                   for g, a, tr in regimes for s in seeds for t in ("B", "C")
                   for w in ("nearest_job", "nearest_head")] if "engine" in args.parts else []
    print(f"{len(jobs_info)} info runs, {len(jobs_search)} search runs, "
          f"{len(jobs_engine)} engine runs", flush=True)
    t0 = time.perf_counter()
    res = {"args": vars(args), "seeds": seeds}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        # the long search jobs first, so the pool stays busy to the end
        res["search"] = list(ex.map(run, jobs_search, chunksize=1))
        print(f"search done at {time.perf_counter() - t0:.0f} s", flush=True)
        res["info"] = list(ex.map(run, jobs_info, chunksize=8))
        print(f"info done at {time.perf_counter() - t0:.0f} s", flush=True)
        res["engine"] = list(ex.map(run_engine, jobs_engine, chunksize=1))
    res["elapsed_s"] = time.perf_counter() - t0
    json.dump(res, open(out_dir / "bench.json", "w"))
    md = tables(res, args)
    (out_dir / "tables.md").write_text(md)
    print(md)
    print(f"elapsed {res['elapsed_s']:.0f} s; written {out_dir}", flush=True)


if __name__ == "__main__":
    main()
