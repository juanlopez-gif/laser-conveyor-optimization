"""Tie-break audit (0.7.26, rerun 0.7.33): the rules' legacy tie-break on the arrival time
against the observable one, on development realizations.

0.7.33: the known-parts search applies its key inside every look-ahead continuation as
well as when ordering the candidates (``SearchTeacher`` passes ``tie_key`` to
``run_lookaheads``); the 0.7.26 run changed only the ordering, so its legacy arm of the
search was incomplete (referee report on drafts 0.5 / 0.8). The pooled contrast is now a
mean per seed over the eight regimes, with the interval over the 30 seed means (the
eight regimes of one seed share the realization and are not independent).

For every regime of phase 4 (real / heavy × Poisson / bursts × nominal / slow head),
treatments B and C, and the policies ``nearest_job``, ``nearest_head`` and the
known-parts search over ``nearest_job``: run the episode with the observable key and,
at every decision with at least two candidates, also evaluate the legacy key on the
same state (decisions changed); and run the legacy policy on the same realization
(paired cost ratio new vs legacy). Seeds 2000–2029 (development); never the reserved
ones.

    python experiments/tiebreak_audit.py --workers 40
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
sys.path.insert(0, str(ROOT / "experiments"))

from rl2.expert_iteration import _init_worker  # noqa: E402
from rl2_eval import LOADS, ARRIVALS, TRAVEL, TREATMENTS, _extra  # noqa: E402

POLICIES = ("nearest_job", "nearest_head", "search")


def _policy(name, legacy):
    from fastsim.policies import KEY_ARRIVAL, KEY_OBSERVED, nearest_head, nearest_job
    from rl2.search import SearchTeacher
    key = KEY_ARRIVAL if legacy else KEY_OBSERVED
    if name == "nearest_job":
        return lambda sim, cands: nearest_job(sim, cands, key=key)
    if name == "nearest_head":
        return lambda sim, cands: nearest_head(sim, cands, key=key)
    return SearchTeacher(None, k=6, tie_key=key)


def run(job):
    from fastsim import FastSim
    from rl2.expert_iteration import val_config
    cfg = val_config(job["load"], job["seed"], "none", extra=tuple(_extra(job)))
    new, old = _policy(job["who"], False), _policy(job["who"], True)
    t0 = time.perf_counter()
    sim = FastSim(cfg)
    decisions = changed = 0
    while sim.run_until_decision():
        cands = sim.candidates()
        if len(cands) > 1:
            a = new(sim, cands)
            b = old(sim, cands)
            decisions += 1
            changed += a != b
        else:
            a = cands[0]
        sim.apply(a)
    cost_new = sim.summary()["penalized_flow_time"]
    sim2 = FastSim(cfg)
    s2 = sim2.run(old)
    return job | {"decisions": decisions, "changed": changed, "cost_new": cost_new,
                  "cost_old": s2["penalized_flow_time"], "secs": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tag", default="tiebreak_audit")
    args = ap.parse_args()
    lo, hi = args.seed_base, args.seed_base + args.seeds - 1
    for b0, n in ((7000, 30), (9000, 60), (10000, 60)):
        if lo <= b0 + n - 1 and hi >= b0:
            sys.exit("reserved seeds")
    regimes = [(g, a, tr) for g in LOADS for a in ARRIVALS for tr in TRAVEL]
    jobs = [{"load": g, "arrivals": a, "travel": tr, "slow": 0.5, "seed": s, "t": t,
             "obs": TREATMENTS[t][0], "mode": TREATMENTS[t][1], "visible": None, "who": w}
            for g, a, tr in regimes for s in range(args.seed_base, args.seed_base + args.seeds)
            for t in ("B", "C") for w in POLICIES]
    print(f"{len(jobs)} audits", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        rows = list(ex.map(run, jobs, chunksize=4))
    out = ROOT / "results/experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "rows": rows}, open(out / "audit.json", "w"))
    L = ["# Tie-break audit: observable key (0.7.26) against the legacy arrival-time key (rerun 0.7.33)", "",
         "Rerun of 0.7.33: the known-parts search applies the key under audit inside every look-ahead "
         "continuation as well as when ordering its candidates (the 0.7.26 run changed only the ordering).", "",
         f"Seeds {args.seed_base}–{args.seed_base + args.seeds - 1} (development), treatments B and C, "
         "eight regimes. ``changed`` = decisions (≥ 2 candidates) of the new episode at which the legacy "
         "key would have chosen another part; Δ = paired cost ratio of the new policy vs the legacy one "
         "[%], 95 % CI, negative = the observable key is cheaper.", "",
         "| regime | t | policy | decisions | changed | Δ new vs legacy [%] |", "|---|---|---|---|---|---|"]
    summary = {}
    for g, a, tr in regimes:
        for t in ("B", "C"):
            for w in POLICIES:
                rs = [r for r in rows if (r["load"], r["arrivals"], r["travel"], r["t"], r["who"]) == (g, a, tr, t, w)]
                d = [r["cost_new"] / r["cost_old"] - 1 for r in rs]
                mu, h = 100 * np.mean(d), 100 * 1.96 * np.std(d, ddof=1) / math.sqrt(len(d))
                dec, ch = sum(r["decisions"] for r in rs), sum(r["changed"] for r in rs)
                summary[(t, w)] = summary.get((t, w), []) + d
                L.append(f"| {g} / {a} / {tr} | {t} | {w} | {dec} | {ch} ({100 * ch / dec:.1f} %) | {mu:+.2f} ±{h:.2f} |")
    L += ["", "Pooled over the eight regimes: one mean contrast per seed over its eight regimes, interval over "
          f"the {args.seeds} seed means (the eight episodes of one seed share the realization):", "",
          "| t | policy | Δ new vs legacy [%] (per-seed means) | max |Δ| over regimes [%] |", "|---|---|---|---|"]
    for t in ("B", "C"):
        for w in POLICIES:
            per_seed = []
            for s in range(args.seed_base, args.seed_base + args.seeds):
                rs = [r for r in rows if (r["t"], r["who"], r["seed"]) == (t, w, s)]
                per_seed.append(np.mean([r["cost_new"] / r["cost_old"] - 1 for r in rs]))
            mu, h = 100 * np.mean(per_seed), 100 * 1.96 * np.std(per_seed, ddof=1) / math.sqrt(len(per_seed))
            per = []
            for g, a, tr in regimes:
                rs = [r for r in rows if (r["load"], r["arrivals"], r["travel"], r["t"], r["who"]) == (g, a, tr, t, w)]
                per.append(abs(100 * np.mean([r["cost_new"] / r["cost_old"] - 1 for r in rs])))
            L.append(f"| {t} | {w} | {mu:+.2f} ±{h:.2f} | {max(per):.2f} |")
    md = "\n".join(L) + "\n"
    (out / "audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
