"""Restriction gap of belt-order (head-only) policies — docs/methods/restriction_gap.md.

Question (literature review, point A): ``nearest_head`` and the RL agent may only
serve the HEAD of a conveyor, i.e. they process every conveyor in belt order. How
much can be lost by that restriction?

For small instances we enumerate every processing order pi (decoder Phi, i.e. the
``fixed_sequence`` policy) and compare

    best over ALL permutations            vs.    best over BELT-ORDER permutations

(belt order = within each conveyor, parts are processed in arrival order; these are
exactly the orders a head-only policy can realise). Screening at dt = 0.05, then the
best order of each class is re-simulated at dt = 0.01 with the formal-model check.

    python experiments/restriction_gap.py --workers 2
    python experiments/restriction_gap.py --n 6 --seeds 2 --workers 1      # quick

Scope: optimal over the decoder Phi (non-delay semi-active schedules, pre-positioning
``none``), not over all schedules of P (deliberate waiting is not explored).
Output: results/experiments/restriction_gap/{rows.csv,summary.md} (not versioned).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization.evaluate import evaluate_sequence  # noqa: E402
from optimization.instances import Instance, generate_instance  # noqa: E402
from optimization.objective_functions import OBJECTIVES  # noqa: E402

OBJS = ("penalized_flow_time", "avg_flow_time", "makespan")
#: "default": default arrival process; "burst": mean inter-arrival 3 s on every
#: conveyor, so the parts compete for the laser (otherwise small instances are trivial)
LOADS = {"default": {}, "burst": {"arrivals.stochastic.mean_interarrival": "[3.0, 3.0, 3.0]"}}


def respects_belt_order(perm, inst: Instance) -> bool:
    last = {}
    rank = {j.id: i for i, j in enumerate(inst.jobs)}          # ids are in arrival order
    for pid in perm:
        k = inst.jobs[rank[pid]].conveyor
        if k in last and rank[pid] < last[k]:
            return False
        last[k] = rank[pid]
    return True


def solve(job):
    load, n, seed = job
    inst = generate_instance(n, seed, "stochastic", overrides=LOADS[load] or None,
                             name=f"{load}_n{n}_s{seed}")
    t0 = time.time()
    best = {(o, c): (math.inf, None) for o in OBJS for c in ("all", "belt")}
    n_perm = n_belt = n_infeasible = 0
    for perm in itertools.permutations(inst.ids):
        ev = evaluate_sequence(inst, list(perm), OBJS[0])
        n_perm += 1
        belt = respects_belt_order(perm, inst)
        n_belt += belt
        if not ev.feasible:
            n_infeasible += 1
            continue
        for o in OBJS:
            v = OBJECTIVES[o](ev.summary)
            for c in (("all", "belt") if belt else ("all",)):
                if v < best[(o, c)][0] - 1e-12:
                    best[(o, c)] = (v, perm)
    rows = []
    for o in OBJS:
        val = {}
        for c in ("all", "belt"):
            perm = best[(o, c)][1]
            ev = evaluate_sequence(inst, list(perm), o, dt=0.01, check=True)
            assert ev.feasible and not ev.violations, (inst.name, o, c, ev.reason)
            val[c] = (ev.objective, perm)
        gap = 100 * (val["belt"][0] - val["all"][0]) / val["all"][0]
        rows.append({"load": load, "n": n, "seed": seed, "objective": o,
                     "best_all": round(val["all"][0], 4), "best_belt": round(val["belt"][0], 4),
                     "gap_pct": round(gap, 4),
                     "best_all_is_belt_order": respects_belt_order(val["all"][1], inst),
                     "permutations": n_perm, "belt_order_permutations": n_belt,
                     "infeasible_permutations": n_infeasible,
                     "best_all_order": " ".join(val["all"][1]),
                     "best_belt_order": " ".join(val["belt"][1]),
                     "seconds": round(time.time() - t0, 1)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--seeds", type=int, default=5, help="seeds for n=6 (n=7 uses min(3, seeds))")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="restriction_gap")
    args = ap.parse_args(argv)
    jobs = [(l, n, s) for l in LOADS for n in args.n
            for s in range(1, (args.seeds if n <= 6 else min(3, args.seeds)) + 1)]
    jobs.sort(key=lambda j: -j[1])
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(solve, jobs) for r in rs]
    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| load | n | objective | instances | mean gap [%] | max gap [%] | "
             "instances with gap > 0.1 % | optimum respects belt order |",
             "|---|---|---|---|---|---|---|---|"]
    for load in LOADS:
        for n in args.n:
            for o in OBJS:
                sub = [r for r in rows if (r["load"], r["n"], r["objective"]) == (load, n, o)]
                if not sub:
                    continue
                g = [r["gap_pct"] for r in sub]
                lines.append(f"| {load} | {n} | {o} | {len(sub)} | {mean(g):.2f} | {max(g):.2f} | "
                             f"{sum(x > 0.1 for x in g)} | "
                             f"{sum(r['best_all_is_belt_order'] for r in sub)}/{len(sub)} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    for r in rows:
        if r["gap_pct"] > 0.1:
            print(r["load"], r["n"], r["seed"], r["objective"], r["gap_pct"], "| all:",
                  r["best_all_order"], "| belt:", r["best_belt_order"])
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
