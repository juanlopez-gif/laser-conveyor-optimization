"""Benchmark harness: every registered method on a bank of instances.

    python -m optimization.benchmark --n 6 8 12 --seeds 3 --time-limit 30
    (run from the repo root with PYTHONPATH=src, or use experiments/benchmark.py)

For each (instance, method) the solution is validated by simulation at dt = 0.01
with the formulation checker, and compared with the best lower bound available
(``optimization.lower_bounds.best_lower_bound`` if that module exists, otherwise
the trivial bounds of the formulation). Output: results/benchmark/<tag>/results.csv
and a summary table (mean gap to the best solution found and to the lower bound).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from optimization.evaluate import VALIDATION_DT, evaluate_policy, evaluate_sequence
from optimization.instances import Instance, generate_instance
from optimization.solvers import discover
from simulation.config import PROJECT_ROOT, load_config


def trivial_lower_bound(inst: Instance) -> float:
    """Makespan bounds of the formulation (Proposition 'Lower bounds')."""
    cfg = load_config(set_items=[f"{k}={v}" for k, v in inst.overrides.items()])
    a = cfg.geometry.active_start
    speed = {c.id: c.speed for c in cfg.conveyors}
    avail = [j.time + a / speed[j.conveyor] for j in inst.jobs]
    lb1 = max(av + j.processing_time for av, j in zip(avail, inst.jobs))
    lb2 = min(avail) + sum(j.processing_time for j in inst.jobs)
    return max(lb1, lb2)


def lower_bound(inst: Instance, objective) -> float | None:
    try:
        from optimization.lower_bounds import best_lower_bound   # provided by the theory agent
        return best_lower_bound(inst, objective)
    except ImportError:
        return trivial_lower_bound(inst) if objective == "makespan" else None


def run_benchmark(instances: list[Instance], methods: list[str] | None, objective="makespan",
                  time_limit: float = 30.0, seed: int = 0) -> list[dict]:
    registry = discover()
    names = methods or sorted(registry)
    rows = []
    for inst in instances:
        lb = lower_bound(inst, objective)
        for name in names:
            t0 = time.time()
            sol = registry[name](inst, objective, time_limit, seed)
            elapsed = time.time() - t0
            if sol.sequence:
                ev = evaluate_sequence(inst, sol.sequence, objective, dt=VALIDATION_DT, check=True)
            elif sol.policy:
                ev = evaluate_policy(inst, sol.policy, sol.policy_params or None, objective)
            else:
                continue                                         # method skipped this instance
            if sol.lower_bound is not None:
                lb = max(lb or -math.inf, sol.lower_bound)
            rows.append({"instance": inst.name, "n": inst.n, "method": name,
                         "objective": ev.objective, "feasible": ev.feasible,
                         "violations": len(ev.violations), "time_s": round(elapsed, 3),
                         "proven_optimal": sol.proven_optimal,
                         "solver_lower_bound": sol.lower_bound,
                         "sequence": " ".join(ev.sequence), "info": json.dumps(sol.info)})
        best = min((r["objective"] for r in rows if r["instance"] == inst.name), default=math.inf)
        for r in rows:
            if r["instance"] == inst.name:
                r["lower_bound"] = lb
                r["gap_to_best_pct"] = 100 * (r["objective"] - best) / best if best else 0.0
                r["gap_to_lb_pct"] = (100 * (r["objective"] - lb) / lb
                                      if lb and math.isfinite(r["objective"]) else None)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mode", default="stochastic")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--objective", default="makespan")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--tag")
    args = ap.parse_args(argv)

    instances = [generate_instance(n, s, args.mode) for n in args.n for s in range(1, args.seeds + 1)]
    rows = run_benchmark(instances, args.methods, args.objective, args.time_limit)
    out = PROJECT_ROOT / "results" / "benchmark" / (args.tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'method':24s} {'mean obj':>10s} {'gap best %':>11s} {'gap LB %':>9s} {'time s':>8s} {'feasible':>9s}")
    for m in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == m]
        gl = [r["gap_to_lb_pct"] for r in sub if r["gap_to_lb_pct"] is not None]
        print(f"{m:24s} {mean(r['objective'] for r in sub):10.2f} "
              f"{mean(r['gap_to_best_pct'] for r in sub):11.2f} "
              f"{(mean(gl) if gl else float('nan')):9.2f} {mean(r['time_s'] for r in sub):8.2f} "
              f"{sum(r['feasible'] for r in sub):>5d}/{len(sub)}")
    print(f"\nResults: {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
