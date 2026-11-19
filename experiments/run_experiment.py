"""Batch experiments: many seeds x several policies, no GUI, optional parallelism.

Every policy is evaluated on the SAME seeds (common random numbers), so the
differences between policies are not due to different random arrivals.

    python run_experiments.py --runs 1000
    python run_experiments.py --runs 200 --policies fifo nearest_job --workers 8
    python run_experiments.py --runs 50 --set simulation.dt=0.02 --set arrivals.stochastic.mean_interarrival=[10,12,14]

Outputs (``results/experiments/<tag>/``):
    runs.csv            one row per (policy, seed) with every KPI
    summary_by_policy.csv / .json  mean, std, 95% CI half-width, min, max per KPI
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policies import available_policies  # noqa: E402
from simulation.config import build_config, load_raw  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.results_io import write_dict_csv  # noqa: E402

KEY_METRICS = ["parts_completed", "parts_exited_unprocessed", "throughput_per_min",
               "avg_waiting_time", "max_waiting_time", "avg_flow_time", "makespan",
               "total_laser_travel_distance", "laser_utilization",
               "laser_processing_utilization", "total_conveyor_stop_time",
               "max_lateness", "blocking_events", "total_blocked_time",
               "blocking_penalty", "objective"]


def run_one(job: tuple[dict, str, int]) -> dict:
    """Worker: run one (raw_config, policy, seed) combination headless."""
    raw, policy, seed = job
    raw = json.loads(json.dumps(raw))            # private copy
    raw["policy"]["name"] = policy
    raw["simulation"]["seed"] = seed
    result = Simulation(build_config(raw), record_trace=False).run()
    row = {"policy": policy, "seed": seed}
    row.update({k: v for k, v in result.summary.items() if k not in ("policy", "seed")})
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for policy in sorted({r["policy"] for r in rows}):
        sub = [r for r in rows if r["policy"] == policy]
        for k in KEY_METRICS + [k for k in sub[0] if k.endswith("_avg_queue_length")]:
            vals = [float(r[k]) for r in sub if r.get(k) is not None]
            if not vals:
                continue
            sd = stdev(vals) if len(vals) > 1 else 0.0
            out.append({"policy": policy, "metric": k, "n": len(vals), "mean": mean(vals),
                        "std": sd, "ci95": 1.96 * sd / math.sqrt(len(vals)),
                        "min": min(vals), "max": max(vals)})
    return out


def run_experiments(raw: dict, policies: list[str], runs: int, seed_base: int,
                    workers: int) -> list[dict]:
    jobs = [(raw, p, seed_base + i) for p in policies for i in range(runs)]
    if workers <= 1:
        return [run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_one, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30, help="replications per policy")
    ap.add_argument("--policies", nargs="+", default=None,
                    help=f"policies to compare (default: all = {available_policies()})")
    ap.add_argument("--config", help="YAML merged on top of the defaults")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], default="stochastic")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", help="output folder name (default: timestamp)")
    args = ap.parse_args(argv)

    raw = load_raw(args.config, set_items=args.set + [f"arrivals.mode={args.mode}"])
    build_config(raw)                                    # validate once before forking
    policies = args.policies or available_policies()
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / raw["output"].get("results_dir", "results") / "experiments" / tag
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Running {args.runs} runs x {len(policies)} policies ({args.mode}) "
          f"with {args.workers} worker(s)...")
    rows = run_experiments(raw, policies, args.runs, args.seed_base, args.workers)
    agg = aggregate(rows)
    write_dict_csv(out / "runs.csv", rows)
    write_dict_csv(out / "summary_by_policy.csv", agg)
    (out / "summary_by_policy.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(f"Done in {time.time() - t0:.1f} s -> {out}")
    show = ["throughput_per_min", "avg_waiting_time", "avg_flow_time", "makespan",
            "total_laser_travel_distance", "laser_utilization", "total_conveyor_stop_time"]
    print(f"\n{'metric':30s}" + "".join(f"{p:>22s}" for p in policies))
    for m in show:
        line = f"{m:30s}"
        for p in policies:
            a = next((x for x in agg if x["policy"] == p and x["metric"] == m), None)
            line += f"{a['mean']:>13.2f} +/-{a['ci95']:>7.2f}" if a else f"{'-':>22s}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
