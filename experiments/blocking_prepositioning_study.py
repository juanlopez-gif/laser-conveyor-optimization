"""Study of version 0.4.0: blocking penalty, nearest_head and laser pre-positioning.

Grid: load (default, heavy) x pre-positioning mode (none, when_idle, always) x
policy x 30 stochastic seeds (common random numbers: every cell uses the same
seeds 1000..1029, as run_experiments.py). The rollout policy is expensive and is
only run with pre-positioning ``none`` and ``when_idle``.

    python experiments/blocking_prepositioning_study.py --workers 2
    python experiments/blocking_prepositioning_study.py --seeds 5 --no-rollout   # quick
    # sub-grid; "rollout@avg_flow_time" = rollout with that look-ahead objective,
    # "rl@model=models/rl_bc.zip" = policy parameter model=...
    python experiments/blocking_prepositioning_study.py --policies nearest_job rollout@avg_flow_time         --modes none when_idle --tag rollout_objective

Outputs (``results/experiments/<tag>/``, not versioned):
    runs.csv     one row per (load, mode, policy, seed) with every KPI
    summary.md   tables used in docs/methods/nearest_head_prepositioning.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimization.objective_functions import OBJECTIVES  # noqa: E402
from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402

LOADS = {"default": None, "heavy": str(ROOT / "config" / "examples" / "heavy_load.yaml")}
MODES = ("none", "when_idle", "always")
CHEAP = ("fifo", "nearest_job", "nearest_head", "weighted_score", "round_robin")
KPIS = ("avg_flow_time", "penalized_flow_time", "blocking_events", "total_blocked_time",
        "blocking_penalty", "makespan", "avg_waiting_time", "total_laser_travel_distance")
BASELINE = ("nearest_job", "none")


def run_one(job: tuple[str, str, str, int]) -> dict:
    load, mode, policy, seed = job
    name, _, objective = policy.partition("@")
    sets = ["arrivals.mode=stochastic", f"simulation.seed={seed}", f"policy.name={name}",
            f"laser.prepositioning={mode}"]
    if objective:                      # rollout@avg_flow_time  or  rl@model=models/rl_bc.zip
        key, eq, value = objective.partition("=")
        sets.append(f"policy.params.{name}.{key}={value}" if eq
                    else f"policy.params.{name}.objective={objective}")
    cfg = load_config(LOADS[load], set_items=sets)
    t0 = time.perf_counter()
    s = Simulation(cfg, record_trace=False).run().summary
    row = {"load": load, "mode": mode, "policy": policy, "seed": seed,
           "wall_s": round(time.perf_counter() - t0, 3)}
    for k in KPIS:
        row[k] = OBJECTIVES[k](s) if k in OBJECTIVES else s[k]
    row["parts_completed"] = s["parts_completed"]
    return row


def jobs(seeds: list[int], with_rollout: bool, policies=None, modes=None,
         loads=None) -> list[tuple[str, str, str, int]]:
    if policies:                       # explicit sub-grid
        return [(l, m, p, s) for l in (loads or LOADS) for m in (modes or MODES)
                for p in policies for s in seeds]
    out = [(l, m, p, s) for l in LOADS for m in MODES for p in CHEAP for s in seeds]
    if with_rollout:
        out += [(l, m, "rollout", s) for l in LOADS for m in ("none", "when_idle") for s in seeds]
    return out


def summarize(rows: list[dict]) -> str:
    lines = []
    for load in LOADS:
        base = {r["seed"]: r for r in rows
                if (r["load"], r["policy"], r["mode"]) == (load, *BASELINE)}
        lines += [f"\n### Load: {load}\n",
                  "| policy | pre-positioning | mean flow [s] | penalized flow [s] | "
                  "blocking events | blocked time [conv-s] | makespan [s] | "
                  f"paired Δ penalized flow vs {BASELINE[0]}/{BASELINE[1]} [%] (95% CI) | "
                  "better on |",
                  "|---|---|---|---|---|---|---|---|---|"]
        cells = sorted({(r["policy"], r["mode"]) for r in rows if r["load"] == load},
                       key=lambda c: mean(r["penalized_flow_time"] for r in rows
                                          if (r["load"], r["policy"], r["mode"]) == (load, *c)))
        for pol, mode in cells:
            sub = [r for r in rows if (r["load"], r["policy"], r["mode"]) == (load, pol, mode)]

            def ms(k):
                v = [r[k] for r in sub]
                return f"{mean(v):.2f} ± {stdev(v) if len(v) > 1 else 0:.2f}"
            wins = sum(1 for r in sub if r["seed"] in base
                       and r["penalized_flow_time"] < base[r["seed"]]["penalized_flow_time"] - 1e-9)
            rel = [100 * (r["penalized_flow_time"] / base[r["seed"]]["penalized_flow_time"] - 1)
                   for r in sub if r["seed"] in base]
            if len(rel) > 1:
                half = 1.96 * stdev(rel) / len(rel) ** 0.5
                delta = f"{mean(rel):+.1f} ({mean(rel) - half:+.1f}, {mean(rel) + half:+.1f})"
            else:
                delta = "–"
            lines.append(f"| {pol} | {mode} | {ms('avg_flow_time')} | {ms('penalized_flow_time')} | "
                         f"{ms('blocking_events')} | {ms('total_blocked_time')} | {ms('makespan')} | "
                         f"{delta} | {wins}/{len(sub)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-rollout", action="store_true")
    ap.add_argument("--tag", default="v0.4_blocking_prepositioning")
    ap.add_argument("--policies", nargs="+", help="sub-grid: policies (name or name@objective)")
    ap.add_argument("--modes", nargs="+", choices=MODES)
    ap.add_argument("--loads", nargs="+", choices=list(LOADS))
    ap.add_argument("--summarize", nargs="+", metavar="TAG",
                    help="do not simulate: merge results/experiments/<TAG>/runs.csv files and "
                         "write the summary into --tag")
    args = ap.parse_args(argv)

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    if args.summarize:
        rows, seen = [], set()
        for tag in args.summarize:
            with open(ROOT / "results" / "experiments" / tag / "runs.csv", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    for k in KPIS + ("wall_s",):
                        r[k] = float(r[k])
                    r["seed"] = int(r["seed"])
                    key = (r["load"], r["mode"], r["policy"], r["seed"])
                    if key not in seen:          # the same cell run twice is identical
                        seen.add(key)
                        rows.append(r)
        out = ROOT / "results" / "experiments" / args.tag
        out.mkdir(parents=True, exist_ok=True)
        text = (f"# Merged summary of {', '.join(args.summarize)}" + chr(10) * 2
                + "Values: mean ± std over seeds; paired Δ = mean relative difference of "
                "penalized flow on the same seed, with a normal 95 % CI." + chr(10)
                + summarize(rows))
        (out / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    todo = jobs(seeds, not args.no_rollout, args.policies, args.modes, args.loads)
    # expensive jobs first so the pool stays busy until the end
    todo.sort(key=lambda j: (not j[2].startswith("rollout"), j[0] != "heavy"))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, todo, chunksize=1))
    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = load_config()
    header = (f"# Blocking penalty / nearest_head / pre-positioning study\n\n"
              f"{len(seeds)} stochastic seeds ({seeds[0]}..{seeds[-1]}), common random numbers. "
              f"Blocking penalty: {cfg.blocking_penalty_per_event} s per event + "
              f"{cfg.blocking_penalty_per_second} s per blocked conveyor-second. "
              f"Penalized flow = mean flow + penalty / n. Values: mean ± std over seeds. "
              f"Wall time {time.time() - t0:.0f} s.\n")
    text = header + summarize(rows)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(text)
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
