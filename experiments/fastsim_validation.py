"""Phase 1 acceptance of the strong-RL plan: fast simulator vs fixed-step engine.

For every (load, pre-positioning mode, rule, seed) the same instance is run on the
fixed-step engine (``simulation.engine``, step ``--dt``) and on the event-driven
simulator (``fastsim``) with the same rule (``nearest_job`` / ``nearest_head``), and

* processing order (identical or not),
* start times of the parts (max |S_fast - S_engine| while the orders agree),
* penalized flow time (relative difference),
* the formal checker (``optimization.schedule_check``) on the fast schedule,
* wall time of both,

are recorded. With ``--dt 0.001`` the engine is close to its continuous-time limit,
which the event-driven simulator computes exactly; comparing both ``--dt`` values
separates fast-simulator errors from the engine's own discretization error.

    python experiments/fastsim_validation.py --seeds 200 --dt 0.01 --workers 6 --tag fastsim_val_dt01

Outputs (``results/experiments/<tag>/``, not versioned): runs.csv, summary.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastsim import POLICIES, FastSim  # noqa: E402
from optimization.objective_functions import OBJECTIVES  # noqa: E402
from optimization.schedule_check import check_schedule  # noqa: E402
from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402

LOADS = {"default": None, "heavy": str(ROOT / "config" / "examples" / "heavy_load.yaml")}
MODES = ("none", "when_idle", "always")
RULES = ("nearest_job", "nearest_head")


def run_one(job: tuple) -> dict:
    load, mode, rule, seed, dt = job
    cfg = load_config(LOADS[load], set_items=[
        "arrivals.mode=stochastic", f"simulation.seed={seed}", f"policy.name={rule}",
        f"laser.prepositioning={mode}", f"simulation.dt={dt}"])
    t0 = time.perf_counter()
    res = Simulation(cfg, record_trace=False).run()
    t1 = time.perf_counter()
    fs = FastSim(cfg)
    s_fast = fs.run(POLICIES[rule])
    t2 = time.perf_counter()
    eng = sorted((p for p in res.parts if p.start_processing_time is not None),
                 key=lambda p: p.start_processing_time)
    o_eng = [p.number - 1 for p in eng]
    same = o_eng == fs.order
    n_common = next((i for i, (a, b) in enumerate(zip(o_eng, fs.order)) if a != b),
                    min(len(o_eng), len(fs.order)))
    s_eng = {p.number - 1: p.start_processing_time for p in res.parts}
    ds = max((abs(s_eng[j] - fs.pS[j]) for j in fs.order[:n_common]), default=0.0)
    k_eng = OBJECTIVES["penalized_flow_time"](res.summary)
    k_fast = s_fast["penalized_flow_time"]
    return {"load": load, "mode": mode, "rule": rule, "seed": seed, "dt": dt,
            "same_order": int(same), "first_diff": "" if same else n_common,
            "n_parts": len(o_eng), "max_dS_common": round(ds, 4),
            "kpi_engine": round(k_eng, 4), "kpi_fast": round(k_fast, 4),
            "kpi_rel_pct": round(100 * (k_fast / k_eng - 1), 4),
            "checker_errors": len(check_schedule(fs.result(), time_tol=1e-6, dist_tol=1e-6)),
            "decisions": s_fast["decisions"],
            "wall_engine_s": round(t1 - t0, 4), "wall_fast_s": round(t2 - t1, 4)}


def engine_vs_engine(job: tuple) -> dict:
    """Baseline: the fixed-step engine at ``dt`` against itself at ``dt / 10``."""
    load, mode, rule, seed, dt = job

    def run(step):
        cfg = load_config(LOADS[load], set_items=[
            "arrivals.mode=stochastic", f"simulation.seed={seed}", f"policy.name={rule}",
            f"laser.prepositioning={mode}", f"simulation.dt={step}"])
        res = Simulation(cfg, record_trace=False).run()
        order = [p.number - 1 for p in sorted(
            (p for p in res.parts if p.start_processing_time is not None),
            key=lambda p: p.start_processing_time)]
        return order, OBJECTIVES["penalized_flow_time"](res.summary)
    (o1, k1), (o2, k2) = run(dt), run(dt / 10)
    return {"load": load, "mode": mode, "rule": rule, "seed": seed, "dt": dt,
            "same_order": int(o1 == o2), "kpi_rel_pct": round(100 * (k1 / k2 - 1), 4)}


def summarize(rows: list[dict]) -> str:
    def block(sub, label):
        n = len(sub)
        same = sum(r["same_order"] for r in sub)
        rel = [abs(r["kpi_rel_pct"]) for r in sub]
        within = sum(v <= 0.5 for v in rel)
        return (f"| {label} | {n} | {100 * same / n:.1f} % | {100 * within / n:.1f} % | "
                f"{mean(rel):.3f} | {max(rel):.3f} | "
                f"{max(r['max_dS_common'] for r in sub):.3f} | "
                f"{sum(r['checker_errors'] > 0 for r in sub)} | "
                f"{sum(r['wall_engine_s'] for r in sub) / sum(r['wall_fast_s'] for r in sub):.0f}× |")
    head = ("| subset | runs | identical order | KPI within 0.5 % | mean abs KPI diff [%] | "
            "max abs KPI diff [%] | max start-time diff while orders agree [s] | "
            "runs with checker errors | speed-up (wall) |\n|---|---|---|---|---|---|---|---|---|")
    lines = [head, block(rows, "**all**")]
    for load in LOADS:
        for mode in MODES:
            sub = [r for r in rows if r["load"] == load and r["mode"] == mode]
            if sub:
                lines.append(block(sub, f"{load} / {mode}"))
    for rule in RULES:
        sub = [r for r in rows if r["rule"] == rule]
        if sub:
            lines.append(block(sub, rule))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--tag", default="fastsim_validation")
    ap.add_argument("--engine-baseline", action="store_true",
                    help="compare the engine at --dt with itself at --dt/10 instead")
    args = ap.parse_args(argv)
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    jobs = [(l, m, r, s, args.dt) for l in LOADS for m in args.modes for r in RULES for s in seeds]
    t0 = time.time()
    fn = engine_vs_engine if args.engine_baseline else run_one
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(fn, jobs, chunksize=4))
    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if args.engine_baseline:
        with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        rel = [abs(r["kpi_rel_pct"]) for r in rows]
        text = (f"# Engine dt = {args.dt} vs engine dt = {args.dt / 10}\n\n"
                f"{n} runs: identical order {100 * sum(r['same_order'] for r in rows) / n:.1f} %, "
                f"KPI within 0.5 % {100 * sum(v <= 0.5 for v in rel) / n:.1f} %, "
                f"mean abs KPI diff {mean(rel):.3f} %, max {max(rel):.3f} %.\n")
        (out / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = (f"# Fast simulator vs fixed-step engine (dt = {args.dt})\n\n"
            f"Seeds {seeds[0]}..{seeds[-1]}, {len(rows)} runs, wall {time.time() - t0:.0f} s.\n\n"
            + summarize(rows))
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
