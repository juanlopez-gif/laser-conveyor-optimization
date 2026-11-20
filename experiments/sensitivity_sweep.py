"""Sensitivity of the policies to the cell parameters (version 0.6.0).

One-at-a-time sweep around the default cell (laser 10 cm/s, processing 2.5 s, belts
3 cm/s, arrivals 10/12/14 s; the user decided on 2026-11-19 to keep these values and
not to model laser acceleration), docs/methods/sensitivity.md:

* ``belt``       conveyor speed  ``conveyors.speed``           [cm/s] (adjustable in the cell)
* ``laser``      laser speed     ``laser.speed_x = speed_y``   [cm/s]
* ``processing`` processing time ``processing.time``           [s]    (adjustable job)
* ``arrival``    arrival rate: ``arrivals.stochastic.mean_interarrival`` = factor x
                 [10, 12, 14] s (factor < 1 = more parts)

Every cell runs every policy on the same seeds (common random numbers) and reports
the paired relative difference in penalized flow time against ``nearest_job`` with
pre-positioning ``none`` *in the same cell*.

    python experiments/sensitivity_sweep.py --workers 2 --seeds 10 \
        --policies nearest_job nearest_head rollout "rl@model=models/rl06_wi_s0.zip" ...

Outputs (``results/experiments/<tag>/``, not versioned): runs.csv, summary.md,
sensitivity.png.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import blocking_prepositioning_study as bps  # noqa: E402
from blocking_prepositioning_study import run_one  # noqa: E402

BASES = {"default": None}     # further base cells can be added here (name -> config path)
bps.LOADS.update(BASES)        # module level: also done in the worker processes

DEFAULT_ARRIVALS = (10.0, 12.0, 14.0)
SWEEPS = {
    "belt": ("conveyor speed [cm/s]", [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
             lambda v: [f"conveyors.speed={v}"]),
    "laser": ("laser speed [cm/s]", [5.0, 10.0, 16.67],
              lambda v: [f"laser.speed_x={v}", f"laser.speed_y={v}"]),
    "processing": ("processing time [s]", [1.5, 2.5, 3.0, 4.0],
                   lambda v: [f"processing.time={v}"]),
    "arrival": ("mean inter-arrival factor (x [10,12,14] s)", [0.6, 0.8, 1.0, 1.2],
                lambda v: ["arrivals.stochastic.mean_interarrival=["
                           + ", ".join(f"{v * a:g}" for a in DEFAULT_ARRIVALS) + "]"]),
}
BASE_VALUE = {"default": {"belt": 3.0, "laser": 10.0, "processing": 2.5, "arrival": 1.0}}
BASELINE = ("nearest_job", "none")


def _job(args):
    base, sweep, value, mode, policy, seed = args
    row = run_one((base, mode, policy, seed, SWEEPS[sweep][2](value)))
    row.update(sweep=sweep, value=value)
    return row


def summarize(rows: list[dict]) -> str:
    out = []
    for sweep, (label, values, _) in SWEEPS.items():
        sub = [r for r in rows if r["sweep"] == sweep]
        if not sub:
            continue
        cells = sorted({(r["policy"], r["mode"]) for r in sub} - {BASELINE})
        out += [f"\n### {sweep}: {label}\n",
                "| policy | mode | " + " | ".join(f"{v:g}" for v in values) + " |",
                "|---|---|" + "---|" * len(values)]
        base_line = ["**nearest_job (abs. penalized flow, s)**", "none"]
        for v in values:
            b = [r["penalized_flow_time"] for r in sub
                 if r["value"] == v and (r["policy"], r["mode"]) == BASELINE]
            base_line.append(f"{mean(b):.1f}" if b else "–")
        out.append("| " + " | ".join(base_line) + " |")
        for pol, mode in cells:
            line = [pol, mode]
            for v in values:
                base = {r["seed"]: r["penalized_flow_time"] for r in sub
                        if r["value"] == v and (r["policy"], r["mode"]) == BASELINE}
                rel = [100 * (r["penalized_flow_time"] / base[r["seed"]] - 1) for r in sub
                       if r["value"] == v and (r["policy"], r["mode"]) == (pol, mode)
                       and r["seed"] in base]
                if len(rel) > 1:
                    half = 1.96 * stdev(rel) / len(rel) ** 0.5
                    line.append(f"{mean(rel):+.1f} ±{half:.1f}")
                else:
                    line.append("–")
            out.append("| " + " | ".join(line) + " |")
    return "\n".join(out) + "\n"


def plot(rows: list[dict], path: Path, base_name: str = "default") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted({(r["policy"], r["mode"]) for r in rows} - {BASELINE})
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(5 * len(SWEEPS), 4), sharey=True)
    for ax, (sweep, (label, values, _)) in zip(axes, SWEEPS.items()):
        sub = [r for r in rows if r["sweep"] == sweep]
        for pol, mode in cells:
            ys = []
            for v in values:
                base = {r["seed"]: r["penalized_flow_time"] for r in sub
                        if r["value"] == v and (r["policy"], r["mode"]) == BASELINE}
                rel = [100 * (r["penalized_flow_time"] / base[r["seed"]] - 1) for r in sub
                       if r["value"] == v and (r["policy"], r["mode"]) == (pol, mode)
                       and r["seed"] in base]
                ys.append(mean(rel) if rel else float("nan"))
            ax.plot(values, ys, marker="o", label=f"{pol.split('@')[0]} / {mode}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(BASE_VALUE[base_name][sweep], color="grey", lw=0.8, ls=":")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ penalized flow vs nearest_job/none [%]")
    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True,
                    help="policy specs as in blocking_prepositioning_study.py; "
                         "nearest_job/none is always added as baseline")
    ap.add_argument("--modes", nargs="+", default=["when_idle"])
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base", default="default", choices=list(BASES))
    ap.add_argument("--tag", default="v0.6_sensitivity")
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary/plot")
    args = ap.parse_args(argv)

    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        with open(out / "runs.csv", encoding="utf-8") as f:
            rows = [{**r, "value": float(r["value"]), "seed": int(r["seed"]),
                     "penalized_flow_time": float(r["penalized_flow_time"])}
                    for r in csv.DictReader(f)]
    else:
        seeds = [args.seed_base + k for k in range(args.seeds)]
        jobs = [(args.base, sw, v, "none", "nearest_job", s) for sw in args.sweeps
                for v in SWEEPS[sw][1] for s in seeds]
        jobs += [(args.base, sw, v, m, p, s) for sw in args.sweeps for v in SWEEPS[sw][1]
                 for m in args.modes for p in args.policies for s in seeds
                 if (p, m) != BASELINE]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_job, jobs, chunksize=1))
        print(f"{len(rows)} runs in {time.time() - t0:.0f} s")
        with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.md").write_text(summarize(rows), encoding="utf-8")
    plot(rows, out / "sensitivity.png", args.base)
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
