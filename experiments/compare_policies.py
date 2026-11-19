"""Compare policies from an experiment folder (or run a fresh one) and plot.

    python experiments/compare_policies.py                       # fresh: 30 runs, all policies
    python experiments/compare_policies.py --from results/experiments/<tag>

Produces box plots of the main KPIs (``comparison.png``) and the Pareto
front (makespan vs laser travel vs conveyor stop time) printed to stdout.
Requires matplotlib (optional dependency).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from optimization.objective_functions import pareto_front  # noqa: E402

PLOT_METRICS = [("throughput_per_min", "Throughput [parts/min]"),
                ("avg_waiting_time", "Avg waiting time [s]"),
                ("avg_flow_time", "Avg flow time [s]"),
                ("total_laser_travel_distance", "Laser travel [cm]"),
                ("laser_utilization", "Laser utilization [-]"),
                ("total_conveyor_stop_time", "Conveyor stop time [s]")]


def load_runs(folder: Path) -> list[dict]:
    with open(folder / "runs.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = sorted({r["policy"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, label) in zip(axes.flat, PLOT_METRICS):
        data = [[float(r[key]) for r in rows if r["policy"] == p and r[key] not in ("", None)]
                for p in policies]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(policies) + 1), policies, rotation=20)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Policy comparison ({len(rows) // max(1, len(policies))} runs per policy)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", help="existing results/experiments/<tag> folder")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None)
    args, rest = ap.parse_known_args(argv)

    if args.src:
        folder = Path(args.src)
    else:
        from run_experiment import main as run_main
        tag = "compare_latest"
        extra = ["--runs", str(args.runs), "--tag", tag] + rest
        if args.workers:
            extra += ["--workers", str(args.workers)]
        run_main(extra)
        folder = ROOT / "results" / "experiments" / tag

    rows = load_runs(folder)
    plot(rows, folder / "comparison.png")
    print(f"Plot saved to {folder / 'comparison.png'}")

    # Policy-level Pareto front on mean values (all minimised).
    policies = sorted({r["policy"] for r in rows})
    keys = ["makespan", "total_laser_travel_distance", "total_conveyor_stop_time"]
    means = []
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        means.append(tuple(sum(float(r[k]) for r in sub) / len(sub) for k in keys))
    front = pareto_front(means)
    print("\nMean values (minimise all):", ", ".join(keys))
    for i, p in enumerate(policies):
        mark = "  <- Pareto-optimal" if i in front else ""
        print(f"  {p:16s} " + "  ".join(f"{v:10.1f}" for v in means[i]) + mark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
