"""Separate the policy effect from the pre-positioning effect (version 0.6.1).

Reads ``results/experiments/<tag>/runs.csv`` written by
``blocking_prepositioning_study.py`` (every policy run with pre-positioning ``none`` and
``when_idle`` on the same seeds) and reports, per load, paired relative differences of
penalized flow time with a normal 95 % CI:

* **total**      policy/when_idle  vs  nearest_job/none   (the column used up to 0.6.0)
* **policy**     policy/mode   vs  nearest_job/mode   (same pre-positioning)
* **prepos.**    policy/when_idle  vs  policy/none    (same policy)
* **head-to-head** pairs given with ``--pairs A B`` (same mode), e.g. the hybrid
  against the plain rollout.

The total is not exactly the sum of the other two (ratios compose multiplicatively and
the means are of per-seed ratios), but it is close.

``--merge TAG ...`` (0.6.2) adds the runs of further tags (same seeds) before the
analysis, and ``--out`` names the output tag (default: ``--tag``). A policy run in only
one mode gets "–" where the other mode is needed.

    python experiments/decompose_effects.py --tag v0.6_confirmatory \\
        --pairs "rl_rollout@model=models/rl06_wi_s0.zip" rollout@avg_flow_time
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
KPI = "penalized_flow_time"
REF = "nearest_job"
MODES = ("none", "when_idle")


def short(policy: str) -> str:
    """Readable policy name: model paths reduced to the model names."""
    name, *specs = policy.split("@")
    keep = []
    for s in specs:
        k, eq, v = s.partition("=")
        if eq and k in ("model", "models"):
            keep.append("+".join(Path(m).stem.replace("rl06_", "") for m in v.split(",")))
        else:
            keep.append(s)
    return "@".join([name, *keep])


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int, int] | None:
    """Mean relative difference a/b - 1 in %, 95 % half-width, wins of a, n."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    rel = [100 * (a[s] / b[s] - 1) for s in seeds]
    half = 1.96 * stdev(rel) / len(rel) ** 0.5
    wins = sum(a[s] < b[s] - 1e-9 for s in seeds)
    return mean(rel), half, wins, len(seeds)


def fmt(r) -> str:
    if r is None:
        return "–"
    m, h, w, n = r
    star = "**" if abs(m) > h else ""          # CI excludes 0
    return f"{star}{m:+.1f} ±{h:.1f}{star} ({w}/{n})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--merge", nargs="+", default=[], metavar="TAG",
                    help="further tags whose runs.csv are merged in")
    ap.add_argument("--out", help="output tag (default: --tag)")
    ap.add_argument("--pairs", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="head-to-head comparison A vs B at the same pre-positioning")
    args = ap.parse_args(argv)

    exp = ROOT / "results" / "experiments"
    data: dict[tuple, dict[int, float]] = defaultdict(dict)   # (load, policy, mode) -> seed -> KPI
    for tag in [args.tag, *args.merge]:
        with open(exp / tag / "runs.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                data[(r["load"], r["policy"], r["mode"])][int(r["seed"])] = float(r[KPI])
    loads = sorted({k[0] for k in data}, key=lambda l: (l != "default", l))
    policies = sorted({k[1] for k in data})
    seeds = sorted({s for v in data.values() for s in v})

    lines = [f"# Policy vs pre-positioning effects: {' + '.join([args.tag, *args.merge])}", "",
             f"Seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), common random numbers. "
             f"Paired relative difference of {KPI} in % with a normal 95 % half-width and "
             "(wins / seeds); bold = the CI excludes 0. Negative = better.", ""]
    for load in loads:
        def cell(p, m):
            return data.get((load, p, m), {})

        def avg(p, m):
            return f"{mean(cell(p, m).values()):.2f}" if cell(p, m) else "–"

        base = cell(REF, "none")
        order = sorted(policies, key=lambda p: min(
            (mean(cell(p, m).values()) for m in MODES if cell(p, m)), default=float("inf")))
        lines += [f"## Load: {load}", "",
                  f"Reference {REF}/none: {mean(base.values()):.2f} s.", "",
                  "| policy | flow none [s] | flow when_idle [s] | total (when_idle vs ref/none) "
                  "| policy effect, none | policy effect, when_idle | pre-positioning effect |",
                  "|---|---|---|---|---|---|---|"]
        for p in order:
            cn, cw = cell(p, "none"), cell(p, "when_idle")
            pol_n = paired(cn, cell(REF, "none")) if p != REF else None
            pol_w = paired(cw, cell(REF, "when_idle")) if p != REF else None
            lines.append(
                f"| {short(p)} | {avg(p, 'none')} | {avg(p, 'when_idle')} | "
                f"{fmt(paired(cw, base))} | {fmt(pol_n)} | {fmt(pol_w)} | {fmt(paired(cw, cn))} |")
        if args.pairs:
            lines += ["", "| head-to-head | none | when_idle |", "|---|---|---|"]
            for a, b in args.pairs:
                cells = [fmt(paired(cell(a, m), cell(b, m))) for m in MODES]
                lines.append(f"| {short(a)} vs {short(b)} | {cells[0]} | {cells[1]} |")
        lines.append("")
    text = "\n".join(lines)
    out = exp / (args.out or args.tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomposition.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
