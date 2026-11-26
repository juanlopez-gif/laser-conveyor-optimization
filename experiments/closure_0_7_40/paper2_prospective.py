"""Prospective replication of Paper 2 v1.5, commit 4892d46db10effa185f99245c6519878a705f35d.
Run with Python 3.11 and the repository dependencies installed:
python -B paper2_prospective.py --repo PATH_TO_CHECKOUT --output-dir NEW_OUTPUT_DIRECTORY
Both modes use the observable first-observation tie key. The fixed seed range is
43700..43759 inclusive; the positive grid, screen and Bonferroni constant come
from the published recheck. The previous results are read only for cohort overlap.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
import statistics as stats
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

COMMIT = "4892d46db10effa185f99245c6519878a705f35d"
SEEDS = range(43700, 43760)
MODES = ("none", "when_idle")
LANES = ((6, 10), (6, 15), (12, 5), (12, 10), (12, 15), (18, 5), (18, 10))
FIELDS = ("spacing_cm", "vy_cm_s", "vx_cm_s", "load", "arrivals", "coverage_s",
          "mode", "seed", "J_s_per_part", "planned_preposition_mean_s")
PREFIX = "paper2_prospective_"

def initialize(repo):
    os.chdir(repo)
    sys.path.insert(0, str(Path(repo) / "experiments"))
    import recheck_0_7_39
    global reference
    reference = recheck_0_7_39

def run_cell(cell):
    rows = []
    def run_coverage(s):
        costs = {}
        for mode in MODES:
            cfg = reference.geometry_config(cell, mode, s)
            costs[mode] = []
            for seed in SEEDS:
                rec = reference.run(cfg, seed, reference.KEY_OBSERVED)
                score = rec["penalized_flow_time"]
                costs[mode].append(score)
                rows.append((*cell, s, mode, seed, score, rec["prepos_t_mean"]))
        return costs

    full = run_coverage(10.)
    benefits = reference.benefits(full)
    gain_mean, gain_hw = reference.interval(benefits)
    selected = reference.screen(benefits)
    summary = dict(zip(FIELDS[:5], cell))
    summary.update(full_gain=gain_mean, full_gain_half95=gain_hw, selected=selected)
    if selected:
        costs = {10.: full}
        for s in reference.GRID[:-1]:
            costs[s] = run_coverage(s)
        curve = reference.curve(costs)
        shorter = costs[10/3]["when_idle"]
        direct, hw = reference.interval([small-long for small, long in zip(shorter, full["when_idle"])])
        assert math.isclose(curve["short_cost"], direct, abs_tol=1e-12)
        summary.update(point_first=curve["point"][0], point_sustained=curve["point"][1],
                       conservative_first=curve["conservative"][0], conservative_sustained=curve["conservative"][1],
                       short_cost=direct, short_half95=hw, short_share=curve["short_share"])
    return cell, summary, rows

def older_cohort(repo):
    selected = set()
    src = repo / "docs/papers/paper2/data/recheck_0_7_39_geometry.txt"
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("observable-key curve "):
            key, marker, _ = line.removeprefix("observable-key curve ").partition(") {")
            if not marker:
                raise ValueError(f"Malformed cohort in {src}")
            selected.add(ast.literal_eval(key+")"))
    if len(selected) != 19:
        raise ValueError("Expected the archived 19-cell retrospective cohort")
    return selected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6, choices=range(1,7))
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    if repo == out or repo in out.parents:
        parser.error("Pass an output directory outside the source checkout.")
    out.mkdir(parents=True, exist_ok=True)
    paths = {ext: out / (PREFIX + name) for ext, name in
             (("raw","episodes.csv.gz"), ("cells","cells.csv"), ("summary","summary.json"))}
    if any(path.exists() for path in paths.values()):
        parser.error("Previous output found. To reproduce, use a new output directory.")
    cells = [(*lane, vx, load, arr) for lane,vx,load,arr in
             product(LANES, (5,10), ("nominal","high"), ("poisson","burst"))]
    results = {}
    total_rows = 0
    print(f"Running {len(cells)} cells, {len(SEEDS)} paired seeds, {args.workers} workers", flush=True)
    with gzip.open(paths["raw"], "xt", encoding="utf-8", newline="", compresslevel=6) as raw:
        writer = csv.writer(raw)
        writer.writerow(FIELDS)
        with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(repo),)) as pool:
            jobs = [pool.submit(run_cell, cell) for cell in cells]
            for job in as_completed(jobs):
                cell, record, rows = job.result()
                results[cell] = record
                writer.writerows(rows)
                total_rows += len(rows)
                if len(results)%7 == 0 or len(results)==len(cells):
                    raw.flush()
                    print(f"{len(results)}/{len(cells)} cells complete; {total_rows} episodes saved", flush=True)
    old = older_cohort(repo)
    passed = {cell for cell, row in results.items() if row["selected"]}
    ids = lambda group: [list(cell) for cell in sorted(group)]
    for cell, row in results.items():
        row["prior_observable_selected"] = cell in old
    fields = list(FIELDS[:5])+["full_gain","full_gain_half95","selected","prior_observable_selected",
           "point_first","point_sustained","conservative_first","conservative_sustained",
           "short_cost","short_half95","short_share"]
    with paths["cells"].open("x",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fields)
        writer.writeheader()
        writer.writerows(results[key] for key in sorted(results))
    assert total_rows == (56 + 8*len(passed))*2*len(SEEDS)
    selected_rows = [results[k] for k in sorted(passed)]
    counters = {field: dict(Counter("undetermined" if r[field] is None else str(r[field]) for r in selected_rows))
                for field in ("point_first","point_sustained","conservative_first","conservative_sustained")}
    costs = [r["short_cost"] for r in selected_rows]
    worst = max(selected_rows,key=lambda r:r["short_cost"]) if costs else None
    counts = {key: sum(r[key] is not None and r[key]<10 for r in selected_rows) for key in counters}
    archive = repo / "experiments/recheck_0_7_39.py"
    report = dict(source_commit=COMMIT, reference_script_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  seed_first=SEEDS.start,seed_last=SEEDS.stop-1,seeds_per_cell=len(SEEDS),cell_count=len(cells),
                  rule="nearest_job, KEY_OBSERVED",grid_s=[0.5,1,1.5,2,2.5,10/3,5,7.5,10],
                  screen="G(10)>=1 and mean - 1.96*SE(paired G)>=0.5",crossing_z=2.773,
                  episode_rows=total_rows, selected_count=len(passed),prior_count=len(old),
                  intersection=ids(old&passed),new_only=ids(passed-old),prior_only=ids(old-passed),
                  crossing_counts=counters,early_crossing_counts=counts,
                  direct_cost_median=stats.median(costs) if costs else None,
                  worst_estimated_mean_shortening_cost_cell=worst)
    with paths["summary"].open("x", encoding="utf-8") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")
    print(f"Done. Selected {len(passed)}; overlap {len(old&passed)}; {counts}; worst cell {worst}",flush=True)
if __name__=="__main__":
    main()
