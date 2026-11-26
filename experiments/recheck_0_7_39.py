"""Retrospective review of 2b895f5 (external review; archived with version 0.7.40).
Run from the repository root.

python /path/to/recheck_0_7_39.py --study reference
python /path/to/recheck_0_7_39.py --study geometry
"""
import argparse
import csv
import gzip
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from fastsim import FastSim
from fastsim.policies import KEY_ARRIVAL, KEY_OBSERVED, nearest_job
from simulation.config import load_config

GRID = (0.5, 1., 1.5, 2., 2.5, 10 / 3, 5., 7.5, 10.)
BURST = ["arrivals.stochastic.distribution=burst",
         "arrivals.stochastic.burst.on_fraction=0.3",
         "arrivals.stochastic.burst.on_mean=20"]
SEEDS = range(9000, 9060)


def interval(values, z=1.96):
    return stats.mean(values), z * stats.stdev(values) / math.sqrt(len(values))


def run(config, seed, key=KEY_OBSERVED):
    seeded = replace(config, simulation=replace(config.simulation, seed=seed))
    return FastSim(seeded).run(partial(nearest_job, key=key))


def reference():
    archive = defaultdict(dict)
    with open(ROOT / "docs/papers/paper1/data/phase4_search.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["who"] in ("net", "nearest_job"):
                k = (r["load"], r["arrivals"], r["travel"], r["t"])
                archive[k][int(r["seed"]), r["who"]] = float(r["cost"])
    for k, case in sorted(archive.items()):
        load, arrivals, travel, treatment = k
        base = str(ROOT / "config/examples/heavy_load.yaml") if load == "heavy" else None
        mode = "none" if treatment == "B" else "when_idle"
        extra = ["arrivals.mode=stochastic", "simulation.observation=belts",
                 "laser.prepositioning=" + mode]
        if arrivals == "burst":
            extra.extend(BURST)
        if travel == "slow":
            laser = load_config(base).laser
            extra.extend([f"laser.speed_x={laser.speed_x * 0.5}",
                          f"laser.speed_y={laser.speed_y * 0.5}"])
        config = load_config(base, set_items=extra)
        diffs = []
        for seed in range(7000, 7030):
            old = run(config, seed, KEY_ARRIVAL)["penalized_flow_time"]
            assert math.isclose(old, case[seed, "nearest_job"], abs_tol=1e-7, rel_tol=0)
            new = run(config, seed)["penalized_flow_time"]
            diffs.append(100 * (case[seed, "net"] / new - 1))
        print("network vs observable rule", k, "mean and interval half-width", interval(diffs), flush=True)


def geometry_config(k, mode, seconds):
    d, vy, vx, load, arrivals = k
    intervals = "[10,12,14]" if load == "nominal" else "[7,8.5,10]"
    extra = ["arrivals.mode=stochastic", "simulation.observation=belts",
             f"laser.prepositioning={mode}", f"simulation.visible_length={round(3*seconds, 6)}",
             "conveyors.y_positions=null", f"conveyors.spacing={d}",
             f"laser.speed_x={vx}", f"laser.speed_y={vy}", "geometry.entry_length=35",
             f"arrivals.stochastic.mean_interarrival={intervals}", "processing.time=2.5"]
    if arrivals == "burst":
        extra.extend(BURST)
    return load_config(None, set_items=extra)


def benefits(costs):
    return [a-b for a, b in zip(costs["none"], costs["when_idle"])]


def screen(gains):
    mean, half = interval(gains)
    return mean >= 1.0 and mean - half >= 0.5


def crossings(flags):
    hits = [s for s, ok in zip(GRID, flags) if ok]
    sustained = [s for i, s in enumerate(GRID) if all(flags[i:])]
    return (hits[0] if hits else None, sustained[0] if sustained else None)


def curve(costs):
    full = benefits(costs[10.])
    point, conservative = [], []
    for s in GRID:
        diffs = [g-0.9*f for g, f in zip(benefits(costs[s]), full)]
        mean, half = interval(diffs, 2.773)
        point.append(mean >= 0)
        conservative.append(mean - half >= 0)
    short = costs[10/3]
    direct = stats.mean(a-b for a, b in zip(short["when_idle"], costs[10.]["when_idle"]))
    share = stats.mean(benefits(short)) / stats.mean(full)
    return {"point": crossings(point), "conservative": crossings(conservative),
            "short_cost": direct, "short_share": share, "gain": stats.mean(full)}


def extract_historical():
    archive = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    path = ROOT / "docs/papers/paper2/data/geometry_bench.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            d, vy, vx = [int(float(r[c])) for c in ("spacing", "vy", "vx")]
            if r["bench"] != "grid" or r["who"] != "nearest_job" or (d, vy) in ((6, 5), (18, 15)):
                continue
            s = float(r["s"])
            if s > 0:
                s = min(GRID, key=lambda grid_s: abs(s-grid_s))
                k = (d, vy, vx, r["load"], r["arrivals"])
                archive[k][s][r["mode"]][int(r["seed"])] = float(r["cost"])
    return {k: {s: {mode: [by_seed[seed] for seed in SEEDS]
                    for mode, by_seed in modes.items()} for s, modes in case.items()}
            for k, case in archive.items()}


def simulate_geometry(k, s, check=None):
    costs, moves = {}, []
    for mode in ("none", "when_idle"):
        config = geometry_config(k, mode, s)
        costs[mode] = []
        for i, seed in enumerate(SEEDS):
            result = run(config, seed)
            costs[mode].append(result["penalized_flow_time"])
            if mode == "when_idle" and s == 10 and result["prepos_t_mean"] is not None:
                moves.append(result["prepos_t_mean"])
            if check is not None:
                old = run(config, seed, KEY_ARRIVAL)["penalized_flow_time"]
                assert math.isclose(old, check[mode][i], abs_tol=1e-7, rel_tol=0)
    return costs, moves


def prediction_errors(selected):
    errors = defaultdict(list)
    for k, held in selected.items():
        rest = [r for key, r in selected.items() if key != k]
        grouped = [r["point"][0] for r in rest if r["vx"] == held["vx"]]
        fitted = sum(r["move"]*r["point"][0] for r in rest) / sum(r["move"]**2 for r in rest)
        predictions = {"constant": stats.mean(r["point"][0] for r in rest),
                       "same_vx": stats.mean(grouped), "fitted_move": fitted*held["move"],
                       "crossing_time": 2*k[0]/k[1]}
        for label, prediction in predictions.items():
            errors[label].append(abs(held["point"][0]-prediction))
    return {label: stats.mean(err) for label, err in errors.items()}


def geometry():
    archived = extract_historical()
    assert len(archived) == 56
    membership = Counter()
    selected = {}
    for k, old in sorted(archived.items()):
        gold = screen(benefits(old[10.]))
        checks = k in ((18, 5, 5, "nominal", "burst"), (6, 15, 10, "high", "burst"))
        full, moves = simulate_geometry(k, 10, old[10.])
        gnew = screen(benefits(full))
        membership[gold, gnew] += 1
        print("full-view screen", k, gold, gnew, flush=True)
        if gold:
            print("published-key curve", k, curve(old))
        if gnew:
            costs = {10.: full}
            for s in GRID[:-1]:
                costs[s], _ = simulate_geometry(k, s, old[s] if checks and s == 10/3 else None)
            result = curve(costs)
            result.update(move=stats.mean(moves), vx=k[2])
            selected[k] = result
            print("observable-key curve", k, result, flush=True)
    print("historical/observable membership", dict(membership))
    print("conservative first crossings", Counter(r["conservative"][0] for r in selected.values()))
    print("direct shortening cost, median and maximum",
          stats.median(r["short_cost"] for r in selected.values()),
          max(r["short_cost"] for r in selected.values()))
    print("leave-one-cell-out error (seconds)", prediction_errors(selected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("reference", "geometry", "both"), default="both")
    study = parser.parse_args().study
    if study in ("reference", "both"):
        reference()
    if study in ("geometry", "both"):
        geometry()
