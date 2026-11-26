"""Re-run the two known-parts search tie-breaks on the archived Phase 4 episodes.

Run from an extracted checkout of 4892d46 with the dependencies in requirements-rl.txt.
Example: python -B /path/to/p1_search_tiebreak.py --workers 8
The CSV is written beside this script unless --output is supplied.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
ARCHIVE = ROOT / "docs/papers/paper1/data/phase4_search.csv"
FIELDS = ["load", "arrivals", "travel", "mode", "seed", "parts",
          "search_legacy_archived", "search_legacy_rerun", "search_observed",
          "network_archived", "search_observed_vs_network_pct",
          "search_observed_vs_legacy_pct", "abs_legacy_discrepancy"]
GROUP = ["load", "arrivals", "travel", "mode"]

def init_worker():
    import torch
    torch.set_num_threads(1)

def pair(job):
    from fastsim import FastSim
    from fastsim.policies import KEY_ARRIVAL, KEY_OBSERVED
    from rl2.expert_iteration import val_config
    from rl2.search import SearchTeacher
    from rl2_eval import _extra
    load, arrivals, travel, mode, seed, archived, net, parts = job
    options = dict(load=load, arrivals=arrivals, travel=travel, slow=0.5,
                   mode="none" if mode == "B" else "when_idle", obs="belts")
    config = val_config(load, seed, "none", extra=tuple(_extra(options)))
    def score(key):
        return FastSim(config).run(SearchTeacher(None, k=6, tie_key=key))["penalized_flow_time"]
    old = score(KEY_ARRIVAL)
    discrepancy = abs(old - archived)
    if discrepancy > 1e-7:
        raise ValueError(f"Archived search cost mismatch {job[:5]}: rerun={old}, archived={archived}")
    observed = score(KEY_OBSERVED)
    return dict(zip(FIELDS, [load,arrivals,travel,mode,seed,parts,archived,old,observed,net,
                             100*(observed/net-1),100*(observed/old-1),discrepancy]))

def normal_interval(values):
    average = mean(values)
    half = 1.96*stdev(values)/math.sqrt(len(values))
    return average, average-half, average+half

def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP)].append(row)
    blocks = []
    for key, subgroup in sorted(groups.items()):
        if len(subgroup) != 30:
            raise ValueError(f"Unexpected size for {key}: {len(subgroup)}")
        record = dict(zip(GROUP, key))
        for f in ("search_observed_vs_network_pct", "search_observed_vs_legacy_pct"):
            m,lb,ub = normal_interval([r[f] for r in subgroup])
            record[f+"_mean"],record[f+"_lower95"],record[f+"_upper95"] = m,lb,ub
        record["observable_search_cheaper_than_net"] = sum(r["search_observed"]<r["network_archived"] for r in subgroup)
        record["normal_two_sided_p_search_vs_net"] = math.erfc(
            abs(record["search_observed_vs_network_pct_mean"]) * 1.96 / math.sqrt(2)
            / (record["search_observed_vs_network_pct_upper95"] - record["search_observed_vs_network_pct_mean"]))
        blocks.append(record)
    for mode in ("B","C"):
        family = sorted([r for r in blocks if r["mode"]==mode],key=lambda r:r["normal_two_sided_p_search_vs_net"])
        running = 0.
        for i,r in enumerate(family):
            running = max(running, min(1.,(len(family)-i)*r["normal_two_sided_p_search_vs_net"]))
            r["holm_adjusted_p_within_mode"] = running
    return blocks

def write_csv(path, records):
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    data=defaultdict(dict)
    with ARCHIVE.open(newline="",encoding="utf-8") as file:
        for r in csv.DictReader(file):
            if r["who"] not in ("search","net"):
                continue
            key=(r["load"],r["arrivals"],r["travel"],r["t"],int(r["seed"]))
            data[key][r["who"]]=float(r["cost"])
            data[key]["parts"]=int(r["parts"])
    if len(data)!=480 or {k[-1] for k in data}!=set(range(7000,7030)):
        raise ValueError("Unexpected experiment/seed coverage")
    jobs=[(*k,v["search"],v["net"],v["parts"]) for k,v in sorted(data.items())]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        for i,row in enumerate(pool.map(pair,jobs,chunksize=2),start=1):
            rows.append(row)
            if i%60==0: print(f"Validated/paired {i}/{len(jobs)}",flush=True)
    args.output.mkdir(exist_ok=True,parents=True)
    blocks=summary(rows)
    write_csv(args.output/"p1_search_paired_seeds.csv",rows)
    write_csv(args.output/"p1_search_summary.csv",blocks)
    print(f"max_abs_legacy_error={max(r['abs_legacy_discrepancy'] for r in rows):.12g}",flush=True)
    for r in blocks:
        print(*(r[k] for k in GROUP),*(round(r[k],5) for k in ["search_observed_vs_network_pct_mean",
              "search_observed_vs_network_pct_lower95","search_observed_vs_network_pct_upper95",
              "search_observed_vs_legacy_pct_mean","holm_adjusted_p_within_mode"]),flush=True)
if __name__=="__main__":
    main()

