"""Compare search-teacher variants over a trained network on the validation cells
(docs/methods/rl2.md §6.2 and §8.5).

Each variant drives the fast simulator on the validation groups (``--groups``, default
all of ``rl2.expert_iteration.VAL_GROUPS``: the real cell, the heavy example and the
real cell with bursty arrivals on seeds 2000+, general cells on seeds 2500+), in the
``--val-mode``; the table gives the paired Δ of penalized flow time vs ``nearest_job``
per group, the mean over the groups, the Δ vs the network alone, and the time per
episode. ``--extra key=value ...`` overrides the configuration of every cell (applied
last), e.g. another burst setting.

    python experiments/rl2_teacher_variants.py --net models/rl2_gen_s_big.pt --val 30 \\
        --variants known:k6 known:k0 trunc:h20:m2:p1 ...

Variant syntax (0.7.6, ``rl2.search``): ``known`` (look-ahead until the known parts
are done), ``roll`` (sampled-future rollout to the end of the window, no value head),
``cut`` (value head at exactly t + H) or ``dcut`` (value head at the first decision
after t + H); then ``:kK`` candidates (0 = all), ``:hH`` horizon [s], ``:wW`` window
[s] of ``roll``, ``:mM`` scenarios, ``:pW`` penalty weight, ``:fF`` futures
(``poisson`` = Poisson at the mean rates, ``cell`` = the cell's arrival model with
its on/off source, 0.7.12). Defaults: k6, h30, w30, m2, p1, fpoisson (``known``: m0,
p0, the 0.7.2 teacher).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl2.expert_iteration import (VAL_GENERAL_SEED_BASE, VAL_GROUPS,  # noqa: E402
                                  VAL_SEED_BASE, _init_worker, val_episode)


def parse_variant(spec: str) -> dict:
    parts = spec.split(":")
    kind = parts[0]
    assert kind in ("known", "roll", "cut", "dcut"), spec
    v = {"name": spec, "k": 6, "h": 30.0, "w": 30.0, "m": 2, "p": 1.0, "f": "poisson"}
    for p in parts[1:]:
        key, val = p[0], p[1:]
        v[key] = type(v[key])(val)
    if kind == "known":
        v["la"] = (0.0, 0, "time", 30.0)
        v["p"] = v["p"] if ":p" in spec else 0.0          # the 0.7.2 teacher has no penalty
        v["m"] = v["m"] if ":m" in spec else 0
    elif kind == "roll":
        v["la"] = (math.inf, v["m"], "time", v["w"])
    else:
        v["la"] = (v["h"], v["m"], "time" if kind == "cut" else "decision", 30.0)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--val", type=int, default=30, help="seeds per group (general: 2x)")
    ap.add_argument("--variants", nargs="+", default=["known:k6", "trunc:h30:m2:p1"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--tag", default="rl2_teacher_variants")
    ap.add_argument("--val-mode", default="design", choices=["design", "none"],
                    help="validation mode (rl2.expert_iteration.VAL_MODES, 0.7.7)")
    ap.add_argument("--groups", nargs="+", default=list(VAL_GROUPS), choices=VAL_GROUPS)
    ap.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE",
                    help="configuration overrides applied to every cell, last")
    args = ap.parse_args(argv)
    groups = list(args.groups)
    cells = [(g, (VAL_GENERAL_SEED_BASE if g == "general" else VAL_SEED_BASE) + i)
             for g in groups for i in range(args.val * (2 if g == "general" else 1))]
    variants = [parse_variant(s) for s in args.variants]
    net = str(ROOT / args.net)
    m = args.val_mode
    base = {"horizon": None, "mode": m, "extra": tuple(args.extra)}
    jobs = [base | {"load": g, "seed": sd, "who": "nearest_job", "net": None, "k": 6,
                    "w_pen": 0.0} for g, sd in cells]
    jobs += [base | {"load": g, "seed": sd, "who": "net", "net": net, "k": 6, "w_pen": 0.0}
             for g, sd in cells]
    for vi, v in enumerate(variants):
        jobs += [base | {"load": g, "seed": sd, "who": "teacher", "net": net, "k": v["k"],
                         "w_pen": v["p"], "la": v["la"], "futures": v["f"], "variant": vi}
                 for g, sd in cells]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        res = list(ex.map(_run, jobs, chunksize=1))
    ref = {(r["load"], r["seed"]): r["cost"] for r in res if r["who"] == "nearest_job"}
    netc = {(r["load"], r["seed"]): r["cost"] for r in res if r["who"] == "net"}

    def stats(rows, base):
        rs = [r["cost"] / base[(r["load"], r["seed"])] - 1 for r in rows]
        return (float(np.mean(rs)), 1.96 * float(np.std(rs, ddof=1)) / math.sqrt(len(rs)))

    out = {"net": args.net, "val": args.val, "val_mode": m, "groups": groups,
           "extra": list(args.extra), "rows": []}
    header = ("| variant | " + " | ".join(groups) + " | score | vs net alone (score) | "
              "s / episode (" + " / ".join(groups) + ") |")
    print(header); print("|---|" + "---|" * (len(groups) + 3))
    for name, rows_of in [("net alone", lambda: [r for r in res if r["who"] == "net"])] + [
            (v["name"], (lambda vi=vi: [r for r in res if r.get("variant") == vi]))
            for vi, v in enumerate(variants)]:
        rows = rows_of()
        line = {"variant": name}
        cols = []
        for g in groups:
            gr = [r for r in rows if r["load"] == g]
            mu, ci = stats(gr, ref)
            line[g] = {"delta": mu, "ci": ci, "secs": float(np.mean([r["secs"] for r in gr]))}
            cols.append(f"{100 * mu:+.2f} ±{100 * ci:.2f}")
        line["score"] = float(np.mean([line[g]["delta"] for g in groups]))
        vs = float(np.mean([stats([r for r in rows if r["load"] == g], netc)[0] for g in groups]))
        line["vs_net"] = vs
        out["rows"].append(line)
        secs = " / ".join(f"{line[g]['secs']:.2f}" for g in groups)
        print(f"| {name} | " + " | ".join(cols) + f" | {100 * line['score']:+.2f} | "
              f"{100 * vs:+.2f} | {secs} |")
    # per-episode rows (0.7.12), so any pair of variants can be contrasted afterwards
    names = {None: None} | {vi: v["name"] for vi, v in enumerate(variants)}
    out["episodes"] = [{"load": r["load"], "seed": r["seed"], "who": r["who"],
                        "variant": names[r.get("variant")], "cost": r["cost"],
                        "secs": r["secs"]} for r in res]
    out["elapsed_s"] = time.perf_counter() - t0
    rdir = ROOT / "results" / "experiments" / args.tag
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "variants.json").write_text(json.dumps(out, indent=2))
    print(f"elapsed {out['elapsed_s']:.0f} s; written {rdir / 'variants.json'}")
    return 0


def _run(job):
    r = val_episode(job)
    r["variant"] = job.get("variant")
    return r


if __name__ == "__main__":
    raise SystemExit(main())
