"""Information bench: treatments A/B/C/D and the visible-length curve across regimes.

Measures, with one strong-RL network (``rl2``) and the reference rules, the value of
the information the cell gives the controller (``docs/plans/papers.md``, Paper 2):

* treatments: A = ``zone``/``none``, B = ``belts``/``none``, C = ``belts``/``when_idle``,
  D = ``full``/``none``;
* the visible-length curve: ``belts`` with ``simulation.visible_length`` in
  ``--visible-lengths`` (cm of the entry zone the camera sees), with ``none`` and with
  ``when_idle`` (B and C are the points with the whole entry zone visible);
* regimes: load (the real cell and the heavy example, validation seeds 2000+),
  arrivals (``regular`` = Poisson, ``burst`` = on/off Poisson source with the same mean
  rate, 0.7.10) and travel (``nominal`` laser speeds or ``slow`` = speeds × ``--slow``).

Every configuration of a regime is run on the same seeds (common random numbers), so
every contrast is paired. Output: ``results/experiments/<tag>/bench.json`` (every run)
and ``tables.md`` (paired Δ of the penalized flow time vs ``nearest_job`` in B of the
same regime, and the network's contrasts between treatments).

    python experiments/rl2_information_bench.py --net models/rl2_gen_s_d.pt --tag info_pilot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl2.expert_iteration import (BURST_VAL, HEAVY, VAL_SEED_BASE,  # noqa: E402
                                  _init_worker, val_config)

TREATMENTS = {"A": ("zone", "none"), "B": ("belts", "none"),
              "C": ("belts", "when_idle"), "D": ("full", "none")}
BURST = BURST_VAL        # the bursty setting of the validation group real_burst

_NET = {}


def _policy(name: str, path: str):
    from fastsim import POLICIES
    if name in POLICIES:
        return POLICIES[name]
    if path not in _NET:
        from rl2.model import NetPolicy, load
        _NET[path] = NetPolicy(load(path))
    return _NET[path]


def _laser_speed(load: str) -> tuple[float, float]:
    from simulation.config import load_config
    c = load_config(HEAVY if load == "heavy" else None)
    return c.laser.speed_x, c.laser.speed_y


def run(job: dict) -> dict:
    from fastsim import FastSim
    extra = [f"simulation.observation={job['obs']}", f"laser.prepositioning={job['mode']}"]
    if job["visible"] is not None:
        extra.append(f"simulation.visible_length={job['visible']}")
    if job["arrivals"] == "burst":
        extra += BURST
    if job["travel"] == "slow":
        vx, vy = _laser_speed(job["load"])
        extra += [f"laser.speed_x={vx * job['slow']:.4f}", f"laser.speed_y={vy * job['slow']:.4f}"]
    cfg = val_config(job["load"], job["seed"], "none", extra=tuple(extra))
    sim = FastSim(cfg)
    s = sim.run(_policy(job["who"], job["net"]))
    return job | {"cost": s["penalized_flow_time"], "blocks": s["blocking_events"],
                  "parts": len(sim.arr)}


def _ci(d):
    d = np.asarray(d, dtype=float)
    return 100 * d.mean(), 100 * 1.96 * d.std(ddof=1) / math.sqrt(len(d))


def _cell(mu, ci):
    return f"{mu:+.2f} ±{ci:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--net", default="models/rl2_gen_s_d.pt")
    ap.add_argument("--tag", default="info_bench")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--loads", nargs="+", default=["real", "heavy"])
    ap.add_argument("--arrivals", nargs="+", default=["regular", "burst"])
    ap.add_argument("--travel", nargs="+", default=["nominal", "slow"])
    ap.add_argument("--slow", type=float, default=0.5, help="laser speed factor of 'slow'")
    ap.add_argument("--visible-lengths", nargs="*", type=float, default=[0.0, 2.5, 5.0])
    ap.add_argument("--policies", nargs="+", default=["nearest_job", "nearest_head", "net"])
    ap.add_argument("--workers", type=int, default=47)
    args = ap.parse_args()

    out_dir = ROOT / "results/experiments" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [{"t": t, "obs": o, "mode": m, "visible": None} for t, (o, m) in TREATMENTS.items()]
    for lv in args.visible_lengths:
        for m in ("none", "when_idle"):
            configs.append({"t": f"L{lv:g}/{m}", "obs": "belts", "mode": m, "visible": lv})
    jobs = [{"load": g, "arrivals": a, "travel": tr, "slow": args.slow, "seed": VAL_SEED_BASE + i,
             "who": w, "net": args.net} | c
            for g in args.loads for a in args.arrivals for tr in args.travel
            for i in range(args.seeds) for c in configs for w in args.policies]
    print(f"{len(jobs)} runs, {len(configs)} configurations per regime", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        res = list(ex.map(run, jobs, chunksize=4))
    json.dump(res, open(out_dir / "bench.json", "w"))

    lines = [f"# Information bench `{args.tag}` — net `{args.net}`, {args.seeds} seeds per regime", ""]
    regimes = [(g, a, tr) for g in args.loads for a in args.arrivals for tr in args.travel]
    names = [c["t"] for c in configs]
    for g, a, tr in regimes:
        sub = [r for r in res if (r["load"], r["arrivals"], r["travel"]) == (g, a, tr)]
        ref = {r["seed"]: r["cost"] for r in sub if r["who"] == "nearest_job" and r["t"] == "B"}
        parts = np.mean([r["parts"] for r in sub if r["who"] == "nearest_job" and r["t"] == "B"])
        ref_cost = np.mean(list(ref.values()))
        lines += [f"## {g} / {a} arrivals / {tr} travel "
                  f"(mean {parts:.0f} parts, `nearest_job` in B: {ref_cost:.1f} s/part)", "",
                  "Paired Δ of the penalized flow time vs `nearest_job` in B [%]:", "",
                  "| policy | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for w in args.policies:
            cells = []
            for t in names:
                d = [r["cost"] / ref[r["seed"]] - 1 for r in sub if r["who"] == w and r["t"] == t]
                cells.append(_cell(*_ci(d)))
            lines.append(f"| `{w}` | " + " | ".join(cells) + " |")
        lines.append("")
        for w in args.policies:
            byt = {t: {r["seed"]: r["cost"] for r in sub if r["who"] == w and r["t"] == t}
                   for t in names}
            con = []
            for x, y in (("A", "B"), ("B", "C"), ("B", "D")):
                d = [byt[y][s] / byt[x][s] - 1 for s in byt[x]]
                con.append(f"{y} vs {x} {_cell(*_ci(d))}")
            lines.append(f"* `{w}` contrasts: " + "; ".join(con))
        lines.append("")
    (out_dir / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
