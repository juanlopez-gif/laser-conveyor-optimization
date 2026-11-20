"""Speed of the fast simulator vs the fixed-step engine (strong-RL plan, phase 1).

Single process, same instances, rule ``nearest_job``. Reports decisions per second
(a decision = one assignment of the laser) for the engine at ``dt = 0.05`` (the step
used to generate RL training data) and ``dt = 0.01`` (evaluation), and for
``fastsim``; and the cost of ``clone()``.

    python experiments/fastsim_benchmark.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastsim import POLICIES, FastSim  # noqa: E402
from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.events import EventKind  # noqa: E402

LOADS = {"default": None, "heavy": str(ROOT / "config" / "examples" / "heavy_load.yaml")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=6000)
    args = ap.parse_args(argv)
    rows = []
    for load, path in LOADS.items():
        res = {}
        for name, dt in (("engine dt=0.05", 0.05), ("engine dt=0.01", 0.01), ("fastsim", None)):
            n_dec, wall = 0, 0.0
            for s in range(args.seed_base, args.seed_base + args.episodes):
                cfg = load_config(path, set_items=[
                    "arrivals.mode=stochastic", f"simulation.seed={s}",
                    "policy.name=nearest_job", f"simulation.dt={dt or 0.01}"])
                t0 = time.perf_counter()
                if dt is None:
                    sim = FastSim(cfg)
                    sim.run(POLICIES["nearest_job"])
                    n_dec += sim.n_decisions
                else:
                    r = Simulation(cfg, record_trace=False).run()
                    n_dec += sum(1 for e in r.events.events if e.kind == EventKind.ASSIGN)
                wall += time.perf_counter() - t0
            res[name] = (n_dec, wall)
        # clone cost in the middle of an episode
        cfg = load_config(path, set_items=["arrivals.mode=stochastic",
                                           f"simulation.seed={args.seed_base}"])
        sim = FastSim(cfg)
        for _ in range(20):
            sim.run_until_decision()
            sim.apply(POLICIES["nearest_job"](sim, sim.candidates()))
        sim.run_until_decision()
        t0 = time.perf_counter()
        for _ in range(2000):
            sim.clone()
        clone_us = (time.perf_counter() - t0) / 2000 * 1e6
        base = res["engine dt=0.05"][0] / res["engine dt=0.05"][1]
        for name, (n, w) in res.items():
            rows.append(f"| {load} | {name} | {n} | {w:.2f} | {n / w:.0f} | {n / w / base:.1f}× |")
        rows.append(f"| {load} | fastsim `clone()` | – | – | {clone_us:.0f} µs per clone | – |")
    text = ("| load | simulator | decisions | wall [s] | decisions / s | vs engine dt=0.05 |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print(text)
    out = ROOT / "results" / "experiments" / "fastsim_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(f"# fastsim benchmark ({args.episodes} episodes per load, "
                                    "single process)\n\n" + text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
