"""Latency of the strong-RL model (plan §3 acceptance; docs/methods/rl2.md §4).

For each size S / M / L (untrained weights; latency does not depend on them):
* parameters;
* feature encoding time per decision (FastSim state -> tokens);
* network forward for one state, 1 thread (the controller case), median and p95;
* batched forward throughput (states/s) with B = 64, all threads (training/search case);
* decisions per second of a full FastSim episode driven by the network (S only).

Usage:  python experiments/rl2_latency.py [--sizes S M L] [--threads 8] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from fastsim import POLICIES, FastSim  # noqa: E402
from rl2.features import encode  # noqa: E402
from rl2.model import SIZES, DispatchNet, NetPolicy, collate  # noqa: E402
from simulation.config import load_config  # noqa: E402

HEAVY = str(ROOT / "config/examples/heavy_load.yaml")


def states(path, seed, n=200):
    cfg = load_config(path, set_items=["arrivals.mode=stochastic", f"simulation.seed={seed}"])
    sim, out, enc_t = FastSim(cfg), [], []
    rule = POLICIES["nearest_job"]
    while sim.run_until_decision() and len(out) < n:
        c = sim.candidates()
        if len(c) > 1:
            t0 = time.perf_counter()
            out.append(encode(sim, c))
            enc_t.append(time.perf_counter() - t0)
        sim.apply(rule(sim, c))
    return out, enc_t


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results/experiments/rl2_latency"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, path, seed in (("default", None, 6100), ("heavy", HEAVY, 6101)):
        obs, enc = states(path, seed)
        data[name] = obs
        print(f"{name}: {len(obs)} states, tokens/state median "
              f"{statistics.median(o.part.shape[0] + 5 for o in obs)}, "
              f"encode {statistics.median(enc) * 1e3:.3f} ms median")
        data[name + "_enc"] = enc

    rows = []
    for size in args.sizes:
        torch.manual_seed(0)
        net = DispatchNet(SIZES[size]).eval()
        row = {"size": size, "params": net.n_params()}
        with torch.inference_mode():
            for load in ("default", "heavy"):
                obs = data[load]
                torch.set_num_threads(1)
                singles = [collate([o]) for o in obs[:100]]
                for b in singles[:5]:
                    net(b)
                ts = []
                for b in singles:
                    t0 = time.perf_counter()
                    net(b)
                    ts.append(time.perf_counter() - t0)
                row[f"{load}_single_ms_median"] = statistics.median(ts) * 1e3
                row[f"{load}_single_ms_p95"] = pct(ts, 0.95) * 1e3
                row[f"{load}_encode_ms_median"] = statistics.median(data[load + "_enc"]) * 1e3
                torch.set_num_threads(args.threads)
                batch = collate(obs[:64])
                net(batch)
                reps = 10 if size != "L" else 3
                t0 = time.perf_counter()
                for _ in range(reps):
                    net(batch)
                row[f"{load}_batch64_states_per_s"] = 64 * reps / (time.perf_counter() - t0)
        if size == "S":
            torch.set_num_threads(1)
            cfg = load_config(None, set_items=["arrivals.mode=stochastic", "simulation.seed=6102"])
            sim = FastSim(cfg)
            t0 = time.perf_counter()
            sim.run(NetPolicy(net))
            row["episode_decisions_per_s_1thread"] = sim.n_decisions / (time.perf_counter() - t0)
        rows.append(row)
        print(json.dumps(row))

    (out / "latency.json").write_text(json.dumps(rows, indent=2))
    lines = ["| size | parameters | single state, 1 thread: median / p95 [ms] (default) | "
             "(heavy) | encode [ms] | batch 64, %d threads [states/s] (default / heavy) |"
             % args.threads, "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['params'] / 1e6:.2f} M | "
            f"{r['default_single_ms_median']:.2f} / {r['default_single_ms_p95']:.2f} | "
            f"{r['heavy_single_ms_median']:.2f} / {r['heavy_single_ms_p95']:.2f} | "
            f"{r['default_encode_ms_median']:.2f} / {r['heavy_encode_ms_median']:.2f} | "
            f"{r['default_batch64_states_per_s']:.0f} / {r['heavy_batch64_states_per_s']:.0f} |")
    if "episode_decisions_per_s_1thread" in rows[0]:
        lines.append("")
        lines.append(f"Full FastSim episode driven by S (1 thread): "
                     f"{rows[0]['episode_decisions_per_s_1thread']:.0f} decisions/s.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
