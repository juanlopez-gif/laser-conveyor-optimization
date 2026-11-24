"""Per-decision latency of a checkpoint (docs/methods/rl2_ppo.md §3.4): single state, 1 thread, median / p95, as rl2_latency.py.
    python experiments/rl2_ppo_latency.py models/rl2_ppo_s2.pt"""
import json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import torch
from rl2.model import collate, load, NetPolicy
from rl2_latency import states, pct, HEAVY
from fastsim import FastSim
from simulation.config import load_config

net = load(ROOT / sys.argv[1]).eval()
row = {"model": sys.argv[1], "params": net.n_params()}
torch.set_num_threads(1)
with torch.inference_mode():
    for name, path, seed in (("default", None, 6100), ("heavy", HEAVY, 6101)):
        obs, enc = states(path, seed)
        singles = [collate([o]) for o in obs[:100]]
        for b in singles[:5]: net(b)
        ts = []
        for b in singles:
            t0 = time.perf_counter(); net(b); ts.append(time.perf_counter() - t0)
        row[f"{name}_single_ms_median"] = statistics.median(ts) * 1e3
        row[f"{name}_single_ms_p95"] = pct(ts, 0.95) * 1e3
        row[f"{name}_encode_ms_median"] = statistics.median(enc) * 1e3
    cfg = load_config(None, set_items=["arrivals.mode=stochastic", "simulation.seed=6102"])
    sim = FastSim(cfg); t0 = time.perf_counter(); sim.run(NetPolicy(net))
    row["episode_decisions_per_s_1thread"] = sim.n_decisions / (time.perf_counter() - t0)
out = ROOT / "results/experiments/rl2_ppo_latency"; out.mkdir(parents=True, exist_ok=True)
(out / "latency.json").write_text(json.dumps(row, indent=2))
print(json.dumps(row, indent=2))
