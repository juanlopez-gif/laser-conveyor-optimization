"""Greedy vs sampled decoding of checkpoints on the validation realizations (docs/methods/rl2_ppo.md §3.5):
    python experiments/rl2_ppo_anomaly.py models/rl2_ppo_s1_it16.pt models/rl2_ppo_s1_it40.pt models/rl2_ppo_s0_it40.pt models/rl2_gen_s_b.pt
(investigation of the seed-1 anomaly: greedy validation worsens after update 16
while the sampled training cost keeps falling)."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from rl2.expert_iteration import VAL_GROUPS, _init_worker, val_config, _net
from rl2.ppo import val_jobs, paired, fmt

def episode(job):
    import torch
    from fastsim import POLICIES, FastSim
    from rl2.model import NetPolicy
    cfg = val_config(job["load"], job["seed"], "design")
    if job["who"] == "net":
        g = torch.Generator().manual_seed(job["seed"] * 10 + job["rep"])
        pol = NetPolicy(_net(job["net"]), sample=job["rep"] > 0, generator=g)
    else:
        pol = POLICIES[job["who"]]
    s = FastSim(cfg).run(pol)
    return {"load": job["load"], "seed": job["seed"], "rep": job["rep"], "cost": s["penalized_flow_time"]}

def run(ex, who, net, reps):
    jobs = [{**j, "rep": r} for j in val_jobs(50, who, net, "design") for r in reps]
    res = list(ex.map(episode, jobs, chunksize=1))
    out = {}
    for r in res:
        out.setdefault((r["load"], r["seed"]), []).append(r["cost"])
    return {k: float(np.mean(v)) for k, v in out.items()}

if __name__ == "__main__":
    out = ROOT / "results/experiments/rl2_ppo_anomaly_s1"; out.mkdir(parents=True, exist_ok=True)
    lines = ["| checkpoint | decoding | " + " | ".join(VAL_GROUPS) + " | score |", "|---|---|" + "---|" * 5]
    res = {}
    with ProcessPoolExecutor(40, initializer=_init_worker) as ex:
        nj = run(ex, "nearest_job", None, [0])
        for ck in sys.argv[1:]:
            p = str(ROOT / ck)
            for name, reps in (("greedy", [0]), ("sampled (mean of 4)", [1, 2, 3, 4])):
                c = run(ex, "net", p, reps)
                v = paired(c, nj); res[f"{ck} {name}"] = v
                print(ck, name, fmt(v), flush=True)
                lines.append(f"| {Path(ck).stem} | {name} | " + " | ".join(
                    f"{100*v[g]['delta']:+.2f} ±{100*v[g]['ci']:.2f}" for g in VAL_GROUPS) + f" | **{100*v['score']:+.2f}** |")
    (out / "anomaly.json").write_text(json.dumps(res, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
