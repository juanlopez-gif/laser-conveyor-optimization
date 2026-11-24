"""Training-cost accounting: PPO runs against the expert-iteration chain
(docs/methods/rl2_ppo.md §1.6).

* PPO: from ``models/<tag>_training.json`` (gen_s, train_s, val_s, CPU seconds measured
  in the workers, episodes, decisions).
* Expert iteration: from ``models/<tag>_training.json`` of the chain (gen_s, train_s,
  val_s per iteration, reference validations, workers). The number of look-ahead
  clones run by the teacher is not logged, so it is measured here on a sample of
  episodes with the same teacher settings (seeds 190000+, never the reserved seeds)
  and scaled by the logged number of labelled decisions.

    python experiments/rl2_ppo_cost.py --ppo rl2_ppo_s0 rl2_ppo_s1 rl2_ppo_s2 \
        --chain rl2_gen_s_big rl2_gen_s_sf rl2_gen_s_d rl2_gen_s_b --workers 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl2.expert_iteration import VAL_GROUPS, _init_worker, play_episode  # noqa: E402

SAMPLE_SEED_BASE = 190_000


def measure_teacher(ex, net: str, la: tuple, n: int, k: int = 6) -> dict:
    """Look-ahead clones and labelled decisions per episode for a teacher setting."""
    jobs = [{"seed": SAMPLE_SEED_BASE + e, "net": net, "k": k, "w_pen": 0.0, "beta": 0.3,
             "eps": 0.1, "la": la} for e in range(n)]
    t0 = time.perf_counter()
    eps = list(ex.map(play_episode, jobs, chunksize=1))
    dec = sum(len(e["samples"]) for e in eps)
    return {"episodes": n, "decisions": dec, "lookaheads": sum(e["lookaheads"] for e in eps),
            "clones_per_decision": sum(e["lookaheads"] for e in eps) / dec,
            "secs_per_episode": float(np.mean([e["secs"] for e in eps])),
            "wall_s": time.perf_counter() - t0}


def ppo_cost(tag: str) -> dict:
    d = json.loads((ROOT / "models" / f"{tag}_training.json").read_text())
    ups, vals = d["updates"], d["validations"]
    w = d["machine"]["workers"]
    val_eps = {g: 2 if g == "general" else 1 for g in VAL_GROUPS}
    per_val = d["args"]["val"] * sum(val_eps.values())
    gen_s, train_s, val_s = (sum(u["gen_s"] for u in ups), sum(u["train_s"] for u in ups),
                             sum(v["val_s"] for v in vals) + d["reference"]["wall_s"])
    return {"tag": tag, "workers": w, "wall_s": d["wall_s"], "gen_s": gen_s, "train_s": train_s,
            "val_s": val_s, "gen_cpu_s": sum(u["gen_cpu_s"] for u in ups),
            "val_cpu_s": sum(v["val_cpu_s"] for v in vals) + d["reference"]["cpu_s"],
            "episodes": sum(u["episodes"] for u in ups),
            "val_episodes": per_val * (len(vals) + 1),
            "decisions": sum(u["decisions"] for u in ups),
            "all_decisions": sum(u["all_decisions"] for u in ups),
            "best_update": d["best"]["update"], "best_score": d["best"]["score"]}


def chain_cost(tag: str, teacher: dict) -> dict:
    d = json.loads((ROOT / "models" / f"{tag}_training.json").read_text())
    its = d["iterations"]
    w = d["machine"][0]["workers"]
    val_n = d["args"]["val"]
    groups = sum(1 for k in its[0]["val"] if k.startswith("net/") and not k.endswith("score"))
    per_val = val_n * (groups + 1)                            # general counts double
    sf = "scenarios" in d["args"] and d["args"]["scenarios"] > 0
    clones = 0.0
    for it in its:
        if it["iteration"] == 0:
            clones += it["decisions"] * teacher["known"]["clones_per_decision"]
        else:
            clones += it["decisions"] * teacher["sf" if sf else "known"]["clones_per_decision"]
    # validation episodes: net + teacher + nearest_job per iteration, plus the reference
    # (nearest_head + teacher + nearest_job twice)
    val_eps = per_val * (3 * len(its) + 4)
    val_clones = sum(it["decisions"] for it in its) / sum(it["episodes"] for it in its) * \
        per_val * len(its) * teacher["sf" if sf else "known"]["clones_per_decision"]
    return {"tag": tag, "workers": w, "iterations": len(its), "teacher": "sampled-future" if sf else "known-parts",
            "gen_s": sum(it["gen_s"] for it in its), "train_s": sum(it["train_s"] for it in its),
            "val_s": sum(it["val_s"] for it in its),
            "episodes": sum(it["episodes"] for it in its),
            "decisions": sum(it["decisions"] for it in its),
            "lookahead_clones": clones, "val_episodes": val_eps, "val_lookahead_clones": val_clones,
            "best": d["best"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ppo", nargs="+", default=["rl2_ppo_s0", "rl2_ppo_s1", "rl2_ppo_s2"])
    ap.add_argument("--chain", nargs="+",
                    default=["rl2_gen_s_big", "rl2_gen_s_sf", "rl2_gen_s_d", "rl2_gen_s_b"])
    ap.add_argument("--sample", type=int, default=40, help="episodes to measure the teacher")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tag", default="rl2_ppo_cost")
    args = ap.parse_args(argv)
    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    net = str(ROOT / "models" / "rl2_gen_s_b.pt")
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        teacher = {"known": measure_teacher(ex, net, (0.0, 0), args.sample),
                   "sf": measure_teacher(ex, net, (float("inf"), 8, "time", 30.0), args.sample)}
    res = {"teacher_sample": teacher, "ppo": [ppo_cost(t) for t in args.ppo],
           "chain": [chain_cost(t, teacher) for t in args.chain]}
    (out / "cost.json").write_text(json.dumps(res, indent=2))

    lines = [f"# {args.tag}: training cost", "",
             f"Teacher sample ({args.sample} episodes, seeds {SAMPLE_SEED_BASE}+, base network "
             "`rl2_gen_s_b`): known-parts teacher "
             f"{teacher['known']['clones_per_decision']:.2f} clones per labelled decision, "
             f"{teacher['known']['secs_per_episode']:.2f} s per episode; sampled-future teacher "
             f"{teacher['sf']['clones_per_decision']:.2f} clones per labelled decision, "
             f"{teacher['sf']['secs_per_episode']:.1f} s per episode.", "",
             "## PPO runs", "",
             "| run | wall | generation | updates | validation | gen CPU-h | val CPU-h | "
             "wall × workers [CPU-h] | episodes (+ validation) | network decisions | best |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in res["ppo"]:
        lines.append(f"| {p['tag']} | {p['wall_s'] / 60:.1f} min | {p['gen_s'] / 60:.1f} min | "
                     f"{p['train_s'] / 60:.1f} min | {p['val_s'] / 60:.1f} min | "
                     f"{p['gen_cpu_s'] / 3600:.2f} | {p['val_cpu_s'] / 3600:.2f} | "
                     f"{p['wall_s'] * p['workers'] / 3600:.1f} | {p['episodes']:,} (+{p['val_episodes']:,}) | "
                     f"{p['decisions']:,} | update {p['best_update']} ({100 * p['best_score']:+.2f} %) |")
    lines += ["", "## Expert-iteration chain", "",
              "| run | teacher | iterations | wall (gen + train + val) | generation | updates | "
              "validation | wall × workers [CPU-h] | acting episodes | labelled decisions | "
              "look-ahead clones (est.) | validation episodes (+ clones, est.) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {k: 0.0 for k in ("gen_s", "train_s", "val_s", "episodes", "decisions",
                            "lookahead_clones", "val_episodes", "val_lookahead_clones", "cpu_h")}
    for c in res["chain"]:
        wall = c["gen_s"] + c["train_s"] + c["val_s"]
        cpu_h = wall * c["workers"] / 3600
        lines.append(f"| {c['tag']} | {c['teacher']} | {c['iterations']} | {wall / 3600:.2f} h | "
                     f"{c['gen_s'] / 3600:.2f} h | {c['train_s'] / 60:.1f} min | {c['val_s'] / 60:.1f} min | "
                     f"{cpu_h:.0f} | {c['episodes']:,} | {c['decisions']:,} | "
                     f"{c['lookahead_clones'] / 1e6:.2f} M | {c['val_episodes']:,} (+{c['val_lookahead_clones'] / 1e6:.2f} M) |")
        for k in tot:
            tot[k] += cpu_h if k == "cpu_h" else c[k]
    wall = tot["gen_s"] + tot["train_s"] + tot["val_s"]
    lines.append(f"| **chain** | | | **{wall / 3600:.2f} h** | {tot['gen_s'] / 3600:.2f} h | "
                 f"{tot['train_s'] / 60:.1f} min | {tot['val_s'] / 60:.1f} min | **{tot['cpu_h']:.0f}** | "
                 f"**{tot['episodes']:,.0f}** | {tot['decisions']:,.0f} | **{tot['lookahead_clones'] / 1e6:.2f} M** | "
                 f"{tot['val_episodes']:,.0f} (+{tot['val_lookahead_clones'] / 1e6:.2f} M) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
