"""Paired evaluation of PPO checkpoints against ``nearest_job`` and a reference network
on the validation realizations (docs/methods/rl2_ppo.md §1.5).

Every policy runs the same realizations (``val_config``: real, heavy, general,
real_burst; design mode; ``--val`` seeds per group, 2x for general), greedy, on the
exact simulator. Per group: mean paired ratio - 1 with a 95 % interval
(1.96 sd / sqrt n), wins; score = mean of the groups. The mean over the PPO seeds
is the per-realization mean of their ratios, then mean and interval over the
realizations.

    python experiments/rl2_ppo_eval.py --models models/rl2_ppo_s0.pt models/rl2_ppo_s1.pt \
        models/rl2_ppo_s2.pt --ref models/rl2_gen_s_b.pt --workers 40 --tag rl2_ppo_eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl2.expert_iteration import VAL_GROUPS, VAL_MODES, _init_worker  # noqa: E402
from rl2.ppo import run_val  # noqa: E402


def ratios(costs: dict, ref: dict) -> dict:
    """{group: [ratio - 1 per realization, sorted by seed]}"""
    return {g: [costs[k] / ref[k] - 1 for k in sorted(costs) if k[0] == g] for g in VAL_GROUPS}


def summarize(rs: dict) -> dict:
    out = {}
    for g in VAL_GROUPS:
        x = np.asarray(rs[g])
        out[g] = {"delta": float(x.mean()), "ci": 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x)),
                  "n": len(x), "wins": int((x < 0).sum())}
    out["score"] = float(np.mean([out[g]["delta"] for g in VAL_GROUPS]))
    return out


def row(name: str, s: dict) -> str:
    cells = [f"{100 * s[g]['delta']:+.2f} ±{100 * s[g]['ci']:.2f} ({s[g]['wins']}/{s[g]['n']})"
             for g in VAL_GROUPS]
    return f"| {name} | " + " | ".join(cells) + f" | **{100 * s['score']:+.2f}** |"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="PPO checkpoints")
    ap.add_argument("--ref", default="models/rl2_gen_s_b.pt", help="reference network")
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--val-mode", default="design", choices=list(VAL_MODES))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tag", default="rl2_ppo_eval")
    ap.add_argument("--horizon", type=float, default=None)
    args = ap.parse_args(argv)
    out = ROOT / "results" / "experiments" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    names = {str(ROOT / m): Path(m).stem for m in args.models}
    ref_path = str(ROOT / args.ref)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
        nj = run_val(ex, args.val, "nearest_job", None, args.val_mode, args.horizon)
        refn = run_val(ex, args.val, "net", ref_path, args.val_mode, args.horizon)
        ppo = {p: run_val(ex, args.val, "net", p, args.val_mode, args.horizon) for p in names}
    wall = time.perf_counter() - t0

    res = {"args": vars(args), "wall_s": wall, "ref": Path(args.ref).stem,
           "costs": {"nearest_job": [[g, s, c] for (g, s), c in sorted(nj["costs"].items())],
                     Path(args.ref).stem: [[g, s, c] for (g, s), c in sorted(refn["costs"].items())],
                     **{n: [[g, s, c] for (g, s), c in sorted(ppo[p]["costs"].items())]
                        for p, n in names.items()}},
           "vs_nearest_job": {}, "vs_ref": {}}
    lines = [f"# {args.tag}: paired evaluation on the validation realizations",
             "", f"Design mode, {args.val} seeds per group (2x general), greedy policies, "
             f"exact simulator; reference network `{args.ref}`; {wall:.0f} s wall on "
             f"{args.workers} workers.", "",
             "## vs `nearest_job` (paired ratio − 1, %, ±95 %, wins)", "",
             "| policy | " + " | ".join(VAL_GROUPS) + " | score |",
             "|---|" + "---|" * (len(VAL_GROUPS) + 1)]
    s = summarize(ratios(refn["costs"], nj["costs"]))
    res["vs_nearest_job"][res["ref"]] = s
    lines.append(row(res["ref"], s))
    per_seed_nj, per_seed_ref = [], []
    for p, n in names.items():
        r = ratios(ppo[p]["costs"], nj["costs"])
        per_seed_nj.append(r)
        s = summarize(r)
        res["vs_nearest_job"][n] = s
        lines.append(row(n, s))
    if len(names) > 1:
        mean_r = {g: list(np.mean([r[g] for r in per_seed_nj], axis=0)) for g in VAL_GROUPS}
        s = summarize(mean_r)
        res["vs_nearest_job"]["mean_over_seeds"] = s
        lines.append(row("mean over seeds", s))
    lines += ["", f"## vs `{res['ref']}` (paired ratio − 1, %, ±95 %, wins; negative = PPO better)",
              "", "| policy | " + " | ".join(VAL_GROUPS) + " | score |",
              "|---|" + "---|" * (len(VAL_GROUPS) + 1)]
    for p, n in names.items():
        r = ratios(ppo[p]["costs"], refn["costs"])
        per_seed_ref.append(r)
        s = summarize(r)
        res["vs_ref"][n] = s
        lines.append(row(n, s))
    if len(names) > 1:
        mean_r = {g: list(np.mean([r[g] for r in per_seed_ref], axis=0)) for g in VAL_GROUPS}
        s = summarize(mean_r)
        res["vs_ref"]["mean_over_seeds"] = s
        lines.append(row("mean over seeds", s))
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
