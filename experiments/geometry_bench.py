"""Geometry bench (confirmatory): how much anticipation does the camera need when the
head can pre-position, and what predicts it: the transverse crossing time or the
moves the head actually makes?

Frozen protocol: ``docs/methods/geometry.md`` §5 (written before the run). 72 regimes
(belt spacing 6 / 12 / 18 cm × v_y 5 / 10 / 15 cm/s × v_x 5 / 10 cm/s × load nominal /
high by the arrival rates × Poisson / bursts), a common coverage grid in seconds of
anticipation, ``none`` against ``when_idle`` at every length, ``nearest_job`` (primary)
and the frozen network (secondary), 60 paired realizations per configuration; plus the
physical-length check (original cell at entry 10 cm against 35 cm). Development
smoke test and the confirmatory run:

    python experiments/geometry_bench.py --tag _geo_smoke --seed-base 3000 --seeds 2
    python experiments/geometry_bench.py --tag geometry_bench --confirmatory
    python experiments/geometry_bench.py --tag geometry_bench --analyse-only

The reserved range (``--confirmatory``: seeds 9000–9059) is used once: the script refuses
to overwrite an existing ``bench.json`` of a confirmatory run. Outputs:
``results/experiments/<tag>/{bench.json,tables.md,extract.csv.gz}`` and
``docs/methods/figures/<tag>_*.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from rl2.expert_iteration import BURST_VAL, _init_worker, val_config  # noqa: E402

# ------------------------------------------------------------------ frozen design
SPACINGS = (6.0, 12.0, 18.0)
VY = (5.0, 10.0, 15.0)
VX = (5.0, 10.0)
LOADS = {"nominal": (10.0, 12.0, 14.0), "high": (7.0, 8.5, 10.0)}   # mean inter-arrivals [s]
PROC = 2.5
ARRIVALS = ("poisson", "burst")
V_BELT = 3.0
ENTRY, ENTRY_ORIG = 35.0, 10.0
GRID_S = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 10 / 3, 5.0, 7.5, 10.0)     # anticipation [s]
CHECK_S = tuple(s for s in GRID_S if s * V_BELT <= ENTRY_ORIG + 1e-9)
MIN_USEFUL, MIN_USEFUL_LB, SHARE = 1.0, 0.5, 0.9                     # s/part; s/part; knee criterion
Z95 = 1.96
Z_BONF = 2.773            # two-sided 95 % simultaneous over the 9 non-zero grid lengths
COST_S = (1.5, 2.5, 10 / 3)   # anticipations whose direct cost against the largest is reported
RESERVED_BASE, RESERVED_N = 9000, 60
NET = "models/rl2_gen_s_b.pt"
KEEP = ("avg_flow_time", "blocking_events", "total_blocked_time", "total_simulation_time",
        "decisions", "idle_time", "wait_time", "prepos_moves", "revocations", "prepos_tx_mean",
        "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "prepos_wip_mean", "anticipation_eff")
EXTRACT = ("bench", "spacing", "vy", "vx", "tau_cross", "load", "arrivals", "entry", "s", "lv", "mode",
           "who", "seed", "cost", "avg_flow_time", "blocking_events", "total_blocked_time", "parts",
           "total_simulation_time", "idle_time", "wait_time", "prepos_moves", "revocations",
           "prepos_tx_mean", "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "prepos_wip_mean",
           "anticipation_eff")
FIG = ROOT / "docs/methods/figures"


def tau_cross(spacing, vy):
    return 2 * spacing / vy


def rho(load):
    return PROC * sum(1 / x for x in LOADS[load])


def lv_of(s):
    return round(s * V_BELT, 6)


REGIMES = [dict(spacing=sp, vy=vy, vx=vx, load=ld, arrivals=ar, tau_cross=tau_cross(sp, vy))
           for sp in SPACINGS for vy in VY for vx in VX for ld in LOADS for ar in ARRIVALS]


def regime_key(r):
    return (r["spacing"], r["vy"], r["vx"], r["load"], r["arrivals"])


def regime_name(r):
    return (f"spacing {r['spacing']:g} / v_y {r['vy']:g} / v_x {r['vx']:g} / {r['load']} / {r['arrivals']}"
            f" (τ_cross {r['tau_cross']:.1f} s)")


# ------------------------------------------------------------------ episodes
def extra(job):
    e = ["simulation.observation=belts", f"laser.prepositioning={job['mode']}",
         f"simulation.visible_length={job['lv']}", "conveyors.y_positions=null",
         f"conveyors.spacing={job['spacing']}", f"laser.speed_x={job['vx']}",
         f"laser.speed_y={job['vy']}", f"geometry.entry_length={job['entry']}",
         f"arrivals.stochastic.mean_interarrival=[{', '.join(str(x) for x in LOADS[job['load']])}]",
         f"processing.time={PROC}"]
    if job["arrivals"] == "burst":
        e += list(BURST_VAL)
    return e


_NET: dict = {}


def policy(job):
    from fastsim import POLICIES
    if job["who"] == "nearest_job":
        return POLICIES["nearest_job"]
    from rl2.model import NetPolicy, load
    p = job["net"]
    if p not in _NET:
        _NET[p] = NetPolicy(load(p))
    return _NET[p]


def run(job):
    from fastsim import FastSim
    from fastsim.core import MOVING
    cfg = val_config("real", job["seed"], "none", extra=tuple(extra(job)))
    sim = FastSim(cfg)
    pol = policy(job)
    bad = 0

    def guarded(s, cands):
        nonlocal bad
        for j in cands:
            if s.pst[j] == MOVING and s.x(j) < s.avis - 1e-6:
                bad += 1
        return pol(s, cands)

    t0 = time.perf_counter()
    s = sim.run(guarded)
    out = {k: v for k, v in job.items() if k != "net"}
    out |= {"cost": s["penalized_flow_time"], "parts": len(sim.arr), "mask_violations": bad,
            "secs": time.perf_counter() - t0}
    out |= {k: s[k] for k in KEEP}
    return out


def jobs(args):
    seeds = [args.seed_base + i for i in range(args.seeds)]
    net = str(ROOT / args.net)
    J = []
    for r in REGIMES:
        for s in GRID_S:
            for mode in ("none", "when_idle"):
                for who in args.who:
                    for seed in seeds:
                        J.append(r | dict(bench="grid", entry=ENTRY, s=s, lv=lv_of(s), mode=mode,
                                          who=who, seed=seed, net=net))
    for r in REGIMES:
        if not (r["spacing"] == 12.0 and r["vy"] == 10.0 and r["vx"] == 10.0):
            continue
        for s in CHECK_S:
            for mode in ("none", "when_idle"):
                for who in args.who:
                    for seed in seeds:
                        J.append(r | dict(bench="entry", entry=ENTRY_ORIG, s=s, lv=lv_of(s), mode=mode,
                                          who=who, seed=seed, net=net))
    return J


# ------------------------------------------------------------------ statistics
def ci(d, z=Z95):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else math.nan), math.nan
    return float(d.mean()), float(z * d.std(ddof=1) / math.sqrt(len(d)))


def cell(mu, h, n=None):
    s = f"{mu:+.2f} ±{h:.2f}"
    return s if n is None else f"{s} ({n})"


class Bench:
    def __init__(self, rows):
        self.rows = rows
        self.by = {}
        for r in rows:
            self.by.setdefault((r["bench"], regime_key(r), r["entry"], round(r["s"], 6), r["mode"], r["who"]), {})[r["seed"]] = r
        self.seeds = sorted({r["seed"] for r in rows})
        self.whos = [w for w in ("nearest_job", "net") if any(r["who"] == w for r in rows)]
        self.regimes = [r for r in REGIMES if any(regime_key(r) == regime_key(x) for x in rows[:20000])] or REGIMES

    def get(self, reg, s, mode, who, bench="grid", entry=ENTRY):
        return self.by.get((bench, regime_key(reg), entry, round(s, 6), mode, who), {})

    def paired(self, x, y, field="cost", z=Z95):
        seeds = sorted(set(x) & set(y))
        d = [x[k][field] - y[k][field] for k in seeds]
        mu, h = ci(d, z)
        return mu, h, sum(v > 1e-12 for v in d), len(seeds)

    def gain(self, reg, s, who, z=Z95, bench="grid", entry=ENTRY):
        """Seconds per part saved by pre-positioning at anticipation s (none − when_idle)."""
        return self.paired(self.get(reg, s, "none", who, bench, entry), self.get(reg, s, "when_idle", who, bench, entry), z=z)

    def level(self, reg, s, mode, who, field="cost"):
        rows = self.get(reg, s, mode, who)
        return float(np.mean([r[field] for r in rows.values()])) if rows else math.nan

    def useful(self, reg, who):
        mu, h, _, _ = self.gain(reg, GRID_S[-1], who)
        return mu >= MIN_USEFUL and mu - h >= MIN_USEFUL_LB

    def knee(self, reg, who, conservative=False):
        """First grid anticipation whose paired contrast G(s) − SHARE·G(s_max) is ≥ 0 (point)
        or has a lower simultaneous-95 % bound ≥ 0 (conservative, Bonferroni over the grid)."""
        rn, rw = self.get(reg, GRID_S[-1], "none", who), self.get(reg, GRID_S[-1], "when_idle", who)
        for s in GRID_S[1:]:
            a, b = self.get(reg, s, "none", who), self.get(reg, s, "when_idle", who)
            seeds = sorted(set(a) & set(b) & set(rn) & set(rw))
            if not seeds:
                return math.nan
            d = [(a[k]["cost"] - b[k]["cost"]) - SHARE * (rn[k]["cost"] - rw[k]["cost"]) for k in seeds]
            mu, h = ci(d, Z_BONF if conservative else Z95)
            if (mu - h >= 0) if conservative else (mu >= 0):
                return s
        return math.nan

    def trace(self, reg, who, s=GRID_S[-1], mode="when_idle"):
        rows = self.get(reg, s, mode, who)
        if not rows:
            return {}

        def m(k):
            xs = [r[k] for r in rows.values() if r.get(k) is not None]
            return float(np.mean(xs)) if xs else math.nan
        T = m("total_simulation_time")
        return {"moves": m("prepos_moves"), "revoked": m("revocations"), "tx": m("prepos_tx_mean"),
                "ty": m("prepos_ty_mean"), "t": m("prepos_t_mean"), "ydom": m("prepos_y_dominated"),
                "wip": m("prepos_wip_mean"), "idle": m("idle_time") / T if T else math.nan,
                "wait": m("wait_time"), "ant": m("anticipation_eff"), "parts": m("parts"),
                "blocks": m("blocking_events"), "blocked": m("total_blocked_time")}


def loo_mae(x, y, model):
    """Leave-one-out mean absolute error of knee ≈ model(x; fit on the others)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        errs.append(abs(y[i] - model(x[m], y[m], x[i])))
    return float(np.mean(errs)) if errs else math.nan


MODELS = {
    "knee = τ_cross": lambda xs, ys, x: x,
    "knee = a·τ_cross": lambda xs, ys, x: (np.sum(xs * ys) / np.sum(xs * xs)) * x,
    "knee = b·τ_pp": lambda xs, ys, x: (np.sum(xs * ys) / np.sum(xs * xs)) * x,
    "knee = constant": lambda xs, ys, x: float(np.mean(ys)),
}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return math.nan
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ tables
def tables(b: Bench, args) -> str:
    L = [f"# Geometry bench `{args.tag}`", "",
         f"Seeds {b.seeds[0]}–{b.seeds[-1]} ({len(b.seeds)} paired realizations per configuration). "
         f"Belts {V_BELT:g} cm/s, entry zone {ENTRY:g} cm, processing {PROC:g} s; loads by the arrival rates: "
         + ", ".join(f"{k} = {'/'.join(f'{x:g}' for x in v)} s (ρ_p {rho(k):.2f})" for k, v in LOADS.items())
         + f"; bursts `{' '.join(BURST_VAL)}`. G = seconds per part saved by pre-positioning (`none` − "
         "`when_idle` at the same anticipation, paired, 95 % CI; (n) = realizations with a positive gain). "
         f"Useful benefit: G(10 s) ≥ {MIN_USEFUL} s/part with lower bound ≥ {MIN_USEFUL_LB}. Knee: first grid "
         f"anticipation retaining {100 * SHARE:.0f} % of G(10 s), point estimate / lower simultaneous-95 % bound "
         f"(z = {Z_BONF}) ≥ 0. Protocol: `docs/methods/geometry.md` §5.", ""]
    for who in b.whos:
        L += [f"## `{who}`: summary across regimes", "",
              "| spacing | v_y | v_x | τ_cross | load | arrivals | `none`(0) [s/part] | G(10 s) [s/part] | useful | "
              "G at 1.5 / 2.5 / 3.33 s [s/part] | cost of 1.5 / 2.5 / 3.33 s vs 10 s [s/part] | knee [s] point / conservative | "
              "moves | idle (`none`) | t_x / t_y / max [s] | y-dom | WIP at move |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for reg in b.regimes:
            g10 = b.gain(reg, GRID_S[-1], who)
            if math.isnan(g10[0]):
                continue
            gs = [b.gain(reg, s, who) for s in COST_S]
            costs = [b.paired(b.get(reg, s, "when_idle", who), b.get(reg, GRID_S[-1], "when_idle", who)) for s in COST_S]
            tr = b.trace(reg, who)
            idle0 = b.trace(reg, who, 0.0, "none").get("idle", math.nan)
            k1, k2 = b.knee(reg, who), b.knee(reg, who, conservative=True)
            u = b.useful(reg, who)
            L.append(f"| {reg['spacing']:g} | {reg['vy']:g} | {reg['vx']:g} | {reg['tau_cross']:.1f} | {reg['load']} | "
                     f"{reg['arrivals']} | {b.level(reg, 0.0, 'none', who):.1f} | {cell(g10[0], g10[1])} | "
                     f"{'yes' if u else 'no'} | " + " / ".join(f"{g[0]:+.2f}" for g in gs) + " | "
                     + " / ".join(f"{c[0]:+.2f}±{c[1]:.2f}" for c in costs) + " | "
                     f"{k1:.2f} / {k2:.2f}" + ("" if u else " (undetermined)") + " | "
                     f"{tr.get('moves', math.nan):.1f} | {100 * idle0:.0f} % | {tr.get('tx', math.nan):.2f} / "
                     f"{tr.get('ty', math.nan):.2f} / {tr.get('t', math.nan):.2f} | {100 * tr.get('ydom', math.nan):.0f} % | "
                     f"{tr.get('wip', math.nan):.1f} |")
        L.append("")
        # curves
        L += [f"## `{who}`: coverage curves (G [s/part] at each anticipation; share of G(10 s) in parentheses)", "",
              "| regime | " + " | ".join(f"{s:.2f} s" for s in GRID_S) + " |", "|---|" + "---|" * len(GRID_S)]
        for reg in b.regimes:
            g10 = b.gain(reg, GRID_S[-1], who)[0]
            if math.isnan(g10):
                continue
            cells = []
            for s in GRID_S:
                mu, h, nb, n = b.gain(reg, s, who)
                cells.append(f"{mu:+.2f}±{h:.2f} ({100 * mu / g10:.0f} %)" if g10 else f"{mu:+.2f}±{h:.2f}")
            L.append(f"| {regime_name(reg)} | " + " | ".join(cells) + " |")
        L.append("")
        if who == "net":
            L += ["## `net`: value of widening the view with the mode fixed (when_idle(s) − when_idle(0), "
                  "s/part, positive = better) and value of seeing with the same actions (none(0) − none(s))", "",
                  "| regime | " + " | ".join(f"{s:.2f} s" for s in GRID_S[1:]) + " |", "|---|" + "---|" * (len(GRID_S) - 1)]
            for reg in b.regimes:
                w0 = b.get(reg, 0.0, "when_idle", who)
                n0 = b.get(reg, 0.0, "none", who)
                if not w0:
                    continue
                cells = []
                for s in GRID_S[1:]:
                    a = b.paired(w0, b.get(reg, s, "when_idle", who))
                    c = b.paired(n0, b.get(reg, s, "none", who))
                    cells.append(f"{a[0]:+.2f}±{a[1]:.2f} / see {c[0]:+.2f}±{c[1]:.2f}")
                L.append(f"| {regime_name(reg)} | " + " | ".join(cells) + " |")
            L.append("")
    # ---- predictors of the knee (rule, useful regimes)
    for who in b.whos:
        pts = [(reg, b.knee(reg, who), b.trace(reg, who)) for reg in b.regimes if b.useful(reg, who)]
        pts = [(reg, k, tr) for reg, k, tr in pts if not math.isnan(k) and tr]
        L += [f"## Predictors of the knee (`{who}`, {len(pts)} regimes with a useful benefit)", ""]
        if len(pts) >= 4:
            knee = [k for _, k, _ in pts]
            tc = [reg["tau_cross"] for reg, _, _ in pts]
            tpp = [tr["t"] for _, _, tr in pts]
            txy = [tr["tx"] + tr["ty"] for _, _, tr in pts]
            L += [f"* knee range {min(knee):.2f}–{max(knee):.2f} s; τ_cross range {min(tc):.1f}–{max(tc):.1f} s; "
                  f"mean move duration τ_pp range {min(tpp):.2f}–{max(tpp):.2f} s.",
                  f"* Spearman rank correlation of the knee with τ_cross: {spearman(tc, knee):+.2f}; with τ_pp: "
                  f"{spearman(tpp, knee):+.2f}; with t_x + t_y: {spearman(txy, knee):+.2f}.",
                  "* Leave-one-out mean absolute error [s]: "
                  f"knee = τ_cross {loo_mae(tc, knee, MODELS['knee = τ_cross']):.2f}; "
                  f"knee = a·τ_cross {loo_mae(tc, knee, MODELS['knee = a·τ_cross']):.2f}; "
                  f"knee = b·τ_pp {loo_mae(tpp, knee, MODELS['knee = b·τ_pp']):.2f}; "
                  f"knee = b·(t_x + t_y) {loo_mae(txy, knee, MODELS['knee = b·τ_pp']):.2f}; "
                  f"knee = constant {loo_mae(tc, knee, MODELS['knee = constant']):.2f}.", ""]
        else:
            L += ["* fewer than four regimes with a useful benefit: no predictor comparison.", ""]
    # ---- H0 triplet
    L += ["## H0: cells with the same τ_cross (2.4 s: 6 / 5, 12 / 10, 18 / 15) per v_x, load and arrivals", ""]
    for who in b.whos:
        for vx in VX:
            for ld in LOADS:
                for ar in ARRIVALS:
                    ref = dict(spacing=12.0, vy=10.0, vx=vx, load=ld, arrivals=ar)
                    dmax, n = 0.0, 0
                    for sp, vy in ((6.0, 5.0), (18.0, 15.0)):
                        other = dict(spacing=sp, vy=vy, vx=vx, load=ld, arrivals=ar)
                        for s in GRID_S:
                            for mode in ("none", "when_idle"):
                                x, y = b.get(other, s, mode, who), b.get(ref, s, mode, who)
                                for k in set(x) & set(y):
                                    dmax = max(dmax, abs(x[k]["cost"] - y[k]["cost"])); n += 1
                    if n:
                        L.append(f"* `{who}`, v_x {vx:g}, {ld}, {ar}: max |ΔJ| = {dmax:.2e} s over {n} paired episodes.")
    L.append("")
    # ---- entry length
    L += ["## H3: physical entry length (spacing 12 / v_y 10 / v_x 10; entry 10 cm against 35 cm)", "",
          f"Free transit added by the longer entry: {(ENTRY - ENTRY_ORIG) / V_BELT:.2f} s per part.", ""]
    for who in b.whos:
        for ld in LOADS:
            for ar in ARRIVALS:
                reg = dict(spacing=12.0, vy=10.0, vx=10.0, load=ld, arrivals=ar, tau_cross=2.4)
                if not b.get(reg, 0.0, "none", who, "entry", ENTRY_ORIG):
                    continue
                L += [f"| `{who}` {ld} / {ar}: s | Δ none (35 − 10) [s] | G entry 10 | G entry 35 | ΔG (35 − 10) |", "|---|---|---|---|---|"]
                for s in CHECK_S:
                    n10, n35 = b.get(reg, s, "none", who, "entry", ENTRY_ORIG), b.get(reg, s, "none", who)
                    w10, w35 = b.get(reg, s, "when_idle", who, "entry", ENTRY_ORIG), b.get(reg, s, "when_idle", who)
                    dn = b.paired(n35, n10)
                    g10, g35 = b.gain(reg, s, who, bench="entry", entry=ENTRY_ORIG), b.gain(reg, s, who)
                    seeds = sorted(set(n10) & set(n35) & set(w10) & set(w35))
                    dg = ci([(n35[k]["cost"] - w35[k]["cost"]) - (n10[k]["cost"] - w10[k]["cost"]) for k in seeds])
                    L.append(f"| {s:.2f} | {cell(dn[0], dn[1])} | {cell(g10[0], g10[1], g10[2])} | "
                             f"{cell(g35[0], g35[1], g35[2])} | {cell(*dg)} |")
                L.append("")
    # ---- checks
    viol = sum(r["mask_violations"] for r in b.rows)
    L += ["## Checks", "", f"* visible stretch respected: {viol} candidate violations over {len(b.rows)} episodes."]
    for who in b.whos:
        dmax = 0.0
        for r in b.rows:
            if r["who"] == who and r["s"] == 0.0 and r["mode"] == "when_idle":
                o = b.by.get((r["bench"], regime_key(r), r["entry"], 0.0, "none", who), {}).get(r["seed"])
                if o:
                    dmax = max(dmax, abs(r["cost"] - o["cost"]))
        L.append(f"* `{who}`: at s = 0, max |J(when_idle) − J(none)| = {dmax:.2e} s"
                 + (" (the mode is a context token of the network)." if who == "net" else "."))
    secs = sum(r["secs"] for r in b.rows)
    L += [f"* {len(b.rows)} episodes, {secs / 3600:.2f} CPU-hours.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def figures(b: Bench, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    tcs = sorted({tau_cross(sp, vy) for sp in SPACINGS for vy in VY})
    for who in b.whos:
        for ar in ARRIVALS:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
            for i, vx in enumerate(VX):
                for j, ld in enumerate(LOADS):
                    ax = axes[i][j]
                    for sp in SPACINGS:
                        for vy in VY:
                            reg = dict(spacing=sp, vy=vy, vx=vx, load=ld, arrivals=ar, tau_cross=tau_cross(sp, vy))
                            g = np.array([b.gain(reg, s, who)[0] for s in GRID_S])
                            h = np.array([b.gain(reg, s, who)[1] for s in GRID_S])
                            if np.isnan(g).all():
                                continue
                            tc = tau_cross(sp, vy)
                            col = cmap(tcs.index(tc) / max(1, len(tcs) - 1))
                            ls = {5.0: "-", 10.0: "--", 15.0: ":"}[vy]
                            ax.plot(GRID_S, g, ls, color=col, marker="o", ms=2.5, lw=1.1,
                                    label=f"{sp:g} cm / {vy:g} cm/s, τ_cross {tc:.1f} s")
                            ax.fill_between(GRID_S, g - h, g + h, color=col, alpha=0.06)
                    ax.axhline(0, color="k", lw=0.5)
                    ax.set_title(f"v_x {vx:g} cm/s, {ld} load", fontsize=9)
                    if i == 1:
                        ax.set_xlabel("anticipation τ_vis [s]")
                    if j == 0:
                        ax.set_ylabel("G [s per part]")
            axes[0][0].legend(fontsize=6, frameon=False, ncol=1)
            fig.suptitle(f"{who}, {ar} arrivals: pre-positioning gain against the camera's anticipation", fontsize=10)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(FIG / f"{tag}_{who}_{ar}.{ext}", dpi=140)
            plt.close(fig)
        # knee against the two candidate predictors
        pts = [(reg, b.knee(reg, who), b.trace(reg, who)) for reg in b.regimes if b.useful(reg, who)]
        pts = [(reg, k, tr) for reg, k, tr in pts if not math.isnan(k) and tr]
        if len(pts) >= 4:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            for ax, key, lab in ((axes[0], "tau_cross", "τ_cross [s]"), (axes[1], "t", "mean move duration τ_pp [s]")):
                for reg, k, tr in pts:
                    x = reg["tau_cross"] if key == "tau_cross" else tr["t"]
                    ax.plot(x, k, marker={5.0: "s", 10.0: "o"}[reg["vx"]],
                            color={"nominal": "#1f77b4", "high": "#d62728"}[reg["load"]],
                            mfc="none" if reg["arrivals"] == "burst" else None, ms=5, ls="none")
                lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([0, lim], [0, lim], color="grey", lw=0.6, ls="--")
                ax.set_xlabel(lab); ax.set_ylabel("knee [s] (point estimate)")
            axes[0].set_title("square v_x 5, circle v_x 10; blue nominal, red high; hollow = bursts", fontsize=8)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(FIG / f"{tag}_{who}_knee.{ext}", dpi=140)
            plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="geometry_bench")
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--confirmatory", action="store_true",
                    help=f"the reserved seeds {RESERVED_BASE}–{RESERVED_BASE + RESERVED_N - 1}, used once")
    ap.add_argument("--who", nargs="+", default=["nearest_job", "net"], choices=["nearest_job", "net"])
    ap.add_argument("--net", default=NET)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "results/experiments" / args.tag
    if args.confirmatory:
        args.seed_base, args.seeds = RESERVED_BASE, RESERVED_N
        if (out_dir / "bench.json").exists() and not args.analyse_only:
            sys.exit(f"{out_dir / 'bench.json'} exists: the reserved seeds are used once")
    elif not args.analyse_only:
        lo, hi = args.seed_base, args.seed_base + args.seeds - 1
        for b0, n in ((7000, 30), (RESERVED_BASE, RESERVED_N)):
            if lo <= b0 + n - 1 and hi >= b0:
                sys.exit(f"seeds {b0}–{b0 + n - 1} are reserved (use --confirmatory for {RESERVED_BASE})")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        J = jobs(args)
        print(f"{len(J)} episodes", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
            J.sort(key=lambda j: (j["who"] != "net", j["load"] != "high"))
            rows = list(ex.map(run, J, chunksize=8))
        res = {"args": vars(args), "design": {"grid_s": GRID_S, "loads": LOADS, "proc": PROC, "entry": ENTRY,
                                              "min_useful": (MIN_USEFUL, MIN_USEFUL_LB), "share": SHARE,
                                              "z_bonf": Z_BONF, "net": args.net},
               "elapsed_s": time.perf_counter() - t0, "rows": rows}
        json.dump(res, open(out_dir / "bench.json", "w"))
        with gzip.open(out_dir / "extract.csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, EXTRACT)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in EXTRACT})
        print(f"elapsed {res['elapsed_s']:.0f} s", flush=True)
    res = json.load(open(out_dir / "bench.json"))
    b = Bench(res["rows"])
    md = tables(b, args)
    (out_dir / "tables.md").write_text(md)
    figures(b, args.tag)
    print(md)
    print(f"written {out_dir} and {FIG}/{args.tag}_*.pdf")


if __name__ == "__main__":
    main()
