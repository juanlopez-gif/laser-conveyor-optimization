"""Geometry pilot (0.7.18): does the head's crossing time predict the camera coverage?

Design, hypotheses and checks: ``docs/methods/geometry.md`` §2–§3 (written before the
run). Nine cells (belt spacing 6 / 12 / 18 cm × transverse head speed 5 / 10 / 15 cm/s,
v_x fixed at 10 cm/s, entry zone 35 cm), Poisson arrivals at the nominal rates, the
coverage grid at fractions of τ_cross = (y_max − y_min) / v_y plus common absolute
lengths, ``none`` against ``when_idle`` at every length, ``nearest_job`` (primary) and
the frozen network (secondary), development seeds. Plus the physical-length check: the
original cell (entry 10 cm) at the same visible lengths.

    python experiments/geometry_pilot.py --tag geometry_pilot --seed-base 3000 --seeds 30
    python experiments/geometry_pilot.py --tag geometry_pilot --analyse-only
    python experiments/geometry_pilot.py --check-original          # check (i) of §3

Outputs ``results/experiments/<tag>/bench.json`` and ``tables.md`` (not versioned) and
the figures ``docs/methods/figures/geometry_pilot_*.pdf``.
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
sys.path.insert(0, str(ROOT / "experiments"))

from rl2.expert_iteration import _init_worker, val_config  # noqa: E402

SPACINGS = (6.0, 12.0, 18.0)
VY = (5.0, 10.0, 15.0)
VX = 10.0
V_BELT = 3.0
ENTRY = 35.0
ENTRY_ORIG = 10.0
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
ABS_LENGTHS = (1.0, 2.5, 5.0, 7.5, 10.0)
CHECK_LENGTHS = (0.0, 1.0, 2.5, 5.0, 7.5, 10.0)
NET = "models/rl2_gen_s_b.pt"
KEEP = ("avg_flow_time", "blocking_events", "total_blocked_time", "total_simulation_time",
        "decisions", "idle_time", "wait_time", "prepos_moves", "revocations", "prepos_tx_mean",
        "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "anticipation_eff")
FIG = ROOT / "docs/methods/figures"


def tau_cross(spacing: float, vy: float) -> float:
    return 2 * spacing / vy            # three belts at 0, spacing, 2 spacing


def geom_name(spacing, vy):
    return f"spacing {spacing:g} cm, v_y {vy:g} cm/s (τ_cross {tau_cross(spacing, vy):.1f} s)"


# ------------------------------------------------------------------ episodes
def extra(job: dict) -> list[str]:
    return ["simulation.observation=belts", f"laser.prepositioning={job['mode']}",
            f"simulation.visible_length={job['lv']}", "conveyors.y_positions=null",
            f"conveyors.spacing={job['spacing']}", f"laser.speed_x={VX}",
            f"laser.speed_y={job['vy']}", f"geometry.entry_length={job['entry']}"]


_NET: dict = {}


def policy(job: dict):
    from fastsim import POLICIES
    if job["who"] == "nearest_job":
        return POLICIES["nearest_job"]
    from rl2.model import NetPolicy, load
    p = job["net"]
    if p not in _NET:
        _NET[p] = NetPolicy(load(p))
    return _NET[p]


def run(job: dict) -> dict:
    """One episode; check (iii) of §3 (the visible stretch is respected) on every decision."""
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
            "a": sim.a, "secs": time.perf_counter() - t0}
    out |= {k: s[k] for k in KEEP}
    return out


def jobs(args) -> list[dict]:
    seeds = [args.seed_base + i for i in range(args.seeds)]
    whos = list(args.who)
    net = str(ROOT / args.net)
    J = []
    for sp in SPACINGS:
        for vy in VY:
            tc = tau_cross(sp, vy)
            specs = [("frac", f, f * tc * V_BELT) for f in FRACS] + [("abs", None, l) for l in ABS_LENGTHS]
            for kind, f, lv in specs:
                for mode in ("none", "when_idle"):
                    for who in whos:
                        for s in seeds:
                            J.append(dict(bench="grid", spacing=sp, vy=vy, tau_cross=tc, entry=ENTRY,
                                          kind=kind, frac=f, lv=round(lv, 6), tau_vis=lv / V_BELT,
                                          mode=mode, who=who, seed=s, net=net))
    for entry in (ENTRY_ORIG, ENTRY):
        for lv in CHECK_LENGTHS:
            for mode in ("none", "when_idle"):
                for who in whos:
                    for s in seeds:
                        J.append(dict(bench="entry", spacing=12.0, vy=10.0, tau_cross=2.4, entry=entry,
                                      kind="abs", frac=None, lv=lv, tau_vis=lv / V_BELT, mode=mode,
                                      who=who, seed=s, net=net))
    return J


# ------------------------------------------------------------------ statistics
def ci(d):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else math.nan, math.nan
    return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))


def cell(mu, h, n=None):
    s = f"{mu:+.2f} ±{h:.2f}"
    return s if n is None else f"{s} ({n})"


class Bench:
    def __init__(self, rows):
        self.rows = rows
        self.by = {}
        for r in rows:
            key = (r["bench"], r["spacing"], r["vy"], r["entry"], round(r["lv"], 6), r["mode"], r["who"])
            self.by.setdefault(key, {})[r["seed"]] = r
        self.seeds = sorted({r["seed"] for r in rows})
        self.whos = [w for w in ("nearest_job", "net") if any(r["who"] == w for r in rows)]

    def get(self, bench, sp, vy, entry, lv, mode, who):
        return self.by.get((bench, sp, vy, entry, round(lv, 6), mode, who), {})

    def lengths(self, sp, vy, who):
        """Visible lengths of the grid for a geometry, sorted, with their fraction (or None)."""
        out = {}
        for r in self.rows:
            if r["bench"] == "grid" and r["spacing"] == sp and r["vy"] == vy and r["who"] == who:
                out.setdefault(round(r["lv"], 6), r["frac"])
        return sorted(out.items())

    def gain_s(self, bench, sp, vy, entry, lv, who):
        """Seconds per part saved by pre-positioning at lv: J_none − J_when_idle, paired."""
        a, b = self.get(bench, sp, vy, entry, lv, "none", who), self.get(bench, sp, vy, entry, lv, "when_idle", who)
        seeds = sorted(set(a) & set(b))
        d = [a[s]["cost"] - b[s]["cost"] for s in seeds]
        return (*ci(d), sum(x > 1e-12 for x in d), len(seeds))

    def diff_s(self, x: dict, y: dict, field="cost"):
        """Paired x − y in seconds over the common seeds."""
        seeds = sorted(set(x) & set(y))
        d = [x[s][field] - y[s][field] for s in seeds]
        return (*ci(d), len(seeds))

    def level(self, bench, sp, vy, entry, lv, mode, who, field="cost"):
        rows = self.get(bench, sp, vy, entry, lv, mode, who)
        return float(np.mean([r[field] for r in rows.values()])) if rows else math.nan

    def curve(self, sp, vy, who):
        """[(lv, frac, tau_vis, G mean, G half, n better)] over the merged grid."""
        out = []
        for lv, f in self.lengths(sp, vy, who):
            mu, h, nb, n = self.gain_s("grid", sp, vy, ENTRY, lv, who)
            out.append((lv, f, lv / V_BELT, mu, h, nb))
        return out

    def knee(self, sp, vy, who, conservative=False):
        """First point of the merged grid (by τ_vis) retaining 90 % of the gain at the
        largest τ_vis; ``conservative``: the paired contrast G(ℓ) − 0.9 G(ℓ_max) must
        have its lower 95 % bound ≥ 0. Returns (tau_vis, lv) or (nan, nan)."""
        pts = self.lengths(sp, vy, who)
        if not pts:
            return math.nan, math.nan
        lmax = pts[-1][0]
        ref_n, ref_w = self.get("grid", sp, vy, ENTRY, lmax, "none", who), self.get("grid", sp, vy, ENTRY, lmax, "when_idle", who)
        for lv, _ in pts:
            a, b = self.get("grid", sp, vy, ENTRY, lv, "none", who), self.get("grid", sp, vy, ENTRY, lv, "when_idle", who)
            seeds = sorted(set(a) & set(b) & set(ref_n) & set(ref_w))
            d = [(a[s]["cost"] - b[s]["cost"]) - 0.9 * (ref_n[s]["cost"] - ref_w[s]["cost"]) for s in seeds]
            mu, h = ci(d)
            if (mu - h >= 0) if conservative else (mu >= 0):
                return lv / V_BELT, lv
        return math.nan, math.nan

    def trace(self, sp, vy, who, lv):
        rows = self.get("grid", sp, vy, ENTRY, lv, "when_idle", who)
        if not rows:
            return {}
        def m(k):
            xs = [r[k] for r in rows.values() if r.get(k) is not None]
            return float(np.mean(xs)) if xs else math.nan
        T = m("total_simulation_time")
        return {"moves": m("prepos_moves"), "revoked": m("revocations"), "tx": m("prepos_tx_mean"),
                "ty": m("prepos_ty_mean"), "t": m("prepos_t_mean"), "ydom": m("prepos_y_dominated"),
                "idle": m("idle_time") / T if T else math.nan, "wait": m("wait_time"),
                "ant": m("anticipation_eff"), "parts": m("parts")}


# ------------------------------------------------------------------ tables
def tables(b: Bench, args) -> str:
    L = [f"# Geometry pilot `{args.tag}`", "",
         f"Seeds {b.seeds[0]}–{b.seeds[-1]} ({len(b.seeds)} paired realizations per configuration), "
         f"v_x {VX:g} cm/s, belts {V_BELT:g} cm/s, entry zone {ENTRY:g} cm, Poisson arrivals at the nominal "
         f"rates. G = seconds per part saved by pre-positioning (`none` − `when_idle` at the same visible "
         "length, paired; 95 % CI; (n) = realizations with a positive gain). Design and hypotheses: "
         "`docs/methods/geometry.md`.", ""]
    for who in b.whos:
        L += [f"## `{who}`: coverage curves", ""]
        for sp in SPACINGS:
            for vy in VY:
                tc = tau_cross(sp, vy)
                cur = b.curve(sp, vy, who)
                if not cur:
                    continue
                gmax = cur[-1][3]
                jn = b.level("grid", sp, vy, ENTRY, 0.0, "none", who)
                tr = b.trace(sp, vy, who, cur[-1][0])
                k1, k2 = b.knee(sp, vy, who), b.knee(sp, vy, who, conservative=True)
                L += [f"### {geom_name(sp, vy)}", "",
                      f"`none` at ℓ_v = 0: {jn:.2f} s/part (net of the free transit {ENTRY / V_BELT:.2f} s: "
                      f"{jn - ENTRY / V_BELT:.2f}). Gain at the largest length ({cur[-1][0]:g} cm, "
                      f"{cur[-1][2]:.2f} s): {cur[-1][3]:+.2f} ±{cur[-1][4]:.2f} s/part "
                      f"({100 * cur[-1][3] / (jn - ENTRY / V_BELT):+.1f} % of the net flow). "
                      f"Knee (90 % of that gain): τ_vis {k1[0]:.2f} s = {k1[0] / tc:.2f} τ_cross (point estimate), "
                      f"{k2[0]:.2f} s = {k2[0] / tc:.2f} τ_cross (lower CI bound ≥ 0). "
                      f"Traces at the largest length (`when_idle`): {tr.get('moves', math.nan):.1f} pre-positioning "
                      f"moves per episode ({tr.get('parts', math.nan):.0f} parts), {tr.get('revoked', math.nan):.1f} revoked; "
                      f"planned travel t_x {tr.get('tx', math.nan):.2f} s, t_y {tr.get('ty', math.nan):.2f} s, "
                      f"max {tr.get('t', math.nan):.2f} s, y-dominated {100 * tr.get('ydom', math.nan):.0f} %; "
                      f"head idle {100 * tr.get('idle', math.nan):.0f} % of the time, waiting at the meeting point "
                      f"{tr.get('wait', math.nan):.0f} s per episode; effective anticipation "
                      f"{tr.get('ant', math.nan):.2f} s against the nominal {cur[-1][2]:.2f} s.", "",
                      "| ℓ_v [cm] | f = τ_vis/τ_cross | τ_vis [s] | G [s/part] | share of G(max) | none level [s] |",
                      "|---|---|---|---|---|---|"]
                for lv, f, tv, mu, h, nb in cur:
                    share = mu / gmax if gmax else math.nan
                    fs = f"{tv / tc:.2f}" + ("" if f is None else " (grid)")
                    L.append(f"| {lv:g} | {fs} | {tv:.2f} | {cell(mu, h, nb)} | {100 * share:.0f} % | "
                             f"{b.level('grid', sp, vy, ENTRY, lv, 'none', who):.2f} |")
                L.append("")
        # summary across geometries
        L += [f"### `{who}`: summary across geometries", "",
              "| spacing | v_y | τ_cross | G(1.5 τ_cross) [s/part] | G(10 cm) [s/part] | knee [s] (point / CI) | "
              "knee / τ_cross | t_y / t_x / max of the moves [s] | y-dominated | idle |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sp in SPACINGS:
            for vy in VY:
                tc = tau_cross(sp, vy)
                g15 = b.gain_s("grid", sp, vy, ENTRY, round(1.5 * tc * V_BELT, 6), who)
                g10 = b.gain_s("grid", sp, vy, ENTRY, 10.0, who)
                k1, k2 = b.knee(sp, vy, who), b.knee(sp, vy, who, conservative=True)
                tr = b.trace(sp, vy, who, round(1.5 * tc * V_BELT, 6))
                L.append(f"| {sp:g} | {vy:g} | {tc:.1f} | {cell(g15[0], g15[1])} | {cell(g10[0], g10[1])} | "
                         f"{k1[0]:.2f} / {k2[0]:.2f} | {k1[0] / tc:.2f} / {k2[0] / tc:.2f} | "
                         f"{tr.get('ty', math.nan):.2f} / {tr.get('tx', math.nan):.2f} / {tr.get('t', math.nan):.2f} | "
                         f"{100 * tr.get('ydom', math.nan):.0f} % | {100 * tr.get('idle', math.nan):.0f} % |")
        L.append("")
    # ---- H0: the 2.4 s triplet
    L += ["## H0: the τ_cross = 2.4 s triplet (6 / 5, 12 / 10, 18 / 15)", "",
          "Maximum absolute difference of the cost, realization by realization, over every grid length and mode, "
          "against spacing 12 / v_y 10:", ""]
    for who in b.whos:
        for sp, vy in ((6.0, 5.0), (18.0, 15.0)):
            dmax, n = 0.0, 0
            for lv, _ in b.lengths(12.0, 10.0, who):
                for mode in ("none", "when_idle"):
                    x, y = b.get("grid", sp, vy, ENTRY, lv, mode, who), b.get("grid", 12.0, 10.0, ENTRY, lv, mode, who)
                    for s in set(x) & set(y):
                        dmax = max(dmax, abs(x[s]["cost"] - y[s]["cost"]))
                        n += 1
            L.append(f"* `{who}`, spacing {sp:g} / v_y {vy:g}: max |ΔJ| = {dmax:.2e} s over {n} paired episodes.")
    L.append("")
    # ---- H3: physical entry length
    L += ["## H3: physical entry length (spacing 12, v_y 10; entry 10 cm against 35 cm, same visible lengths)", "",
          f"Free transit added by the longer entry: {(ENTRY - ENTRY_ORIG) / V_BELT:.2f} s per part.", ""]
    for who in b.whos:
        L += [f"| `{who}` ℓ_v [cm] | J none, entry 10 | J none, entry 35 | Δ none (35 − 10) [s] | "
              "G entry 10 [s/part] | G entry 35 [s/part] | ΔG (35 − 10) [s/part] |", "|---|---|---|---|---|---|---|"]
        for lv in CHECK_LENGTHS:
            n10, n35 = b.get("entry", 12.0, 10.0, ENTRY_ORIG, lv, "none", who), b.get("entry", 12.0, 10.0, ENTRY, lv, "none", who)
            w10, w35 = b.get("entry", 12.0, 10.0, ENTRY_ORIG, lv, "when_idle", who), b.get("entry", 12.0, 10.0, ENTRY, lv, "when_idle", who)
            dn = b.diff_s(n35, n10)
            g10, g35 = b.gain_s("entry", 12.0, 10.0, ENTRY_ORIG, lv, who), b.gain_s("entry", 12.0, 10.0, ENTRY, lv, who)
            seeds = sorted(set(n10) & set(n35) & set(w10) & set(w35))
            dg = ci([(n35[s]["cost"] - w35[s]["cost"]) - (n10[s]["cost"] - w10[s]["cost"]) for s in seeds])
            L.append(f"| {lv:g} | {b.level('entry', 12.0, 10.0, ENTRY_ORIG, lv, 'none', who):.2f} | "
                     f"{b.level('entry', 12.0, 10.0, ENTRY, lv, 'none', who):.2f} | {cell(dn[0], dn[1])} | "
                     f"{cell(g10[0], g10[1], g10[2])} | {cell(g35[0], g35[1], g35[2])} | {cell(*dg)} |")
        L.append("")
    # ---- checks
    viol = sum(r["mask_violations"] for r in b.rows)
    L += ["## Checks", "", f"* (iii) visible stretch respected: {viol} candidate violations over {len(b.rows)} episodes."]
    for who in b.whos:
        dmax = 0.0
        for r in b.rows:
            if r["who"] == who and r["lv"] == 0.0 and r["mode"] == "when_idle":
                other = b.get(r["bench"], r["spacing"], r["vy"], r["entry"], 0.0, "none", who).get(r["seed"])
                if other:
                    dmax = max(dmax, abs(r["cost"] - other["cost"]))
        L.append(f"* (ii) `{who}`: at ℓ_v = 0, max |J(when_idle) − J(none)| = {dmax:.2e} s.")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def figures(b: Bench, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    tcs = sorted({tau_cross(sp, vy) for sp in SPACINGS for vy in VY})
    for who in b.whos:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for sp in SPACINGS:
            for vy in VY:
                tc = tau_cross(sp, vy)
                cur = b.curve(sp, vy, who)
                if not cur:
                    continue
                col = cmap(tcs.index(tc) / max(1, len(tcs) - 1))
                tv = np.array([c[2] for c in cur]); g = np.array([c[3] for c in cur]); h = np.array([c[4] for c in cur])
                ls = {5.0: "-", 10.0: "--", 15.0: ":"}[vy]
                lab = f"{sp:g} cm / {vy:g} cm/s, τ_cross {tc:.1f} s"
                axes[0].plot(tv, g, ls, color=col, marker="o", ms=3, lw=1.2, label=lab)
                axes[0].fill_between(tv, g - h, g + h, color=col, alpha=0.08)
                gmax = cur[-1][3]
                if gmax:
                    axes[1].plot(tv / tc, g / gmax, ls, color=col, marker="o", ms=3, lw=1.2)
                moves = [b.trace(sp, vy, who, c[0]).get("moves", math.nan) for c in cur]
                axes[2].plot(tv, moves, ls, color=col, marker="o", ms=3, lw=1.2)
        axes[2].set_xlabel("anticipation τ_vis [s]")
        axes[2].set_ylabel("pre-positioning moves per episode (≈ 70 parts)")
        axes[2].set_ylim(bottom=0)
        axes[2].set_title("opportunities taken", fontsize=9)
        axes[0].set_xlabel("anticipation τ_vis = ℓ_v / v_belt [s]")
        axes[0].set_ylabel("gain of pre-positioning G [s per part]")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_title("in seconds", fontsize=9)
        axes[1].set_xlabel("τ_vis / τ_cross")
        axes[1].set_ylabel("share of the gain at the largest length")
        axes[1].axhline(0.9, color="grey", lw=0.6, ls="--")
        axes[1].axvline(1.0, color="grey", lw=0.6, ls="--")
        axes[1].set_xlim(0, 4.5)
        axes[1].set_title("normalized by the crossing time", fontsize=9)
        axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
        fig.suptitle(f"{who}: pre-positioning gain against the camera's anticipation, nine geometries", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"{tag}_{who}.{ext}", dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ checks
def check_original(args) -> None:
    """Check (i) of §3: the original cell run by this script equals ``rl2_eval.run``."""
    import rl2_eval
    _init_worker()
    worst = 0.0
    for s in range(args.seed_base, args.seed_base + 3):
        for lv in (0.0, 5.0, 10.0):
            for mode in ("none", "when_idle"):
                for who in args.who:
                    ours = run(dict(bench="entry", spacing=12.0, vy=10.0, tau_cross=2.4, entry=ENTRY_ORIG,
                                    kind="abs", frac=None, lv=lv, tau_vis=lv / V_BELT, mode=mode, who=who,
                                    seed=s, net=str(ROOT / args.net)))
                    theirs = rl2_eval.run(dict(load="real", arrivals="regular", travel="nominal", slow=0.5,
                                               seed=s, net=str(ROOT / args.net), ref=None,
                                               t=f"L{lv:g}/{mode}", obs="belts", mode=mode, visible=lv, who=who))
                    d = abs(ours["cost"] - theirs["cost"])
                    worst = max(worst, d)
                    print(f"seed {s} L{lv:g}/{mode} {who}: {ours['cost']:.6f} vs {theirs['cost']:.6f} (Δ {d:.1e})")
    print(f"check (i): max |Δ| = {worst:.1e} s")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="geometry_pilot")
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--who", nargs="+", default=["nearest_job", "net"], choices=["nearest_job", "net"])
    ap.add_argument("--net", default=NET)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--check-original", action="store_true")
    args = ap.parse_args()
    if args.check_original:
        check_original(args)
        return
    if 7000 <= args.seed_base <= 7029:
        sys.exit("seeds 7000–7029 are the reserved confirmatory seeds, used once")
    out_dir = ROOT / "results/experiments" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        J = jobs(args)
        print(f"{len(J)} episodes", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
            # the network's episodes first (slower), so the pool stays busy to the end
            J.sort(key=lambda j: j["who"] != "net")
            rows = list(ex.map(run, J, chunksize=4))
        res = {"args": vars(args), "elapsed_s": time.perf_counter() - t0, "rows": rows}
        json.dump(res, open(out_dir / "bench.json", "w"))
        print(f"elapsed {res['elapsed_s']:.0f} s", flush=True)
    res = json.load(open(out_dir / "bench.json"))
    b = Bench(res["rows"])
    md = tables(b, args)
    (out_dir / "tables.md").write_text(md)
    figures(b, args.tag)
    print(md)
    print(f"written {out_dir / 'tables.md'} and {FIG}/{args.tag}_*.pdf")


if __name__ == "__main__":
    main()
