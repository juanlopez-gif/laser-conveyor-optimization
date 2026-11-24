"""Belt-speed bench (confirmatory design, NOT yet run): when the belt runs faster, which
scale of the camera's reach transfers, the centimetres of belt or the seconds of
anticipation, or neither because the interception changes?

Frozen protocol: ``docs/methods/geometry.md`` §7 (written before the run). Belt speed
3 and 5 cm/s × four cells of the geometry bench (spacing 12 / v_y 10 and 6 / 15, each
with v_x 5 and 10 cm/s), nominal load, Poisson arrivals, a common grid of visible
lengths in centimetres, ``none`` against ``when_idle`` at every length, the SAME
arrival realizations at both speeds (the arrival streams are seeded per belt and do
not depend on the belt speed), ``nearest_job`` (primary) and the frozen network
(secondary), a common 50 cm entry zone (10 s at 5 cm/s). Smoke test on development
seeds and the confirmatory run:

    .venv/bin/python experiments/belt_speed_bench.py --tag _belt_smoke --seed-base 3100 --seeds 2 --cells 0
    .venv/bin/python experiments/belt_speed_bench.py --tag belt_speed_bench --confirmatory
    python experiments/belt_speed_bench.py --tag belt_speed_bench --analyse-only
    python experiments/belt_speed_bench.py --from-extract docs/papers/paper2/data/belt_speed_bench.csv.gz

The reserved seed range (``--confirmatory``) is fixed by the owner and used once: the
script refuses ``--confirmatory`` while ``RESERVED_BASE`` is None, refuses to overwrite
the ``bench.json`` of a confirmatory run, and refuses the used ranges 7000–7029 and
9000–9059 (and the reserved one) without the flag. Outputs:
``results/experiments/<tag>/{bench.json,tables.md,extract.csv.gz}`` and, for a tag
without a leading underscore, ``docs/methods/figures/<tag>_*.pdf`` (development tags
keep their figures under ``results/experiments/<tag>/figures/``).
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


def _rl2():
    """``rl2.expert_iteration`` imports PyTorch; it is needed to run episodes, not to analyse
    the extract, so the import is deferred (tables and figures regenerate with NumPy and
    Matplotlib only, as in ``geometry_bench.py``)."""
    from rl2.expert_iteration import _init_worker, val_config
    return _init_worker, val_config


# ------------------------------------------------------------------ frozen design (geometry.md §7)
V_BELTS = (3.0, 5.0)                                   # belt speed [cm/s]; 3 = the geometry bench
V_REF = 3.0                                            # the speed already measured
CELLS = (dict(spacing=12.0, vy=10.0, vx=5.0), dict(spacing=12.0, vy=10.0, vx=10.0),
         dict(spacing=6.0, vy=15.0, vx=5.0), dict(spacing=6.0, vy=15.0, vx=10.0))
LOAD = (10.0, 12.0, 14.0)                              # nominal: mean inter-arrivals [s] (ρ_p 0.64)
PROC = 2.5
ENTRY = 50.0                                           # common physical entry zone [cm]: 10 s at 5 cm/s
GRID_CM = (0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 10.0, 15.0, 50 / 3, 25.0, 30.0, 50.0)   # visible length [cm]
L_REF = 50.0                                           # the full view the costs are taken against
L_10, L_167 = 10.0, 50 / 3                             # the two candidate lengths of the frozen question
MARGIN = 0.5                                           # s/part: "keeps the benefit" (owner's decision, §7.2)
MIN_USEFUL, MIN_USEFUL_LB, SHARE = 1.0, 0.5, 0.9      # benefit criterion and knee share, as in §5
Z95 = 1.96
Z_BONF = 2.838            # two-sided 95 % simultaneous over the 11 non-zero grid lengths (2.773 for 9 in §5)
RESERVED_BASE = None      # reserved range to be fixed by the owner; used once
RESERVED_N = 60
USED = ((7000, 30), (9000, 60))                        # ranges already used once (phase 4, geometry bench)
NET = "models/rl2_gen_s_b.pt"
KEEP = ("avg_flow_time", "blocking_events", "total_blocked_time", "total_simulation_time",
        "decisions", "idle_time", "wait_time", "prepos_moves", "revocations", "prepos_tx_mean",
        "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "prepos_wip_mean", "anticipation_eff")
EXTRACT = ("bench", "vbelt", "spacing", "vy", "vx", "tau_cross", "load", "arrivals", "entry", "lv", "s",
           "mode", "who", "seed", "cost", "avg_flow_time", "blocking_events", "total_blocked_time", "parts",
           "arr_sum", "total_simulation_time", "idle_time", "wait_time", "prepos_moves", "revocations",
           "prepos_tx_mean", "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "prepos_wip_mean",
           "anticipation_eff")
FIG = ROOT / "docs/methods/figures"


def tau_cross(spacing, vy):
    return 2 * spacing / vy


def rho():
    return PROC * sum(1 / x for x in LOAD)


def s_of(lv, vbelt):
    """Nominal anticipation [s] of a visible length [cm] at a belt speed."""
    return lv / vbelt


def free_transit(vbelt):
    return ENTRY / vbelt


def same_s_pairs():
    """Grid lengths that give the same anticipation at the two speeds: (ℓ at 3, ℓ at 5)."""
    out = []
    for a in GRID_CM:
        for c in GRID_CM:
            if a > 0 and abs(s_of(a, V_BELTS[0]) - s_of(c, V_BELTS[1])) < 1e-6:
                out.append((a, c))
    return out


def cell_key(c):
    return (c["spacing"], c["vy"], c["vx"])


def cell_name(c):
    return f"spacing {c['spacing']:g} / v_y {c['vy']:g} / v_x {c['vx']:g} (τ_cross {tau_cross(c['spacing'], c['vy']):.1f} s)"


def fmt_l(lv):
    return f"{lv:.1f}" if abs(lv - round(lv)) > 1e-6 else f"{lv:g}"


# ------------------------------------------------------------------ episodes
def extra(job):
    return ["simulation.observation=belts", f"laser.prepositioning={job['mode']}",
            f"simulation.visible_length={job['lv']}", "conveyors.y_positions=null",
            f"conveyors.spacing={job['spacing']}", f"conveyors.speed={job['vbelt']}",
            f"laser.speed_x={job['vx']}", f"laser.speed_y={job['vy']}",
            f"geometry.entry_length={job['entry']}",
            f"arrivals.stochastic.mean_interarrival=[{', '.join(str(x) for x in LOAD)}]",
            f"processing.time={PROC}"]


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
    cfg = _rl2()[1]("real", job["seed"], "none", extra=tuple(extra(job)))
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
            "arr_sum": float(sum(a[0] for a in sim.arr)),      # same-arrivals check across speeds
            "secs": time.perf_counter() - t0}
    out |= {k: s[k] for k in KEEP}
    return out


def jobs(args):
    seeds = [args.seed_base + i for i in range(args.seeds)]
    net = str(ROOT / args.net)
    cells = [CELLS[i] for i in args.cells] if args.cells else list(CELLS)
    J = []
    for vb in V_BELTS:
        for c in cells:
            for lv in GRID_CM:
                for mode in ("none", "when_idle"):
                    for who in args.who:
                        for seed in seeds:
                            J.append(c | dict(bench="speed", vbelt=vb, tau_cross=tau_cross(c["spacing"], c["vy"]),
                                              load="nominal", arrivals="poisson", entry=ENTRY,
                                              lv=round(lv, 6), s=round(s_of(lv, vb), 6), mode=mode, who=who,
                                              seed=seed, net=net))
    return J


# ------------------------------------------------------------------ statistics
def ci(d, z=Z95):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else math.nan), math.nan
    return float(d.mean()), float(z * d.std(ddof=1) / math.sqrt(len(d)))


def cell_txt(mu, h, n=None):
    s = f"{mu:+.2f} ±{h:.2f}"
    return s if n is None else f"{s} ({n})"


class Bench:
    """Per-episode rows indexed by (belt speed, cell, visible length, mode, controller)."""

    def __init__(self, rows):
        self.rows = rows
        self.by = {}
        for r in rows:
            self.by.setdefault((r["vbelt"], cell_key(r), round(r["lv"], 6), r["mode"], r["who"]), {})[r["seed"]] = r
        self.seeds = sorted({r["seed"] for r in rows})
        self.whos = [w for w in ("nearest_job", "net") if any(r["who"] == w for r in rows)]
        keys = {cell_key(x) for x in rows}
        self.cells = [c for c in CELLS if cell_key(c) in keys] or list(CELLS)
        self.vbelts = [v for v in V_BELTS if any(r["vbelt"] == v for r in rows)]

    def get(self, c, vb, lv, mode, who):
        return self.by.get((vb, cell_key(c), round(lv, 6), mode, who), {})

    def paired(self, x, y, field="cost", z=Z95):
        seeds = sorted(set(x) & set(y))
        d = [x[k][field] - y[k][field] for k in seeds]
        mu, h = ci(d, z)
        return mu, h, sum(v > 1e-12 for v in d), len(seeds)

    def gain(self, c, vb, lv, who, z=Z95):
        """G_v(ℓ): seconds per part saved by pre-positioning at visible length ℓ (none − when_idle)."""
        return self.paired(self.get(c, vb, lv, "none", who), self.get(c, vb, lv, "when_idle", who), z=z)

    def gain_d(self, c, vb, lv, who):
        """Per-seed gains (for cross-speed pairing: the same seed is the same arrival realization)."""
        a, b = self.get(c, vb, lv, "none", who), self.get(c, vb, lv, "when_idle", who)
        return {k: a[k]["cost"] - b[k]["cost"] for k in set(a) & set(b)}

    def cost(self, c, vb, lv, who):
        """C_v(ℓ): J_when_idle(ℓ) − J_when_idle(ℓ_ref), paired; positive = the shorter view costs."""
        return self.paired(self.get(c, vb, lv, "when_idle", who), self.get(c, vb, L_REF, "when_idle", who))

    def level(self, c, vb, lv, mode, who, field="cost"):
        rows = self.get(c, vb, lv, mode, who)
        return float(np.mean([r[field] for r in rows.values()])) if rows else math.nan

    def useful(self, c, vb, who):
        mu, h, _, _ = self.gain(c, vb, L_REF, who)
        return mu >= MIN_USEFUL and mu - h >= MIN_USEFUL_LB

    def knee(self, c, vb, who, conservative=False):
        """First grid length [cm] whose paired contrast G(ℓ) − SHARE·G(ℓ_ref) is ≥ 0 (point) or has a
        lower simultaneous-95 % bound ≥ 0 (conservative, Bonferroni over the 11 non-zero lengths)."""
        rn, rw = self.get(c, vb, L_REF, "none", who), self.get(c, vb, L_REF, "when_idle", who)
        for lv in GRID_CM[1:]:
            a, b = self.get(c, vb, lv, "none", who), self.get(c, vb, lv, "when_idle", who)
            seeds = sorted(set(a) & set(b) & set(rn) & set(rw))
            if not seeds:
                return math.nan
            d = [(a[k]["cost"] - b[k]["cost"]) - SHARE * (rn[k]["cost"] - rw[k]["cost"]) for k in seeds]
            mu, h = ci(d, Z_BONF if conservative else Z95)
            if (mu - h >= 0) if conservative else (mu >= 0):
                return lv
        return math.nan

    def trace(self, c, vb, who, lv=L_REF, mode="when_idle"):
        rows = self.get(c, vb, lv, mode, who)
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

    # ---- the frozen decision (§7.3)
    @staticmethod
    def keeps(cost, conservative=True):
        """Does a shorter view keep the benefit? Conservative: the upper 95 % bound of its paired
        cost against the full view is ≤ MARGIN (yes), its lower bound is > MARGIN (no), else
        undetermined. Point: the point estimate against MARGIN."""
        mu, h = cost[0], cost[1]
        if math.isnan(mu):
            return "n/a"
        if not conservative:
            return "yes" if mu <= MARGIN else "no"
        if mu + h <= MARGIN:
            return "yes"
        if mu - h > MARGIN:
            return "no"
        return "undetermined"

    def verdict(self, c, who, vb=5.0, conservative=True):
        """H_cm / H_s / H_neither / undetermined for one cell at the faster belt; 'benefit lost' when the
        cell no longer passes the usefulness criterion there (a design result, the question moot)."""
        if math.isnan(self.gain(c, vb, L_REF, who)[0]):
            return "n/a"
        if not self.useful(c, vb, who):
            return "benefit lost at this speed"
        k10 = self.keeps(self.cost(c, vb, L_10, who), conservative)
        k167 = self.keeps(self.cost(c, vb, L_167, who), conservative)
        if k10 == "yes":
            return "H_cm"
        if k10 == "no" and k167 == "yes":
            return "H_s"
        if k167 == "no":
            return "H_neither"
        return "undetermined"


# ------------------------------------------------------------------ tables
def tables(b: Bench, args) -> str:
    pairs = same_s_pairs()
    L = [f"# Belt-speed bench `{args.tag}`", "",
         f"Seeds {b.seeds[0]}–{b.seeds[-1]} ({len(b.seeds)} paired realizations per configuration; the same "
         f"arrival realizations at both belt speeds). Belt speeds {', '.join(f'{v:g}' for v in V_BELTS)} cm/s; "
         f"entry zone {ENTRY:g} cm (free transit {free_transit(V_BELTS[0]):.2f} / {free_transit(V_BELTS[1]):.2f} s); "
         f"processing {PROC:g} s; nominal load {'/'.join(f'{x:g}' for x in LOAD)} s (ρ_p {rho():.2f}), Poisson. "
         "G_v(ℓ) = seconds per part saved by pre-positioning at belt speed v and visible length ℓ (`none` − "
         "`when_idle` at the same length, paired, 95 % CI). C_v(ℓ) = `when_idle`(ℓ) − `when_idle`"
         f"({L_REF:g} cm), paired: the cost of the shorter view. Useful benefit: G_v({L_REF:g} cm) ≥ {MIN_USEFUL} "
         f"s/part with lower bound ≥ {MIN_USEFUL_LB}. A view keeps the benefit when C_v(ℓ) ≤ {MARGIN} s/part "
         "(conservative: upper 95 % bound; point: the estimate). Knee: first grid length retaining "
         f"{100 * SHARE:.0f} % of G_v({L_REF:g} cm), point / lower simultaneous-95 % bound (z = {Z_BONF}) ≥ 0. "
         "Protocol: `docs/methods/geometry.md` §7.", ""]
    for who in b.whos:
        # ---- frozen verdict
        L += [f"## `{who}`: the frozen question at {V_BELTS[1]:g} cm/s (H_cm: 10 cm keeps the benefit; H_s: 10 cm "
              f"does not but {L_167:.1f} cm does; H_neither: neither does)", "",
              f"| cell | G_5({L_REF:g} cm) [s/part] | useful at 5 | C_5(10 cm = {s_of(L_10, 5.0):.2f} s) | "
              f"C_5({L_167:.1f} cm = {s_of(L_167, 5.0):.2f} s) | keeps (10 / 16.7) conservative | keeps point | "
              f"verdict conservative | verdict point | C_3(10 cm = {s_of(L_10, 3.0):.2f} s) (check) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in b.cells:
            g5 = b.gain(c, 5.0, L_REF, who)
            if math.isnan(g5[0]):
                continue
            c10, c167 = b.cost(c, 5.0, L_10, who), b.cost(c, 5.0, L_167, who)
            c3 = b.cost(c, 3.0, L_10, who)
            L.append(f"| {cell_name(c)} | {cell_txt(g5[0], g5[1])} | {'yes' if b.useful(c, 5.0, who) else 'no'} | "
                     f"{cell_txt(c10[0], c10[1])} | {cell_txt(c167[0], c167[1])} | "
                     f"{b.keeps(c10)} / {b.keeps(c167)} | {b.keeps(c10, False)} / {b.keeps(c167, False)} | "
                     f"**{b.verdict(c, who)}** | {b.verdict(c, who, conservative=False)} | "
                     f"{cell_txt(c3[0], c3[1])} ({b.keeps(c3)}) |")
        L.append("")
        # ---- summary per cell and speed
        L += [f"## `{who}`: summary per cell and belt speed", "",
              f"| cell | v_belt | `none`(0) [s/part] | net of free transit | G({L_REF:g} cm) | useful | "
              f"G at 10 / {L_167:.1f} cm | C at 10 / {L_167:.1f} cm vs {L_REF:g} cm | knee [cm] point / cons. | knee [s] | "
              "moves | idle (`none`) | t_x / t_y / max [s] | y-dom | eff. anticipation [s] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in b.cells:
            for vb in b.vbelts:
                g = b.gain(c, vb, L_REF, who)
                if math.isnan(g[0]):
                    continue
                gs = [b.gain(c, vb, lv, who) for lv in (L_10, L_167)]
                cs = [b.cost(c, vb, lv, who) for lv in (L_10, L_167)]
                tr = b.trace(c, vb, who)
                idle0 = b.trace(c, vb, who, 0.0, "none").get("idle", math.nan)
                k1, k2 = b.knee(c, vb, who), b.knee(c, vb, who, conservative=True)
                u = b.useful(c, vb, who)
                n0 = b.level(c, vb, 0.0, "none", who)
                L.append(f"| {cell_name(c)} | {vb:g} | {n0:.1f} | {n0 - free_transit(vb):.1f} | {cell_txt(g[0], g[1])} | "
                         f"{'yes' if u else 'no'} | " + " / ".join(f"{x[0]:+.2f}" for x in gs) + " | "
                         + " / ".join(f"{x[0]:+.2f}±{x[1]:.2f}" for x in cs) + " | "
                         f"{k1:.1f} / {k2:.1f}" + ("" if u else " (undetermined)") + " | "
                         f"{s_of(k1, vb):.2f} / {s_of(k2, vb):.2f} | {tr.get('moves', math.nan):.1f} | {100 * idle0:.0f} % | "
                         f"{tr.get('tx', math.nan):.2f} / {tr.get('ty', math.nan):.2f} / {tr.get('t', math.nan):.2f} | "
                         f"{100 * tr.get('ydom', math.nan):.0f} % | {tr.get('ant', math.nan):.2f} |")
        L.append("")
        # ---- curves
        L += [f"## `{who}`: coverage curves (G_v [s/part] at each visible length; share of G_v({L_REF:g} cm) in parentheses)", "",
              "| cell | v_belt | " + " | ".join(f"{fmt_l(lv)} cm" for lv in GRID_CM) + " |", "|---|---|" + "---|" * len(GRID_CM),
              "| anticipation [s] at 3 cm/s | | " + " | ".join(f"{s_of(lv, 3.0):.2f}" for lv in GRID_CM) + " |",
              "| anticipation [s] at 5 cm/s | | " + " | ".join(f"{s_of(lv, 5.0):.2f}" for lv in GRID_CM) + " |"]
        for c in b.cells:
            for vb in b.vbelts:
                gref = b.gain(c, vb, L_REF, who)[0]
                if math.isnan(gref):
                    continue
                cells = []
                for lv in GRID_CM:
                    mu, h, nb, n = b.gain(c, vb, lv, who)
                    cells.append(f"{mu:+.2f}±{h:.2f} ({100 * mu / gref:.0f} %)" if gref else f"{mu:+.2f}±{h:.2f}")
                L.append(f"| {cell_name(c)} | {vb:g} | " + " | ".join(cells) + " |")
        L.append("")
        # ---- cross-speed comparisons (paired per seed: the same arrivals at both speeds)
        if len(b.vbelts) == 2:
            v3, v5 = b.vbelts
            L += [f"## `{who}`: same distance, G_{v3:g}(ℓ) − G_{v5:g}(ℓ) [s/part], paired per realization", "",
                  "| cell | " + " | ".join(f"{fmt_l(lv)} cm" for lv in GRID_CM) + " | mean abs. share difference |",
                  "|---|" + "---|" * (len(GRID_CM) + 1)]
            for c in b.cells:
                g3r, g5r = b.gain(c, v3, L_REF, who)[0], b.gain(c, v5, L_REF, who)[0]
                if math.isnan(g3r) or math.isnan(g5r):
                    continue
                cells, dsh = [], []
                for lv in GRID_CM:
                    a, d = b.gain_d(c, v3, lv, who), b.gain_d(c, v5, lv, who)
                    mu, h = ci([a[k] - d[k] for k in sorted(set(a) & set(d))])
                    cells.append(f"{mu:+.2f}±{h:.2f}")
                    if lv > 0 and g3r and g5r:
                        dsh.append(abs(np.mean(list(a.values())) / g3r - np.mean(list(d.values())) / g5r))
                L.append(f"| {cell_name(c)} | " + " | ".join(cells) + f" | {100 * np.mean(dsh):.0f} pts |")
            L += ["", f"## `{who}`: same anticipation, G_{v3:g}(ℓ₃) − G_{v5:g}(ℓ₅) [s/part] with ℓ₃ / {v3:g} = ℓ₅ / {v5:g}, paired", "",
                  "| cell | " + " | ".join(f"{s_of(a, v3):.2f} s ({fmt_l(a)} / {fmt_l(d)} cm)" for a, d in pairs)
                  + f" | G_{v3:g}({L_REF:g}) − G_{v5:g}({L_REF:g}) | mean abs. share difference |",
                  "|---|" + "---|" * (len(pairs) + 2)]
            for c in b.cells:
                g3r, g5r = b.gain(c, v3, L_REF, who)[0], b.gain(c, v5, L_REF, who)[0]
                if math.isnan(g3r) or math.isnan(g5r):
                    continue
                cells, dsh = [], []
                for a_l, d_l in pairs:
                    a, d = b.gain_d(c, v3, a_l, who), b.gain_d(c, v5, d_l, who)
                    mu, h = ci([a[k] - d[k] for k in sorted(set(a) & set(d))])
                    cells.append(f"{mu:+.2f}±{h:.2f}")
                    if g3r and g5r:
                        dsh.append(abs(np.mean(list(a.values())) / g3r - np.mean(list(d.values())) / g5r))
                a, d = b.gain_d(c, v3, L_REF, who), b.gain_d(c, v5, L_REF, who)
                mu, h = ci([a[k] - d[k] for k in sorted(set(a) & set(d))])
                L.append(f"| {cell_name(c)} | " + " | ".join(cells) + f" | {mu:+.2f}±{h:.2f} | {100 * np.mean(dsh):.0f} pts |")
            L.append("")
    # ---- checks
    viol = sum(r["mask_violations"] for r in b.rows)
    L += ["## Checks", "", f"* visible stretch respected: {viol} candidate violations over {len(b.rows)} episodes."]
    sig = {}
    for r in b.rows:
        sig.setdefault((cell_key(r), r["seed"]), set()).add((r["parts"], round(r["arr_sum"], 6)))
    bad = sum(1 for v in sig.values() if len(v) > 1)
    L.append(f"* same arrivals at both speeds, every length, mode and controller: {bad} of {len(sig)} (cell, seed) "
             "pairs differ in the number of parts or the sum of their arrival instants"
             + (" (as required)." if bad == 0 else " (**the realizations are not common**)."))
    for who in b.whos:
        dmax = 0.0
        for r in b.rows:
            if r["who"] == who and r["lv"] == 0.0 and r["mode"] == "when_idle":
                o = b.by.get((r["vbelt"], cell_key(r), 0.0, "none", who), {}).get(r["seed"])
                if o:
                    dmax = max(dmax, abs(r["cost"] - o["cost"]))
        L.append(f"* `{who}`: at ℓ = 0, max |J(when_idle) − J(none)| = {dmax:.2e} s"
                 + (" (the mode is a context token of the network)." if who == "net" else "."))
    secs = sum(r["secs"] for r in b.rows)
    L += [f"* {len(b.rows)} episodes, {secs / 3600:.2f} CPU-hours.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def figures(b: Bench, tag: str, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig_dir.mkdir(parents=True, exist_ok=True)
    col = {3.0: "#1f77b4", 5.0: "#d62728"}
    for who in b.whos:
        # the curves twice: x in cm and x in s, both speeds overlaid in every panel
        n = len(b.cells)
        fig, axes = plt.subplots(n, 2, figsize=(9, 2.4 * n + 0.6), squeeze=False)
        for i, c in enumerate(b.cells):
            for j, unit in enumerate(("cm", "s")):
                ax = axes[i][j]
                for vb in b.vbelts:
                    g = np.array([b.gain(c, vb, lv, who)[0] for lv in GRID_CM])
                    h = np.array([b.gain(c, vb, lv, who)[1] for lv in GRID_CM])
                    if np.isnan(g).all():
                        continue
                    x = np.array(GRID_CM) if unit == "cm" else np.array([s_of(lv, vb) for lv in GRID_CM])
                    ax.plot(x, g, "-", color=col[vb], marker="o", ms=2.5, lw=1.1, label=f"belt {vb:g} cm/s")
                    ax.fill_between(x, g - h, g + h, color=col[vb], alpha=0.10)
                ax.axhline(0, color="k", lw=0.5)
                if unit == "cm":
                    for lv in (L_10, L_167):
                        ax.axvline(lv, color="grey", lw=0.6, ls=":")
                    ax.set_xlabel("visible length ℓ_v [cm]")
                else:
                    ax.axvline(s_of(L_10, 3.0), color="grey", lw=0.6, ls=":")
                    ax.set_xlabel("anticipation τ_vis = ℓ_v / v_belt [s]")
                if j == 0:
                    ax.set_ylabel("G [s per part]")
                ax.set_title(cell_name(c), fontsize=8)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{who}: pre-positioning gain against the visible length (left) and the anticipation (right); "
                     f"dotted: 10 and {L_167:.1f} cm, {s_of(L_10, 3.0):.2f} s", fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"{tag}_{who}_curves.{ext}", dpi=140)
        plt.close(fig)
        # the costs of the two candidate lengths at each speed, against the margin
        fig, ax = plt.subplots(figsize=(7, 3.2))
        xs, labels = [], []
        for i, c in enumerate(b.cells):
            for k, vb in enumerate(b.vbelts):
                for m, lv in enumerate((L_10, L_167)):
                    mu, h, _, _ = b.cost(c, vb, lv, who)
                    x = i * 5 + k * 2 + m * 0.8
                    ax.errorbar(x, mu, yerr=h, fmt={3.0: "o", 5.0: "s"}[vb], color=col[vb], ms=4, capsize=2,
                                mfc="none" if m == 1 else None)
            xs.append(i * 5 + 1.9); labels.append(f"{c['spacing']:g}/{c['vy']:g}/v_x {c['vx']:g}")
        ax.axhline(MARGIN, color="grey", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(f"C_v(ℓ) = when_idle(ℓ) − when_idle({L_REF:g} cm) [s/part]")
        ax.set_title(f"{who}: cost of 10 cm (filled) and {L_167:.1f} cm (hollow) of visible belt\n"
                     f"at 3 (blue circles) and 5 cm/s (red squares); dashed = margin {MARGIN} s/part", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fig_dir / f"{tag}_{who}_cost.{ext}", dpi=140)
        plt.close(fig)


# ------------------------------------------------------------------ main
def load_extract(path: Path):
    NUM = {"vbelt", "spacing", "vy", "vx", "tau_cross", "entry", "lv", "s", "cost", "avg_flow_time",
           "total_blocked_time", "arr_sum", "total_simulation_time", "idle_time", "wait_time",
           "prepos_tx_mean", "prepos_ty_mean", "prepos_t_mean", "prepos_y_dominated", "prepos_wip_mean",
           "anticipation_eff"}
    INT = {"seed", "blocking_events", "parts", "prepos_moves", "revocations"}
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            row = {k: (None if v == "" else float(v) if k in NUM else int(v) if k in INT else v)
                   for k, v in r.items()}
            rows.append(row | {"mask_violations": 0, "secs": 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="belt_speed_bench")
    ap.add_argument("--seed-base", type=int, default=3100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--confirmatory", action="store_true",
                    help="the reserved seeds (RESERVED_BASE, fixed by the owner), used once")
    ap.add_argument("--cells", type=int, nargs="*", default=None,
                    help=f"development only: indices into CELLS (0–{len(CELLS) - 1}) to run a subset")
    ap.add_argument("--who", nargs="+", default=["nearest_job", "net"], choices=["nearest_job", "net"])
    ap.add_argument("--net", default=NET)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--from-extract", type=Path, default=None,
                    help="analyse the versioned per-episode extract (csv.gz) instead of bench.json")
    args = ap.parse_args()
    out_dir = ROOT / "results/experiments" / args.tag
    fig_dir = out_dir / "figures" if args.tag.startswith("_") else FIG
    if args.from_extract is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        b = Bench(load_extract(args.from_extract))
        md = tables(b, args)
        (out_dir / "tables_from_extract.md").write_text(md)
        figures(b, args.tag, fig_dir)
        print(md)
        return
    if args.confirmatory:
        if RESERVED_BASE is None:
            sys.exit("the reserved seed range is not fixed yet (RESERVED_BASE): the owner sets it before the run")
        if args.cells:
            sys.exit("--cells is for development runs only")
        args.seed_base, args.seeds = RESERVED_BASE, RESERVED_N
        if (out_dir / "bench.json").exists() and not args.analyse_only:
            sys.exit(f"{out_dir / 'bench.json'} exists: the reserved seeds are used once")
    elif not args.analyse_only:
        lo, hi = args.seed_base, args.seed_base + args.seeds - 1
        for b0, n in USED + (((RESERVED_BASE, RESERVED_N),) if RESERVED_BASE is not None else ()):
            if lo <= b0 + n - 1 and hi >= b0:
                sys.exit(f"seeds {b0}–{b0 + n - 1} are reserved or already used (--confirmatory runs the reserved range once)")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        J = jobs(args)
        print(f"{len(J)} episodes", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers, initializer=_rl2()[0]) as ex:
            J.sort(key=lambda j: j["who"] != "net")
            rows = list(ex.map(run, J, chunksize=8))
        res = {"args": vars(args), "design": {"v_belts": V_BELTS, "cells": CELLS, "grid_cm": GRID_CM, "load": LOAD,
                                              "proc": PROC, "entry": ENTRY, "l_ref": L_REF, "margin": MARGIN,
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
    figures(b, args.tag, fig_dir)
    print(md)
    print(f"written {out_dir} and {fig_dir}/{args.tag}_*.pdf")


if __name__ == "__main__":
    main()
