"""Tables and figures of Paper 2's geometry bench (docs/papers/paper2, Section 6) from the
versioned per-episode extract ``docs/papers/paper2/data/geometry_bench.csv.gz``
(written by ``experiments/geometry_bench.py --confirmatory``; protocol and results in
``docs/methods/geometry.md`` §5–§6).

    python experiments/paper2_bench_figures.py

Writes ``docs/papers/paper2/tables/bench_*.tex`` and ``figures/bench_*.pdf`` and prints
the headline numbers quoted in the text.
"""

from __future__ import annotations

import csv
import gzip
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from geometry_bench import (ARRIVALS, COST_S, GRID_S, LOADS, MODELS, SPACINGS, VX, VY, Bench,  # noqa: E402
                            loo_mae, regime_key, rho, spearman, tau_cross)

OUT = ROOT / "docs/papers/paper2"
DATA = OUT / "data/geometry_bench.csv.gz"
DISTINCT = [(sp, vy) for sp in SPACINGS for vy in VY if (sp, vy) not in ((6.0, 5.0), (18.0, 15.0))]
TC_ORDER = sorted({tau_cross(sp, vy) for sp, vy in DISTINCT})
ARR = {"poisson": "Poisson", "burst": "bursts"}
NUM = {"spacing", "vy", "vx", "tau_cross", "entry", "s", "lv", "cost", "avg_flow_time", "total_blocked_time",
       "total_simulation_time", "idle_time", "wait_time", "prepos_tx_mean", "prepos_ty_mean", "prepos_t_mean",
       "prepos_y_dominated", "prepos_wip_mean", "anticipation_eff"}
INT = {"seed", "blocking_events", "parts", "prepos_moves", "revocations"}


def load(path: Path) -> Bench:
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (None if v == "" else float(v) if k in NUM else int(v) if k in INT else v)
                         for k, v in r.items()} | {"mask_violations": 0, "secs": 0.0})
    return Bench(rows)


def fmt(mu, h, bold=False):
    s = f"${mu:+.2f} \\pm {h:.2f}$"
    return "{\\boldmath" + s + "}" if bold else s


def reg(sp, vy, vx, ld, ar):
    return dict(spacing=sp, vy=vy, vx=vx, load=ld, arrivals=ar, tau_cross=tau_cross(sp, vy))


def distinct_regimes():
    return [reg(sp, vy, vx, ld, ar) for sp, vy in DISTINCT for vx in VX for ld in LOADS for ar in ARRIVALS]


def useful(b, r, who="nearest_job"):
    return b.useful(r, who)


# ------------------------------------------------------------------ tables
BELT_DATA = OUT / "data/belt_speed_bench.csv.gz"


def t_belt():
    """Belt-speed bench (3 vs 5 cm/s, nearest_job): the frozen question per cell, from the versioned
    extract (`docs/methods/geometry.md` §7–§8, `experiments/belt_speed_bench.py`)."""
    import belt_speed_bench as bs
    b = bs.Bench(bs.load_extract(BELT_DATA))
    who = "nearest_job"
    L = [r"\begin{tabular}{@{}lcccccccl@{}}", r"\toprule",
         r"cell & $G_3(50\,\mathrm{cm})$ & useful at 3 & $G_5(50\,\mathrm{cm})$ & useful at 5 & $C_5(10\,\mathrm{cm})$ & $C_5(16.7\,\mathrm{cm})$ & $C_3(10\,\mathrm{cm})$ & verdict at 5~cm/s \\",
         r" & [s/part] & & [s/part] & & $=2.00$~s & $=3.33$~s & $=3.33$~s (check) & (conservative) \\", r"\midrule"]
    names = {"benefit lost at this speed": "benefit lost at this speed", "H_cm": "10~cm keep the benefit ($H_{\\mathrm{cm}}$)",
             "H_s": "16.7~cm needed ($H_{\\mathrm{s}}$)", "H_neither": "neither ($H_{\\mathrm{neither}}$)", "undetermined": "undetermined"}
    for c in b.cells:
        g3, g5 = b.gain(c, 3.0, bs.L_REF, who), b.gain(c, 5.0, bs.L_REF, who)
        c10, c167, c3 = b.cost(c, 5.0, bs.L_10, who), b.cost(c, 5.0, bs.L_167, who), b.cost(c, 3.0, bs.L_10, who)
        v = b.verdict(c, who, 5.0, True)
        L.append(f"$d={c['spacing']:g}$, $v_y={c['vy']:g}$, $v_x={c['vx']:g}$ ($\\tcross={bs.tau_cross(c['spacing'], c['vy']):.1f}$~s) & "
                 f"{fmt(g3[0], g3[1])} & {'yes' if b.useful(c, 3.0, who) else 'no'} & {fmt(g5[0], g5[1])} & {'yes' if b.useful(c, 5.0, who) else 'no'} & "
                 f"{fmt(c10[0], c10[1])} & {fmt(c167[0], c167[1])} & {fmt(c3[0], c3[1])} & {names[v]} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_gains(b: Bench):
    """G(10 s) of the rule, 56 distinct cells: rows = v_x × load × arrivals, columns = τ_cross."""
    geo = {tau_cross(sp, vy): (sp, vy) for sp, vy in DISTINCT}
    L = [r"\begin{tabular}{@{}lll" + "c" * len(TC_ORDER) + "@{}}", r"\toprule",
         r"$v_x$ & load & arrivals & " + " & ".join(f"$\\tau_{{\\mathrm{{cross}}}}={tc:.1f}$~s" for tc in TC_ORDER) + r" \\",
         r" & & & " + " & ".join(f"{geo[tc][0]:g}~cm / {geo[tc][1]:g}~cm/s" for tc in TC_ORDER) + r" \\", r"\midrule"]
    for vx in VX:
        for ld in LOADS:
            for ar in ARRIVALS:
                cells = []
                for tc in TC_ORDER:
                    sp, vy = geo[tc]
                    r = reg(sp, vy, vx, ld, ar)
                    mu, h, _, _ = b.gain(r, GRID_S[-1], "nearest_job")
                    cells.append(fmt(mu, h, bold=useful(b, r)))
                L.append(f"{vx:g} & {ld} & {ARR[ar]} & " + " & ".join(cells) + r" \\")
        if vx == VX[0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_sufficient(b: Bench):
    L = [r"\begin{tabular}{@{}ccccl ccccc@{}}", r"\toprule",
         r"spacing & $v_y$ & $v_x$ & $\tau_{\mathrm{cross}}$ & load / arrivals & $G(10\,\mathrm{s})$ & share at 3.33~s & "
         r"cost of 3.33~s & knee & knee (cons.) \\",
         r"{[cm]} & [cm/s] & [cm/s] & [s] & & [s/part] & [\%] & [s/part] & [s] & [s] \\", r"\midrule"]
    for r in distinct_regimes():
        if not useful(b, r):
            continue
        g10 = b.gain(r, GRID_S[-1], "nearest_job")
        sh = 100 * b.gain(r, 10 / 3, "nearest_job")[0] / g10[0]
        c25 = b.paired(b.get(r, 2.5, "when_idle", "nearest_job"), b.get(r, GRID_S[-1], "when_idle", "nearest_job"))
        c33 = b.paired(b.get(r, 10 / 3, "when_idle", "nearest_job"), b.get(r, GRID_S[-1], "when_idle", "nearest_job"))
        k1, k2 = b.knee(r, "nearest_job"), b.knee(r, "nearest_job", conservative=True)
        L.append(f"{r['spacing']:g} & {r['vy']:g} & {r['vx']:g} & {r['tau_cross']:.1f} & {r['load']} / {ARR[r['arrivals']]} & "
                 f"{fmt(g10[0], g10[1])} & {sh:.0f} & "
                 f"{fmt(c33[0], c33[1], bold=c33[0] - c33[1] > 0)} & {k1:.2f} & "
                 + ("undet." if math.isnan(k2) else f"{k2:.2f}") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def predictor_points(b: Bench):
    pts = []
    for r in distinct_regimes():
        if not useful(b, r):
            continue
        k, tr = b.knee(r, "nearest_job"), b.trace(r, "nearest_job")
        if not math.isnan(k) and tr:
            pts.append((r, k, tr))
    return pts


def t_predictors(b: Bench):
    pts = predictor_points(b)
    knee = np.array([k for _, k, _ in pts]); tc = np.array([r["tau_cross"] for r, _, _ in pts])
    tpp = np.array([tr["t"] for _, _, tr in pts]); txy = np.array([tr["tx"] + tr["ty"] for _, _, tr in pts])
    fit = lambda x: float(np.sum(x * knee) / np.sum(x * x))
    rows = [(r"knee $=\tau_{\mathrm{cross}}$", spearman(tc, knee), loo_mae(tc, knee, MODELS["knee = τ_cross"]), "---"),
            (r"knee $=a\,\tau_{\mathrm{cross}}$", spearman(tc, knee), loo_mae(tc, knee, MODELS["knee = a·τ_cross"]), f"$a={fit(tc):.2f}$"),
            (r"knee $=b\,\tau_{\mathrm{pp}}$ (mean planned duration of the moves started)", spearman(tpp, knee), loo_mae(tpp, knee, MODELS["knee = b·τ_pp"]), f"$b={fit(tpp):.2f}$"),
            (r"knee $=b'\,(t_x+t_y)$", spearman(txy, knee), loo_mae(txy, knee, MODELS["knee = b·τ_pp"]), f"$b'={fit(txy):.2f}$"),
            (r"knee $=$ constant", math.nan, loo_mae(tc, knee, MODELS["knee = constant"]), f"${knee.mean():.2f}$~s")]
    best = min(r[2] for r in rows)
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"predictor & Spearman $\rho$ with the knee & leave-one-out MAE [s] & fitted coefficient \\", r"\midrule"]
    for name, sp, mae, coef in rows:
        L.append(f"{name} & " + ("---" if math.isnan(sp) else f"${sp:+.2f}$") + " & "
                 + (f"\\textbf{{{mae:.2f}}}" if mae == best else f"{mae:.2f}") + f" & {coef} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L), dict(n=len(pts), knee=(knee.min(), knee.max()), tpp=(tpp.min(), tpp.max()),
                              ratio=(knee / tpp).min(), ratio_max=(knee / tpp).max(), b=fit(tpp), const=knee.mean())


def t_entry(b: Bench):
    from geometry_bench import CHECK_S, ENTRY, ENTRY_ORIG
    L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"load & arrivals & $\Delta$ `none' level (35 $-$ 10~cm) & $\Delta G$ at 1~s & $\Delta G$ at 2~s & $\Delta G$ at 3.33~s \\",
         r" & & [s/part] & [s/part] & [s/part] & [s/part] \\", r"\midrule"]
    for ld in LOADS:
        for ar in ARRIVALS:
            r = reg(12.0, 10.0, 10.0, ld, ar)
            n10, n35 = b.get(r, 0.0, "none", "nearest_job", "entry", ENTRY_ORIG), b.get(r, 0.0, "none", "nearest_job")
            dn = b.paired(n35, n10)
            cells = [fmt(dn[0], dn[1])]
            for s in (1.0, 2.0, 10 / 3):
                a10, w10 = b.get(r, s, "none", "nearest_job", "entry", ENTRY_ORIG), b.get(r, s, "when_idle", "nearest_job", "entry", ENTRY_ORIG)
                a35, w35 = b.get(r, s, "none", "nearest_job"), b.get(r, s, "when_idle", "nearest_job")
                seeds = sorted(set(a10) & set(w10) & set(a35) & set(w35))
                d = [(a35[k]["cost"] - w35[k]["cost"]) - (a10[k]["cost"] - w10[k]["cost"]) for k in seeds]
                mu, h = float(np.mean(d)), float(1.96 * np.std(d, ddof=1) / math.sqrt(len(d)))
                cells.append(fmt(mu, h, bold=abs(mu) > h))
            L.append(f"{ld} & {ARR[ar]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_net(b: Bench):
    L = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
         r"$v_x$ & load & arrivals & useful cells & $G(10\,\mathrm{s})$ range & knee range & seeing, `none'(0) $-$ `none'(10~s) \\",
         r"{[cm/s]} & & & (of 9) & [s/part] & [s] & [s/part], range \\", r"\midrule"]
    for vx in VX:
        for ld in LOADS:
            for ar in ARRIVALS:
                regs = [reg(sp, vy, vx, ld, ar) for sp in SPACINGS for vy in VY]
                g = [b.gain(r, GRID_S[-1], "net")[0] for r in regs]
                u = [r for r in regs if b.useful(r, "net")]
                k = [b.knee(r, "net") for r in u]
                see = [b.paired(b.get(r, 0.0, "none", "net"), b.get(r, GRID_S[-1], "none", "net")) for r in regs]
                nsig = sum(1 for s in see if s[0] - s[1] > 0)
                L.append(f"{vx:g} & {ld} & {ARR[ar]} & {len(u)} & {min(g):+.2f} to {max(g):+.2f} & "
                         + (f"{min(k):.1f}--{max(k):.1f}" if k else "---") + " & "
                         f"{min(s[0] for s in see):+.2f} to {max(s[0] for s in see):+.2f} ({nsig} sig.) \\\\")
        if vx == VX[0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def _color(plt, tc):
    return plt.get_cmap("viridis")(TC_ORDER.index(tc) / (len(TC_ORDER) - 1))


def fig_curves(b: Bench, path: Path, arrivals: str):
    plt = _rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for i, vx in enumerate(VX):
        for j, ld in enumerate(LOADS):
            ax = axes[i][j]
            for sp, vy in DISTINCT:
                r = reg(sp, vy, vx, ld, arrivals)
                g = np.array([b.gain(r, s, "nearest_job")[0] for s in GRID_S])
                h = np.array([b.gain(r, s, "nearest_job")[1] for s in GRID_S])
                tc = r["tau_cross"]
                u = useful(b, r)
                ax.plot(GRID_S, g, "-" if u else "--", color=_color(plt, tc), marker="o" if u else None, ms=2.5,
                        lw=1.3 if u else 0.8, alpha=1.0 if u else 0.55,
                        label=f"$\\tau_{{\\mathrm{{cross}}}}$ {tc:.1f} s ({sp:g} cm, {vy:g} cm/s)")
                if u:
                    ax.fill_between(GRID_S, g - h, g + h, color=_color(plt, tc), alpha=0.07, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(10 / 3, color="grey", lw=0.6, ls=":")
            ax.set_title(f"$v_x={vx:g}$ cm/s, {ld} load ($\\rho_p$ {rho(ld):.2f})")
            if i == 1:
                ax.set_xlabel("anticipation $\\tau_v=\\ell_v/v_{\\mathrm{belt}}$ [s]")
            if j == 0:
                ax.set_ylabel("gain of pre-positioning $G$ [s per part]")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_normalized(b: Bench, path: Path):
    plt = _rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for r in distinct_regimes():
        if not useful(b, r):
            continue
        g10 = b.gain(r, GRID_S[-1], "nearest_job")[0]
        sh = np.array([b.gain(r, s, "nearest_job")[0] for s in GRID_S]) / g10
        tc = r["tau_cross"]
        mk = {5.0: "s", 10.0: "o"}[r["vx"]]
        axes[0].plot(np.array(GRID_S) / tc, sh, "-", color=_color(plt, tc), marker=mk, ms=2.5, lw=0.9, alpha=0.85)
        axes[1].plot(GRID_S, sh, "-", color=_color(plt, tc), marker=mk, ms=2.5, lw=0.9, alpha=0.85)
    for ax in axes:
        ax.axhline(0.9, color="grey", lw=0.6, ls="--")
        ax.set_ylabel("share of $G(10\\,\\mathrm{s})$")
        ax.set_ylim(-0.1, 1.35)
    axes[0].axvline(1.0, color="grey", lw=0.6, ls="--")
    axes[0].set_xlabel("$\\tau_v/\\tau_{\\mathrm{cross}}$")
    axes[0].set_xlim(0, 6)
    axes[0].set_title("normalized by the crossing time")
    axes[1].axvline(10 / 3, color="grey", lw=0.6, ls=":")
    axes[1].set_xlabel("anticipation $\\tau_v$ [s]")
    axes[1].set_title("in seconds")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=_color(plt, tc), lw=1.5, label=f"$\\tau_{{\\mathrm{{cross}}}}$ {tc:.1f} s") for tc in TC_ORDER]
    hs += [Line2D([], [], color="k", marker="s", ls="", ms=3, label="$v_x=5$"),
           Line2D([], [], color="k", marker="o", ls="", ms=3, label="$v_x=10$")]
    axes[1].legend(handles=hs, loc="lower right", ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_knee(b: Bench, path: Path, b_fit: float, const: float):
    plt = _rc()
    pts = predictor_points(b)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, key, lab in ((axes[0], "tau_cross", "$\\tau_{\\mathrm{cross}}$ [s]"),
                         (axes[1], "t", "mean duration of the pre-positioning moves $\\tau_{\\mathrm{pp}}$ [s]")):
        for r, k, tr in pts:
            x = r["tau_cross"] if key == "tau_cross" else tr["t"]
            ax.plot(x, k, marker={5.0: "s", 10.0: "o"}[r["vx"]], color={"nominal": "#1f77b4", "high": "#d62728"}[r["load"]],
                    mfc="none" if r["arrivals"] == "burst" else None, ms=5, ls="none", alpha=0.9)
        ax.set_xlabel(lab)
        ax.set_ylabel("knee: 90 % of $G(10\\,\\mathrm{s})$ [s]")
        ax.set_ylim(0, 8)
    axes[0].plot([0, 7.5], [0, 7.5], color="grey", lw=0.6, ls="--", label="knee $=\\tau_{\\mathrm{cross}}$")
    axes[0].set_xlim(0, 7.5)
    axes[1].plot([0, 3], [0, b_fit * 3], color="grey", lw=0.6, ls="--", label=f"knee $={b_fit:.2f}\\,\\tau_{{\\mathrm{{pp}}}}$")
    axes[1].axhline(const, color="grey", lw=0.6, ls=":", label=f"constant {const:.1f} s")
    axes[1].set_xlim(0, 3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="#1f77b4", marker="o", ls="", ms=4, label="nominal load"),
          Line2D([], [], color="#d62728", marker="o", ls="", ms=4, label="high load"),
          Line2D([], [], color="k", marker="s", ls="", ms=4, label="$v_x=5$ (square), 10 (circle)"),
          Line2D([], [], color="k", marker="o", ls="", ms=4, mfc="none", label="hollow: bursts")]
    axes[0].legend(handles=hs + axes[0].get_legend_handles_labels()[0], loc="upper left", frameon=False, fontsize=6.5)
    axes[1].legend(loc="upper left", frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def fig_belt(path: Path):
    """Belt-speed bench: G_v(ℓ) against the visible length in cm, belts at 3 and 5 cm/s, four cells."""
    import belt_speed_bench as bs
    b = bs.Bench(bs.load_extract(BELT_DATA))
    plt = _rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for ax, c in zip(axes.ravel(), b.cells):
        for vb, col, lab in ((3.0, "#1f77b4", "belt 3 cm/s"), (5.0, "#d62728", "belt 5 cm/s")):
            xs, ys, hs = [], [], []
            for lv in bs.GRID_CM:
                mu, h, _, _ = b.gain(c, vb, lv, "nearest_job")
                xs.append(lv); ys.append(mu); hs.append(h)
            ys, hs = np.array(ys), np.array(hs)
            ax.plot(xs, ys, "-o", color=col, ms=2.5, lw=1.0, label=lab)
            ax.fill_between(xs, ys - hs, ys + hs, color=col, alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        for x in (10.0, 50 / 3):
            ax.axvline(x, color="k", lw=0.5, ls=":")
        ax.set_title(f"$d={c['spacing']:g}$ cm, $v_y={c['vy']:g}$, $v_x={c['vx']:g}$ cm/s ($\\tau_{{cross}}={bs.tau_cross(c['spacing'], c['vy']):.1f}$ s)")
        ax.set_ylabel("$G_v(\\ell)$ [s per part]")
    for ax in axes[1]:
        ax.set_xlabel("visible length $\\ell$ [cm]")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    b = load(DATA)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    pred_tex, info = t_predictors(b)
    tables = {"bench_gains": t_gains(b), "bench_sufficient": t_sufficient(b), "bench_predictors": pred_tex,
              "bench_entry": t_entry(b), "bench_net": t_net(b), "bench_belt": t_belt()}
    for name, tex in tables.items():
        src = "data/belt_speed_bench.csv.gz" if name == "bench_belt" else "data/geometry_bench.csv.gz"
        (OUT / "tables" / f"{name}.tex").write_text(
            f"% generated by experiments/paper2_bench_figures.py from {src}\n" + tex + "\n")
    fig_curves(b, OUT / "figures/bench_curves_poisson.pdf", "poisson")
    fig_curves(b, OUT / "figures/bench_curves_burst.pdf", "burst")
    fig_normalized(b, OUT / "figures/bench_normalized.pdf")
    fig_knee(b, OUT / "figures/bench_knee.pdf", info["b"], info["const"])
    fig_belt(OUT / "figures/bench_belt.pdf")
    # headline numbers
    U = [r for r in distinct_regimes() if useful(b, r)]
    print(f"useful cells (rule): {len(U)} of {len(distinct_regimes())}")
    for s in COST_S:
        c = [b.paired(b.get(r, s, "when_idle", "nearest_job"), b.get(r, GRID_S[-1], "when_idle", "nearest_job")) for r in U]
        print(f"cost of {s:.2f} s vs 10 s: median {np.median([x[0] for x in c]):+.2f}, max {max(x[0] for x in c):+.2f}, "
              f"significant {sum(x[0] - x[1] > 0 for x in c)}")
    kp = [b.knee(r, "nearest_job") for r in U]; kc = [b.knee(r, "nearest_job", True) for r in U]
    print("knee point", min(kp), max(kp), "conservative", np.nanmin(kc), np.nanmax(kc), "undetermined", sum(math.isnan(x) for x in kc))
    print("predictors:", info)
    print("written:", sorted(tables), [p.name for p in (OUT / "figures").glob("bench_*.pdf")])


if __name__ == "__main__":
    main()
