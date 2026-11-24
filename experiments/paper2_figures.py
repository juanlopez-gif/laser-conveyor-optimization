"""Figures and tables of Paper 2 from the phase 4 bench (``docs/papers/paper2``).

Reads the per-episode results of the confirmatory run (``experiments/rl2_eval.py`` on
the reserved seeds, 0.7.13) and writes, so that every number of the paper comes from one
computation:

* ``docs/papers/paper2/figures/coverage_curve.pdf``: the gain of pre-positioning at
  each visible length (``when_idle`` vs ``none`` at the same length, paired), against
  the seconds of anticipation ``visible_length / belt_speed``, one panel per regime,
  with the head's travel time across the conveyors marked;
* ``docs/papers/paper2/figures/contrasts.pdf``: the treatment contrasts of the network
  (B vs A, C vs B, D vs B) on the eight regimes;
* ``docs/papers/paper2/tables/*.tex``: the tables included by ``paper2.tex``.

Paired Δ = mean over seeds of ``cost_x / cost_y - 1`` (penalized flow time), with a
normal 95 % confidence interval; negative = the first configuration is better.

The per-episode rows used here (the ``info`` part of ``bench.json``) are versioned as
``docs/papers/paper2/data/phase4_info.csv`` so that the figures and tables can be rebuilt
without the local ``results/`` directory; ``--bench`` regenerates that extract from a
``bench.json`` first.

    python experiments/paper2_figures.py                      # from the versioned extract
    python experiments/paper2_figures.py --bench results/experiments/phase4/bench.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/papers/paper2"
LOADS = ("real", "heavy")
ARRIVALS = ("regular", "burst")
TRAVEL = ("nominal", "slow")
REGIMES = [(g, a, tr) for g in LOADS for a in ARRIVALS for tr in TRAVEL]
LENGTHS = (0.0, 1.0, 2.5, 5.0, 7.5, 10.0)
POL = {"nearest_job": r"\textsc{nj}", "nearest_head": r"\textsc{nh}",
       "net": "network", "net_ref": "reference network"}
ARR = {"regular": "Poisson", "burst": "bursts"}
TRV = {"nominal": "nominal", "slow": "slow"}


def ci(d):
    d = np.asarray(d, dtype=float)
    return 100 * d.mean(), 100 * 1.96 * d.std(ddof=1) / math.sqrt(len(d))


def fmt(mu, half, n=None, bold=False):
    s = f"${mu:+.2f} \\pm {half:.2f}$"
    if bold:
        s = "{\\boldmath" + s + "}"
    if n is not None:
        s += f" ({n})"
    return s


def regime_name(g, a, tr):
    return f"{g} / {ARR[a]} / {TRV[tr]} head"


FIELDS = ("load", "arrivals", "travel", "belt_speed", "head_span_s", "seed", "t", "who",
          "cost", "flow", "blocks", "blocked", "parts")
NUM = {"belt_speed": float, "head_span_s": float, "seed": int, "cost": float, "flow": float,
       "blocks": int, "blocked": float, "parts": int}


def extract(bench_json: Path, csv_path: Path) -> None:
    """Write the per-episode rows of the information bench to the versioned extract."""
    raw = json.load(open(bench_json))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in raw["info"]:
            w.writerow({k: r[k] for k in FIELDS})


class Bench:
    def __init__(self, csv_path: Path):
        with open(csv_path, newline="") as f:
            self.info = [{k: (NUM[k](v) if k in NUM else v) for k, v in row.items()}
                         for row in csv.DictReader(f)]
        self.seeds = sorted({r["seed"] for r in self.info})
        self.by = {}
        for r in self.info:
            self.by.setdefault((r["load"], r["arrivals"], r["travel"], r["who"], r["t"]), {})[r["seed"]] = r
        self.meta = {}
        for r in self.info:
            self.meta[(r["load"], r["arrivals"], r["travel"])] = (r["belt_speed"], r["head_span_s"])

    def costs(self, reg, who, t):
        return {s: r["cost"] for s, r in self.by[(*reg, who, t)].items()}

    def rows(self, reg, who, t):
        return self.by[(*reg, who, t)]

    def delta(self, reg, who_x, t_x, who_y, t_y):
        """Paired Δ of (who_x, t_x) relative to (who_y, t_y): mean, half-CI, n better."""
        x, y = self.costs(reg, who_x, t_x), self.costs(reg, who_y, t_y)
        d = [x[s] / y[s] - 1 for s in self.seeds]
        n = sum(1 for s in self.seeds if x[s] < y[s] - 1e-12)
        return (*ci(d), n)

    def gain(self, reg, who, lv):
        """Pre-positioning gain at visible length lv: when_idle vs none, same length."""
        return self.delta(reg, who, f"L{lv:g}/when_idle", who, f"L{lv:g}/none")

    def level(self, reg, who, t):
        """Δ vs nearest_job in B."""
        return self.delta(reg, who, t, "nearest_job", "B")

    def gain_loss(self, reg, who, lv):
        """Pre-positioning benefit lost with lv against 10 cm: per realization,
        gain(lv) - gain(10) in points (positive = the shorter view retains less)."""
        def g(l):
            return {s: self.costs(reg, who, f"L{l:g}/when_idle")[s]
                    / self.costs(reg, who, f"L{l:g}/none")[s] - 1 for s in self.seeds}
        a, c = g(lv), g(10.0)
        return ci([a[s] - c[s] for s in self.seeds])

    def net_diff(self, reg, t_x, t_y):
        """Frozen minus reference network, paired: per realization Δ_net(X vs Y) −
        Δ_ref(X vs Y), in points (negative = the frozen network obtains more from X)."""
        d = []
        for s in self.seeds:
            dn = self.costs(reg, "net", t_x)[s] / self.costs(reg, "net", t_y)[s] - 1
            dr = self.costs(reg, "net_ref", t_x)[s] / self.costs(reg, "net_ref", t_y)[s] - 1
            d.append(dn - dr)
        return ci(d)

    def share_alt(self, reg, who, lv):
        """Share of the whole-zone gain with the mean ``none`` level over the lengths as
        the baseline (the ``none`` levels estimate one quantity, since B vs A is null)."""
        nj = self.costs(reg, "nearest_job", "B")
        none = np.mean([[self.costs(reg, who, f"L{l:g}/none")[s] / nj[s] - 1 for s in self.seeds]
                        for l in LENGTHS], axis=0)
        wi = lambda l: np.array([self.costs(reg, who, f"L{l:g}/when_idle")[s] / nj[s] - 1 for s in self.seeds])
        full = (wi(10.0) - none).mean()
        return (wi(lv) - none).mean() / full if abs(full) > 1e-9 else float("nan")


# ------------------------------------------------------------------ tables
def t_regimes(b: Bench):
    L = [r"\begin{tabular}{@{}lccccrrrr@{}}", r"\toprule",
         r"regime & belt [cm/s] & head [cm/s] & $\tau_{\mathrm{cross}}$ [s] & entry zone [s] & "
         r"parts & \textsc{nj} in B [s/part] & blocking events & blocked [conv.\,s] \\", r"\midrule"]
    for reg in REGIMES:
        v, th = b.meta[reg]
        rows = b.rows(reg, "nearest_job", "B").values()
        parts = np.mean([r["parts"] for r in rows])
        cost = np.mean([r["cost"] for r in rows])
        blocks = np.mean([r["blocks"] for r in rows])
        blocked = np.mean([r["blocked"] for r in rows])
        head = (10.0 if reg[0] == "real" else 5.0) * (0.5 if reg[2] == "slow" else 1.0)
        L.append(f"{regime_name(*reg)} & {v:.0f} & {head:g} & {th:.1f} & {10.0 / v:.2f} & "
                 f"{parts:.0f} & {cost:.1f} & {blocks:.1f} & {blocked:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_levels(b: Bench):
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"regime & \textsc{nj} C & \textsc{nh} B & \textsc{nh} C & network A & network B & network C \\",
         r"\midrule"]
    for reg in REGIMES:
        cells = [fmt(*b.level(reg, w, t)[:2]) for w, t in
                 (("nearest_job", "C"), ("nearest_head", "B"), ("nearest_head", "C"),
                  ("net", "A"), ("net", "B"), ("net", "C"))]
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_contrasts(b: Bench):
    L = [r"\begin{tabular}{@{}lcccc|c@{}}", r"\toprule",
         r" & \multicolumn{4}{c|}{network} & \textsc{nj} \\",
         r"regime & B vs A & C vs B & D vs B & C vs A & C vs B \\", r"\midrule"]
    for reg in REGIMES:
        cells = []
        for x, y in (("A", "B"), ("B", "C"), ("B", "D"), ("A", "C")):
            mu, h, n = b.delta(reg, "net", y, "net", x)
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        mu, h, n = b.delta(reg, "nearest_job", "C", "nearest_job", "B")
        cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_coverage(b: Bench, load: str):
    regs = [r for r in REGIMES if r[0] == load]
    v = b.meta[regs[0]][0]
    lens = LENGTHS[1:]
    L = [r"\begin{tabular}{@{}llc" + "c" * len(lens) + "@{}}", r"\toprule",
         r"regime & policy & $\tau_{\mathrm{cross}}$ & " +
         " & ".join(f"{l:g} cm" for l in lens) + r" \\",
         r" & & [s] & " + " & ".join(f"{l / v:.2f} s" for l in lens) + r" \\", r"\midrule"]
    for reg in regs:
        th = b.meta[reg][1]
        for i, who in enumerate(("nearest_job", "net")):
            full = b.gain(reg, who, 10.0)[0]
            cells = []
            for l in lens:
                mu, h, _ = b.gain(reg, who, l)
                cell = f"${mu:+.2f} \\pm {h:.2f}$"
                if load == "real":            # shares are meaningless when the whole-zone gain is ~0
                    cell += f" ({100 * mu / full:.0f}\\,\\%)"
                cells.append(cell)
            name = regime_name(*reg) if i == 0 else ""
            thc = f"{th:.1f}" if i == 0 else ""
            L.append(f"{name} & {POL[who]} & {thc} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_share_alt(b: Bench):
    regs = [r for r in REGIMES if r[0] == "real"]
    lens = LENGTHS[1:-1]
    L = [r"\begin{tabular}{@{}ll" + "cc" * len(lens) + "@{}}", r"\toprule",
         r"regime & policy & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l:g} cm}}" for l in lens) + r" \\",
         r" & & " + " & ".join("same-$\\ell$ & pooled" for _ in lens) + r" \\", r"\midrule"]
    for reg in regs:
        for i, who in enumerate(("nearest_job", "net")):
            full = b.gain(reg, who, 10.0)[0]
            cells = []
            for l in lens:
                s1 = 100 * b.gain(reg, who, l)[0] / full
                s2 = 100 * b.share_alt(reg, who, l)
                cells.append(f"{s1:.0f} & {s2:.0f}")
            name = regime_name(*reg) if i == 0 else ""
            L.append(f"{name} & {POL[who]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_direct(b: Bench):
    """Two answers to "is l cm enough?": the cost of pre-positioning with l cm against
    with 10 cm (direct, paired) and the pre-positioning benefit lost (gain(l) - gain(10))."""
    lens = (5.0, 7.5)
    L = [r"\begin{tabular}{@{}ll" + "cc" * len(lens) + "@{}}", r"\toprule",
         r"regime & policy & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l:g} cm against 10 cm}}"
                                           for l in lens) + r" \\",
         r" & & " + " & ".join(r"cost [\%] & benefit lost [pts]" for _ in lens) + r" \\",
         r"\midrule"]
    for k, reg in enumerate(REGIMES):
        if k == 4:
            L.append(r"\midrule")
        for i, who in enumerate(("nearest_job", "net")):
            cells = []
            for l in lens:
                mu, h, n = b.delta(reg, who, f"L{l:g}/when_idle", who, "L10/when_idle")
                cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
                mu2, h2 = b.gain_loss(reg, who, l)
                cells.append(fmt(mu2, h2, bold=abs(mu2) > h2))
            name = regime_name(*reg) if i == 0 else ""
            L.append(f"{name} & {POL[who]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_robustness(b: Bench):
    L = [r"\begin{tabular}{@{}lccc|ccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c|}{reference network} & "
         r"\multicolumn{3}{c}{frozen $-$ reference, paired [pts]} \\",
         r"regime & B vs A & C vs B & D vs B & B vs A & C vs B & D vs B \\",
         r"\midrule"]
    for reg in REGIMES:
        cells = []
        for x, y in (("A", "B"), ("B", "C"), ("B", "D")):
            mu, h, n = b.delta(reg, "net_ref", y, "net_ref", x)
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        for x, y in (("A", "B"), ("B", "C"), ("B", "D")):
            mu, h = b.net_diff(reg, y, x)
            cells.append(fmt(mu, h, bold=abs(mu) > h))
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_robustness_levels(b: Bench):
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
         r"regime & in B & in C \\", r"\midrule"]
    for reg in REGIMES:
        cells = []
        for t in ("B", "C"):
            mu, h, n = b.delta(reg, "net", t, "net_ref", t)
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ----------------------------------------------------------------- figures
STYLE = {"nearest_job": dict(color="#555555", marker="s", label="nearest_job rule"),
         "net": dict(color="#1f77b4", marker="o", label="learned policy (network)")}


def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
        "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    })
    return plt


def _panel_title(reg):
    return f"{ARR[reg[1]]} · {TRV[reg[2]]} head"


def fig_coverage(b: Bench, path: Path, load: str):
    """One row of panels for one load: gain of pre-positioning vs anticipation."""
    plt = _rc()
    regs = [r for r in REGIMES if r[0] == load]
    v = b.meta[regs[0]][0]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.3), sharey=True)
    zone_s = 10.0 / v
    xmax = max([zone_s] + [b.meta[r][1] for r in regs if b.meta[r][1] <= 1.6 * zone_s]) * 1.08
    for ax, reg in zip(axes, regs):
        th = b.meta[reg][1]
        for who, st in STYLE.items():
            xs = np.array([l / v for l in LENGTHS])
            g = np.array([b.gain(reg, who, l)[:2] for l in LENGTHS])
            ax.fill_between(xs, -g[:, 0] - g[:, 1], -g[:, 0] + g[:, 1], color=st["color"], alpha=0.12, lw=0)
            ax.plot(xs, -g[:, 0], color=st["color"], marker=st["marker"], label=st["label"])
        zone = 10.0 / v
        label = f"$\\tau_{{\\mathrm{{cross}}}}$ = {th:.1f} s"
        if th <= xmax:
            ax.axvline(th, color="#c0392b", ls=(0, (4, 2)), lw=1.1)
            ax.text(th - 0.05 if th > zone else th + 0.05, 0.03, label, transform=ax.get_xaxis_transform(),
                    ha="right" if th > zone else "left", va="bottom", color="#c0392b", fontsize=8.5, rotation=90)
            ax.axvspan(zone, xmax, color="0.5", alpha=0.07, lw=0)
        else:                                   # the head's travel time is off the axis
            ax.annotate(label + " $\\rightarrow$", (1.0, 0.97), xycoords="axes fraction", ha="right",
                        va="top", color="#c0392b", fontsize=8.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(_panel_title(reg))
        ax.set_xlim(-0.05, xmax)
        ticks = [0, 2.5, 5, 7.5, 10]
        ax.set_xticks([l / v for l in ticks])
        ax.set_xticklabels([f"{l:g}\n{l / v:.1f}" for l in ticks])
        ax.set_xlabel("visible length [cm]\nanticipation [s]")
    axes[0].set_ylabel("gain of pre-positioning [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_contrasts(b: Bench, path: Path):
    plt = _rc()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    names = [f"{r[0]} · {ARR[r[1]]} · {TRV[r[2]]} head" for r in REGIMES]
    ys = np.arange(len(REGIMES))[::-1]
    panels = (("A", "B", "B vs A\nwider view, same actions"),
              ("B", "C", "C vs B\npre-positioning on the same view"),
              ("B", "D", "D vs B\nhidden upstream queue"))
    for ax, (x, y, title) in zip(axes, panels):
        for yy, reg in zip(ys, REGIMES):
            mu, h, _ = b.delta(reg, "net", y, "net", x)
            sig = abs(mu) > h
            ax.errorbar(mu, yy, xerr=h, fmt="o", color="#1f77b4", mfc="#1f77b4" if sig else "white",
                        mec="#1f77b4", capsize=3, elinewidth=1.2)
        ax.axvline(0, color="0.4", lw=0.9)
        ax.axhline(3.5, color="0.8", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("paired $\\Delta$ of penalized flow time [%]")
        ax.set_xlim(-10.5, 3.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", type=Path, default=None,
                    help="bench.json of rl2_eval.py; regenerates data/phase4_info.csv first")
    args = ap.parse_args()
    data = OUT / "data/phase4_info.csv"
    if args.bench is not None:
        extract(args.bench, data)
        print(f"extract written: {data}")
    b = Bench(data)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tables = {"regimes": t_regimes(b), "levels": t_levels(b), "contrasts": t_contrasts(b),
              "coverage_real": t_coverage(b, "real"), "coverage_heavy": t_coverage(b, "heavy"),
              "share_alt": t_share_alt(b), "robustness": t_robustness(b),
              "robustness_levels": t_robustness_levels(b), "direct": t_direct(b)}
    for name, tex in tables.items():
        (OUT / "tables" / f"{name}.tex").write_text(
            "% generated by experiments/paper2_figures.py from data/phase4_info.csv\n"
            + tex + "\n")
    fig_coverage(b, OUT / "figures/coverage_real.pdf", "real")
    fig_coverage(b, OUT / "figures/coverage_heavy.pdf", "heavy")
    fig_contrasts(b, OUT / "figures/contrasts.pdf")
    # headline numbers for the text
    for reg in [r for r in REGIMES if r[0] == "real"]:
        v, th = b.meta[reg]
        for who in ("nearest_job", "net"):
            full = b.gain(reg, who, 10.0)[0]
            shares = [100 * b.gain(reg, who, l)[0] / full for l in LENGTHS[1:]]
            alts = [100 * b.share_alt(reg, who, l) for l in LENGTHS[1:]]
            print(f"{regime_name(*reg):34s} {who:12s} full {full:+.2f}  share "
                  + " ".join(f"{s:4.0f}" for s in shares) + "  pooled " + " ".join(f"{s:4.0f}" for s in alts))
    # direct cost of a shorter view (when_idle at l vs when_idle at 10 cm) and the
    # none level at l vs at 10 cm, which is what separates the two readings
    for reg in [r for r in REGIMES if r[0] == "real"]:
        for who in ("nearest_job", "net"):
            for l in (5.0, 7.5):
                mu, h, n = b.delta(reg, who, f"L{l:g}/when_idle", who, "L10/when_idle")
                mu0, h0, _ = b.delta(reg, who, f"L{l:g}/none", who, "L10/none")
                mu2, h2 = b.gain_loss(reg, who, l)
                print(f"{regime_name(*reg):34s} {who:12s} {l:4g} vs 10 cm: cost {mu:+.2f} ±{h:.2f} "
                      f"({n}/30)  none-level {mu0:+.2f} ±{h0:.2f}  benefit lost {mu2:+.2f} ±{h2:.2f}")
    # paired frozen − reference differences over the 24 contrast cells
    diffs = [(x, y, reg, *b.net_diff(reg, y, x)) for x, y in (("A", "B"), ("B", "C"), ("B", "D"))
             for reg in REGIMES]
    sig = [(f"{y} vs {x}", regime_name(*reg), mu, h) for x, y, reg, mu, h in diffs if abs(mu) > h]
    print(f"paired net-ref: |mean|<1 pt in {sum(abs(d[3]) < 1 for d in diffs)}/24, "
          f"<0.5 in {sum(abs(d[3]) < 0.5 for d in diffs)}/24, max {max(abs(d[3]) for d in diffs):.2f}; "
          f"CI excludes zero in {len(sig)}: {sig}")
    print("written:", sorted(p.name for p in (OUT / "tables").iterdir()),
          sorted(p.name for p in (OUT / "figures").iterdir()))


if __name__ == "__main__":
    main()
