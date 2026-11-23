"""Figures and tables of Paper 2 from the phase 4 bench (``docs/papers/paper2``).

Reads ``results/experiments/<tag>/bench.json`` written by ``experiments/rl2_eval.py``
(the confirmatory run on the reserved seeds, 0.7.13) and writes, so that every number
of the paper comes from one computation:

* ``docs/papers/paper2/figures/coverage_curve.pdf``: the gain of pre-positioning at
  each visible length (``when_idle`` vs ``none`` at the same length, paired), against
  the seconds of anticipation ``visible_length / belt_speed``, one panel per regime,
  with the head's travel time across the conveyors marked;
* ``docs/papers/paper2/figures/contrasts.pdf``: the treatment contrasts of the network
  (B vs A, C vs B, D vs B) on the eight regimes;
* ``docs/papers/paper2/tables/*.tex``: the tables included by ``paper2.tex``.

Paired Δ = mean over seeds of ``cost_x / cost_y - 1`` (penalized flow time), with a
normal 95 % confidence interval; negative = the first configuration is better.

    python experiments/paper2_figures.py --tag phase4
"""

from __future__ import annotations

import argparse
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


class Bench:
    def __init__(self, path: Path):
        raw = json.load(open(path))
        self.args = raw["args"]
        self.seeds = raw["seeds"]
        self.info = raw["info"]
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
         r"regime & belt [cm/s] & head [cm/s] & $\tau_{\mathrm{head}}$ [s] & entry zone [s] & "
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
         r"regime & policy & $\tau_{\mathrm{head}}$ & " +
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


def t_robustness(b: Bench):
    L = [r"\begin{tabular}{@{}lccc|cc@{}}", r"\toprule",
         r" & \multicolumn{3}{c|}{reference network} & \multicolumn{2}{c}{frozen vs reference} \\",
         r"regime & B vs A & C vs B & D vs B & in B & in C \\", r"\midrule"]
    for reg in REGIMES:
        cells = []
        for x, y in (("A", "B"), ("B", "C"), ("B", "D")):
            mu, h, n = b.delta(reg, "net_ref", y, "net_ref", x)
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        for t in ("B", "C"):
            mu, h, n = b.delta(reg, "net", t, "net_ref", t)
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ----------------------------------------------------------------- figures
def fig_coverage(b: Bench, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                         "axes.titlesize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.0), sharey="row")
    style = {"nearest_job": dict(color="0.35", marker="s", label=r"$\mathrm{nearest\_job}$"),
             "net": dict(color="tab:blue", marker="o", label="network")}
    for ax, reg in zip(axes.flat, REGIMES):
        v, th = b.meta[reg]
        for who, st in style.items():
            xs, ys, hs = [], [], []
            for l in LENGTHS:
                mu, h, _ = b.gain(reg, who, l)
                xs.append(l / v); ys.append(-mu); hs.append(h)
            ax.errorbar(xs, ys, yerr=hs, capsize=2, lw=1.2, ms=4, **st)
        ax.axvline(th, color="tab:red", ls="--", lw=1)
        ax.annotate(r"$\tau_{\mathrm{head}}$", (th, 0), xytext=(-3, 3), textcoords="offset points",
                    ha="right", va="bottom", color="tab:red", fontsize=8)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(regime_name(*reg) + f"\n$\\tau_{{\\mathrm{{head}}}}={th:.1f}$ s")
        ax.set_xlim(-0.1, max(10.0 / v, th) + 0.3)
        ax.set_xlabel("anticipation $\\ell_v / v_{\\mathrm{belt}}$ [s]")
        sec = ax.secondary_xaxis("top", functions=(lambda x, v=v: x * v, lambda x, v=v: x / v))
        sec.set_xticks([0, 2.5, 5, 7.5, 10] if reg[0] == "real" else [0, 5, 10])
        sec.set_xlabel("visible length $\\ell_v$ [cm]", fontsize=8)
        sec.tick_params(labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("reduction of penalized flow time [%]\n(when\\_idle vs none, same $\\ell_v$)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_contrasts(b: Bench, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)
    names = [regime_name(*r) for r in REGIMES]
    ys = np.arange(len(REGIMES))[::-1]
    for ax, (x, y, title) in zip(axes, (("A", "B", "B vs A: seeing, same actions"),
                                        ("B", "C", "C vs B: pre-positioning, same view"),
                                        ("B", "D", "D vs B: hidden upstream queue"))):
        mus, hs = [], []
        for reg in REGIMES:
            mu, h, _ = b.delta(reg, "net", y, "net", x)
            mus.append(mu); hs.append(h)
        ax.errorbar(mus, ys, xerr=hs, fmt="o", color="tab:blue", capsize=3, ms=4)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("paired $\\Delta$ of penalized flow time [%]")
        ax.set_xlim(-10, 3)
        ax.grid(axis="x", color="0.9")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="phase4")
    args = ap.parse_args()
    b = Bench(ROOT / "results/experiments" / args.tag / "bench.json")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    tables = {"regimes": t_regimes(b), "levels": t_levels(b), "contrasts": t_contrasts(b),
              "coverage_real": t_coverage(b, "real"), "coverage_heavy": t_coverage(b, "heavy"),
              "share_alt": t_share_alt(b), "robustness": t_robustness(b)}
    for name, tex in tables.items():
        (OUT / "tables" / f"{name}.tex").write_text(
            f"% generated by experiments/paper2_figures.py from results/experiments/{args.tag}/bench.json\n"
            + tex + "\n")
    fig_coverage(b, OUT / "figures/coverage_curve.pdf")
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
    print("written:", sorted(p.name for p in (OUT / "tables").iterdir()),
          sorted(p.name for p in (OUT / "figures").iterdir()))


if __name__ == "__main__":
    main()
