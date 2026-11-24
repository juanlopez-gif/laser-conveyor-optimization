"""Tables and figures of Paper 1 (docs/papers/paper1) from versioned data.

Inputs (all distributed with the source):

* ``docs/papers/paper1/data/phase4_search.csv``: per-episode results of the phase-4
  confirmatory run for the Paper-1 policies (known-parts search, sampled-future teacher,
  network, and the rules) in treatments B and C, seeds 7000–7029;
* ``docs/papers/paper1/data/phase4_latency.csv.gz``: per-decision latency of every policy
  on the decisions of the network's episodes (one thread, loaded machine);
* ``docs/papers/paper1/data/teacher_variants_*.json`` and ``teacher_futures_*.json``:
  teacher variants over a trained network on development seeds (rl2.md §6.4, §8.9);
* ``models/*_training.json``: the expert-iteration logs (validation per iteration).

``--bench results/experiments/phase4/bench.json`` regenerates the two phase-4 extracts
and copies the variant files from ``results/experiments/``.

    python experiments/paper1_figures.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/papers/paper1"
DATA = OUT / "data"
LOADS, ARRIVALS, TRAVEL = ("real", "heavy"), ("regular", "burst"), ("nominal", "slow")
REGIMES = [(g, a, tr) for g in LOADS for a in ARRIVALS for tr in TRAVEL]
ARR = {"regular": "Poisson", "burst": "bursts"}
POL = {"nearest_job": r"\textsc{nj}", "nearest_head": r"\textsc{nh}", "search": "known-parts search",
       "teacher": "sampled-future search", "net": "network"}
FIELDS = ("load", "arrivals", "travel", "belt_speed", "head_span_s", "seed", "t", "who", "cost", "flow",
          "blocks", "blocked", "parts", "secs", "lookaheads")
NUM = {"belt_speed": float, "head_span_s": float, "seed": int, "cost": float, "flow": float, "blocks": int,
       "blocked": float, "parts": int, "secs": float, "lookaheads": int}
VARIANT_FILES = {"teacher_variants_s": "rl2_teacher_variants_s", "teacher_variants_s2": "rl2_teacher_variants_s2",
                 "teacher_variants_s3": "rl2_teacher_variants_s3", "teacher_futures_s_b": "teacher_futures_s_b",
                 "teacher_futures_s_b_strong": "teacher_futures_s_b_strong"}


def regime_name(g, a, tr):
    return f"{g} / {ARR[a]} / {tr} head"


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


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else math.nan


# ------------------------------------------------------------------ extracts
def extract(bench_json: Path) -> None:
    raw = json.load(open(bench_json))
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in raw["search"]:
        rows.append({k: r.get(k) for k in FIELDS})
    for r in raw["info"]:
        if r["who"] in ("nearest_job", "nearest_head") and r["t"] in ("B", "C"):
            rows.append({k: r.get(k) for k in FIELDS})
    with open(DATA / "phase4_search.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    with gzip.open(DATA / "phase4_latency.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["load", "arrivals", "travel", "seed", "t", "policy", "ms"])
        for r in raw["search"]:
            if r.get("probe"):
                for name, xs in r["latency_ms"].items():
                    for x in xs:
                        w.writerow([r["load"], r["arrivals"], r["travel"], r["seed"], r["t"], name, x])
    for dst, src in VARIANT_FILES.items():
        p = ROOT / "results/experiments" / src / "variants.json"
        if p.exists():
            d = json.load(open(p))
            d.pop("episodes", None)
            json.dump(d, open(DATA / f"{dst}.json", "w"), indent=0)
    print("extracts written to", DATA)


# ------------------------------------------------------------------ data
class Bench:
    def __init__(self):
        with open(DATA / "phase4_search.csv", newline="") as f:
            self.rows = [{k: (NUM[k](v) if k in NUM and v not in ("", None) else v) for k, v in r.items()}
                         for r in csv.DictReader(f)]
        self.by = {}
        for r in self.rows:
            self.by.setdefault((r["load"], r["arrivals"], r["travel"], r["t"], r["who"]), {})[r["seed"]] = r
        self.seeds = sorted({r["seed"] for r in self.rows})
        self.lat = {}
        with gzip.open(DATA / "phase4_latency.csv.gz", "rt", newline="") as f:
            for r in csv.DictReader(f):
                self.lat.setdefault((r["load"], r["arrivals"], r["travel"], r["t"], r["policy"]), []).append(float(r["ms"]))

    def costs(self, reg, t, who):
        return {s: r["cost"] for s, r in self.by.get((*reg, t, who), {}).items()}

    def delta(self, reg, t, who_x, who_y, field="cost"):
        """Paired Δ of who_x vs who_y in treatment t [%]: mean, half-CI, n with x < y."""
        x = {s: r[field] for s, r in self.by[(*reg, t, who_x)].items()}
        y = {s: r[field] for s, r in self.by[(*reg, t, who_y)].items()}
        seeds = sorted(set(x) & set(y))
        d = [x[s] / y[s] - 1 for s in seeds]
        return (*ci(d), sum(1 for s in seeds if x[s] < y[s] - 1e-12))

    def level(self, reg, t, who, field):
        return float(np.mean([r[field] for r in self.by[(*reg, t, who)].values()]))

    def latency(self, reg, t, who):
        return self.lat.get((*reg, t, who), [])


# ------------------------------------------------------------------ tables
def t_regimes(b: Bench):
    L = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"regime & belt [cm/s] & head [cm/s] & $\tau_{\mathrm{cross}}$ [s] & parts & "
         r"$J$ (\textsc{nj}, B) [s/part] & flow [s] & blocking events & blocked [belt-s] \\", r"\midrule"]
    for reg in REGIMES:
        rows = list(b.by[(*reg, "B", "nearest_job")].values())
        belt = rows[0]["belt_speed"]; th = rows[0]["head_span_s"]
        head = 24 / th
        L.append(f"{regime_name(*reg)} & {belt:g} & {head:g} & {th:.1f} & {np.mean([r['parts'] for r in rows]):.0f} & "
                 f"{np.mean([r['cost'] for r in rows]):.1f} & {np.mean([r['flow'] for r in rows]):.1f} & "
                 f"{np.mean([r['blocks'] for r in rows]):.1f} & {np.mean([r['blocked'] for r in rows]):.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_confirm(b: Bench, t: str):
    L = [r"\begin{tabular}{@{}lccc|cc|ccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c|}{paired $\Delta$ vs \textsc{nj} [\%]} & \multicolumn{2}{c|}{contrasts [\%]} & "
         r"\multicolumn{3}{c}{latency p50 / p95 [ms]} \\",
         r"regime & known-parts & sampled-future & network & search vs net & teacher vs net & search & network & teacher \\",
         r"\midrule"]
    for reg in REGIMES:
        cells = []
        for who in ("search", "teacher", "net"):
            mu, h, n = b.delta(reg, t, who, "nearest_job")
            cells.append(fmt(mu, h, bold=abs(mu) > h))
        for who in ("search", "teacher"):
            mu, h, n = b.delta(reg, t, who, "net")
            cells.append(fmt(mu, h, f"{n}/30", bold=abs(mu) > h))
        for who in ("search", "net", "teacher"):
            xs = b.latency(reg, t, who)
            cells.append(f"{pct(xs, .5):.1f} / {pct(xs, .95):.1f}" if who != "teacher"
                         else f"{pct(xs, .5) / 1000:.2f} / {pct(xs, .95) / 1000:.2f} s")
        L.append(f"{regime_name(*reg)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_latency(b: Bench):
    L = [r"\begin{tabular}{@{}lcccc|cccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c|}{treatment B [ms]} & \multicolumn{4}{c}{treatment C [ms]} \\",
         r"policy & p50 & p95 & p99 & max & p50 & p95 & p99 & max \\", r"\midrule"]
    for who in ("nearest_job", "nearest_head", "search", "net", "teacher"):
        cells = []
        for t in ("B", "C"):
            xs = [x for reg in REGIMES for x in b.latency(reg, t, who)]
            cells += [f"{pct(xs, .5):.2f}", f"{pct(xs, .95):.2f}", f"{pct(xs, .99):.2f}", f"{max(xs):.1f}"]
        L.append(f"{POL[who]} & " + " & ".join(cells) + r" \\")
    nB = sum(len(b.latency(reg, "B", "net")) for reg in REGIMES)
    nC = sum(len(b.latency(reg, "C", "net")) for reg in REGIMES)
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L), nB, nC


def load_training(tag):
    return json.load(open(ROOT / "models" / f"{tag}_training.json"))


def t_ladder():
    L = [r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         r"run & size & params & episodes it.\ 0 / later & best it. & net real & net heavy & net general & score & search over net (score) \\",
         r"\midrule"]
    for tag, size, params in (("rl2_gen_s_big", "S", "0.88 M"), ("rl2_gen_m", "M", "6.7 M"), ("rl2_gen_l", "L", "39 M")):
        d = load_training(tag)
        its = d["iterations"]
        best = max((it for it in its if it["accepted"]), key=lambda it: -it["val"]["net/score"])
        v = best["val"]
        cells = [fmt(100 * v[f"net/{g}"]["delta"], 100 * v[f"net/{g}"]["ci"]) for g in ("real", "heavy", "general")]
        L.append(f"\\texttt{{{tag.replace('_', chr(92) + '_')}}} & {size} & {params} & {its[0]['episodes']} / "
                 f"{its[1]['episodes'] if len(its) > 1 else '---'} & {best['iteration']} of {len(its) - 1} & "
                 + " & ".join(cells) + f" & {100 * v['net/score']:+.2f} & {100 * v['teacher/score']:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_sf_runs():
    runs = (("rl2_gen_s_sf", "sampled-future teacher, from \\texttt{rl2\\_gen\\_s\\_big}; `none'", ("real", "heavy", "general")),
            ("rl2_gen_s_d", "+ design mode, corrected tokens, observation levels; from \\texttt{s\\_sf}", ("real", "heavy", "general")),
            ("rl2_gen_s_b", "+ bursts and slow heads in the mix; from \\texttt{s\\_d}", ("real", "heavy", "general", "real_burst")))
    L = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"run & it. & net real & net heavy & net general & net real/bursts & score & teacher score \\", r"\midrule"]
    for tag, desc, groups in runs:
        d = load_training(tag)
        L.append(r"\multicolumn{8}{@{}l}{\texttt{" + tag.replace("_", r"\_") + "}: " + desc + r"} \\")
        for it in d["iterations"]:
            v = it["val"]
            cells = []
            for g in ("real", "heavy", "general", "real_burst"):
                cells.append(fmt(100 * v[f"net/{g}"]["delta"], 100 * v[f"net/{g}"]["ci"]) if f"net/{g}" in v else "---")
            mark = r"\textbf{" + str(it["iteration"]) + "}" if it["accepted"] else str(it["iteration"])
            L.append(f" & {mark} & " + " & ".join(cells) + f" & {100 * v['net/score']:+.2f} & {100 * v['teacher/score']:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def variant_label(v: str) -> str:
    if v == "net alone":
        return "network alone (no search)"
    if v.startswith("known"):
        return "known parts, " + ("all candidates" if ":k0" in v else "$K=6$")
    parts = v.split(":")
    kind = parts[0]
    h = next((p[1:] for p in parts if p.startswith("h")), None)
    m = next((p[1:] for p in parts if p.startswith("m")), None)
    w = next((p[1:] for p in parts if p.startswith("w")), None)
    pen = next((p[1:] for p in parts if p.startswith("p") and p[1:].replace(".", "").isdigit()), "1")
    k0 = ":k0" in v
    if kind == "trunc":
        s = f"value head at the first decision after $t+{h}$~s, {m} future{'s' if m != '1' else ''}"
    elif kind == "cut":
        s = f"value head at exactly $t+{h}$~s, {m} future{'s' if m != '1' else ''}"
    elif kind == "roll":
        s = f"{m} sampled future{'s' if m != '1' else ''}, {w}~s window"
    else:
        s = v
    if pen == "1" and kind in ("roll", "cut", "trunc"):
        s += ", penalty in the cost"
    elif pen == "0.5":
        s += ", half-weight penalty"
    if k0:
        s += ", all candidates"
    if ":fcell" in v:
        s += " (cell's arrival model)"
    return s


SELECT = ("net alone", "known:k6", "known:k0", "trunc:h20:m4:p1", "cut:h20:m4", "roll:m1:w30", "roll:m2:w30",
          "roll:m4:w30", "roll:m4:w30:p0", "roll:m4:w20:p0", "roll:m4:w45:p0", "roll:m4:w60", "roll:m6:w30:p0",
          "roll:m8:w30:p0", "roll:m8:w20:p0", "roll:m4:w30:p0:k0", "roll:m4:w30:p0.5")


def t_variants():
    found = {}
    for name in ("teacher_variants_s", "teacher_variants_s2", "teacher_variants_s3"):
        d = json.load(open(DATA / f"{name}.json"))
        for r in d["rows"]:
            found.setdefault(r["variant"], r)
    rows = [found[v] for v in SELECT if v in found]
    best = min(r["score"] for r in rows)
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"teacher over the trained S network & real & heavy & general & score & s / episode (real / heavy / general) \\",
         r"\midrule"]
    for r in rows:
        cells = [fmt(100 * r[g]["delta"], 100 * r[g]["ci"]) for g in ("real", "heavy", "general")]
        score = f"{100 * r['score']:+.2f}"
        if r["score"] == best:
            score = r"\textbf{" + score + "}"
        secs = " / ".join(f"{r[g]['secs']:.1f}" for g in ("real", "heavy", "general"))
        L.append(f"{variant_label(r['variant'])} & " + " & ".join(cells) + f" & {score} & {secs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def t_futures():
    d1 = json.load(open(DATA / "teacher_futures_s_b.json"))
    d2 = json.load(open(DATA / "teacher_futures_s_b_strong.json"))
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"policy & real & heavy & general & real, bursts (0.3 / 20 s) & heavy, strong bursts (0.2 / 40 s) & real, strong bursts \\",
         r"\midrule"]
    names = {"net alone": "network alone", "roll:m8:w30:p0": "teacher, Poisson futures",
             "roll:m8:w30:p0:fcell": "teacher, cell futures"}
    for r1 in d1["rows"]:
        r2 = next(r for r in d2["rows"] if r["variant"] == r1["variant"])
        cells = [fmt(100 * r1[g]["delta"], 100 * r1[g]["ci"]) for g in ("real", "heavy", "general", "real_burst")]
        cells += [fmt(100 * r2["heavy"]["delta"], 100 * r2["heavy"]["ci"]), fmt(100 * r2["real_burst"]["delta"], 100 * r2["real_burst"]["ci"])]
        L.append(f"{names[r1['variant']]} & " + " & ".join(cells) + r" \\")
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


def fig_training(path: Path):
    plt = _rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    for tag, lab, col in (("rl2_gen_s_big", "S (0.88 M)", "#1f77b4"), ("rl2_gen_m", "M (6.7 M)", "#ff7f0e"), ("rl2_gen_l", "L (39 M)", "#2ca02c")):
        its = load_training(tag)["iterations"]
        x = [it["iteration"] for it in its]
        ax.plot(x, [100 * it["val"]["net/score"] for it in its], "-o", color=col, ms=3, label=f"{lab}, network alone")
        ax.plot(x, [100 * it["val"]["teacher/score"] for it in its], "--", color=col, lw=0.9, label=f"{lab}, search over the network")
    ax.set_xlabel("expert-iteration round")
    ax.set_ylabel("validation score: mean paired $\\Delta$ vs NJ [%]")
    ax.set_title("known-parts teacher, three sizes")
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    its = load_training("rl2_gen_s_sf")["iterations"]
    x = [it["iteration"] for it in its]
    for g, col in (("real", "#1f77b4"), ("heavy", "#d62728"), ("general", "#7f7f7f")):
        y = [100 * it["val"][f"net/{g}"]["delta"] for it in its]
        h = [100 * it["val"][f"net/{g}"]["ci"] for it in its]
        ax.errorbar(x, y, yerr=h, fmt="-o", color=col, ms=3, capsize=2, label=f"{g}: network alone")
        ax.plot(x, [100 * it["val"][f"teacher/{g}"]["delta"] for it in its], "--", color=col, lw=0.9, label=f"{g}: teacher")
    best = load_training("rl2_gen_s_big")["best"]["score"] * 100
    ax.axhline(best, color="k", lw=0.6, ls=":", label=f"best S with the known-parts teacher (score {best:+.1f})")
    ax.set_xlabel("expert-iteration round (continued from the best S)")
    ax.set_ylabel("paired $\\Delta$ vs NJ [%]")
    ax.set_title("sampled-future teacher, size S")
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=5.5, loc="center right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confirm(b: Bench, path: Path):
    plt = _rc()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4), sharey=True)
    names = [regime_name(*reg) for reg in REGIMES]
    y = np.arange(len(REGIMES))[::-1]
    for ax, (title, pairs) in zip(axes, (("known-parts search vs NJ", [("search", "nearest_job")]),
                                        ("teacher and network vs NJ", [("teacher", "nearest_job"), ("net", "nearest_job")]),
                                        ("searches vs network", [("search", "net"), ("teacher", "net")]))):
        for k, (x_, y_) in enumerate(pairs):
            col = {"search": "#7f7f7f", "teacher": "#d62728", "net": "#1f77b4"}[x_]
            off = (k - (len(pairs) - 1) / 2) * 0.28
            for t, filled in (("B", True), ("C", False)):
                mus = []; hs = []
                for reg in REGIMES:
                    mu, h, _ = b.delta(reg, t, x_, y_)
                    mus.append(mu); hs.append(h)
                ax.errorbar(mus, y + off + (0.0 if filled else -0.12), xerr=hs, fmt="o" if filled else "o", color=col,
                            mfc=col if filled else "white", ms=3.5, capsize=1.5, lw=0.9,
                            label=f"{POL[x_].replace(chr(92) + 'textsc{nj}', 'NJ')} ({t})")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("paired $\\Delta$ of $J$ [%]")
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        ax.axhline(3.5, color="grey", lw=0.5, ls=":")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_latency(b: Bench, path: Path):
    plt = _rc()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for who, col in (("nearest_job", "#2ca02c"), ("search", "#7f7f7f"), ("net", "#1f77b4"), ("teacher", "#d62728")):
        xs = np.sort([x for reg in REGIMES for x in b.latency(reg, "B", who)])
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=col, lw=1.3, label=f"{POL[who].replace(chr(92) + 'textsc{nj}', 'NJ')} (p50 {pct(list(xs), .5):.3g} ms, p95 {pct(list(xs), .95):.3g} ms)")
    ax.set_xscale("log")
    ax.axvline(5, color="k", lw=0.6, ls=":")
    ax.text(5.2, 0.05, "5 ms budget", fontsize=7)
    ax.set_xlabel("latency per decision [ms], one thread, machine fully loaded (log scale)")
    ax.set_ylabel("fraction of decisions")
    ax.set_title("Decision latency, treatment B, eight regimes pooled")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", type=Path, default=None, help="phase-4 bench.json; regenerates the extracts first")
    args = ap.parse_args()
    if args.bench is not None:
        extract(args.bench)
    b = Bench()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    lat_tex, nB, nC = t_latency(b)
    tables = {"regimes": t_regimes(b), "confirm_B": t_confirm(b, "B"), "confirm_C": t_confirm(b, "C"),
              "latency": lat_tex, "ladder": t_ladder(), "sf_runs": t_sf_runs(), "variants": t_variants(),
              "futures": t_futures()}
    for name, tex in tables.items():
        (OUT / "tables" / f"{name}.tex").write_text("% generated by experiments/paper1_figures.py\n" + tex + "\n")
    fig_training(OUT / "figures/training.pdf")
    fig_confirm(b, OUT / "figures/confirm.pdf")
    fig_latency(b, OUT / "figures/latency.pdf")
    # headline numbers
    print(f"latency decisions pooled: B {nB}, C {nC}")
    for t in ("B", "C"):
        for who in ("search", "teacher", "net"):
            ds = [b.delta(reg, t, who, "nearest_job") for reg in REGIMES]
            print(f"{t} {who:8s} vs NJ: " + "  ".join(f"{mu:+.2f}±{h:.2f}" for mu, h, _ in ds))
        for who in ("search", "teacher"):
            ds = [b.delta(reg, t, who, "net") for reg in REGIMES]
            print(f"{t} {who:8s} vs net: " + "  ".join(f"{mu:+.2f}±{h:.2f}({n})" for mu, h, n in ds)
                  + f"  sig {sum(abs(mu) > h for mu, h, _ in ds)}/8")
        xs = [x for reg in REGIMES for x in b.latency(reg, t, "net")]
        print(f"{t} net latency pooled p50 {pct(xs, .5):.2f} p95 {pct(xs, .95):.2f} p99 {pct(xs, .99):.2f} max {max(xs):.1f} ms")
        xs = [x for reg in REGIMES for x in b.latency(reg, t, "teacher")]
        print(f"{t} teacher latency pooled p50 {pct(xs, .5):.0f} p95 {pct(xs, .95):.0f} max {max(xs):.0f} ms")
    la = [r["lookaheads"] for r in b.rows if r["who"] == "teacher" and r["lookaheads"]]
    print(f"teacher look-aheads per episode: mean {np.mean(la):.0f}, range {min(la)}-{max(la)}")
    print("written:", sorted(tables), [p.name for p in (OUT / "figures").glob("*.pdf")])


if __name__ == "__main__":
    main()
