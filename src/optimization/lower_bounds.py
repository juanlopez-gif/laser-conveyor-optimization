"""Provably valid lower bounds for the laser-conveyor problem (P).

Every bound in this module is valid for EVERY feasible schedule of the exact
continuous-time model P of ``docs/formal_formulation.md`` (Section 1), not only
for the schedules produced by the decoder or by the online policies. The
proofs are in ``docs/methods/theory.md``; the numbering below (L1, L2, ...)
follows that document.

Building blocks (all proved in theory.md):

* L1  earliest start  ``S_j >= e_j``: belt kinematics + upstream queue
  (spacing delta) + forced predecessors (blocking) + first move from home.
* L2  minimum gap ``S_j - C_i >= s_ij`` for i processed before j:
  |dy|/v_y between conveyors, delta/(v_x+v_k) (part behind) or delta/v_x
  (part ahead) on the same conveyor.
* L3  Jackson / Hamiltonian-path makespan bound over the tail sets
  ``{j : e_j >= t}``.
* L4  per-conveyor bounds (belt stops during processing of own parts).
* L5  preemptive SRPT bound for sum C_j (1|r_j,pmtn|sum C_j), with and
  without setup inflation.
* L6  sequence relaxation: for every order pi a recursion gives
  componentwise lower bounds of C_j; its minimum over pi is computed exactly
  by dynamic programming over subsets (small n, or on windows of parts).

Public API: :func:`best_lower_bound`, :func:`all_lower_bounds`,
:func:`sequence_lower_bound`, :func:`bound_data`.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from optimization.instances import Instance, instance_config

#: DP over subsets is exact for the relaxation up to this many parts;
#: larger instances use windows of this size.
DP_MAX_PARTS = 10
DP_WINDOW = 8

SUPPORTED = ("makespan", "avg_flow_time", "avg_waiting_time")


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class BoundData:
    """Instance + parameters in the notation of the formal model."""

    n: int
    ids: list[str]
    r: list[float]
    p: list[float]
    conv: list[int]            # conveyor id (1-based)
    y: list[float]
    v: list[float]             # belt speed of the part's conveyor
    m: list[int]               # 0-based index in belt (arrival) order of its conveyor
    a: float
    b: float
    delta: float
    vx: float
    vy: float
    model: str
    x0: float
    y0: float
    all_stop: bool             # sigma = all ones (every conveyor stops while processing)
    blocking: bool             # unprocessed parts block at b (C4)
    by_conv: dict[int, list[int]]   # conveyor -> part indices in belt order

    # ---- elementary quantities ------------------------------------------- #
    def tau_home(self, j: int) -> float:
        """min over X in [a,b] of tau(P0, (X, y_j))  (first move from home)."""
        dx = max(self.a - self.x0, 0.0, self.x0 - self.b)
        dy = abs(self.y[j] - self.y0)
        tx, ty = dx / self.vx, dy / self.vy
        return max(tx, ty) if self.model == "simultaneous" else tx + ty

    def gap(self, i: int, j: int) -> float:
        """L2: lower bound of S_j - C_i when i is processed before j."""
        if self.conv[i] != self.conv[j]:
            return abs(self.y[i] - self.y[j]) / self.vy
        dm = self.m[j] - self.m[i]
        if dm > 0:                                   # j behind i on the belt
            return dm * self.delta / (self.vx + self.v[j])
        return (-dm) * self.delta / self.vx          # j ahead of i

    def same_conv_gap(self, k: int) -> float:
        """Smallest L2 gap between two parts of conveyor k."""
        j = self.by_conv[k][0]
        return self.delta / (self.vx + self.v[j])


def bound_data(inst: Instance) -> BoundData:
    """Read every parameter from ``instance_config`` (never hard-coded)."""
    cfg = instance_config(inst)
    g, las, sim = cfg.geometry, cfg.laser, cfg.simulation
    spec = {c.id: c for c in cfg.conveyors}
    jobs = list(inst.jobs)
    by_conv: dict[int, list[int]] = {}
    for idx, j in sorted(enumerate(jobs), key=lambda t: (t[1].time, t[1].id)):
        by_conv.setdefault(j.conveyor, []).append(idx)
    m = [0] * len(jobs)
    for lst in by_conv.values():
        for pos, idx in enumerate(lst):
            m[idx] = pos
    return BoundData(
        n=len(jobs), ids=[j.id for j in jobs], r=[j.time for j in jobs],
        p=[j.processing_time for j in jobs], conv=[j.conveyor for j in jobs],
        y=[spec[j.conveyor].y for j in jobs], v=[spec[j.conveyor].speed for j in jobs], m=m,
        a=g.active_start, b=g.active_end, delta=sim.min_part_spacing,
        vx=las.speed_x, vy=las.speed_y, model=las.motion_model, x0=las.home_x, y0=las.home_y,
        all_stop=sim.conveyor_stop_policy == "all_conveyors",
        blocking=sim.unprocessed_at_zone_end == "block", by_conv=by_conv)


# --------------------------------------------------------------------------- #
# L1: earliest start                                                          #
# --------------------------------------------------------------------------- #
def _forced_before(d: BoundData, j: int) -> list[int]:
    """Parts of k(j) ahead of j that must be COMPLETED before S_j (blocking):
    i ahead of j with a + (m_j - m_i) delta > b."""
    if not d.blocking:
        return []
    return [i for i in d.by_conv[d.conv[j]]
            if d.m[i] < d.m[j] and d.a + (d.m[j] - d.m[i]) * d.delta > d.b + 1e-12]


def earliest_starts(d: BoundData) -> list[float]:
    """L1: e_j <= S_j for every feasible schedule of P."""
    e = [0.0] * d.n
    for k, lst in d.by_conv.items():
        for j in lst:                                   # belt order: forced parts already done
            forced = _forced_before(d, j)
            fset = set(forced)
            val = d.tau_home(j)
            for h in lst:
                if d.m[h] > d.m[j]:
                    break
                stop = sum(d.p[i] for i in forced if d.m[i] >= d.m[h])
                val = max(val, d.r[h] + (d.a + (d.m[j] - d.m[h]) * d.delta) / d.v[j] + stop)
            for i in fset:
                val = max(val, e[i] + d.p[i] + d.gap(i, j))
            e[j] = val
    return e


# --------------------------------------------------------------------------- #
# L3: Jackson tail sets with Hamiltonian-path gap bound                       #
# --------------------------------------------------------------------------- #
def path_gap_bound(d: BoundData, U: list[int]) -> float:
    """H(U): lower bound on the sum of the |U|-1 gaps between consecutive
    parts of U in any processing order (Lemma H in theory.md)."""
    if len(U) <= 1:
        return 0.0
    counts: dict[int, int] = {}
    for j in U:
        counts[d.conv[j]] = counts.get(d.conv[j], 0) + 1
    ks = sorted(counts)
    K = len(ks)
    c = {k: d.same_conv_gap(k) for k in ks}
    intra = sum((counts[k] - 1) * c[k] for k in ks)
    if K == 1:
        return intra
    yk = [d.y[d.by_conv[k][0]] for k in ks]
    span = max(yk) - min(yk)
    gmin = min((abs(y1 - y2) for i1, y1 in enumerate(yk) for y2 in yk[i1 + 1:]), default=0.0)
    cmax = max(c.values())
    best = math.inf
    for T in range(K - 1, len(U)):
        val = intra - (T - K + 1) * cmax + max(span, T * gmin) / d.vy
        best = min(best, val)
    return max(best, 0.0)


def makespan_jackson(d: BoundData, e: list[float], with_gaps: bool = True) -> float:
    """L3: C_max >= t + sum_{U_t} p + H(U_t), U_t = {j: e_j >= t}."""
    order = sorted(range(d.n), key=lambda j: e[j])
    best = 0.0
    for q in range(d.n):
        U = order[q:]
        val = e[order[q]] + sum(d.p[j] for j in U)
        if with_gaps:
            val += path_gap_bound(d, U)
        best = max(best, val)
    return best


# --------------------------------------------------------------------------- #
# L4: per-conveyor bounds                                                     #
# --------------------------------------------------------------------------- #
def makespan_per_conveyor(d: BoundData) -> float:
    """L4a: C_max >= r_h + (a + (m_last - m_h) delta)/v_k + sum_{i >= h} p_i."""
    best = 0.0
    for k, lst in d.by_conv.items():
        last = lst[-1]
        for h in lst:
            tail = sum(d.p[i] for i in lst if d.m[i] >= d.m[h])
            best = max(best, d.r[h] + (d.a + (d.m[last] - d.m[h]) * d.delta) / d.v[h] + tail)
    return best


def sum_completion_per_conveyor(d: BoundData) -> float:
    """L4b: sum_j C_j >= sum_k sum_l LB(C_(l)) (order statistics per conveyor)."""
    total = 0.0
    for k, lst in d.by_conv.items():
        nk = len(lst)
        for l in range(1, nk + 1):                    # l-th completion among parts of k
            best = -math.inf
            for h in range(1, l + 1):
                ps = sorted(d.p[i] for i in lst[h - 1:])[: l - h + 1]
                best = max(best, d.r[lst[h - 1]] + (d.a + (l - h) * d.delta) / d.v[lst[0]] + sum(ps))
            total += best
    return total


# --------------------------------------------------------------------------- #
# L5: preemptive SRPT                                                         #
# --------------------------------------------------------------------------- #
def srpt_sum_completion(release: list[float], proc: list[float]) -> float:
    """Optimal value of 1|r_j, pmtn|sum C_j (SRPT, Schrage 1968)."""
    jobs = sorted(zip(release, proc))
    heap: list[float] = []
    t, i, total, n = -math.inf, 0, 0.0, len(jobs)
    while i < n or heap:
        if not heap:
            t = max(t, jobs[i][0])
        while i < n and jobs[i][0] <= t + 1e-12:
            heapq.heappush(heap, jobs[i][1])
            i += 1
        rem = heapq.heappop(heap)
        nxt = jobs[i][0] if i < n else math.inf
        if t + rem <= nxt + 1e-12:
            t += rem
            total += t
        else:
            heapq.heappush(heap, rem - (nxt - t))
            t = nxt
    return total


def sum_completion_srpt(d: BoundData, e: list[float]) -> float:
    """L5: max of SRPT on (e_j, p_j) and on the setup-inflated jobs
    (e_j - g_j, p_j + g_j), g_j = min_{i != j} s_ij."""
    plain = srpt_sum_completion(e, d.p)
    if d.n <= 1:
        return plain
    g = [min(d.gap(i, j) for i in range(d.n) if i != j) for j in range(d.n)]
    inflated = srpt_sum_completion([e[j] - g[j] for j in range(d.n)],
                                   [d.p[j] + g[j] for j in range(d.n)])
    return max(plain, inflated)


# --------------------------------------------------------------------------- #
# L6: sequence relaxation                                                     #
# --------------------------------------------------------------------------- #
class _Relaxation:
    """Recursion of Proposition L6 restricted to a subset U of the parts."""

    def __init__(self, d: BoundData, U: list[int]):
        self.d, self.U = d, list(U)
        self.pos = {j: q for q, j in enumerate(self.U)}
        nU = len(self.U)
        self.home = [d.tau_home(j) for j in self.U]
        # per local part: list of (base_h, mask_h) and forced mask
        self.terms: list[list[tuple[float, int]]] = []
        self.forced: list[int] = []
        for j in self.U:
            k, terms = d.conv[j], []
            for h in d.by_conv[k]:
                if d.m[h] > d.m[j]:
                    break
                base = d.r[h] + (d.a + (d.m[j] - d.m[h]) * d.delta) / d.v[j]
                mask = 0
                for i in self.U:
                    if i == j:
                        continue
                    if d.all_stop:
                        if d.r[i] >= d.r[h]:
                            mask |= 1 << self.pos[i]
                    elif d.conv[i] == k and d.m[i] >= d.m[h]:
                        mask |= 1 << self.pos[i]
                terms.append((base, mask))
            self.terms.append(terms)
            fm = 0
            for i in _forced_before(d, j):
                if i in self.pos:
                    fm |= 1 << self.pos[i]
            self.forced.append(fm)
        self.gap = [[d.gap(i, j) if i != j else 0.0 for j in self.U] for i in self.U]
        self.p = [d.p[j] for j in self.U]
        self.nU = nU

    def release(self, q: int, done: int, psum) -> float:
        """Lower bound of S_q given the set ``done`` processed before it."""
        if self.forced[q] & ~done:
            return math.inf
        val = self.home[q]
        for base, mask in self.terms[q]:
            s = base + psum(done & mask)
            if s > val:
                val = s
        return val

    def sequence(self, order: list[int]) -> list[float]:
        """Completion lower bounds along a fixed order (local indices)."""
        done, C, prev = 0, [0.0] * self.nU, None
        psum = lambda msk: sum(self.p[q] for q in range(self.nU) if msk >> q & 1)
        for q in order:
            s = self.release(q, done, psum)
            if prev is not None:
                s = max(s, C[prev] + self.gap[prev][q])
            C[q] = s + self.p[q]
            done |= 1 << q
            prev = q
        return C

    def dp(self) -> tuple[float, float]:
        """(min C_max, min sum C) of the relaxation over all orders of U."""
        nU = self.nU
        if nU == 0:
            return 0.0, 0.0
        full = (1 << nU) - 1
        ps = [0.0] * (1 << nU)
        for msk in range(1, 1 << nU):
            low = msk & -msk
            ps[msk] = ps[msk ^ low] + self.p[low.bit_length() - 1]
        psum = ps.__getitem__
        INF = math.inf
        bestC = [[INF] * nU for _ in range(1 << nU)]
        bestF = [[INF] * nU for _ in range(1 << nU)]
        for q in range(nU):
            c = self.release(q, 0, psum) + self.p[q]
            bestC[1 << q][q] = c
            bestF[1 << q][q] = c
        rel_cache: dict[tuple[int, int], float] = {}
        for msk in range(1, full + 1):
            rowC, rowF = bestC[msk], bestF[msk]
            for i in range(nU):
                ci = rowC[i]
                if ci == INF:
                    continue
                fi, gi = rowF[i], self.gap[i]
                for q in range(nU):
                    if msk >> q & 1:
                        continue
                    key = (msk, q)
                    rq = rel_cache.get(key)
                    if rq is None:
                        rq = self.release(q, msk, psum)
                        rel_cache[key] = rq
                    s = ci + gi[q]
                    if rq > s:
                        s = rq
                    c = s + self.p[q]
                    nm = msk | (1 << q)
                    if c < bestC[nm][q]:
                        bestC[nm][q] = c
                    if fi + c < bestF[nm][q]:
                        bestF[nm][q] = fi + c
        return min(bestC[full]), min(bestF[full])


def sequence_lower_bound(inst: Instance, order: list[str]) -> dict:
    """L6 for a FIXED processing order: valid lower bounds of every P-schedule
    that processes the parts in ``order``. Returns makespan / sum_C / avg_flow."""
    d = bound_data(inst)
    rel = _Relaxation(d, list(range(d.n)))
    idx = {pid: q for q, pid in enumerate(d.ids)}
    C = rel.sequence([idx[pid] for pid in order])
    sc = sum(C)
    return {"makespan": max(C), "sum_C": sc, "avg_flow_time": (sc - sum(d.r)) / d.n,
            "avg_waiting_time": (sc - sum(d.p) - sum(d.r)) / d.n, "C": C}


def relaxation_bounds(d: BoundData, e: list[float]) -> tuple[float, float]:
    """L6 by DP: exact on the whole instance if n <= DP_MAX_PARTS; otherwise
    max over windows (makespan) and sum over a partition into windows (sum C)."""
    if d.n == 0:
        return 0.0, 0.0
    if d.n <= DP_MAX_PARTS:
        return _Relaxation(d, list(range(d.n))).dp()
    order = sorted(range(d.n), key=lambda j: (e[j], j))
    cache: dict[tuple[int, ...], tuple[float, float]] = {}

    def solve(B: list[int]) -> tuple[float, float]:
        key = tuple(B)
        if key not in cache:
            cache[key] = _Relaxation(d, B).dp()
        return cache[key]

    cm, best_sc = 0.0, 0.0
    # several partitions into consecutive windows (shifted by `off`); each one
    # gives a valid bound on sum C_j (Corollary L6'), keep the best
    for off in range(0, DP_WINDOW, max(1, DP_WINDOW // 4)):
        cuts = ([0] if off else []) + list(range(off, d.n, DP_WINDOW)) + [d.n]
        sc = 0.0
        for lo, hi in zip(cuts, cuts[1:]):
            if hi > lo:
                c, f = solve(order[lo:hi])
                cm, sc = max(cm, c), sc + f
        best_sc = max(best_sc, sc)
    # the last window (largest e) is the most informative for the makespan
    cm = max(cm, solve(order[-DP_WINDOW:])[0])
    return cm, best_sc


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def trivial_makespan(d: BoundData) -> float:
    """Section 5.4 of the formulation (= benchmark.trivial_lower_bound)."""
    avail = [d.r[j] + d.a / d.v[j] for j in range(d.n)]
    return max(max(av + d.p[j] for j, av in enumerate(avail)), min(avail) + sum(d.p))


def all_lower_bounds(inst: Instance, objective: str, use_dp: bool = True) -> dict[str, float]:
    """Every bound for ``objective`` (dict name -> value)."""
    if objective not in SUPPORTED:
        raise ValueError(f"no lower bound for objective {objective!r}")
    d = bound_data(inst)
    if d.n == 0:
        return {"empty": 0.0}
    e = earliest_starts(d)
    out: dict[str, float] = {}
    dp = relaxation_bounds(d, e) if use_dp else None
    if objective == "makespan":
        out["trivial"] = trivial_makespan(d)
        out["L1_release"] = max(e[j] + d.p[j] for j in range(d.n))
        out["L3_jackson"] = makespan_jackson(d, e, with_gaps=False)
        out["L3_jackson_gaps"] = makespan_jackson(d, e, with_gaps=True)
        out["L4_conveyor"] = makespan_per_conveyor(d)
        if dp is not None:
            out["L6_relaxation"] = dp[0]
        return out
    sum_r, sum_p, n = sum(d.r), sum(d.p), d.n
    sums = {"L1_release": sum(e[j] + d.p[j] for j in range(n)),
            "L4_conveyor": sum_completion_per_conveyor(d),
            "L5_srpt": sum_completion_srpt(d, e)}
    if dp is not None:
        sums["L6_relaxation"] = dp[1]
    shift = sum_r + (sum_p if objective == "avg_waiting_time" else 0.0)
    return {k: (v - shift) / n for k, v in sums.items()}


def best_lower_bound(instance: Instance, objective) -> float | None:
    """Best valid lower bound of ``objective`` for every feasible schedule of P.

    ``objective`` is "makespan", "avg_flow_time" or "avg_waiting_time";
    anything else (e.g. a composite weights dict) returns None.
    """
    if not isinstance(objective, str) or objective not in SUPPORTED:
        return None
    return max(all_lower_bounds(instance, objective).values())
