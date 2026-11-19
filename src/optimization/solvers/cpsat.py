"""Exact method: the time-indexed model P_Delta solved with OR-Tools CP-SAT.

Implements Section 2 of ``docs/formal_formulation.md`` (constraints M1-M14). Time is
discretised in steps of ``step`` seconds (Delta) and every length is expressed in an
integer *position unit* chosen so that the belt advance per step ``v_k * Delta``, the
zone limits ``a, b``, the spacing ``delta`` and the laser home ``x_0`` are integers
(1 mm with the default configuration). All parameters are read from the simulator
``Config`` of the instance; nothing is hard-coded.

Model (the mapping to M1-M14 is documented in ``docs/methods/exact_cpsat.md``):

* ``s[j][t]``   start binaries (M1), ``c[j][t]`` = started at or before step t;
* ``u[k][t]``   belt run binaries, ``D[k][t]`` belt odometers (M2);
* ``E[j]``      entry coordinate = max(odometer at arrival, upstream part + delta) (M3);
* ``X[j]``      processing position ``D[k][S_j] - E_j`` through ``AddElement`` (M4);
* ``a <= X_j <= b``: reachability window and blocking (M5);
* belt stopped while a part with sigma = 1 is processed (M6);
* pairwise laser disjunction with travel times + redundant ``NoOverlap`` (M7-M9);
* non-idling conveyors with blocking indicators ``beta[j][t]`` (M12-M14).

Solving strategy: a *discrete decoder* (:func:`decode_order`) computes, for a given
processing order, the earliest schedule of P_Delta itself. It is used to build a
complete feasible warm start (best of the dispatching policies' orders improved by a
short swap descent) and to derive valid start-time domains from the warm-start
objective value, so the horizon never cuts off an improving solution.

The returned processing order is replayed in the simulator by the benchmark (and,
when ``validate=True``, inside :func:`solve_cpsat` for reporting in ``info``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

from ortools.sat.python import cp_model

from optimization.evaluate import VALIDATION_DT, evaluate_sequence
from optimization.instances import Instance, instance_config
from optimization.solvers import Solution, register_method
from simulation.config import Config
from simulation.engine import Simulation

#: default time step Delta [s]
DEFAULT_STEP = 0.1
#: objectives supported by the model (names of optimization.objective_functions.OBJECTIVES)
SUPPORTED_OBJECTIVES = ("makespan", "avg_flow_time", "avg_waiting_time")
#: makespan is modelled in units of Delta / MAKESPAN_RES so p_j need not be a multiple of Delta
MAKESPAN_RES = 100
#: policies whose realised orders seed the warm start
WARM_START_POLICIES = ("nearest_job", "fifo", "round_robin", "weighted_score")
#: CP-SAT parameters. Probing is disabled: on these models it otherwise spends most of
#: the time limit adding binary clauses before the search starts (measured: 2-3x faster)
SOLVER_PARAMS: dict = {"cp_model_probing_level": 0}

_STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
           cp_model.INFEASIBLE: "INFEASIBLE", cp_model.MODEL_INVALID: "MODEL_INVALID",
           cp_model.UNKNOWN: "UNKNOWN"}


# =============================================================================== scaling
def _frac(x: float) -> Fraction:
    return Fraction(x).limit_denominator(10_000)


def _lcm(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out = out * v // math.gcd(out, v)
    return out


def _ceil_div(num: int, den: int) -> int:
    return -((-num) // den)


@dataclass
class ScaledData:
    """Instance data in integer steps (Delta) and integer position units."""

    step: float                        # Delta [s]
    unit: Fraction                     # position unit [cm]
    conveyors: list[int]               # conveyor ids (1-based)
    w: dict[int, int]                  # belt advance per running step, v_k*Delta/unit
    a: int                             # start of the laser zone [units]
    b: int                             # end of the laser zone [units]
    spacing: int                       # delta [units]
    x0: int                            # laser home x [units]
    r: list[int]                       # release step ceil(r_j / Delta)
    p: list[int]                       # processing steps ceil(p_j / Delta)
    k: list[int]                       # conveyor of each part
    pred: list[int | None]             # previous part on the same conveyor
    stops: dict[int, list[int]]        # parts whose processing stops conveyor k (sigma_{k,k(j)} = 1)
    blocking: bool                     # unprocessed_at_zone_end == "block"
    x_steps: Fraction                  # laser steps per x-unit, 1 / (v_x * Delta / unit)
    y_frac: dict[tuple[int, int], Fraction] = field(default_factory=dict)  # |dy| / (v_y * Delta)
    sequential: bool = False           # motion_model == "sequential"

    def travel_steps(self, dx: int, rows: tuple[int, int]) -> int:
        """Smallest integer number of steps covering the laser move (row 0 = home)."""
        if self.sequential:
            return math.ceil(self.x_steps * dx + self.y_frac[rows])
        return max(math.ceil(self.x_steps * dx), math.ceil(self.y_frac[rows]))

    def min_gap(self, i: int, j: int) -> int:
        """Lower bound (steps) on S_j - C_i when j is processed after i.

        Different conveyors: the y move. Same conveyor: at C_i the two parts are at least
        ``delta`` apart along the belt and the laser and the part close that distance at
        most at ``v_x + v_k``.
        """
        if self.k[i] != self.k[j]:                 # both motion models need >= |dy| / v_y
            return math.ceil(self.y_frac[(self.k[i], self.k[j])])
        closing = 1 / self.x_steps + self.w[self.k[i]]          # units per step
        return math.ceil(Fraction(self.spacing) / closing)

    def est(self, j: int) -> int:
        """Earliest start of part j: reach the zone (a / v) and the first laser move in y."""
        return max(self.r[j] + _ceil_div(self.a, self.w[self.k[j]]),
                   math.ceil(self.y_frac[(self.k[j], 0)]))


def scale_instance(inst: Instance, cfg: Config, step: float) -> ScaledData:
    """Convert the instance and its configuration to the integer data of P_Delta."""
    g, las, sim = cfg.geometry, cfg.laser, cfg.simulation
    speed = {c.id: c.speed for c in cfg.conveyors}
    lengths = [_frac(v * step) for v in speed.values()]
    lengths += [_frac(g.active_start), _frac(g.active_end), _frac(sim.min_part_spacing),
                _frac(las.home_x), Fraction(1, 10)]          # at least mm resolution
    denominator = _lcm([f.denominator for f in lengths])
    if denominator > 10_000:
        raise ValueError("cannot find an integral position unit for this configuration")
    unit = Fraction(1, denominator)

    def to_units(x: float) -> int:
        return int(_frac(x) / unit)

    jobs = list(inst.jobs)
    last_on: dict[int, int] = {}
    pred: list[int | None] = []
    for idx, j in enumerate(jobs):          # jobs are sorted by (time, conveyor) = belt order
        pred.append(last_on.get(j.conveyor))
        last_on[j.conveyor] = idx
    stop_all = sim.conveyor_stop_policy == "all_conveyors"
    stops = {kk: [i for i, j in enumerate(jobs) if stop_all or j.conveyor == kk] for kk in speed}
    rows = {**{c.id: c.y for c in cfg.conveyors}, 0: las.home_y}      # row 0 = laser home
    y_frac = {(k1, k2): _frac(abs(y1 - y2) / (las.speed_y * step))
              for k1, y1 in rows.items() for k2, y2 in rows.items()}
    eps = 1e-9
    return ScaledData(
        step=step, unit=unit, conveyors=sorted(speed),
        w={kk: to_units(v * step) for kk, v in speed.items()},
        a=to_units(g.active_start), b=to_units(g.active_end),
        spacing=to_units(sim.min_part_spacing), x0=to_units(las.home_x),
        r=[int(math.ceil(j.time / step - eps)) for j in jobs],
        p=[int(math.ceil(j.processing_time / step - eps)) for j in jobs],
        k=[j.conveyor for j in jobs], pred=pred, stops=stops,
        blocking=sim.unprocessed_at_zone_end == "block",
        x_steps=1 / (_frac(las.speed_x * step) / unit),
        y_frac=y_frac, sequential=las.motion_model == "sequential")


# =============================================================================== decoder
@dataclass
class DiscreteSchedule:
    """A complete solution of P_Delta (earliest schedule of a processing order)."""

    order: list[int]
    S: list[int]                       # start steps
    E: list[int]                       # entry coordinates [units]
    X: list[int]                       # processing positions [units]
    D: dict[int, list[int]]            # odometers, t = 0..len
    u: dict[int, list[int]]            # run indicators, t = 0..len-1
    R: list[int]                       # ready steps (first step with x_j >= a)

    def value(self, objective: str, inst: Instance, d: ScaledData) -> float:
        """Objective in seconds, computed like the model does."""
        if objective == "makespan":
            return max(d.step * s + j.processing_time for s, j in zip(self.S, inst.jobs))
        total = sum(d.step * s - j.time for s, j in zip(self.S, inst.jobs))
        if objective == "avg_flow_time":
            total += sum(j.processing_time for j in inst.jobs)
        return total / inst.n


def decode_order(d: ScaledData, order: Sequence[int], max_steps: int,
                 ready_dispatch: bool = False) -> DiscreteSchedule | None:
    """Earliest (semi-active) schedule of P_Delta that processes the parts in ``order``.

    Discrete twin of the simulator's fixed-sequence decoder. Conveyors run unless
    processing (sigma) or blocked (an unstarted part within one step of ``b``). With
    ``ready_dispatch=False`` (formulation P) the laser may travel towards a part before it
    enters the laser zone; with ``True`` (simulator semantics) it leaves its previous
    point only at ``max(C_prev, R_j)``, ``R_j`` = first step with ``x_j >= a``.
    Returns ``None`` if the order deadlocks or exceeds ``max_steps``.
    """
    n = len(d.r)
    E: list[int | None] = [None] * n
    S: list[int | None] = [None] * n
    R: list[int | None] = [None] * n
    X: list[int] = [0] * n
    D = {kk: [0] for kk in d.conveyors}
    u: dict[int, list[int]] = {kk: [] for kk in d.conveyors}
    arrivals = sorted(range(n), key=lambda j: (d.r[j], j))
    next_arrival, pos, free = 0, 0, 0
    point, row = d.x0, 0
    end = 0
    for t in range(max_steps):
        while next_arrival < n and d.r[arrivals[next_arrival]] <= t:      # (M3)
            j = arrivals[next_arrival]
            pj = d.pred[j]
            E[j] = D[d.k[j]][t] if pj is None else max(D[d.k[j]][t], E[pj] + d.spacing)
            next_arrival += 1
        for j in range(n):
            if R[j] is None and E[j] is not None and D[d.k[j]][t] - E[j] >= d.a:
                R[j] = t
        if pos < n and t >= free:
            j = order[pos]
            if E[j] is not None:
                x = D[d.k[j]][t] - E[j]
                leave = max(free, R[j] if R[j] is not None else t) if ready_dispatch else free
                if d.a <= x <= d.b and t >= leave + d.travel_steps(abs(x - point), (row, d.k[j])) \
                        and t >= d.travel_steps(abs(x - d.x0), (0, d.k[j])):
                    S[j], X[j] = t, x
                    point, row, free = x, d.k[j], t + d.p[j]
                    end = max(end, free)
                    pos += 1
        if pos == n and t >= end:
            return DiscreteSchedule(list(order), S, E, X, D, u, R)   # type: ignore[arg-type]
        for kk in d.conveyors:
            busy = any(S[i] is not None and S[i] <= t < S[i] + d.p[i] for i in d.stops[kk])
            blocked = d.blocking and any(
                E[i] is not None and S[i] is None and D[kk][t] - E[i] > d.b - d.w[kk]
                for i in range(n) if d.k[i] == kk)
            run = 0 if busy or blocked else 1
            u[kk].append(run)
            D[kk].append(D[kk][t] + d.w[kk] * run)
    return None


def policy_order(inst: Instance, policy: str, dt: float = 0.05) -> list[int]:
    """Processing order realised by an online policy in the simulator (indices of jobs)."""
    cfg = instance_config(inst, policy, dt=dt)
    result = Simulation(cfg, record_trace=False).run()
    index = {j.id: i for i, j in enumerate(inst.jobs)}
    started = sorted((p for p in result.parts if p.start_processing_time is not None),
                     key=lambda p: p.start_processing_time)
    return [index[p.id] for p in started]


def warm_start(inst: Instance, d: ScaledData, objective: str, max_steps: int,
               budget_s: float, ready_dispatch: bool) -> tuple[DiscreteSchedule | None, float, str]:
    """Best decoded policy order, improved by a first-improvement swap descent."""
    best, best_val, source = None, math.inf, ""
    for pol in WARM_START_POLICIES:
        try:
            order = policy_order(inst, pol)
        except Exception:                      # a policy failing must not break the solver
            continue
        if len(order) != inst.n:
            continue
        sched = decode_order(d, order, max_steps, ready_dispatch)
        if sched is not None and sched.value(objective, inst, d) < best_val - 1e-9:
            best, best_val, source = sched, sched.value(objective, inst, d), pol
    if best is None:
        return None, math.inf, ""
    deadline, improved = time.time() + budget_s, True
    while improved and time.time() < deadline:
        improved = False
        for i in range(inst.n - 1):
            for j in range(i + 1, inst.n):
                if time.time() >= deadline:
                    break
                order = list(best.order)
                order[i], order[j] = order[j], order[i]
                sched = decode_order(d, order, max_steps, ready_dispatch)
                if sched is not None and sched.value(objective, inst, d) < best_val - 1e-9:
                    best, best_val, improved = sched, sched.value(objective, inst, d), True
                    source = source.split("+")[0] + "+swaps"
    return best, best_val, source


# =============================================================================== model
@dataclass
class ModelHandles:
    """The CP-SAT model plus the variables needed to hint it and read a solution."""

    model: cp_model.CpModel
    S: list[cp_model.IntVar]
    X: list[cp_model.IntVar]
    s: list[dict[int, cp_model.IntVar]]
    objective_scale: float             # seconds per objective unit
    objective_offset: float            # seconds added to scale * objective
    horizon: int
    stats: dict
    variables: dict = field(default_factory=dict)   # u, D, c, E, R, z, before (for hints)


def _travel_lb(m: cp_model.CpModel, T: cp_model.IntVar, dx_abs: cp_model.IntVar,
               d: ScaledData, rows: tuple[int, int]) -> None:
    """T (steps) >= travel time of a move with |dx| = dx_abs units between ``rows``."""
    ax = d.x_steps
    if d.sequential:
        fy = d.y_frac[rows]
        den = _lcm([ax.denominator, fy.denominator])
        m.add(den * T >= int(ax * den) * dx_abs + int(fy * den))
    else:
        m.add(ax.denominator * T >= ax.numerator * dx_abs)
        m.add(T >= math.ceil(d.y_frac[rows]))


def start_windows(inst: Instance, d: ScaledData, objective: str,
                  incumbent: DiscreteSchedule) -> tuple[list[int], list[int]]:
    """Start-step domains [est_j, lst_j] that contain every solution at least as good as
    ``incumbent`` (so they never cut off an optimum)."""
    n = inst.n
    est = [d.est(j) for j in range(n)]
    if objective == "makespan":
        cmax = max(s + p for s, p in zip(incumbent.S, d.p))
        lst = [cmax - d.p[j] for j in range(n)]
    else:                                   # sum of starts <= incumbent sum
        total, base = sum(incumbent.S), sum(est)
        lst = [total - (base - est[j]) for j in range(n)]
    return est, lst


def min_gap_after(d: ScaledData, j: int) -> int:
    """Lower bound (steps) on the idle laser time after part j, whatever comes next."""
    others = [d.min_gap(j, i) for i in range(len(d.r)) if i != j]
    return min(others) if others else 0


def build_model(inst: Instance, objective: str, d: ScaledData, est: list[int], lst: list[int],
                ready_dispatch: bool = False) -> ModelHandles:
    """Build P_Delta for ``inst`` with start domains [est_j, lst_j].

    ``ready_dispatch=True`` adds the simulator's dispatching restriction (the laser
    starts moving towards a part only once it is in the laser zone), which needs
    immediate-successor literals (``AddCircuit``) and ready-time variables ``R_j``.
    """
    m = cp_model.CpModel()
    n = inst.n
    jobs = range(n)
    H = max(l + p for l, p in zip(lst, d.p))
    if any(e > l for e, l in zip(est, lst)):
        raise ValueError("empty start window")

    # (M2) odometers D[k][t], t = 0..H, and run binaries u[k][t], t = 0..H-1
    u = {kk: [m.new_bool_var(f"u_{kk}_{t}") for t in range(H)] for kk in d.conveyors}
    D = {kk: [m.new_constant(0)] + [m.new_int_var(0, d.w[kk] * t, f"D_{kk}_{t}")
                                     for t in range(1, H + 1)] for kk in d.conveyors}
    for kk in d.conveyors:
        for t in range(H):
            m.add(D[kk][t + 1] == D[kk][t] + d.w[kk] * u[kk][t])

    # (M1) start binaries s[j][t] on [est_j, lst_j]; c[j][t] = started at or before t
    s: list[dict[int, cp_model.IntVar]] = []
    c: list[dict[int, cp_model.IntVar]] = []
    S: list[cp_model.IntVar] = []
    for j in jobs:
        sj = {t: m.new_bool_var(f"s_{j}_{t}") for t in range(est[j], lst[j] + 1)}
        m.add_exactly_one(sj.values())
        Sj = m.new_int_var(est[j], lst[j], f"S_{j}")
        m.add(Sj == sum(t * v for t, v in sj.items()))
        cj: dict[int, cp_model.IntVar] = {}
        prev = None
        for t in range(est[j], lst[j] + 1):
            ct = m.new_bool_var(f"c_{j}_{t}")
            m.add(ct == sj[t] if prev is None else ct == prev + sj[t])
            cj[t], prev = ct, ct
        s.append(sj), c.append(cj), S.append(Sj)

    def started(j: int, t: int):
        """1 if part j started at a step <= t."""
        if t < est[j]:
            return 0
        return c[j][min(t, lst[j])]

    def processing(j: int, t: int):
        """1 if part j is being processed during step t."""
        return started(j, t) - started(j, t - d.p[j])

    def active(j: int, t: int) -> bool:
        return est[j] <= t < lst[j] + d.p[j]

    # (M3) E_j = max(D_k(r_j), E_pred + delta); (M4) X_j = D_k(S_j) - E_j; (M5) a <= X_j <= b
    E, X = [], []
    ub_pos = max(d.w.values()) * H
    for j in jobs:
        Ej = m.new_int_var(0, ub_pos, f"E_{j}")
        terms = [D[d.k[j]][d.r[j]]]
        if d.pred[j] is not None:
            terms.append(E[d.pred[j]] + d.spacing)
        m.add_max_equality(Ej, terms)
        DSj = m.new_int_var(0, ub_pos, f"DS_{j}")
        m.add_element(S[j], D[d.k[j]], DSj)
        Xj = m.new_int_var(d.a, d.b, f"X_{j}")
        m.add(Xj == DSj - Ej)
        E.append(Ej), X.append(Xj)

    # (M6) a conveyor does not run while a part with sigma = 1 is processed
    for kk in d.conveyors:
        for t in range(H):
            busy = [processing(j, t) for j in d.stops[kk] if active(j, t)]
            if busy:
                m.add(u[kk][t] + sum(busy) <= 1)

    # (M7)-(M8) disjunctive laser with travel times; (M9) first move from home
    # redundant NoOverlap: each interval is extended by a lower bound on the laser gap
    # to whatever part comes next (valid for every successor; see min_gap_after)
    m.add_no_overlap([m.new_fixed_size_interval_var(S[j], d.p[j] + min_gap_after(d, j), f"I_{j}")
                      for j in jobs])
    T0s, Ts, before = [], {}, {}
    for j in jobs:
        dx0 = m.new_int_var(0, max(abs(d.x0 - d.a), abs(d.x0 - d.b)), f"dx0_{j}")
        m.add_abs_equality(dx0, X[j] - d.x0)
        T0 = m.new_int_var(0, H, f"T0_{j}")
        _travel_lb(m, T0, dx0, d, (d.k[j], 0))
        m.add(S[j] >= T0)
        T0s.append(T0)
    for i in jobs:
        for j in range(i + 1, n):
            dx = m.new_int_var(0, d.b - d.a, f"dx_{i}_{j}")
            m.add_abs_equality(dx, X[i] - X[j])
            T = m.new_int_var(0, H, f"T_{i}_{j}")
            _travel_lb(m, T, dx, d, (d.k[i], d.k[j]))
            w_ij = m.new_bool_var(f"w_{i}_{j}")        # w_ij; w_ji is its negation (M7)
            m.add(S[j] >= S[i] + d.p[i] + T).only_enforce_if(w_ij)
            m.add(S[i] >= S[j] + d.p[j] + T).only_enforce_if(~w_ij)
            if d.k[i] == d.k[j]:                       # redundant: belt-spacing gap
                m.add(S[j] >= S[i] + d.p[i] + d.min_gap(i, j)).only_enforce_if(w_ij)
                m.add(S[i] >= S[j] + d.p[j] + d.min_gap(j, i)).only_enforce_if(~w_ij)
            Ts[(i, j)] = Ts[(j, i)] = T
            before[(i, j)], before[(j, i)] = w_ij, ~w_ij

    # simulator dispatching: S_j >= max(C_i, R_j) + tau_ij for the immediate predecessor i
    R: list[cp_model.IntVar] = []
    z: dict[tuple[int, int], cp_model.IntVar] = {}
    if ready_dispatch:
        for j in jobs:
            r_min = d.r[j] + _ceil_div(d.a, d.w[d.k[j]])
            Rj = m.new_int_var(r_min, lst[j], f"R_{j}")
            Rm = m.new_int_var(r_min - 1, lst[j] - 1, f"Rm_{j}")
            m.add(Rm == Rj - 1)
            DR = m.new_int_var(0, ub_pos, f"DR_{j}")
            DRm = m.new_int_var(0, ub_pos, f"DRm_{j}")
            m.add_element(Rj, D[d.k[j]], DR)
            m.add_element(Rm, D[d.k[j]], DRm)
            m.add(DR - E[j] >= d.a)            # in the zone at R_j ...
            m.add(DRm - E[j] <= d.a - 1)       # ... and not one step earlier
            m.add(S[j] >= Rj)
            R.append(Rj)
        arcs = []
        for j in jobs:                          # node 0 = laser home / end of the route
            z[(-1, j)] = m.new_bool_var(f"z_home_{j}")
            arcs.append((0, j + 1, z[(-1, j)]))
            arcs.append((j + 1, 0, m.new_bool_var(f"z_{j}_end")))
            m.add(S[j] >= R[j] + T0s[j]).only_enforce_if(z[(-1, j)])
            for i in jobs:
                if i != j:
                    z[(i, j)] = m.new_bool_var(f"z_{i}_{j}")
                    arcs.append((i + 1, j + 1, z[(i, j)]))
                    m.add_implication(z[(i, j)], before[(i, j)])
                    m.add(S[j] >= R[j] + Ts[(i, j)]).only_enforce_if(z[(i, j)])
        m.add_circuit(arcs)

    # (M12)-(M14) non-idling: a belt may stop only while processing (sigma) or when an
    # unstarted part of it is within one step of advance of b (discretisation slack)
    n_beta = 0
    for kk in d.conveyors:
        own = [j for j in jobs if d.k[j] == kk]
        wk = d.w[kk]
        first_block = {j: d.r[j] + _ceil_div(d.b - wk + 1, wk) for j in own}
        for t in range(H):
            reasons = [processing(j, t) for j in d.stops[kk] if active(j, t)]
            if d.blocking:
                for j in own:
                    if first_block[j] <= t <= lst[j]:
                        beta = m.new_bool_var(f"beta_{j}_{t}")
                        n_beta += 1
                        m.add(started(j, t) == 0).only_enforce_if(beta)                 # (M13)
                        m.add(D[kk][t] - E[j] >= d.b - wk + 1).only_enforce_if(beta)    # (M14)
                        reasons.append(beta)
            m.add(u[kk][t] + sum(reasons) >= 1)                                         # (M12)

    # objective in integer units; seconds = scale * value + offset
    r_sum = sum(j.time for j in inst.jobs)
    p_sum = sum(j.processing_time for j in inst.jobs)
    if objective == "makespan":
        cmax = m.new_int_var(0, MAKESPAN_RES * (H + 1), "Cmax")
        for j in jobs:
            p_res = math.ceil(inst.jobs[j].processing_time / d.step * MAKESPAN_RES - 1e-9)
            m.add(cmax >= MAKESPAN_RES * S[j] + p_res)
        m.minimize(cmax)
        scale, offset = d.step / MAKESPAN_RES, 0.0
    elif objective in ("avg_flow_time", "avg_waiting_time"):
        m.minimize(sum(S))
        scale = d.step / n
        offset = ((p_sum if objective == "avg_flow_time" else 0.0) - r_sum) / n
    else:
        raise ValueError(f"objective '{objective}' not supported by cpsat")

    stats = {"horizon_steps": H, "start_vars": sum(len(x) for x in s), "beta_vars": n_beta,
             "pairs": n * (n - 1) // 2}
    variables = {"u": u, "D": D, "c": c, "E": E, "R": R, "z": z, "before": before}
    return ModelHandles(m, S, X, s, scale, offset, H, stats, variables)


def add_full_hint(h: ModelHandles, sched: DiscreteSchedule) -> None:
    """Hint the primary variables with a decoded (feasible) schedule."""
    m = h.model
    parts = h.variables
    for j, Sj in enumerate(sched.S):
        m.add_hint(h.S[j], Sj)
        m.add_hint(h.X[j], sched.X[j])
        m.add_hint(parts["E"][j], sched.E[j])
        for t, v in h.s[j].items():
            m.add_hint(v, int(t == Sj))
        for t, v in parts["c"][j].items():
            m.add_hint(v, int(t >= Sj))
    position = {j: i for i, j in enumerate(sched.order)}
    for (i, j), lit in parts["before"].items():
        if i < j:
            m.add_hint(lit, int(position[i] < position[j]))
    for j, Rj in enumerate(parts["R"]):
        m.add_hint(Rj, sched.R[j])
    for (i, j), lit in parts["z"].items():
        prev = sched.order[position[j] - 1] if position[j] > 0 else -1
        m.add_hint(lit, int(prev == i))
    for kk, row in parts["u"].items():
        for t, v in enumerate(row):
            run = sched.u[kk][t] if t < len(sched.u[kk]) else 1
            m.add_hint(v, run)


def complete_hint(m: cp_model.CpModel, time_limit: float = 10.0) -> bool:
    """Extend a hint of the primary variables to *every* variable of the model.

    CP-SAT only tries briefly to complete a partial hint and may drop it; solving once
    with the hinted variables fixed (a pure propagation problem) yields a full
    assignment that is then used as a complete hint. Returns True on success.
    """
    solver = cp_model.CpSolver()
    solver.parameters.fix_variables_to_their_hinted_value = True
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    if solver.solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False
    values = [solver.value(m.get_int_var_from_proto_index(i))
              for i in range(len(m.proto.variables))]
    m.clear_hints()
    for i, v in enumerate(values):
        m.add_hint(m.get_int_var_from_proto_index(i), v)
    return True


# =============================================================================== solver
def _workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def solve_cpsat(inst: Instance, objective: str = "makespan", time_limit: float = 60.0,
                seed: int = 0, step: float = DEFAULT_STEP, validate: bool = True,
                num_workers: int | None = None, warm_start_share: float = 0.1,
                ready_dispatch: bool = False, solver_params: dict | None = None) -> Solution:
    """Solve P_Delta for ``inst`` and return the processing order, bound and diagnostics.

    Parameters
    ----------
    step : Delta [s]; ``v_k * step`` must be integral in some unit <= 1 mm.
    validate : replay the order in the simulator (dt = 0.01, formulation checker) and
        store the replayed objective in ``info``.
    warm_start_share : fraction of ``time_limit`` given to the warm-start swap descent.
    ready_dispatch : False = formulation P_Delta (the laser may anticipate a part);
        True = add the simulator's restriction (move only towards parts already in
        the zone), so that model and replay agree up to discretisation.
    solver_params : extra CP-SAT parameters (override :data:`SOLVER_PARAMS`).
    """
    if not isinstance(objective, str) or objective not in SUPPORTED_OBJECTIVES:
        return Solution(info={"skipped": f"objective {objective!r} not supported by cpsat"})
    t_start = time.time()
    cfg = instance_config(inst)
    d = scale_instance(inst, cfg, step)
    max_steps = int((max(j.time for j in inst.jobs) + 60.0 * inst.n + 300.0) / step)
    incumbent, inc_val, source = warm_start(inst, d, objective, max_steps,
                                            warm_start_share * time_limit, ready_dispatch)
    info: dict = {"step": step, "position_unit_cm": float(d.unit), "ready_dispatch": ready_dispatch,
                  "warm_start_value": round(inc_val, 4), "warm_start_source": source}
    if incumbent is None:
        info["status"] = "NO_WARM_START"
        return Solution(info=info)

    est, lst = start_windows(inst, d, objective, incumbent)
    h = build_model(inst, objective, d, est, lst, ready_dispatch)
    add_full_hint(h, incumbent)
    hint_complete = complete_hint(h.model)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.5, time_limit - (time.time() - t_start))
    solver.parameters.random_seed = int(seed)
    solver.parameters.num_workers = num_workers or _workers()
    for key, value in {**SOLVER_PARAMS, **(solver_params or {})}.items():
        setattr(solver.parameters, key, value)
    status = solver.solve(h.model)
    info.update({"status": _STATUS.get(status, str(status)),
                 "num_variables": len(h.model.proto.variables),
                 "num_constraints": len(h.model.proto.constraints), **h.stats,
                 "solver_time": round(solver.wall_time, 3), "hint_complete": hint_complete})
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        starts = [solver.value(v) for v in h.S]
        positions = [solver.value(v) for v in h.X]
        model_value = h.objective_scale * solver.objective_value + h.objective_offset
        bound: float | None = h.objective_scale * solver.best_objective_bound + h.objective_offset
    elif status == cp_model.UNKNOWN:
        # time limit hit before the search reported a solution (e.g. long presolve):
        # fall back to the warm start, which is a feasible solution of the same model
        starts, positions, model_value, bound = incumbent.S, incumbent.X, inc_val, None
        info["status"] = "UNKNOWN_WARM_START_RETURNED"
    else:
        info["wall_time"] = round(time.time() - t_start, 3)
        return Solution(info=info)

    order = sorted(range(inst.n), key=lambda j: (starts[j], j))
    sequence = [inst.jobs[j].id for j in order]
    info.update({
        "model_objective": round(model_value, 4),
        "model_bound": round(bound, 4) if bound is not None else None,
        "gap_pct": (round(100 * (model_value - bound) / model_value, 3)
                    if bound is not None and model_value else None),
        "model_starts": {inst.jobs[j].id: round(starts[j] * step, 3) for j in range(inst.n)},
        "model_x_cm": {inst.jobs[j].id: float(positions[j] * d.unit) for j in range(inst.n)},
        "wall_time": round(time.time() - t_start, 3),
    })
    if validate:
        ev = evaluate_sequence(inst, sequence, objective, dt=VALIDATION_DT, check=True)
        info.update({"replay_objective": round(ev.objective, 4), "replay_feasible": ev.feasible,
                     "replay_violations": len(ev.violations),
                     "replay_minus_model": round(ev.objective - model_value, 4)})
    # The P_Delta bound is reported as a lower bound of the problem. With ready_dispatch
    # the model is a *restriction* of P (not a relaxation), so its bound is not a valid
    # bound for P and is only kept in info["model_bound"].
    return Solution(sequence=sequence, lower_bound=None if ready_dispatch else bound,
                    proven_optimal=status == cp_model.OPTIMAL, info=info)


@register_method("cpsat")
def cpsat(instance: Instance, objective: str, time_limit: float, seed: int) -> Solution:
    """P_Delta of the formulation (Delta = 0.1 s), warm-started, validated by replay."""
    return solve_cpsat(instance, objective, time_limit, seed)


@register_method("cpsat_ready")
def cpsat_ready(instance: Instance, objective: str, time_limit: float, seed: int) -> Solution:
    """P_Delta + the simulator's ready-dispatch rule (model value ~ replayed value)."""
    return solve_cpsat(instance, objective, time_limit, seed, ready_dispatch=True)
