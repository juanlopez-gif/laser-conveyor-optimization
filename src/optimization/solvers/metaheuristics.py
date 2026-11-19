"""Permutation metaheuristics over the decoder Phi(pi) (simulator as exact evaluator).

Registered methods
------------------
* ``"sa"``  — simulated annealing with a restricted (time-local) neighbourhood;
* ``"ils"`` — iterated local search: first-improvement descent over the restricted
  neighbourhood (closest moves first) + random-insertion perturbation.

Both methods share the machinery of this module:

* **Warm start** from the realised orders of the online policies ``fifo``,
  ``nearest_job``, ``round_robin`` and ``weighted_score`` (the best one is the
  starting point; all of them are kept as fall-back answers, so the result is never
  worse than the best policy after validation at ``dt = 0.01``).
* **Restricted neighbourhood**: swap, insertion and block (segment) moves where
  every part moves at most ``window`` positions. Moving a part far from its
  arrival rank is almost always infeasible (deadlock) or bad.
* **Repair instead of rejection**: when the decoder deadlocks because the next
  part of the order is upstream of a conveyor blocked by an unprocessed part B,
  B is moved to just before that part and the evaluation resumes. This turns most
  infeasible neighbours (out-of-belt-order on one conveyor) into feasible ones.
* **Incremental evaluation**: the evaluator keeps snapshots of the simulation of
  the *current* sequence after every ``stride`` assigned parts. A neighbour that
  shares the first ``i`` positions with it is simulated only from the last snapshot
  ``<= i`` (the decoder state after assigning ``pi[:i]`` depends on that prefix
  only), which roughly halves the cost of an evaluation. The result is identical to
  a full simulation (tested).
* **Memoisation** of every evaluated sequence, ``SEARCH_DT`` during the search and
  a strict wall-clock deadline that also interrupts a running simulation.

Determinism: for a fixed ``seed`` and evaluation budget (``max_evals``) the search
trajectory is deterministic. When the wall-clock ``time_limit`` is binding the
result depends, unavoidably, on how many evaluations the machine completes.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from optimization.evaluate import (SEARCH_DT, VALIDATION_DT, evaluate_policy,
                                   evaluate_sequence, objective_value)
from optimization.instances import Instance, instance_config
from optimization.solvers import Solution, register_method
from policies.fixed_sequence import FixedSequencePolicy, InfeasibleSequence
from simulation.conveyor import STOP_BLOCKED
from simulation.engine import Simulation

#: online policies whose realised orders seed the search
WARM_START_POLICIES: tuple[str, ...] = ("fifo", "nearest_job", "round_robin", "weighted_score")

INF = math.inf
_IMPROVE_EPS = 1e-9


class SearchTimeout(Exception):
    """Raised when the search deadline is reached (also inside a simulation)."""


class _NullEventLog:
    """Event log that discards everything (search runs need no event log)."""

    events: list = []

    def log(self, *args, **kwargs) -> None:   # noqa: D401 - intentionally empty
        return None


# ---------------------------------------------------------------- evaluator
@dataclass
class EvalResult:
    """Outcome of evaluating one (possibly repaired) sequence."""

    value: float                 # objective at the search dt (inf if infeasible)
    sequence: tuple[str, ...]    # sequence actually simulated (after repairs)
    repairs: int = 0


@dataclass
class EvaluatorStats:
    evaluations: int = 0         # simulations started (incl. resumed ones)
    cache_hits: int = 0
    repairs: int = 0
    infeasible: int = 0
    reference_builds: int = 0
    steps_simulated: int = 0
    steps_saved: int = 0         # steps skipped thanks to snapshots


class SequenceEvaluator:
    """Memoised, incremental evaluation of processing orders with deadlock repair.

    ``evaluate(seq)`` returns the objective of ``Phi(seq)`` exactly as
    :func:`optimization.evaluate.evaluate_sequence` would at the same ``dt``
    (without the formulation check), except that an order that deadlocks is
    repaired (up to ``max_repairs`` times) and the repaired order is returned.
    """

    def __init__(self, inst: Instance, objective: str | dict, dt: float = SEARCH_DT,
                 stride: int | None = None, max_repairs: int = 6,
                 deadline: float | None = None) -> None:
        self.inst = inst
        self.objective = objective
        self.dt = dt
        self.n = inst.n
        self.stride = stride or max(2, round(self.n / 10))
        self.max_repairs = max_repairs
        self.deadline = deadline
        horizon = max(j.time for j in inst.jobs) + 60.0 * inst.n + 300.0   # as evaluate_sequence
        self.cfg = instance_config(inst, "fixed_sequence", {"sequence": list(inst.ids)}, dt,
                                   [f"simulation.max_time={horizon}"])
        self.cache: dict[tuple[str, ...], EvalResult] = {}
        self.stats = EvaluatorStats()
        self.reference: tuple[str, ...] | None = None
        self.snapshots: dict[int, Simulation] = {}     # position -> state after pi[:position]

    # ------------------------------------------------------------ simulation
    def _fresh(self, seq: Sequence[str]) -> Simulation:
        sim = Simulation(self.cfg, policy=FixedSequencePolicy(sequence=list(seq)),
                         record_trace=False)
        sim.log = _NullEventLog()
        return sim

    def _clone(self, sim: Simulation) -> Simulation:
        return copy.deepcopy(sim, memo={id(self.cfg): self.cfg})

    def _start_state(self, seq: Sequence[str]) -> Simulation:
        """Simulation positioned at the deepest valid snapshot for ``seq``."""
        if self.reference is None or not self.snapshots:
            return self._fresh(seq)
        prefix = 0
        for a, b in zip(seq, self.reference):
            if a != b:
                break
            prefix += 1
        best = max((p for p in self.snapshots if p <= prefix), default=0)
        if best == 0:
            return self._fresh(seq)
        sim = self._clone(self.snapshots[best])
        sim.policy.sequence = list(seq)
        self.stats.steps_saved += sim.step_index
        return sim

    def _run(self, sim: Simulation, capture: dict[int, Simulation] | None = None) -> None:
        """Step ``sim`` to the end; optionally capture snapshots at stride positions."""
        steps = 0
        last_pos = sim.policy.position
        while not sim.finished:
            sim.step()
            steps += 1
            if capture is not None and sim.policy.position != last_pos:
                last_pos = sim.policy.position
                if last_pos % self.stride == 0 and last_pos < self.n and last_pos not in capture:
                    capture[last_pos] = self._clone(sim)
            if steps % 400 == 0 and self.deadline is not None and time.time() > self.deadline:
                self.stats.steps_simulated += steps
                raise SearchTimeout
        self.stats.steps_simulated += steps

    def _value(self, sim: Simulation) -> float:
        if sim.n_completed != self.n:
            return INF
        return objective_value(sim.result().summary, self.objective)

    @staticmethod
    def _blocker(sim: Simulation) -> str | None:
        """Id of the unprocessed part blocking the conveyor of the awaited part."""
        target = sim.policy.next_id()
        for conv in sim.conveyors:
            ids = [p.id for p in conv.buffer] + [p.id for p in conv.parts]
            if target in ids and STOP_BLOCKED in conv.stop_reasons:
                bp = conv.blocking_part(sim.block)
                return bp.id if bp is not None else None
        return None

    # ---------------------------------------------------------------- public
    def evaluate(self, seq: Sequence[str]) -> EvalResult:
        """Objective of ``seq`` (repairing deadlocks). Memoised."""
        key = tuple(seq)
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]
        current, repairs, visited = key, 0, [key]
        while True:
            if current in self.cache:               # repaired into a known sequence
                res = self.cache[current]
                res = EvalResult(res.value, res.sequence, res.repairs + repairs)
                break
            self.stats.evaluations += 1
            sim = self._start_state(current)
            try:
                self._run(sim)
            except InfeasibleSequence:
                blocker = self._blocker(sim)
                pos = sim.policy.position
                if blocker is None or repairs >= self.max_repairs:
                    res = EvalResult(INF, current, repairs)
                    self.stats.infeasible += 1
                    break
                lst = [p for p in current if p != blocker]
                lst.insert(pos, blocker)
                current = tuple(lst)
                repairs += 1
                self.stats.repairs += 1
                visited.append(current)
                continue
            res = EvalResult(self._value(sim), current, repairs)
            if res.value == INF:
                self.stats.infeasible += 1
            break
        for k in visited:
            self.cache.setdefault(k, res)
        return res

    def set_reference(self, seq: Sequence[str]) -> None:
        """Make ``seq`` (a feasible, evaluated order) the base of incremental evaluation."""
        seq = tuple(seq)
        if seq == self.reference:
            return
        if self.reference is not None:            # keep snapshots of the common prefix
            prefix = 0
            for a, b in zip(seq, self.reference):
                if a != b:
                    break
                prefix += 1
            kept = {p: s for p, s in self.snapshots.items() if p <= prefix}
        else:
            kept = {}
        self.reference, self.snapshots = seq, kept
        start = max(kept, default=0)
        sim = self._clone(kept[start]) if start else self._fresh(seq)
        sim.policy.sequence = list(seq)
        self.stats.reference_builds += 1
        try:
            self._run(sim, capture=self.snapshots)
        except InfeasibleSequence:                # should not happen for an evaluated order
            pass


# --------------------------------------------------------------- neighbourhood
def random_move(seq: Sequence[str], rng: random.Random, window: int) -> list[str]:
    """Random restricted move: swap / insertion / block insertion within ``window`` positions."""
    n = len(seq)
    s = list(seq)
    if n < 2:
        return s
    r = rng.random()
    i = rng.randrange(n)
    # displacement d in 1..window with probability ~ 1/d (close moves are preferred)
    d = rng.choices(range(1, max(1, window) + 1),
                    weights=[1.0 / k for k in range(1, max(1, window) + 1)])[0]
    j = min(n - 1, max(0, i + (d if rng.random() < 0.5 else -d)))
    if j == i:
        j = i + 1 if i + 1 < n else i - 1
    if r < 0.4:                                    # swap
        s[i], s[j] = s[j], s[i]
    elif r < 0.85 or n < 4:                        # insertion: move one part
        s.insert(j, s.pop(i))
    else:                                          # block (segment of 2-3) insertion
        length = rng.randint(2, 3)
        a = min(i, n - length)
        block = s[a:a + length]
        del s[a:a + length]
        target = min(len(s), max(0, a + (d if rng.random() < 0.5 else -d)))
        s[target:target] = block
    return s


def neighbourhood(seq: Sequence[str], window: int, rng: random.Random) -> list[list[str]]:
    """All swap and insertion neighbours with displacement <= window.

    Ordered by displacement (adjacent moves first), shuffled within each distance.
    Duplicates are removed (an adjacent swap equals an adjacent insertion).
    """
    n = len(seq)
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = {tuple(seq)}
    for d in range(1, window + 1):
        bucket: list[list[str]] = []
        for i in range(n - d):
            j = i + d
            sw = list(seq)
            sw[i], sw[j] = sw[j], sw[i]
            fw = list(seq)
            fw.insert(j, fw.pop(i))                # move part i forward (later)
            bw = list(seq)
            bw.insert(i, bw.pop(j))                # move part j backward (earlier)
            for cand in (sw, fw, bw):
                t = tuple(cand)
                if t not in seen:
                    seen.add(t)
                    bucket.append(cand)
        rng.shuffle(bucket)
        out.extend(bucket)
    return out


# ------------------------------------------------------------------ framework
@dataclass
class SearchContext:
    """Shared state of one metaheuristic run (budget, incumbents, elite pool)."""

    inst: Instance
    objective: str | dict
    rng: random.Random
    evaluator: SequenceEvaluator
    search_deadline: float
    max_evals: int | None
    best_value: float = INF
    best_seq: tuple[str, ...] = ()
    elite: dict[tuple[str, ...], float] = field(default_factory=dict)
    trace: list[tuple[int, float]] = field(default_factory=list)
    requests: int = 0                                 # evaluations + cache hits
    window: int = 1                                   # max displacement of a move
    time_limit: float = 30.0
    extra: dict = field(default_factory=dict)         # method-specific statistics

    def budget_left(self) -> bool:
        if time.time() >= self.search_deadline:
            return False
        # max_evals counts every request (cache hits too): on tiny instances the whole
        # neighbourhood ends up cached and the search would otherwise spin until the deadline
        if self.max_evals is not None and self.requests >= self.max_evals:
            return False
        return True

    def evaluate(self, seq: Sequence[str]) -> EvalResult:
        if not self.budget_left():
            raise SearchTimeout
        self.requests += 1
        res = self.evaluator.evaluate(seq)
        if res.value < INF:
            self.elite[res.sequence] = res.value
            if res.value < self.best_value - _IMPROVE_EPS:
                self.best_value, self.best_seq = res.value, res.sequence
                self.trace.append((self.evaluator.stats.evaluations, res.value))
        return res


def default_window(n: int) -> int:
    """Maximum displacement of a part in one move (3 conveyors -> a few parts in the zone)."""
    return max(1, min(6, n - 1))


def _warm_starts(inst: Instance, objective: str | dict) -> tuple[dict[str, tuple[str, ...]],
                                                                 dict[str, float], float]:
    """Realised orders and validated objective (dt=0.01) of the online policies."""
    seqs, vals = {}, {}
    t0 = time.time()
    for name in WARM_START_POLICIES:
        ev = evaluate_policy(inst, name, objective=objective)
        seqs[name], vals[name] = tuple(ev.sequence), (ev.objective if ev.feasible else INF)
    return seqs, vals, (time.time() - t0) / len(WARM_START_POLICIES)


def run_metaheuristic(inst: Instance, objective: str | dict, time_limit: float, seed: int,
                      core: Callable[[SearchContext, tuple[str, ...]], None], name: str,
                      max_evals: int | None = None, window: int | None = None) -> Solution:
    """Common driver: warm start, run ``core`` under the budget, validate, fall back."""
    start = time.time()
    rng = random.Random(seed)
    if inst.n <= 1:
        return Solution(sequence=list(inst.ids), info={"method": name, "trivial": True})
    seqs, pol_vals, val_cost = _warm_starts(inst, objective)
    best_pol = min(pol_vals, key=lambda k: pol_vals[k])
    # reserve time for validating up to two candidates at dt = 0.01 (+5 % margin)
    reserve = 2.2 * val_cost + 0.05 * time_limit
    search_deadline = start + time_limit - reserve
    evaluator = SequenceEvaluator(inst, objective, SEARCH_DT, deadline=search_deadline)
    ctx = SearchContext(inst, objective, rng, evaluator, search_deadline, max_evals)
    ctx.window = window or default_window(inst.n)
    ctx.time_limit = time_limit

    timed_out = False
    try:
        starts = {}
        for pname in WARM_START_POLICIES:                     # deterministic order
            res = ctx.evaluate(seqs[pname])
            starts[pname] = res.value
        initial = min((r for r in ctx.elite), key=lambda s: ctx.elite[s], default=None)
        if initial is not None:
            core(ctx, initial)
    except SearchTimeout:
        timed_out = True
    search_time = time.time() - start

    # ---- validation at dt = 0.01 with the formulation checker
    deadline = start + time_limit
    candidates = sorted(ctx.elite, key=lambda s: ctx.elite[s])[:3]
    best_seq, best_val, src = seqs[best_pol], pol_vals[best_pol], f"policy:{best_pol}"
    validated = 0
    for cand in candidates:
        if validated and time.time() + 1.2 * val_cost > deadline:
            break
        if list(cand) == list(best_seq):
            continue
        ev = evaluate_sequence(inst, list(cand), objective, dt=VALIDATION_DT, check=True)
        validated += 1
        if ev.feasible and ev.objective < best_val - _IMPROVE_EPS:
            best_seq, best_val, src = cand, ev.objective, "search"
        if time.time() + 1.2 * val_cost > deadline:
            break
    st = evaluator.stats
    info = {
        "method": name, "seed": seed, "value": best_val, "source": src,
        "search_best_value_dt_search": ctx.best_value,
        "policy_values": pol_vals, "best_policy": best_pol,
        "improvement_vs_best_policy_pct": (100 * (pol_vals[best_pol] - best_val) / pol_vals[best_pol]
                                           if math.isfinite(pol_vals[best_pol]) and pol_vals[best_pol]
                                           else None),
        "evaluations": st.evaluations, "cache_hits": st.cache_hits, "repairs": st.repairs,
        "infeasible": st.infeasible, "reference_builds": st.reference_builds,
        "steps_saved_pct": round(100 * st.steps_saved / max(1, st.steps_saved + st.steps_simulated), 1),
        "timed_out": timed_out, "search_time_s": round(search_time, 3),
        "validated": validated, "window": ctx.window,
        "improvements": ctx.trace[-10:],
    }
    info.update(ctx.extra)
    return Solution(sequence=list(best_seq), info=info)


# ------------------------------------------------------- simulated annealing
def _sa_core(ctx: SearchContext, initial: tuple[str, ...]) -> None:
    """Simulated annealing with an evaluation-count geometric schedule and reheats."""
    rng, ev, n = ctx.rng, ctx.evaluator, ctx.inst.n
    window = ctx.window
    cur, f_cur = initial, ctx.elite[initial]
    ev.set_reference(cur)

    # Calibrate T0: a lower-quartile worsening move is accepted with probability 0.2.
    # (Random moves are disruptive: the median worsening is far larger than the gains
    # available, and a median-based T0 turns the search into a random walk.)
    deltas = []
    for _ in range(min(12, 2 * n)):
        r = ctx.evaluate(random_move(cur, rng, window))
        if math.isfinite(r.value) and r.value > f_cur:
            deltas.append(r.value - f_cur)
    if ctx.best_value < f_cur - _IMPROVE_EPS:                # calibration found something
        cur, f_cur = ctx.best_seq, ctx.best_value
        ev.set_reference(cur)
    deltas.sort()
    t0 = (deltas[len(deltas) // 4] / math.log(5)) if deltas else max(1e-3, 0.002 * abs(f_cur))

    # Evaluation budget of one annealing cycle. It is derived from a fixed cost model
    # (~4 ms per part per evaluation at SEARCH_DT), not from measured machine time, so the
    # trajectory stays deterministic; about two cycles fit in the time limit.
    est_evals = ctx.time_limit / (0.004 * n)
    cycle = int(min(max(60, est_evals / 2), 40 * n))
    t_end = t0 * 1e-3
    alpha = (t_end / t0) ** (1.0 / cycle)
    temp, k_cycle, cycles, accepted, rejected_in_row = t0, 0, 1, 0, 0
    while True:
        cand = random_move(cur, rng, window)
        r = ctx.evaluate(cand)
        k_cycle += 1
        if math.isfinite(r.value):
            delta = r.value - f_cur
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                if r.sequence != cur:
                    cur, f_cur = r.sequence, r.value
                    accepted += 1
                    rejected_in_row = -1
        rejected_in_row += 1
        # Rebuilding the snapshots costs about one evaluation: do it lazily, once the
        # current order has survived a few rejected neighbours (low acceptance phase).
        if rejected_in_row == 3:
            ev.set_reference(cur)
        temp *= alpha
        if k_cycle >= cycle:                                   # reheat from the best
            cycles += 1
            k_cycle, temp = 0, t0 / (2 ** min(cycles, 4))
            cur, f_cur = ctx.best_seq, ctx.best_value
            ev.set_reference(cur)
        ctx.extra = {"sa_T0": t0, "sa_cycle_evals": cycle,
                     "sa_cycles": cycles, "sa_accepted": accepted}


# ---------------------------------------------------------- iterated local search
def _local_search(ctx: SearchContext, seq: tuple[str, ...], f: float) -> tuple[tuple[str, ...], float]:
    """First-improvement descent over the restricted neighbourhood (closest moves first)."""
    window = ctx.window
    ctx.evaluator.set_reference(seq)
    improved = True
    while improved:
        improved = False
        for cand in neighbourhood(seq, window, ctx.rng):
            r = ctx.evaluate(cand)
            if r.value < f - _IMPROVE_EPS:
                seq, f = r.sequence, r.value
                ctx.evaluator.set_reference(seq)
                improved = True
                break
    return seq, f


def _perturb(seq: Sequence[str], rng: random.Random, window: int, strength: int) -> list[str]:
    """``strength`` random insertion moves inside one local region of the order."""
    s = list(seq)
    n = len(s)
    centre = rng.randrange(n)
    for _ in range(strength):
        i = min(n - 1, max(0, centre + rng.randint(-window, window)))
        j = min(n - 1, max(0, i + rng.randint(-window, window)))
        s.insert(j, s.pop(i))
    return s


def _ils_core(ctx: SearchContext, initial: tuple[str, ...]) -> None:
    """Iterated local search: descent, local perturbation, 'better-or-near-best' acceptance."""
    rng, window = ctx.rng, ctx.window
    cur, f_cur = _local_search(ctx, initial, ctx.elite[initial])
    strength, iters, stall = 2, 0, 0
    while stall < 200:                     # 200 kicks without a new evaluation: space exhausted
        iters += 1
        ctx.extra = {"ils_iterations": iters}
        before = ctx.evaluator.stats.evaluations
        cand = ctx.evaluate(_perturb(cur, rng, window, strength))
        if not math.isfinite(cand.value):
            stall = stall + 1 if ctx.evaluator.stats.evaluations == before else 0
            continue
        new, f_new = _local_search(ctx, cand.sequence, cand.value)
        if f_new < f_cur - _IMPROVE_EPS:
            cur, f_cur, strength = new, f_new, 2
        elif f_new <= ctx.best_value * 1.002:                  # sideways / near-best walk
            cur, f_cur = new, f_new
            strength = min(strength + 1, 6)
        else:                                                  # restart from the best
            cur, f_cur = ctx.best_seq, ctx.best_value
            strength = min(strength + 1, 6)
        stall = stall + 1 if ctx.evaluator.stats.evaluations == before else 0


# ------------------------------------------------------------------ public API
def simulated_annealing(instance: Instance, objective: str | dict = "makespan",
                        time_limit: float = 30.0, seed: int = 0, max_evals: int | None = None,
                        window: int | None = None) -> Solution:
    """Simulated annealing over processing orders (see module docstring)."""
    return run_metaheuristic(instance, objective, time_limit, seed, _sa_core, "sa",
                             max_evals, window)


def iterated_local_search(instance: Instance, objective: str | dict = "makespan",
                          time_limit: float = 30.0, seed: int = 0, max_evals: int | None = None,
                          window: int | None = None) -> Solution:
    """Iterated local search over processing orders (see module docstring)."""
    return run_metaheuristic(instance, objective, time_limit, seed, _ils_core, "ils",
                             max_evals, window)


@register_method("sa")
def sa_method(instance: Instance, objective: str | dict, time_limit: float, seed: int) -> Solution:
    return simulated_annealing(instance, objective, time_limit, seed)


@register_method("ils")
def ils_method(instance: Instance, objective: str | dict, time_limit: float, seed: int) -> Solution:
    return iterated_local_search(instance, objective, time_limit, seed)
