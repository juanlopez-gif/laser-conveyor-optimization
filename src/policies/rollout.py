"""Rollout policy: one-step look-ahead with a base policy (Bertsekas' rollout).

At every decision epoch (laser idle, at least one READY part) the policy

1. ranks the candidates by laser travel time and keeps the best
   ``max_candidates`` (the base policy's own choice is always kept, which makes
   the rollout *fortified*: it can always fall back on the base decision);
2. for every kept candidate ``c`` clones the running simulation, forces ``c``
   as the next assignment and lets the **base policy** (default
   ``nearest_job``) drive the clone until every part *known at decision time*
   is processed (or for ``horizon_s`` simulated seconds);
3. picks the candidate whose look-ahead has the smallest cost (default: average
   flow time of the known parts, i.e. the online counterpart of
   ``avg_flow_time``). Penalty-aware look-ahead objectives are available
   (``penalized_flow_time``...), but in the 30-seed study of 0.4.0 they were
   WORSE even on the penalized metric (docs/CHANGELOG.md, 0.4.0 §4).

If the look-ahead were exact (no future arrivals, same ``dt``, untruncated
horizon) and the base policy is a state-feedback rule (``nearest_job`` is),
the cost-improvement property of rollout guarantees that the rollout run is
never worse than the base run for the look-ahead objective. In the real cell
future arrivals are unknown, so the guarantee holds only for the parts present
at each decision; see docs/methods/online_rollout.md.

Fairness (online information only)
----------------------------------
The clone is built with :func:`copy.deepcopy` and then:

* ``pending_arrivals`` (the pre-computed list of FUTURE arrivals) is replaced by
  an empty deque before anything runs, so the clone never sees a part that has
  not arrived yet. Optionally (``future_samples > 0``) it is filled with
  *synthetic* arrivals sampled from the known arrival rates, never from the
  real future list;
* the event log is replaced by a no-op log, the trace recorder by ``None`` and
  the queue series by a fresh list (nothing is written into the original);
* the configuration and its frozen specs are shared (immutable), everything
  else (parts, conveyors, laser, metrics) is a private copy;
* the clone's policy is the base policy (never rollout: no recursion).

Forcing the first decision reuses the engine itself: the clone is taken in the
middle of step 5 (dispatch) of the original, after steps 1-4 have run. Steps
1-4 are idempotent in that state (laser idle, no due arrival left, admission
already done), so the clone simply re-executes the whole step with a policy
that returns the forced part first. The test suite checks that this reproduces
the base-policy run bit for bit.
"""

from __future__ import annotations

import copy
import math
import random
import time as _time
from collections import deque
from typing import Any, Mapping, Sequence

from simulation.arrivals import ArrivalRecord, ProcessingTimeSampler
from simulation.engine import Simulation
from simulation.events import EventLog
from simulation.part import Part

from . import get_policy, register_policy
from .base_policy import BasePolicy, DecisionContext

#: Objectives understood by the look-ahead (same names as optimization.OBJECTIVES).
LOOKAHEAD_OBJECTIVES = ("avg_flow_time", "avg_waiting_time", "makespan", "max_lateness",
                        "total_laser_travel", "total_conveyor_stop_time",
                        "blocking_events", "total_blocked_time", "blocking_penalty",
                        "penalized_flow_time", "penalized_makespan")

_TIE_TOL = 1e-9


def _settled(part: Part) -> bool:
    """Processed, or gone for good (``pass`` mode: exited unprocessed)."""
    return part.completion_time is not None or part.exit_time is not None


class _NullEventLog(EventLog):
    """Event log that discards everything (the clone must not log anywhere)."""

    def log(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - intentionally empty
        return None


class _LookaheadSimulation(Simulation):
    """A cloned simulation whose clock starts at ``_t0`` and may use another ``dt``.

    The engine defines ``time = step_index * dt``; changing ``dt`` in a clone
    would break that identity, so the clone measures time from the decision
    instant instead. Only used on deep copies, never instantiated directly.
    """

    _t0: float = 0.0
    _k0: int = 0

    @property
    def time(self) -> float:  # type: ignore[override]
        return self._t0 + (self.step_index - self._k0) * self.dt


class _ForcedFirstPolicy(BasePolicy):
    """Returns the part ``forced_id`` at the first decision, then defers to ``base``."""

    name = "rollout_forced"

    def __init__(self, forced_id: str, base: BasePolicy) -> None:
        super().__init__()
        self.forced_id: str | None = forced_id
        self.base = base

    def select(self, candidates: Sequence[Part], ctx: DecisionContext) -> Part | None:
        if self.forced_id is not None:
            target, self.forced_id = self.forced_id, None
            for p in candidates:
                if p.id == target:
                    return p
            raise RuntimeError(f"rollout: forced part {target} is not a candidate in the clone")
        return self.base.select(candidates, ctx)


@register_policy("rollout")
class RolloutPolicy(BasePolicy):
    """One-step look-ahead over a base policy, using only information available online.

    Parameters (``policy.params.rollout.<name>`` in the config):

    base_policy:   name of the registered policy simulated after the forced choice
                   (default ``nearest_job``; ``rollout`` itself is refused).
    base_params:   parameters of the base policy (default ``{}``).
    max_candidates: candidates evaluated per decision, best by travel time first
                   (default 6; the base choice is always included; ``None`` = all).
    horizon_s:     cap on the simulated look-ahead [s] (default ``None`` = until all
                   known parts are processed, bounded by ``simulation.max_time``).
    dt_lookahead:  time step of the clone [s] (default 0.05; ``None`` = same dt as
                   the running simulation, which makes the look-ahead exact).
    objective:     look-ahead cost: one of :data:`LOOKAHEAD_OBJECTIVES` or a dict of
                   weights over them (default ``avg_flow_time``; the blocking
                   objectives use the weights of ``objective.blocking_penalty``).
    future_samples: number of synthetic future-arrival scenarios averaged per
                   candidate (default 0 = look-ahead with known parts only).
    future_window_s: length of the window in which synthetic arrivals are sampled.
    """

    def __init__(self, base_policy: str = "nearest_job", base_params: Mapping | None = None,
                 max_candidates: int | None = 6, horizon_s: float | None = None,
                 dt_lookahead: float | None = 0.05,
                 objective: str | Mapping[str, float] = "avg_flow_time",
                 future_samples: int = 0, future_window_s: float = 30.0, **params: Any) -> None:
        super().__init__(base_policy=base_policy, base_params=dict(base_params or {}),
                         max_candidates=max_candidates, horizon_s=horizon_s,
                         dt_lookahead=dt_lookahead, objective=objective,
                         future_samples=future_samples, future_window_s=future_window_s,
                         **params)
        if base_policy == "rollout":
            raise ValueError("rollout cannot use itself as base policy (infinite recursion)")
        self.base_name = base_policy
        self.base_params = dict(base_params or {})
        get_policy(self.base_name, self.base_params)        # fail fast on unknown names
        self.max_candidates = None if max_candidates is None else int(max_candidates)
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1 (or None)")
        self.horizon_s = None if horizon_s is None else float(horizon_s)
        self.dt_lookahead = None if dt_lookahead is None else float(dt_lookahead)
        if self.dt_lookahead is not None and self.dt_lookahead <= 0:
            raise ValueError("dt_lookahead must be > 0 (or None)")
        self.weights = self._parse_objective(objective)
        self.future_samples = int(future_samples)
        self.future_window_s = float(future_window_s)
        self.reset()

    # ------------------------------------------------------------- bookkeeping
    @staticmethod
    def _parse_objective(objective: str | Mapping[str, float]) -> dict[str, float]:
        weights = ({str(objective): 1.0} if isinstance(objective, str)
                   else {str(k): float(v) for k, v in objective.items() if float(v) != 0.0})
        unknown = [k for k in weights if k not in LOOKAHEAD_OBJECTIVES]
        if unknown or not weights:
            raise ValueError(f"rollout objective {objective!r} not supported; "
                             f"use one of {LOOKAHEAD_OBJECTIVES} or a dict of weights")
        return weights

    def reset(self) -> None:
        """Clear the statistics collected during a run."""
        self.decision_times: list[float] = []       # wall-clock seconds per decision
        self.clones_simulated = 0
        self.lookahead_steps = 0
        self.last_costs: dict[str, float] = {}

    def stats(self) -> dict[str, float]:
        """Summary of the computational effort of the last run."""
        d = self.decision_times
        return {"decisions": len(d), "clones": self.clones_simulated,
                "lookahead_steps": self.lookahead_steps,
                "decision_time_total_s": sum(d),
                "decision_time_mean_ms": 1000 * sum(d) / len(d) if d else 0.0,
                "decision_time_max_ms": 1000 * max(d) if d else 0.0}

    def _new_base(self) -> BasePolicy:
        base = get_policy(self.base_name, self.base_params)
        base.reset()
        return base

    # --------------------------------------------------------------- decision
    def select(self, candidates: Sequence[Part], ctx: DecisionContext) -> Part | None:
        start = _time.perf_counter()
        try:
            return self._select(candidates, ctx)
        finally:
            self.decision_times.append(_time.perf_counter() - start)

    def _select(self, candidates: Sequence[Part], ctx: DecisionContext) -> Part | None:
        sim = ctx.simulation
        base_choice = self._new_base().select(candidates, ctx)
        if sim is None:                       # no simulation to clone: behave as the base
            return base_choice
        ranked = sorted(candidates, key=lambda p: (ctx.travel_time(p), p.arrival_time, p.number))
        if self.max_candidates is not None:
            kept = ranked[:self.max_candidates]
            if base_choice is not None and base_choice not in kept:
                kept.append(base_choice)
        else:
            kept = ranked
        if len(kept) == 1:
            self.last_costs = {}
            return kept[0]
        # Evaluate the base choice first: on ties the base decision is kept.
        if base_choice is not None:
            kept.sort(key=lambda p: p is not base_choice)

        scenarios = self._sample_scenarios(sim) if self.future_samples > 0 else [[]]
        costs: dict[str, float] = {}
        best, best_cost = kept[0], math.inf
        for cand in kept:
            cost = sum(self.lookahead_cost(sim, cand.id, sc) for sc in scenarios) / len(scenarios)
            costs[cand.id] = cost
            if cost < best_cost - _TIE_TOL:
                best, best_cost = cand, cost
        self.last_costs = costs
        return best

    # ------------------------------------------------------------- look-ahead
    def clone(self, sim: Simulation, policy: BasePolicy) -> _LookaheadSimulation:
        """Private copy of ``sim`` for look-ahead: no future arrivals, no log, no trace.

        The original is only read. Immutable configuration objects are shared;
        every mutable object is deep-copied.
        """
        cfg = sim.cfg
        memo: dict[int, Any] = {
            id(cfg): cfg, id(cfg.geometry): cfg.geometry, id(cfg.laser): cfg.laser,
            id(sim.stop_policy): sim.stop_policy,          # stateless strategy object
            id(sim.policy): policy,                        # never copy/recurse into rollout
            id(sim.log): _NullEventLog(),
            id(sim.metrics.queue_series): [],
            id(sim.pending_arrivals): deque(),             # FAIRNESS: future is unknown
        }
        for spec in cfg.conveyors:
            memo[id(spec)] = spec
        if sim.trace is not None:
            memo[id(sim.trace)] = None
        clone = copy.deepcopy(sim, memo)
        clone.__class__ = _LookaheadSimulation
        if self.dt_lookahead is not None and self.dt_lookahead != sim.dt:
            clone.dt = self.dt_lookahead
            clone._t0, clone._k0 = sim.time, clone.step_index
        else:                        # same dt: keep time = step_index * dt bit for bit
            clone._t0, clone._k0 = 0.0, 0
        clone.policy = policy
        clone.trace = None
        clone.pending_arrivals = deque()
        clone.trace_every = 1 << 60                        # never sample queues/trace
        return clone

    def _strip_future(self, clone: Simulation, original: Simulation,
                      scenario: Sequence[ArrivalRecord]) -> None:
        """Set the clone's future arrivals: none, or a synthetic scenario.

        Hook kept separate so tests can verify that the decision changes if the
        real future WERE visible (it never is in this class).
        """
        clone.pending_arrivals = deque(scenario)

    def lookahead_cost(self, sim: Simulation, forced_id: str,
                       scenario: Sequence[ArrivalRecord] = ()) -> float:
        """Cost of: assign ``forced_id`` now, then follow the base policy."""
        t0 = sim.time
        clone = self.clone(sim, _ForcedFirstPolicy(forced_id, self._new_base()))
        self._strip_future(clone, sim, scenario)
        n_known = len(sim.parts)
        tracked = [p for p in clone.parts if p.completion_time is None and p.exit_time is None]
        stop0 = sum(clone.metrics.stopped_time)
        dist0 = clone.laser.distance
        block0 = sum(clone.metrics.stopped_blocked)
        t_end = math.inf if self.horizon_s is None else t0 + self.horizon_s
        self.clones_simulated += 1

        clone.step()                         # re-executes the current step with the forced choice
        steps, done_count = 1, -1
        while not clone.finished and clone.time < t_end - 1e-9:
            if clone.n_completed != done_count:
                done_count = clone.n_completed
                if not clone.pending_arrivals and all(
                        _settled(p) for p in tracked) and all(
                        _settled(p) for p in clone.parts[n_known:]):
                    break
            clone.step()
            steps += 1
        self.lookahead_steps += steps
        tracked += clone.parts[n_known:]     # synthetic parts (only when sampling futures)
        return self._cost(tracked, clone, clone.time, stop0, dist0, block0)

    def _cost(self, parts: Sequence[Part], clone: Simulation, t_end: float,
              stop0: float, dist0: float, block0: float = 0.0) -> float:
        """Weighted look-ahead cost. Unfinished parts count as finishing at ``t_end``
        (a lower bound on their true completion).

        Blocking terms are measured from the decision instant: N_B counts the known
        parts that reach (or already sit at) the zone end unprocessed, T_B the
        conveyor time blocked during the look-ahead. Parts already blocking add the
        same constant to every candidate, so they do not bias the choice."""
        cfg = clone.cfg
        n_block = sum(1 for p in parts if p.block_time is not None)
        t_block = sum(clone.metrics.stopped_blocked) - block0
        penalty = (cfg.blocking_penalty_per_event * n_block
                   + cfg.blocking_penalty_per_second * t_block)
        flow = ((sum((p.completion_time if p.completion_time is not None else t_end)
                     - p.arrival_time for p in parts) / len(parts)) if parts else 0.0)
        total = 0.0
        for name, w in self.weights.items():
            if name in ("blocking_events", "total_blocked_time", "blocking_penalty"):
                total += w * {"blocking_events": n_block, "total_blocked_time": t_block,
                              "blocking_penalty": penalty}[name]
                continue
            if name == "penalized_flow_time":
                total += w * (flow + (penalty / len(parts) if parts else 0.0))
                continue
            if name == "penalized_makespan":
                total += w * (max(((p.completion_time if p.completion_time is not None
                                    else t_end) for p in parts), default=0.0) + penalty)
                continue
            if name == "avg_flow_time":
                v = (sum((p.completion_time if p.completion_time is not None else t_end)
                         - p.arrival_time for p in parts) / len(parts)) if parts else 0.0
            elif name == "avg_waiting_time":
                v = (sum((p.start_processing_time if p.start_processing_time is not None
                          else t_end) - p.arrival_time for p in parts) / len(parts)) if parts else 0.0
            elif name == "makespan":
                v = max(((p.completion_time if p.completion_time is not None else t_end)
                         for p in parts), default=0.0)
            elif name == "max_lateness":
                v = max(((p.completion_time if p.completion_time is not None else t_end)
                         - p.due_date for p in parts), default=0.0)
            elif name == "total_laser_travel":
                v = clone.laser.distance - dist0
            else:  # total_conveyor_stop_time
                v = sum(clone.metrics.stopped_time) - stop0
            total += w * v
        return total

    # ------------------------------------------------------- synthetic futures
    def _sample_scenarios(self, sim: Simulation) -> list[list[ArrivalRecord]]:
        """Synthetic arrivals in ``(t, t + future_window_s]`` from the KNOWN rates.

        Rates come from the configuration (stochastic: mean inter-arrival times;
        deterministic: the nominal intervals, used as Poisson means) or, for
        explicit instances, from the arrivals observed so far. The real list of
        future arrivals is never read. The random stream depends only on the
        seed, the decision time and the scenario index, so all candidates are
        compared on the same scenarios and runs are reproducible.
        """
        cfg, t = sim.cfg, sim.time
        means = self._known_interarrival_means(sim)
        scenarios = []
        for s in range(self.future_samples):
            rng = random.Random(f"rollout/{cfg.simulation.seed}/{t:.6f}/{s}")
            sampler = ProcessingTimeSampler(cfg, rng)
            recs: list[ArrivalRecord] = []
            for k, m in enumerate(means):
                if not m or m <= 0 or not math.isfinite(m):
                    continue
                ta = t + rng.expovariate(1.0 / m)
                while ta <= t + self.future_window_s:
                    recs.append(ArrivalRecord(ta, k, sampler.sample()))
                    ta += rng.expovariate(1.0 / m)
            scenarios.append(sorted(recs, key=lambda r: (r.time, r.conveyor_index)))
        return scenarios

    @staticmethod
    def _known_interarrival_means(sim: Simulation) -> list[float | None]:
        a, n = sim.cfg.arrivals, sim.cfg.n_conveyors
        if a.mode == "stochastic":
            return list(a.mean_interarrival)
        if a.mode == "deterministic":
            return [iv if iv > 0 else None for iv in a.intervals]
        t = sim.time                         # explicit: empirical rate of what has arrived
        counts = [0] * n
        for p in sim.parts:
            counts[p.conveyor_id - 1] += 1
        return [t / c if (c > 0 and t > 0) else None for c in counts]
