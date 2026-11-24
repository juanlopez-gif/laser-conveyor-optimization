"""Search teacher for expert iteration (strong-RL plan, phase 3).

At a decision the teacher scores each kept candidate c by a look-ahead:

1. clone the state with :meth:`FastSim.lookahead_clone`, so the real future
   arrivals and every part the controller cannot observe are dropped (online
   information only; the real cell does not see parts waiting upstream, 0.7.3). With ``scenarios > 0`` they are
   replaced by arrivals sampled from the **known** rates in a window of
   ``window_s`` seconds (``futures="poisson"``: Poisson at the mean rates, the
   teacher of every run up to 0.7.11; ``futures="cell"``: the cell's own arrival
   model, i.e. the on/off source of ``distribution: burst`` with its parameters as
   statistical knowledge, 0.7.12);
2. force c, then let the **base policy** drive the clone until every part known at the
   decision is processed;
3. cost(c) = Σ completion times of the known parts (the online counterpart of the
   mean flow time, as in ``rollout@avg_flow_time``) + ``w_pen`` · (λ_N ΔN_B + λ_T ΔT_B),
   averaged over the scenarios.

**Cost-to-go look-aheads (0.7.6, plan §4).** With ``horizon_s > 0`` and a network
base the look-ahead scores the realized cost-to-go of all parts instead of the
completion times of the known ones:

   cost(c) = Δ(WIP integral) + w_pen · (λ_N ΔN_B + λ_T ΔT_B) over the look-ahead
             + V̂(state at the cut),

averaged over ``scenarios`` sampled futures (common random numbers: every candidate
sees the same futures). This is the quantity the value head is trained on (the
realized cost-to-go of ``play_episode``), so it sees the effect of a choice on the
parts that arrive later, which the known-parts look-ahead cannot. Three cuts:

* ``horizon_s = inf``: no cut; the clone runs until it drains after the synthetic
  window ``window_s`` and V̂ = 0 (a sampled-future rollout, no value head);
* ``cut = "time"``: the clone stops at exactly ``t + horizon_s`` and V̂ is the value
  head there (the same instant for every candidate, so the realized parts are
  comparable and the value errors are correlated);
* ``cut = "decision"``: the clone stops at the first decision with ≥ 2 candidates at
  or after ``t + horizon_s`` (in-distribution states for the value head, but a
  different instant per candidate).

The synthetic window of the cut modes is ``horizon_s + 60 s``. ``scale``
(:class:`rl2.model.RunningScale`, saved with the model) denormalizes the value head.
With the rule base (no network, iteration 0) a ``horizon_s > 0`` falls back to the
known-parts look-ahead without scenarios.

The base policy is either the rule ``nearest_job`` (iteration 0, i.e. the rollout
of 0.4–0.6 on the exact simulator) or the current network (iterations ≥ 1, i.e. the
``rl_rollout`` hybrid of 0.6 with a better base each round). Rollout improves on its
base policy (Bertsekas), so each teacher should beat the network it is built on, and
the next network imitates that teacher.

**Batching.** A network-driven look-ahead would call the network once per simulated
decision (4 ms each, ``docs/methods/rl2.md`` §5). All look-aheads of one decision
(candidates × scenarios) therefore advance in lock-step: each runs until it needs
the network, then their states are evaluated in one batched forward pass.

Candidates kept: all if at most ``k``; otherwise the top ``k`` by the base policy's
preference (network logits, or travel time for the rule), always including the base
choice. The base choice is scored first and kept on ties (fortified rollout).
"""

from __future__ import annotations

import math
import random

import torch

from fastsim import FastSim
from fastsim.policies import KEY_OBSERVED, nearest_job
from rl2.features import arrival_rates, encode
from rl2.model import DispatchNet, collate
from simulation.arrivals import ArrivalRecord, ProcessingTimeSampler, _burst_times

_TIE = 1e-9

#: how the teacher samples the unknown future: Poisson at the mean rates (the
#: teacher of every run up to 0.7.11) or the cell's own arrival model (0.7.12)
FUTURES = ("poisson", "cell")


def _burst_params(cfg) -> list[tuple[float, float] | None]:
    """Per conveyor ``(on_fraction, on_mean)`` of the on/off source, ``None`` where
    the arrivals are Poisson (``on_fraction`` 1, or another distribution)."""
    a = cfg.arrivals
    if a.mode != "stochastic" or a.distribution != "burst":
        return [None] * len(cfg.conveyors)
    return [None if f >= 1.0 else (f, m) for f, m in zip(a.burst_on_fraction, a.burst_on_mean)]


def sample_scenarios(sim: FastSim, m: int, window_s: float, key: str,
                     futures: str = "poisson") -> list[list]:
    """``m`` synthetic arrival lists in (t, t + window_s].

    ``futures="poisson"``: Poisson at the known mean rates on every conveyor.
    ``futures="cell"``: the cell's arrival model; on a conveyor with an on/off source
    (``arrivals.stochastic.distribution: burst``) the future is an on/off Poisson
    source with the cell's ``on_fraction`` and ``on_mean`` (statistical knowledge of
    the process, not an observation of it), whose initial state is drawn from the
    stationary distribution (on with probability ``on_fraction``: the controller does
    not know whether the source is on right now). On a Poisson conveyor both choices
    draw exactly the same arrivals. Common random numbers: the same ``key``, instant
    and scenario index give the same future to every candidate.
    """
    if futures not in FUTURES:
        raise ValueError(f"futures must be one of {FUTURES}, not {futures!r}")
    out = []
    rates = arrival_rates(sim.cfg)
    bursts = _burst_params(sim.cfg) if futures == "cell" else [None] * len(rates)
    for s in range(m):
        rng = random.Random(f"rl2search/{key}/{sim.t:.6f}/{s}")
        sampler = ProcessingTimeSampler(sim.cfg, rng)
        recs = []
        for k, lam in enumerate(rates):
            if lam <= 0:
                continue
            if bursts[k] is None:
                ta = sim.t + rng.expovariate(lam)
                while ta <= sim.t + window_s:
                    recs.append(ArrivalRecord(ta, k, sampler.sample()))
                    ta += rng.expovariate(lam)
            else:
                on_fraction, on_mean = bursts[k]
                sw = random.Random(f"rl2search/{key}/{sim.t:.6f}/{s}/sw/{k}")
                for t in _burst_times(1.0 / lam, on_fraction, on_mean, window_s, rng, sw):
                    recs.append(ArrivalRecord(sim.t + t, k, sampler.sample()))
        out.append(recs)
    return out


#: synthetic-arrival window beyond the truncation horizon (value-truncated mode)
WINDOW_MARGIN_S = 60.0


class _Look:
    __slots__ = ("sim", "known", "done", "nb0", "tb0", "w0", "cand", "sc", "t_stop", "stopped")

    def __init__(self, sim, known, cand, sc, t_stop=None):
        self.sim, self.known, self.cand, self.sc = sim, known, cand, sc
        self.done = False
        self.nb0, self.tb0, self.w0 = sim.n_block_events, sim.blocked_area, sim.wip_area
        self.t_stop = t_stop      # None: run until the known parts are done
        self.stopped = False      # cut at a decision at or after t_stop


def _known_done(lk: _Look) -> bool:
    pC = lk.sim.pC
    return all(pC[j] is not None for j in lk.known)


@torch.inference_mode()
def run_lookaheads(looks: list[_Look], net: DispatchNet | None, rule=nearest_job) -> None:
    """Advance every look-ahead until its known parts are processed, or (``t_stop``
    set) until the first decision with ≥ 2 candidates at or after ``t_stop``
    (lock-step batching of the network calls)."""
    active = [lk for lk in looks if not lk.done]
    while active:
        need = []
        for lk in active:
            s = lk.sim
            while True:
                if lk.t_stop is None and _known_done(lk):
                    lk.done = True
                    break
                if not s.run_until_decision():
                    lk.done = True
                    break
                c = s.candidates()
                if lk.t_stop is not None and len(c) >= 2 and s.t >= lk.t_stop - _TIE:
                    lk.done = lk.stopped = True
                    break
                if len(c) == 1:
                    s.apply(c[0])
                elif net is None:
                    s.apply(rule(s, c))
                else:
                    need.append(lk)
                    break
        if need:
            obs = [encode(lk.sim) for lk in need]
            logits, _ = net(collate(obs))
            idx = logits.argmax(-1).tolist()
            for lk, o, i in zip(need, obs, idx):
                lk.sim.apply(o.part_ids[i])
        active = [lk for lk in active if not lk.done]


class SearchTeacher:
    """Rollout teacher over a base policy (rule or network)."""

    def __init__(self, net: DispatchNet | None = None, k: int = 6, scenarios: int = 0,
                 window_s: float = 30.0, w_pen: float = 0.0, key: str = "0",
                 horizon_s: float = 0.0, scale=None, cut: str = "time", tie_key=None,
                 futures: str = "poisson"):
        if cut not in ("time", "decision"):
            raise ValueError(f"cut must be 'time' or 'decision', not {cut!r}")
        if futures not in FUTURES:
            raise ValueError(f"futures must be one of {FUTURES}, not {futures!r}")
        self.cut, self.futures = cut, futures
        self.net = None if net is None else net.eval()
        if net is None and horizon_s > 0:
            # value truncation needs the value head: with the rule base (iteration 0)
            # fall back to the known-parts look-ahead, without the sampled futures that
            # were requested for the truncated mode (they hurt that objective, §6.2)
            horizon_s, scenarios = 0.0, 0
        self.k, self.m, self.window, self.w_pen, self.key = k, scenarios, window_s, w_pen, key
        self.tie_key = KEY_OBSERVED if tie_key is None else tie_key   # rule tie-break (0.7.26)
        self.horizon = float(horizon_s)
        self.scale = scale
        self.last_costs: dict[int, float] = {}
        self.lookaheads = 0

    @property
    def truncated(self) -> bool:
        return self.horizon > 0

    @torch.inference_mode()
    def _values(self, looks: list[_Look], max_time: float) -> list[float]:
        """Denormalized value-head estimate of the cost-to-go at the cut of each
        look-ahead: 0 for a clone that drained (nothing left of the synthetic
        future) or reached the episode's time limit."""
        out = [0.0] * len(looks)
        idx = [i for i, lk in enumerate(looks)
               if lk.stopped and lk.sim.t < max_time - _TIE]
        if not idx:
            return out
        obs = [encode(looks[i].sim, looks[i].sim.candidates()) for i in idx]
        _, val = self.net(collate(obs))
        for i, v in zip(idx, val.tolist()):
            out[i] = float(self.scale.denormalize(v)) if self.scale is not None else v
        return out

    @torch.inference_mode()
    def preference(self, sim: FastSim, cands: list[int]) -> list[int]:
        """Candidates ordered by the base policy's preference (best first)."""
        if self.net is None:
            first = nearest_job(sim, cands, key=self.tie_key)
            rest = sorted((j for j in cands if j != first), key=lambda j: self.tie_key(sim, j))
            return [first] + rest
        obs = encode(sim, cands)
        logits, _ = self.net(collate([obs]))
        order = torch.argsort(logits[0], descending=True).tolist()
        return [obs.part_ids[i] for i in order if obs.cand[i]]

    def costs(self, sim: FastSim, cands: list[int]) -> dict[int, float]:
        """Look-ahead cost of every kept candidate (base choice first in the dict)."""
        pref = self.preference(sim, cands)
        kept = pref[: self.k] if self.k else pref
        known = [j for j in range(len(sim.pk)) if sim.pC[j] is None and sim.observable(j)]
        trunc = self.truncated
        finite = trunc and math.isfinite(self.horizon)
        window = self.horizon + WINDOW_MARGIN_S if finite else self.window
        scen = (sample_scenarios(sim, self.m, window, self.key, self.futures)
                if self.m > 0 else [[]])
        t_stop = None
        if finite and self.cut == "decision":
            t_stop = sim.t + self.horizon
        elif trunc:
            t_stop = math.inf                  # run until the clone ends
        looks = []
        for c in kept:
            for si, sc in enumerate(scen):
                cl = sim.lookahead_clone(sc, window)
                if finite and self.cut == "time":
                    cl.max_time = min(cl.max_time, sim.t + self.horizon)
                lk = _Look(cl, known, c, si, t_stop)
                cl.apply(c)
                looks.append(lk)
        run_lookaheads(looks, self.net)
        if finite and self.cut == "time":      # the clone stopped at its time limit
            for lk in looks:
                lk.stopped = lk.sim.t >= sim.t + self.horizon - _TIE
        self.lookaheads += len(looks)
        values = self._values(looks, sim.max_time) if trunc else None
        lam_n = sim.cfg.blocking_penalty_per_event
        lam_t = sim.cfg.blocking_penalty_per_second
        tot = {c: 0.0 for c in kept}
        for i, lk in enumerate(looks):
            s = lk.sim
            if trunc:
                v = (s.wip_area - lk.w0) + values[i]
            else:
                v = sum(s.pC[j] for j in known)
            if self.w_pen:
                v += self.w_pen * (lam_n * (s.n_block_events - lk.nb0)
                                   + lam_t * (s.blocked_area - lk.tb0))
            tot[lk.cand] += v
        return {c: tot[c] / len(scen) for c in kept}

    def choose(self, costs: dict[int, float]) -> int:
        best, bc = None, math.inf
        for c, v in costs.items():          # base choice first: kept on ties
            if v < bc - _TIE:
                best, bc = c, v
        return best

    def __call__(self, sim: FastSim, cands: list[int]) -> int:
        if len(cands) == 1:
            return cands[0]
        self.last_costs = self.costs(sim, cands)
        return self.choose(self.last_costs)


def soft_target(costs: dict[int, float], temperature: float) -> dict[int, float]:
    """Softmin over the searched candidates (unsearched ones get 0)."""
    cmin = min(costs.values())
    if temperature <= 0:
        best = min(costs, key=costs.get)
        return {c: float(c == best) for c in costs}
    w = {c: math.exp(-(v - cmin) / temperature) for c, v in costs.items()}
    z = sum(w.values())
    return {c: x / z for c, x in w.items()}
