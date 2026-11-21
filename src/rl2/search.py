"""Search teacher for expert iteration (strong-RL plan, phase 3).

At a decision the teacher scores each kept candidate c by a look-ahead:

1. clone the state with :meth:`FastSim.lookahead_clone`, so the real future
   arrivals and every part the controller cannot observe are dropped (online
   information only; the real cell does not see parts waiting upstream, 0.7.3). With ``scenarios > 0`` they are
   replaced by arrivals sampled from the **known** rates in a window of
   ``window_s`` seconds;
2. force c, then let the **base policy** drive the clone until every part known at the
   decision is processed;
3. cost(c) = Σ completion times of the known parts (the online counterpart of the
   mean flow time, as in ``rollout@avg_flow_time``) + ``w_pen`` · (λ_N ΔN_B + λ_T ΔT_B),
   averaged over the scenarios.

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
from fastsim.policies import nearest_job
from rl2.features import arrival_rates, encode
from rl2.model import DispatchNet, collate
from simulation.arrivals import ArrivalRecord, ProcessingTimeSampler

_TIE = 1e-9


def sample_scenarios(sim: FastSim, m: int, window_s: float, key: str) -> list[list]:
    """``m`` synthetic arrival lists in (t, t + window_s] from the known rates."""
    out = []
    rates = arrival_rates(sim.cfg)
    for s in range(m):
        rng = random.Random(f"rl2search/{key}/{sim.t:.6f}/{s}")
        sampler = ProcessingTimeSampler(sim.cfg, rng)
        recs = []
        for k, lam in enumerate(rates):
            if lam <= 0:
                continue
            ta = sim.t + rng.expovariate(lam)
            while ta <= sim.t + window_s:
                recs.append(ArrivalRecord(ta, k, sampler.sample()))
                ta += rng.expovariate(lam)
        out.append(recs)
    return out


class _Look:
    __slots__ = ("sim", "known", "done", "nb0", "tb0", "cand", "sc")

    def __init__(self, sim, known, cand, sc):
        self.sim, self.known, self.cand, self.sc = sim, known, cand, sc
        self.done = False
        self.nb0, self.tb0 = sim.n_block_events, sim.blocked_area


def _known_done(lk: _Look) -> bool:
    pC = lk.sim.pC
    return all(pC[j] is not None for j in lk.known)


@torch.inference_mode()
def run_lookaheads(looks: list[_Look], net: DispatchNet | None, rule=nearest_job) -> None:
    """Advance every look-ahead until its known parts are processed (lock-step
    batching of the network calls)."""
    active = [lk for lk in looks if not lk.done]
    while active:
        need = []
        for lk in active:
            s = lk.sim
            while True:
                if _known_done(lk) or not s.run_until_decision():
                    lk.done = True
                    break
                c = s.candidates()
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
                 window_s: float = 30.0, w_pen: float = 0.0, key: str = "0"):
        self.net = None if net is None else net.eval()
        self.k, self.m, self.window, self.w_pen, self.key = k, scenarios, window_s, w_pen, key
        self.last_costs: dict[int, float] = {}
        self.lookaheads = 0

    @torch.inference_mode()
    def preference(self, sim: FastSim, cands: list[int]) -> list[int]:
        """Candidates ordered by the base policy's preference (best first)."""
        if self.net is None:
            first = nearest_job(sim, cands)
            rest = sorted((j for j in cands if j != first),
                          key=lambda j: (round(sim.travel_time(j), 9), sim.pr[j], j))
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
        scen = sample_scenarios(sim, self.m, self.window, self.key) if self.m > 0 else [[]]
        looks = []
        for c in kept:
            for si, sc in enumerate(scen):
                cl = sim.lookahead_clone(sc, self.window)
                lk = _Look(cl, known, c, si)
                cl.apply(c)
                looks.append(lk)
        run_lookaheads(looks, self.net)
        self.lookaheads += len(looks)
        lam_n = sim.cfg.blocking_penalty_per_event
        lam_t = sim.cfg.blocking_penalty_per_second
        tot = {c: 0.0 for c in kept}
        for lk in looks:
            s = lk.sim
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
