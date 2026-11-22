"""Information model (0.7.3): policies use only observable parts.

The real cell has an overhead camera that sees every belt and its whole working area
(owner, 2026-11-20): ``simulation.observation: belts``. Parts waiting upstream of the
belt entry are not seen. ``zone`` (laser zone only) is a what-if level.

Each test changes the parts that are unobservable at a restricted level and checks that
the policy's view is unchanged; the same change under ``full`` must be visible, so each
test can detect a leak.
"""

from __future__ import annotations

import pytest

from fastsim import POLICIES, FastSim
from policies import get_policy
from policies.base_policy import BasePolicy
from policies.rollout import RolloutPolicy
from simulation.config import load_config
from simulation.engine import Simulation
from simulation.observation import visible_status

HEAVY = "config/examples/heavy_load.yaml"
RESTRICTED = ["belts", "zone"]


def cfg(obs="auto", mode="none", seed=4, path=HEAVY):
    return load_config(path, set_items=["arrivals.mode=stochastic", f"simulation.seed={seed}",
                                        f"laser.prepositioning={mode}",
                                        f"simulation.observation={obs}"])


def test_config_resolution():
    assert load_config().simulation.observation == "belts"          # real cell
    assert cfg(mode="when_idle").simulation.observation == "belts"
    assert cfg(obs="full").simulation.observation == "full"
    assert cfg(obs="zone").simulation.observation == "zone"
    with pytest.raises(ValueError):
        cfg(obs="zone", mode="always")
    with pytest.raises(ValueError):
        cfg(obs="everything")


# ------------------------------------------------------------------ engine
class _Probe(BasePolicy):
    """nearest_job that runs ``check`` at decisions with parts hidden at ``level``."""

    name = "probe"

    def __init__(self, check, level, limit=5):
        super().__init__()
        self.check, self.level, self.limit, self.done = check, level, limit, 0
        self.base = get_policy("nearest_job")

    def select(self, candidates, ctx):
        sim = ctx.simulation
        hidden = [p for p in sim.parts if p.completion_time is None
                  and not visible_status(p.status.value, self.level)]
        if hidden and len(candidates) > 1 and self.done < self.limit:
            self.check(candidates, ctx, hidden)
            self.done += 1
        return self.base.select(candidates, ctx)


def _probe(c, check, level):
    pol = _Probe(check, level)
    Simulation(c, policy=pol, record_trace=False).run()
    assert pol.done > 0, "no decision with unobservable parts was probed"


@pytest.mark.parametrize("level", RESTRICTED)
def test_engine_rollout_clone_has_only_observable_parts(level):
    def check(candidates, ctx, hidden):
        clone = RolloutPolicy().clone(ctx.simulation, get_policy("nearest_job"))
        left = [p for p in clone.parts if p.completion_time is None]
        assert left and all(visible_status(p.status.value, level) for p in left)
        assert {p.id for p in hidden}.isdisjoint(p.id for p in clone.parts)
    _probe(cfg(obs=level), check, level)


def _sensitivity(c, level, view, mutate):
    """Number of probed decisions where ``mutate`` of the hidden parts changes ``view``."""
    changed = []

    def check(candidates, ctx, hidden):
        a = view(candidates, ctx)
        undo = mutate(hidden)
        b = view(candidates, ctx)
        undo()
        changed.append(not _equal(a, b))
    _probe(c, check, level)
    return sum(changed)


def _equal(a, b):
    try:
        import numpy as np
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
    except ImportError:
        pass
    return a == b


def _bump(hidden):
    old = [(p.processing_time, p.arrival_time) for p in hidden]
    for p in hidden:
        p.processing_time += 7.0
        p.arrival_time -= 20.0

    def undo():
        for p, (pt, at) in zip(hidden, old):
            p.processing_time, p.arrival_time = pt, at
    return undo


@pytest.mark.parametrize("level", RESTRICTED)
def test_engine_rollout_costs_ignore_unobservable_parts(level):
    def view(candidates, ctx):
        r = RolloutPolicy(objective="avg_flow_time", dt_lookahead=0.05)
        return {c.id: r.lookahead_cost(ctx.simulation, c.id) for c in candidates}
    assert _sensitivity(cfg(obs=level), level, view, _bump) == 0
    assert _sensitivity(cfg(obs="full"), level, view, _bump) > 0


@pytest.mark.parametrize("level", RESTRICTED)
def test_rl_features_ignore_unobservable_parts(level):
    pytest.importorskip("numpy")
    from rl.features import encode

    state = {}

    def view(candidates, ctx):
        state["sim"] = ctx.simulation
        return encode(ctx.simulation, candidates)

    def bump_and_drop(hidden):
        """Change the hidden parts and take the waiting ones out of their buffers (the
        agent sees waiting parts only through the buffer length)."""
        undo_bump = _bump(hidden)
        ids = {p.id for p in hidden}
        saved = [(c, c.buffer.copy()) for c in state["sim"].conveyors]
        for c in state["sim"].conveyors:
            for p in [p for p in c.buffer if p.id in ids]:
                c.buffer.remove(p)

        def undo():
            undo_bump()
            for c, buf in saved:
                c.buffer = buf
        return undo
    assert _sensitivity(cfg(obs=level), level, view, bump_and_drop) == 0
    assert _sensitivity(cfg(obs="full"), level, view, bump_and_drop) > 0


# ---------------------------------------------------------------- fastsim / rl2
def _fast_decisions(c, level, n=8):
    """FastSim states at decisions with parts that are hidden at ``level``."""
    sim, seen = FastSim(c), 0
    rule = POLICIES["nearest_job"]
    names = ("WAITING", "MOVING", "READY", "PROCESSING", "COMPLETED", "EXITED")
    while sim.run_until_decision() and seen < n:
        cands = sim.candidates()
        hidden = [j for j in range(len(sim.pk)) if sim.pC[j] is None
                  and not visible_status(names[sim.pst[j]], level)]
        if hidden and len(cands) > 1:
            yield sim, cands, hidden
            seen += 1
        sim.apply(rule(sim, cands))


@pytest.mark.parametrize("level", RESTRICTED)
def test_fastsim_teacher_ignores_unobservable_parts(level):
    from rl2.search import SearchTeacher

    def changes(c):
        out = []
        for sim, cands, hidden in _fast_decisions(c, level):
            cl = sim.lookahead_clone()
            if c.simulation.observation == level:
                assert all(j not in q for q in cl.belt + cl.buf for j in hidden)
            t = SearchTeacher(None, k=6)
            a = t.costs(sim, cands)
            other = sim.clone()
            for j in hidden:
                other.pp[j] += 7.0
            out.append(t.costs(other, cands) != a)
        assert out
        return sum(out)
    assert changes(cfg(obs=level)) == 0
    assert changes(cfg(obs="full")) > 0


@pytest.mark.parametrize("level", RESTRICTED)
def test_rl2_tokens_only_observable_parts(level):
    pytest.importorskip("torch")
    from rl2.features import OBSERVATIONS, encode
    n = 0
    for sim, cands, hidden in _fast_decisions(cfg(obs=level), level):
        obs = encode(sim, cands)
        assert set(obs.part_ids).isdisjoint(hidden)
        assert all(sim.observable(j) for j in obs.part_ids)
        assert obs.part[:, 0].sum() == 0                     # no WAITING token
        if level == "zone":
            assert obs.part[:, 1].sum() == 0                 # no entry-zone token
        assert (obs.conv[:, 6] == 0).all()                   # upstream buffer unseen
        k = len(obs.ctx) - 9                                 # observation one-hot
        assert list(obs.ctx[k:k + 3]) == [1.0 if o == level else 0.0 for o in OBSERVATIONS]
        n += 1
    assert n > 0
    for sim, cands, hidden in _fast_decisions(cfg(obs="full"), level, n=3):
        assert set(hidden) <= set(encode(sim, cands).part_ids)


# ------------------------------------------------ 0.7.6: past and future leaks
def _obs_equal(a, b) -> bool:
    import numpy as np
    return (a.part_ids == b.part_ids and np.array_equal(a.cand, b.cand)
            and all(np.allclose(getattr(a, f), getattr(b, f), atol=1e-12)
                    for f in ("part", "conv", "laser", "ctx")))


def _tokens_at_decisions(c, level, n=6):
    """(FastSim state, tokens) at the first ``n`` multi-candidate decisions."""
    from rl2.features import encode
    sim = FastSim(c)
    out = []
    while sim.run_until_decision() and len(out) < n:
        cands = sim.candidates()
        if len(cands) > 1:
            out.append((sim.clone(), encode(sim, cands)))
        sim.apply(POLICIES["nearest_job"](sim, cands))
    return out


@pytest.mark.parametrize("level", RESTRICTED)
def test_rl2_waiting_time_counts_from_the_instant_the_part_is_seen(level):
    """Under ``belts``/``zone`` the arrival time of a part that waited upstream is not
    observable: shifting it must not change the tokens; under ``full`` it must."""
    from rl2.features import encode
    c = cfg(level, seed=4)
    states = _tokens_at_decisions(c, level)
    moved = 0
    for sim, obs in states:
        waited = [j for j in obs.part_ids if sim.pentry[j] is not None
                  and sim.pentry[j] > sim.pr[j] + 1e-9]
        if not waited:
            continue
        alt = sim.clone()
        for j in waited:
            alt.pr[j] -= 3.0                       # arrived earlier in the hidden queue
        assert _obs_equal(encode(alt, alt.candidates()), obs)
        moved += 1
    assert moved > 0, "no part waited upstream in this instance"
    # the same change is visible with full information
    cf = cfg("full", seed=4)
    seen = False
    for sim, obs in _tokens_at_decisions(cf, "full"):
        waited = [j for j in obs.part_ids if sim.pentry[j] is not None
                  and sim.pentry[j] > sim.pr[j] + 1e-9]
        if waited:
            alt = sim.clone()
            alt.pr[waited[0]] -= 3.0
            assert not _obs_equal(encode(alt, alt.candidates()), obs)
            seen = True
            break
    assert seen


@pytest.mark.parametrize("level", ["belts", "zone", "full"])
def test_rl2_tokens_ignore_the_real_future_arrivals(level):
    """Removing every future arrival (the controller cannot know there are none)
    leaves the tokens unchanged at every level."""
    from rl2.features import encode
    c = cfg(level, seed=5)
    for sim, obs in _tokens_at_decisions(c, level):
        alt = sim.clone()
        alt.arr = sim.arr[: sim.next_arr]          # no more real arrivals
        assert _obs_equal(encode(alt, alt.candidates()), obs)
