"""Strong-RL phase 2: token features and the Transformer pointer model
(docs/methods/rl2.md). Acceptance (plan §3): permutation invariance, masking,
latency."""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastsim import POLICIES, FastSim  # noqa: E402
from optimization.schedule_check import check_schedule  # noqa: E402
from rl2.features import CONV_F, CTX_F, LASER_F, PART_F, Observation, encode  # noqa: E402
from rl2.model import SIZES, DispatchNet, NetPolicy, collate, load, save  # noqa: E402
from simulation.config import load_config  # noqa: E402

HEAVY = "config/examples/heavy_load.yaml"


def cfg(path=None, seed=7, mode="none"):
    return load_config(path, set_items=["arrivals.mode=stochastic", f"simulation.seed={seed}",
                                        f"laser.prepositioning={mode}"])


def decision_states(path=None, seed=7, mode="none", every=3, limit=40):
    """Observations at decisions of a nearest_job episode (only multi-candidate ones)."""
    sim, out, i = FastSim(cfg(path, seed, mode)), [], 0
    rule = POLICIES["nearest_job"]
    while sim.run_until_decision() and len(out) < limit:
        c = sim.candidates()
        if len(c) > 1 and i % every == 0:
            out.append((encode(sim), sorted(c)))
        i += 1
        sim.apply(rule(sim, c))
    return out


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return DispatchNet(SIZES["S"]).eval()


@pytest.fixture(scope="module")
def states():
    return (decision_states(None, 7, "none") + decision_states(HEAVY, 8, "when_idle")
            + decision_states(HEAVY, 9, "always", limit=15))


def test_features_shapes_and_candidates(states):
    assert len(states) > 40
    for obs, cands in states:
        assert obs.part.shape[1] == PART_F and obs.conv.shape == (3, CONV_F)
        assert obs.laser.shape == (LASER_F,) and obs.ctx.shape == (CTX_F,)
        assert np.isfinite(obs.part).all() and np.isfinite(obs.conv).all()
        assert sorted(obs.candidate_ids()) == cands
        assert len(set(obs.part_ids)) == len(obs.part_ids)
        assert np.abs(obs.part).max() < 20 and np.abs(obs.ctx).max() < 20


def test_size_s_parameter_count(net):
    assert 0.5e6 < net.n_params() < 1.5e6        # plan: ~1 M


def _permuted(obs: Observation, rng) -> tuple[Observation, np.ndarray]:
    pp = rng.permutation(obs.part.shape[0])
    pc = rng.permutation(obs.conv.shape[0])
    return Observation(obs.part[pp], obs.conv[pc], obs.laser, obs.ctx, obs.cand[pp],
                       [obs.part_ids[i] for i in pp]), pp


def test_permutation_invariance(net, states):
    """Shuffling part and conveyor tokens permutes the logits the same way and leaves
    the value unchanged."""
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for obs, _ in states[:30]:
            l0, v0 = net(collate([obs]))
            ob2, pp = _permuted(obs, rng)
            l1, v1 = net(collate([ob2]))
            a, b = l0[0][torch.from_numpy(pp)], l1[0]
            fin = torch.isfinite(a)
            assert torch.equal(fin, torch.isfinite(b))
            assert torch.allclose(a[fin], b[fin], atol=1e-4)
            assert torch.allclose(v0, v1, atol=1e-4)


def test_batched_equals_single(net, states):
    """Padding in a batch does not change any state's output."""
    obs = [o for o, _ in states[:12]]
    with torch.no_grad():
        lb, vb = net(collate(obs))
        for i, o in enumerate(obs):
            l1, v1 = net(collate([o]))
            p = o.part.shape[0]
            fin = torch.isfinite(l1[0])
            assert torch.allclose(lb[i, :p][fin], l1[0][fin], atol=1e-4)
            assert torch.isinf(lb[i, p:]).all()
            assert torch.allclose(vb[i], v1[0], atol=1e-4)


def test_masking_never_selects_a_non_candidate(states):
    """Several random networks: non-candidates and padding get probability 0 and are
    never the argmax or a sample."""
    obs = [o for o, _ in states]
    b = collate(obs)
    g = torch.Generator().manual_seed(1)
    for seed in range(5):
        torch.manual_seed(seed)
        n = DispatchNet(SIZES["S"]).eval()
        with torch.no_grad():
            logits, _ = n(b)
            prob = torch.softmax(logits, -1)
        allowed = b.cand & b.part_valid
        assert (prob[~allowed] == 0).all()
        assert torch.allclose(prob.sum(-1), torch.ones(len(obs)))
        assert allowed.gather(1, logits.argmax(-1, keepdim=True)).all()
        s = torch.multinomial(prob, 20, replacement=True, generator=g)
        assert allowed.gather(1, s).all()


@pytest.mark.parametrize("path,mode", [(None, "none"), (HEAVY, "when_idle")])
def test_untrained_net_runs_a_valid_episode(net, path, mode):
    sim = FastSim(load_config(path, set_items=["arrivals.mode=stochastic", "simulation.seed=21",
                                               f"laser.prepositioning={mode}",
                                               "simulation.arrival_horizon=200"]))
    sim.run(NetPolicy(net))
    assert sim.n_completed == len(sim.pk) > 0
    assert check_schedule(sim.result(), time_tol=1e-6, dist_tol=1e-6) == []


def test_any_candidate_not_only_heads(states):
    """The action space contains non-head candidates (the point of the pointer)."""
    assert any(((o.part[:, 15] == 1) & (o.part[:, 14] == 0)).any() for o, _ in states)


def test_save_and_load_roundtrip(net, states, tmp_path):
    p = tmp_path / "m.pt"
    save(net, p)
    n2 = load(p)
    b = collate([o for o, _ in states[:5]])
    with torch.no_grad():
        assert torch.equal(net(b)[1], n2(b)[1])


def test_latency_size_s(net, states):
    """Network-only decision latency, single state, 1 thread. The plan's budget is
    5 ms; the test allows margin for a loaded machine (numbers: rl2.md §4)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        b = collate([states[0][0]])
        with torch.no_grad():
            for _ in range(5):
                net(b)
            t0 = time.perf_counter()
            for _ in range(50):
                net(b)
        ms = (time.perf_counter() - t0) / 50 * 1e3
    finally:
        torch.set_num_threads(old)
    assert ms < 25
