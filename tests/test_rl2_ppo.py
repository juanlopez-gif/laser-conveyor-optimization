"""PPO baseline over the token policy (docs/methods/rl2_ppo.md): masking, advantages,
loss shapes and one tiny end-to-end run."""

from __future__ import annotations

import json
import shutil
import uuid

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl2.expert_iteration import ROOT  # noqa: E402
from rl2.model import SIZES, DispatchNet, RunningScale, collate, load, save  # noqa: E402
from rl2.ppo import (PPOConfig, gae_costs, main, play_episode, ppo_loss,  # noqa: E402
                     ppo_update, prepare_batch, sample_action)
from test_rl2_model import HEAVY, decision_states  # noqa: E402


@pytest.fixture(scope="module")
def states():
    return [o for o, _ in decision_states(None, 7, "none") + decision_states(HEAVY, 8, "when_idle")]


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    """Short episodes played by an untrained network (sampled actions); a light cell
    may have no decision with >= 2 candidates, so only episodes with samples are kept."""
    torch.manual_seed(0)
    net = DispatchNet(SIZES["S"]).eval()
    path = tmp_path_factory.mktemp("ppo") / "act.pt"
    save(net, path, {"scale": [0, 0.0, 0.0]})
    eps = [play_episode({"seed": 200_000 + e, "net": str(path), "horizon": 120})
           for e in range(8)]
    out = [e for e in eps if e["obs"]]
    assert len(out) >= 3
    return out


def test_sampling_never_picks_a_non_candidate(states):
    """Several random networks, many samples per state: the sampled index is always a
    candidate, its log-prob is finite and the value is a finite scalar."""
    g = torch.Generator().manual_seed(1)
    for seed in range(3):
        torch.manual_seed(seed)
        net = DispatchNet(SIZES["S"]).eval()
        for obs in states:
            for _ in range(10):
                i, lp, v = sample_action(net, obs, g)
                assert obs.cand[i] and np.isfinite(lp) and lp <= 1e-6 and np.isfinite(v)


def test_episode_samples_are_multi_candidate_decisions(episodes):
    """Decisions with one candidate are excluded; each sample's action is a candidate;
    the cost at the decisions is non-decreasing and the final cost bounds it."""
    for e in episodes:
        assert len(e["obs"]) == len(e["act"]) == len(e["logp"]) == len(e["cost_at"]) > 0
        assert len(e["obs"]) < e["decisions"]
        for o, a in zip(e["obs"], e["act"]):
            assert o.cand.sum() >= 2 and o.cand[a]
        c = np.asarray(e["cost_at"] + [e["final_cost"]])
        assert (np.diff(c) >= -1e-9).all()


def test_gae_reduces_to_monte_carlo_cost_to_go():
    """lambda = 1 with a zero value function: the advantage is the cost-to-go; with a
    perfect value function every advantage is 0."""
    cost_at, final = [0.0, 3.0, 7.0, 12.0], 20.0
    adv, ret, mc = gae_costs(cost_at, final, [0.0] * 4, lam=1.0)
    assert np.allclose(adv, [20, 17, 13, 8]) and np.allclose(mc, adv) and np.allclose(ret, adv)
    adv, ret, _ = gae_costs(cost_at, final, [20.0, 17.0, 13.0, 8.0], lam=0.95)
    assert np.allclose(adv, 0) and np.allclose(ret, [20, 17, 13, 8])
    adv, _, _ = gae_costs(cost_at, final, [0.0] * 4, lam=0.0)   # one-step TD
    assert np.allclose(adv, [3, 4, 5, 8])


def test_prepare_batch_normalizes_and_updates_the_scale(episodes):
    sc = RunningScale()
    b = prepare_batch(episodes, sc, lam=0.95)
    n = sum(len(e["obs"]) for e in episodes)
    assert len(b["obs"]) == len(b["act"]) == len(b["logp"]) == len(b["adv"]) == len(b["vt"]) == n
    assert abs(b["adv"].mean()) < 1e-6 and abs(b["adv"].std() - 1) < 1e-6
    assert sc.n == n and sc.mean == pytest.approx(b["mc"].mean())
    assert abs(b["vt"].mean()) < 0.5          # targets in units of the running scale


def test_loss_shapes_and_ratio_one(episodes):
    """At the acting weights the ratio is 1: the clipped fraction is 0, the KL ~ 0 and
    the policy loss equals minus the mean advantage."""
    torch.manual_seed(0)
    net = DispatchNet(SIZES["S"]).eval()
    sc = RunningScale()
    b = prepare_batch(episodes, sc, lam=0.95)
    idx = np.arange(min(64, len(b["obs"])))
    batch = collate([b["obs"][i] for i in idx])
    with torch.no_grad():
        logits, value = net(batch)
    act = torch.from_numpy(b["act"][idx])
    adv = torch.from_numpy(b["adv"][idx].astype(np.float32))
    logp_all = torch.log_softmax(logits, -1)
    logp_old = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
    loss, st = ppo_loss(logits, value, act, logp_old, adv,
                        torch.from_numpy(b["vt"][idx].astype(np.float32)), PPOConfig())
    assert loss.shape == () and torch.isfinite(loss)
    assert logits.shape == (len(idx), batch.part.shape[1]) and value.shape == (len(idx),)
    assert st["clipfrac"] == 0 and abs(st["kl"]) < 1e-5
    assert st["policy"] == pytest.approx(-float(adv.mean()), abs=1e-5)
    assert 0 <= st["entropy"] <= float(torch.log(batch.cand.sum(-1).float()).max())


def test_update_moves_the_policy_towards_positive_advantages(episodes):
    """A handful of updates with a large step lower the policy loss on the same batch
    and leave the network on the CPU."""
    torch.manual_seed(0)
    net = DispatchNet(SIZES["S"]).eval()
    b = prepare_batch(episodes, RunningScale(), lam=0.95)
    cfg = PPOConfig(epochs=3, minibatch=64, lr=1e-3)
    h = ppo_update(net, b, cfg, seed=0, threads=2, device="cpu")
    assert h["samples"] == len(b["obs"]) and len(h["epochs"]) == 3
    assert h["epochs"][-1]["policy"] < h["epochs"][0]["policy"]
    assert all(p.device.type == "cpu" for p in net.parameters())


def test_tiny_end_to_end_run():
    """Two updates of four episodes, validation on two seeds per group: the log and the
    best model are written, the best is one of the candidates, no reserved seed is
    used; repository-relative paths only."""
    tag = f"test_ppo_{uuid.uuid4().hex[:8]}"
    try:
        assert main(["--tag", tag, "--seed", "3", "--episodes", "8", "--batch-episodes", "4",
                     "--val-every", "4", "--val", "2", "--workers", "2", "--threads", "2",
                     "--device", "cpu", "--horizon", "60", "--minibatch", "32"]) == 0
        log = json.loads((ROOT / "models" / f"{tag}_training.json").read_text())
        ups = log["updates"]
        assert [u["update"] for u in ups] == [1, 2] and sum(u["episodes"] for u in ups) == 8
        assert ups[0]["seeds"][0] == 500_000 and ups[1]["seeds"][1] == 500_007
        vals = [v for v in log["validations"] if v["candidate"]]
        assert [v["update"] for v in vals] == [1, 2] and all(len(v["val"]) == 5 for v in vals)
        assert log["best"]["checkpoint"] in {v["model"] for v in vals}
        assert not log["best"]["model"].startswith(("C:", "/"))
        net = load(ROOT / log["best"]["model"])
        assert net.n_params() > 0
        assert all("_it" in v["model"].rsplit("/", 1)[-1] for v in vals)
    finally:
        for p in (ROOT / "models").glob(f"{tag}*"):
            p.unlink()
        shutil.rmtree(ROOT / "results" / "experiments" / tag, ignore_errors=True)
