"""Strong-RL phase 3: portable training (device selection, checkpoint and resume)."""

from __future__ import annotations

import json
import shutil
import uuid

import pytest

torch = pytest.importorskip("torch")

from rl2.expert_iteration import ROOT, main, play_episode, train  # noqa: E402
from rl2.model import (SIZES, DispatchNet, RunningScale, collate, load,  # noqa: E402
                       resolve_device, save)


@pytest.fixture(scope="module")
def samples():
    ep = play_episode({"seed": 100011, "net": None, "k": 6, "w_pen": 0.0, "beta": 1.0,
                       "eps": 0.1, "mix": {"p_real": 1.0}, "horizon": 120})
    return ep["samples"]


def _train(samples, device):
    torch.manual_seed(0)
    net = DispatchNet(SIZES["S"])
    sc = RunningScale()
    sc.update([s[2] for s in samples])
    h = train(net, samples, sc, epochs=2, lr=1e-3, batch=32, temp=1.0, value_coef=0.25,
              seed=0, threads=2, device=device)
    return net, h


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_training_on_cpu_returns_a_cpu_model(samples):
    net, h = _train(samples, "cpu")
    assert h["device"] == "cpu"
    assert all(p.device.type == "cpu" for p in net.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_training_on_cuda_matches_cpu(samples):
    """Same data, same seed: GPU and CPU losses agree to float tolerance and the GPU
    model comes back on the CPU."""
    _, hc = _train(samples, "cpu")
    net, hg = _train(samples, "cuda")
    assert hg["device"].startswith("cuda")
    assert all(p.device.type == "cpu" for p in net.parameters())
    assert hg["epochs"][0]["policy"] == pytest.approx(hc["epochs"][0]["policy"], rel=1e-3)


def test_batch_to_device_and_cpu_save(samples, tmp_path):
    b = collate([s[0] for s in samples[:4]]).to("cpu")
    assert b.part.device.type == "cpu"
    net = DispatchNet(SIZES["S"])
    save(net, tmp_path / "m.pt")
    ck = torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in ck["state"].values())
    load(tmp_path / "m.pt")


def test_checkpoint_and_resume_continue_the_same_run():
    """Run iteration 0, stop, resume to iteration 1. The resumed run continues the
    seeds, keeps the experiment arguments (a changed --episodes is ignored) and logs
    both iterations; repository-relative paths only."""
    tag = f"test_resume_{uuid.uuid4().hex[:8]}"
    base = ["--tag", tag, "--size", "S", "--episodes0", "4", "--episodes", "2",
            "--epochs0", "1", "--epochs", "1", "--val", "2", "--workers", "2",
            "--threads", "2", "--device", "cpu", "--horizon", "60"]
    try:
        assert main(base + ["--iters", "0"]) == 0
        state = ROOT / "results" / "experiments" / tag / "state.pkl"
        assert state.exists()
        assert main(["--tag", tag, "--resume", "--iters", "1", "--episodes", "99",
                     "--workers", "2", "--threads", "2", "--device", "cpu"]) == 0
        log = json.loads((ROOT / "models" / f"{tag}_training.json").read_text())
        its = log["iterations"]
        assert [i["iteration"] for i in its] == [0, 1]
        assert its[1]["seeds"][0] == its[0]["seeds"][1] + 1
        assert its[1]["episodes"] == 2 and log["args"]["episodes"] == 2
        assert len(log["resumed"]) == 1 and len(log["machine"]) == 2
        assert not its[0]["model"].startswith(("C:", "/"))
        assert (ROOT / log["best"]["model"]).exists()
    finally:
        for p in (ROOT / "models").glob(f"{tag}*"):
            p.unlink()
        shutil.rmtree(ROOT / "results" / "experiments" / tag, ignore_errors=True)
