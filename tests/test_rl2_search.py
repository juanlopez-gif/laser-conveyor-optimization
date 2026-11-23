"""Strong-RL phase 3: search teacher and expert-iteration pieces (docs/methods/rl2.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fastsim import POLICIES, FastSim  # noqa: E402
from rl2.expert_iteration import (GENERAL, _targets, general_config, play_episode,  # noqa: E402
                                  sample_config, train)
from rl2.model import SIZES, DispatchNet, ModelConfig, RunningScale  # noqa: E402
from rl2.search import SearchTeacher, run_lookaheads, sample_scenarios, soft_target  # noqa: E402
from simulation.config import load_config  # noqa: E402

HEAVY = "config/examples/heavy_load.yaml"


def cfg(path=None, seed=7, mode="none", **extra):
    items = ["arrivals.mode=stochastic", f"simulation.seed={seed}",
             f"laser.prepositioning={mode}"] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(path, set_items=items)


def first_multi_decision(sim, skip=5):
    rule, seen = POLICIES["nearest_job"], 0
    while sim.run_until_decision():
        c = sim.candidates()
        if len(c) > 1:
            if seen == skip:
                return c
            seen += 1
        sim.apply(rule(sim, c))
    raise AssertionError("no multi-candidate decision")


def test_lookahead_clone_hides_the_real_future():
    sim = FastSim(cfg(HEAVY, 4))
    first_multi_decision(sim)
    assert sim.next_arr < len(sim.arr)                  # real future exists
    c = sim.lookahead_clone()
    assert c.arr == () and c.next_arr == 0
    assert sim.next_arr < len(sim.arr)                  # original untouched
    sc = sample_scenarios(sim, 1, 30.0, "t")[0]
    c2 = sim.lookahead_clone(sc, 30.0)
    assert len(c2.arr) == len(sc) and all(a[0] > sim.t for a in c2.arr)


def test_teacher_ignores_the_real_future():
    """Changing the real future arrivals does not change the teacher's costs."""
    sim = FastSim(cfg(HEAVY, 5))
    cands = first_multi_decision(sim)
    t = SearchTeacher(None, k=6)
    a = t.costs(sim, cands)
    other = sim.clone()
    other.arr = sim.arr[: sim.next_arr] + tuple((r + 0.5, (k + 1) % 3, p)
                                               for r, k, p in sim.arr[sim.next_arr:])
    assert t.costs(other, cands) == a


def test_rule_teacher_is_the_rollout_over_nearest_job():
    """The base choice is kept (scored first) and the choice minimizes the cost; with
    one look-ahead per candidate the known parts all finish."""
    sim = FastSim(cfg(None, 9))
    cands = first_multi_decision(sim, skip=2)
    t = SearchTeacher(None, k=6)
    costs = t.costs(sim, cands)
    assert next(iter(costs)) == POLICIES["nearest_job"](sim, cands)
    assert set(costs) <= set(cands) and len(costs) == min(6, len(cands))
    ch = t.choose(costs)
    assert costs[ch] == min(costs.values())


def test_rule_teacher_beats_nearest_job_on_average():
    d = []
    for seed in range(2000, 2006):
        c = cfg(None, seed, **{"simulation.arrival_horizon": 200})
        ref = FastSim(c).run(POLICIES["nearest_job"])["penalized_flow_time"]
        d.append(FastSim(c).run(SearchTeacher(None, k=6))["penalized_flow_time"] / ref - 1)
    assert np.mean(d) < 0


def test_batched_network_lookahead_equals_sequential():
    """Lock-step batching gives the same look-ahead as running each clone alone."""
    torch.manual_seed(0)
    net = DispatchNet(ModelConfig(1, 32, 2)).eval()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    t = SearchTeacher(net, k=6)
    together = t.costs(sim, cands)
    alone = {}
    for c in together:
        t1 = SearchTeacher(net, k=6)
        t1.preference = lambda s, cs, c=c: [c]          # score only c
        alone.update(t1.costs(sim, cands))
    for c in together:
        assert together[c] == pytest.approx(alone[c], abs=1e-9)


def test_soft_target_and_training_targets():
    st = soft_target({1: 10.0, 2: 10.0 + math.log(3)}, 1.0)
    assert st[1] == pytest.approx(0.75) and st[2] == pytest.approx(0.25)
    tg = _targets([np.array([0.0, np.nan, math.log(3)], np.float32)], 1.0, 4)
    assert torch.allclose(tg[0], torch.tensor([0.75, 0.0, 0.25, 0.0]))
    hard = _targets([np.array([0.5, 0.0, np.nan], np.float32)], 0.0, 3)
    assert torch.equal(hard[0], torch.tensor([0.0, 1.0, 0.0]))


def test_sample_config_mix():
    kinds = [sample_config(100000 + i)[1].split("/")[0] for i in range(80)]
    assert {"real", "heavy", "general"} <= set(kinds)
    assert 0.45 < kinds.count("general") / len(kinds) < 0.85     # nominal 65 %


def test_general_cells_cover_the_ranges():
    G = GENERAL
    ds = [general_config(9000 + i) for i in range(150)]
    ns = {d["n"] for _, d in ds}
    assert ns == set(range(G["n_conveyors"][0], G["n_conveyors"][1] + 1))
    for c, d in ds:
        assert len(c.conveyors) == d["n"]
        assert all(1.5 <= x.speed <= 6.0 for x in c.conveyors)
        assert 5.0 <= c.laser.speed_x <= 20.0 and 5.0 <= c.laser.speed_y <= 20.0
        rho = sum(1.0 / m for m in c.arrivals.mean_interarrival) * d["p_mean"]
        assert 0.3 - 1e-3 <= rho <= 0.95 + 1e-3
    assert {d["dist"] for _, d in ds} == {"constant", "uniform", "normal"}
    assert {d["motion"] for _, d in ds} == {"simultaneous", "sequential"}
    assert {d["stop"] for _, d in ds} == {"all_conveyors", "processing_conveyor_only"}
    # independent belt speeds
    assert any(len({x.speed for x in c.conveyors}) == d["n"] > 1 for c, d in ds)


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_general_cells_run_valid_with_rule_teacher_and_net(n):
    """Fast simulator, rule teacher and an untrained network on general cells with
    1-6 belts: complete schedules, no violation of the formal model."""
    from optimization.schedule_check import check_schedule
    from rl2.model import NetPolicy
    seed = next(9000 + i for i in range(500) if general_config(9000 + i)[1]["n"] == n)
    c, _ = general_config(seed, extra=("simulation.arrival_horizon=150",))
    torch.manual_seed(0)
    for pol in (SearchTeacher(None, k=6), NetPolicy(DispatchNet(ModelConfig(1, 32, 2)))):
        sim = FastSim(c)
        sim.run(pol)
        assert sim.n_completed == len(sim.pk) > 0
        assert check_schedule(sim.result(), time_tol=1e-6, dist_tol=1e-6) == []


def test_episode_labels_and_one_training_step(tmp_path):
    """A labelled episode has finite targets with a zero-cost best candidate, the
    cost-to-go decreases along the episode, and a short training lowers the loss."""
    ep = play_episode({"seed": 100003, "net": None, "k": 6, "w_pen": 0.0, "beta": 1.0,
                       "eps": 0.1, "mix": {"p_real": 1.0}})
    assert ep["samples"]
    for obs, cvec, ctg in ep["samples"]:
        ok = ~np.isnan(cvec)
        assert ok.sum() >= 2 and np.nanmin(cvec) == 0.0
        assert all(obs.cand[ok]) and ctg >= 0
    ctgs = [s[2] for s in ep["samples"]]
    assert ctgs[0] > ctgs[-1]
    net = DispatchNet(SIZES["S"])
    sc = RunningScale()
    sc.update(ctgs)
    h = train(net, ep["samples"], sc, epochs=4, lr=1e-3, batch=32, temp=1.0,
              value_coef=0.25, seed=0, threads=2)["epochs"]
    assert h[-1]["policy"] < h[0]["policy"]


# ------------------------------------------------- value-truncated teacher (0.7.6)
def _small_net():
    torch.manual_seed(0)
    return DispatchNet(ModelConfig(1, 32, 2)).eval()


def test_truncated_lookaheads_cut_at_a_decision_after_the_horizon():
    """Every look-ahead stops at a decision with >= 2 candidates at or after
    t + horizon, or at the end of its clone; the costs are finite."""
    from rl2.search import _Look
    net = _small_net()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    t0 = sim.t
    scen = sample_scenarios(sim, 2, 80.0, "t")
    looks = []
    for c in cands[:2]:
        for sc in scen:
            cl = sim.lookahead_clone(sc, 80.0)
            lk = _Look(cl, [], c, 0, t0 + 20.0)
            cl.apply(c)
            looks.append(lk)
    run_lookaheads(looks, net)
    assert all(lk.done for lk in looks)
    for lk in looks:
        if lk.stopped:
            assert lk.sim.t >= t0 + 20.0 - 1e-9 and len(lk.sim.candidates()) >= 2
        else:
            assert lk.sim.done
    t = SearchTeacher(net, k=6, horizon_s=20.0, scenarios=2, scale=RunningScale(),
                      cut="decision")
    costs = t.costs(sim, cands)
    assert t.truncated and set(costs) <= set(cands)
    assert all(math.isfinite(v) for v in costs.values())
    assert next(iter(costs)) == t.preference(sim, cands)[0]        # base choice first


def test_truncated_teacher_batched_equals_sequential_and_ignores_the_real_future():
    net = _small_net()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    other = sim.clone()
    other.arr = sim.arr[: sim.next_arr] + tuple((r + 0.5, (k + 1) % 3, p)
                                               for r, k, p in sim.arr[sim.next_arr:])
    for kw in ({"horizon_s": 20.0, "cut": "decision"}, {"horizon_s": 20.0, "cut": "time"},
               {"horizon_s": math.inf, "window_s": 40.0}):
        t = SearchTeacher(net, k=6, scenarios=2, w_pen=1.0, scale=RunningScale(), **kw)
        together = t.costs(sim, cands)
        alone = {}
        for c in together:
            t1 = SearchTeacher(net, k=6, scenarios=2, w_pen=1.0, scale=RunningScale(), **kw)
            t1.preference = lambda s, cs, c=c: [c]
            alone.update(t1.costs(sim, cands))
        for c in together:
            assert together[c] == pytest.approx(alone[c], abs=1e-6)
        assert t.costs(other, cands) == together


def test_truncated_cost_is_realized_cost_plus_scaled_value():
    """cost = Δ WIP integral + λ_N ΔN_B + λ_T ΔT_B (w_pen = 1) + denormalized value."""
    from rl2.search import WINDOW_MARGIN_S, _Look
    net = _small_net()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    sc = RunningScale()
    sc.n, sc.mean, sc.m2 = 10, 100.0, 9 * 25.0          # std 5
    t = SearchTeacher(net, k=6, horizon_s=15.0, scenarios=1, w_pen=1.0, scale=sc,
                      cut="decision")
    costs = t.costs(sim, cands)
    c = next(iter(costs))
    scen = sample_scenarios(sim, 1, 15.0 + WINDOW_MARGIN_S, t.key)[0]
    cl = sim.lookahead_clone(scen, 15.0 + WINDOW_MARGIN_S)
    lk = _Look(cl, [], c, 0, sim.t + 15.0)
    cl.apply(c)
    run_lookaheads([lk], net)
    assert lk.stopped
    from rl2.features import encode
    from rl2.model import collate
    with torch.inference_mode():
        v = float(net(collate([encode(cl, cl.candidates())]))[1][0])
    lam_n, lam_t = sim.cfg.blocking_penalty_per_event, sim.cfg.blocking_penalty_per_second
    expect = (cl.wip_area - sim.wip_area) + lam_n * (cl.n_block_events - sim.n_block_events) \
        + lam_t * (cl.blocked_area - sim.blocked_area) + sc.denormalize(v)
    assert costs[c] == pytest.approx(expect, abs=1e-6)


def test_rule_base_keeps_the_known_parts_lookahead():
    sim = FastSim(cfg(None, 9))
    cands = first_multi_decision(sim, skip=2)
    assert SearchTeacher(None, k=6, horizon_s=20.0, scenarios=2).costs(sim, cands) == \
        SearchTeacher(None, k=6).costs(sim, cands)


def test_load_scale_roundtrip(tmp_path):
    from rl2.model import load_scale, save
    net = _small_net()
    save(net, tmp_path / "m.pt", {"scale": [7, 12.5, 48.0]})
    sc = load_scale(tmp_path / "m.pt")
    assert (sc.n, sc.mean, sc.m2) == (7, 12.5, 48.0)
    assert sc.denormalize(sc.normalize(3.0)) == pytest.approx(3.0)
    save(net, tmp_path / "n.pt")
    assert load_scale(tmp_path / "n.pt") is None


def test_time_cut_stops_every_clone_at_the_same_instant():
    net = _small_net()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    t = SearchTeacher(net, k=6, horizon_s=12.0, scenarios=2, scale=RunningScale(), cut="time")
    costs = t.costs(sim, cands)
    assert all(math.isfinite(v) for v in costs.values())
    cl = sim.lookahead_clone(sample_scenarios(sim, 1, 72.0, t.key)[0], 72.0)
    cl.max_time = sim.t + 12.0
    cl.apply(next(iter(costs)))
    assert not cl.run(POLICIES["nearest_job"]) is None
    assert cl.t == pytest.approx(sim.t + 12.0)


def test_rollout_to_the_end_uses_no_value_head():
    """horizon inf: cost = realized cost-to-go of the drained clone, V = 0."""
    net = _small_net()
    sim = FastSim(cfg(HEAVY, 6))
    cands = first_multi_decision(sim)
    sc = RunningScale()
    sc.n, sc.mean, sc.m2 = 10, 1e6, 9.0          # a huge value offset would show up
    t = SearchTeacher(net, k=6, horizon_s=math.inf, scenarios=1, window_s=20.0, w_pen=1.0,
                      scale=sc)
    costs = t.costs(sim, cands)
    assert all(v < 1e5 for v in costs.values())
    c = next(iter(costs))
    cl = sim.lookahead_clone(sample_scenarios(sim, 1, 20.0, t.key)[0], 20.0)
    cl.apply(c)
    from rl2.model import NetPolicy
    cl.run(NetPolicy(net))
    lam_n, lam_t = sim.cfg.blocking_penalty_per_event, sim.cfg.blocking_penalty_per_second
    expect = (cl.wip_area - sim.wip_area) + lam_n * (cl.n_block_events - sim.n_block_events) \
        + lam_t * (cl.blocked_area - sim.blocked_area)
    assert costs[c] == pytest.approx(expect, abs=1e-6)


def test_training_cells_draw_observation_levels_and_validation_cells_do_not():
    """0.7.9: training cells carry belts / zone / full (zone with none); the general
    validation cells (no p_obs) stay belts and are unchanged by the extra draw."""
    kinds, zone_modes = [], set()
    for i in range(120):
        cfg, kind = sample_config(100000 + i)
        group, mode, obs, _regime = kind.split("/")
        kinds.append(obs)
        assert cfg.simulation.observation == obs and cfg.laser.prepositioning == mode
        if obs == "zone":
            zone_modes.add(mode)
    assert {"belts", "zone", "full"} <= set(kinds)
    assert 0.5 < kinds.count("belts") / len(kinds) < 0.9                  # nominal 70 %
    assert zone_modes == {"none"}
    for seed in (2500, 2501, 2507):
        cfg, d = general_config(seed, mode="none")
        assert cfg.simulation.observation == "belts" and d["obs"] == "belts"
        cfg2, d2 = general_config(seed, mode="none", p_obs={"belts": 1.0})
        assert d2 == {**d, "obs": "belts"}
        assert cfg2.conveyors[0].speed == cfg.conveyors[0].speed


def test_training_cells_draw_regimes_and_validation_cells_do_not():
    """0.7.11: half the training cells get bursty arrivals and some general cells a
    slow head, from a separate random stream; every other parameter of a cell, and
    the validation cells, are unchanged."""
    from rl2.expert_iteration import BURST_VAL, VAL_GROUPS, draw_regime, val_config
    regimes, slow_v = [], []
    for i in range(160):
        cfg, kind = sample_config(100000 + i)
        group, mode, obs, regime = kind.split("/")
        regimes.append(regime)
        burst = cfg.arrivals.distribution == "burst"
        assert burst == ("burst" in regime)
        if burst:
            assert 0.2 <= cfg.arrivals.burst_on_fraction[0] <= 1.0
            assert 10.0 <= cfg.arrivals.burst_on_mean[0] <= 40.0
        if "slow" in regime:
            assert group == "general" and 2.5 <= cfg.laser.speed_x <= 5.0
            slow_v.append(cfg.laser.speed_x)
        elif group == "general":
            assert cfg.laser.speed_x >= 5.0
    bursty = sum("burst" in r for r in regimes) / len(regimes)
    assert 0.35 < bursty < 0.65 and len(slow_v) > 5                  # nominal 50 %, 30 % of general
    # the regime draw leaves every other parameter of the cell as it was
    for seed in (100003, 100011, 100042):
        r = draw_regime(seed)
        c0, d0 = general_config(seed, p_obs=GENERAL["p_observation"])
        c1, d1 = general_config(seed, p_obs=GENERAL["p_observation"], regime=r)
        keep = {k: v for k, v in d0.items() if k not in ("vx", "vy", "regime")}
        assert {k: d1[k] for k in keep} == keep
        assert c1.conveyors[0].speed == c0.conveyors[0].speed
        assert c1.arrivals.mean_interarrival == c0.arrivals.mean_interarrival
        if not r["slow"]:
            assert (c1.laser.speed_x, c1.laser.speed_y) == (c0.laser.speed_x, c0.laser.speed_y)
    # validation cells: Poisson arrivals and the main laser draw (no regime)
    for seed in (2500, 2501, 2507):
        cfg = val_config("general", seed)
        assert cfg.arrivals.distribution == "exponential" and cfg.laser.speed_x >= 5.0
    assert VAL_GROUPS == ("real", "heavy", "general", "real_burst")
    rb, real = val_config("real_burst", 2000), val_config("real", 2000)
    assert rb.arrivals.distribution == "burst" and real.arrivals.distribution == "exponential"
    assert rb.arrivals.burst_on_fraction == (0.3,) * 3 and rb.arrivals.burst_on_mean == (20.0,) * 3
    assert (rb.laser.speed_x, rb.laser.prepositioning) == (real.laser.speed_x, real.laser.prepositioning)
    assert len(BURST_VAL) == 3


# ---------------------------------------------------------- cell futures (0.7.12)
def _burst_cfg(seed=7, on_fraction=0.3, on_mean=20.0, **extra):
    return cfg(seed=seed, **{"arrivals.stochastic.distribution": "burst",
                             "arrivals.stochastic.burst.on_fraction": on_fraction,
                             "arrivals.stochastic.burst.on_mean": on_mean, **extra})


def test_cell_futures_equal_poisson_futures_on_a_poisson_cell():
    sim = FastSim(cfg(HEAVY, 4))
    first_multi_decision(sim)
    assert sample_scenarios(sim, 3, 40.0, "t", "cell") == sample_scenarios(sim, 3, 40.0, "t")
    # an on/off source with on_fraction 1 is Poisson too
    sim = FastSim(_burst_cfg(on_fraction=1.0))
    first_multi_decision(sim)
    assert sample_scenarios(sim, 2, 40.0, "t", "cell") == sample_scenarios(sim, 2, 40.0, "t", "poisson")
    with pytest.raises(ValueError):
        sample_scenarios(sim, 1, 40.0, "t", "gamma")
    with pytest.raises(ValueError):
        SearchTeacher(None, futures="gamma")


def test_cell_futures_follow_the_burst_model():
    """On a bursty cell the cell futures differ from the Poisson ones, keep the mean
    rate, are over-dispersed, stay in (t, t + window], and are reproducible."""
    sim = FastSim(_burst_cfg())
    first_multi_decision(sim)
    window, m = 400.0, 60
    cell = sample_scenarios(sim, m, window, "t", "cell")
    pois = sample_scenarios(sim, m, window, "t", "poisson")
    assert cell != pois
    assert cell == sample_scenarios(sim, m, window, "t", "cell")
    assert cell != sample_scenarios(sim, m, window, "u", "cell")
    rates = [1.0 / x for x in sim.cfg.arrivals.mean_interarrival]
    for k, lam in enumerate(rates):
        n_cell = [sum(1 for r in sc if r.conveyor_index == k) for sc in cell]
        n_pois = [sum(1 for r in sc if r.conveyor_index == k) for sc in pois]
        assert np.mean(n_cell) == pytest.approx(lam * window, rel=0.15)
        assert np.mean(n_pois) == pytest.approx(lam * window, rel=0.15)
        assert np.var(n_cell) / np.mean(n_cell) > 1.8 > 1.4 > np.var(n_pois) / np.mean(n_pois)
    for sc in cell:                       # per conveyor in time order (the clone sorts)
        for k in range(len(rates)):
            ts = [r.time for r in sc if r.conveyor_index == k]
            assert ts == sorted(ts) and all(sim.t < t <= sim.t + window for t in ts)
        assert all(r.processing_time > 0 for r in sc)


def test_teacher_with_cell_futures_runs_and_differs_on_a_bursty_cell():
    net = _small_net()
    sim = FastSim(_burst_cfg(seed=9, mode="when_idle"))
    cands = first_multi_decision(sim)
    kw = dict(k=6, horizon_s=math.inf, scenarios=2, window_s=30.0, w_pen=0.0,
              scale=RunningScale())
    tp = SearchTeacher(net, **kw)
    tc = SearchTeacher(net, futures="cell", **kw)
    a, b = tp.costs(sim, cands), tc.costs(sim, cands)
    assert set(a) == set(b) and all(math.isfinite(v) for v in b.values())
    assert b == tc.costs(sim, cands)                  # deterministic
    assert a != b                                     # different futures, different costs
    # the whole episode runs with the cell teacher and is reproducible
    s1 = FastSim(_burst_cfg(seed=9, mode="when_idle"))
    s1.run(SearchTeacher(net, futures="cell", **kw))
    s2 = FastSim(_burst_cfg(seed=9, mode="when_idle"))
    s2.run(SearchTeacher(net, futures="cell", **kw))
    assert s1.order == s2.order and s1.n_completed == len(s1.pk)
