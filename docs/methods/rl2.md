# Strong RL: representation, model and expert iteration (0.7.1–0.7.2, strong-RL plan phases 2–3)

Code: `src/rl2/features.py` (tokens), `src/rl2/model.py` (network, batching, policy,
save/load). Tests: `tests/test_rl2_model.py`. Latency: `experiments/rl2_latency.py`.
Plan: [../plans/strong_rl_plan.md](../plans/strong_rl_plan.md) §3. Changes and
rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.7.1).

Phase 2 (§1–§5) builds the model. Phase 3 (§6–§8) trains it by expert iteration on
**the general problem**: N belts with their own speeds and arrival rates, any
processing time, laser speeds, geometry. Final results come in phase 4.

## 1. Why a new representation

The 0.5–0.6 agent sees 87 hand-made features in 3 slots per belt. Its action is
"serve the head of conveyor k". That design has three limits (plan §1):

* it cannot see the whole queue;
* it is tied to N = 3 conveyors;
* it can only pick belt heads. With bursts, this restriction costs up to 3.5 %.

The new model reads the state as an **unordered set of tokens**, one per part, belt,
laser and context. Its action is a pointer to **any candidate part**.

## 2. Tokens (`rl2.features.encode`)

Scales:

* lengths are divided by the belt length L;
* speeds are divided by 3 cm/s, the real belt speed;
* times go through `slog(t) = sign(t)·log(1 + |t|/10 s)`, which keeps long waits at
  heavy load bounded but still distinguishable.

No token holds a list position or a conveyor index. A part is linked to its belt
through the y offset from the laser and a copy of its belt's state.

| token | count | features |
|---|---|---|
| part | one per arrived unfinished part (buffer, belt, being processed) | 23 features: status one-hot (waiting, moving, ready, processing); x/L; distance to b; time to b at belt speed; interception time from the laser and a reachable flag; waiting time; p_j / mean p; remaining processing time; blocking flag; rank on the belt among unstarted parts and a head flag; **candidate flag**; committed flag; x and y offset from the laser; nominal and current belt speed; belt running; belt blocked |
| conveyor | N | 11 features: y offset from the laser; nominal and current speed; running; stopped for processing; blocked; buffer length; parts on the belt; arrival rate; offered load; has a candidate |
| laser | 1 | 9 features: x, y; state one-hot (idle, heading, busy); remaining processing time; committed flag and target offset |
| context | 1 | 18 features: laser speeds; motion model; mean processing time; total offered load; a, b, spacing, workspace height (relative to L); pre-positioning mode one-hot; λ_N, λ_T; arrivals still coming; time left to the episode's arrival horizon (added in phase 3, for the value head); number of parts and of candidates |

* The context token carries the cell parameters. With it, one model can be trained
  with domain randomization (phase 3) and used on other cells.
* Encoding takes 0.14 ms per decision at default load and 0.19 ms at heavy load.

## 3. Network (`rl2.model.DispatchNet`)

* **Embedding:** one linear embedding per token type to width d, plus a learned type
  vector.
* **Encoder:** Transformer, pre-norm, ReLU feed-forward (4·d), **no positional
  encoding**. Padding is masked.
  * Self-attention without positions is permutation-equivariant.
  * g is the masked mean over all tokens.
* **Policy:** logit_i = MLP([h_i, g]) for each part token. Non-candidates and padding
  get −∞, so the action is **any candidate** of the open decision.
* **Value:** MLP([g, h_ctx, h_laser]) predicts the normalized cost-to-go.
  `RunningScale` keeps the normalization; phase 3 trains this head.
* **`NetPolicy`:** a greedy or sampled dispatcher for `FastSim.run`. When a decision
  has a single candidate, it takes it without calling the network.
* `save` / `load` store the size config together with the weights.

| size | layers × d × heads | parameters |
|---|---|---|
| **S** | 4 × 128 × 4 | **0.88 M** |
| M | 8 × 256 × 8 | 6.66 M |
| L | 12 × 512 × 8 | 39.2 M |

## 4. Acceptance (plan §3)

**Tests** (`tests/test_rl2_model.py`, 10 tests, all passing):

| test | what it checks |
|---|---|
| permutation invariance | Part and conveyor tokens of 30 real decision states are shuffled. Logits permute identically and the value is unchanged (|diff| < 1e-4). |
| masking | 5 random networks on 100+ states: non-candidates and padding get probability exactly 0 and are never the argmax or a sample. |
| batching | A padded batch gives the same output per state as single calls. |
| features | Shapes are right, values are finite and bounded, and the candidate flags equal `sim.candidates()`. |
| any candidate | Non-head candidates appear among the states, so the action space is wider than heads only. |
| valid episode | An untrained network drives a full FastSim episode (default/`none` and heavy/`when_idle`), and the formal checker finds 0 violations. |
| size and save/load | S has ~1 M parameters; weights survive a round trip. |
| latency | S forward pass < 25 ms (a loose bound for a loaded machine; the real numbers are below). |

**Latency** (`experiments/rl2_latency.py`; untrained weights; real decision states
from seeds 6100/6101):

| size | single state, 1 thread: median / p95 [ms], default load | heavy load | batch of 64, 8 threads [states/s], default / heavy |
|---|---|---|---|
| **S** | **4.1 / 5.3** | **4.7 / 5.8** | 2 360 / 1 580 |
| M | 15.4 / 17.8 | 18.9 / 22.7 | 440 / 265 |
| L | 64 / 78 | 81 / 103 | 87 / 53 |

* **Size S meets the plan's budget of < 5 ms per decision on the median at both
  loads. The p95 is 5.3–5.8 ms.**
  * The physical slack before a block is seconds, so this is enough for a
    controller.
  * Most of the time is per-layer overhead on very small tensors: about 1.2 ms fixed
    plus 0.85 ms per layer. It is not the arithmetic.
  * Switching from GELU to ReLU and from `no_grad` to `inference_mode` saved about
    10 %.
* **M and L are 4× and 16× slower** on this CPU. They are only viable as offline
  teachers or with a GPU, which confirms the plan's §0 choice of S.

## 5. Consequence for phase 3

A full FastSim episode driven by S runs at **340 decisions/s** on one thread. The
simulator alone runs at 5 500 decisions/s, so the network, not the simulator, is now
the bottleneck of network-guided search.

* **Per-decision network calls cost about 16× more than one simulation step.** Phase
  3 must therefore **batch network calls**: the search rollouts and scenarios of one
  decision advance in lock-step, and their states are evaluated together.
* A batch of 64 gives 1 600–2 400 states/s, against ~240 for single calls.
* Planned design for `src/rl2/search.py`:
  * K candidates × M scenarios rollouts are advanced in lock-step;
  * one batched forward pass per step;
  * the value head truncates the horizon, which cuts the number of network steps;
  * rollouts may use a cheap rule (`nearest_job`) where the network adds nothing.

## 5b. Reproduce (phase 2)

```bash
python -m pytest tests/test_rl2_model.py -q
python experiments/rl2_latency.py --sizes S M L --threads 8
```

The output goes to `results/experiments/rl2_latency/` (`summary.md`, `latency.json`).
Phase 3 commands are in §9.

## 6. Phase 3: the general problem (0.7.2)

**Target.** The user's instruction (2026-11-20) is that the problem to solve is the
general one:

* N conveyors with speeds v_1 … v_N;
* arrival rates λ_1 … λ_N;
* processing time t;
* laser speeds vx, vy;
* any other parameter that can be generalized.

The real cell (3 belts at 3 cm/s, arrivals every 10/12/14 s, 2.5 s, 10 cm/s) is one
instance of it.

The token model of §2 already accepts any N and any per-belt parameters. A first
training run (`rl2_s`) sampled only a few parameters of the 3-belt cell, so it was
stopped after its first iteration and replaced by the general training below. Its
first iteration is kept as a reference in §8.

### 6.1 Training distribution (`rl2.expert_iteration.GENERAL`, `sample_config`)

Every training episode (seeds 100000+) draws one cell:

* **25 %:** the real cell;
* **10 %:** the heavy example (`config/examples/heavy_load.yaml`);
* **65 %:** a general cell (`general_config`), every draw independent:

| parameter | range |
|---|---|
| number of belts N | 1 to 6 (uniform) |
| belt speed v_k | U[1.5, 6] cm/s, per belt |
| arrivals | Poisson per belt. First the processing load ρ = Σ_k λ_k · mean p ~ U[0.3, 0.95] is drawn, then it is split between the belts with weights U[0.5, 1.5]. This keeps cells neither empty nor hopelessly overloaded (the real cell has ρ = 0.64, the heavy example 0.93). |
| processing time | mean U[1.5, 4] s; constant, uniform (±30 %) or normal (σ = 15 %, min 50 %), one third each |
| laser vx, vy | U[5, 20] cm/s each, independent |
| laser motion | simultaneous axes (70 %) or sequential (30 %) |
| geometry | entry, laser-zone and exit lengths, belt spacing and minimum part spacing, each × U[0.5, 1.5] of the real value |
| blocking penalty | λ_N ~ U[0, 20] s per event, λ_T ~ U[0, 2] s per blocked belt-second |
| conveyor stop rule | only the processing belt (70 %) or all belts (30 %) |
| pre-positioning | `none` (70 %) or `when_idle` (30 %) |

All values reach the network through the conveyor and context tokens.

**Not generalized yet:**

* a processing-time distribution per belt or per part type (the configuration has one
  distribution);
* laser acceleration (a decision of the user);
* the `pass` rule (the fast simulator implements `block` only).

### 6.2 Search teacher (`rl2/search.py`)

* **Look-ahead copy.** `FastSim.lookahead_clone` drops the real future arrivals:
  online information only. A test checks that changing the real future does not
  change any teacher cost.
* **Score of a candidate.** For each candidate (all of them, or the top 6 by the base
  policy, always including the base policy's choice):
  1. force the candidate;
  2. let the base policy run until every part known at the decision is processed;
  3. the cost is the sum of their completion times. This is the online counterpart
     of the mean flow time, as in `rollout@avg_flow_time`.
* **Choice.** The base choice is scored first and is kept on ties (fortified
  rollout).
* **Base policy.** In iteration 0 it is `nearest_job`; the teacher is then the
  rollout of 0.4–0.6 on the exact simulator. In iteration k ≥ 1 it is the current
  network, as in the `rl_rollout` hybrid.
* **Lock-step batching.** All look-aheads of one decision advance together: each runs
  until it needs the network, then one batched forward pass serves them all. A test
  checks that this gives exactly the same costs as running each look-ahead alone.

**Teacher variants checked on validation seeds 2000–2019** (`none`, paired Δ vs
`nearest_job`, rule base):

| teacher | real cell | heavy example | time per episode |
|---|---|---|---|
| known parts, no penalty (**used**) | −3.8 ±2.2 % | −2.1 ±1.6 % | 0.15 s / 1.3 s |
| + blocking penalty in the look-ahead (w = 1) | −5.6 ±2.7 % | +5.8 ±3.0 % | 0.17 s / 1.4 s |
| + 4 sampled future-arrival scenarios (30 s) | +7.9 ±3.6 % | +13.2 ±2.9 % | 1.3 s / 8.7 s |

* Adding the penalty helps at the real load but hurts badly at heavy load. This agrees
  with the 0.4 finding that a penalty-aware look-ahead is worse overall.
* Sampled futures with a known-parts objective are much worse. A likely cause: the
  objective counts only the known parts, so look-aheads with synthetic arrivals favour
  starving the new parts. This was not investigated further. Scenarios stay available
  (`scenarios > 0`) but are not used.

### 6.3 Expert iteration (`rl2/expert_iteration.py`, `experiments/rl2_train.py`)

1. **Iteration 0.** The rule teacher acts, with 10 % random actions for coverage, and
   labels every decision with ≥ 2 candidates. `net_0` is trained from scratch.
2. **Iteration k ≥ 1.** The teacher is the rollout over `net_{k-1}`. The acting
   policy is `net_{k-1}`, except that the teacher acts with probability β = 0.3
   (DAgger). `net_k` is fine-tuned from `net_{k-1}` on a replay buffer of the last 2
   iterations.
3. **Losses.**
   * Policy: cross-entropy to the softmin of the relative look-ahead costs, with
     temperature 1 s.
   * Value: 0.25 × MSE of the normalized cost-to-go, where cost-to-go = remaining
     Σ F + λ_N ΔN_B + λ_T ΔT_B. The normalization uses a running mean/std of the
     targets.
   * Optimizer: AdamW with a one-cycle learning rate, batch 256, gradient clipping at 1.
4. **Validation and gating.**
   * Three groups, all with `none`: the real cell and the heavy example (seeds
     2000–2029), and 60 general cells (seeds 2500–2559).
   * Per group: the mean paired Δ of penalized flow time vs `nearest_job`, on the
     exact simulator.
   * The score is the mean of the three groups. A network is kept only if the score
     improves; training stops after 2 rejections or 6 iterations.
   * The teacher of every iteration is scored too.
5. **Logging.** Seeds, arguments, versions, times, losses and validation are written
   to `models/<tag>_training.json`.

## 7. Fast simulator on general cells (check before training)

The fast simulator was validated in 0.7.0 on the 3-belt cells only. It was rerun
against the engine on 40 general cells × 2 modes × 2 rules
(`experiments/fastsim_validation.py --loads general`, seeds 5500–5539).

* **Formal checker:** 0 violations.
* **Agreement with the engine at `dt = 0.01`:**
  * identical order in 66.9 % of runs;
  * KPI within 0.5 % in 49.4 %;
  * mean |KPI diff| 0.97 %, max 6.4 %.
  * This is worse than on the real cell (84.7 %, 0.40 %).
* **The engine is the one that is wrong.** Four of the worst cases were rerun with
  smaller engine steps:

  | seed | engine `dt` | same order | max start-time diff while orders agree [s] | KPI diff [%] |
  |---|---|---|---|---|
  | 5518 | 0.01 / 0.001 / 0.0002 | yes / yes / yes | 0.71 / 0.078 / 0.012 | −0.91 / −0.09 / −0.016 |
  | 5530 | 0.01 / 0.001 / 0.0002 | no / no / yes | 1.05 / 0.21 / 0.043 | +0.31 / −0.80 / −0.022 |
  | 5534 | 0.01 / 0.001 / 0.0002 | no / yes / yes | 0.72 / 0.072 / 0.015 | +0.20 / −0.05 / −0.010 |
  | 5523 | 0.01 / 0.001 / 0.0002 | yes / yes / yes | 0.59 / 0.077 / 0.014 | −0.56 / −0.08 / −0.014 |

  The difference shrinks linearly with the engine's step, so it is the engine's O(dt)
  lag per event. That lag grows in general cells, where belts and the laser are faster
  and events are more frequent.
* **Consequence for phase 4.** On general cells, engine results at `dt = 0.01` carry
  about 1 % of discretization error, which is comparable to the policy effects. Final
  numbers on general cells must use `dt ≤ 0.001` or the exact event-driven
  simulator. On the real cell the error is about 0.4 %.

## 8. Training results

### 8.1 First training run, narrow distribution (`rl2_s`, stopped after iteration 0)

* **Training mix:** 50 % real cell, 20 % heavy example, 30 % real cell with belt
  speed, processing time and an arrival factor randomized.
* **Data:** 800 episodes, 44 565 labelled decisions; 12 epochs (2 026 s).
* **Validation** (30 seeds per load, `none`, vs `nearest_job`):

  | policy | real cell | heavy example |
  |---|---|---|
  | `net_0` alone | **−3.8 ±1.8 %** | **−3.9 ±1.4 %** |
  | rollout over `net_0` | −6.0 ±2.2 % | −4.9 ±1.2 % |

* After one iteration, the network alone matches its teacher (the rule rollout, −3.8 %
  / −2.1 %) at no search cost. Search over the network gains another 1–2 points.
* The run was stopped to switch to the general problem (§6). The model and log are
  kept outside the repository, in `results/experiments/rl2_s_narrow/`.

### 8.2 General training with full information (`rl2_gen_s`, stopped)

* **What happened:** the first general run was stopped during its iteration 0, when
  it became clear that the controller of the real cell does not see the parts waiting
  upstream of the belts. The model and the teacher had been using them.
* **Fix and restart (0.7.3):** the tokens and the look-ahead now hold observable parts
  only (`simulation.observation`; real cell: `belts`). The context token gained the
  observation level (`CTX_F` 18 → 21), so earlier checkpoints no longer fit the model.
  The run was restarted from scratch.
* Its log is kept outside the repository in
  `results/experiments/archive/rl2_gen_s_fullinfo_stopped/`.

### 8.3 General training with the real information model (`rl2_gen_s`, 0.7.3)

RESULTS_PLACEHOLDER

## 9. Reproduce (phase 3)

```bash
python -m pytest tests/test_rl2_search.py -q
python experiments/fastsim_validation.py --loads general --seeds 40 --seed-base 5500 \
    --dt 0.01 --workers 8 --modes none when_idle --tag fastsim_val_general_dt01
python experiments/rl2_train.py --tag rl2_gen_s --iters 6 --episodes0 1000 --episodes 500 \
    --epochs0 8 --epochs 4 --replay 2 --val 30 --workers 8
```
