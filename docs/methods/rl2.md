# Strong RL: representation, model and expert iteration (0.7.1–0.7.6, strong-RL plan phases 2–3)

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
| part | one per arrived unfinished part (buffer, belt, being processed) | 23 features: status one-hot (waiting, moving, ready, processing); x/L; distance to b; time to b at belt speed; interception time from the laser and a reachable flag; waiting time since the part became observable (0.7.7; was since arrival); p_j / mean p; remaining processing time; blocking flag; rank on the belt among unstarted parts and a head flag; **candidate flag**; committed flag; x and y offset from the laser; nominal and current belt speed; belt running; belt blocked |
| conveyor | N | 11 features: y offset from the laser; nominal and current speed; running; stopped for processing; blocked; buffer length; parts on the belt; arrival rate; offered load; has a candidate |
| laser | 1 | 9 features: x, y; state one-hot (idle, heading, busy); remaining processing time; committed flag and target offset |
| context | 1 | 18 features: laser speeds; motion model; mean processing time; total offered load; a, b, spacing, workspace height (relative to L); pre-positioning mode one-hot; λ_N, λ_T; arrivals still coming (0.7.7: t < T_A, the known horizon; was read from the real arrival list); time left to the episode's arrival horizon (added in phase 3, for the value head); number of parts and of candidates |

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
| pre-positioning | `none` or `when_idle`, 50 % each since 0.7.7 (`when_idle` is the design mode of the cell, `none` the variant without entry detection; 70 / 30 up to 0.7.6) |
| observation level | `belts` 70 %, `zone` 15 % (forces `none`), `full` 15 % (0.7.9) |
| arrival regime (0.7.11) | Poisson, or with probability 0.5 the on/off source of `arrivals.stochastic.distribution: burst` with `on_fraction` ~ U[0.2, 1] and `on_mean` ~ U[10, 40] s at the same mean rate; also for the real and heavy training cells |
| slow head (0.7.11) | with probability 0.3 a general cell gets vx, vy ~ U[2.5, 5] cm/s instead of the main draw |

All values reach the network through the conveyor and context tokens, **except the
burst parameters and the regime itself** (0.7.11): the network knows the mean rates
and must infer the burstiness from what it sees, so the tokens and the saved models
stay compatible across 0.7.9–0.7.11. The observation level and the regimes are drawn
from their own random streams (`draw_observation`, `draw_regime`), so the other
parameters of a cell with a given seed, and the validation cells (which take no
draw), are unchanged by their introduction.

**Validation groups** (`VAL_GROUPS`): the real cell and the heavy example (seeds
2000+), 2n general cells (2500+, Poisson, main laser draw) and, since 0.7.11, the real
cell with bursty arrivals (`real_burst`: `on_fraction 0.3`, `on_mean 20 s`, seeds
2000+); the score is the mean of the four group deltas, so scores from 0.7.11 on are
not comparable with the three-group scores of §8.4–§8.6 (the per-group deltas are).

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
   * Three groups: the real cell and the heavy example (seeds 2000–2029) and 60
     general cells (seeds 2500–2559). Up to 0.7.6 all with `none`; since 0.7.7
     (`--val-mode design`, the default) the real and heavy cells use `when_idle`, the
     design mode, and the general cells draw their mode; `--val-mode none` reproduces
     the old validation.
   * Per group: the mean paired Δ of penalized flow time vs `nearest_job`, on the
     exact simulator.
   * The score is the mean of the three groups. A network is kept only if the score
     improves; training stops after 2 rejections or 6 iterations.
   * The teacher of every iteration is scored too.
5. **Logging.** Seeds, arguments, versions, times, losses and validation are written
   to `models/<tag>_training.json`.

### 6.4 A stronger teacher over the trained network (0.7.6)

The large runs of §8.4 showed that the known-parts look-ahead is the ceiling: every
network reaches it within 0.3–0.7 points and later iterations are rejected. The plan
(§4) names three levers: value-head truncation, sampled futures and a larger K. They
were compared over the trained S network on the 30-seed validation cells
(`experiments/rl2_teacher_variants.py`, paired Δ vs `nearest_job`; the network alone
gives −4.55 / −2.52 / −1.59 %, score −2.88 %). All variants score the **realized
cost-to-go of all parts** (Δ WIP integral, optionally + penalties) instead of the
completion times of the known parts, with common random numbers across candidates.

| teacher over the S network | real | heavy | general | score | s / episode (real / heavy / general) |
|---|---|---|---|---|---|
| known parts, K = 6 (0.7.2, used so far) | −5.23 ±1.91 | −2.69 ±1.72 | −2.49 ±1.52 | −3.47 | 2.0 / 7.0 / 3.1 |
| known parts, all candidates | −5.23 ±1.91 | −2.69 ±1.72 | −2.49 ±1.52 | −3.47 | 1.8 / 8.4 / 3.7 |
| value head at the first decision after t + 20 s, 4 futures | +18.1 ±4.5 | +37.9 ±3.0 | +22.8 ±7.5 | +26.3 | 7.9 / 26.3 / 13.1 |
| value head at exactly t + 20 s, 4 futures | +54.4 ±7.8 | +67.0 ±5.4 | +37.7 ±10.7 | +53.0 | 10.4 / 26.7 / 13.9 |
| 1 sampled future, 30 s window, no value head | −1.59 ±2.04 | +6.73 ±2.12 | +2.82 ±2.79 | +2.65 | 5.7 / 18.5 / 8.1 |
| 2 sampled futures, 30 s | −4.11 ±1.73 | +0.47 ±1.42 | +0.12 ±2.03 | −1.17 | 7.8 / 25.9 / 12.3 |
| 4 sampled futures, 30 s, penalty in the cost | −6.68 ±1.93 | −1.40 ±1.76 | −0.94 ±1.79 | −3.01 | 11.9 / 45.5 / 23.4 |
| 4 sampled futures, 30 s | −6.67 ±1.78 | −3.52 ±1.51 | −3.00 ±1.53 | −4.40 | 11.6 / 47.7 / 25.0 |
| 4 sampled futures, 20 s | −6.60 ±1.85 | −3.82 ±1.47 | −2.43 ±1.68 | −4.28 | 10.6 / 36.8 / 19.6 |
| 4 sampled futures, 45 s | −6.85 ±2.14 | −3.65 ±1.35 | −2.20 ±1.37 | −4.23 | 18.3 / 68.0 / 33.3 |
| 4 sampled futures, 60 s | −4.66 ±1.93 | −0.23 ±1.64 | −0.49 ±1.79 | −1.79 | 20.2 / 77.9 / 39.0 |
| 6 sampled futures, 30 s | −6.51 ±1.94 | −4.01 ±1.46 | −2.96 ±1.52 | −4.49 | 19.0 / 78.1 / 39.7 |
| **8 sampled futures, 30 s (used from 0.7.6)** | **−7.50 ±1.92** | **−3.83 ±1.33** | **−3.19 ±1.53** | **−4.84** | 21.9 / 112.2 / 58.3 |
| 8 sampled futures, 20 s | −7.39 ±1.83 | −3.53 ±1.57 | −3.15 ±1.51 | −4.69 | 17.7 / 85.9 / 42.4 |

Other rows checked: all candidates instead of K = 6 changes nothing with 4 futures
(the top 6 already hold the best); half-weight penalty −4.54; value-head cuts at 10 and
40 s, and without the penalty, all between +24 and +58 %. Full tables:
`results/experiments/rl2_teacher_variants_s{,2,3}/variants.json`.

* **The value head cannot bootstrap the look-ahead.** The trained cost-to-go depends
  mostly on the random future arrivals, so its scale (hundreds of part-seconds) and
  the value head's error (normalized MSE 0.06, i.e. a quarter of that scale) are far
  larger than the differences between candidates (seconds). Cutting at a fixed instant,
  where the value errors should be correlated, is even worse than cutting at a
  decision: the value head was trained on decision states only. Truncation with a
  learned value is therefore out until the value head is much more accurate (plan §4,
  Gumbel search, would need the same).
* **Sampled futures work once the objective counts every part.** The 0.7.2 attempt
  (§6.2, +7.9 %) counted only the known parts, so synthetic arrivals rewarded starving
  them. With the WIP integral of all parts, 4 futures already beat the known-parts
  teacher in every group and 8 futures gain another half point. One or two futures
  are worse than none: the sampled future is noise until it is averaged.
* **Window.** 20–30 s is best; 45–60 s add noise faster than information. The clone
  drains after the window, so the look-ahead is also bounded.
* **Penalty.** Including λ_N, λ_T in the look-ahead cost hurts (−3.01 vs −4.40 with 4
  futures), as in 0.4 and 0.7.2: the policies are judged on the penalized flow time,
  yet a flow-time look-ahead ranks candidates better.
* **Cost.** The chosen teacher is about 10× the known-parts one per episode (the
  look-aheads are 8× more numerous and longer). On the workstation that is 40 min per
  iteration of 2 000 episodes, acceptable.
* **What the futures know (0.7.12).** `SearchTeacher(futures=...)`,
  `sample_scenarios(..., futures)`: with `"poisson"` (every run up to 0.7.11) the
  futures are Poisson at the cell's mean rates on every conveyor; with `"cell"` they
  follow the cell's arrival model, i.e. on a conveyor with an on/off source
  (`arrivals.stochastic.distribution: burst`) the sampled future is an on/off Poisson
  source with the cell's `on_fraction` and `on_mean`, started in the stationary state
  (on with probability `on_fraction`: the controller knows the statistics of the
  process, not whether the source is on now). On a Poisson conveyor both draw exactly
  the same arrivals (same random stream), so the choice only matters on bursty cells.
  Measured in §8.9. The network's tokens do not carry the burst parameters in either
  case.

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

**Run.** Pilot on the laptop (Ryzen 7 5700U, 8 workers, CPU), 2026-11-20, 1 h 38 min:

* model S;
* 6 iterations: 1000 episodes and 8 epochs in iteration 0, then 500 episodes and
  4 epochs each, replay 2;
* teacher k = 6, `w_pen` 0;
* gating on 30 validation seeds per group (real 2000+, heavy 2000+, general 2500+,
  60 general episodes).

Log: `models/rl2_gen_s_training.json`. The accepted model is `models/rl2_gen_s.pt`
(iteration 3).

**Validation.** The table gives the paired Δ of penalized flow time vs `nearest_job`
(95 % CI, wins). "Network" is the network alone, a single forward pass per decision.
"Search" is the search teacher over the network.

| iteration | network real | network heavy | network general | search real | search heavy | search general | gate |
|---|---|---|---|---|---|---|---|
| 0 | −3.7 ±1.6 % | −2.0 ±1.6 % | −1.2 ±1.2 % | −5.5 % | −1.8 % | −2.6 % | accepted |
| 1 | −4.0 ±1.6 % | −1.6 ±1.8 % | −1.4 ±1.2 % | −5.6 % | −1.6 % | −2.4 % | accepted |
| 2 | −4.0 ±1.6 % | −3.2 ±1.5 % | −1.1 ±1.3 % | −5.4 % | −2.4 % | −2.7 % | accepted |
| **3** | **−3.4 ±1.7 %**, 25/30 | **−3.3 ±1.7 %**, 25/30 | **−1.8 ±1.4 %**, 24/60 | −5.5 ±1.8 % | −2.6 ±1.7 % | −2.6 ±1.6 % | **accepted (best)** |
| 4 | −3.9 ±1.6 % | −2.7 ±1.8 % | −1.4 ±1.4 % | −5.4 % | −2.6 % | −2.7 % | rejected |
| 5 | −3.9 ±1.8 % | −2.6 ±1.7 % | −1.3 ±1.3 % | −5.4 % | −2.3 % | −2.6 % | rejected |

Reference policies on the same validation seeds:

| policy | real | heavy | general |
|---|---|---|---|
| `nearest_head` | −1.6 ±1.7 % | +0.4 ±1.4 % | +1.1 ±1.5 % |
| search over `nearest_job` (rule teacher) | −4.9 ±1.8 % | +0.5 ±2.2 % | −2.2 ±1.3 % |

The run stopped after two rejected iterations (patience 2).

**Reading.**

* **The network alone beats `nearest_job` in all three groups**, significantly in each
  (the 95 % CI excludes 0). It is a single forward pass of about 4 ms. It also beats
  `nearest_head` in all three groups.
* **The network alone beats the rule teacher at heavy load** (−3.3 % vs +0.5 %). It is
  close to the rule teacher in the real cell (−3.4 % vs −4.9 %) and in general cells
  (−1.8 % vs −2.2 %).
* **Search over the network is the best policy** in every group: −5.5 % (real),
  −2.6 % (heavy), −2.6 % (general). Its cost is k = 6 look-aheads per decision.
* **Plateau.** After iteration 2, the training agreement with the teacher keeps
  rising (0.88 → 0.90), but validation no longer improves. The gap between network and
  search (about 2 points in the real cell, 1 point in general cells) is not closed by
  more iterations at this size and data volume. The large run on the other machine
  (sizes S and L, more episodes; `docs/methods/remote_training.md`) tests whether
  capacity or data is the limit.
* **Caveats.**
  * The validation seeds were used for gating, so the best iteration is selected on
    them and its numbers are slightly optimistic.
  * The unbiased comparison is the final evaluation on the reserved seeds 7000–7029
    (phase 4).
  * The general group has only 60 episodes over very different cells. Per-cell
    breakdowns are left for phase 4.

### 8.4 Large-scale runs on a GPU workstation (`rl2_gen_s_big`, `rl2_gen_m`, `rl2_gen_l`, 0.7.5)

The three sizes of the scaling ladder (plan §3) were trained on a 48-thread
workstation with an RTX 3080 (`docs/methods/remote_training.md`; CHANGELOG 0.7.5),
with the real information model of 0.7.3. Same code, same seeds (training 100000+,
validation 2000+ and 2500+), same gating; only the size and the amount of data differ.
Validation here uses 50 seeds per group (100 general cells), so the intervals are
narrower than in §8.1.

| run | size | parameters | episodes it. 0 / later | data per iteration | iterations run | best |
|---|---|---|---|---|---|---|
| `rl2_gen_s_big` | S | 0.88 M | 4 000 / 2 000 | 194 k / 98 k decisions | 0–5 | it. 3 |
| `rl2_gen_m` | M | 6.7 M | 8 000 / 4 000 | 394 k / 197 k | 0–5 | it. 3 |
| `rl2_gen_l` | L | 39 M | 16 000 / 8 000 | 793 k / 394 k | 0–1 (iteration 2 running) | it. 1 |

Both S and M stopped by the rule of §6.3 (two rejections after their iteration 3).
Reference on the same 50/100 seeds: `nearest_head` −1.81 ±1.32 / +0.21 ±0.96 /
+0.89 ±1.02 %; search over `nearest_job` −4.39 ±1.35 / +0.09 ±1.62 / −1.60 ±1.05 %
(real / heavy / general, paired Δ vs `nearest_job`).

**Per iteration, size S** (`net` = network alone, `search` = rollout over that network):

| it | net real | net heavy | net general | score | search real | search heavy | search general | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −3.65 ±1.41 | −0.61 ±1.67 | −1.38 ±0.97 | −1.88 | −4.97 ±1.52 | −1.99 ±1.29 | −2.49 ±1.06 | yes | 18 / 115 / 18 |
| 1 | −4.15 ±1.48 | −0.68 ±1.71 | −1.78 ±1.05 | −2.20 | −4.68 ±1.51 | −2.40 ±1.19 | −2.33 ±1.06 | yes | 161 / 85 / 20 |
| 2 | −4.33 ±1.40 | −1.89 ±1.66 | −1.87 ±1.07 | −2.70 | −5.00 ±1.59 | −2.21 ±1.43 | −2.43 ±1.08 | yes | 189 / 57 / 20 |
| **3** | **−4.42 ±1.44** | **−2.32 ±1.63** | **−1.64 ±1.06** | **−2.79** | −4.71 ±1.50 | −2.23 ±1.39 | −2.37 ±1.05 | yes | 190 / 59 / 21 |
| 4 | −4.69 ±1.45 | −1.60 ±1.61 | −1.45 ±1.10 | −2.58 | −4.84 ±1.49 | −2.19 ±1.37 | −2.39 ±1.04 | no | 177 / 57 / 21 |
| 5 | −4.44 ±1.47 | −1.77 ±1.59 | −1.47 ±1.04 | −2.56 | −4.72 ±1.52 | −2.26 ±1.30 | −2.25 ±1.04 | no | 174 / 55 / 20 |

**Per iteration, size M:**

| it | net real | net heavy | net general | score | search real | search heavy | search general | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −4.30 ±1.28 | −0.81 ±1.29 | −1.75 ±0.99 | −2.29 | −4.94 ±1.53 | −1.70 ±1.33 | −2.05 ±1.21 | yes | 38 / 729 / 165 |
| 1 | −4.59 ±1.32 | −1.20 ±1.75 | −1.58 ±1.02 | −2.46 | −4.85 ±1.51 | −2.30 ±1.23 | −2.26 ±1.12 | yes | 3 253 / 548 / 187 |
| 2 | −4.27 ±1.49 | −2.28 ±1.48 | −1.25 ±1.12 | −2.60 | −4.92 ±1.49 | −2.45 ±1.24 | −2.29 ±1.07 | yes | 3 577 / 351 / 186 |
| **3** | **−4.49 ±1.55** | **−2.22 ±1.38** | **−1.91 ±1.08** | **−2.88** | −4.96 ±1.50 | −2.35 ±1.25 | −2.24 ±1.21 | yes | 3 437 / 331 / 185 |
| 4 | −4.14 ±1.49 | −2.08 ±1.34 | −1.61 ±1.10 | −2.61 | −5.03 ±1.51 | −2.67 ±1.28 | −2.42 ±1.04 | no | 3 528 / 346 / 185 |
| 5 | −4.41 ±1.48 | −2.05 ±1.29 | −1.53 ±1.04 | −2.67 | −5.09 ±1.49 | −2.46 ±1.22 | −1.87 ±1.17 | no | 3 494 / 342 / 182 |

**Per iteration, size L:**

| it | net real | net heavy | net general | score | search real | search heavy | search general | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −3.88 ±1.38 | −1.07 ±1.76 | −1.49 ±1.03 | −2.15 | −5.02 ±1.48 | −1.73 ±1.33 | −2.02 ±1.09 | yes | 77 / 5 184 / 1 108 |
| **1** | **−4.25 ±1.53** | **−1.67 ±1.32** | **−1.40 ±1.04** | **−2.44** | −4.90 ±1.50 | −1.77 ±1.27 | −2.21 ±1.03 | yes | 39 250 / 3 805 / 1 100 |

Iteration 1 of L took 10 h 54 min to generate (8 000 episodes labelled by the search
over the L network, 4.9 s each on the CPU workers) and was accepted; after two
iterations L (−2.44 %) is still below the best S (−2.79 %) and M (−2.88 %).

**Best models on the 30-seed validation set of §8.1 and of the laptop references**
(seeds 2000–2029 and 2500–2559, `results/experiments/compare_val30_*.json`):

| policy | real cell | heavy example | general cells |
|---|---|---|---|
| `nearest_head` | −1.60 ±1.68 | +0.37 ±1.38 | +1.15 ±1.54 |
| search over `nearest_job` (rule teacher) | −4.86 ±1.81 | +0.48 ±2.22 | −2.18 ±1.32 |
| **S alone** (`rl2_gen_s_big.pt`) | −4.55 ±1.88 | −2.52 ±2.26 | −1.59 ±1.42 |
| search over S | −5.23 ±1.91 | −2.69 ±1.72 | −2.49 ±1.52 |
| **M alone** (`rl2_gen_m.pt`) | −5.25 ±2.03 | −1.96 ±1.75 | −2.02 ±1.42 |
| search over M | −5.59 ±1.85 | −2.45 ±1.47 | −2.43 ±1.76 |
| L alone (it. 0) | −4.01 ±1.58 | −0.69 ±2.37 | −1.47 ±1.42 |
| search over L (it. 0) | −5.63 ±1.89 | −1.95 ±1.71 | −2.20 ±1.61 |

* The `nearest_head` row reproduces the laptop reference (−1.6 / +0.4 / +1.2 %)
  exactly, so the seeds and groups are the same.
* The rule teacher differs from the laptop reference at heavy load (+0.5 % here,
  −2.2 % there). The laptop reference predates 0.7.3: its look-ahead still saw the
  parts waiting upstream, which is worth about 2.7 points at heavy load. The real and
  general values agree. For the same reason the narrow pilot of §8.1 (−3.9 % heavy,
  full information) is not comparable at heavy load.

**Findings.**

* **Every network alone beats its rule teacher at the real load and generalizes.**
  S alone gives −4.4 % (real) with 50 seeds, M −4.5 %, against −4.4 % for the search
  over `nearest_job`, at no search cost; both give about −2 % on the general cells
  where `nearest_head` loses.
* **The search over the network saturates at −4.7 … −5.1 / −2.2 … −2.7 /
  −2.2 … −2.4 %** from iteration 1 on, for both sizes. From iteration 2 the network
  alone is within 0.3–0.7 points of its own teacher, so the teacher has little left to
  teach: the limit is the depth of the look-ahead (known parts, `nearest_job`-style
  horizon), not the capacity of the model. Later iterations trade one group against
  another within the confidence intervals and are rejected. A stronger teacher (value-
  head truncation with sampled futures, larger K, or the Gumbel-MCTS option of plan
  §4) is the next lever, not a larger network.
* **M versus S.** With twice the data, M's best score is −2.88 % against −2.79 %:
  inside the confidence intervals. On the 30-seed set M is better at the real load
  (−5.3 vs −4.6 %) and on general cells (−2.0 vs −1.6 %) and worse at heavy load
  (−2.0 vs −2.5 %), all within ±2 points. M costs 8.5× more per generated episode
  (the teacher runs the network on the CPU workers) and 4× per decision at inference
  (15–19 ms vs 4 ms, §4): the gain does not pay for the cost under the 5 ms budget.
  **L** at iteration 0, with 4× the data of S, scores −2.15 %: between S (−1.88 %)
  and M (−2.29 %) at the same stage, with a higher final training loss (0.469 vs
  0.454 for M) after the same 8 epochs; at iteration 1 it reaches −2.44 % against
  −2.20 % (S) and −2.46 % (M). The 39 M model is not under-fitted by lack
  of data; it learns the same teacher more slowly, and its teacher runs about 6×
  slower than M's (validation 1 108 s vs 185 s). On this evidence the ladder stops at
  M: more parameters do not raise the ceiling set by the teacher.
* **Training losses** keep falling across iterations (policy cross-entropy 0.465 →
  0.404 for S, 0.454 → 0.391 for M; agreement with the teacher's argmax 0.88 → 0.92)
  while validation stalls: the networks fit their teacher ever better, which confirms
  that the teacher is the ceiling.

**Times on the workstation** (47 simulation workers, trainer on the GPU):

| step | S | M | L |
|---|---|---|---|
| generation, rule teacher (it. 0) | 4 000 episodes in 18 s | 8 000 in 38 s | 16 000 in 77 s |
| generation, network teacher (it. ≥ 1) | 2 000 episodes in 160–190 s (0.09 s each) | 4 000 in 3 250–3 580 s (0.85 s each) | 8 000 in 39 250 s (4.9 s each) |
| training, per epoch | 14 s for 194 k decisions (≈ 13 500 decisions/s) | 90 s for 394 k (≈ 4 400/s) | 650 s for 793 k (≈ 1 200/s) |
| validation (50/50/100 cells × net, search, `nearest_job`) | 20 s | 185 s | 1 108 s |
| whole run | 25 min (6 iterations) | 5 h 53 min (6 iterations) | 14 h for iterations 0–1 (12 h per iteration ≥ 1) |

The laptop needed 4–5 min per epoch of 50 000 decisions on the CPU; the GPU does the
same in about 4 s. On this machine the bottleneck moved to data generation with a
network teacher, which runs single-state network calls on the CPU workers.

### 8.5 Expert iteration with the sampled-future teacher (`rl2_gen_s_sf`, 0.7.6)

Continuation of the best S model (`rl2_gen_s_big.pt`, §8.4) with the teacher of
§6.4 (8 sampled futures, 30 s window, no penalty in the look-ahead): `--init`, so the
run starts at iteration 1; 2 000 episodes and 4 epochs per iteration, replay 2, 50
validation seeds per group (100 general), patience 2. Same seeds as §8.4 (training
continues from 100000; the run's seeds are its own).

Run of 2026-11-21/22 on the workstation (3 h 53 min; 47 workers; about 50 min per
iteration to generate, 1 min to train, 5 min to validate). Validation in `none`
(the mode of every result up to 0.7.6) with the 0.7.6 tokens, i.e. before the two
leak fixes of 0.7.7 (which change the numbers by less than the confidence intervals,
`CHANGELOG` 0.7.7).

| it | net real | net heavy | net general | score | search real | search heavy | search general | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −7.73 ±1.49 | −3.55 ±1.26 | −2.23 ±0.98 | −4.50 | −7.64 ±1.57 | −3.93 ±1.16 | −2.70 ±1.11 | yes | 2 868 / 28 / 309 |
| **2** | **−7.64 ±1.44** | **−4.42 ±1.13** | **−2.42 ±1.10** | **−4.83** | −7.20 ±1.60 | −3.52 ±1.22 | −2.72 ±1.16 | yes | 2 980 / 55 / 313 |
| 3 | −6.99 ±1.38 | −3.71 ±1.25 | −2.47 ±1.05 | −4.39 | −7.83 ±1.50 | −3.30 ±1.14 | −2.73 ±1.14 | no | 3 323 / 55 / 320 |
| 4 | −7.16 ±1.48 | −2.91 ±1.40 | −2.83 ±1.09 | −4.30 | −7.63 ±1.52 | −3.39 ±1.09 | −2.75 ±1.15 | no | 3 335 / 55 / 291 |

Reference on the same seeds: `nearest_head` −1.81 / +0.21 / +0.89 %; search over
`nearest_job` −4.39 / +0.09 / −1.60 %; best S of §8.4 (known-parts teacher)
−4.42 / −2.32 / −1.64 %, score −2.79 %.

* **The network absorbed the new teacher in one iteration.** From −2.79 % (best of
  §8.4) to −4.50 % after iteration 1 and **−4.83 %** after iteration 2: −7.6 % at the
  real load, −4.4 % at heavy load and −2.4 % on the general cells, at 4 ms per
  decision. It matches or beats every rule and every earlier hybrid in all three
  groups.
* **The network reaches its teacher again.** From iteration 2 the network alone is at
  or above the search over itself at the real and heavy loads (−7.6 vs −7.2 %, −4.4 vs
  −3.5 %): the search is noisy (8 sampled futures) and the network learns its average.
  Iterations 3 and 4 trade groups within the intervals and are rejected; the run
  stopped by the patience rule. The next step is again a stronger teacher or more
  futures, not more iterations.
* **Training losses** fall from 0.377 to 0.314 (policy cross-entropy) while validation
  stalls, as in §8.4.
* `models/rl2_gen_s_sf.pt` is the iteration-2 model; `models/rl2_gen_s_sf_training.json`
  the log. On the design mode (`when_idle`, 30 seeds) this model gives −13.8 % at the
  real load (`docs/assumptions.md` C4).


### 8.6 Design mode, corrected tokens and observation levels (`rl2_gen_s_d`, 0.7.9)

Continuation of `rl2_gen_s_sf.pt` (§8.5) with the same teacher (8 sampled futures,
30 s), the tokens of 0.7.7, pre-positioning `when_idle` in half of the training cells
and the observation levels of 0.7.9 (`belts` 70 %, `zone` 15 %, `full` 15 %).
Validation in the design mode: the real cell and the heavy example with `when_idle`,
general cells with the mode drawn. The `nearest_job` reference of every paired Δ is
run in the same mode as the policy, so these Δ measure the policy alone; the value of
pre-positioning itself is in `docs/assumptions.md` C4.

Run of 2026-11-22 on the workstation (4 h 21 min; iterations 1–4, stopped by the
patience rule; best **iteration 2**). Reference on the same seeds in the design mode
(both policies with `when_idle`): `nearest_head` −0.37 ±1.34 / +0.48 ±1.02 /
+1.23 ±1.07 %; search over `nearest_job` −3.73 ±1.37 / +0.51 ±1.33 / −1.74 ±1.13 %.

| it | net real | net heavy | net general | score | search real | search heavy | search general | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −6.40 ±1.50 | −2.04 ±1.28 | −3.08 ±1.08 | −3.84 | −6.54 ±1.32 | −2.86 ±1.28 | −3.06 ±1.20 | yes | 3 275 / 32 / 310 |
| **2** | **−6.61 ±1.40** | **−2.45 ±1.20** | **−2.88 ±1.07** | **−3.98** | −6.69 ±1.31 | −3.14 ±1.22 | −3.22 ±1.29 | yes | 3 431 / 62 / 305 |
| 3 | −6.67 ±1.36 | −2.40 ±1.32 | −2.87 ±1.06 | −3.98 | −6.40 ±1.29 | −2.29 ±1.36 | −3.31 ±1.29 | no | 3 718 / 61 / 291 |
| 4 | −6.37 ±1.43 | −2.45 ±1.32 | −2.82 ±1.05 | −3.88 | −6.54 ±1.30 | −2.89 ±1.32 | −3.17 ±1.20 | no | 3 765 / 62 / 306 |

* In the design mode the paired Δ is measured against `nearest_job` **with**
  pre-positioning, which already gains 7.4 points on its own at the real load
  (`docs/assumptions.md` C4), so the network's −6.6 % here is on top of that: against
  `nearest_job` without pre-positioning it is −13.3 % (table below).
* The network is again at its teacher from iteration 2 (−6.6 vs −6.7 % real); the
  run stopped after two iterations within the intervals.
* `models/rl2_gen_s_d.pt` is the iteration-2 model, `models/rl2_gen_s_d_training.json`
  the log.

**Information treatments A/B/C/D with one network** (`results/experiments/abcd_rl2_gen_s_d.json`;
30 seeds, paired Δ vs `nearest_job` in B; A = `zone`/`none`, B = `belts`/`none`,
C = `belts`/`when_idle`, D = `full`/`none`, the treatments of `docs/plans/papers.md`):

| policy | real A | real B | real C | real D | heavy A | heavy B | heavy C | heavy D |
|---|---|---|---|---|---|---|---|---|
| `nearest_job` | 0 | 0 | −7.43 ±2.48 | 0 | 0 | 0 | +0.05 ±1.54 | 0 |
| `nearest_head` | −1.60 ±1.68 | −1.60 ±1.68 | −8.14 ±2.22 | −1.60 ±1.68 | +0.37 ±1.38 | +0.37 ±1.38 | +0.75 ±1.48 | +0.37 ±1.38 |
| **`rl2_gen_s_d` alone** | −7.72 ±1.99 | −8.02 ±1.75 | **−13.33 ±1.81** | −7.34 ±1.83 | −3.77 ±1.46 | −4.01 ±1.81 | −3.26 ±1.62 | −5.24 ±1.21 |

Paired contrasts of the network between treatments (same seeds): **B vs A −0.22 ±1.33 %**
(real) and −0.22 ±1.35 % (heavy); **C vs B −5.67 ±1.71 %** (real) and +0.91 ±1.64 %
(heavy); **D vs B +0.76 ±0.92 %** (real) and −1.16 ±1.24 % (heavy).

* **Seeing the entry zone without being allowed to act on it is worth nothing** (B vs
  A), at either load and with a network that anticipates the future. The rules are
  identical in A, B and D by construction (they choose among candidates only).
* **Acting on it is worth 5.7 points** at the design load (C vs B) and nothing at the
  heavy load, where the head is never idle.
* **Seeing the hidden upstream queue** (D) adds nothing at the design load and about
  1.2 points at the heavy load.
* Caveat: the network saw `zone` cells in 15 % of its training episodes only; a
  network trained more on A could narrow or widen the B–A gap. This is the first
  measurement of the main contrast of Paper 2, on the nominal cell only.

### 8.7 Information bench across regimes with `rl2_gen_s_d` (0.7.10)

Pilot of 2026-11-22 (`experiments/rl2_information_bench.py --tag info_pilot_s_d`,
~5 min on 47 workers; `results/experiments/info_pilot_s_d/`): the network of §8.6
unchanged, the rules `nearest_job` and `nearest_head`, 30 validation seeds (2000–2029)
per regime, paired. Regimes: load (real, heavy) × arrivals (regular = Poisson; burst =
on/off source, `on_fraction 0.3`, `on_mean 20 s`, same mean rate) × head travel
(nominal; slow = laser speeds × 0.5, i.e. 5 cm/s on the real cell and 2.5 cm/s on the
heavy one). Treatments as in §8.6; the visible-length curve uses `belts` with
`simulation.visible_length` ∈ {0, 2.5, 5, 10} cm (10 = the whole entry zone = B/C).

**Network contrasts between treatments** (paired Δ of the penalized flow time, %):

| load | arrivals | travel | `nearest_job` in B [s/part] | net in B vs `nearest_job` | B vs A (see without acting) | C vs B (act) | D vs B (hidden queue) |
|---|---|---|---|---|---|---|---|
| real | regular | nominal | 20.0 | −8.02 ±1.75 | −0.22 ±1.33 | **−5.67 ±1.71** | +0.76 ±0.92 |
| real | regular | slow | 40.9 | −7.33 ±2.03 | +0.26 ±1.45 | **−2.98 ±2.25** | −0.32 ±1.11 |
| real | burst | nominal | 29.3 | −2.87 ±1.18 | +0.26 ±0.41 | **−3.13 ±1.09** | −0.56 ±0.53 |
| real | burst | slow | 41.5 | −4.54 ±1.74 | +0.05 ±0.83 | **−3.96 ±1.56** | +0.13 ±0.64 |
| heavy | regular | nominal | 70.6 | −4.01 ±1.81 | −0.22 ±1.35 | +0.91 ±1.64 | −1.16 ±1.24 |
| heavy | regular | slow | 101.4 | +0.13 ±2.98 | −0.88 ±1.82 | +0.46 ±2.61 | **−4.35 ±1.86** |
| heavy | burst | nominal | 73.3 | −4.93 ±1.32 | −0.34 ±0.78 | −0.57 ±1.56 | −0.36 ±0.92 |
| heavy | burst | slow | 97.0 | −4.58 ±1.66 | −0.43 ±0.36 | −0.33 ±1.22 | −0.03 ±1.11 |

**Visible-length curve with pre-positioning** (`when_idle`; Δ vs `nearest_job` in B,
%; the rules with `none` and the network with `none` are flat in ℓ_v in every regime,
within ±0.5 points, as B vs A says):

| regime | policy | ℓ_v = 0 | 2.5 cm | 5 cm | 10 cm (= C) |
|---|---|---|---|---|---|
| real / regular / nominal | net | −7.31 ±1.98 | −10.64 ±1.72 | −12.77 ±1.70 | −13.33 ±1.81 |
| | `nearest_job` | 0 | −4.21 ±1.35 | −5.98 ±2.37 | −7.43 ±2.48 |
| real / regular / slow | net | −7.28 ±2.44 | −8.71 ±2.10 | −9.33 ±2.38 | −10.11 ±2.83 |
| | `nearest_job` | 0 | −0.48 ±1.96 | +0.51 ±2.26 | −2.26 ±2.36 |
| real / burst / nominal | net | −2.94 ±1.16 | −5.06 ±1.35 | −5.63 ±1.26 | −5.91 ±1.47 |
| | `nearest_job` | 0 | −2.48 ±1.01 | −2.91 ±1.41 | −2.88 ±1.51 |
| real / burst / slow | net | −4.46 ±1.58 | −6.50 ±1.76 | −7.54 ±1.88 | −8.41 ±1.72 |
| | `nearest_job` | 0 | −2.52 ±1.28 | −2.53 ±1.59 | −3.88 ±1.67 |
| heavy (all four regimes) | both | flat: every point within ±1.6 of B | | | |

**Gain of pre-positioning at each visible length** (paired Δ of `when_idle` versus
`none` **at the same ℓ_v**, %; this isolates what advancing the head adds at a given
coverage, which is the curve Paper 2 should report; anticipation in seconds at the belt
speed, 3 cm/s on the real cell):

| regime | policy | ℓ_v = 0 (0 s) | 2.5 cm (0.8 s) | 5 cm (1.7 s) | 10 cm (3.3 s) |
|---|---|---|---|---|---|
| real / regular / nominal | net | −0.03 ±0.64 | −2.96 ±1.59 | **−5.52 ±1.56** | −5.67 ±1.71 |
| | `nearest_job` | 0 | −4.21 ±1.35 | −5.98 ±2.37 | −7.43 ±2.48 |
| real / regular / slow | net | −0.08 ±0.92 | −1.22 ±2.01 | −1.57 ±2.15 | −2.98 ±2.25 |
| | `nearest_job` | 0 | −0.48 ±1.96 | +0.51 ±2.26 | −2.26 ±2.36 |
| real / burst / nominal | net | −0.12 ±0.13 | −1.88 ±0.87 | −2.76 ±1.14 | −3.13 ±1.09 |
| | `nearest_job` | 0 | −2.48 ±1.01 | −2.91 ±1.41 | −2.88 ±1.51 |
| real / burst / slow | net | +0.35 ±0.39 | −2.08 ±0.83 | −3.15 ±1.28 | −3.96 ±1.56 |
| | `nearest_job` | 0 | −2.52 ±1.28 | −2.53 ±1.59 | −3.88 ±1.67 |
| heavy (all four regimes) | both | within ±1 point at every ℓ_v (no idle head) | | | |

With the nominal head, 5 cm of coverage (1.7 s) gives 97 % of the pre-positioning gain
of the whole zone (−5.52 of −5.67); with a head half as fast only 53 % (−1.57 of
−2.98); with bursts 88 % (nominal head) and 80 % (slow head). The coverage the
pre-positioning needs therefore grows with the head's travel time across the workspace
(3.5 s nominal, 7 s slow), which is the axis against which the paper should plot it.

Findings:

* **Seeing the entry zone without acting on it is worth nothing in every regime**
  tested: B vs A is within ±0.9 points in all eight, for the network as for the rules
  (identical candidates by construction). Bursts and slow heads do not create a value
  of information alone. The main hypothesis of Paper 2 as written (upstream visibility
  is worth more under bursty arrivals and long travel) is **not supported** by this
  pilot at this burst setting; what the visibility buys is the **action** it enables.
* **The value of acting is largest on the nominal regime** (−5.7 points at the real
  load) and smaller, not larger, with bursts (−3.1) or a slow head (−3.0): bursts
  leave the head idle during the gaps with nothing to pre-position to, and a slow head
  is idle less often. It is zero at the heavy load in every regime (the head is never
  idle). `nearest_job` with `when_idle` gains the same order (−7.4 / −2.3 / −2.9 /
  −3.9 points), so this is a property of the cell, not of the learned policy.
* **The curve saturates at about half the entry zone**: with 5 cm visible (1.7 s of
  belt travel at 3 cm/s) the network reaches −12.8 of the −13.3 points it gets with
  the whole zone on the nominal regime, and the shape is the same in the other real-
  load regimes. For a cell design this says how far upstream the camera must see for
  pre-positioning to pay: a few seconds of belt travel, roughly the head's travel time
  across the workspace, which is the quantity the curve should be plotted against in
  the paper.
* **The hidden upstream queue (D) is worth 4.4 points only at the heavy load with a
  slow head**, where every belt queues and knowing the queue lengths changes which
  belt to serve; elsewhere it is within ±1.2 points. Note that this regime (laser at
  2.5 cm/s) lies **outside the training range** of the network (`GENERAL` laser
  speeds 5–20 cm/s) and there the network alone no longer beats `nearest_job`
  (+0.13 ±2.98 in B); its D result there should be taken as indicative.
* **Generalization to bursts without retraining**: trained on Poisson arrivals only,
  the network keeps −2.9 to −4.9 points over `nearest_job` on the bursty regimes (less
  than its −8.0 on the regular real cell). Whether training on bursty cells recovers
  the gap is the retraining question left open in 0.7.10.
* Consistency checks in the tables: `belts` with ℓ_v = 0 reproduces A for the rules
  exactly and for the network within the intervals (its context token differs); the
  `none` columns are identical across A, B, D and every ℓ_v for the rules.
* Limits of the pilot: one burst setting (0.3 / 20 s), one slow factor, two cells, the
  validation seeds; the confirmatory bench of the plan (36 configurations, fresh seeds
  7000–7029) is still to run, once the regimes and the training mix are frozen.
* Decisions taken by the owner on this pilot (hypothesis reformulated, retraining
  conditional on the scope, phase 4 after freezing): `docs/plans/papers.md`.

### 8.8 Retraining with bursty arrivals and slow heads in the mix (`rl2_gen_s_b`, 0.7.11)

The owner included bursty arrivals and slow heads in the scope (2026-11-22). Run
`rl2_gen_s_b`: size S, continuation of `rl2_gen_s_d.pt`, the sampled-future teacher
(8 futures, 30 s window, w_pen 0), validation in the design mode on the four groups
of §6.1 with 50 seeds (100 general), 8 iterations at most, patience 2, 2 000 episodes
per iteration, replay 2. The training mix is that of §6.1 with the 0.7.11 rows; the
teacher's futures stay Poisson at the mean rates. `rl2_gen_s_d` is kept as the
reference; the choice between the two is to be made on development seeds, never on
the confirmatory seeds 7000–7029.

Run of 2026-11-22 on the workstation (12:33–16:38, 4 h 05 min; iterations 1–3,
stopped by the patience rule; best **iteration 1**, score −3.82 % over the four
groups). Reference rows on the same seeds (design mode): `nearest_head` −0.37 / +0.48 /
+1.23 / +0.06 %, search over `nearest_job` −3.73 / +0.51 / −1.74 / −1.84 % (real /
heavy / general / real_burst); the first three are identical to those of §8.6, which
confirms that the validation cells did not change with the 0.7.11 draws.

| it | net real | net heavy | net general | net real_burst | score | search real | search heavy | search general | search real_burst | accepted | gen / train / val [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **−6.19 ±1.48** | **−2.86 ±1.09** | **−2.73 ±1.05** | **−3.51 ±1.00** | **−3.82** | −6.44 ±1.30 | −2.35 ±1.46 | −3.51 ±1.27 | −4.07 ±0.96 | yes | 4 228 / 37 / 361 |
| 2 | −6.20 ±1.36 | −2.60 ±1.23 | −2.73 ±1.06 | −3.66 ±0.99 | −3.80 | −6.34 ±1.31 | −2.55 ±1.21 | −3.12 ±1.22 | −3.90 ±0.97 | no | 4 576 / 79 / 363 |
| 3 | −6.48 ±1.45 | −2.25 ±1.06 | −2.72 ±1.08 | −3.74 ±0.92 | −3.80 | −6.58 ±1.29 | −2.50 ±1.30 | −3.06 ±1.28 | −3.80 ±0.95 | no | 4 599 / 81 / 357 |

`models/rl2_gen_s_b.pt` is the iteration-1 model (weights identical to
`rl2_gen_s_b_it1.pt`), `models/rl2_gen_s_b_training.json` the log. Generation took
30–40 % longer than in §8.6 (bursty and slow-head cells have more decisions).

**Head to head with the reference** `rl2_gen_s_d` (paired cost ratio, design mode,
validation seeds; negative = the new network is better):

| group | real (50) | heavy (50) | general (100) | real_burst (50) |
|---|---|---|---|---|
| `rl2_gen_s_b` vs `rl2_gen_s_d` | +0.47 ±0.69 % | −0.36 ±0.83 % | +0.20 ±0.48 % | +0.09 ±0.36 % |

The two networks are indistinguishable, also on the bursty cell: `rl2_gen_s_d` alone
already scores −3.59 ±0.98 % on `real_burst` (measured for this comparison), the new
one −3.51 ±1.00 %.

**Information bench with the new network** (`info_bench_s_b`, same regimes and seeds
as §8.7; network contrasts, with the §8.7 values of `rl2_gen_s_d` for comparison):

| regime | net | B vs `nearest_job` | C vs `nearest_job` | B vs A | C vs B | D vs B | new vs ref in C |
|---|---|---|---|---|---|---|---|
| real / regular / nominal | ref | −8.02 ±1.75 | −13.33 ±1.81 | −0.22 ±1.33 | −5.67 ±1.71 | +0.76 ±0.92 | |
| | new | −7.38 ±1.69 | −13.04 ±1.92 | −0.06 ±1.49 | −6.03 ±1.80 | −0.07 ±0.72 | +0.34 ±0.83 |
| real / regular / slow | ref | −7.33 ±2.03 | −10.11 ±2.83 | +0.26 ±1.45 | −2.98 ±2.25 | −0.32 ±1.11 | |
| | new | −8.84 ±1.94 | −10.39 ±2.48 | −1.04 ±1.33 | −1.72 ±1.62 | +0.66 ±1.14 | −0.19 ±1.03 |
| real / burst / nominal | ref | −2.87 ±1.18 | −5.91 ±1.47 | +0.26 ±0.41 | −3.13 ±1.09 | −0.56 ±0.53 | |
| | new | −2.85 ±1.13 | −5.65 ±1.50 | +0.02 ±0.47 | −2.87 ±1.14 | −0.07 ±0.33 | +0.28 ±0.39 |
| real / burst / slow | ref | −4.54 ±1.74 | −8.41 ±1.72 | +0.05 ±0.83 | −3.96 ±1.56 | +0.13 ±0.64 | |
| | new | −4.58 ±1.61 | −8.73 ±1.76 | +0.36 ±0.66 | −4.32 ±1.28 | +0.01 ±0.49 | −0.34 ±0.87 |
| heavy / regular / nominal | ref | −4.01 ±1.81 | −3.26 ±1.62 | −0.22 ±1.35 | +0.91 ±1.64 | −1.16 ±1.24 | |
| | new | −4.33 ±1.46 | −3.10 ±1.58 | −0.29 ±1.03 | +1.36 ±1.55 | −1.34 ±0.84 | +0.19 ±0.91 |
| heavy / regular / slow | ref | +0.13 ±2.98 | +0.33 ±2.87 | −0.88 ±1.82 | +0.46 ±2.61 | −4.35 ±1.86 | |
| | new | +0.91 ±2.71 | −1.07 ±2.53 | +1.64 ±2.40 | −1.63 ±2.80 | **−6.35 ±2.42** | −1.20 ±1.85 |
| heavy / burst / nominal | ref | −4.93 ±1.32 | −5.54 ±1.53 | −0.34 ±0.78 | −0.57 ±1.56 | −0.36 ±0.92 | |
| | new | −4.56 ±1.38 | −5.26 ±1.51 | +0.11 ±0.97 | −0.67 ±1.45 | −0.83 ±0.82 | +0.31 ±0.38 |
| heavy / burst / slow | ref | −4.58 ±1.66 | −4.94 ±1.72 | −0.43 ±0.36 | −0.33 ±1.22 | −0.03 ±1.11 | |
| | new | −5.54 ±1.78 | −5.86 ±1.89 | −1.13 ±0.73 | −0.28 ±1.51 | −0.78 ±0.97 | −0.98 ±0.81 |

Findings:

* **Training on bursty cells and slow heads changes nothing measurable.** The new
  network equals the reference in every validation group and in every bench regime
  (the largest difference, −1.0 ±0.8 % on the heavy cell with bursts and a slow head,
  is at the edge of its interval). As with S/M/L (§8.4), the network is at its teacher
  from the first iteration (−3.51 vs −4.07 on `real_burst`), and the teacher, whose
  futures are Poisson at the mean rates, is the ceiling: a network trained on bursts
  cannot exceed a teacher that does not model them.
* **The 0.7.10 conclusions hold with a network trained in the regimes**: B vs A is
  within ±1.6 points in every regime, the value is in the action (C vs B −6.0 nominal,
  −1.7 slow, −2.9 burst, −4.3 both at the real load, ≈ 0 at the heavy load), and the
  hidden upstream queue is worth something only at the heavy load with a slow head,
  now −6.35 ±2.42 % (D vs B) with a model that saw 2.5 cm/s heads in training. That
  regime remains the one where neither network beats `nearest_job` in B
  (+0.9 ±2.7 %): with a head that slow and every belt queueing, the nearest-part rule
  is close to the best one can do without seeing the queues.
* **Choice of network.** Both are equivalent on development seeds. `rl2_gen_s_b` was
  trained on the declared scope (bursts, slow heads) and loses nothing elsewhere, so
  it is the natural candidate to freeze for phase 4; `rl2_gen_s_d` stays available as
  the 0.7.9 reference. **Decision (owner, 2026-11-22): `rl2_gen_s_b`, iteration 1, is
  the frozen network of phase 4**; the defensible statement is that extending the
  training produced no detectable improvement on this bench
  (`docs/plans/papers.md`, decisions before phase 4).
* **If more is wanted on bursts**, the lever is the teacher, not the training mix: give
  `rl2.search.sample_scenarios` the burst model of the cell (statistical knowledge,
  not an observation) so its futures are bursty when the cell is, and tell the
  network the burst parameters in the context token (a new token layout, trained from
  scratch). Not done. Owner's decision (2026-11-22): the teacher explanation is
  plausible but unproven; before any training from scratch, compare the Poisson
  teacher with a burst-aware teacher on the same realizations and the same search
  budget (development seeds).

### 8.9 Teacher futures: Poisson against the cell's arrival model (0.7.12)

The reading of §8.8 (a network trained on bursts cannot exceed a teacher whose
sampled futures are Poisson) was tested directly, as the owner asked, before any
training from scratch: the same network (`rl2_gen_s_b`, frozen), the same
realizations (validation seeds, design mode), the same search budget (8 futures, 30 s
window, w_pen 0, K = 6), and two ways of sampling the unknown future (§6.4):
`poisson` (the teacher of every run so far) and `cell` (an on/off source with the
cell's `on_fraction` and `on_mean`, started in its stationary state). On Poisson cells
the two are the same teacher by construction, which the first three columns confirm
end to end. `experiments/rl2_teacher_variants.py`, 50 seeds per group (100 general);
`results/experiments/teacher_futures_s_b{,_pairs,_strong}/variants.json`.

Paired Δ of the penalized flow time vs `nearest_job` [%]:

| policy | real | heavy | general | real_burst (0.3 / 20 s) | heavy, strong bursts (0.2 / 40 s) | real, strong bursts (0.2 / 40 s) |
|---|---|---|---|---|---|---|
| network alone | −6.19 ±1.48 | −2.86 ±1.09 | −2.73 ±1.05 | −3.51 ±1.00 | −1.06 ±0.61 | −1.85 ±0.75 |
| teacher, Poisson futures | −6.44 ±1.30 | −2.35 ±1.46 | −3.51 ±1.27 | −4.07 ±0.96 | −1.13 ±0.58 | −1.60 ±0.71 |
| teacher, cell futures | −6.44 ±1.30 | −2.35 ±1.46 | −3.51 ±1.27 | −3.60 ±0.89 | −1.27 ±0.61 | −1.56 ±0.69 |
| s / episode (net / teachers) | 0.4 / 23 | 0.7 / 116 | 0.4 / 45 | 0.5 / 50 | 0.8 / 190 | 0.5 / 103 |

Paired contrasts on the bursty cells (negative = the second is better; "better on"
counts the seeds where it is):

| cell | cell futures vs Poisson futures | Poisson teacher vs network | cell teacher vs network |
|---|---|---|---|
| real_burst (0.3 / 20 s) | **+0.52 ±0.49 %** (21/50) | −0.56 ±0.51 % (29/50) | −0.06 ±0.56 % (21/50) |
| heavy, strong bursts | −0.14 ±0.38 % (19/50) | −0.06 ±0.40 % (20/50) | −0.20 ±0.38 % (21/50) |
| real, strong bursts | +0.07 ±0.57 % (12/50) | +0.28 ±0.61 % (12/50) | +0.31 ±0.38 % (12/50) |

Findings:

* **The explanation is not supported.** Giving the teacher the cell's burst model does
  not make it a better teacher: on the validation bursty cell it is half a point
  *worse* than the Poisson teacher (paired +0.52 ±0.49 %, the interval barely
  excludes zero), and with strong bursts the two are equal within ±0.4–0.6 points.
  The ceiling the network reaches on bursty cells is therefore not an artefact of
  Poisson futures; retraining with a burst-aware teacher would not move it.
* **Why the informed future does not help at this budget.** The stationary start
  (on with probability `on_fraction`) adds to every sampled future the uncertainty of
  whether the source is emitting now, which 8 futures do not average out; the smoother
  Poisson future at the mean rate ranks the candidates with less noise. A future that
  knew the *current* on/off state would need to infer it from the recent arrivals
  (an observation-based estimate, a different model than the statistical knowledge
  tested here), and the pilot of §8.7 found no value of information on these cells.
* **With strong bursts there is little to decide.** At `on_fraction` 0.2 the cell
  alternates between an idle head and a saturated burst; the network gains
  1–2 % over `nearest_job`, no teacher beats the network (paired contrasts within
  ±0.6 points), and the known-parts order is close to the best achievable without
  seeing the hidden queue. This matches the pilot (§8.7): bursts reduce, not raise,
  the value of deciding better with the same information.
* **Consequence for Paper 1.** The sampled-future teacher with Poisson futures stays
  the teacher of record; the frozen network `rl2_gen_s_b` stands; no training from
  scratch on this ground. The contrast itself is a result for the paper: the quality
  of the amortized policy on bursty cells is bounded by the problem, not by the
  teacher's arrival model.

## 9. Reproduce (phase 3)

```bash
python -m pytest tests/test_rl2_search.py -q
python experiments/fastsim_validation.py --loads general --seeds 40 --seed-base 5500 \
    --dt 0.01 --workers 8 --modes none when_idle --tag fastsim_val_general_dt01
python experiments/rl2_train.py --tag rl2_gen_s --iters 6 --episodes0 1000 --episodes 500 \
    --epochs0 8 --epochs 4 --replay 2 --val 30 --workers 8
# 0.7.6: teacher variants over a trained network, then training with the chosen one
python experiments/rl2_teacher_variants.py --net models/rl2_gen_s_big.pt --val 30 \
    --variants known:k6 roll:m4:w30:p0 roll:m8:w30:p0 cut:h20:m4 dcut:h20:m4
python experiments/rl2_train.py --tag rl2_gen_s_sf --init models/rl2_gen_s_big.pt \
    --la-horizon inf --scenarios 8 --la-window 30 --w-pen 0 --iters 8 --episodes 2000 \
    --epochs 4 --replay 2 --val 50
# 0.7.10: information treatments and visible-length curve across regimes (§8.7)
python experiments/rl2_information_bench.py --net models/rl2_gen_s_d.pt --tag info_pilot_s_d --seeds 30
# 0.7.12: teacher futures, Poisson vs the cell's arrival model, same seeds and budget (§8.9)
python experiments/rl2_teacher_variants.py --net models/rl2_gen_s_b.pt --val 50 \
    --groups real heavy general real_burst --variants roll:m8:w30:p0 roll:m8:w30:p0:fcell \
    --tag teacher_futures_s_b
python experiments/rl2_teacher_variants.py --net models/rl2_gen_s_b.pt --val 50 \
    --groups heavy real_burst --variants roll:m8:w30:p0 roll:m8:w30:p0:fcell \
    --extra arrivals.stochastic.distribution=burst arrivals.stochastic.burst.on_fraction=0.2 \
            arrivals.stochastic.burst.on_mean=40 --tag teacher_futures_s_b_strong
```
