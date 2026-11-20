# Reinforcement learning dispatcher (version 0.5.0, updated 0.6.0)

> **0.7.0:** the engine's tie-break was corrected and the reference `nearest_job`
> became about 2.5 % stronger at the real load. Current numbers: [confirmatory.md](confirmatory.md) §7.

> **0.6.0:** §1–§5 describe the 0.5 agent and are kept as the historical record. The
> 0.6 training (DAgger, γ = 1, validation-based selection) and its results are in §6.
> The default model of the policy `rl` is now `models/rl06_wi_s0.zip`. The hybrids
> that combine the agent with the rollout are in [hybrid.md](hybrid.md).

Code: `src/rl/` (environment, features, training) and the policy `rl`
(`src/policies/rl_policy.py`). Formulation: [../formal_formulation.md](../formal_formulation.md)
§4. Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.5.0).

## 1. Formulation: a semi-Markov decision process

| element | definition |
|---|---|
| decision epochs | the laser is free and **at least two** conveyors have a dispatch candidate (if only one has, the decision is forced and taken automatically) |
| action | $k\in\lbrace 1..N\rbrace$: serve the **head** of conveyor $k$ (its most downstream candidate); actions of conveyors without candidates are **masked** |
| observation (87 features for N = 3) | laser position; per conveyor: running, blocked, upstream buffer, has candidate; for its first 3 unstarted parts: present, position, in zone, time to the zone end, interception time, waiting time, processing time, blocking now; total unfinished parts (`rl/features.py`) |
| cost between epochs $t,t'$ | $\int_t^{t'}\lvert\lbrace j: r_j\le s<C_j\rbrace\rvert ds+\lambda_T T_B(t,t')+\lambda_N N_B(t,t')$ |
| return | undiscounted sum $=-(\sum_jF_j+\Pi)=-n\cdot$`penalized_flow_time` (Little's law); reward scaled by 1/100 |
| **caveat (0.5.2)** | the identity holds for the **undiscounted** return. PPO was trained with $\gamma=0.99$ per decision (and the critic of the cloning step on discounted returns), so it optimized a discounted proxy that weights early costs more; with $n$ exogenous (every arriving part is processed, the system is drained) the per-episode scaling $1/n$ does not change the optimal policy, but across episodes the return weights long episodes more than the mean-per-part objective does. Next iteration: $\gamma=1$ (finite drained episodes) or average-reward. |
| episode | 300 s of stochastic arrivals, then drained; load sampled per episode (default / heavy), training seeds ≥ 100000 (never the evaluation seeds 1000–1029) |

Design choices and why:
* **Head-only actions** use the structural result of phase 2 (good sequences keep belt
  order within each conveyor). The action space is small and fixed (N), so a plain MLP
  works; serving a later part of a conveyor first is not representable — by design.
* **Return = objective.** The reward integrates work in progress instead of rewarding
  completions, so maximizing the return is *exactly* minimizing the default objective
  (checked in `tests/test_rl.py`: return × 100 = Σ F_j + Π up to one step `dt` per part).
* **Forced decisions are hidden** (≈ 2/3 of the dispatches at default load), which
  shortens the credit-assignment chain.
* The engine was split around the decision (`Simulation.begin_step()`,
  `decision_candidates()`, `end_step(choice)`); `step()` is exactly
  `begin_step(); end_step()` (bit-identical, tested).

## 2. Training

```bash
pip install -r requirements-rl.txt
PYTHONPATH=src python -m rl.train --teacher rollout --episodes 60 --ppo-steps 150000 --n-envs 2
```

1. **Teacher data:** 60 episodes (30 default, 30 heavy) driven by `rollout`
   (flow look-ahead, `dt = 0.05`): 3 838 decisions, 919 s. The teacher may choose any
   candidate; its label is the conveyor of the chosen part.
2. **Behaviour cloning** (60 epochs): actor by masked cross-entropy, critic by
   regression on the teacher's discounted returns. Training accuracy **0.903**.
   → `models/rl_bc.zip`.
3. **PPO fine-tuning** (MaskablePPO, 150 000 decisions, 2 parallel simulators,
   lr 1e-4, clip 0.1, γ = 0.99 per decision): 894 s. → `models/rl_ppo.zip`. The learning
   curve (stochastic policy) did not improve over the cloned policy
   (`models/rl_training.json`).

Hardware note: the bottleneck is the Python simulator, not the network (87→128→128→3).
More **CPU cores** help (`--n-envs` = number of cores); a GPU does not.

## 3. Evaluation (30 seeds, `dt = 0.01`, paired with the other policies)

Same protocol as [nearest_head_prepositioning.md](nearest_head_prepositioning.md):
pre-positioning `none` (the mode used in training), penalized flow time, paired Δ vs
`nearest_job`:

| policy | default load | heavy load | decision cost |
|---|---|---|---|
| rollout (flow look-ahead) — the teacher | **−4.2 % (−5.4, −3.0)**, 26/30 | **−3.7 % (−5.1, −2.4)**, 25/30 | ≈ 9 s / 53 s per run |
| `rl` behaviour cloning (`rl_bc.zip`) | −1.4 % (−3.3, +0.5), 18/30 | −1.8 % (−3.3, −0.3), 23/30 | ≈ 1–2 s per run (incl. simulation) |
| `rl` BC + PPO (`rl_ppo.zip`) | +1.9 % (−0.4, +4.3), 13/30 | −1.0 % (−2.6, +0.5), 20/30 | same |
| `nearest_head` | −2.6 % (−4.4, −0.7), 22/30 | +0.5 % (−0.9, +1.9), 16/30 | ms |
| `nearest_job` | 0 | 0 | ms |

Reproduce:

```bash
python experiments/blocking_prepositioning_study.py --policies "rl@model=models/rl_bc.zip" \
    "rl@model=models/rl_ppo.zip" --modes none --tag rl_eval
python experiments/blocking_prepositioning_study.py \
    --summarize v0.4_blocking_prepositioning rollout_objective rl_eval --tag v0.5_summary
```

## 4. Findings

1. **Imitation recovers about half of the teacher's gain at heavy load**
   (−1.8 % vs −3.7 %, significant) at a negligible decision cost; at default load the
   gain (−1.4 %) is not significant and below the zero-cost rule `nearest_head`
   (−2.6 %).
2. **PPO fine-tuning degraded the cloned policy** with this budget (150k decisions,
   ~15 min on 2 cores). This is a common failure mode (the critic and the entropy
   bonus pull the actor away from the teacher before the value estimates are good);
   it is reported as is.
3. Hence the default model of the policy `rl` was `models/rl_bc.zip` (until 0.5.2;
   superseded in 0.6.0, see §6).
4. Taken together with phase 2: the structure of the problem (belt order,
   revocable pre-positioning) gives most of the attainable improvement cheaply;
   learning adds little at this scale **so far**.

## 5. Open items after 0.5 (status in 0.6.0 marked in brackets)

* DAgger (query the teacher on the states visited by the learner) to fix the
  distribution shift of behaviour cloning **[done in 0.6.0]**; more teacher data
  (teacher at `dt = 0.01`) [open].
* PPO with a larger budget on a many-core machine, KL penalty towards the cloned
  policy, critic pre-trained longer; sweep of γ and of the reward scale.
* $\gamma=1$ (or average reward) so that the trained objective is exactly the
  evaluated one (see the caveat in §1); several training seeds **[both done in 0.6.0]**.
* Train with pre-positioning `when_idle` (the best mode for rollout and
  `nearest_head`); the 0.5 agent was trained and evaluated with `none` only
  **[done in 0.6.0]**.
* Calibrate the cell parameters before investing in large-scale training.

## 6. Version 0.6.0: DAgger, γ = 1, validation-based selection

Code: `src/rl/train.py`. Each run writes `models/<tag>.zip` and
`models/<tag>_training.json` (arguments, teacher data, validation history).

### 6.1 Training changes and why

| change | reason |
|---|---|
| **DAgger** after behaviour cloning (3 iterations × 24 episodes; mixing β_i = 0.5^i; 30 epochs per iteration). The learner drives and the teacher (`rollout`) labels every visited state. | Plain cloning only sees the teacher's state distribution (Ross et al. 2011). |
| **γ = 1**, reward scale 1/1000 | The return is now exactly minus the episode objective (removes the caveat in §1). |
| **Checkpoint selection on validation seeds 2000+** (10 episodes per check). Score = mean ratio of the episode cost to `nearest_head` on the same episode; lower is better. | The evaluation seeds 1000–1029 are never used for selection. The delivered model is the best checkpoint among BC, DAgger and PPO. |
| PPO fine-tuning reduced to 60 000 decisions, validated every 15 000 | In 0.5 PPO degraded the cloned policy, so it is now only kept if validation says it helps. |
| Pre-positioning `when_idle` (3 seeds) and `none` (1 seed) | `when_idle` is the best mode for rollout and `nearest_head`. |
| Parallel episode collection, 40 teacher episodes (default + heavy load) | Time: about 37 min per agent on 2 cores. |

### 6.2 Validation history (score, lower is better; selected stage in bold)

| agent | BC | DAgger 1 | DAgger 2 | DAgger 3 | PPO 15k | PPO 30k | PPO 45k | PPO 60k |
|---|---|---|---|---|---|---|---|---|
| `rl06_wi_s0` | 1.003 | 0.992 | 0.988 | **0.980** | 0.984 | 0.999 | 1.006 | 1.026 |
| `rl06_wi_s1` | 1.000 | 0.991 | **0.985** | 0.994 | 0.990 | 1.010 | 1.007 | 1.015 |
| `rl06_wi_s2` | 0.997 | 1.011 | 1.005 | 0.994 | **0.988** | 1.015 | 1.012 | 1.034 |
| `rl06_none_s0` | 1.027 | 1.003 | 1.001 | **0.985** | 0.988 | 1.021 | 1.013 | 0.998 |

* DAgger improved on cloning in all four runs. The gain over `nearest_head` on
  validation is small (1.2–2 %).
* Longer PPO always made things worse. Only `rl06_wi_s2` kept its first PPO checkpoint
  (0.988 vs 0.994 after DAgger), a difference within the noise of 10 validation
  episodes.

### 6.3 Evaluation (30 seeds 1000–1029, paired Δ of penalized flow vs `nearest_job`/`none`)

| agent | pre-positioning | default load | heavy load |
|---|---|---|---|
| `rl06_wi_s0` (new default) | `when_idle` | −6.2 % (−8.4, −4.1), 26/30 | −3.0 % (−4.8, −1.3), 24/30 |
| `rl06_wi_s1` | `when_idle` | −7.1 % (−8.8, −5.3), 28/30 | −2.8 % (−4.4, −1.2), 21/30 |
| `rl06_wi_s2` | `when_idle` | −5.8 % (−7.9, −3.6), 25/30 | −2.8 % (−4.6, −0.9), 23/30 |
| `rl_ensemble` (the 3 above) | `when_idle` | −7.0 % (−8.6, −5.3), 28/30 | −4.0 % (−5.4, −2.5), 26/30 |
| `rl06_none_s0` | `none` | −1.4 % (−3.8, +1.0), 19/30 | −3.4 % (−4.9, −1.8), 24/30 |
| `rl_bc` (0.5 default) | `none` | −1.4 % (−3.3, +0.5), 18/30 | −1.8 % (−3.3, −0.3), 23/30 |
| `nearest_head` | `when_idle` | −6.4 % (−9.0, −3.9), 29/30 | −0.1 % (−1.5, +1.3), 15/30 |
| `nearest_job` | `when_idle` | −4.9 % (−6.9, −2.8), 23/30 | −0.5 % (−1.9, +0.8), 21/30 |

Source: `results/experiments/v0.6_summary/summary.md` (tags `rl06_agents`,
`rl06_agents_none`).

### 6.4 Findings

> **Revised by the confirmatory run (0.6.1, [confirmatory.md](confirmatory.md)).**
> * On fresh seeds the agent's own policy effect at default load is −0.2 ±1.9 %, not
>   significant. The −6.2 % below was almost entirely the pre-positioning effect.
> * At heavy load it is −2.4 % and significant, and the agent beats `nearest_head` by
>   about 2.5 %.
> * The ensemble does not improve on a single agent.


1. **At equal pre-positioning the 0.6 agent beats the 0.5 agent at heavy load**:
   `rl06_none_s0` gets −3.4 % against −1.8 % for `rl_bc`, as good as the 0.5
   rollout (−3.7 %). At default load it does not improve (−1.4 % for both).
2. **With `when_idle`, the agent alone is about as good as `nearest_head`** at default
   load (−6.2 % vs −6.4 %). It is clearly better at heavy load (−3.0 % vs −0.1 %).
3. **The three training seeds give similar agents** (−5.8 to −7.1 % default). The
   majority vote is the best agent-only policy at heavy load (−4.0 %).
4. **The agent is most useful inside the rollout.** `rl_rollout` is the best policy at
   default load (−11.4 %); see [hybrid.md](hybrid.md).
5. **Out of its training cell the agent degrades.** With 40 % more arrivals it is worse
   than the reference; see [sensitivity.md](sensitivity.md). Domain randomization is
   planned in `docs/plans/strong_rl_plan.md`.
6. Default model: `rl06_wi_s0`. It has the best validation score. On the evaluation
   seeds `rl06_wi_s1` is slightly better, but choosing it on those seeds would reuse
   them for selection.

### 6.5 Reproduce

```bash
PYTHONPATH=src python -m rl.train --teacher rollout --prepositioning when_idle \
    --episodes 40 --dagger-iters 3 --dagger-episodes 24 --ppo-steps 60000 \
    --gamma 1 --reward-scale 1000 --seed 0 --tag rl06_wi_s0 --workers 2 --n-envs 2
python experiments/blocking_prepositioning_study.py --modes when_idle --tag rl06_agents \
    --policies "rl@model=models/rl06_wi_s0.zip" "rl@model=models/rl06_wi_s1.zip" \
    "rl@model=models/rl06_wi_s2.zip" \
    "rl_ensemble@models=models/rl06_wi_s0.zip,models/rl06_wi_s1.zip,models/rl06_wi_s2.zip"
```
