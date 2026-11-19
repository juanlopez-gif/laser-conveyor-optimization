# Reinforcement learning dispatcher (version 0.5.0)

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
3. Hence the default model of the policy `rl` is `models/rl_bc.zip`.
4. Taken together with phase 2: the structure of the problem (belt order,
   revocable pre-positioning) gives most of the attainable improvement cheaply;
   learning adds little at this scale **so far**.

## 5. Open items (not done)

* DAgger (query the teacher on the states visited by the learner) to fix the
  distribution shift of behaviour cloning; more teacher data (teacher at `dt = 0.01`).
* PPO with a larger budget on a many-core machine, KL penalty towards the cloned
  policy, critic pre-trained longer; sweep of γ and of the reward scale.
* Train with pre-positioning `when_idle` (the best mode for rollout and
  `nearest_head`); the current agent was trained and evaluated with `none` only.
* Calibrate the cell parameters before investing in large-scale training.
