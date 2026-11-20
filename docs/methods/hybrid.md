# Hybrid policies: learned dispatcher + look-ahead (version 0.6.0)

Code: `src/policies/hybrid.py`, tests `tests/test_hybrid.py`. Agent:
[rl.md](rl.md). Rollout: [online_rollout.md](online_rollout.md). Changes and rationale:
[../CHANGELOG.md](../CHANGELOG.md) (0.6.0).

## 1. Motivation

Up to 0.5 the RL agent alone had about the same cost as the zero-cost rule
`nearest_head`. The rollout was clearly better but about 100 times slower. The four
hybrids test whether the agent becomes useful once it is combined with search.

## 2. The four policies

| policy | idea | parameters |
|---|---|---|
| `rl_rollout` | Rollout whose **base policy is the agent**. Each candidate is scored by simulating the known parts with the agent instead of `nearest_job`. A rollout improves on its base policy (Bertsekas), so a better base should give a better rollout. | `model` plus every rollout parameter |
| `rl_filter` | The agent proposes its `top_k` best conveyors (default 2). The rollout then simulates only their heads. About 6/`top_k` times cheaper than the plain rollout. | `model`, `top_k`, `base_policy` (`nearest_job` or `rl`) |
| `rl_gated` | The agent decides directly when it is confident (largest action probability ≥ `confidence`, default 0.8) or when there is no time to think (slack < `min_slack_s`, default 0 = never). Otherwise the rollout decides. | `model`, `confidence`, `min_slack_s` |
| `rl_ensemble` | **Majority vote** of several agents. Ties are broken by the sum of the action probabilities. | `models` (comma-separated paths) |

Common rules:

* All four use the agent's action space: serve the head of a conveyor.
* Forced decisions (only one conveyor has a candidate) are taken directly, as in the RL
  environment.
* All four are `parametric` (they need a trained model), so they run only when named
  explicitly.
* Policy specs in the study harness: `name@k=v@k2=v2`, for example
  `rl_rollout@model=models/rl06_wi_s0.zip`.
* The harness records `decision_ms` (mean time per decision). `rl_gated` also records
  `rollout_share` (fraction of decisions sent to the rollout).

## 3. Evaluation (30 seeds 1000–1029, `dt = 0.01`, pre-positioning `when_idle`)

* Agent: `models/rl06_wi_s0.zip`. It was selected on validation seeds 2000+ (see
  [rl.md](rl.md) §6).
* Metric: penalized flow time.
* Paired Δ is measured against `nearest_job` with pre-positioning `none`, on the same
  seeds (common random numbers), with a normal 95 % CI.
* Source: `results/experiments/v0.6_summary/summary.md` (merged from `rl06_hybrids`,
  `rl06_agents`, `rollout_objective` and `v0.4_blocking_prepositioning`).

| policy | default: penalized flow [s] | default: paired Δ [%] | heavy: penalized flow [s] | heavy: paired Δ [%] | mean time per decision (default / heavy) |
|---|---|---|---|---|---|
| **`rl_rollout`** | **20.32** | **−11.4 (−13.0, −9.8)**, 30/30 | 71.49 | **−3.7 (−5.4, −2.1)**, 23/30 | 101 ms / 550 ms |
| `rollout@avg_flow_time` | 20.64 | −10.1 (−11.6, −8.5), 30/30 | 72.68 | −2.0 (−3.5, −0.5), 17/30 | — |
| `rollout` (makespan objective) | 20.96 | −9.6 (−12.6, −6.7), 26/30 | 75.53 | +1.8 (+0.4, +3.2), 9/30 | — |
| `rl_filter` (top 2) | 21.05 | −8.9 (−11.4, −6.5), 28/30 | 73.60 | −0.8 (−2.2, +0.7), 15/30 | 26 ms / 172 ms |
| `rl_gated` (0.8) | 21.34 | −7.6 (−9.6, −5.5), 28/30 | 72.15 | −2.9 (−4.7, −1.0), 23/30 | 7 ms / 4 ms |
| `rl_ensemble` (3 seeds) | 21.34 | −7.0 (−8.6, −5.3), 28/30 | **71.32** | **−4.0 (−5.4, −2.5)**, 26/30 | ms |
| `rl` alone (`rl06_wi_s0`) | 21.64 | −6.2 (−8.4, −4.1), 26/30 | 72.03 | −3.0 (−4.8, −1.3), 24/30 | ms |
| `nearest_head` | 21.63 | −6.4 (−9.0, −3.9), 29/30 | 74.11 | −0.1 (−1.5, +1.3), 15/30 | µs |
| `nearest_job` / `none` (reference) | 22.80 | 0 | 74.18 | 0 | µs |

* "30/30" is the number of seeds on which the policy beats the reference.
* Wall time per run is the whole simulation:

  | policy | default load | heavy load |
  |---|---|---|
  | `rl_rollout` | 10 s | 55 s |
  | `rollout` | 6 s | 47 s |
  | `rl_filter` | 3.5 s | 19 s |
  | `rl_gated` | 1.9 s | 2.4 s |

* `rl_gated` called the rollout on 14 % of the decisions at default load and on 0.3 %
  at heavy load. At heavy load the agent is almost always confident, so `rl_gated`
  behaves like the agent alone.

## 4. Findings

1. **`rl_rollout` is the best policy at default load.**
   * Δ = −11.4 %, and it beats the reference on all 30 seeds.
   * It also improves on the best plain rollout (`rollout@avg_flow_time`, −10.1 %),
     which matches the rollout-improvement property: the agent is a better base policy
     than `nearest_job`.
   * At heavy load it matches the best rollout from 0.5 (−3.7 %).
2. **The ensemble is the best policy at heavy load** (−4.0 %), at negligible cost. The
   differences among the top four at heavy load are within their confidence intervals.
3. **`rl_gated` gives most of the gain of the agent at almost no extra cost.** It does
   not reach the rollout because the confidence threshold sends few decisions to it.
4. **`rl_filter` loses at heavy load.** Pruning to two conveyors with the agent's
   ranking removes the option the rollout would have chosen too often.
5. The policy effect and the pre-positioning effect are mixed in the column "paired Δ":
   the reference uses `none`, the hybrids use `when_idle`. At equal pre-positioning, the
   reference is `nearest_job`/`when_idle` (−4.9 % default, −0.5 % heavy). Relative to
   it, `rl_rollout` gains about 6.5 points at default load and 3 points at heavy load.
   The confirmatory tables will report both effects separately (audit point 2).

## 5. Caveats

* The evaluation seeds 1000–1029 have been seen before. The rollout objective and the
  0.5 BC model were chosen on them. The final comparison will rerun the best variants on
  **fresh seeds 3000+** (audit point 1).
* One agent, trained at the two loads of the study with `when_idle`. Its behaviour
  outside those loads is in [sensitivity.md](sensitivity.md): it degrades when the cell
  is much busier than in training.
* `decision_ms` is measured on a 2-core laptop with the Python simulator. A real
  controller would need the rollout to run in well under one processing time (2.5 s).
  `rl_rollout` stays below this limit on average at both loads.

## 6. Reproduce

```bash
python experiments/blocking_prepositioning_study.py --modes when_idle --tag rl06_hybrids \
    --policies "rl_rollout@model=models/rl06_wi_s0.zip" "rl_filter@model=models/rl06_wi_s0.zip" \
    "rl_gated@model=models/rl06_wi_s0.zip"
python experiments/blocking_prepositioning_study.py --summarize v0.4_blocking_prepositioning \
    rollout_objective rl_eval rl06_agents rl06_agents_none rl06_hybrids --tag v0.6_summary
```

On Windows set `PYTHONIOENCODING=utf-8` so the summary can print "Δ".
