# Confirmatory evaluation on fresh seeds (version 0.6.1)

Code: `experiments/blocking_prepositioning_study.py` (runs) and
`experiments/decompose_effects.py` (effect decomposition). Outputs (not versioned):
`results/experiments/v0.6_confirmatory/` (`runs.csv`, `summary.md`,
`decomposition.md`). Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.6.1).

## 1. Why

The external audit of 2026-11-19 raised two methodological points, and both were
accepted:

1. **Seed reuse.** Several choices were made after looking at the evaluation seeds
   1000–1029: the rollout objective, the 0.5 BC model, and which hybrids to keep. Results
   on those seeds are therefore exploratory. The final variants must be compared on
   seeds never used before.
2. **Mixed effects.** The headline numbers up to 0.6.0 compared a policy *with*
   pre-positioning `when_idle` against `nearest_job` *without* it. That mixes two
   effects: a better dispatching rule, and moving the laser early. They must be
   reported separately.

## 2. Design (fixed before running)

* **Seeds:** 3000–3029. They were not used for training (100000+), validation (2000+)
  or any earlier evaluation (1000–1029).
* **Simulation:** common random numbers, `dt = 0.01`, default cell (the real values,
  see [../assumptions.md](../assumptions.md) §F) and heavy load.
* **Policies (6),** each run with pre-positioning `none` and `when_idle`:
  * `nearest_job` (reference);
  * `nearest_head`;
  * `rollout@avg_flow_time` (the best rollout);
  * `rl` (`rl06_wi_s0`, the default model);
  * `rl_ensemble` (3 agents);
  * `rl_rollout` (`rl06_wi_s0`).
* **Size and time:** 720 runs, 2000 s on 2 workers.
* **Metric:** penalized flow time, as a paired relative difference with a normal 95 %
  CI. Three effects are reported per policy:

| effect | comparison |
|---|---|
| total | policy / `when_idle` vs `nearest_job` / `none` (the column used up to 0.6.0) |
| policy | policy vs `nearest_job` at the **same** pre-positioning |
| pre-positioning | the same policy, `when_idle` vs `none` |

## 3. Results

Values are Δ % ± 95 % half-width, followed by (wins / 30). **Bold** = the CI excludes 0.
Negative = better.

### Default load (reference `nearest_job`/`none`: 17.66 s)

| policy | total | policy effect, `none` | policy effect, `when_idle` | pre-positioning effect |
|---|---|---|---|---|
| `rl_rollout` | **−13.1 ±1.6** (30/30) | **−4.4 ±1.8** (24/30) | **−4.7 ±1.2** (28/30) | **−9.0 ±2.0** (29/30) |
| `rollout@avg_flow_time` | **−12.1 ±1.7** (30/30) | **−4.0 ±1.6** (25/30) | **−3.5 ±1.3** (24/30) | **−8.3 ±1.6** (30/30) |
| `nearest_head` | **−10.1 ±2.6** (29/30) | −1.2 ±1.7 (17/30) | −1.4 ±1.7 (21/30) | **−8.9 ±2.6** (27/30) |
| `rl_ensemble` | **−9.3 ±2.1** (29/30) | −1.8 ±1.9 (17/30) | −0.4 ±1.8 (14/30) | **−7.5 ±1.9** (29/30) |
| `rl` | **−9.1 ±2.1** (28/30) | −1.7 ±1.9 (19/30) | −0.2 ±1.9 (16/30) | **−7.3 ±2.3** (28/30) |
| `nearest_job` | **−8.8 ±1.8** (29/30) | – | – | **−8.8 ±1.8** (29/30) |

### Heavy load (reference `nearest_job`/`none`: 62.65 s)

| policy | total | policy effect, `none` | policy effect, `when_idle` | pre-positioning effect |
|---|---|---|---|---|
| `rl_rollout` | **−4.0 ±1.6** (27/30) | **−3.7 ±1.4** (26/30) | **−4.0 ±1.7** (24/30) | −0.3 ±1.0 (20/30) |
| `rl_ensemble` | **−2.5 ±1.7** (19/30) | **−2.3 ±1.5** (21/30) | **−2.5 ±1.9** (20/30) | −0.2 ±1.3 (21/30) |
| `rl` | **−2.4 ±1.7** (21/30) | **−1.7 ±1.6** (18/30) | **−2.4 ±1.8** (21/30) | −0.8 ±1.0 (23/30) |
| `rollout@avg_flow_time` | −1.0 ±1.5 (19/30) | **−2.1 ±1.2** (20/30) | −1.0 ±1.9 (18/30) | +1.1 ±1.6 (13/30) |
| `nearest_job` | +0.1 ±1.1 (19/30) | – | – | +0.1 ±1.1 (19/30) |
| `nearest_head` | +0.3 ±1.2 (15/30) | +0.8 ±1.3 (11/30) | +0.3 ±1.5 (14/30) | −0.4 ±1.3 (22/30) |

### Head-to-head (same pre-positioning)

| comparison | default, `none` | default, `when_idle` | heavy, `none` | heavy, `when_idle` |
|---|---|---|---|---|
| `rl_rollout` vs `rollout@avg_flow_time` | −0.4 ±0.8 | −1.1 ±1.3 | −1.6 ±1.6 | **−2.9 ±1.4** |
| `rl_rollout` vs `rl_ensemble` | **−2.5 ±1.7** | **−4.0 ±1.9** | **−1.4 ±1.3** | **−1.4 ±1.4** |
| `rl` vs `nearest_head` | −0.4 ±2.0 | +1.3 ±2.0 | **−2.4 ±1.5** | **−2.7 ±1.8** |
| `rl_ensemble` vs `rl` | −0.0 ±1.6 | −0.2 ±0.7 | −0.6 ±0.9 | −0.0 ±1.1 |

### Decision time (this run, 2-core laptop)

| policy | default load: mean / max ms per decision | heavy load: mean / max ms per decision |
|---|---|---|
| `rl_rollout` | 38 / 95 | 280 / 449 |
| `rollout@avg_flow_time` | 28 / 64 | 216 / 305 |

Even the worst `rl_rollout` decision is well below one processing time (2.5 s). These
times are lower than in the 0.6.0 run (550 ms mean at heavy load), which ran
alongside other jobs, so the timings depend on machine load.

## 4. Conclusions

What is confirmed:

1. **`rl_rollout` is the best policy at both loads.** Total effect: −13.1 % (default)
   and −4.0 % (heavy). Its own policy effect is significant at both loads and in both
   modes (−3.7 to −4.7 %).
2. **It beats the plain rollout only at heavy load** (−2.9 %, with `when_idle`). At
   default load the difference is not significant (−1.1 ±1.3). The exploratory 0.6.0
   claim that it improves on the rollout at default load is therefore **not
   confirmed**.
3. **At heavy load the agent is a better rule than `nearest_head`** (−2.4 to −2.7 %).
   At default load the two are equivalent.

What changes the reading of earlier results:

4. **At default load, pre-positioning is the largest effect.** Moving the laser early
   (`when_idle`) gives −7 to −9 % with *any* policy, `nearest_job` included. Most of
   the "−6.2 % of the RL agent" in 0.6.0 was this effect: the agent's own policy
   effect at default load is −0.2 ±1.9, not significant. **In practice, the most
   valuable change for the real cell at its real load is to pre-position the laser.**
   This needs the cell to detect parts before they enter the laser zone, which is still
   unconfirmed (assumption C4).
5. **At heavy load, pre-positioning does nothing** (−0.8 to +1.1 %, not significant):
   the laser is never idle, so it never moves early. There, all the gain comes from
   the policy.
6. **The ensemble adds nothing over a single agent** (−0.0 to −0.6 %, not significant).
   The 0.6.0 finding "ensemble best at heavy load" was seed noise.
7. **The fresh seeds give a lighter default load** (reference 17.7 s vs 22.8 s on seeds
   1000–1029). This is sampling variation of 300 s of Poisson arrivals and shows why
   the paired design is needed.

## 5. Reproduce

```bash
PYTHONIOENCODING=utf-8 python experiments/blocking_prepositioning_study.py --seeds 30 \
    --seed-base 3000 --workers 2 --modes none when_idle --tag v0.6_confirmatory \
    --policies nearest_job nearest_head rollout@avg_flow_time "rl@model=models/rl06_wi_s0.zip" \
    "rl_ensemble@models=models/rl06_wi_s0.zip,models/rl06_wi_s1.zip,models/rl06_wi_s2.zip" \
    "rl_rollout@model=models/rl06_wi_s0.zip"
PYTHONIOENCODING=utf-8 python experiments/decompose_effects.py --tag v0.6_confirmatory \
    --pairs "rl_rollout@model=models/rl06_wi_s0.zip" rollout@avg_flow_time \
    --pairs "rl_rollout@model=models/rl06_wi_s0.zip" "rl_ensemble@models=models/rl06_wi_s0.zip,models/rl06_wi_s1.zip,models/rl06_wi_s2.zip" \
    --pairs "rl@model=models/rl06_wi_s0.zip" nearest_head \
    --pairs "rl_ensemble@models=models/rl06_wi_s0.zip,models/rl06_wi_s1.zip,models/rl06_wi_s2.zip" "rl@model=models/rl06_wi_s0.zip"
```
