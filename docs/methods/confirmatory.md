# Confirmatory evaluation on fresh seeds (version 0.6.1)

Code: `experiments/blocking_prepositioning_study.py` (runs) and
`experiments/decompose_effects.py` (effect decomposition). Outputs (not versioned):
`results/experiments/v0.6_confirmatory/` (`runs.csv`, `summary.md`,
`decomposition.md`). Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.6.1).

> **Update 0.7.0 (2026-11-20): rerun with the corrected tie-break, §7.**
> * In the engine, exact ties between candidates were decided by floating-point
>   round-off instead of the documented rule (earliest arrival first); see
>   [fastsim.md](fastsim.md) §3.
> * With the rule applied, the reference `nearest_job` is about 2.5 % better at the
>   real load, so every gain against it shrinks.
> * **The current numbers are in §7.** §3–§5 are kept as the record of the 0.6.1 run.

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
   This needs the cell to detect parts before they enter the laser zone. **On
   2026-11-20 the owner confirmed that the real cell cannot do this for now**
   (assumption C4), so it is a possible hardware upgrade, not an option today.
5. **At heavy load, pre-positioning does nothing** (−0.8 to +1.1 %, not significant):
   the laser is never idle, so it never moves early. There, all the gain comes from
   the policy.
6. **The ensemble adds nothing over a single agent** (−0.0 to −0.6 %, not significant).
   The 0.6.0 finding "ensemble best at heavy load" was seed noise.
7. **The fresh seeds give a lighter default load** (reference 17.7 s vs 22.8 s on seeds
   1000–1029). This is sampling variation of 300 s of Poisson arrivals and shows why
   the paired design is needed.

## 5. Recommendation for the real cell (2026-11-20)

> **Superseded in part (0.7.7, 2026-11-22).** The cell is not built yet; its design
> mode is now `when_idle` (`docs/assumptions.md` C4), and the strong-RL dispatcher of
> `rl2.md` §8.5–§8.6 replaces `rl_rollout` as the recommendation. The numbers below
> stay valid for the `none` variant and as the reference the new models are
> compared with.

At the time, the real cell was taken to run without pre-positioning (`none`), because
it could not detect parts before the laser zone. In that mode:

| policy | default (real) load | heavy load | decision time |
|---|---|---|---|
| **`rl_rollout`** (recommended) | **−4.4 ±1.8 %** (24/30) | **−3.7 ±1.4 %** (26/30) | ≈ 40 ms / 280 ms mean |
| `rollout@avg_flow_time` | −4.0 ±1.6 % (25/30) | −2.1 ±1.2 % (20/30) | ≈ 30 ms / 215 ms mean |
| `rl` | −1.7 ±1.9 % | −1.7 ±1.6 % | ms |
| `nearest_head` | −1.2 ±1.7 % | +0.8 ±1.3 % | µs |

* `rl_rollout` is the recommendation: it is the best at both loads and is fast enough.
* At the real load the plain rollout is statistically equivalent and needs no trained
  model, so it is the fallback if the RL dependencies are unwanted on the controller.
* The agent used here (`rl06_wi_s0`) was trained with `when_idle`. The agent trained
  with `none` (`rl06_none_s0`) was tested in §5.1; it makes no significant difference.
* An entry sensor would add about 9 points on top of the policy, at the real load.

### 5.1 Does the agent trained without pre-positioning help? (2026-11-20)

* **Question:** the real cell runs with `none`, but the agent inside `rl_rollout` was
  trained with `when_idle`. Is `rl06_none_s0`, trained with `none`, a better base
  policy?
* **Setup:** `rl_rollout` and `rl` with `rl06_none_s0`, seeds 3000–3029, `none`, both
  loads (120 runs, tag `v0.6_confirmatory_none_agent`). They were compared with the
  runs above on the same seeds using
  `decompose_effects.py --merge` (output `v0.6_confirmatory_merged`).

| comparison (mode `none`) | default (real) load | heavy load |
|---|---|---|
| `rl_rollout`, `none_s0` vs reference | **−4.9 ±1.7** (25/30) | **−3.1 ±1.6** (24/30) |
| `rl_rollout`, `wi_s0` vs reference | **−4.4 ±1.8** (24/30) | **−3.7 ±1.4** (26/30) |
| `rl_rollout`: `none_s0` vs `wi_s0` | −0.5 ±0.9 (12/30) | +0.6 ±1.1 (14/30) |
| `rl_rollout` `none_s0` vs `rollout@avg_flow_time` | **−0.9 ±0.6** (20/30) | −1.1 ±1.6 (22/30) |
| `rl`: `none_s0` vs `wi_s0` | −0.0 ±1.6 | +0.6 ±1.1 |

* **Answer: no significant difference** between the two agents, alone or inside the
  rollout. The two point estimates go in opposite directions at the two loads.
* With `none_s0` as base, `rl_rollout` is slightly but significantly better than the
  plain rollout at the real load (−0.9 ±0.6 %). With `wi_s0` the same comparison was
  not significant (−0.4 ±0.8 %).
* **Decision:** the recommendation stays `rl_rollout` with `rl06_wi_s0`, the default
  model. Either agent is acceptable, and keeping one model for every mode is simpler.
* Decision time with `none_s0`: 42 ms mean (114 ms max) at the real load and 305 ms
  mean (719 ms max) at heavy load. Even the maximum is below one processing time.

## 6. Reproduce

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
# §5.1
PYTHONIOENCODING=utf-8 python experiments/blocking_prepositioning_study.py --seeds 30     --seed-base 3000 --workers 2 --modes none --tag v0.6_confirmatory_none_agent     --policies "rl_rollout@model=models/rl06_none_s0.zip" "rl@model=models/rl06_none_s0.zip"
PYTHONIOENCODING=utf-8 python experiments/decompose_effects.py --tag v0.6_confirmatory     --merge v0.6_confirmatory_none_agent --out v0.6_confirmatory_merged     --pairs "rl_rollout@model=models/rl06_none_s0.zip" "rl_rollout@model=models/rl06_wi_s0.zip"     --pairs "rl_rollout@model=models/rl06_none_s0.zip" rollout@avg_flow_time
```

## 7. Rerun with the corrected tie-break (0.7.0, 2026-11-20)

* **What changed:** the same design, seeds 3000–3029 and policies as §2, rerun on the
  engine with the tie fix of [fastsim.md](fastsim.md) §3. Tag `v0.7_confirmatory_ties`,
  720 runs on 8 workers.
* **What did not change:** the RL agents were not retrained. Their behaviour changes
  only through the slightly different states they now visit.

### Real cell: default load, pre-positioning `none`

Reference `nearest_job` / `none`: 17.23 s. It was 17.66 s with the old tie-breaking.

| policy | penalized flow [s] | Δ vs `nearest_job` (95 % CI), wins | in 0.6.1 (§5) |
|---|---|---|---|
| **`rl_rollout`** | 16.82 | **−2.3 ±1.5 %**, 21/30 | −4.4 % |
| `rollout@avg_flow_time` | 16.87 | **−1.9 ±1.3 %**, 21/30 | −4.0 % |
| `nearest_head` | 17.27 | +0.4 ±1.6 %, 11/30 | −1.2 % |
| `rl_ensemble` | 17.29 | +0.4 ±1.6 %, 12/30 | −1.8 % |
| `rl` | 17.32 | +0.5 ±1.8 %, 13/30 | −1.7 % |

### Heavy load, `none`

| policy | Δ vs `nearest_job` | in 0.6.1 |
|---|---|---|
| **`rl_rollout`** | **−3.4 ±1.4 %**, 25/30 | −3.7 % |
| `rl_ensemble` | **−2.0 ±1.6 %** | −2.3 % |
| `rollout@avg_flow_time` | **−1.7 ±1.4 %** | −2.1 % |
| `rl` | −1.4 ±1.6 % | −1.7 % |
| `nearest_head` | +1.0 ±1.3 % | +0.8 % |

### Other results of the rerun

* **Pre-positioning effect** at default load: −7.3 to −9.1 % for every policy, as
  before. At heavy load there is no effect.
* **`rl_rollout` vs `rollout`:**
  * default load: −0.4 ±1.1 % (`none`) and −0.2 ±1.2 % (`when_idle`), not significant;
  * heavy load with `when_idle`: −2.4 ±1.4 %, significant.
* **`rl` vs `nearest_head`:** equal at default load; −2.3 % (`none`) and −3.0 %
  (`when_idle`) at heavy load.

### Conclusions after the rerun

1. **The recommendation is unchanged:** `rl_rollout` with `none` is the best policy
   for the real cell, at both loads.
2. **At the real load the attainable gain over a well-specified nearest-job rule is
   small:**
   * about 2 % (`rl_rollout` −2.3 %, `rollout` −1.9 %);
   * `nearest_head` and the agents alone give nothing there.
   * Half of the 0.6.1 gain at the real load came from the reference being handicapped
     by round-off.
3. **At heavy load the learned policies clearly help:** `rl_rollout` −3.4 %; the
   agent alone beats `nearest_head` by 2–3 %.
4. This sets the bar for the strong-RL plan at the real load:
   * beat `nearest_job` / `none` by clearly more than 2 %;
   * beat `rl_rollout` / `none` significantly.

## 8. Rerun with the real information model (0.7.3, 2026-11-20)

**Why.** The real cell has an overhead camera that sees every belt (entry zone and
laser zone) and the positions of the parts, but not the parts waiting upstream of the
belt entry (`docs/assumptions.md` C4b). Up to 0.7.2:

* `rollout` copied every arrived part into its look-ahead, including the waiting ones;
* the RL agent saw the upstream buffer length.

Version 0.7.3 removes that information (`simulation.observation: belts`,
`docs/formal_formulation.md` §4.1). The same design as §7 was rerun: seeds 3000–3029,
`none`, tag `v0.7.3_confirmatory_belts`. The agents were not retrained; the agent now
receives a zero buffer length.

**Paired Δ of penalized flow time vs `nearest_job` / `none`** (95 % CI, wins out of 30):

| policy | real cell, 0.7.3 | real cell, §7 (0.7.0) | heavy load, 0.7.3 | heavy load, §7 |
|---|---|---|---|---|
| `rollout@avg_flow_time` | **−2.4 ±1.3 %**, 22/30 | −1.9 ±1.3 % | −0.9 ±1.3 %, 17/30 | −1.7 ±1.4 % |
| `rl_rollout` | **−2.3 ±1.4 %**, 22/30 | −2.3 ±1.5 % | **−2.1 ±1.5 %**, 20/30 | −3.4 ±1.4 % |
| `rl` | −0.1 ±1.3 % | +0.5 ±1.8 % | −0.7 ±1.4 % | −1.4 ±1.6 % |
| `nearest_head` | +0.4 ±1.6 % | +0.4 % | +1.0 ±1.3 % | +1.0 % |

* **Control.** `nearest_job` and `nearest_head` give results identical to §7 on all
  60 runs: they never used unobservable parts.
* **Real cell:** the gains do not depend on the waiting parts. `rollout` and
  `rl_rollout` are equivalent (−2.4 % and −2.3 %, both significant).
* **Heavy load:** about one point of the earlier gains came from knowing the parts
  queued upstream, which are many at that load. `rl_rollout` stays significant
  (−2.1 %); the plain `rollout` no longer is.
* **Recommendation for the real cell:** `rollout@avg_flow_time` or `rl_rollout` with
  `none`. The two are equivalent; the plain rollout needs no trained model.

## 9. Phase 4: confirmatory evaluation on the reserved seeds (0.7.13, 2026-11-22)

Code: `experiments/rl2_eval.py` (one invocation; protocol frozen in
`docs/plans/papers.md`, "Phase 4 protocol", confirmed by the owner). Outputs (not
versioned): `results/experiments/phase4/bench.json`, `tables.md`; the full tables are
copied to [phase4_tables.md](phase4_tables.md). Run of 2026-11-22, 23:03–23:41 on the
workstation (38 min, 47 workers): 15 360 information runs, 1 440 search runs, 960
engine runs.

**Design (as frozen).** Network `models/rl2_gen_s_b.pt` (frozen, iteration 1),
reference `models/rl2_gen_s_d.pt` (0.7.9); **seeds 7000–7029**, used here once; the
exact event-driven simulator for every run; eight regimes: load real / heavy ×
arrivals Poisson / burst (on-fraction 0.3, on-mean 20 s) × head nominal / slow (speeds
× 0.5); treatments A (`zone`/`none`), B (`belts`/`none`), C (`belts`/`when_idle`), D
(`full`/`none`) and the coverage grid `visible_length` ∈ {0, 1, 2.5, 5, 7.5, 10} cm with
`none` and `when_idle` at each length; policies `nearest_job`, `nearest_head`, the two
networks and, in B and C, the known-parts search over `nearest_job` (K = 6) and the
sampled-future teacher over the frozen network (8 futures, 30 s, w_pen 0). Paired Δ of
the penalized flow time with 95 % CI; "n/30" counts the seeds where the second policy
is better. Latencies are measured on the decisions of the network's own episode, one
thread per worker, with 47 workers busy (an upper bound of the single-process figures
of `rl2.md` §5).

### 9.1 Overview

| regime | `nearest_job` B [s/part] | net B vs NJ | net C vs NJ | B vs A (net) | C vs B (net) | D vs B (net) | teacher vs net (B) |
|---|---|---|---|---|---|---|---|
| real / Poisson / nominal head | 21.8 | −7.22 ±1.95 | −14.11 ±2.21 | +0.85 ±1.14 | **−7.37 ±1.87** | −0.32 ±0.78 | −0.39 ±1.24 |
| real / Poisson / slow head | 42.1 | −7.25 ±1.74 | −10.63 ±2.52 | −0.89 ±1.24 | **−3.56 ±2.48** | +0.03 ±0.99 | −0.53 ±1.12 |
| real / burst / nominal head | 27.6 | −1.96 ±1.74 | −4.74 ±1.70 | +0.71 ±1.16 | **−2.75 ±1.50** | −0.74 ±1.05 | −0.65 ±1.12 |
| real / burst / slow head | 41.3 | −6.24 ±2.00 | −10.35 ±2.24 | −0.03 ±1.06 | **−4.26 ±2.19** | +0.24 ±1.00 | −0.48 ±1.22 |
| heavy / Poisson / nominal head | 73.5 | −2.78 ±1.85 | −4.14 ±1.28 | −0.63 ±1.08 | −1.27 ±1.34 | **−2.66 ±1.43** | −0.81 ±1.46 |
| heavy / Poisson / slow head | 104.0 | −1.32 ±2.32 | −1.50 ±2.15 | −1.34 ±1.66 | −0.01 ±1.98 | **−3.74 ±1.87** | +0.05 ±2.24 |
| heavy / burst / nominal head | 72.0 | −3.29 ±1.66 | −4.90 ±1.57 | −0.42 ±0.63 | **−1.60 ±1.14** | −0.42 ±0.77 | −0.27 ±0.55 |
| heavy / burst / slow head | 93.6 | −4.15 ±1.74 | −3.56 ±1.94 | **−1.31 ±0.74** | +0.64 ±1.16 | +0.04 ±0.85 | +0.01 ±0.79 |

### 9.2 Paper 2: value of upstream visibility

* **B vs A (seeing the entry zone with the same actions) is a null result in every
  regime**: for the network, between −1.3 and +0.9 points, with intervals of ±0.6 to
  ±1.7; the only interval that excludes zero (heavy load, bursts, slow head, −1.31
  ±0.74) is the smallest effect of the table. For `nearest_job` and `nearest_head`
  the contrast is exactly zero by construction (they never use upstream parts). The
  pilot's bound (±0.9 on development seeds) is confirmed at about ±1.5 points on the
  reserved seeds.
* **C vs B (acting on the same view) is where the value is, at the real load**: −7.4
  (nominal head), −3.6 (slow head), −2.8 (bursts) and −4.3 (bursts and slow head)
  points for the network, 22–27 of 30 seeds, and the same for `nearest_job` (−7.4,
  −3.6, −2.7, −3.9). At the heavy load the head is rarely idle and the contrast is
  within ±1.3 points (significant only with bursts and the nominal head, −1.6 ±1.1).
  The secondary hypothesis holds: bursts and a slow head reduce the value of acting
  early (from −7.4 to −2.8 / −3.6), they do not create a value of seeing early.
* **D vs B (the hidden upstream queue) is worth something only at the heavy load**:
  −2.7 ±1.4 (nominal head) and −3.7 ±1.9 (slow head) with Poisson arrivals, 20–23 of
  30 seeds; within ±1 point in the six other regimes. The pilot's −6.35 ±2.42 on
  development seeds (0.7.11) is confirmed in sign, with a smaller size.
* **Coverage curve** (`when_idle` vs `none` at the same visible length; share of the
  whole-zone gain; seconds = length at the belt speed, 3 cm/s at the real load):

  | regime (real load) | policy | whole-zone gain | 1 cm (0.33 s) | 2.5 cm (0.83 s) | 5 cm (1.67 s) | 7.5 cm (2.5 s) |
  |---|---|---|---|---|---|---|
  | Poisson / nominal head (head span 2.4 s) | `nearest_job` | −7.37 ±2.49 | 24 % | 62 % | 84 % | 99 % |
  | | network | −7.37 ±1.87 | 9 % | 39 % | 64 % | 78 % |
  | Poisson / slow head (4.8 s) | `nearest_job` | −3.61 ±2.55 | −12 % | 3 % | 36 % | 85 % |
  | | network | −3.56 ±2.48 | 19 % | 44 % | 58 % | 78 % |
  | burst / nominal head (2.4 s) | `nearest_job` | −2.71 ±1.72 | 49 % | 41 % | 109 % | 99 % |
  | | network | −2.75 ±1.50 | 27 % | 13 % | 86 % | 89 % |
  | burst / slow head (4.8 s) | `nearest_job` | −3.88 ±2.26 | 8 % | 48 % | 74 % | 87 % |
  | | network | −4.26 ±2.19 | −4 % | 39 % | 73 % | 104 % |

  The required coverage is larger than the pilot suggested (97 % of the gain at 5 cm
  for the 0.7.9 network on development seeds): on the reserved seeds half the entry
  zone (1.7 s of belt travel) gives 84 % of the gain for `nearest_job` and 64 % for
  the network with the nominal head, and 36 % / 58 % with a head twice as slow; the
  curve saturates at 7.5 cm (2.5 s, the head's travel time across the conveyors at the
  nominal speed) for the nominal head and needs the whole zone (3.3 s) for the slow
  head. The coverage to specify is therefore the head's travel time across the
  conveyors, in seconds of belt travel, not a fixed fraction of the entry zone. At
  the heavy load every point of the curve is within ±1–2 points of zero (the full
  table is in `phase4_tables.md`).
* **Robustness to the network.** The reference network reproduces every contrast
  within the noise (B vs A, C vs B, D vs B: the two networks differ by less than 1 point
  in 18 of the 24 cells and by at most 2.1 points, heavy / Poisson / slow head B vs A,
  −1.34 ±1.66 vs +0.79 ±2.10; in every cell by less than the sum of the two half-widths;
  same sign wherever either interval excludes zero; same significant cells except C vs B
  at heavy / burst / nominal, −1.60 ±1.14 vs −0.30 ±0.86; `phase4_tables.md`).
  *Corrected in 0.7.16: the first version of this bullet said "within ±0.5 points in
  every regime", which was wrong (12 of 24 cells).* The reference network is the
  checkpoint from which `rl2_gen_s_b` was retrained (0.7.11), not an independent run. In C the frozen network is 0.5–1.5 points better than
  the reference in every regime (significant in three), in B the two are equal.

### 9.3 Paper 1: search, teacher and network

Treatment B (paired Δ vs `nearest_job` [%]; latency per decision [ms], p50 / p95):

| regime | known-parts search | teacher (8 futures) | network | search vs network | teacher vs network | latency search | latency network | latency teacher [s] |
|---|---|---|---|---|---|---|---|---|
| real / Poisson / nominal | −4.16 ±1.98 | −7.60 ±2.17 | −7.22 ±1.95 | **+3.35 ±1.10** | −0.39 ±1.24 | 2.0 / 6.9 | 8.5 / 13.7 | 0.43 / 1.43 |
| real / Poisson / slow | −3.73 ±2.17 | −7.72 ±2.14 | −7.25 ±1.74 | **+3.83 ±1.56** | −0.53 ±1.12 | 3.3 / 10.5 | 10.2 / 15.3 | 0.82 / 2.22 |
| real / burst / nominal | −0.57 ±1.54 | −2.68 ±1.44 | −1.96 ±1.74 | +1.54 ±1.60 | −0.65 ±1.12 | 4.3 / 12.6 | 9.8 / 14.7 | 0.84 / 2.31 |
| real / burst / slow | −4.25 ±1.91 | −6.72 ±2.12 | −6.24 ±2.00 | **+2.22 ±1.42** | −0.48 ±1.22 | 5.1 / 13.4 | 10.9 / 16.2 | 1.14 / 2.67 |
| heavy / Poisson / nominal | +0.26 ±1.45 | −3.68 ±1.58 | −2.78 ±1.85 | **+3.31 ±1.87** | −0.81 ±1.46 | 5.0 / 14.9 | 11.3 / 17.5 | 1.37 / 3.33 |
| heavy / Poisson / slow | +2.31 ±2.94 | −1.40 ±2.53 | −1.32 ±2.32 | **+3.86 ±2.86** | +0.05 ±2.24 | 5.6 / 16.0 | 11.7 / 18.0 | 1.47 / 3.54 |
| heavy / burst / nominal | −1.02 ±1.33 | −3.56 ±1.74 | −3.29 ±1.66 | **+2.46 ±1.36** | −0.27 ±0.55 | 7.3 / 21.3 | 12.2 / 19.6 | 1.86 / 5.12 |
| heavy / burst / slow | −1.89 ±1.87 | −4.15 ±1.82 | −4.15 ±1.74 | **+2.40 ±1.29** | +0.01 ±0.79 | 6.3 / 19.9 | 11.4 / 18.6 | 1.58 / 4.44 |

("search vs network" is the search's cost relative to the network's, paired:
positive = the network is better. Treatment C gives the same picture,
`phase4_tables.md`.)

* **The network alone beats `nearest_job` in 7 of 8 regimes** (−2.0 to −7.3 %; the
  exception is the heavy load with a slow head, −1.3 ±2.3), and **beats the
  known-parts search in every regime** (by 1.5 to 3.9 points, significant in 7 of 8),
  at 2–3× the search's latency and 100× less than the teacher's. Pooled over the
  regimes the network decides in 10.8 ms (p50), 17.5 ms (p95), 25.5 ms (p99), 106 ms
  at most, on one thread with the machine fully loaded.
* **The network equals its teacher**: the teacher over the network is within ±0.8
  points of the network alone in every regime, never significantly better (13–18 of
  30 seeds), at 0.4–1.9 s per decision (p50) and up to 12 s. This confirms on reserved
  seeds what every training run showed on development seeds (`rl2.md` §8.4–§8.8):
  the amortized policy recovers the sampled-future search at 1/100 of its latency.
* **The known-parts search is no longer a competitive reference** at the heavy load
  (+0.3 and +2.3 % vs `nearest_job` with Poisson arrivals): the look-ahead over the
  parts in view, without the sampled future, ranks the candidates worse than the
  rule when every belt is queueing.
* **Against the plan's success criteria** (`strong_rl_plan.md` §5): the network alone
  in the design mode is −14.1 % vs `nearest_job`/`none` at the real load (target
  ≥ −10 %); the decision latency is 8.5 ms (p50) under full load against the 5 ms
  target measured single-process in `rl2.md` §5 (4 ms), both three orders of
  magnitude below the physical slack; the hindsight gap and the generalization grid
  were deferred to a second phase-4 run (`papers.md`).

### 9.4 Engine check

The fixed-step engine at `dt = 0.001` with `nearest_job` and `nearest_head` in B and
C on every regime (960 episodes): identical processing order in 464 / 480 (B) and
461 / 480 (C) episodes, mean |Δ J| 0.03 % (B) and 0.06 % (C), 18 episodes with |Δ J|
> 0.5 % (max 1.4 % in B, 5.2 % in C), all of them order flips at near-ties, as in the
acceptance of `fastsim.md` §4 (95.8 % identical order at this `dt`). The fast
simulator's numbers stand; the network has no engine adapter and is validated through
the fast simulator only.

### 9.5 Conclusions for the papers

* Paper 1: the method to present is the sampled-future teacher plus the network that
  amortizes it; its gain over `nearest_job` and over the known-parts search is
  confirmed on reserved seeds in every regime but the saturated heavy cell with a slow
  head, where no policy without the hidden queue does better than the rule.
* Paper 2: the value of upstream visibility is the value of the action it enables
  (C vs B), the pure-information contrast (B vs A) is a null result bounded at about
  ±1.5 points, the hidden queue matters only at the heavy load (D vs B, 2.7–3.7
  points), and the coverage to specify is the head's travel time across the
  conveyors in seconds of belt travel: 7.5 cm (2.5 s) for the nominal head, the whole
  10 cm zone (3.3 s) for a head twice as slow.
