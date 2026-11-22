# Blocking penalty, `nearest_head` and laser pre-positioning (version 0.4.0)

Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.4.0). Model:
[../formal_formulation.md](../formal_formulation.md) §1.4 (penalty), §1.5 (pre-positioning),
§4 (policies).

## Set-up

* Two loads: **default** (conveyors 3 cm/s, laser 10 cm/s, processing 2.5 s, mean
  inter-arrival 10/12/14 s, ρ ≈ 0.64) and **heavy** (`config/examples/heavy_load.yaml`).
* 30 stochastic seeds (1000–1029), 300 s of arrivals then drained, `dt = 0.01`. Every
  cell uses the **same seeds** (common random numbers), so policies are compared
  **pairwise per seed**.
* Objective: `penalized_flow_time` = mean flow time + Π/n, Π = 10·N_B + 1·T_B
  (placeholder weights).
* Paired Δ = mean over seeds of the relative difference of penalized flow time with
  respect to `nearest_job` without pre-positioning (the previous default), with a
  normal 95 % confidence interval; "better on" counts the seeds where the cell is
  strictly better than that baseline.
* Two rollout variants: **rollout (flow)** = look-ahead objective `avg_flow_time`
  (default), **rollout (penalized)** = look-ahead objective `penalized_flow_time`.

Reproduce:

```bash
python experiments/blocking_prepositioning_study.py --workers 2
python experiments/blocking_prepositioning_study.py --policies nearest_job rollout@avg_flow_time \
    --modes none when_idle --tag rollout_objective
python experiments/blocking_prepositioning_study.py \
    --summarize v0.4_blocking_prepositioning rollout_objective --tag v0.4_summary
```

The full table (all 5 cheap policies × 3 modes, both rollout variants) is written to
`results/experiments/v0.4_summary/summary.md` (results are not versioned). Main rows:

### Default load

| policy | pre-positioning | mean flow [s] | blocking events | blocked time [conv-s] | penalized flow [s] | paired Δ vs nearest_job/none [%] (95 % CI) | better on |
|---|---|---|---|---|---|---|---|
| rollout (flow) | when_idle | 18.16 | 9.60 | 106.7 | **20.64** | **−10.1 (−11.6, −8.5)** | 30/30 |
| rollout (penalized) | when_idle | 19.22 | 7.03 | 73.8 | 20.96 | −9.6 (−12.6, −6.7) | 26/30 |
| **nearest_head** | **when_idle** | 19.41 | 8.90 | 93.6 | 21.63 | **−6.4 (−9.0, −3.9)** | **29/30** |
| nearest_job | when_idle | 19.30 | 9.03 | 121.1 | 21.90 | −4.9 (−6.9, −2.8) | 23/30 |
| rollout (flow) | none | 19.09 | 10.10 | 120.0 | 21.80 | −4.2 (−5.4, −3.0) | 26/30 |
| nearest_head | none | 20.01 | 8.83 | 94.7 | 22.26 | −2.6 (−4.4, −0.7) | 22/30 |
| nearest_job | none | 19.95 | 9.80 | 132.5 | 22.80 | 0 (baseline) | – |
| nearest_job | always | 19.57 | 9.77 | 154.6 | 22.69 | −1.0 (−3.4, +1.4) | 15/30 |
| nearest_head | always | 21.70 | 7.50 | 151.5 | 24.50 | +4.6 (+0.1, +9.1) | 10/30 |
| fifo | none | 24.43 | 24.03 | 138.8 | 29.06 | +23.3 | 3/30 |
| weighted_score | none | 25.36 | 29.93 | 148.9 | 30.81 | +29.3 | 3/30 |
| round_robin | none | 32.36 | 34.53 | 187.4 | 38.89 | +63.2 | 0/30 |

### Heavy load

| policy | pre-positioning | mean flow [s] | blocking events | blocked time [conv-s] | penalized flow [s] | paired Δ vs nearest_job/none [%] (95 % CI) | better on |
|---|---|---|---|---|---|---|---|
| rollout (flow) | none | 62.97 | 20.63 | 594.0 | **71.58** | **−3.7 (−5.1, −2.4)** | 25/30 |
| rollout (flow) | when_idle | 63.88 | 21.23 | 605.9 | 72.68 | −2.0 (−3.5, −0.5) | 17/30 |
| nearest_head | always | 65.56 | 13.70 | 539.1 | 72.85 | −1.8 (−3.6, +0.0) | 19/30 |
| nearest_job | always | 65.50 | 15.10 | 568.5 | 73.27 | −1.2 (−3.2, +0.8) | 20/30 |
| nearest_head | when_idle | 65.96 | 18.63 | 569.9 | 74.11 | −0.1 (−1.5, +1.3) | 15/30 |
| nearest_job | none | 65.56 | 19.17 | 610.3 | 74.18 | 0 (baseline) | – |
| nearest_head | none | 66.40 | 18.47 | 572.5 | 74.56 | +0.5 (−0.9, +1.9) | 16/30 |
| rollout (penalized) | none | 68.30 | 21.23 | 531.0 | 76.29 | +2.5 (+0.3, +4.8) | 11/30 |
| fifo | none | 109.16 | 58.13 | 942.1 | 125.51 | +69.6 | 0/30 |

(Standard deviations across seeds are large — 7–17 s of mean flow — because the seeds
differ in how many parts arrive; this is why the comparison is paired.)

## Findings

1. **`nearest_head` reduces blocking.** At default load it cuts the blocked conveyor
   time by 29 % (94.7 vs 132.5 conveyor-s) and the penalized flow time by 2.6 %
   without pre-positioning; with `when_idle` pre-positioning by **6.4 %, better on
   29/30 seeds**. It is the best rule that costs no computation. At heavy load its
   gain is within noise unless pre-positioning is `always`.
2. **Pre-positioning helps only if it is revocable or the cell is saturated.**
   `when_idle` improves every good policy at default load (nearest_job −4.9 %,
   nearest_head −6.4 %, rollout −10.1 %). `always` (non-revocable commitment to a part
   that has not arrived) is neutral for `nearest_job` and **harmful** for
   `nearest_head` (+4.6 %) at default load; at heavy load, when the laser is always
   busy and the next part is almost always already coming, `always` is the best mode
   for the simple rules (blocking events −29 %).
3. **Do not optimize the penalty inside a short look-ahead.** The rollout whose
   look-ahead minimizes `penalized_flow_time` blocks less (73.8 vs 106.7 conveyor-s at
   default load) but loses flow time, and at heavy load it is worse than `nearest_job`
   **even on the penalized metric** (+2.5 %). The look-ahead only sees the parts
   already present and finishes them, so it over-weights the blocks it can see and
   ignores that delaying flow causes future blocks. Minimizing flow time in the
   look-ahead reduces blocking implicitly and gives the best penalized result
   (−10.1 % default, −3.7 % heavy). The rollout default objective is therefore kept at
   `avg_flow_time`.
4. **Ranking (penalized flow, default load):** rollout (flow) + when_idle <
   rollout (penalized) + when_idle < nearest_head + when_idle < nearest_job + when_idle ≈
   rollout (flow) < nearest_head < nearest_job ≪ fifo < weighted_score ≪ round_robin.
   FIFO, weighted score and round robin are 20–120 % worse and block 2.5–4 times more
   often; pre-positioning does not rescue them.
5. **Cost of rollout:** ≈ 9 s of computation per 300 s run at default load and ≈ 53 s
   at heavy load (one core), versus milliseconds for `nearest_head`.

## Limitations

* The penalty weights (10 s per event, 1 s per conveyor-second) are placeholders; the
  ranking between "block less" and "flow faster" policies may change with other
  weights.
* Parameters of the cell (speeds, processing and arrival distributions) are not yet
  calibrated with real data.
* `when_idle` / `always` require knowing that a part is on the entry zone (sensor);
  not confirmed for the real cell at the time, so the default remained `none` until
  0.7.6. Since 0.7.7 the planned cell's camera covers the entry zone and `when_idle`
  is the design mode (`docs/assumptions.md` C4).
