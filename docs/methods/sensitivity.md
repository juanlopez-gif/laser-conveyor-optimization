# Sensitivity of the policies to the cell parameters (version 0.6.0)

Code: `experiments/sensitivity_sweep.py`. Outputs (not versioned):
`results/experiments/v0.6_sensitivity/` (`runs.csv`, `summary.md`, `sensitivity.png`).
Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.6.0).

> In every cell the policies run with `when_idle` and are compared with `nearest_job` /
> `none`, so the numbers mix the policy and pre-positioning effects. The fresh-seed run
> of 0.6.1 separates them at the default and heavy loads
> ([confirmatory.md](confirmatory.md)).

## 1. Why

* On 2026-11-19 the user decided to keep the default cell: laser 10 cm/s with no
  acceleration, processing 2.5 s, belts 3 cm/s, mean inter-arrival 10/12/14 s.
* Belt speed and job time can be adjusted in the real cell, and the arrival rate is
  not measured yet.
* So the question is how the ranking of the policies changes when these parameters
  move.

## 2. Design

**One-at-a-time sweep around the default cell**, at the default load:

| parameter | config key | values (default in bold) |
|---|---|---|
| belt speed [cm/s] | `conveyors.speed` | 1.5, 2, **3**, 4, 5, 6 |
| laser speed [cm/s] | `laser.speed_x = laser.speed_y` | 5, **10**, 16.67 |
| processing time [s] | `processing.time` | 1.5, **2.5**, 3, 4 |
| arrival factor | `arrivals.stochastic.mean_interarrival` = factor × [10, 12, 14] s | 0.6, 0.8, **1**, 1.2 (< 1 = more parts) |

* Policies:
  * reference: `nearest_job` / `none`;
  * all others with pre-positioning `when_idle`: `nearest_head`, `rollout`, `rl` (agent
    `rl06_wi_s0`), `rl_ensemble` (3 agents), `rl_filter`, `rl_gated`, `rl_rollout`.
* Seeds: **10 per cell** (1000–1009), common random numbers within each cell.
* Size and time: 17 sweep points (14 distinct cells; the default cell is run once per
  sweep) × 8 policies × 10 seeds = 1360 runs, in 3574 s on 2 workers.
* Metric: paired relative difference in penalized flow time against the reference **in
  the same cell**, with a normal 95 % half-width (±).
* The agents were trained only in the default cell (default and heavy loads). Every
  other cell is out of their training distribution.

## 3. Results

Paired Δ [%] vs `nearest_job`/`none` in the same cell; negative is better. The first row
of each table is the reference's absolute penalized flow [s].

**Belt speed [cm/s]**

| policy | 1.5 | 2 | **3** | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| reference [s] | 27.1 | 26.1 | 25.4 | 25.5 | 25.6 | 25.5 |
| `rl_rollout` | **−11.0 ±3.7** | −10.7 ±3.6 | **−9.7 ±2.2** | **−9.8 ±4.3** | **−9.5 ±4.9** | **−6.3 ±3.5** |
| `rollout` | −9.5 ±2.8 | **−11.6 ±3.6** | −8.2 ±2.0 | −8.5 ±5.7 | −8.5 ±4.3 | −5.1 ±3.6 |
| `rl_filter` | −7.4 ±3.7 | −9.9 ±4.7 | −6.9 ±3.0 | −6.7 ±4.4 | −6.9 ±3.8 | −5.4 ±4.2 |
| `rl_gated` | −7.6 ±4.4 | −7.5 ±5.6 | −5.6 ±2.9 | −6.2 ±5.7 | −5.2 ±5.6 | −4.3 ±3.8 |
| `rl_ensemble` | −5.4 ±5.2 | −6.9 ±2.6 | −6.1 ±2.7 | −3.9 ±5.9 | −5.3 ±5.7 | −4.9 ±4.8 |
| `rl` | −5.6 ±5.1 | −5.8 ±4.0 | −4.0 ±3.1 | −3.9 ±6.5 | −5.2 ±6.0 | −3.9 ±3.7 |
| `nearest_head` | −5.6 ±5.0 | −9.1 ±4.0 | −3.8 ±2.5 | −5.8 ±5.0 | −5.3 ±4.1 | −4.7 ±3.1 |

**Laser speed [cm/s]**

| policy | 5 | **10** | 16.67 |
|---|---|---|---|
| reference [s] | 50.1 | 25.4 | 17.3 |
| `rl_rollout` | **−11.9 ±3.1** | **−9.7 ±2.2** | **−8.3 ±1.7** |
| `rollout` | −9.3 ±3.6 | −8.2 ±2.0 | −6.8 ±1.6 |
| `rl_filter` | −7.1 ±3.5 | −6.9 ±3.0 | −5.4 ±3.6 |
| `rl_gated` | −8.7 ±2.7 | −5.6 ±2.9 | −4.5 ±3.0 |
| `rl_ensemble` | −7.7 ±3.8 | −6.1 ±2.7 | −3.9 ±2.9 |
| `rl` | −8.3 ±2.6 | −4.0 ±3.1 | −2.9 ±4.9 |
| `nearest_head` | −8.4 ±2.9 | −3.8 ±2.5 | −5.5 ±2.0 |

**Processing time [s]**

| policy | 1.5 | **2.5** | 3 | 4 |
|---|---|---|---|---|
| reference [s] | 11.0 | 25.4 | 38.0 | 72.2 |
| `rl_rollout` | **−16.6 ±2.5** | **−9.7 ±2.2** | **−6.7 ±3.9** | −3.1 ±3.4 |
| `rollout` | −16.1 ±2.1 | −8.2 ±2.0 | −6.4 ±3.4 | **−4.6 ±3.4** |
| `rl_filter` | −15.7 ±2.0 | −6.9 ±3.0 | −4.2 ±4.2 | −1.9 ±3.6 |
| `rl_gated` | −15.1 ±3.1 | −5.6 ±2.9 | −0.8 ±3.8 | −0.4 ±3.4 |
| `rl_ensemble` | −14.1 ±2.5 | −6.1 ±2.7 | −1.8 ±3.5 | −2.0 ±2.9 |
| `rl` | −13.2 ±2.7 | −4.0 ±3.1 | −0.2 ±3.7 | −0.1 ±2.9 |
| `nearest_head` | −15.4 ±2.3 | −3.8 ±2.5 | −0.9 ±3.5 | −1.6 ±3.0 |

**Arrival factor (mean inter-arrival = factor × [10, 12, 14] s)**

| policy | 0.6 | 0.8 | **1** | 1.2 |
|---|---|---|---|---|
| reference [s] | 78.4 | 42.5 | 25.4 | 18.0 |
| `rl_rollout` | −0.6 ±3.7 | −6.6 ±3.7 | **−9.7 ±2.2** | **−11.8 ±2.2** |
| `rollout` | **−4.8 ±3.0** | **−8.7 ±3.7** | −8.2 ±2.0 | −11.4 ±2.4 |
| `rl_filter` | +0.3 ±3.3 | −6.2 ±5.5 | −6.9 ±3.0 | −11.2 ±2.7 |
| `rl_gated` | +2.4 ±3.5 | −0.5 ±5.6 | −5.6 ±2.9 | −10.4 ±2.5 |
| `rl_ensemble` | +1.3 ±2.4 | −1.3 ±3.7 | −6.1 ±2.7 | −9.1 ±3.3 |
| `rl` | +2.3 ±2.9 | −0.1 ±3.6 | −4.0 ±3.1 | −9.9 ±3.5 |
| `nearest_head` | −0.1 ±3.5 | −3.4 ±5.3 | −3.8 ±2.5 | −10.8 ±1.9 |

The default column (bold heading) is the same cell in all four tables; it was simulated
once per sweep with the same seeds, so the four copies are identical.

## 4. Findings

1. **`rl_rollout` is the best or near-best policy in 11 of the 14 distinct cells.**
   This covers every belt speed, every laser speed, processing times up to 3 s, and the
   default and lighter arrival rates.
2. **The plain `rollout` is better when the cell is congested**: arrival factor 0.6 and
   0.8, and processing 4 s. The agents were trained at the default parameters and do
   not generalize to much higher load:
   * `rl` alone is *worse* than the reference at factor 0.6 (+2.3 %), and so is
     `rl_gated` (+2.4 %);
   * `rl_rollout` loses most of its gain there (−0.6 %).
   Training on randomized cell parameters (domain randomization,
   `docs/plans/strong_rl_plan.md`) targets exactly this.
3. **The agent alone performs about like `nearest_head`.** Its value comes from being
   combined with the rollout.
4. **Processing time is the parameter that matters most.**
   * The reference goes from 11 s (1.5 s jobs) to 72 s (4 s jobs).
   * Every policy gains most with short jobs (−13 to −17 %).
   * If the real job can be shortened, that improves the cell far more than any
     dispatching policy.
5. **Belt speed barely matters.** The reference stays at 25–27 s from 1.5 to 6 cm/s.
   The relative gains shrink slightly at 6 cm/s.
6. **A faster laser lowers the reference cost** (50 → 17 s) **and the relative gains
   of the policies**: less travel time leaves less to optimize.

## 5. Caveats

* 10 seeds per cell; the half-widths (±2 to ±6 points) are often larger than the
  differences between the top policies. The sweep shows trends, not rankings.
* The seeds 1000–1009 are a subset of the evaluation seeds. The confirmatory runs will
  use fresh seeds 3000+.
* One-at-a-time design: interactions are not studied, for example a fast belt combined
  with long jobs.
* The real values of the arrival rate, belt speed and job time still have to be
  measured in the cell.

## 6. Reproduce

```bash
PYTHONIOENCODING=utf-8 python experiments/sensitivity_sweep.py --workers 2 --seeds 10 \
    --tag v0.6_sensitivity --policies nearest_job nearest_head rollout \
    "rl@model=models/rl06_wi_s0.zip" \
    "rl_ensemble@models=models/rl06_wi_s0.zip,models/rl06_wi_s1.zip,models/rl06_wi_s2.zip" \
    "rl_filter@model=models/rl06_wi_s0.zip" "rl_gated@model=models/rl06_wi_s0.zip" \
    "rl_rollout@model=models/rl06_wi_s0.zip"
```
