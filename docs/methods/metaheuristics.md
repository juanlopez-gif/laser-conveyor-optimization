# Metaheuristics over the decoder Φ(π)

Owner: METAHEURISTICS agent. Code: `src/optimization/solvers/metaheuristics.py`.
Tests: `tests/test_metaheuristics.py`.

Two methods are registered with the benchmark: `"sa"` (simulated annealing) and
`"ils"` (iterated local search). Both search over processing orders π and use the
simulator, through the fixed-sequence decoder Φ(π) (§3 of
[formal_formulation.md](../formal_formulation.md)), as the exact evaluator. They
accept any key of `OBJECTIVES` (tested with `avg_flow_time` and `makespan`) and a
weights dict (composite objective), because the objective is computed by
`optimization.evaluate.objective_value`.

Python API (extra knobs not exposed by the benchmark contract):

```python
from optimization.solvers.metaheuristics import iterated_local_search, simulated_annealing
sol = iterated_local_search(inst, "avg_flow_time", time_limit=30, seed=0,
                            max_evals=None, window=None)
sol.sequence, sol.info   # info: value, source, policy_values, evaluations, repairs, ...
```

## Shared machinery

| component | choice | why |
|---|---|---|
| warm start | realised orders of `fifo`, `nearest_job`, `round_robin`, `weighted_score` (`evaluate_policy(...).sequence`, dt = 0.01) | good orders are close to the policy orders; replaying a realised order gives exactly the policy's value (checked on 36 instance/policy pairs) |
| start point | the best warm start at the search dt | |
| fall-back | the best policy order is always a candidate answer | the returned order is never worse than the best policy after validation |
| neighbourhood | swap, insertion (move one part), block insertion (segment of 2–3); every part moves at most `window = min(6, n-1)` positions | large moves are almost always deadlocks or much worse; parts are roughly ordered by arrival |
| move distance | P(d) ∝ 1/d, d = 1..window (SA); ILS scans d = 1 first, then 2, ... | most improving moves are short |
| repair | on a deadlock "next part X is upstream of conveyor k, which is blocked by the unprocessed part B at the zone end", B is moved to just before X and the simulation resumes (≤ 6 repairs per evaluation) | out-of-belt-order on one conveyor is the only cause of deadlock; the repair is the minimal change that restores feasibility, so infeasible neighbours become useful ones |
| incremental evaluation | snapshots (`copy.deepcopy` of the `Simulation`) of the current order after every `stride = max(2, round(n/10))` assigned parts; a neighbour that shares the first i positions resumes from the deepest snapshot ≤ i | the decoder state right after assigning π(i−1) depends only on π[:i]; results are bit-identical to `evaluate_sequence` (tested); saved 34 % of the simulated steps on average in the experiments |
| other speed-ups | memo of every evaluated (and every repaired) order; no event log in search runs; `SEARCH_DT = 0.05` | |
| time limit | search deadline = limit − (2.2 × validation cost + 5 % of the limit); the deadline is also checked inside a running simulation (every 400 steps) | max overrun observed: none (all 96 runs finished 0.3–6 s before the limit) |
| validation | up to the 3 best distinct orders found are replayed with `evaluate_sequence(dt=0.01, check=True)`; the best feasible validated one is returned unless the best policy is better | search-dt values differ from dt = 0.01 values by ~0.05 s |
| determinism | `random.Random(seed)`; all schedules (SA cycle length, perturbation strength) are driven by evaluation counts, never by wall-clock time | identical results for the same seed and evaluation budget (`max_evals`); with a binding time limit the result depends on how many evaluations the machine completes |

### Simulated annealing (`sa`)

* T₀: 12 random moves from the start; T₀ = (lower-quartile worsening) / ln 5, i.e. a
  lower-quartile worsening is accepted with probability 0.2. A median-based T₀
  was tried first and turned the search into a random walk (random moves are
  disruptive: the median worsening is ~10× the available gains).
* Geometric cooling to T₀·10⁻³ over a cycle of `clamp(time_limit / (0.004 n) / 2, 60, 40 n)`
  evaluations (fixed cost model, so ≈ two cycles fit in the budget and the
  schedule is machine-independent). After a cycle: reheat to T₀/2^c and restart
  from the best order.
* Snapshots are rebuilt lazily, after three consecutive rejections of the current
  order (rebuilding costs about one evaluation, which is not worth it in the
  high-acceptance phase).

### Iterated local search (`ils`)

* Local search: first improvement over the full restricted neighbourhood (all
  swaps and both insertion directions with displacement ≤ window), ordered by
  displacement (adjacent moves first, random order within a displacement).
* Perturbation: `strength` (2…6) random insertions inside one local region
  (±window around a random centre).
* Acceptance: better than the current local optimum → accept and reset strength
  to 2; within 0.2 % of the best → accept (sideways walk) and increase strength;
  otherwise restart from the best and increase strength.
* Stops early when 200 consecutive kicks produce no new (uncached) evaluation:
  the reachable space is exhausted (tiny instances).

`max_evals` counts every evaluation request, cache hits included, so a budgeted
run terminates even when the whole neighbourhood is cached. (The ILS early stop
and this accounting were added after the experiments below; they only matter
when nothing new can be evaluated, i.e. not for the n >= 20 rows.)

A GA was not implemented: with 100–1000 evaluations per run (each one a full or
partial simulation) a population-based method gets only a handful of
generations; the evaluation budget is better spent on a trajectory method.

## Experiments

`generate_instance(n, seed)`, n ∈ {10, 20, 30, 40}, seeds 1–3, both objectives,
time limits 30 s and 120 s, method seed 0. Policies evaluated at dt = 0.01; every
returned order re-validated with `evaluate_sequence(dt=0.01, check=True)`:
**96/96 feasible, 0 violations**. Runs were executed 8 in parallel (other agents
were also using the machine), so evaluation counts are conservative.

Improvement (%) = 100 · (policy − method) / policy, mean over the 3 seeds;
"> NJ" = number of instances where the method is strictly better than `nearest_job`.

| objective | T [s] | method | n | vs fifo | vs nearest_job | vs round_robin | vs weighted_score | vs best policy | > NJ | runtime s | evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| avg_flow_time | 30 | ils | 10 | 3.52 | 4.57 | 11.04 | 3.52 | 3.52 | 2/3 | 28.5 | 297 |
| avg_flow_time | 30 | ils | 20 | 3.83 | 3.27 | 14.19 | 4.93 | 2.33 | 3/3 | 28.9 | 255 |
| avg_flow_time | 30 | ils | 30 | 5.09 | 2.80 | 16.66 | 5.97 | 2.12 | 3/3 | 29.2 | 159 |
| avg_flow_time | 30 | ils | 40 | 5.41 | 1.92 | 19.52 | 5.91 | 1.68 | 3/3 | 29.4 | 112 |
| avg_flow_time | 30 | sa | 10 | 3.52 | 4.57 | 11.04 | 3.52 | 3.52 | 2/3 | 28.6 | 293 |
| avg_flow_time | 30 | sa | 20 | 2.55 | 1.99 | 13.07 | 3.65 | 1.03 | 3/3 | 28.7 | 225 |
| avg_flow_time | 30 | sa | 30 | 4.07 | 1.72 | 15.74 | 5.05 | 1.05 | 2/3 | 28.8 | 146 |
| avg_flow_time | 30 | sa | 40 | 4.38 | 0.84 | 18.62 | 4.91 | 0.60 | 3/3 | 29.2 | 103 |
| avg_flow_time | 120 | ils | 10 | 3.52 | 4.57 | 11.04 | 3.52 | 3.52 | 2/3 | 113.8 | 894 |
| avg_flow_time | 120 | ils | 20 | 6.01 | 5.43 | 16.07 | 7.14 | 4.54 | 3/3 | 114.4 | 924 |
| avg_flow_time | 120 | ils | 30 | 6.76 | 4.54 | 18.11 | 7.61 | 3.87 | 3/3 | 114.1 | 670 |
| avg_flow_time | 120 | ils | 40 | 7.53 | 4.11 | 21.31 | 8.01 | 3.88 | 3/3 | 115.3 | 561 |
| avg_flow_time | 120 | sa | 10 | 3.52 | 4.57 | 11.04 | 3.52 | 3.52 | 2/3 | 113.9 | 693 |
| avg_flow_time | 120 | sa | 20 | 3.91 | 3.35 | 14.26 | 5.00 | 2.41 | 3/3 | 114.4 | 881 |
| avg_flow_time | 120 | sa | 30 | 4.42 | 2.11 | 16.06 | 5.32 | 1.43 | 3/3 | 114.5 | 663 |
| avg_flow_time | 120 | sa | 40 | 5.10 | 1.57 | 19.26 | 5.65 | 1.34 | 3/3 | 115.1 | 499 |
| makespan | 30 | ils | 10 | 1.41 | 1.89 | 4.56 | 1.41 | 1.29 | 3/3 | 28.7 | 304 |
| makespan | 30 | ils | 20 | 3.06 | 2.15 | 4.79 | 3.46 | 2.15 | 3/3 | 28.9 | 246 |
| makespan | 30 | ils | 30 | 0.64 | 0.00 | 2.82 | 0.91 | 0.00 | 0/3 | 28.6 | 163 |
| makespan | 30 | ils | 40 | 0.41 | 0.47 | 2.76 | 0.14 | 0.14 | 2/3 | 29.0 | 109 |
| makespan | 30 | sa | 10 | 0.17 | 0.66 | 3.36 | 0.17 | 0.04 | 3/3 | 28.6 | 274 |
| makespan | 30 | sa | 20 | 1.85 | 0.92 | 3.61 | 2.24 | 0.92 | 2/3 | 28.7 | 189 |
| makespan | 30 | sa | 30 | 0.64 | 0.00 | 2.82 | 0.91 | 0.00 | 0/3 | 28.8 | 122 |
| makespan | 30 | sa | 40 | 0.36 | 0.43 | 2.72 | 0.09 | 0.09 | 2/3 | 28.7 | 82 |
| makespan | 120 | ils | 10 | 1.41 | 1.89 | 4.56 | 1.41 | 1.29 | 3/3 | 114.5 | 994 |
| makespan | 120 | ils | 20 | 3.06 | 2.15 | 4.79 | 3.46 | 2.15 | 3/3 | 114.9 | 1003 |
| makespan | 120 | ils | 30 | 0.64 | 0.00 | 2.82 | 0.91 | 0.00 | 0/3 | 113.9 | 637 |
| makespan | 120 | ils | 40 | 0.60 | 0.66 | 2.95 | 0.33 | 0.33 | 2/3 | 114.5 | 584 |
| makespan | 120 | sa | 10 | 0.17 | 0.66 | 3.36 | 0.17 | 0.04 | 3/3 | 113.9 | 805 |
| makespan | 120 | sa | 20 | 1.66 | 0.73 | 3.42 | 2.07 | 0.73 | 2/3 | 114.3 | 787 |
| makespan | 120 | sa | 30 | 0.64 | 0.00 | 2.82 | 0.91 | 0.00 | 0/3 | 113.5 | 545 |
| makespan | 120 | sa | 40 | 0.41 | 0.47 | 2.76 | 0.14 | 0.14 | 2/3 | 114.8 | 343 |

Repairs: 100 per run on average (out of a few hundred evaluations); after repair,
0 neighbours remained infeasible. Snapshots skipped 34 % of the simulated steps.

### Findings

* **ILS dominates SA** on both objectives and at both budgets (e.g. avg flow time,
  n = 40, 120 s: 3.9 % vs 1.3 % over the best policy). With a few hundred
  evaluations, the systematic "short moves first" scan of ILS uses the budget
  better than random sampling; SA only ties ILS on n = 10.
* **avg_flow_time** has real room: the search beats `nearest_job` on 34/36 ILS
  runs (all except n = 10, seed 1, where FIFO = NJ appears optimal), by 2–5 %, and
  the gain grows with the budget (30 s → 120 s roughly doubles it for n ≥ 20).
  n = 30–40 at 30 s are budget-limited (110–160 evaluations).
* **makespan** offers little: in this low-load regime the makespan is dominated by
  the last arrivals, and only the last busy period matters. Gains are 0–2 % and
  for n = 30 none of the methods improved on the best policy.
* **Structure of good orders** (ILS, 120 s; "same" = pairs of parts of the same
  conveyor processed out of belt order, "cross" = inversions between conveyors,
  w.r.t. arrival order):

  | instance | NJ same / cross | best avg-flow same / cross |
  |---|---|---|
  | n40 s1 | 7 / 15 | 2 / 14 |
  | n40 s2 | 4 / 12 | 1 / 8 |
  | n40 s3 | 6 / 10 | 1 / 8 |
  | n30 s1 | 7 / 14 | 2 / 14 |
  | n30 s3 | 4 / 7 | 0 / 5 |
  | n20 s3 | 4 / 6 | 1 / 4 |

  Good orders keep almost every conveyor in belt order and deviate from arrival
  order mainly *across* conveyors. `nearest_job` loses time precisely by
  processing a downstream-close but later part of the same conveyor first (e.g.
  `P015 P013 P014 P012 P011` on conveyor 1 for n = 30, seed 1): the earlier part
  stays on the belt longer, and it soon blocks the conveyor at the zone end.
  The improving moves restore belt order on a conveyor and pull parts of a lightly
  loaded conveyor (here conveyor 3) earlier, behind a late part of a busy conveyor
  (`... P017 P020/3 P016/3 P010/2 P018 ...`). Parts never moved more than 10
  positions from their arrival rank; this justifies the restricted neighbourhood.
  A practical consequence for online policies: *nearest-job with per-conveyor FIFO*
  (pick the nearest conveyor, but its first part) is the natural candidate rule.

## Limitations

* Φ only produces semi-active (earliest) schedules; deliberate waiting is out of
  reach (see §3 of the formulation), so the values are upper bounds on the optimum
  of 𝒫, not optima.
* Search at dt = 0.05: the order found best at the search dt may not be best at
  dt = 0.01 (differences ~0.05 s); the top-3 candidates are validated to soften this.
* For n ≥ 30 and 30 s the budget is only ~100–160 evaluations; the methods are
  budget-limited there (a faster simulator or event-driven decoder would help more
  than any parameter tuning).
* The time limit is respected but a run uses ~95 % of it whatever the instance
  (no early stop at local optima for SA; ILS keeps perturbing).
* Experiments: 3 seeds per size, one method seed; parallel runs on a shared
  machine, so absolute evaluation counts are machine-dependent.
