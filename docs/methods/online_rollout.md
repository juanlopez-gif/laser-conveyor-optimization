# Online method: rollout (one-step look-ahead) policy

Owner: ONLINE agent. Code: `src/policies/rollout.py` (policy `rollout`),
`src/optimization/solvers/online.py` (benchmark methods `policy:rollout`,
`policy:rollout_exact`), tests: `tests/test_rollout.py`.

## 1. Method

Rollout (Bertsekas, *Rollout algorithms*, see e.g. Bertsekas, Tsitsiklis & Wu 1997;
Bertsekas, *Reinforcement Learning and Optimal Control*, 2019, ch. 2) improves a
*base policy* $\mu$ by one step of look-ahead. At a decision epoch with state $x$
(laser idle, set $C(x)$ of READY reachable parts) it chooses

$$\tilde\mu(x) \in \arg\min_{c \in \hat C(x)} \; \hat J_\mu\big(f(x, c)\big),$$

where $f(x,c)$ is the state after assigning part $c$ and $\hat J_\mu$ is the cost of
continuing from there with the base policy, obtained by **simulating a clone of the
running simulation**. Here:

* base policy $\mu$ = `nearest_job` (configurable);
* $\hat C(x)$ = the `max_candidates` candidates with the smallest interception time,
  **always including the base choice $\mu(x)$** ("fortified" rollout);
* $\hat J_\mu$ = cost of the parts *known at decision time* (arrived and not yet
  completed): the clone runs until all of them are processed (or `horizon_s`
  simulated seconds). Default cost: their average flow time (sunk flow time of
  already completed parts does not depend on the decision, so this is the online
  counterpart of `avg_flow_time`). Unfinished parts at the end of a truncated
  horizon count as finishing at the horizon (lower bound on their completion).
* Ties are broken in favour of the base choice, then by travel time.

### Cost-improvement property and when it applies

If (i) the base policy is a state-feedback rule (`nearest_job` is: its choice
depends only on the current state), (ii) the look-ahead is exact (same `dt` as the
real run, no horizon truncation, every candidate or at least the base choice
evaluated) and (iii) no new part arrives after the first decision, then by
induction over decision epochs
$J_{\tilde\mu}(x_0) \le \hat J_\mu(f(x_0, \tilde\mu(x_0))) \le \hat J_\mu(f(x_0,\mu(x_0))) = J_\mu(x_0)$:
the rollout run is never worse than the base run for the look-ahead objective. The
test suite checks this on instances where all parts arrive at $t=0$
(`test_never_worse_than_base_when_lookahead_is_exact`, flow time, makespan,
waiting time).

In the real cell (iii) fails: arrivals are unknown, so the look-ahead optimises the
*known* parts only and a choice that is best for them can delay a part that
arrives a few seconds later. Using a coarser `dt_lookahead` also breaks (ii)
slightly. Hence on stochastic instances the improvement is empirical, not
guaranteed: it holds on average and on most seeds (section 4), not on every seed.

## 2. Fairness: what the look-ahead may and may not see

The engine pre-computes all arrivals in `Simulation.pending_arrivals`; a policy
that read that deque would be clairvoyant. The clone (`RolloutPolicy.clone`) is
built with `copy.deepcopy` and a pre-filled memo so that:

| object of the original | in the clone |
|---|---|
| `pending_arrivals` (real future) | **empty deque** (optionally synthetic arrivals, below) |
| `log` (EventLog) | no-op log (nothing written anywhere) |
| `trace` | `None`; `trace_every` set to a huge value (no sampling) |
| `metrics.queue_series` | fresh empty list (no O(T) copy) |
| `policy` (the rollout itself) | the base policy, wrapped to force the first choice (no recursion) |
| `cfg`, geometry, conveyor/laser specs, stop-policy strategy | shared (immutable / stateless) |
| parts, conveyors, laser, metrics counters, everything else | private deep copies |

**Forcing the first decision.** The clone is taken inside step 5 (dispatch) of the
original, after steps 1–4. In that state steps 1–4 are idempotent (the laser is
idle, due arrivals have been generated, admission already ran), so the clone
re-executes the whole step with a policy that returns the forced part first and
then delegates to the base policy. This reuses the engine code instead of copying
it. `test_forced_clone_reproduces_base_run` verifies that a clone given the real
future and the same `dt`, forced to the base choice, reproduces the base run bit for
bit (all start/completion times, laser distance, end time).

**Different `dt` in the clone.** The engine defines `time = step_index * dt`. The
clone's class is switched to a subclass whose clock is
`t0 + (step_index - k0) * dt`, so the clone can use `dt_lookahead` without
corrupting time; with the same `dt` the original formula is kept exactly.

**Optional synthetic futures** (`future_samples > 0`, off by default): each scenario
draws Poisson arrivals in `(t, t + future_window_s]` from the *known rates* (config
mean inter-arrival times; for explicit instances the empirical rate observed so
far) and processing times from the configured distribution. The random stream
depends only on (seed, decision time, scenario index), so all candidates are
compared on the same scenarios and runs stay deterministic. The real future list
is never read.

Tests (`tests/test_rollout.py`):

* `test_no_future_information_is_used` — instance A = two parts at $t=1.2$; instance B
  = A plus one part on conveyor 2 at $t=11.4$. A *clairvoyant* variant (test-only
  subclass that copies the real future into the clone) changes its first decision
  (P001 vs P002 at $t=4.54$), so the instance is discriminating; the real rollout
  (default, exact, and with synthetic futures) makes identical decisions in A and B
  before $t=11.4$.
* `test_clone_is_isolated_and_has_no_future` — no shared mutable objects, empty
  future, null log, no trace, base (non-rollout) policy.
* `test_lookahead_leaves_original_untouched` — a full fingerprint of the original
  (parts, conveyors, laser, metrics, log length, pending arrivals, clock) is
  identical before/after evaluating every candidate, and the run then continues
  exactly as without look-ahead.
* `test_determinism` — two rollout runs give identical parts, events and summary.
* `test_schedule_passes_formal_check` — `check_schedule` returns no violations.

## 3. Parameters (`policy.params.rollout.<name>`)

| name | default | meaning |
|---|---|---|
| `base_policy` | `nearest_job` | registered policy simulated after the forced choice (`rollout` refused) |
| `base_params` | `{}` | its parameters |
| `max_candidates` | 6 | candidates evaluated (best by travel time); base choice always added; `null` = all |
| `horizon_s` | `null` | look-ahead cap [s]; `null` = until known parts are processed (bounded by `max_time`) |
| `dt_lookahead` | 0.05 | clone time step [s]; `null` = simulation `dt` (exact, ~4–5x slower) |
| `objective` | `avg_flow_time` | `avg_flow_time`, `avg_waiting_time`, `makespan`, `max_lateness`, `total_laser_travel`, `total_conveyor_stop_time`, or a dict of weights |
| `future_samples` | 0 | synthetic future scenarios per candidate (0 = known parts only) |
| `future_window_s` | 30 | window of the synthetic arrivals [s] |

Example: `--set policy.name=rollout --set policy.params.rollout.dt_lookahead=null`.
The benchmark method `policy:rollout` uses the benchmark objective as look-ahead
objective when supported (otherwise `avg_flow_time`); `policy:rollout_exact` also
sets `dt_lookahead=null, max_candidates=null`.

## 4. Experiments

Setup: stochastic arrivals (`arrivals.mode=stochastic`), 30 seeds (1000–1029), common
random numbers (all policies see the same parts), full default horizon
(`simulation.arrival_horizon=300`, drain), rollout with default parameters.
Values: mean ± 95% CI half-width ($1.96\,s/\sqrt{n}$) over the 30 runs. Raw data:
`results/experiments/rollout_default_30/` and `results/experiments/rollout_heavy_30/`.

```
python run_experiments.py --runs 30 --policies fifo nearest_job round_robin weighted_score rollout --workers 2 --tag rollout_default_30
python run_experiments.py --runs 30 --policies fifo nearest_job round_robin weighted_score rollout --workers 2 --config config/examples/heavy_load.yaml --tag rollout_heavy_30
```

### 4.1 Default load (config/default_config.yaml)

| metric | fifo | nearest_job | round_robin | weighted_score | rollout |
|---|---|---|---|---|---|
| avg flow time [s] | 24.43 ± 3.83 | 19.95 ± 2.32 | 32.36 ± 5.56 | 25.36 ± 4.31 | **19.09 ± 2.15** |
| avg waiting time [s] | 21.93 ± 3.83 | 17.45 ± 2.32 | 29.86 ± 5.56 | 22.86 ± 4.31 | **16.59 ± 2.15** |
| makespan [s] | 328.63 ± 6.10 | 319.86 ± 3.44 | 342.95 ± 8.54 | 331.81 ± 7.46 | **317.61 ± 3.31** |
| laser travel [cm] | 1139.53 ± 41.28 | 1039.76 ± 35.58 | 1351.99 ± 61.50 | 1157.55 ± 47.67 | **1015.56 ± 37.89** |
| conveyor stop time [s] | 332.41 ± 48.49 | 305.72 ± 33.77 | 381.08 ± 54.21 | 342.57 ± 54.21 | **296.42 ± 33.48** |

Paired differences rollout − nearest_job (same seed), mean ± 95% CI, and number of
seeds where rollout is better / worse / equal:

| metric | difference | better / worse / equal |
|---|---|---|
| avg flow time | −0.863 ± 0.295 s | 28 / 2 / 0 |
| makespan | −2.248 ± 0.838 s | 25 / 1 / 4 |
| laser travel | −24.2 ± 11.9 cm | 22 / 8 / 0 |
| conveyor stop time | −9.31 ± 6.88 s | 20 / 10 / 0 |

### 4.2 Heavy load (config/examples/heavy_load.yaml)

| metric | fifo | nearest_job | round_robin | weighted_score | rollout |
|---|---|---|---|---|---|
| avg flow time [s] | 109.16 ± 9.33 | 65.56 ± 5.81 | 145.29 ± 11.80 | 115.89 ± 10.14 | **62.97 ± 5.92** |
| avg waiting time [s] | 106.16 ± 9.33 | 62.56 ± 5.81 | 142.29 ± 11.80 | 112.89 ± 10.14 | **59.97 ± 5.92** |
| makespan [s] | 498.93 ± 17.36 | 402.90 ± 10.80 | 555.73 ± 21.42 | 514.43 ± 18.78 | **398.17 ± 10.14** |
| laser travel [cm] | 1117.83 ± 40.82 | 651.06 ± 18.48 | 1375.62 ± 58.31 | 1182.91 ± 44.59 | **628.75 ± 19.25** |
| conveyor stop time [s] | 1221.48 ± 59.87 | 849.93 ± 46.18 | 1300.70 ± 74.57 | 1267.05 ± 66.20 | **841.57 ± 42.32** |

| metric | rollout − nearest_job | better / worse / equal |
|---|---|---|
| avg flow time | −2.588 ± 0.818 s | 25 / 5 / 0 |
| makespan | −4.730 ± 1.775 s | 25 / 5 / 0 |
| laser travel | −22.3 ± 11.2 cm | 23 / 7 / 0 |
| conveyor stop time | −8.36 ± 19.72 s (not significant) | 14 / 16 / 0 |

All policies complete every part in every run (77.47 ± 3.39 parts at default load,
93.13 ± 3.60 at heavy load).

### 4.3 Computational effort

Measured on this PC (Windows, Python 3.11, 2 cores) with `RolloutPolicy.stats()`.
Single runs of seed 42 with nothing else running (300 s arrival horizon + drain; a
plain `nearest_job` run takes about 1 s):

| config | arrivals | `dt_lookahead` | wall time / run | decisions | mean / max decision time |
|---|---|---|---|---|---|
| default | deterministic | 0.05 (default) | 3.1 s | 78 | 23.5 / 53 ms |
| default | stochastic | 0.05 (default) | 3.7 s | 80 | 31.0 / 119 ms |
| default | deterministic | = dt (0.01, exact) | 8.5 s | 78 | 92.4 / 212 ms |
| default | stochastic | = dt (0.01, exact) | 11.0 s | 80 | 122 / 457 ms |
| heavy | deterministic | 0.05 (default) | 24.9 s | 93 | 251 / 488 ms |
| heavy | stochastic | 0.05 (default) | 31.3 s | 101 | 294 / 598 ms |
| heavy | deterministic | = dt (0.01, exact) | 107 s | 93 | 1136 / 2097 ms |
| heavy | stochastic | = dt (0.01, exact) | 130 s | 101 | 1273 / 2362 ms |

Over 10 stochastic seeds (1000–1009, default parameters; measured while the test
suite was running on the other core, so these wall times are inflated): default
load 8.5 s/run (max 20.0 s), 80 decisions/run, decision time mean 80.6 ms, max
303 ms; heavy load 35.0 s/run (max 73.6 s), 95 decisions/run, mean 334 ms, max
1073 ms. A 300 s default run therefore stays well under the 2-minute target; the
exact look-ahead is 3–4x slower and only stays under 2 min at default load.

Exact look-ahead vs. default (`dt_lookahead=0.05`) on seed 42: average flow time
14.93 vs 14.90 s (default deterministic), 16.44 vs 16.27 s (default stochastic),
66.46 vs 66.46 s (heavy deterministic), 67.22 vs 69.46 s (heavy stochastic). The
coarse step does not systematically hurt; single-seed differences come from the
different decisions it induces.

Wall time of the experiment batches on this PC (2 workers, all 5 policies, 30 seeds):
167 s at default load, 954 s at heavy load (dominated by rollout).

### 4.4 Interpretation

* Rollout is the best of the five policies on every reported metric at both load
  levels; against the strongest baseline (`nearest_job`) the paired gain in average
  flow time is about 4% (default) and 4% (heavy), significant at 95%.
* It is not better on every seed (2/30 and 5/30 seeds worse in flow time). This is
  expected and does not contradict the rollout theory: the cost-improvement
  guarantee needs an exact look-ahead, and here (a) future arrivals are unknown —
  the decision that is best for the parts present can delay a part that arrives
  shortly after — and (b) the look-ahead uses `dt = 0.05` instead of 0.01. The
  guarantee is verified exactly (tests) where its assumptions hold.

## 5. Limitations and recommended engine changes (not implemented)

* **Engine coupling.** The forced first decision relies on steps 1–4 of
  `Simulation.step` being idempotent when re-run within the same step, and the
  `dt` change relies on overriding the `time` property. Both are tested, but a
  public API would be more robust; recommended: split `step()` into
  `_pre_dispatch()` / `_post_dispatch()` (or add `Simulation.fork(policy,
  keep_future=False)`) and store the clock as `t0 + (step_index-k0)*dt`.
* **Copy cost.** `deepcopy` copies every part ever generated (including exited
  ones) and the laser history; an engine-level `fork()` could skip exited parts.
* **Stateful base policies** (e.g. `round_robin`) are restarted from their reset
  state in every clone, because the rollout cannot know their internal state.
* **Speed.** The look-ahead runs until all known parts are processed; under heavy
  load (long queues) this is ~10x more steps per decision than at default load.
  `horizon_s` and `max_candidates` trade accuracy for time.
* `run_simulation.py` runs every available policy for the comparison view, so it
  now also runs rollout (a few seconds on the default deterministic config).
