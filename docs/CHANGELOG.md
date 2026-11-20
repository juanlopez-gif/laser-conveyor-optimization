# Changelog and decision log

Every change of the model, the simulator or the methods is recorded here with **what**
changed, **why**, and **where** (files, tests, documents). Earlier versions are
summarized from the git history.

---

## 0.7.0 — 2026-11-20 — Strong-RL plan, phase 1: event-driven fast simulator; tie-break fix

The user asked to start the strong-RL plan (`docs/plans/strong_rl_plan.md`). This entry
covers phase 1. The plan now has a status section (§0) with the facts learned since it
was written:

* the real cell runs with `none`;
* the reference results are those of `docs/methods/confirmatory.md`;
* already-used seeds, and fresh seeds 7000–7029 for the final evaluation;
* hardware: Ryzen 7 5700U, 8 cores / 16 threads, 15 GB, no CUDA GPU, hence model size S
  and up to 8 workers.

### Fast simulator (`src/fastsim/`, `docs/methods/fastsim.md`)

* **What it is:** an event-driven simulator with the same model and per-instant rules
  as the engine. Crossing times are computed in closed form, and interception uses the
  same function as the engine.
  * Decision API: `run_until_decision`, `candidates`, `apply`, `clone` (about 10 µs),
    `cost_so_far`, `result()` for the checker.
  * Rules: `nearest_job`, `nearest_head`.
  * Scope: `block` mode only.
* **Why:** the strong-RL plan needs millions of decisions and cheap state copies for
  search.
* **Speed:** 5 150–5 770 decisions/s on one core:
  * 20.6× (real load) and 16.8× (heavy) faster than the engine at `dt = 0.05`;
  * 73–81× faster than the engine at `dt = 0.01`.
  Using 8 workers instead of 2 adds about 4×.
* **Agreement** (`experiments/fastsim_validation.py`, 2400 + 240 runs):
  * 0 formal-checker violations at tolerance 1e-6;
  * against the engine at `dt = 0.001`: identical order in 95.8 % of runs, mean
    |KPI diff| 0.09 %;
  * against the engine at `dt = 0.01`: 84.7 % and 0.40 %.
* **Acceptance criterion revised:**
  * The plan's target (≥ 99 % identical order vs `dt = 0.01`) is unreachable by
    construction: the engine agrees with *itself* across `dt = 0.01` / `0.001` in only
    90.8 % of runs.
  * The residual differences were traced to exact continuous-time coincidences (equal
    travel times, simultaneous events). The engine breaks these with an O(v·dt) bias.
  * Mean KPI bias is below 0.1 %.
  * Details in `docs/methods/fastsim.md` §4.
* **Bug found by the new tests:** processing a part upstream of a blocking part
  released the block (T_B undercounted). It was fixed before any result was produced.
* Tests: `tests/test_fastsim.py` (21). Benchmark: `experiments/fastsim_benchmark.py`.

### Tie-break fix in the engine (changes results)

* **The problem:** exact ties between candidates are frequent:
  * equal Y moves between belts (1.2 s);
  * several parts stopped at b.
  The interception root carried ~1e-16 s of round-off, and positions at b were
  45 − 1e-15. That noise, not the documented rule (earliest arrival, then id), decided
  these ties.
* **Fix:**
  * `simulation/motion.py`: `find_intercept` polishes the root to the fixed point
    t = τ(x(t)), which returns the dominant constant travel time bit for bit;
  * `policies/base_policy.py`: new `DecisionContext.travel_key` (travel time rounded to
    1e-9 s);
  * `nearest_job`, `nearest_head` and the candidate ranking of `rollout` now use it.
* **Effect,** measured on seeds 3000–3029 against the previous commit:
  * `nearest_job` / `none` at the real load changes its order on 26 of 30 seeds and
    improves by 2.5 %;
  * smaller effects elsewhere (`docs/methods/fastsim.md` §3).
* **Consequence:**
  * The reference of every Δ % up to 0.6.2 was about 2.5 % weaker than the documented
    rule at the real load.
  * The confirmatory comparison was rerun (tag `v0.7_confirmatory_ties`,
    `docs/methods/confirmatory.md` §7).
  * At the real load with `none`, the gains are now `rl_rollout` −2.3 ±1.5 % and
    `rollout` −1.9 ±1.3 % (both significant; they were −4.4 % and −4.0 %).
  * At heavy load: `rl_rollout` −3.4 %.
  * The recommendation (`rl_rollout` with `none`) is unchanged.
  * Earlier result pages keep their numbers as historical record, with pointers to §7.
* All 192 tests pass with the fix.

---

## 0.6.2 — 2026-11-20 — The real cell cannot pre-position; agent check for `none`

* **What:** the owner confirmed that the real cell cannot detect parts before the laser
  zone for now.
* **Consequences:**
  * `laser.prepositioning: none` stays the default and is now the confirmed mode of the
    real cell (`docs/assumptions.md` C4).
  * `when_idle` and `always` remain what-if studies of a possible entry sensor. On
    fresh seeds it would be worth −7 to −9 % at the real load.
* **Recommendation for the real cell** (`docs/methods/confirmatory.md` §5): `rl_rollout`
  with `none`, at −4.4 % (real load) and −3.7 % (heavy load) vs `nearest_job`, both
  significant. The plain rollout (−4.0 % / −2.1 %) is the fallback without RL
  dependencies.
* **Agent check** (`docs/methods/confirmatory.md` §5.1):
  * Question: since the real cell runs with `none`, is `rl_rollout` better with the
    agent trained with `none` (`rl06_none_s0`) than with the default `rl06_wi_s0`?
  * Run: 120 runs on seeds 3000–3029.
  * Answer: no significant difference (−0.5 ±0.9 % at the real load, +0.6 ±1.1 % at
    heavy load).
  * Decision: the recommendation keeps the default model, because one model for every
    mode is simpler.
* **`experiments/decompose_effects.py`:**
  * `--merge` combines runs of several tags on the same seeds, and `--out` names the
    output.
  * Model paths are shortened to the model name in the tables, so two agents of the
    same policy can be told apart.
  * Reason: comparing the new runs with the confirmatory ones without rerunning them.
* No model, configuration or simulator code changed.

---

## 0.6.1 — 2026-11-20 — Confirmatory evaluation, effect decomposition, licence, real cell values

### Confirmatory evaluation on fresh seeds (`docs/methods/confirmatory.md`)

* **What was done:**
  * 6 policies × pre-positioning `none`/`when_idle` × 2 loads × seeds 3000–3029, for
    720 runs.
  * The design was fixed before running.
* **Why:** it applies audit points 1 and 2, which were accepted in 0.6.0. The seeds
  1000–1029 had already influenced several choices, and the headline numbers mixed the
  policy effect with the pre-positioning effect.
* **New script `experiments/decompose_effects.py`.** It reports three effects:
  * total: vs `nearest_job`/`none`;
  * policy: vs `nearest_job` at the same pre-positioning;
  * pre-positioning: the same policy, `when_idle` vs `none`.

  It also reports head-to-head pairs. It was checked on the old `rollout_objective`
  data: the rollout's own effect there is −5.3 %, consistent with the audit's −5.4 %.
* **Confirmed:**
  * `rl_rollout` is the best policy at both loads: total −13.1 % default and −4.0 %
    heavy.
  * Its own policy effect is significant at both loads (−4.7 % and −4.0 % with
    `when_idle`).
  * At heavy load the agent beats `nearest_head` (−2.7 %).
* **Not confirmed (exploratory 0.6.0 claims withdrawn):**
  * `rl_rollout` beats the plain rollout only at heavy load (−2.9 %). At default load
    the difference is not significant (−1.1 ±1.3 %).
  * The ensemble does not improve on one agent.
  * The agent's −6.2 % at default load was almost entirely pre-positioning: its own
    effect is −0.2 ±1.9 %.
* **New finding:** at the real (default) load, pre-positioning (`when_idle`) gives −7
  to −9 % with any policy, including `nearest_job`. It is the most valuable single
  change for the real cell, but it needs part detection before the laser zone
  (assumption C4, still unconfirmed). At heavy load it has no effect.
* Pointers to the confirmatory page were added to `hybrid.md`, `rl.md` §6.4,
  `sensitivity.md` and `README.md`. The exploratory tables were kept unchanged, as the
  record of what was seen on seeds 1000–1029.

### Licence

* **What:** MIT (`LICENSE`, README section). It covers code, documentation, trained
  models and configuration.
* **Why:** the owner delegated the choice. MIT is the shortest permissive licence and
  the usual one for academic research code. It lets others reuse the simulator, which
  helps citations. The code has no patent-sensitive parts, so the patent clause of
  Apache-2.0 adds nothing here. CC-BY is meant for text and media, not code.

### Cell parameters are the real ones

* On 2026-11-20 the owner stated that the data we have are the real values: belt
  3 cm/s, laser 10 cm/s, processing 2.5 s, arrivals 10/12/14 s.
* This is recorded in `docs/assumptions.md` §F.
* The "measure the real cell" item is closed. The heavy load stays a stress scenario.

---

## 0.6.0 — 2026-11-20 — DAgger agents, hybrids, sensitivity, external audit fixes

Closed on 2026-11-20 after the sensitivity sweep finished. New method pages:
`docs/methods/hybrid.md`, `docs/methods/sensitivity.md`; `docs/methods/rl.md` §6.
Merged paired summary: `results/experiments/v0.6_summary/summary.md`.
`README.md` lists the hybrids and links the new pages.

**Main results** (30 seeds, paired Δ of penalized flow vs `nearest_job`/`none`):

| policy | default load | heavy load |
|---|---|---|
| `rl_rollout` (rollout with the agent as base policy) | **−11.4 %** (−13.0, −9.8) | −3.7 % (−5.4, −2.1) |
| `rl_ensemble` (vote of 3 agents) | −7.0 % | **−4.0 %** (−5.4, −2.5) |
| best rollout of 0.5 (`rollout@avg_flow_time`, `when_idle`) | −10.1 % | −2.0 % |
| `rl` alone (`rl06_wi_s0`, new default) | −6.2 % | −3.0 % |
| `nearest_head` / `when_idle` | −6.4 % | −0.1 % |

These use the evaluation seeds 1000–1029, which earlier choices have already seen;
the confirmatory rerun on fresh seeds 3000+ is pending (see the audit section).

### User decisions (2026-11-19)

* The real cell uses a Longer Ray5 5 W laser, but the model **keeps laser 10 cm/s, no
  acceleration and processing 2.5 s**. The user's reason: acceleration complicates the
  model too much.
* Belt speed and job time are adjustable in the real cell, so they are studied in the
  sensitivity sweep instead of changing the default.
* A Ray5-specific config was drafted and then removed for the same reason.

### RL training (`src/rl/train.py`, `tests/test_rl.py`)

* **DAgger** (Ross et al. 2011) after behaviour cloning: the learner drives, the teacher
  (`rollout`) labels every visited state, with mixing β_i = β0^i. Reason: plain cloning
  only sees the teacher's state distribution.
* **γ = 1** and reward scale 1000. The return now equals minus the episode objective;
  before it was a discounted proxy (review point E).
* **Checkpoint selection on validation seeds 2000+**, never on the evaluation seeds
  1000–1029. Score = mean ratio of the episode cost to `nearest_head` on the same
  episode. The delivered model is the best checkpoint among BC, DAgger and PPO.
* Parallel episode collection (`ProcessPoolExecutor`).
* Result (validation ratio, lower is better):

  | model | ratio | selected stage |
  |---|---|---|
  | `rl06_wi_s0` | 0.980 | DAgger 3 |
  | `rl06_wi_s1` | 0.985 | DAgger 2 |
  | `rl06_wi_s2` | 0.988 | PPO 15k |
  | `rl06_none_s0` | 0.985 | DAgger 3 |

* **Longer PPO fine-tuning degraded the cloned/DAgger policy in every run.** Validation
  selection kept the degraded checkpoints out of the delivered models. Only
  `rl06_wi_s2` kept its first PPO checkpoint (15k decisions, 0.988 vs 0.994 after
  DAgger), a difference within validation noise. (Corrected on 2026-11-20: the
  selected stages of s1, s2 and `none_s0` were missing from the table above.)
* **Default model of the policy `rl` changed** from `models/rl_bc.zip` to
  `models/rl06_wi_s0.zip` (`src/policies/rl_policy.py`).
  * Reason: it has the best validation score.
  * On the evaluation seeds it improves on `rl_bc` from −1.4 % to −6.2 % at default
    load and from −1.8 % to −3.0 % at heavy load (part of this is pre-positioning
    `when_idle`).
  * `rl06_wi_s1` is slightly better on the evaluation seeds, but it was not chosen,
    so that those seeds are not used for selection.
  * It was trained with `when_idle`; with `none`, use `models/rl06_none_s0.zip`.

### Hybrid policies (`src/policies/hybrid.py`, `tests/test_hybrid.py`)

* Four hybrids:
  * `rl_rollout`: rollout with the RL agent as base policy.
  * `rl_filter`: the RL proposes its top-k conveyors and the rollout decides among them.
  * `rl_gated`: the RL decides when it is confident, the rollout otherwise.
  * `rl_ensemble`: majority vote of several agents.
* The study harness accepts `name@k=v@k2=v2` policy specs and records `decision_ms` and
  `rollout_share`.
* Results (30 seeds, `when_idle`; paired Δ vs `nearest_job`/`none`; full table in
  `docs/methods/hybrid.md` §3):

  | policy | default load | heavy load | time per decision (default / heavy) |
  |---|---|---|---|
  | `rl_rollout` | −11.4 % (−13.0, −9.8) | −3.7 % (−5.4, −2.1) | 101 / 550 ms |
  | `rl_filter` (top 2) | −8.9 % (−11.4, −6.5) | −0.8 % (−2.2, +0.7) | 26 / 172 ms |
  | `rl_gated` (0.8) | −7.6 % (−9.6, −5.5) | −2.9 % (−4.7, −1.0) | 7 / 4 ms |
  | `rl_ensemble` | −7.0 % (−8.6, −5.3) | −4.0 % (−5.4, −2.5) | ms |

* `rl_rollout` improves on the best plain rollout (−10.1 % / −2.0 %). This matches the
  rollout-improvement property: the agent is a better base policy than `nearest_job`.
* `rl_gated` sent only 14 % (default) and 0.3 % (heavy) of the decisions to the
  rollout, so it behaves almost like the agent alone.
* `rl_filter` loses at heavy load: pruning to the agent's top two conveyors too often
  removes the option the rollout would have chosen.

### Sensitivity sweep (`experiments/sensitivity_sweep.py`)

* One-at-a-time sweep around the default cell: belt speed, laser speed, processing time
  and arrival rate.
* Every policy is compared with `nearest_job`/`none` in the same cell, with common random
  numbers.
* 10 seeds per cell (1000–1009), 1360 runs, 3574 s on 2 workers. Full tables and
  caveats: `docs/methods/sensitivity.md`.
* Findings (Δ vs the reference in the same cell):
  * `rl_rollout` is the best or near-best policy in 11 of the 14 distinct cells: every
    belt speed (−6.3 to −11.0 %), every laser speed (−8.3 to −11.9 %), and processing
    times up to 3 s.
  * The plain `rollout` is better when the cell is congested:
    * arrival factor 0.6: −4.8 % vs −0.6 %;
    * arrival factor 0.8: −8.7 % vs −6.6 %;
    * processing 4 s: −4.6 % vs −3.1 %.
  * The agents were trained only in the default cell and do not generalize to much
    higher load: at arrival factor 0.6, `rl` alone gets +2.3 %, worse than the
    reference.
  * The agent alone performs about like `nearest_head` across the sweep.
  * Processing time is the dominant parameter: the reference goes from 11 s (1.5 s
    jobs) to 72 s (4 s jobs). Belt speed barely matters (25–27 s from 1.5 to
    6 cm/s).
  * With 10 seeds the half-widths are ±2 to ±6 points, so the sweep shows trends, not
    rankings.

### Literature and plans

* `docs/literature/closest_papers.md`: full reading of the six closest papers.
  `review_assessment.md` §5 corrects earlier claims about them.
* `docs/plans/strong_rl_plan.md`: working plan for a stronger learned dispatcher:
  * fast event-driven simulator;
  * Transformer over parts with pointer and value heads;
  * expert iteration with a search teacher;
  * domain randomization;
  * S/M/L scaling ladder.

### Fixes after the external audit (2026-11-19)

An independent audit covered the public commit 2a28321 and the local 0.6 copy. Each
finding below was reproduced before fixing and now has a regression test.

| finding | fix | files / tests |
|---|---|---|
| The benchmark used the CP-SAT grid bound as a lower bound of P. P_Δ rounds up and is not a relaxation, so a negative gap could appear next to `proven_optimal`. | `cpsat` returns `lower_bound=None` and keeps the grid certificate in `info["model_bound"]`. New benchmark columns: `objective` (replay), `lower_bound` (rigorous bounds only), `proven_optimal_model`, `model_objective`, `model_bound`. | `solvers/cpsat.py`, `benchmark.py`, `docs/methods/exact_cpsat.md` §4; `test_cpsat.py::test_benchmark_never_reports_negative_gap_to_lower_bound` |
| The schedule checker accepted zero-length processing and truncated runs. | New checks: C_j − S_j = p_j, and completeness (on by default when `simulation.drain` is set; `require_complete=False` for deliberately partial runs). | `schedule_check.py`; `test_formulation.py` (two negative controls) |
| In the MILP penalty, β could be 0 while a part sat at b if processing had already stopped the belt. | New constraint (M15) ties β to the physical state. | `docs/formal_formulation.md`, `docs/formulation/formal_formulation.tex/.pdf` (doc only; CP-SAT does not implement Π) |
| Interception missed an earlier meeting when the belt is faster than the laser's X axis, because the search assumed a decreasing gap. | Exact solution: travel time is piecewise linear in x, so the earliest root is found segment by segment. In every configuration studied the belt is slower than the laser, where old and new agree to ≤ 3·10⁻¹³ s (20 000 random nominal cases). Such round-off can still flip an *exact* tie between two decisions, so a rerun may differ on rare tied decisions; no systematic effect. One test (`test_rollout.py::test_no_future_information_is_used`) relied on round-off to break an exact tie (9.04 s either way) and was rewritten to check the tie and the strict preference explicitly. | `simulation/motion.py`; `test_blocking_prepositioning.py` (audit counterexample + brute-force check), `test_rollout.py` |
| An RL episode with only forced decisions ran entirely inside `reset`, so its cost was never reported. | `reset` returns `summary` and `episode_cost` when the episode ends there, and the collector keeps them. | `rl/env.py`, `rl/train.py`; `test_rl.py::test_decision_free_episode_reports_its_cost` |
| The README said OR-Tools is not used. | Dependency text corrected. | `README.md` |

Also fixed: in the 0.5.2 entry and in `review_assessment.md`, two formulas had `\t`
turned into a tab (`\tau`, `\text`).

**Methodological points accepted from the audit.** They will be applied before the
final tables.

1. The rollout objective and the BC model were chosen after looking at evaluation seeds
   1000–1029. The confirmatory comparison of the final variants will therefore use
   **fresh seeds** (3000+).
2. The −10.1 % of `rollout`/`when_idle` against `nearest_job`/`none` combines two
   effects: the policy itself (−5.4 % at equal pre-positioning, per the audit) and
   pre-positioning. The tables will report the two separately.
3. The repository has no licence yet; choosing one is the owner's decision.

**Chain incident.**

* The step that picked the best `when_idle` agent for the hybrid and sensitivity runs
  had a shell-quoting error and passed an empty model path, so the first hybrid run used
  the old default model.
  * It was stopped and rerun with `rl06_wi_s0`.
  * The hybrid results above come from the rerun.
* Summary printing also failed on Windows, because cp1252 cannot encode "Δ".
  * Runs now set `PYTHONIOENCODING=utf-8`.
  * The result files were not affected.

---

## 0.5.2 — 2026-11-19 — Technical points of the literature review

Closes points A, B, D, E (documentation), G and the blocked-time overlap of
`docs/literature/review_assessment.md`.

* **Restriction gap (point A).** New `experiments/restriction_gap.py` and
  `docs/methods/restriction_gap.md`: exhaustive search over all orders vs belt-order
  orders on small instances. **Belt order is not dominant**: gap 0 % at default load
  (8/8 optima keep belt order) but up to 3.5 % (mean 0.3–1.5 %) with bursty arrivals.
  Head-only policies (`nearest_head`, RL action space) are a restricted class.
  Counterexample fixed in `tests/test_restriction_gap.py`. Formulation §5 property 2
  and the PDF (new remark) corrected ("causes blocking" → "tends to cause blocking;
  not dominant").
* **NP-hardness (point B).** The sketch based on limiting cases (infinite speeds,
  infinite zone) was replaced by a realizable reduction from $1|r_j|L_{\max}$ and
  $1|r_j|\sum C_j$ (time scaling by $M>n^2\tau_{\max}$, one part per conveyor). Scope
  stated: stop rule $\sigma=I$, number of conveyors part of the input. Open: fixed
  $N$ (e.g. 3) and `all_conveyors`. The makespan/TSP-path remark is no longer claimed.
* **Notation (point D).** Classification rewritten as
  $1|r_j, s_{ij}(t,z), \text{blocking}, \text{endogenous windows}|\ldots$: travel and
  accessibility windows depend on the belt state, i.e. on the schedule.
* **Pre-positioning semantics (point G).** §1.5: decision epochs, candidate sets per
  mode, irrevocable vs revocable commitments, what happens on revocation, `dt`
  granularity of the simulator.
* **Blocked-time attribution.** §1.4: $T_B$ is attributed downtime; overlapping
  processing and blocking stops count in both per-reason times (physical downtime
  counts once).
* **RL discount (point E).** `docs/methods/rl.md`: the return equals the objective only
  undiscounted; PPO was trained with γ = 0.99 (a discounted proxy).
* No change to the simulator, policies or default parameters.

---

## 0.5.1 — 2026-11-19 — Literature evidence map (documentation only)

* `docs/literature/systematic_review_2026-11-19.md`: PRISMA-informed evidence map
  produced by an external AI tool from the project's review prompt (21 studies;
  access-limited, not a publishable SLR).
* `docs/literature/references.bib`: its 26 references; **all DOIs / arXiv ids verified
  to exist** (Crossref / arXiv, 2026-11-19); S01 DOI added.
* `docs/literature/review_assessment.md`: verification report, point-by-point check of
  the review's technical remarks against our model and code, and an action list.
  Main open points: restriction gap of head-only policies; scope of the NP-hardness
  reduction (general N, own-belt stop rule); endogenous windows $s_{ij}(t,z)$; PPO
  trained with γ = 0.99 (return = objective only for γ = 1); formal semantics of
  revocable pre-positioning and of blocked-time attribution.
* No change to model, code or results.

---

## 0.5.0 — 2026-11-19 — Reinforcement-learning dispatcher

Requested by the cell owner: formulate the dispatching problem as RL and try it.
Details and all numbers: [methods/rl.md](methods/rl.md).

**What.**
* Engine: `Simulation.step()` split into `begin_step()` / `decision_candidates()` /
  `end_step(choice)` (bit-identical to the old `step()`, tested), plus two counters:
  `wip_area` (integral of unfinished parts) and `n_block_events` (counted at S_j).
* `src/rl/`: SMDP environment (`env.py`, Gymnasium), observation/action/mask
  (`features.py`), training by behaviour cloning from `rollout` + MaskablePPO
  (`train.py`, `--n-envs` for parallel simulators). Policy `rl`
  (`src/policies/rl_policy.py`, parametric, loads `models/rl_bc.zip`).
* Trained models and training log in `models/` (versioned, < 1 MB).
* Optional dependencies in `requirements-rl.txt` (gymnasium, torch, stable-baselines3,
  sb3-contrib); the tests skip the RL tests if they are missing.
* Formulation §4 (MD + PDF): SMDP, cost between epochs, return = −n·penalized flow.
* Study script: `name@key=value` policy parameters (e.g. `rl@model=...`) and
  `--summarize` to merge runs with paired 95 % CIs; duplicate cells are dropped.

**Why these design choices.** Head-only actions (structural result of phase 2, small
fixed action space); reward = work-in-progress integral + blocking penalty so that the
return equals the default objective exactly; forced decisions hidden; imitation of the
best online policy (rollout) because PPO from scratch would need far more simulation.

**Results (30 seeds, paired vs nearest_job, penalized flow).** Behaviour cloning:
−1.4 % default (not significant), −1.8 % heavy (significant) — about half of the
teacher's gain (−4.2 % / −3.7 %) at negligible decision cost. PPO fine-tuning (150k
decisions) **degraded** it (+1.9 % / −1.0 %). Hence the default model is the cloned
one. At default load the zero-cost rule `nearest_head` (−2.6 %) is better.

**Hardware.** The bottleneck is the simulator (CPU), not the network: more CPU cores
help, GPUs do not.

**Not done (open):** DAgger, larger PPO budget, training with `when_idle`
pre-positioning, calibration with real cell data.

---

## 0.4.0 — 2026-11-19 — Blocking penalty, `nearest_head`, laser pre-positioning

Requested by the cell owner: (1) penalize explicitly that a part reaches the end of the
laser zone unprocessed and stops the whole conveyor; (2) implement the belt-order rule
suggested by the metaheuristics; (3) let the laser move towards a part before it enters
the laser zone; (4) update the formulation and document everything.

### 1. Blocking penalty (model + metrics + objectives)

**What.** New quantities per part and per run:

| symbol | meaning | code |
|---|---|---|
| $B_j$ | time part $j$ reaches $x=b$ unprocessed (stops its belt) | `Part.block_time` |
| $\beta_j=\mathbb 1[S_j>B_j]$ | part $j$ blocked its conveyor | `Part.caused_block`, `Part.blocked_duration` |
| $N_B=\sum_j\beta_j$ | blocking events | summary `blocking_events` |
| $T_B=\sum_j(S_j-B_j)^+$ | conveyor time stopped by blocking [conveyor-s] | summary `total_blocked_time` |
| $\Pi=\lambda_N N_B+\lambda_T T_B$ | blocking penalty [s-equivalent] | summary `blocking_penalty` |

New objectives (`optimization/objective_functions.py`): `blocking_events`,
`total_blocked_time`, `blocking_penalty`, `penalized_flow_time`
$=\overline F+\Pi/n$ and `penalized_makespan` $=C_{\max}+\Pi$. Configuration:
`objective.blocking_penalty.per_event` ($\lambda_N$, default 10 s) and `per_second`
($\lambda_T$, default 1 s per blocked conveyor-second).

**Why.** The time objectives already contain the delay caused by blocking, but a
stopped conveyor has costs that they do not see (restart, accumulation upstream, risk
for the upstream process), and the cell owner considers it important to avoid it.
Two terms are used because both the *frequency* (each stop is a disturbance) and the
*duration* (how long the belt stays stopped) matter. The penalty is added per part
($\Pi/n$) to the mean flow time so that both are in seconds per part.

**Decisions and assumptions.**
* **The weights are placeholders** ($\lambda_N=10$, $\lambda_T=1$): one blocked
  conveyor-second costs as much as one second of flow time, plus a fixed 10 s per event.
  They must be calibrated with the real cost of a stopped conveyor. Every experiment
  prints the weights it used.
* A part reaching $b$ exactly when the laser starts on it is **not** an event
  ($S_j=B_j$; the belt is stopped by processing anyway).
* The block ends at $S_j$: from then on the belt is held by processing. The engine now
  releases the blocking stop reason at the start of processing (`_start_processing`),
  so that $T_B$ equals $\sum_j(S_j-B_j)$ exactly (before, one extra step `dt` per event
  was counted as blocked). This does not change any trajectory: the belt was stopped
  by processing in that step anyway.
* **Default objective** changed from `makespan` to `penalized_flow_time`
  (`objective.weights`): at stable load the makespan is almost fixed by the last
  arrival and does not see blocking.
* `rollout`: the blocking objectives are accepted as look-ahead objectives. Its default
  look-ahead objective was first switched to `penalized_flow_time` and then **reverted
  to `avg_flow_time`** after the 30-seed study (§4 below): the penalized look-ahead was
  worse even on the penalized metric.
* Lower bounds: for the penalized objectives the bound of the unpenalized objective is
  used ($\Pi\ge0$), which is valid but ignores the penalty.
* CP-SAT: the linear form of the penalty is written in the formulation (§2), but it is
  **not implemented** in `solvers/cpsat.py`; for penalized objectives CP-SAT is skipped.

**Where.** `config/default_config.yaml`, `simulation/config.py`, `simulation/part.py`,
`simulation/engine.py` (`_update_blocking`, `_start_processing`, `result`),
`simulation/metrics.py`, `optimization/objective_functions.py`,
`optimization/lower_bounds.py`, `policies/rollout.py`, `optimization/solvers/online.py`,
`experiments/run_experiment.py` (new KPIs), visualization dashboard (blocking events and
blocked time). Formulation: `docs/formal_formulation.md` §1.4 and §2, PDF §2.4 and §3.3.

### 2. Policy `nearest_head`

**What.** On every conveyor only the *head* (most downstream unstarted part) is
eligible; among the heads the one with the smallest interception time is chosen.

**Why.** The metaheuristics agent found that good sequences keep belt order inside
each conveyor and only interleave conveyors; `nearest_job` breaks it by serving a later
part first, which lets the earlier part reach $b$ and block (property 2 of §5 of the
formulation). The rule is the simplest online policy with that structure, and it costs
no computation.

**Properties (tested).** Never idles when a candidate exists (the head of a conveyor
enters the zone before any other part of that conveyor); its schedules keep belt order
on every conveyor; feasible for the formal model in every pre-positioning mode.

**Where.** `src/policies/nearest_head.py`, registered as method `policy:nearest_head` in
`optimization/solvers/baselines.py`; tests in `tests/test_blocking_prepositioning.py`;
results in `docs/methods/nearest_head_prepositioning.md`.

### 3. Laser pre-positioning

**What.** `laser.prepositioning: none | when_idle | always` (YAML `false`/`true` are
accepted as `none`/`always`).

* `none` (default): only READY parts are candidates — model $\mathcal P^{\rm sim}$
  (constraint (C7)), behaviour of 0.1–0.3.
* `always`: parts in the entry zone of a running belt are candidates too; the laser goes
  to the meeting point at $x\ge a$ and waits — model $\mathcal P$.
* `when_idle`: entry-zone parts are candidates only when no part is READY; if a part
  becomes READY while the laser is heading to / waiting for an entry-zone part, the
  pre-positioning is revoked (event `UNASSIGN … revoked`) and the policy decides again.

`motion.find_intercept` got an argument `x_min` (start of the laser zone): for a part
upstream of it the meeting point is searched only from the instant it enters the zone;
an upstream part on a stopped belt is unreachable for now.

**Why.** The integration of phase 2 showed that the formulation $\mathcal P$ allows
pre-positioning and the simulator did not; the value for an isolated part is
7.19 − 5.83 = 1.36 s. Implementing it closes the gap between model and simulator
(`always` + `fixed_sequence` is now the decoder of $\mathcal P$) and lets us measure its
value online.

**Why three modes.** A first implementation (`always`) made the myopic policies
*worse* (single run, default load: `nearest_job` mean flow 15.4 → 16.5 s): the laser
committed to a part that had not arrived yet while a READY part elsewhere travelled on
towards $b$ and blocked. Pre-positioning is only harmless when the laser would
otherwise be idle, and the commitment must be revocable; hence `when_idle`.

**Default kept at `none`.** Pre-positioning needs to know that a part is on the entry
zone (an entry sensor). This is a process question that has **not been confirmed** for
the real cell, so the confirmed behaviour stays the default.

**Where.** `config/default_config.yaml` (`laser.prepositioning`),
`simulation/config.py` (`PREPOSITIONING_MODES`), `simulation/motion.py`,
`simulation/engine.py` (`_dispatch`, `_drive_laser`, `_revoke_prepositioning`),
`run_simulation.py --viz-prepositioning MODE`, tests in
`tests/test_blocking_prepositioning.py`; formulation §1.5 (table of modes), §4, §6.

### 4. Results (30 seeds, paired) and the decisions they triggered

Full tables: [methods/nearest_head_prepositioning.md](methods/nearest_head_prepositioning.md).
Penalized flow time, paired relative difference vs `nearest_job` without
pre-positioning (95 % CI):

| | default load | heavy load |
|---|---|---|
| nearest_head, none | −2.6 % (−4.4, −0.7), 22/30 | +0.5 % (−0.9, +1.9) |
| nearest_head, when_idle | **−6.4 % (−9.0, −3.9), 29/30** | −0.1 % (−1.5, +1.3) |
| nearest_head, always | +4.6 % (+0.1, +9.1) | −1.8 % (−3.6, +0.0) |
| rollout (flow look-ahead), none | −4.2 % (−5.4, −3.0) | **−3.7 % (−5.1, −2.4)** |
| rollout (flow look-ahead), when_idle | **−10.1 % (−11.6, −8.5), 30/30** | −2.0 % (−3.5, −0.5) |
| rollout (penalized look-ahead), none | −4.5 % (−7.1, −1.9) | +2.5 % (+0.3, +4.8) |

Decisions:
* **rollout default objective reverted to `avg_flow_time`** (penalized look-ahead worse
  at heavy load even on the penalized metric; it over-weights the blocks it can see).
  `tests/test_rollout.py` checks the guarantee on `avg_flow_time` again, and on the
  penalized objectives when they are requested explicitly.
* `laser.prepositioning` default kept at `none` (sensor not confirmed); `when_idle` is
  the recommended mode if the cell can detect parts in the entry zone, `always` only
  for saturated operation.
* `nearest_head` is the recommended zero-cost rule; rollout (flow) the best online
  policy when ~10–50 s of computation per 300 s of operation is acceptable.

### 5. Documentation and tests

* Formulation (MD + LaTeX/PDF, now 10 pages): blocking penalty (§1.4), pre-positioning
  modes (§1.5), linear form of the penalty in $\mathcal P_\Delta$ (§2), `nearest_head`
  and pre-positioning in the online section (§4), correspondence table (§6).
* `docs/assumptions.md` C4 (pre-positioning), E4 (blocking penalty), E5 (default
  objective); README (defaults, policies, metrics); `docs/methods/online_rollout.md`
  (new default objective); `docs/methods/theory.md`, `docs/methods/exact_cpsat.md`,
  `docs/future_optimization.md`, `docs/agent_architecture.md` (items resolved).
* New study script `experiments/blocking_prepositioning_study.py` and results document
  `docs/methods/nearest_head_prepositioning.md`.
* Tests: new `tests/test_blocking_prepositioning.py` (27 tests: identity
  $T_B=\sum(S_j-B_j)$, penalty formula, `nearest_head` belt order and no idling,
  interception of upstream parts, value of pre-positioning on counterexample D1,
  feasibility of all policies in every mode, revocation invariant of `when_idle`,
  `always` decoder vs (C7) decoder); `tests/test_rollout.py` adapted to the new default
  objective. Full suite after 0.5.0: 156 tests pass.

---

## 0.3.0 — 2026-11-19 — Optimization agents (phase 2)

Foundation (instances, decoder `fixed_sequence`, `evaluate`, `schedule_check`,
benchmark) and four specialist methods: CP-SAT (`cpsat`, `cpsat_ready`), SA/ILS,
proved lower bounds L1–L6, rollout policy. Formulation corrected with (C7) /
$\mathcal P^{\rm sim}$. Details: `docs/agent_architecture.md`.

## 0.2.0 — 2026-11-18 — Formal formulation

Exact continuous-time model $\mathcal P$, time-indexed MILP $\mathcal P_\Delta$, decoder,
online version, complexity; schedule checker; LaTeX/PDF.

## 0.1.x — 2026-11-18 — Simulator V1

Fixed-step hybrid engine, four policies, metrics, HTML visualization, experiments.
Lighter default load (ρ ≈ 0.64), policy selector, English docs; blocking at the zone end
confirmed for the real cell.
