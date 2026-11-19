# Changelog and decision log

Every change of the model, the simulator or the methods is recorded here with **what**
changed, **why**, and **where** (files, tests, documents). Earlier versions are
summarized from the git history.

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
