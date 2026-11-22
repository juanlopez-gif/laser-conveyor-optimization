# Changelog and decision log

Every change of the model, the simulator or the methods is recorded here with **what**
changed, **why**, and **where** (files, tests, documents). Earlier versions are
summarized from the git history.

---

## 0.7.11 — 2026-11-22 — Bursty arrivals and slow heads in the training mix; validation group `real_burst`; run `rl2_gen_s_b`

* **Why.** After the information pilot (0.7.10) the owner decided that bursty
  arrivals and slow heads are part of the scope to be defended, and to retrain the S
  network once, keeping `rl2_gen_s_d` as the reference. Up to 0.7.10 every training
  cell had Poisson arrivals and laser speeds in 5–20 cm/s, and the network did not
  beat `nearest_job` on the heavy cell with a 2.5 cm/s head.
* **What** (`rl2.expert_iteration`, `docs/methods/rl2.md` §6.1): `GENERAL` gains
  `p_burst 0.5`, `burst_on_fraction (0.2, 1)`, `burst_on_mean (10, 40) s`, `p_slow
  0.3`, `slow_laser_speed (2.5, 5) cm/s`. `draw_regime(seed)` draws them from a
  separate random stream (`rl2regime/<seed>`), so every other parameter of a cell
  with a given seed is unchanged and the general validation cells (no draw) are the
  same cells as in 0.7.9 (a test checks both). General training cells may get
  bursts and/or a slow head; the real and heavy training cells bursts only (their
  head is the cell's). The kind string of an episode becomes
  `group/mode/observation/regime`. The previous `laser_speed` range of the main draw
  is untouched (widening it would have changed every existing cell).
* **Validation:** fourth group `real_burst` (the real cell with `on_fraction 0.3`,
  `on_mean 20 s`, the setting of the information bench, now `BURST_VAL`; seeds
  2000+); the score is the mean of the four groups, so it is not comparable with the
  three-group scores of earlier runs (the per-group deltas are).
* **Not changed:** the tokens (the network is not told the burst parameters or the
  regime; it keeps the mean rates and must infer the burstiness), so `rl2_gen_s_d`
  can initialize the new run and remains loadable; the teacher's sampled futures
  (Poisson at the mean rates); the validation seeds; the cell model.
* **Run `rl2_gen_s_b`** launched on the workstation: size S from `rl2_gen_s_d.pt`,
  sampled-future teacher (8 futures, 30 s, w_pen 0), `--val-mode design`, 50
  validation seeds per group, up to 8 iterations, patience 2, 2 000 episodes per
  iteration. Results: `docs/methods/rl2.md` §8.8.
* **Tests:** `tests/test_rl2_search.py` gains the regime test (mix fractions,
  ranges, separate stream, validation cells unchanged, `real_burst` group); the kind
  split updated. Full suite passes.
* **`docs/formulation/formal_formulation.pdf` rebuilt** (13 pages) from the `.tex`
  with the 0.7.7–0.7.10 corrections (information set with the visible length, the
  bursty arrival process, `when_idle` as the design mode): TeX Live was installed on
  the workstation at the owner's request (`texlive-latex-extra`,
  `texlive-fonts-recommended`, `texlive-pictures`, `latexmk`); the PDF had been out of
  date since 0.7.7 because no LaTeX was available here.

---

## 0.7.10 — 2026-11-22 — Bursty arrivals and the visible length of the entry zone; information bench

* **Why.** The first A/B/C/D measurement (0.7.9) showed that, on the nominal cell,
  seeing the entry zone without acting on it is worth nothing: the value is in the
  action (pre-positioning). The main hypothesis of Paper 2 (`docs/plans/papers.md`,
  upstream visibility is worth more under bursty arrivals and long head travel) could
  not be tested because the generators produced only deterministic or Poisson
  arrivals, and treatment B ("a preceding stretch of defined length") had no length
  parameter: `belts` always saw the whole entry zone. Both were listed as missing in
  the plan's review notes (points 3 and 5). The agreed order was: build these two
  pieces first, run a cheap pilot with the existing network, and only then decide on
  retraining and on the phase-4 confirmatory evaluation, which should run once.
* **Bursty arrivals** (`arrivals.stochastic.distribution: burst`,
  `src/simulation/arrivals.py`): an on/off Poisson source (interrupted Poisson
  process) with **the same mean rate** as the Poisson process. Two parameters per
  conveyor: `burst.on_fraction` (fraction of time the source emits; 1 = Poisson) and
  `burst.on_mean` (mean length of an emitting period, s). While on, parts arrive as a
  Poisson process of rate `1 / (mean_interarrival · on_fraction)`; the off-periods
  have mean `on_mean · (1 − on_fraction) / on_fraction`, so the long-run rate is
  exactly `1 / mean_interarrival`. The inter-arrival times are hyper-exponential, which
  is the model the review notes asked for. The switching uses its own random stream
  (`<seed>/bursts/<k>`), so every `exponential` instance is unchanged (a test checks
  that `on_fraction: 1` reproduces the Poisson stream exactly). Checked numerically:
  with `on_fraction 0.3`, `on_mean 20 s` the rate is preserved within 2 % over
  20 000 s and the index of dispersion of 10-s counts rises from 1.0 to 2.3–2.9.
  The controller's statistical knowledge stays the mean rates: the sampled-future
  teacher and `rollout` still draw Poisson futures (`rl2.search.sample_scenarios`),
  and the context token does not carry the burst parameters. This is deliberate for
  the pilot (the network is unchanged and must stay loadable); whether the burst
  parameters become part of θ in the training mix is a decision for the owner, since
  it changes `GENERAL` and needs a retraining.
* **Visible length** (`simulation.visible_length`, cm; `null` = the whole entry zone,
  the behaviour of 0.7.3–0.7.9): at level `belts` the camera sees the laser zone and
  only the stretch `[a − visible_length, a)` of the entry zone. A MOVING part upstream
  of it is **not observable**: it is not a token of the network, not a candidate for
  pre-positioning (engine and fast simulator), it is dropped from the look-ahead
  clones, and its time on the belt is counted from the instant it becomes visible
  (`rl2.features.observed_since` now uses that instant, `FastSim.pvis`). `0` makes
  `belts` see exactly what `zone` sees, which is a check. The fast simulator gained
  the crossing of `x = a − visible_length` as an event, so a part becoming visible is
  a decision opportunity, as entering the belt was. Ignored by `zone` and `full`.
  `simulation/observation.py`: `resolve_visible_length`, `visible_moving`.
* **No change to results.** With the defaults (whole entry zone visible, Poisson
  arrivals) the fast simulator gives bit-identical costs and blocking counts to the
  0.7.9 code on 48 real/heavy cases at every observation level and mode; the engine
  and the fast simulator agree under a visible length (same processing order, start
  times within 0.05 s, two new agreement cases).
* **Information bench** (`experiments/rl2_information_bench.py`): treatments A/B/C/D
  and the visible-length curve (0, 2.5, 5, 10 cm, with `none` and with `when_idle`)
  for `nearest_job`, `nearest_head` and one network, across load (real, heavy),
  arrivals (regular, burst 0.3/20 s) and head travel (nominal, laser speeds × 0.5), 30
  validation seeds per regime, paired. Output: `results/experiments/<tag>/bench.json`
  and `tables.md`. Not changed: `GENERAL`, the seeds, the cell model.
* **Tests:** `tests/test_visible_length_burst.py` (12): config resolution and
  validation; mean rate, over-dispersion, ordering and reproducibility of the bursty
  stream; visibility follows the visible stretch in the fast simulator (both modes)
  and in the engine (candidates); `visible_length: 0` equals the `zone` view; tokens
  and the search teacher ignore a part upstream of the stretch while the whole-zone
  camera sees the same change; engine/fast-simulator agreement. Full suite passes.
* **Docs:** `config/default_config.yaml` (keys and comments), `docs/assumptions.md`
  C4b and D1, `docs/formal_formulation.md` §4.1–4.2 (and the `.tex`, where the
  design mode is also corrected to `when_idle`), `docs/system_description.md`,
  `docs/methods/fastsim.md`, `docs/plans/papers.md` (status table, load grid),
  README §8, `docs/methods/rl2.md` §8.7 (pilot results).
* **Pilot results** (`docs/methods/rl2.md` §8.7, 8 regimes × 30 seeds, network of
  0.7.9 unchanged): seeing the entry zone without acting on it is worth nothing in
  every regime (B vs A within ±0.9 points), also with bursts and a slow head, so the
  main hypothesis of Paper 2 as written is not supported at this burst setting; the
  value is in the action (C vs B −5.7 points on the nominal real cell, −3.0 with a
  slow head, −3.1 with bursts, −4.0 with both, 0 at the heavy load), and the curve
  against the visible length saturates at about half the entry zone (5 cm, 1.7 s of
  belt travel: −12.8 of −13.3 points). Seeing the hidden upstream queue is worth 4.4
  points only at the heavy load with a slow head, a regime outside the network's
  training range (laser 2.5 cm/s). Trained on Poisson arrivals only, the network keeps
  3–5 points over `nearest_job` under bursts.
* **Gain of pre-positioning per visible length** (owner's request, same data): the
  paired contrast `when_idle` vs `none` at the same ℓ_v shows 97 % of the gain at 5 cm
  (1.7 s of anticipation) with the nominal head and 53 % with a head half as fast,
  i.e. the coverage needed grows with the head's travel time (`rl2.md` §8.7).
* **Decisions (owner, 2026-11-22, `docs/plans/papers.md`):** the Paper 2 hypothesis
  is reformulated around the coverage curve (how much of the entry stretch the camera
  must see for advancing the head to pay off), with B vs A kept as the pure-information
  contrast expected to be a null result; the S network is retrained once only if
  bursts and slow heads are in the defended scope, keeping `rl2_gen_s_d` as the
  reference; phase 4 runs once, after network, regimes and contrasts are frozen.
  Whether the burst parameters and slower heads enter `GENERAL` is still open.

---

## 0.7.9 — 2026-11-22 — Observation levels in the training mix; first round in the design mode

* **Why.** The research plan (0.7.8, `docs/plans/papers.md`) compares three
  information treatments: the laser zone only (A), the whole belt (B) and the whole
  belt with pre-positioning (C). One network that receives the observation level in
  its context token can serve the three if it has seen them in training; up to 0.7.8
  every training cell used `belts`.
* **What.** `rl2.expert_iteration.GENERAL["p_observation"]` draws the observation
  level of every training cell (`belts` 70 %, `zone` 15 %, `full` 15 %; `zone`
  forces pre-positioning `none`, since it cannot see the entry zone). The draw uses
  its own random stream (`draw_observation`), so every other parameter of a cell is
  unchanged for a given seed, and the validation cells (no draw) stay `belts`: a test
  checks both. The kind string of an episode becomes `group/mode/observation`.
* **Run `rl2_gen_s_d`:** continuation of `rl2_gen_s_sf.pt` with the sampled-future
  teacher, the corrected tokens of 0.7.7, the 50/50 mode mix and validation in the
  design mode (`--val-mode design`: real and heavy cells with `when_idle`, general
  cells with the mode drawn). Its scores are therefore not comparable with the
  `none` scores of §8.4–§8.5; the reference rows of its log give the new baselines.
* **Results** (`docs/methods/rl2.md` §8.6): 4 h 21 min, best iteration 2, network
  alone −6.6 / −2.5 / −2.9 % against `nearest_job` **with** pre-positioning (score
  −3.98 %), i.e. −13.3 % at the design load against `nearest_job` without it.
  Committed: `models/rl2_gen_s_d.pt`, `models/rl2_gen_s_d_training.json`.
* **First measurement of the information treatments A/B/C/D** of the research plan,
  with this network on the nominal cells (30 seeds, paired): seeing the entry zone
  without acting on it (B vs A) is worth −0.2 ±1.3 % at both loads, i.e. nothing;
  acting on it (C vs B) is worth 5.7 points at the design load and nothing at the
  heavy load; seeing the hidden upstream queue (D vs B) adds 1.2 points at the heavy
  load only. The network saw `zone` cells in 15 % of its training, so the A–B gap may
  move with more A training; the bench of the plan is still to run.

---

## 0.7.8 — 2026-11-22 — Research plan for two papers

The owner shared a research plan (two papers: the general problem and its algorithm;
the value of upstream visibility) and asked for an opinion and for it to be added to
the repository. `docs/plans/papers.md` holds the plan translated into English, with
review notes that map its treatments and tests to what the repository already has and
list the points where the 0.7.5–0.7.7 evidence argues for a change before the design
is frozen: the known-parts search is the ceiling rather than the method, the
difficulty reference has moved from the simple rollout to the sampled-future search
and the network, bursty arrivals and the upstream visible length need a definition
and a parameter, and the load grid should reach the saturated regime. No code change.

---

## 0.7.7 — 2026-11-22 — Review of the model and of the information: pre-positioning as the design mode, two leaks closed

The owner stated that **the cell does not exist yet**: the project develops the
simulation and the dispatcher first, to be taken to a physical cell later. The owner
then asked for an exhaustive review of the mathematical model and of the information
the controller has, and to change whatever the review required.

### Review

* **Mathematical model** (`docs/formal_formulation.md` §0–§3, §5): no error found. The
  odometer formulation, (C1)–(C7), the blocking penalty and its MILP witness, the
  decoder view and the NP-hardness proof are consistent with each other and with the
  simulator (the formal checker and `tests/test_formulation.py` still pass).
* **Information model** (§4.1): two features of the `rl2` tokens used information the
  camera does not give:
  1. the **waiting time** of a part was counted from its arrival to the upstream
     queue ($r_j$), which is hidden under `belts`: a part that waited upstream
     revealed the hidden queue. It is now counted from the instant the part became
     observable (entry on the belt; entry in the laser zone for `zone`; arrival for
     `full`), `rl2.features.observed_since`;
  2. **"arrivals still coming"** was read from the real arrival list, so the
     controller knew that no more parts would come before it could know it. It is
     now $t < T_A$, the known horizon.
  * The information set of §4.1 (Markdown and LaTeX) now lists $o_j$ and $T_A$
    instead of $r_j$, and says why the objectives lose nothing: $\sum_j F_j=\sum_j
    C_j-\sum_j r_j$ with the $r_j$ constants.
  * Both are small (they matter only for parts that waited upstream and near the end
    of an episode), and the number of features is unchanged, so earlier models still
    load. Measured on the 0.7.6 network (30 seeds, `none`): −8.39 % at the design load
    with the corrected features against −8.41 % with the old ones, and −4.52 % against
    −4.80 % at heavy load, both within the confidence intervals. The next training
    round uses the corrected features. `tests/test_observation.py` gains two tests: shifting the arrival time
    of a part that waited upstream, and deleting every future arrival, leave the
    tokens unchanged (and the first change is visible with `full`).
  * The old `rl` agent (`src/rl/features.py`, 0.5–0.6) keeps its features: it is a
    historical reference, no longer trained.
* **Restricted model pinned where it is assumed.** `optimization.instances.instance_config`
  now sets `laser.prepositioning: none` unless the instance overrides it: the
  sequence-only view, the exact models and the counterexamples are defined on
  P^sim (§1.5, §3). Four tests that check (C7) behaviour or analytic values of the
  restricted model state the mode explicitly.
* **Design point, not measurements.** `docs/assumptions.md` now states at the top that
  "the real cell" names the design point of the planned cell, that every parameter is
  a design choice, and that the sensing and kinematic assumptions are requirements for
  whoever builds it (§F reworded accordingly).

### Design decision: pre-positioning `when_idle` (`docs/assumptions.md` C4)

* **Why.** The default `none` came from the 2026-11-20 assumption that the cell could
  not detect parts before the laser zone. C4b (same day) gave the cell an overhead
  camera that sees the entry zone, which removes that reason, but the default was not
  revisited. Measured with the 0.7.6 network on the validation seeds (30 seeds, Δ vs
  `nearest_job`/`none`): the network alone goes from −8.4 % with `none` to **−13.8 %
  with `when_idle`** at the design load, and `nearest_job` itself gains 7.4 points;
  at heavy load nothing changes (the laser is never idle). The owner approved.
* **What changed.**
  * `config/default_config.yaml`: `laser.prepositioning: when_idle`; `none` stays as
    the variant without entry detection. Every result before 0.7.7 was obtained with
    `none` and is labelled so.
  * Training mix (`rl2.expert_iteration.GENERAL`, `sample_config`): `when_idle` with
    probability 0.5 (was 0.3), for the general cells and for the real and heavy ones.
  * Validation: new `--val-mode` (`design`, the default: real and heavy cells with
    `when_idle`, general cells with the mode drawn; `none`: the validation of
    0.7.2–0.7.6). Runs saved before 0.7.7 resume with `none`. Scores of the two modes
    are not comparable; `experiments/rl2_teacher_variants.py` takes the same option.
  * Docs: `formal_formulation.md` §1.5, §4, §4.1 (and the `.tex`; the PDF is **not**
    rebuilt, this machine has no LaTeX), `rl2.md` §2, §6.1, §6.3, README.
* **Requirements this puts on the cell:** the camera must cover the entry zone with a
  reliable detection, and the laser controller must accept "go to this point and wait"
  (and a revocation while heading there).

### Training

The 0.7.6 run `rl2_gen_s_sf` (validation in `none`, old features) continues; its
result goes to `rl2.md` §8.5. The first round with the design mode and the corrected
features will be `rl2_gen_s_d` (§8.6).

---

## 0.7.6 — 2026-11-21 — Stronger search teacher: sampled-future rollout over the network

The large runs of 0.7.5 showed that the search teacher, not the model size, limits
expert iteration: every network reaches its known-parts look-ahead within 0.3–0.7
points and the next iterations are rejected. The user asked to stop the L run and to
develop the stronger teacher (plan §4). Details and tables: `docs/methods/rl2.md`
§6.4 and §8.5.

### What changed (`src/rl2/search.py`, `expert_iteration.py`, `model.py`)

* **Cost-to-go look-aheads** (`SearchTeacher(horizon_s, scenarios, cut, window_s,
  scale)`). With a network base and `horizon_s > 0` the look-ahead scores the realized
  cost-to-go of all parts, Δ(WIP integral) + `w_pen` · penalties, averaged over
  `scenarios` sampled futures with common random numbers, plus a value-head estimate
  at the cut:
  * `horizon_s = inf`: no cut, the clone drains after the synthetic window, no value
    head (**the teacher adopted**, with 8 futures and a 30 s window);
  * `cut = "time"`: cut at exactly t + `horizon_s`, value head there;
  * `cut = "decision"`: cut at the first decision with ≥ 2 candidates after it.
  * The known-parts look-ahead of 0.7.2 is unchanged and remains the default
    (`horizon_s = 0`); with the rule base (iteration 0) a `horizon_s > 0` falls back
    to it without scenarios.
  * *Why the objective changed:* with synthetic arrivals, the completion times of the
    known parts reward starving the new ones (the +7.9 % of 0.7.2); the WIP integral
    of all parts is the cost the policies are judged on.
* `rl2.model.load_extra` / `load_scale`: read the value scale saved with a model, so
  a worker can denormalize the value head.
* `experiments/rl2_train.py`: `--la-horizon`, `--scenarios`, `--la-cut`,
  `--la-window`; the teacher of each job is built by `_teacher`. Runs saved before
  0.7.6 resume with the known-parts teacher.
* `experiments/rl2_teacher_variants.py`: compares teacher variants over a trained
  network on the validation cells (Δ vs `nearest_job`, Δ vs the network alone, time
  per episode).
* **Tests** (`tests/test_rl2_search.py`, 7 new): cut at a decision after the horizon,
  cut at the exact instant, rollout to the end without value head, cost = realized
  cost-to-go + scaled value, lock-step batching equals sequential and the real future
  is ignored in every mode, the rule base keeps the known-parts look-ahead, value
  scale round-trip. 243 tests pass.

### Findings (30 validation seeds per group, teacher over `rl2_gen_s_big.pt`)

| teacher | real | heavy | general | score |
|---|---|---|---|---|
| network alone | −4.6 % | −2.5 % | −1.6 % | −2.88 |
| known parts (0.7.2) | −5.2 % | −2.7 % | −2.5 % | −3.47 |
| value head at a cut (any variant) | +17 … +58 % | | | |
| 4 sampled futures, 30 s, no value head | −6.7 % | −3.5 % | −3.0 % | −4.40 |
| **8 sampled futures, 30 s** | **−7.5 %** | **−3.8 %** | **−3.2 %** | **−4.84** |

* **The value head cannot bootstrap.** Its error (a quarter of the cost-to-go scale,
  which is dominated by the random future) is far larger than the differences
  between candidates. Every value-head variant is catastrophic; the fixed-instant cut
  is worse still because the head only saw decision states.
* **Averaged sampled futures beat the known-parts teacher in every group**, by 1.4
  points of score with 8 futures; one or two futures are worse than none. A 20–30 s
  window is best; the penalty in the look-ahead hurts, as in 0.4 and 0.7.2; K = all
  changes nothing.
* The chosen teacher costs about 10× the old one per episode (40 min per 2 000
  episodes on the workstation).

### Training (`rl2_gen_s_sf`)

Continuation of the best S model with the new teacher (2 000 episodes and 4 epochs per
iteration, 50 validation seeds per group, `none`): 3 h 53 min on the workstation,
iterations 1–4, stopped by the patience rule with **iteration 2** as the best model.

| policy | real cell | heavy example | general cells | score |
|---|---|---|---|---|
| best S of 0.7.5 (known-parts teacher) | −4.4 ±1.4 % | −2.3 ±1.6 % | −1.6 ±1.1 % | −2.79 |
| **`rl2_gen_s_sf` alone (it. 2)** | **−7.6 ±1.4 %** | **−4.4 ±1.1 %** | **−2.4 ±1.1 %** | **−4.83** |
| search over it | −7.2 ±1.6 % | −3.5 ±1.2 % | −2.7 ±1.2 % | −4.48 |

The network absorbed the new teacher in one iteration and, from iteration 2, matches or
beats the search over itself at the real and heavy loads; later iterations are
rejected. Tables and discussion: `docs/methods/rl2.md` §8.5. Committed:
`models/rl2_gen_s_sf.pt` and `models/rl2_gen_s_sf_training.json`.

---

## 0.7.5 — 2026-11-21 — Large-scale training on a GPU workstation (sizes S, M, L)

The user asked to train the expert-iteration dispatcher of 0.7.2–0.7.3 on a larger
machine, following `docs/methods/remote_training.md`: first a short timing round, then
the size-S run with the defaults of `run_train.sh`, then M, and (asked during the run)
L, chained overnight. Results and tables: `docs/methods/rl2.md` §8.4. This entry adds
no code; it adds the trained models and documents the runs and what was learned.

### Machine and setup

* **Machine:** `isecapstone`, AMD Threadripper 3960X (24 cores / 48 threads), 62 GB,
  NVIDIA RTX 3080 10 GB, driver 580.178 (CUDA 13.0), no Slurm, Python 3.12.3. A shared
  workstation (a database and other services run on it), so the runs used 47 workers
  and left the rest to those services.
* **Environment:** `setup_env.sh` with `TORCH_INDEX=…/cu128` (PyTorch 2.11.0+cu128).
  The first run failed before the tests: the login shell exports `PYTHONPATH` with ROS
  Jazzy, and pytest tried to load the ROS `launch_testing` plugin. With
  `env -u PYTHONPATH` the script ended with `setup OK`: 49 tests passed, none skipped,
  so the CUDA-vs-CPU check of `tests/test_rl2_training.py` ran on this machine. Noted
  in `remote_training.md` §3; no script changed.
* **Timing round** (`rl2_timing_s`, S, 400 + 200 episodes, 10 validation seeds):
  1 minute in total. It gave the per-step costs used to size the runs (§8.4, times).

### Runs (branch `remote-training`)

| run | size | episodes it. 0 / later | iterations | best | wall time |
|---|---|---|---|---|---|
| `rl2_gen_s_big` | S | 4 000 / 2 000 (defaults) | 0–5, stopped by patience | it. 3 | 25 min |
| `rl2_gen_m` | M | 8 000 / 4 000 | 0–5, stopped by patience | it. 3 | 5 h 53 min |
| `rl2_gen_l` | L | 16 000 / 8 000 | 0–1; stopped by the user during iteration 2 (0.7.6) | it. 1 | 14 h for its. 0–1 |

* **Data for L** was set at 4× S and 2× M by the rule of the plan (§3: a larger model
  needs proportionally more search data). The user did not fix this number; it is the
  one decision taken here.
* Launched with `nohup bash experiments/remote/run_train.sh` through two small chain
  scripts kept outside the repository (`results/experiments/chain_*.sh`); the
  experiment arguments were never changed, and no run needed a resume.
* **Committed:** `models/<tag>.pt` and `models/<tag>_training.json` of each run. Not
  committed: `state.pkl`, `results/`, the per-iteration models.

### Results (50 validation seeds per group, 100 general cells, Δ vs `nearest_job`)

| policy | real cell | heavy example | general cells |
|---|---|---|---|
| `nearest_head` | −1.8 ±1.3 % | +0.2 ±1.0 % | +0.9 ±1.0 % |
| search over `nearest_job` (rule teacher) | −4.4 ±1.4 % | +0.1 ±1.6 % | −1.6 ±1.1 % |
| **S alone** (it. 3) | −4.4 ±1.4 % | −2.3 ±1.6 % | −1.6 ±1.1 % |
| search over S | −4.7 ±1.5 % | −2.2 ±1.4 % | −2.4 ±1.1 % |
| **M alone** (it. 3) | −4.5 ±1.6 % | −2.2 ±1.4 % | −1.9 ±1.1 % |
| search over M | −5.0 ±1.5 % | −2.4 ±1.3 % | −2.2 ±1.2 % |
| **L alone** (it. 1) | −4.3 ±1.5 % | −1.7 ±1.3 % | −1.4 ±1.0 % |
| search over L (it. 1) | −4.9 ±1.5 % | −1.8 ±1.3 % | −2.2 ±1.0 % |

* **Against the laptop references** (30 seeds, `rl2.md` §8.4): `nearest_head` matches
  them exactly. The rule teacher matches at the real load and on general cells but
  not at heavy load (+0.5 % here, −2.2 % there): the laptop reference predates the
  information model of 0.7.3, whose look-ahead could still see the parts waiting
  upstream. On those 30 seeds S alone gives −4.6 / −2.5 / −1.6 % and M alone
  −5.3 / −2.0 / −2.0 %; the narrow pilot's −3.8 / −3.9 % (full information, 3-belt
  cell only) is beaten at the real load by a model that also covers the general
  problem.
* **What the runs show:**
  * the network alone reaches its rule teacher at the real load and generalizes
    (about −2 % on cells where `nearest_head` loses), at no search cost;
  * the search over the network saturates at about −5 / −2.5 / −2.4 % from iteration
    1 on, for every size, and the network closes to within 0.3–0.7 points of it by
    iteration 2–3; the remaining iterations are rejected. The teacher, not the model,
    is the ceiling;
  * M's best score (−2.88 %) is not distinguishable from S's (−2.79 %) and M costs
    8.5× more per generated episode and 4× per decision; L (4× the data) scores −2.15 % at
    iteration 0 and −2.44 % at iteration 1, below S and M at the same stage, with a
    higher training loss and 12 h per iteration. The scaling ladder stops at M.
* **Recommendation:** keep **S** as the deployed model (`rl2_gen_s_big.pt`) for the
  5 ms budget and the phase-4 evaluation; spend the next effort on a stronger teacher
  (plan §4: value-head truncation with sampled futures, larger K, Gumbel search)
  rather than on a larger network.

### Files

* `models/rl2_gen_s_big.pt`, `models/rl2_gen_m.pt` and the `_training.json` logs of
  the three runs (seeds, arguments, machine, times, losses, validation).
  `models/rl2_gen_l.pt` (157 MB) exceeds GitHub's 100 MB file limit and is kept outside
  the commit until the user decides (Git LFS or an external store).
* `docs/methods/rl2.md` §8.4, `docs/methods/remote_training.md` §3 and §4.
## 0.7.4 — 2026-11-20 — Results of the general pilot training (`rl2_gen_s`)

**What.** The pilot of phase 3 finished on the laptop with the real information model
(`belts`, 0.7.3):

* model S, 6 iterations;
* 3500 episodes and about 168 000 labelled decisions;
* 1 h 38 min on 8 CPU workers.

The run stopped early after two iterations without validation improvement. The accepted
model is iteration 3.

**Validation, paired Δ vs `nearest_job`** (seeds 2000+ / 2500+, used for gating):

| policy | real cell | heavy load | general cells |
|---|---|---|---|
| network alone (`models/rl2_gen_s.pt`) | **−3.4 ±1.7 %** | **−3.3 ±1.7 %** | **−1.8 ±1.4 %** |
| search over the network (k = 6) | −5.5 ±1.8 % | −2.6 ±1.7 % | −2.6 ±1.6 % |
| search over `nearest_job` (rule teacher) | −4.9 ±1.8 % | +0.5 ±2.2 % | −2.2 ±1.3 % |
| `nearest_head` | −1.6 % | +0.4 % | +1.1 % |

* The network alone, at about 4 ms per decision, beats `nearest_job` significantly in
  all three groups.
* At heavy load it also beats the rule teacher.
* Validation plateaued after iteration 3. The large run on the remote machine (S and L,
  more data) tests whether model size or data volume is the limit.
* These numbers are selected on the validation seeds. The unbiased comparison is the
  phase-4 final evaluation on the reserved seeds 7000–7029.

**Why it matters.** It is the first policy trained on the general problem that agreed
with the owner (N belts, per-belt speeds and arrival rates, processing time, laser
vx/vy, ...) that improves on `nearest_job` across that family. It uses only the
information the real cell observes.

**Where.**

* Full table per iteration: `docs/methods/rl2.md` §8.3.
* Status: `docs/plans/strong_rl_plan.md` §0.
* Versioned files: `models/rl2_gen_s.pt` (3.5 MB, identical to the iteration-3
  checkpoint) and `models/rl2_gen_s_training.json`.
* `.gitignore` now excludes the per-iteration checkpoints `models/*_it[0-9]*.pt`. They
  are about 3.5 MB each and stay local; only the accepted model is versioned. The
  training log still names them for traceability.

No code or model change.

---

## 0.7.3 — 2026-11-20 — Information model of the controller; general stochastic problem in the formulation

The user asked whether the mathematical formulation was correct. The review found that
the deterministic model is right but two things were missing:

1. the **information** the controller may use;
2. the **general stochastic problem** that the learned policies target.

The user then stated what the real cell observes: an **overhead camera sees every
belt and its whole working area** (entry zone and laser zone), so parts and their
positions are known. Parts waiting upstream of the belt entry are not seen.
Pre-positioning stays `none` (C4).

### Information model (`src/simulation/observation.py`)

* **New option `simulation.observation`:**
  * `belts` is the real cell and the default (`auto`): parts on the belts are
    observable;
  * `zone` is a what-if: laser-zone parts only, incompatible with pre-positioning;
  * `full` is a what-if: also the parts waiting upstream, i.e. the implicit behaviour
    of every earlier version.
* **Why:** a policy may only use what the cell observes. Up to 0.7.2:
  * the look-ahead (`rollout`, hybrids) and the 0.7 search teacher simulated the
    waiting parts too;
  * the RL agent saw the upstream buffer length;
  * the `rl2` tokens included waiting parts.
* **Where it is enforced:**
  * `Simulation.observable` and `FastSim.observable`;
  * `policies/rollout.py`: unobservable parts are removed from the look-ahead clone;
  * `rl/features.py`: slots and counts use observable parts only;
  * `FastSim.lookahead_clone`, `rl2/search.py` (known parts) and `rl2/features.py`
    (tokens and belt counts, plus the observation level in the context token,
    `CTX_F` 18 → 21).
* The rule policies select among candidates only, which are always observable, so
  they are unchanged.
* **Tests:** `tests/test_observation.py` (13 tests). For the `belts` and `zone` levels,
  changing or removing the unobservable parts at a decision never changes:
  * the engine rollout costs;
  * the RL observation;
  * the fast-simulator teacher costs;
  * the `rl2` tokens.

  The same change is checked to be visible under `full`, so each test can detect a
  leak. The suite has 231 tests, all passing; one is skipped (CUDA).

### Effect on the results (`docs/methods/confirmatory.md` §8)

The real-cell comparison was rerun (seeds 3000–3029, `none`, tag
`v0.7.3_confirmatory_belts`):

| policy | real cell | heavy load |
|---|---|---|
| `rollout@avg_flow_time` | −2.4 ±1.3 % (was −1.9 %) | −0.9 % (was −1.7 %) |
| `rl_rollout` | −2.3 ±1.4 % (unchanged) | −2.1 % (was −3.4 %) |

* `nearest_job` and `nearest_head` are identical to before on all 60 runs.
* The real-cell gains do not depend on the waiting parts. At heavy load about one point
  did.
* Recommendation: `rollout@avg_flow_time` or `rl_rollout` with `none` (equivalent).

### Formulation (`docs/formal_formulation.md`, `docs/formulation/formal_formulation.tex`, PDF rebuilt)

* **§4.1 Information model:** the information set $\mathcal I_t$ with observable
  parts $O(t)$, the three levels, and what changed in 0.7.3.
* **§4.2 The general stochastic problem:**
  * the cell parameter vector θ (N, per-belt y, v, λ, G, vx, vy, motion model,
    geometry, stop rule, λ_N, λ_T, pre-positioning, observation);
  * Poisson or deterministic arrivals per belt, i.i.d. processing times;
  * finite episodes with drain;
  * the single-cell objective $J(\mu;\theta)$ and the general objective
    $\mathbb E_\theta[J(\mu;\theta)/J(\mu_{NJ};\theta)]$, and how it is estimated;
  * load $\rho_p$ vs effective load ρ. Overloaded cells are deliberately included
    in the training distribution.
* **Updated:**
  * the RL paragraph: decisions with ≥ 2 candidates, any-part action in `rl2`;
  * the tie-break rule (0.7.0), now part of the definition of a policy;
  * notation (λ_k, G);
  * the simulator correspondence: information model and the exact event-driven
    simulator.
* **Checked and correct (no change):** odometer and entry coordinates, (C1)–(C7), the
  blocking penalty, the Little identity of the RL cost, the lower bounds, and the
  strong NP-hardness proof.
* `docs/assumptions.md` C4b records the camera.

### Training

* The general run `rl2_gen_s` (0.7.2) was stopped during iteration 0 because it used
  the waiting parts. Its log is in
  `results/experiments/archive/rl2_gen_s_fullinfo_stopped/`.
* It is restarted with the real information model (`docs/methods/rl2.md` §8.2–8.3).

### Small fix

* `play_episode` clips relative look-ahead costs below the tie tolerance (−5.7·10⁻¹⁴
  round-off) to 0, so the best candidate of a label always has cost exactly 0.

---

## 0.7.2 — 2026-11-20 — Strong-RL plan, phase 3: expert iteration on the general problem; portable training

The user asked to start phase 3. During it, the user stated the target precisely:
**the problem to solve is the general one**:

* N conveyors with speeds v_1 … v_N;
* arrival rates λ_1 … λ_N;
* processing time t;
* laser speeds vx, vy;
* any other parameter that can be generalized.

The real cell is one instance of it. The user approved the ranges below and asked to
make training portable to a larger machine. Details: `docs/methods/rl2.md` §6–§9 and
`docs/methods/remote_training.md`.

### Search teacher (`src/rl2/search.py`)

* **What:** a rollout teacher on the fast simulator.
  * It tries each candidate (top 6 by the base policy, always including the base
    policy's choice).
  * The base policy then runs until every part known at the decision is processed.
  * The cost is the sum of their completion times. The base choice is kept on ties.
  * The base policy is `nearest_job` in iteration 0 and the current network afterwards
    (as in `rl_rollout`).
  * All look-aheads of one decision advance in lock-step, with one batched network
    call per step.
* **Why:** 0.7.1 showed that per-decision network calls are the bottleneck; batching
  them removes most of that cost.
* **Fairness:**
  * new `FastSim.lookahead_clone(scenario, window_s)` drops the real future arrivals;
  * a test checks that changing the real future does not change any cost;
  * another test checks that lock-step batching gives exactly the costs of running each
    look-ahead alone.
* **Variants checked** (validation seeds 2000–2019, rule base, vs `nearest_job`):

  | teacher | real cell | heavy example |
  |---|---|---|
  | known parts (**used**) | −3.8 % | −2.1 % |
  | with the blocking penalty | −5.6 % | +5.8 % |
  | with 4 sampled futures | +7.9 % | +13.2 % |

  The penalty and sampled futures stay available as options but are not used (same
  conclusion as the 0.4 penalty-aware look-ahead).

### Expert iteration (`src/rl2/expert_iteration.py`, `experiments/rl2_train.py`)

* **Iteration 0:** the rule teacher acts (10 % random actions) and labels every
  decision with ≥ 2 candidates. Iterations k ≥ 1: the rollout over `net_{k-1}` labels,
  while `net_{k-1}` acts and the teacher acts with probability 0.3 (DAgger).
* **Losses:** cross-entropy to the softmin of the look-ahead costs (temperature 1 s),
  plus 0.25 × MSE of the normalized cost-to-go.
* **Gating:** validation on the real cell, the heavy example and general cells. A
  network is kept only if the mean paired Δ vs `nearest_job` improves.
* **Context token:** one more feature, the time left to the episode's arrival
  horizon (`CTX_F` 17 → 18). *Why:* the value head predicts the episode's cost-to-go,
  which depends on it.

### General problem (`GENERAL`, `general_config`, `sample_config`)

* **Training mix** (seeds 100000+): 25 % the real cell, 10 % the heavy example, 65 %
  general cells with:
  * N = 1–6 belts;
  * v_k ~ U[1.5, 6] cm/s per belt;
  * Poisson λ_k per belt, with processing load Σ λ_k · mean p ~ U[0.3, 0.95] split by
    random weights;
  * processing time mean U[1.5, 4] s, constant, uniform or normal;
  * vx, vy ~ U[5, 20] cm/s each;
  * simultaneous or sequential axes;
  * geometry × U[0.5, 1.5];
  * λ_N ~ U[0, 20] s, λ_T ~ U[0, 2];
  * either conveyor stop rule;
  * `none` (70 %) or `when_idle`.
* **Validation:** the real cell and the heavy example (seeds 2000–2029, `none`), plus
  60 general cells (seeds 2500–2559, `none`).
* **Not generalized yet** (stated to the user):
  * a processing-time distribution per belt or per part type;
  * laser acceleration;
  * the `pass` rule.
* **First training run stopped.** `rl2_s` had sampled only a few parameters of the
  3-belt cell, so it was stopped after its iteration 0 and replaced by `rl2_gen_s`.
  * Its iteration-0 network alone gave −3.8 ±1.8 % (real) and −3.9 ±1.4 % (heavy) vs
    `nearest_job`; the rollout over it gave −6.0 % and −4.9 %.
  * Kept outside the repository in `results/experiments/rl2_s_narrow/`.

### Fast simulator on general cells (`experiments/fastsim_validation.py --loads general`)

* **Checker:** 0 violations on 40 general cells × 2 modes × 2 rules.
* **Agreement with the engine at `dt = 0.01`** is lower than on the real cell
  (identical order 67 %, mean |KPI diff| 0.97 %, max 6.4 %).
* **The engine is wrong, not the fast simulator.** Rerunning the worst cases at
  `dt = 0.001` and `0.0002` makes the difference shrink linearly to about 0.01 %: it is
  the engine's O(dt) lag, larger with faster belts and laser.
* **Consequence for phase 4:** general-cell numbers need `dt ≤ 0.001` or the exact
  simulator.
* `fastsim_validation.py` gained `--loads` (including `general`) and a shared
  `make_config`.

### Portable training (`docs/methods/remote_training.md`, `experiments/remote/`)

* **Why:** network training is the bottleneck on the laptop (≈ 5 min per epoch of
  50 000 decisions, CPU), and sizes M/L need a GPU. The user agreed to prepare the code
  for a larger machine.
* **What:**
  * `--device auto|cpu|cuda|mps` for training. Workers stay on the CPU, and models are
    saved with CPU weights (`model.save`, `Batch.to`, `resolve_device`).
  * **Checkpoint/resume:** the full run state is written atomically to
    `results/experiments/<tag>/state.pkl` after each iteration, and `--resume`
    continues it. Experiment arguments are taken from the saved run; only machine and
    stopping arguments may change.
  * Repository-relative paths in logs, and a per-machine record in the log.
  * `--workers` defaults to all CPUs; `--horizon` for quick checks.
  * Scripts: `setup_env.sh` (venv, CUDA PyTorch, tests, 2-minute check),
    `run_train.sh` (resume-aware launcher) and `slurm_train.sbatch` (requeue-safe
    template).
  * `.gitattributes` keeps shell scripts with LF line endings.
* **Tests:** `tests/test_rl2_training.py`: device, CPU save, an end-to-end checkpoint
  and resume, and a CUDA-vs-CPU check that runs only where CUDA exists.

### Results

General training run `rl2_gen_s` (laptop, size S): in progress; results will be added
to `docs/methods/rl2.md` §8.2.

---

## 0.7.1 — 2026-11-20 — Strong-RL plan, phase 2: token representation and Transformer pointer model

The user accepted the revised phase-1 acceptance and asked to start phase 2
(`docs/plans/strong_rl_plan.md` §3). This entry adds the model only. It is untrained,
and no published result changes.

### What was added (`src/rl2/`, `docs/methods/rl2.md`)

* **`rl2/features.py`:** the state as an unordered set of tokens:
  * one per arrived unfinished part (23 features);
  * one per conveyor (11);
  * one laser token (9);
  * one context token with the cell parameters (17).
  * *Why:* the 0.5–0.6 agent saw 3 slots per belt and was tied to N = 3. Tokens carry
    no list position and no conveyor index, so the model sees the whole queue and does
    not depend on the number of parts or belts. Belt membership is given by the y
    offset from the laser and a copy of the belt state. Times use
    `slog(t) = sign(t)·log(1 + |t|/10 s)`, so long waits at heavy load stay bounded.
* **`rl2/model.py`:**
  * `DispatchNet`: type embeddings; a pre-norm Transformer encoder without positional
    encoding; a pointer policy over **any candidate part** (non-candidates masked to
    −∞); a value head for the normalized cost-to-go.
  * Also: `collate` (padding), `NetPolicy` (dispatcher for `FastSim.run`),
    `RunningScale`, and `save` / `load`.
  * Sizes S (0.88 M parameters), M (6.7 M) and L (39 M) share the code.
  * *Why a pointer:* a head-only action loses up to 3.5 % with bursts (plan §1).
* **Activation ReLU instead of GELU** in the feed-forward layers, and `inference_mode`
  in `NetPolicy`. *Why:* about 10 % lower latency for size S. The plan does not fix
  the activation.
* **Tests:** `tests/test_rl2_model.py`, 10 tests: permutation invariance, masking,
  batching equals single calls, features, non-head candidates, a valid episode
  checked by the formal checker, save/load, latency.
* **Benchmark:** `experiments/rl2_latency.py`.

### Acceptance

* **Permutation invariance and masking:** both pass.
* **Latency, one state, 1 thread:**
  * S: median 4.1 ms (default load) and 4.7 ms (heavy), so the plan's 5 ms budget is
    met on the median; p95 is 5.3–5.8 ms.
  * M: 15–19 ms. L: 64–81 ms.
  * Size S stays the main track, as in §0 of the plan.

### Finding that shapes phase 3

* A FastSim episode driven by S runs at 340 decisions/s, against 5 500 for the
  simulator with a rule. The network is now the bottleneck.
* A batch of 64 states gives 1 600–2 400 states/s.
* The phase-3 search will therefore advance its candidate × scenario rollouts in
  lock-step with one batched forward pass per step, and truncate with the value
  head (`docs/methods/rl2.md` §5).

### Not done yet

* No engine adapter: the `rl2` policy for the fixed-step engine (used for the final
  phase-4 numbers) needs the same features computed from `simulation.engine` state.
  It will be added in phase 4.

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
