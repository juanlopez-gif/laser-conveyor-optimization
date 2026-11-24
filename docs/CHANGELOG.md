# Changelog and decision log

Every change of the model, the simulator or the methods is recorded here with **what**
changed, **why**, and **where** (files, tests, documents). Earlier versions are
summarized from the git history.

---

## 0.7.30 — 2026-11-23 — Paper 1 draft 0.4: the PPO baseline integrated (V1 from scratch, V2 fine-tuning)

* **Why.** The owner asked to wait for the PPO session's final report before draft 0.4 and
  to present V1 as PPO from scratch and V2 as a fine-tuning of the learned policy, the
  improvement judged by the evaluation performance and the training cost, a larger KL
  read only as "the policy changed more during training".
* **What.** `docs/papers/paper1/paper1.tex` (draft 0.4): the "work in progress" line of
  the title page removed; abstract, introduction, §8.4 (now "the earlier agents and the
  PPO baseline" plus a new §8.5 `sec:ppo` with the protocol, V1, V2, cost and latency,
  and a reading), Limits and Conclusions rewritten with the result: from scratch PPO
  stays +3.6 / +4.5 / +7.9 % above `nearest_job` and +7.9 to +12.3 % above the frozen
  expert-iteration network at a comparable number of acting episodes (two runs still
  improving at the budget, 0.5–1 point per 4 000 episodes; the greedy/sampled decoding
  divergence of one seed reported), fine-tuning the learned policy makes it worse on all
  three seeds (+0.67 to +1.72 % vs the frozen network, the first checkpoint the best,
  KL 0.06–0.15 at the first updates), one PPO run costs 22 min / 14.5 CPU-h against
  12.7 h / 596 CPU-h for the chain (≈ 39 M look-ahead clone runs), latency 1.06 ms per
  decision as for any size-S network. `experiments/paper1_figures.py`: tables `ppo_v1`,
  `ppo_v2`, `ppo_cost` and figure `ppo` (two panels, legends below each panel) generated
  from the versioned `docs/papers/paper1/data/ppo_*.json` and the `models/rl2_ppo_*_training.json`
  logs. PDF rebuilt.
* **Where.** `docs/papers/paper1/{paper1.tex,paper1.pdf,tables/ppo_*.tex,figures/ppo.pdf,data/ppo_*.json}`,
  `experiments/paper1_figures.py`, `docs/plans/papers.md`.

---

## 0.7.29 — 2026-11-23 — PPO baseline on the token policy (`docs/methods/rl2_ppo.md`), merged from `claude/ppo-baseline-e08600`

* **Why.** Paper 1 claims that the search teacher is what makes the amortized
  policy good; the only reinforcement-learning comparison so far was the old PPO of
  0.5–0.6 on another representation. A PPO baseline with the **same** tokens,
  network (`DispatchNet` size S, 0.88 M parameters), action mask, simulator, cell
  distribution and validation realizations as the expert-iteration chain answers
  "what does the teacher add over the reward alone?" at equal inference cost. The
  protocol (reward, advantages, hyperparameters, budget, seeds, checkpoint
  selection, evaluation, cost accounting) was frozen and committed before the full
  runs and not changed afterwards (rl2_ppo.md §1).
* **What.** `src/rl2/ppo.py` (semi-MDP PPO between network decisions: reward = minus
  the increase of the penalized objective of `cost_so_far()`, γ = 1, GAE 0.95 in cost
  units, per-batch advantage normalization, value targets on the running scale,
  clipped surrogate 0.2, 4 epochs, minibatch 256, entropy 0.01, AdamW 3 × 10⁻⁴;
  workers return the sampled trajectories, the GPU updates; one update per 1 000
  episodes, checkpoint every 4 000 scored greedy on the validation groups, best
  kept), `experiments/rl2_ppo.py` (CLI), `experiments/rl2_ppo_eval.py` (paired
  evaluation), `experiments/rl2_ppo_cost.py` (cost of the PPO runs and of the
  expert-iteration chain, teacher look-aheads measured on a sample),
  `experiments/rl2_ppo_anomaly.py` (greedy vs sampled decoding),
  `experiments/rl2_ppo_latency.py`; `tests/test_rl2_ppo.py` (7 tests: masking,
  GAE identities, normalization, loss at ratio 1, update direction, tiny end-to-end
  run). Models `models/rl2_ppo_s0.pt`, `_s1`, `_s2` (from scratch, 40 000 episodes
  each, training seeds 200 000+ / 300 000+ / 400 000+) and `models/rl2_ppo_ft_s3.pt`,
  `_s4`, `_s5` (fine-tuning of `rl2_gen_s_b`, 12 000 episodes), with their
  `_training.json`. Results in `results/experiments/rl2_ppo_*` (not versioned),
  tables in rl2_ppo.md §3.
* **Result.** From scratch, with a comparable number of acting episodes to the
  chain (40 000 vs 36 000), PPO stays above `nearest_job` on every seed: best greedy
  scores +4.49 / +7.88 / +3.62 % (mean over seeds +5.33 %), against −3.82 % for the
  frozen expert-iteration network on the same realizations; vs that network PPO is
  +8.78 / +12.31 / +7.89 % worse (mean +9.66 %), losing on every group. Two of three
  runs were still improving at the end of the budget (≈ 0.5–1 % per 4 000
  episodes); seed 1's greedy score worsened after 16 000 episodes while its sampled
  policy kept improving (§3.5: the PPO policies are still stochastic, 0.8–1.0 nats,
  and their sampled score is 15–40 points worse than greedy). Fine-tuning the frozen
  network with PPO (V2) makes it worse on all three seeds (−2.26 / −3.22 / −2.73 %
  at the best checkpoint, the first; +1.20 % vs the frozen network on average), the
  first update moving the policy far (KL 0.06–0.15 per update). Cost: one PPO run
  = 22 min wall / 14.5 CPU-h upper bound on 40 workers, the chain = 12.7 h /
  596 CPU-h on 47 workers with ≈ 39 M look-ahead clone runs; the PPO checkpoint has
  the network's latency (1.06 ms per decision, one thread, same machine as the
  untrained reference).
* **Where.** `docs/methods/rl2_ppo.md` (§1 frozen protocol, §2 smoke test, §3
  results: learning curves, paired evaluation, cost, latency, anomaly, V2),
  `src/rl2/ppo.py`, `experiments/rl2_ppo*.py`, `tests/test_rl2_ppo.py`,
  `models/rl2_ppo_*`. Not done (rl2_ppo.md §1.7): hyperparameter search, reward
  shaping, KL penalty towards the frozen network, longer budgets.
* **Integration.** Run in its own session and worktree (branch `claude/ppo-baseline-e08600`,
  commits 3f8eff4, 40df8a4, 8c93a60; merge commit 0fa2301 on `remote-training`, no
  conflicts). The branch's document and §4 text carried the number 0.7.25, already used
  by Paper 1 draft 0.2; they are renumbered 0.7.29 here. Its per-run results under
  `results/experiments/rl2_ppo_*` are git-ignored; the paired evaluations, the cost
  accounting, the latency and the decoding anomaly are versioned for Paper 1 as
  `docs/papers/paper1/data/ppo_{eval,eval_ft,cost,latency,anomaly_s1}.json` (0.7.30).

---

## 0.7.28 — 2026-11-23 — Belt-speed bench run on the reserved seeds 10000–10059 (`geometry.md` §8)

* **Decisions (owner, `geometry.md` §7.6).** Reserved seeds **10000–10059** (checked
  unused in every `results/experiments/**/*.json`; the test suites use 100000+), margin
  **δ = 0.5 s/part**, **`nearest_job` only** (the network did not run), explicit go.
  `experiments/belt_speed_bench.py`: `RESERVED_BASE = 10000`, docstring updated; the
  protocol of §7.1–§7.5 was not changed after the smoke test or the run.
* **Run.** Smoke test on development seeds 3100–3101 (one cell, 96 episodes: 0
  violations, identical arrivals at both speeds, exact identity at ℓ = 0), then
  `--tag belt_speed_bench --confirmatory --who nearest_job`: 11 520 episodes (2 speeds
  × 4 cells × 12 lengths × 2 modes × 60 seeds), 7 s on 40 workers. Outputs under
  `results/experiments/belt_speed_bench/` (git-ignored); the extract is versioned as
  `docs/papers/paper2/data/belt_speed_bench.csv.gz` (0.8 MB; `--from-extract`
  regenerates identical tables without PyTorch, checked); figures
  `docs/methods/figures/belt_speed_bench_nearest_job_{curves,cost}.{pdf,png}`.
* **Results (`docs/methods/geometry.md` §8, criteria as frozen).** Checks pass (0
  violations over 11 520 episodes, common arrivals in all 240 (cell, seed) pairs); the
  C_3(10 cm) column reproduces the 3 cm/s costs of §6.2 within the precision (the
  observable tie-break of 0.7.26 moved nothing detectably). At 5 cm/s, **10 cm keep the
  benefit in the two cells with v_x = 10 cm/s** (H_cm conservative: upper bounds of the
  cost 0.32 and 0.23 s/part against δ = 0.5; 16.7 cm add nothing detectable), and
  **the benefit of pre-positioning is lost in the two cells with v_x = 5 cm/s**
  (G_5(50 cm) = +0.62 ±0.73 and +0.47 ±0.55 s/part against +1.74 ±0.87 and +1.58 ±0.89
  at 3 cm/s: the design result of §7.1, not H_neither, since C_5(16.7 cm) stays ≤ δ).
  The scale question (centimetres or seconds) is **undetermined**: the discriminating
  cells lost the benefit, the others answer H_cm under either hypothesis (foreseen in
  §7.1 (i)). The frozen alignment reading favours the time scale in three cells and the
  distance scale in one, with a post hoc caveat recorded. G_3(50) − G_5(50) paired:
  −0.77 ±0.60 (12 / 10 / v_x 10, more benefit at 5 cm/s) and +1.11 ±1.02 (6 / 15 /
  v_x 5, less), the other two not detected. Premise check: with the 50 cm entry the
  cell 12 / 10 / v_x 10 sits on the benefit threshold at 3 cm/s (0.99 ±0.38 against
  1.0; lower bound 0.61), reported as it falls. Traces (moves, idle share, t_x up at
  5 cm/s, t_y unchanged) are compatible with the interception moving downstream and
  fewer idle windows; a reading, not a tested hypothesis.
* **Where it goes.** Paper 2 §6.7's "untested" statement on belt speed is replaced by
  §8.5 of `geometry.md`; the paper itself (draft 0.7, shared with co-authors) is not
  edited in this version. Used reserved ranges are now 7000–7029, 9000–9059 and
  10000–10059.

---

## 0.7.27 — 2026-11-23 — Belt-speed bench (3 vs 5 cm/s): frozen protocol and script, not run

* **Why.** Every cell of the geometry bench runs its belts at 3 cm/s, so seconds of
  anticipation and centimetres of visible belt are one variable (`ℓ_v = 3 τ_v`); the
  audit of Paper 2 draft 0.6 and the paper's Limits section (0.7.25) name a bench that
  varies the belt speed as the next experiment: at 5 cm/s, does keeping 10 cm keep the
  benefit of pre-positioning, must the view grow to 16.7 cm to keep 3.33 s, or neither,
  because the interception changes?
* **What.** `docs/methods/geometry.md` §7, written before any run: three exclusive
  hypotheses (H_cm, H_s, H_neither) decided per cell on the paired cost of the shorter
  view against the full 50 cm view, a margin δ = 0.5 s/part proposed as the owner's
  decision (the largest cost of 3.33 s observed in §6.2 was 0.44), numerical predictions
  under each hypothesis from the 3 cm/s extract, the design (3 and 5 cm/s × four cells,
  spacing 12 / v_y 10 and 6 / 15 with v_x 5 and 10, nominal load, Poisson, common 50 cm
  entry, 12-point grid in cm including 10, 15 and 16.7 cm, the same arrival realizations
  at both speeds), frozen measures (equal-distance and equal-anticipation paired
  comparisons, knees in cm and s, Bonferroni z = 2.838 for 11 non-zero lengths) and the
  pending owner decisions. `experiments/belt_speed_bench.py`: structure of
  `geometry_bench.py` (jobs / run / Bench / tables / figures, `--confirmatory`,
  `--from-extract`, `--analyse-only`), `RESERVED_BASE = None` until the owner fixes the
  range, guards against the used ranges 7000–7029 and 9000–9059, curves drawn in cm and
  in s with both speeds overlaid; the analysis of the extract runs without PyTorch.
* **Checks.** Arrivals are seeded per belt from the mean inter-arrival only, so they are
  identical at both speeds (80 equal instants on seed 3100); `geometry.entry_length=50`
  moves the laser zone, the workspace and the head's home together. Smoke test on
  development seeds 3100–3101, one cell (`--tag _belt_smoke`): 192 episodes, 0 visible-
  stretch violations, exact identity at ℓ = 0. The confirmatory design was NOT run.
* **Note.** The predictions in §7.1 come from the extract written before 0.7.26
  (observable tie-break); the C_3(10 cm) column of the bench is the check.
* **Pending (owner, §7.6).** Reserved seed range, margin δ, whether the network runs,
  and the go.

## 0.7.26 — 2026-11-23 — Observable tie-break, MILP blocking witness re-scoped, Paper 1 draft 0.3

* **Why.** The owner's review of Paper 1 draft 0.2 (2026-11-23) found two technical
  problems: (1) `nearest_job`, `nearest_head` and the candidate ranking of the
  known-parts search broke exact ties by the real arrival time `r_j`, which the `belts`
  and `zone` information models do not observe (a part that waited in the hidden queue
  shows no trace of it); (2) the time-indexed MILP's blocking witness (M14/M15 of the
  formulation, M11/M12 of the paper) used a non-strict threshold `x ≥ b − vΔ`, which
  charges a full blocking event one step early when a part at exactly `b − vΔ`
  advances and starts on reaching `b` (S_j = B_j, zero cost in the continuous model),
  an error that does not vanish with Δ. Plus several claims to align with the tables.
* **What, tie-break.** `FastSim.observed_since(j)` (the instant a part became
  observable at the cell's level; `rl2.features.observed_since` delegates to it);
  `fastsim/policies.py` keys `KEY_OBSERVED` (default) and `KEY_ARRIVAL` (legacy, for
  the audit), `nearest_job`/`nearest_head` take a `key`; `SearchTeacher(tie_key=...)`
  uses the same key for its candidate ranking; engine: `Part.observed_time` recorded by
  `Simulation._record_observation` after admission (level `full`: also the queue), and
  `DecisionContext.observed_key`, used by `policies/nearest_job.py`, `nearest_head.py`
  and the rollout's ranking. `fifo` and `weighted_score` (not used in the papers) still
  use `r_j`. Tests: `tests/test_tiebreak.py` (shifting the candidates' arrival times
  changes no decision on the fast simulator or the engine; the legacy key does; the
  observation instant matches the level). Engine/fast-simulator agreement tests pass.
* **Audit** (`experiments/tiebreak_audit.py`, `docs/methods/tiebreak_audit.md`;
  seeds 2000–2029, eight regimes, B and C): the observable key changes 0.1–2.5 % of the
  rules' decisions and 0–0.5 % of the search's; paired cost new vs legacy, pooled:
  `nearest_job` +0.24 ±0.23 % (B), +0.29 ±0.25 % (C); `nearest_head` +0.35 ±0.16 /
  +0.18 ±0.20; search +0.04 ±0.07 / −0.01 ±0.05; by regime within ±0.6 points at the
  real load and up to +1.5 at the heavy load with Poisson arrivals (the legacy key,
  informed by `r_j`, was slightly better). **The phase-4 confirmatory run (0.7.13) used
  the legacy key** for the rules and the known-parts search; its comparators were
  therefore slightly favoured, the network's margins are conservative by those amounts,
  and the reserved seeds were not rerun. Papers 1 and 2 state this.
* **What, MILP.** `docs/formulation/formal_formulation.tex`: (M14) and (M15) now use the
  strict threshold `x_j > b − v_kΔ` (the CP-SAT decoder already did); the tie-break
  paragraph updated; note that (M15) and the penalized objective are specified, not
  implemented, so for the blocking penalty the model is an approximate discretization
  not validated against the continuous definition. Paper 1 no longer calls it an exact
  reference for that objective.
* **Paper 1 draft 0.3** (`docs/papers/paper1/`): abstract, contributions, §3.4 (the
  "85–95 % flow, insensitive to the weights" claim removed), §3.6, §4.1 (return to be
  maximized, weighting by n, observable tie-break and the audit), §4.3 (the selection
  criterion is the mean paired ratio, not a ratio of means; expert scores unpenalized
  flow, PPO maximizes the return, both evaluated by the criterion), §5.3 and Table 2
  (enumeration only for n ≤ 8, heuristic upper bound from n = 10), §5.4/§10/§11
  (light traffic: the rules coincide and are within I* of the offline optimum, not
  optimal), §7.2 and Figure 4 (the 5 ms budget is met unloaded and exceeded under
  load), §9.2 ("no difference from its expert was detected", with the +2.0 ±2.8 cell
  in C), Appendix A (strict witness, the review's case, implementation status),
  Figure 3 legend moved below the panels, `extra.bib` DAgger authors as in PMLR, a
  footnote flagging the Lenstra theorem numbers as pending. PDF rebuilt.
* **Same day, after the owner's reading of the 0.3 PDF:** the figures overlapped
  (legends over the curves in Figure 2, over the NJ curve in Figure 4, the right-hand
  x label of Figure 2 cut) — every legend now sits below its panel, shorter labels,
  taller figures (`experiments/paper1_figures.py`); and the PPO baseline in progress
  is stated explicitly on the title page, in the abstract, in a framed paragraph of
  §8.4 with the frozen protocol, and in the conclusions. 28 pages.
* **Owner's three-task list (same day):** (1) in Paper 1, every result now says which
  tie-break it was obtained with (a "tie-break convention" paragraph opening §8, a
  sentence in the protocol of §9.1, and a note in the captions of Tables 3–6, 8 and 9:
  everything except the audit used the legacy arrival-time key); (2) the Lenstra et
  al. citation was checked against the CWI copy of the 1975 preprint (Mathematisch
  Centrum BW 43/75, OCR text): it proves the NP-completeness of
  n|1|r_j≥0,w_j=1|Σw_jC_j and n|1|r_j≥0|L_max by reductions from KNAPSACK (Theorem 4(a)
  and 4(c)), i.e. ordinary NP-hardness, with no 3-PARTITION and no "strong"; the strong
  NP-hardness our Proposition 4 relies on is attributed to the published 1977 version,
  which could not be read. Paper 1 §5.2 and Appendix B, and the formulation document's
  complexity proof, now say exactly that and state that the reduction inherits the
  hardness of its source (strong if the source is strongly NP-hard); the owner must
  check the 1977 version before submission.
* **Owner checked the published 1977 version (same day):** 1|r_j|ΣC_j is strongly
  NP-hard by its Theorem 5 (p. 357, 3-PARTITION; unary encoding p. 359); for
  1|r_j|L_max the paper gives Theorem 4(g) (p. 356, KNAPSACK, ordinary), so Paper 1's
  Appendix B now states and proves the classical 3-PARTITION reduction for L_max
  (Lemma B.1, the owner's construction: unit separator jobs in fixed slots, gaps of
  length B filled by exactly three jobs). Proposition 4 keeps the strong sense for both
  objectives; §5.2, Appendix B and the formulation document cite the exact theorems.
  The provisional footnote is gone.

## 0.7.25 — 2026-11-23 — Paper 1 draft 0.2: the formulation in the body, proofs and MILP in appendices

* **Why.** The owner's reading of draft 0.1 (2026-11-23): the paper left the formal
  formulation out, although it is the plan's first contribution; it should read "this is
  the problem, this is how we formulate it, these are its properties, and this is how we
  solve it and with what result". Decision (a): formulation complete in the body, long
  proofs, the detailed MILP and the special cases in appendices. The owner also asked
  for a PPO baseline on the new representation, run in parallel in its own session
  (branch `ppo-baseline`), and to call the method "expert iteration with scenario
  search".
* **What.** `docs/papers/paper1/paper1.tex` draft 0.2 (26 pages): §3 The problem
  (system and notation, belt odometer and entry coordinates, decisions and constraints
  C1–C6, objectives with the blocking penalty, pre-positioning and P^sim, problem
  statement), §4 The online problem (SMDP and the return identity, information model
  with the three levels, the general stochastic problem over θ, load and stability),
  §5 Properties (structure, strong NP-hardness, the lower bounds in words with their
  measured quality, the light-traffic theorem and the three counterexamples), §6–§7 the
  method renamed (expert = scenario search; the RL relation stated), §8 development
  results plus §8.4 on the earlier PPO/BC agents and the pending PPO baseline, §9
  confirmatory, §10 discussion ("what the formulation buys"), §11 conclusions;
  Appendix A time-indexed MILP (M1–M12 with the blocking witness), B proof of the
  complexity proposition, C lower bounds with proofs (L1, L2, H, L3, L4, L5, L6), D
  special cases (counterexamples, light-traffic theorems with proofs, the delay
  proposition). Ported from `docs/formulation/formal_formulation.tex` and
  `docs/methods/theory.md`; the earliest-start bound is written $\underline s_j$ to keep
  $e_j$ for the entry time. New `docs/papers/paper1/extra.bib` with seven classical
  references added from memory (Lenstra 1977, Schrage 1968, Baker 1974, Ross 2011,
  Anthony 2017, Silver 2018, Schulman 2017), kept apart from the verified
  `references.bib` and flagged for checking before submission.

## 0.7.24 — 2026-11-23 — Paper 1 draft 0.1 in LaTeX (`docs/papers/paper1/`)

* **Why.** The owner closed Paper 2 at draft 0.6 and asked for Paper 1 (2026-11-23).
  Per the plan's review notes and the owner's decisions of 2026-11-22, the method of
  Paper 1 is the sampled-future look-ahead plus the learned policy that amortizes it;
  the known-parts rollout is the reference it improves on, and the exact pruned search
  of the original plan is named as not built.
* **What.** `docs/papers/paper1/paper1.tex` (draft 0.1, 14 pages): problem with
  endogenous blocking and the reference information model, properties (strong
  NP-hardness with N in the input, belt order not dominant, lower bounds, light
  traffic; proofs cited to the formulation and theory documents), the two look-aheads
  (known-parts rollout; sampled futures scored on every part) with the teacher design
  study (`rl2.md` §6.4), the token/Transformer policy and expert iteration (`rl2.md`
  §2–§3, §6.3), the training distribution (§6.1), development results (scaling ladder
  §8.4, sampled-future continuation §8.5, design-mode and bursts continuations
  §8.6/§8.8, burst-aware teacher §8.9), the confirmatory evaluation of phase 4
  (`confirmatory.md` §9.3: network vs NJ, vs the known-parts search, vs its teacher, in
  B and C, with per-decision latencies), discussion and limits (no optimality gap;
  rollout K = 6 instead of the exact search; one 3-belt cell in the confirmatory run).
  Generator `experiments/paper1_figures.py`: extracts
  `data/phase4_search.csv` (1 920 episodes of the Paper-1 policies and the rules in B
  and C) and `data/phase4_latency.csv.gz` (166 810 per-decision latencies) from the
  phase-4 `bench.json`, copies the teacher-variant and teacher-futures result files
  (`data/teacher_*.json`), reads the versioned training logs `models/*_training.json`;
  tables `regimes`, `confirm_B`, `confirm_C`, `latency`, `ladder`, `sf_runs`,
  `variants`, `futures`; figures `training`, `confirm`, `latency`. All numbers in the
  text come from those tables or from `rl2.md`/`confirmatory.md`; PDF versioned.
* **Not in the draft** (owner's call): the hindsight/optimality reference, the 36-cell
  bench and the budget ablations of the original plan (second phase-4 run, new reserved
  seeds), and an expert-iteration reference in the bibliography (none verified in
  `references.bib`; the scheme is described without citation).

## 0.7.23 — 2026-11-23 — Paper 2 draft 0.6: three text corrections before sharing

* **Why.** The owner's reading of draft 0.5 (2026-11-23): (1) "3.33 s is enough" in the
  abstract overstated the result; the practical result is the measured cost (median
  0.05, max 0.44 s/part in the 18 cells), one cell retained only 71 %, and the
  conservative 90 % criterion certified 2–7.5 s with one cell unresolved; (2) §7.1 did
  not match Table 7: summarized over the cells that qualified, the conservative range is
  2–5 s with v_x = 10 and 5–7.5 s with v_x = 5 plus the unresolved case, and the
  abstract had to say that the doubling of the gain with v_x halved was observed in the
  nominal-load Poisson subset of §6.4; (3) the queue traces are compatible with the
  transmission of the saving at the nominal load, they do not confirm it, and at the
  high load the stop cascade is a hypothesis.
* **What.** `paper2.tex` draft 0.6: abstract, contribution 5, §6.5, §7.1 and §7.3
  reworded accordingly; no number changed. PDF rebuilt. The owner would share this
  version with co-authors.

## 0.7.22 — 2026-11-23 — Paper 2 draft 0.5: rewritten around the geometry bench

* **Why.** Owner's decisions on the bench results (2026-11-23): 1(a) rewrite Paper 2
  around the geometry bench, with the phase-4 study as the reference cell where
  seeing and acting are separated; 2(b) headline the practical consequence of 3.33 s
  of anticipation (median 0.05, max 0.44 s/part against 10 s in the 18 cells worth
  sizing), report 1.7 × the mean move as the best predictor among those tested
  (1.02 s against 1.13 s for a constant, too small a margin for the headline), and
  the conservative 90 % range apart; 3(a) keep the pre-specified threshold (G ≥ 1 s/part,
  lower bound ≥ 0.5) and leave the other 38 of 56 cells undetermined with their
  estimated effect. Decisions 4–6 (network's role, high-load mechanism, high-load
  precision) wait for the next reading; the economic translation stays deferred.
* **What.** `docs/papers/paper2/paper2.tex` draft 0.5 (20 pages): new abstract,
  introduction and contributions; cell section generalized (spacing d, v_x, v_y,
  τ_cross = 2d/v_y, τ_head renamed τ_cross throughout, also in the generated
  reference-cell tables); design split into the reference-cell study and the geometry
  bench with its frozen measures and criteria; checks extended (run-time mask check,
  H0 exact, both extracts); §5 the reference cell (contrasts, coverage curve, direct
  7.5 vs 10 cm, paired robustness; levels, pooled-share and heavy coverage tables
  dropped from the paper, their numbers kept in the text); §6 the geometry bench
  (where sizing is worth it, how much anticipation with the practical and the
  conservative answers, what predicts it, the x axis, mechanism, entry length, the
  network); discussion and conclusions rewritten. New generator
  `experiments/paper2_bench_figures.py` (tables `bench_gains`, `bench_sufficient`,
  `bench_predictors`, `bench_entry`, `bench_net`; figures `bench_curves_poisson`,
  `bench_curves_burst`, `bench_normalized`, `bench_knee`) from the versioned extract.
  PDF rebuilt, every page checked visually.

## 0.7.21 — 2026-11-23 — Geometry bench run on the reserved seeds 9000–9059 (`geometry.md` §6)

* **Why.** Owner's go (2026-11-23) on the frozen protocol of 0.7.20.
* **What.** `python experiments/geometry_bench.py --tag geometry_bench --confirmatory
  --workers 40`: 179 520 episodes, 1 139 s wall (12.0 CPU-hours). **Seeds 9000–9059
  are now used once**, like 7000–7029. Per-episode extract versioned at
  `docs/papers/paper2/data/geometry_bench.csv.gz` (10 MB); `--from-extract` regenerates
  the tables from it (checked identical to the `bench.json` tables). Figures
  `docs/methods/figures/geometry_bench_*.{pdf,png}`. Two analysis fixes after the run,
  before any result was written: the regime list was built from the first 20 000 rows
  (now from all), and the rule's predictor analysis counts the τ_cross = 2.4 s triplet
  once (identical cells by H0).
* **Results** (`geometry.md` §6). Checks: 0 mask violations; H0 exact (≤ 2.6 · 10⁻¹³ s)
  in all eight blocks, so the rule has 56 distinct cells. Coverage worth sizing (G(10 s)
  ≥ 1 s/part, lower bound ≥ 0.5) in 18 of 56 cells for the rule (13 of 28 Poisson, 5 of
  28 bursts); elsewhere undetermined as pre-registered. Knee in the useful cells:
  0.5–5 s point, 2.0–7.5 s conservative; 3.33 s costs at most 0.44 s/part against 10 s.
  **v_x is a first-order factor:** halving it doubles the gain (1.0–1.3 → 2.1–2.4 s/part
  at nominal Poisson) and moves the knee from 1.5–3.3 to 3.3–5 s. Predictors (18 cells,
  leave-one-out MAE): τ_cross 1.53 s, a τ_cross 1.75, **b τ_pp 1.02** (b = 1.74), b′ (t_x +
  t_y) 1.18, constant 1.13: the moves the head makes predict the coverage better than
  the transverse crossing, which is the worst candidate; no predictor is precise. H5:
  implied per-part saving within 1.0–2.0 × the move duration in all ten useful
  nominal cells (mechanism supported), 2.1–9.5 × at the high load (not supported;
  interpretation dropped there). Network: useful in 17 of 72, noisier, no predictor
  works; `when_idle(τ) − when_idle(0)` equals G at the nominal load; it uses upstream
  visibility with the same actions at the high load with bursts (+0.6–1.4 s/part). H3
  holds (ΔG within ±0.26 / ±0.46 s/part, never significant); the longer entry adds
  8.33 s of transit plus 2.0–4.9 s of belt stops growing with the load.

## 0.7.20 — 2026-11-23 — Geometry bench: frozen confirmatory protocol and script (`experiments/geometry_bench.py`), not yet run

* **Why.** The owner's reading of the pilot (0.7.18): the knee in seconds barely moved
  while τ_cross went from 0.8 to 7.2 s, and phase 4's slow head (both axes) needed more
  coverage than the pilot's slow transverse axis alone, so v_x must be varied in a
  controlled way. Instructions: common grid in seconds; v_x 5 / 10 crossed with the
  nine spacing × v_y cells; `nearest_job` primary, network secondary; seconds of
  penalized flow saved per part and the cost of less coverage; for the network also
  `when_idle(ℓ)` vs `when_idle(0)`; 60 paired realizations on seeds 9000–9059 once
  checked unused; grid, minimum useful benefit and 90 % criterion frozen before the
  run; mechanism separated from its interpretation.
* **What.** `docs/methods/geometry.md` §5: the frozen protocol (72 regimes = 9
  geometries × v_x 5 / 10 × load nominal / high by the arrival rates, ρ_p 0.64 / 0.90
  × Poisson / bursts; grid τ_vis ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3.33, 5, 7.5, 10} s; useful
  benefit G(10 s) ≥ 1 s/part with lower bound ≥ 0.5; knee = first grid point with
  G(τ) − 0.9 G(10 s) ≥ 0, point and Bonferroni-conservative (z 2.773); predictor
  comparison by leave-one-out error of knee = τ_cross / a τ_cross / b τ_pp / constant;
  H5 on the queue mechanism with the number of unfinished parts at each move).
  `experiments/geometry_bench.py`: the bench, its analysis, figures and a gzip
  per-episode extract; refuses the reserved ranges without `--confirmatory` and
  refuses to overwrite a confirmatory `bench.json`. `src/fastsim/core.py`:
  `prepos_wip_mean` in the trace statistics. §4.1 of `geometry.md` reworded: "gain per
  move" is an aggregate ratio; the queue transmission is an interpretation to be
  tested (H5). Seed check: no episode result used 9000–9059; `tests/test_rl2_search.py`
  draws *general* cells with `general_config(9000 + i)`, different cells.
* **Smoke test** on seeds 3000–3001: 5 984 episodes in 42 s on 40 workers (0.44
  CPU-hours); H0 exact for the rule in all eight v_x × load × arrivals blocks; 0 mask
  violations; tests pass. Estimated confirmatory run: 179 520 episodes, ≈ 20 min.
  **The confirmatory run waits for the owner's go** (the reserved seeds are used once).

## 0.7.19 — 2026-11-23 — `models/rl2_gen_l.pt` versioned through Git LFS

* **Why.** The size-L model (157 MB, final checkpoint of run `rl2_gen_l`, iteration 1,
  `rl2.md` §8.4) exceeds GitHub's 100 MB blob limit and had stayed outside the
  repository since 0.7.5. The owner decided (2026-11-23) to keep it in the repository
  through Git LFS rather than an external store.
* **What.** `git-lfs` 3.4.1 installed on the workstation; `.gitattributes` tracks
  `models/rl2_gen_l.pt` with the LFS filter; the file is committed as an LFS pointer
  (object sha256 `7bff321f5d2da66652380578e9ea0aacada673ef87604d259848ebc054c176e2`).
  Per-iteration checkpoints stay ignored. The other models (≤ 27 MB) stay as ordinary
  blobs. **Cloning:** install git-lfs, then `git lfs install` once and `git lfs pull`
  in the clone to fetch the 157 MB object; without git-lfs the file is a 130-byte
  pointer and `rl2.model.load` fails on it.

## 0.7.18 — 2026-11-23 — Geometry pilot: does the head's crossing time predict the camera coverage? (`experiments/geometry_pilot.py`)

* **Why.** The owner's brief of 2026-11-23 proposes extending Paper 2 with a bench
  over belt spacing × transverse head speed (36 regimes, seven coverages at
  fractions of τ_cross = (y_max − y_min) / v_y), to test whether τ_cross predicts
  the sufficient coverage across geometries. Agreed first step (owner): a pilot with
  `nearest_job` as the primary controller, nine geometries, Poisson at the nominal
  load, development seeds, v_x fixed, benefit in seconds per part, hypotheses
  written before the run, confirmatory bench only after the protocol is fixed.
* **What.** `docs/methods/geometry.md`: §1–§3 hypotheses H0–H4 and design, written
  before the run; §4 results. `experiments/geometry_pilot.py`: nine cells (spacing
  6 / 12 / 18 cm × v_y 5 / 10 / 15 cm/s, v_x 10, entry zone 35 cm), coverage at
  0–150 % of τ_cross plus the common lengths 1–10 cm, `none` vs `when_idle`,
  `nearest_job` and the frozen network, seeds 3000–3029; the physical-length check
  (original cell at entry 10 cm vs 35 cm, same visible lengths); checks (i)–(iv);
  tables and figures (`docs/methods/figures/geometry_pilot_*.pdf`). Reserved seeds
  7000–7029 are refused by the script. `src/fastsim/core.py`: trace statistics in
  `summary()` (idle time, time waiting at the meeting point, pre-positioning moves
  with their planned t_x / t_y, revocations, effective anticipation); no change to
  the dynamics (`tests/test_fastsim.py`, `test_observation.py`,
  `test_visible_length_burst.py` pass; the original cell reproduces `rl2_eval.run`
  with Δ = 0).
* **Results** (14 400 episodes, 45 s). H0 exact: the three cells with τ_cross = 2.4 s
  are identical for the rule (|ΔJ| ≤ 10⁻¹³), since with simultaneous axes the
  geometry enters only through spacing / v_y; question 2 of the brief is answered by
  the model's structure. The rule's knee (90 % of the whole-zone gain) lies between
  1.2 and 3.3 s of anticipation in all nine geometries while τ_cross spans 0.8–7.2 s:
  normalization by τ_cross holds only for τ_cross 1.2–2.4 s (knee 1.0–1.25 τ_cross),
  fails on both sides (2.1–3.1 at 0.8 s; 0.2–0.5 at 4.8–7.2 s). 7.5 cm (2.5 s) keeps
  91–105 % of the gain in every geometry. Traces: the number of pre-positioning
  moves is fixed by the head's idle time (31 % → 8 %) and barely depends on the
  anticipation; what grows and saturates at 1.2–3 s is the gain per move (up to
  27 s per move with the slowest head, since a second saved on the head's cycle is
  saved by every part queued behind it); moves last 0.7–3.0 s, below τ_cross, with
  45–58 % y-dominated. Phase 4's slow head (both axes at half speed) needed the whole
  zone; slowing v_y alone does not, so v_x must be a factor of the confirmatory bench.
  H3 holds: the 35 cm entry adds 10.0–10.2 s per part (8.33 s free transit + ≈1.8 s)
  and leaves the gain at a given length unchanged within ±0.5 s. Network: same curves
  as the rule, no degradation at the out-of-range entry; its `none` vs `when_idle` at
  ℓ_v = 0 differ because the mode is a context token, so its baseline should be
  `when_idle` at 0. Seeds 3000–3029 are a lighter draw than 7000–7029 (70.5 vs 77.5
  parts). Recommendations for the protocol in `geometry.md` §4.4.

## 0.7.17 — 2026-11-23 — Paper 2 draft 0.4: direct 7.5 vs 10 cm contrast, paired robustness, two sentences corrected

* **Why.** The owner's reading of draft 0.3 (2026-11-23) asked for three adjustments
  before submission: (1) distinguish the two answers to "is 7.5 cm enough?": the share
  of the pre-positioning benefit retained (each gain against `none` at the same
  length; widening the network's view from 7.5 to 10 cm adds 1.59 ±1.08 points of
  benefit, the 78 %) and the direct cost of the cell already pre-positioning with
  7.5 cm instead of 10 (+0.24 ±0.57 % for the network), which is what the coverage
  decision needs; (2) measure the robustness to the network with the paired per-seed
  difference between the two networks' contrasts, since comparing intervals by the sum
  of their half-widths loses the pairing; (3) the abstract said that at the heavy load
  the head "is never idle and the gain disappears" although C vs B is −1.60 ±1.14 with
  bursts and the nominal head, and §§5.3 and 6.4 described the hidden queue as if the
  network had not trained with `full`, while the training mix has `full` in 15 % of
  the episodes (`rl2.md` §8.6). The economic translation stays deferred (owner).
* **What.** `experiments/paper2_figures.py`: `Bench.gain_loss` (gain at ℓ minus gain
  at 10 cm, per seed), `Bench.net_diff` (frozen − reference contrast, per seed), new
  table `direct` (cost of 5 / 7.5 cm against 10 cm with `when_idle` at both lengths,
  and benefit lost, 8 regimes × 2 policies), table `robustness` with three paired
  columns added and its level columns (frozen vs reference in B and C) moved to the
  new table `robustness_levels` for legibility; `main()` prints the direct contrasts, the `none` level at ℓ vs 10 cm
  and the paired summary. All four numbers of the owner's memo reproduce exactly from
  the versioned extract. Finding: the network's `none` level at 7.5 cm is 1.41 ±1.04
  points *below* its level at 10 cm, which is what separates the two readings.
  Paired robustness: within 1 point in 18 of 24 cells (same point values as 0.7.16,
  since the mean of the paired difference is the difference of the means), max 2.13;
  the paired interval excludes zero in two cells (C vs B: real / Poisson / nominal
  −0.92 ±0.87; heavy / burst / nominal −1.30 ±1.05), where the frozen network obtains
  more from pre-positioning. The 0.7.16 sentence "within ±0.8 points in B in seven
  regimes" was also wrong (six regimes); corrected.
* **Paper** (`docs/papers/paper2/paper2.tex`, draft 0.4, PDF rebuilt): abstract,
  contributions 3–4, §4.1 (training mix), §5.2 new paragraph "Two answers to 'is
  7.5 cm enough?'" with Table `direct`, §5.3 ("seldom idle", "nearly flat", 15 %
  `full`), §5.4 rewritten around the paired difference with the conclusion "pattern
  conserved, differences of magnitude in some regimes", §6.1, §6.4, conclusions.
* **Docs.** `confirmatory.md` §9.2: new bullet on the direct cost, robustness bullet
  rewritten with the paired statement (0.7.16 version kept as a note);
  `docs/plans/papers.md` updated.

## 0.7.16 — 2026-11-23 — Paper 2 draft 0.3: owner's four corrections, related work, robustness claim corrected

* **Why.** The owner's reading of draft 0.2 found four statements that went beyond
  the data: (1) τ_head = 2.4 s covers only the lane change, the along-belt travel can
  reach 35 cm / 10 cm/s = 3.5 s, so "the head is in place whenever τ_v ≥ τ_head" was
  not justified; (2) "widening the view is worth nothing" ignored the −1.31 ±0.74 cell
  (heavy / bursts / slow head), and §5.2 explained B vs A with "waiting for an
  announced part", which `none` cannot do (a decision epoch has a non-empty candidate
  set and no wait action, `src/fastsim/core.py`); (3) the two networks were called
  "independently trained" while `rl2_gen_s_b` continued the training of
  `rl2_gen_s_d` (`rl2.md` §8.8), and "the same set of significant cells" had an
  exception; (4) the sign convention was stated both ways (§3.3 "negative = first
  configuration better", Figure 2 "negative = second treatment better"). The owner
  also asked to go on with the paper, so the related-work section proposed earlier
  was written.
* **What** (`docs/papers/paper2/paper2.tex`, draft 0.3; PDF rebuilt, 14 pages).
  * §2.4 and §5.1: τ_head presented as the lane component of the travel, with the
    along-belt component (up to 3.5 s) stated and the knee of the curve read as
    evidence that it seldom matters at pre-positioning decisions, not that it cannot.
  * §5.2: the "waiting" channel removed; B vs A can act only through the order of
    the zone parts; the heavy / bursts / slow-head cell kept as the one where the
    wider view helped.
  * Networks: "reference network" defined as the checkpoint the frozen network was
    retrained from (a check against the retraining, not an independent run) in the
    contributions, §3.1, §4.4 and the conclusions.
  * §3.3: the contrast written once as Δ = 100 (mean of J_X / J_Y − 1) for "X vs Y",
    negative favours X, the count is the realizations with J_X < J_Y; Table 2,
    Figure 2 and Table 7 captions aligned.
  * New §2 "Related work" from `docs/literature/` (closest_papers.md,
    review_assessment.md, the evidence map): moving-conveyor picking (S01–S05),
    moving-target interception (S06–S08), blocking in conveyor systems (S10, S11),
    learned dispatching (M02, M04, S12–S21); only statements verified in those notes.
* **Erratum found while checking (3).** The claim "the reference network reproduces
  every contrast within ±0.5 points" (confirmatory.md §9.2, CHANGELOG 0.7.13, paper
  drafts 0.1–0.2 "within ±1 point, ±0.5 in 20 of 24 cells") was wrong: recomputed
  from the extract, the two networks differ by < 0.5 points in 12 cells, < 1 point in
  18, at most 2.1 (heavy / Poisson / slow head, B vs A: −1.34 ±1.66 vs +0.79 ±2.10),
  always by less than the sum of the two half-widths; same sign wherever either
  interval excludes zero; same significant cells except C vs B at heavy / bursts /
  nominal. Corrected in confirmatory.md (with a note), annotated in the 0.7.13 entry,
  and stated this way in §4.4 of the paper. The conclusion (the contrasts survive the
  retraining) stands; the precision claimed for it did not.

---

## 0.7.15 — 2026-11-23 — Paper 2 draft 0.2: figures redesigned, headline softened, data extract versioned

* **Why.** The owner's review of draft 0.1: the figures were unreadable (eight cramped
  panels, overlapping secondary axes); the headline overstated the evidence (the data
  support the head's travel time as a *reference* for sizing the coverage, not
  ℓ_v ≥ τ_head · v_belt as a universal minimum; 14.4 cm for the slow head is a
  prediction beyond the 10 cm grid); the network's 78 % at 7.5 cm (same-length
  baseline, the frozen protocol) must be the primary estimate and the 97 % (pooled
  baseline) a sensitivity analysis, since the baseline changes the answer to "how
  much is enough"; "seeing without acting is worth nothing" must become "with the
  same actions, widening the view produced small effects in the regimes tested", which
  keeps the heavy / bursts / slow-head cell where the wider view did help (−1.3 ±0.7);
  the PDF stays versioned and the confirmatory data the generator uses must be in the
  repository so that anyone with the checkout can rebuild the figures and tables.
* **What.**
  * `experiments/paper2_figures.py`: reads the versioned extract
    `docs/papers/paper2/data/phase4_info.csv` (the 15 360 per-episode rows of the
    information bench: regime, seed, configuration, policy, cost, flow, blocking,
    parts), written from `bench.json` with `--bench`; figures redesigned:
    `coverage_real.pdf` and `coverage_heavy.pdf` (one row of four panels per load,
    CI bands instead of error bars, a single x axis labelled in cm and seconds, τ_head
    as a dashed line with its label, or an arrow when it lies beyond the axis, the
    stretch beyond the entry zone shaded), `contrasts.pdf` (filled markers where the
    interval excludes zero, nominal and heavy regimes separated). Tables unchanged.
  * `docs/papers/paper2/paper2.tex` (draft 0.2): abstract, contributions, §4.1 title
    and B vs A paragraph, §4.2 reading and §4.2.1 (now "sensitivity of the share to
    the baseline"), §4.4, §5.1 (now "a reference in seconds", with the limits of the
    evidence stated), §5.2 title, conclusions; new reproducibility item in §3.4; the
    heavy-load coverage figure added to §4.3. PDF rebuilt.
* **Checked.** The tables are byte-identical to 0.7.14 apart from the provenance
  comment; the shares quoted in the text (78 / 99 % at 7.5 cm, 64 / 84 % at 5 cm;
  pooled 16 / 53 / 81 / 97 %; reference network 76 / 97 % same-length, 97 / 112 %
  pooled) were recomputed from the extract.

---

## 0.7.14 — 2026-11-23 — Paper 2 draft in LaTeX (`docs/papers/paper2/`)

* **Why.** With phase 4 closed (0.7.13) the owner decided to write Paper 2 first, in
  LaTeX inside the repository, starting from the confirmatory results (A/B/C/D, the
  coverage curve, with and without pre-positioning at each visible length), with the
  coverage expressed in **seconds of anticipation** at the belt speed and its
  equivalent in centimetres for the nominal belt, and the central result stating how
  far the camera must see and why. The economic translation is deferred.
* **What.**
  * `experiments/paper2_figures.py`: reads `results/experiments/phase4/bench.json`
    and writes the paper's figures (`figures/coverage_curve.pdf`: gain of
    pre-positioning against the anticipation ℓ_v / v_belt with the head's travel
    time τ_head marked, one panel per regime; `figures/contrasts.pdf`: the network's
    B vs A, C vs B, D vs B on the eight regimes) and tables (`tables/*.tex`: regimes
    and physical scales, levels vs `nearest_job` in B, contrasts, coverage curve at
    the real and the heavy load, share of the whole-zone gain with the same-length
    and the pooled `none` baseline, robustness with the reference network), so that
    every number in the paper comes from one computation on the versioned data of
    phase 4. The generated files are versioned because `results/` is not.
  * `docs/papers/paper2/paper2.tex` (+ `paper2.pdf`, built with `latexmk -pdf`;
    bibliography `docs/literature/references.bib`): *How far must the camera see?
    The value of upstream visibility and pre-positioning in a shared-head conveyor
    cell*. Sections: the cell and its information model (observation levels,
    pre-positioning `when_idle`, visible length, anticipation τ_v = ℓ_v / v_belt and
    the head's travel time across the conveyors τ_head = 24 cm / v_y); the design
    (two unrelated controllers, eight regimes, paired reserved realizations used
    once, pilot vs confirmatory, causality and simulator checks); results (seeing is
    worth nothing, acting 3–7 %; the coverage curve read against τ_head; the hidden
    queue; robustness to the controller); discussion (design rule ℓ_v ≥ τ_head ·
    v_belt, why seeing alone is worth nothing here, bursts and slow heads, limits,
    economic translation deferred). Draft 0.1; every figure quoted in the text was
    checked against `bench.json`.
  * `.gitignore`: LaTeX build files under `docs/papers/`.
* **Decisions recorded in the paper.** The share of the whole-zone gain is reported
  as frozen in the protocol (`when_idle` vs `none` at the same length) and, for the
  network, also with the mean `none` level over the six lengths as baseline, because
  the network's `none` levels move by up to 1.3 points between lengths (the B vs A
  null's noise); with the pooled baseline the network's shares (16/53/81/97 % at
  1/2.5/5/7.5 cm, real load, nominal head) match the rule's (24/62/84/99 %). The
  pilot's "97 % at 5 cm" is quoted as not confirmed. The heavy coverage table carries
  no share (the whole-zone gain is within noise).

---

## 0.7.13 — 2026-11-22 — Phase 4: confirmatory evaluation on the reserved seeds (`experiments/rl2_eval.py`)

* **Why.** The owner confirmed the phase 4 protocol drafted in
  `docs/plans/papers.md` (network `rl2_gen_s_b` frozen, the teacher contrast of
  0.7.12 closed): one run on the reserved seeds 7000–7029, with the two priority
  contrasts of the papers and the coverage curve, so that the results of the pilots
  (0.7.9–0.7.12, all on development seeds) are confirmed on realizations never seen.
* **What** (`experiments/rl2_eval.py`): one invocation runs, on the eight regimes of
  the pilot (real / heavy × Poisson / burst × nominal / slow head) and 30 paired
  realizations per configuration: (i) *Paper 2*, treatments A/B/C/D and the coverage
  grid (`visible_length` 0, 1, 2.5, 5, 7.5, 10 cm with `none` and `when_idle` at each
  length) for `nearest_job`, `nearest_head`, the frozen network and the 0.7.9
  reference network; (ii) *Paper 1*, in B and in C, the known-parts search over
  `nearest_job`, the sampled-future teacher over the network (8 futures, 30 s, w_pen
  0) and the network alone, each driving its own episode, plus the per-decision
  latency of every policy measured on the decisions of the network's episode (same
  states, one thread); (iii) the fixed-step engine at `dt = 0.001` with the rules in
  B and C on every regime against the fast simulator (the network has no engine
  adapter; it is validated through the fast simulator, whose acceptance is
  `docs/methods/fastsim.md` §4). Outputs `results/experiments/phase4/bench.json` and
  `tables.md`; the script refuses to overwrite a run on the reserved seeds. Smoke
  test on seeds 2000–2001 before the launch.
* **Results** (`docs/methods/confirmatory.md` §9; full tables in
  `docs/methods/phase4_tables.md`; run of 38 min on the workstation). *Paper 2:* B
  vs A is a null result in every regime (network: −1.3 to +0.9 points, intervals
  ±0.6–1.7); C vs B is −7.4 / −3.6 / −2.8 / −4.3 % at the real load (nominal, slow
  head, bursts, both) and within ±1.3 points at the heavy load; D vs B is worth
  −2.7 ±1.4 and −3.7 ±1.9 % at the heavy load with Poisson arrivals only; the
  coverage curve saturates at 7.5 cm (2.5 s of belt travel, the head's travel time
  across the conveyors) with the nominal head and needs the whole zone with a head
  twice as slow, i.e. more coverage than the pilot suggested (half the zone gives
  64–84 % of the gain, not 97 %). *Paper 1:* the network alone beats `nearest_job`
  in 7 of 8 regimes (−2.0 to −7.3 %), beats the known-parts search in every regime
  (1.5–3.9 points) and equals its sampled-future teacher everywhere (within ±0.8
  points) at 10.8 ms per decision (p50, one thread, machine fully loaded) against
  0.4–1.9 s for the teacher. *Engine check:* identical order in 96 % of the 960
  episodes, mean |ΔJ| 0.03–0.06 %, the rest near-tie flips as in `fastsim.md` §4.
  The reference network reproduces every contrast within ±0.5 points
  *[corrected in 0.7.16: within the noise; < 1 point in 18 of 24 cells, at most 2.1]*.

---

## 0.7.12 — 2026-11-22 — Teacher futures from the cell's arrival model; contrast Poisson vs burst-aware teacher

* **Why.** The reading of 0.7.11 (the network trained on bursts cannot exceed a
  teacher whose sampled futures are Poisson) was plausible, not shown. The owner
  decided to measure it directly before any training from scratch: the Poisson
  teacher against a teacher whose futures follow the burst model of the cell, on the
  same realizations, over the same network (`rl2_gen_s_b`, frozen) and with the same
  search budget (8 futures, 30 s window, w_pen 0).
* **What** (`rl2.search`, `docs/methods/rl2.md` §6.4): `sample_scenarios(...,
  futures)` and `SearchTeacher(futures=...)` with `"poisson"` (the teacher of every run
  so far, default) or `"cell"`: on a conveyor with an on/off source the future is an
  on/off Poisson source with the cell's `on_fraction` and `on_mean` as statistical
  knowledge, started in the stationary state; on a Poisson conveyor both choices draw
  the same arrivals from the same stream, so every earlier result is reproduced with
  the default. Jobs carry `futures`; `_extra` also applies a job's `extra` overrides.
  The tokens are unchanged: the network is still not told the burst parameters.
* **`experiments/rl2_teacher_variants.py`:** variant option `:fcell`; `--groups`
  (default the four validation groups; the script had failed since 0.7.11 because its
  seed table did not know `real_burst`), `--extra key=value` applied to every cell
  (used for a stronger burst setting), tables generic in the groups.
* **Tests:** `tests/test_rl2_search.py` gains three tests (cell futures equal Poisson
  futures on Poisson cells and with `on_fraction` 1; on a bursty cell they keep the
  mean rate, are over-dispersed, reproducible and keyed; the teacher with cell futures
  runs, is deterministic and differs from the Poisson one).
* **Results (`rl2.md` §8.9; 50 seeds per group, design mode).** On Poisson cells
  both teachers are the same policy end to end. On the validation bursty cell the
  burst-aware teacher is **worse** than the Poisson one (paired +0.52 ±0.49 %); with
  strong bursts (on-fraction 0.2, on-mean 40 s) the two are equal (−0.14 ±0.38 % on
  the heavy cell, +0.07 ±0.57 % on the real cell) and neither beats the network
  alone. The teacher-ceiling explanation of 0.7.11 is therefore **not supported**: the
  limit on bursty cells is the problem (with strong bursts the head is idle or
  saturated, and sequencing gains 1–2 % at most), not the teacher's arrival model.
  The stationary start of the informed future adds variance that 8 futures do not
  average out. Decision recorded in `docs/plans/papers.md`: Poisson futures remain
  the teacher of record, `rl2_gen_s_b` stands, no training from scratch on this
  ground.
* **Not changed:** the frozen network, the training mix, the validation seeds, the
  cell model, the tokens.

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
* **Results.** 4 h 05 min, stopped by patience after 3 iterations, best iteration 1
  (score −3.82 % over four groups; −6.2 / −2.9 / −2.7 / −3.5 % real / heavy / general
  / real_burst). Head to head with `rl2_gen_s_d` on the validation seeds: +0.5 / −0.4
  / +0.2 / +0.1 %, i.e. **indistinguishable**, also on the bursty cell (the reference
  already scored −3.6 % there). The information bench with the new network reproduces
  every 0.7.10 conclusion; the hidden-queue contrast at the heavy load with a slow
  head becomes −6.4 ±2.4 % with a model trained in that regime. Reading: the network
  is at its teacher from iteration 1 and the teacher's futures are Poisson, so the
  training mix cannot add what the teacher does not model. Committed:
  `models/rl2_gen_s_b.pt`, `models/rl2_gen_s_b_training.json`. Choice of the network
  to freeze for phase 4 left to the owner (`rl2.md` §8.8 recommends `rl2_gen_s_b`,
  trained on the declared scope, with `rl2_gen_s_d` as the 0.7.9 reference).
* **Decisions (owner, 2026-11-22, after the results; `docs/plans/papers.md`).**
  (1) **Network frozen for phase 4: `rl2_gen_s_b`, iteration 1**, because its
  training covers the bursts and slow heads in the scope and it is statistically equal
  to `rl2_gen_s_d` on development seeds; `rl2_gen_s_d` stays as the 0.7.9 reference.
  (2) The teacher-ceiling explanation is to be tested, not assumed: Poisson teacher vs
  burst-aware teacher on the same realizations with the same search budget, before any
  training from scratch. (3) The repeated bench is presented as robust to the change
  of network, pending the reserved seeds. (4) Phase 4 starts with two contrasts:
  network alone vs sampled-future teacher (quality and latency) and with vs without
  pre-positioning at each visible length. A phase 4 protocol (network, seeds
  7000–7029, simulator, regimes, treatments, outputs) is drafted in `papers.md` for
  the owner to confirm; nothing has been run on the reserved seeds.
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
  the commit until the user decides (Git LFS or an external store) [added through Git
  LFS in 0.7.19].
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
