# Assessment of the systematic evidence map (2026-11-19)

The evidence map in [systematic_review_2026-11-19.md](systematic_review_2026-11-19.md) was
produced by an external AI tool from the prompt prepared in this project (PRISMA 2020
method). This document records (1) the verification of its references, (2) a
point-by-point check of its technical remarks against our model and code, and (3) the
resulting action list. Nothing in the model or code was changed because of it yet.

## 1. Reference verification

All 26 DOIs / arXiv ids were resolved on 2026-11-19 against the Crossref REST API and
the arXiv API (script kept outside the repository; it only queried the identifiers).

| result | references |
|---|---|
| DOI resolves, title / authors / venue match | S02–S18, S20, S21, M01–M05 (24) |
| arXiv id resolves, title matches | S02 (1912.08009), S08 (2403.04917), S16 (2303.14693), S19 (2010.12367) |
| no DOI in the review; found by bibliographic search | S01 → `10.1016/S0736-5845(99)00041-1` (Mattone, Divona, Wolf, 2000), added to `references.bib` |
| not found / mismatch | **none** |

Minor: S07 and S10 have Crossref (online-first) years 2020 and 2024; the review uses
the issue years 2022 and 2025 — both are correct.

**Scope of the verification:** it establishes that the papers *exist* with the stated
metadata. It does **not** verify what the review says about their content; that
requires reading at least the closest works (S02, S03, S04, S08, S10, S11), which is
also needed for the paper.

**Status of the review itself:** by its own account it is an access-limited,
PRISMA-*informed* evidence map (native database searches failed, no hit counts, no
complete screening ledger, single AI reviewer). It is useful for positioning, **not**
publishable as an SLR. A paper would need export-based searches (Scopus / WoS) with
complete PRISMA counts.

## 2. Technical remarks checked against our work

| review point | status in our work | action |
|---|---|---|
| **A. Belt order is not automatically processing order**: head-only actions are a restricted policy class; measure the restriction gap. | Correct. `nearest_head` and the RL action space only serve conveyor heads. The belt-order observation of phase 2 is empirical (metaheuristic sequences), not a dominance theorem. | **To do:** on small instances compare the best permutation over all orders (`exhaustive`) with the best belt-order-respecting permutation; report the gap. |
| **B. NP-hardness needs a realizable reduction**: check whether it holds for fixed N = 3, whether release dates stay exogenous, which stop rule. | Our reduction (formulation §5.5) uses one part per conveyor, negligible travel and a zone long enough that blocking never binds. It therefore proves NP-hardness for **general N** (N = n) and the **own-conveyor** stop rule only; with `all_conveyors` the release dates become schedule-dependent. Fixed N = 3 is **not** covered. | **To do:** state the scope precisely in §5.5; optionally look for a reduction with fixed N. |
| **C. CP-SAT exactness is for the discretized model.** | Already stated (`docs/methods/exact_cpsat.md`, agent_architecture "optimal up to discretization"). | none |
| **D. Time windows are endogenous; write $s_{ij}(t,z)$.** | Correct: windows and travel depend on the belt state (stops caused by earlier decisions). The classification line in §5 writes $s_{ij}(t)$ and "time windows". | **To do:** rewrite the classification as $s_{ij}(t,z)$ with state-induced (endogenous) windows. |
| **E. Return = objective only with γ = 1 and consistent 1/M scaling.** | The undiscounted return equals $-(\sum F_j+\Pi)$ (tested); divided by the exogenous $n$ it is exactly `penalized_flow_time`. **But PPO was trained with γ = 0.99 per decision**, so the trained objective is a discounted proxy. | **To do:** say so in `docs/methods/rl.md`; try γ = 1 (or average-reward) in the next RL iteration. |
| **F. Admission loophole** (flow measured from belt admission would reward delaying entry). | Already handled: $F_j=C_j-r_j$ with $r_j$ the external arrival; the work-in-progress integral counts parts from arrival, including the upstream buffer. | none (mention explicitly in the paper) |
| **G. Revocable pre-positioning needs decision semantics.** | Defined in the engine: `when_idle` revokes at the first step in which another part is READY; no cancellation cost; the laser keeps its position and the policy decides again. The RL agent does not use pre-positioning. | **To do:** write these semantics formally in §1.5 (decision events during travel/wait). |
| **Overlap of processing and blocking stops.** | $T_B$ counts a conveyor as blocked whenever the "blocked" stop reason is active. With `processing_conveyor_only` the block ends at $S_j$, so there is no overlap; with `all_conveyors` a blocked conveyor may simultaneously be stopped for processing elsewhere, and it is counted as blocked (attributed downtime ≠ physical downtime). | **To do:** state this attribution rule in §1.4. |
| **Blocking-time optimization is already known (S11).** | Our penalty is new only in combination (event + duration, inside the interception/dispatch model). | Claim the coupling, not the penalty itself. |

## 3. What the review means for a paper (summary)

* The defensible contribution is the **coupled formulation**: a shared head intercepting
  moving jobs whose belt trajectories are changed by processing stops and by boundary
  blocking, with online arrivals and an explicit blocking cost. No verified study in
  the map combines these features → *provisional* novelty, not a priority claim.
* Methods (MILP, CP-SAT, ILS, rollout, PPO, masking, imitation) are known; they are the
  evaluation toolkit, not the contribution.
* Closest works to read in full: S02 (Han et al. 2020), S03 (Nielsen et al. 2023),
  S04 (Schettini et al. 2023), S08 (Philip et al. 2024), S10 (Briskorn et al. 2025),
  S11 (Salvia Hornos et al. 2026).
* Venues suggested by the map: Journal of Scheduling (theory/optimization route),
  Computers & Operations Research, RCIM or IEEE CASE (with industrial data).

## 4. Action list (not started; to be prioritized with the cell owner)

1. Restriction gap of head-only policies on small instances (point A).
2. Precise scope of the NP-hardness result (point B) and endogenous-window notation
   (point D) in the formulation.
3. Formal semantics of revocable pre-positioning and of blocked-time attribution
   (points G, overlap).
4. RL: γ = 1 / average reward, several training seeds, DAgger (point E + rl.md §5).
5. Additional experiments suggested by the map: event-only vs duration-only vs joint
   penalty; own-belt vs all-belts stop rule; decision-time tails.
6. Read the six closest papers; later, an export-based search for a complete SLR.
