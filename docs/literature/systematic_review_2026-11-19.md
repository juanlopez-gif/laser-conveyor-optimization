# PRISMA-informed evidence map and novelty assessment (2026-11-19)

> **Provenance.** Produced on 2026-11-19 by an external AI research tool from the
> review prompt prepared in this project (PRISMA 2020 method, research questions
> RQ1–RQ6). Reproduced here as received, except that the BibTeX section is in
> [references.bib](references.bib) (with the S01 DOI added). The references were
> verified and the technical remarks checked against our work in
> [review_assessment.md](review_assessment.md). Statements about our work in this
> text are the tool's reading of the project summary it was given, not an audit of
> the code.

**Your strongest potential contribution is the coupled problem formulation: interception of moving jobs whose conveyor trajectories change because of processing and boundary blocking.** The individual methods—MILP, CP-SAT, ILS, rollout, PPO, action masking, and learning from expert schedules—are established.

I verified **21 relevant studies**, with accessible full-text material for **16** and abstract-level evidence for **five**. I did **not** find a verified study combining all your stated features. That supports a **provisional novelty claim**, not a definitive priority claim.

**Review status:** this is a **PRISMA-informed, access-limited systematic evidence map and novelty assessment**. It is not a completed, publication-ready SLR: native database searches, Google Scholar's first 200 results per query, comprehensive citation chasing, and complete screening counts could not be obtained. Missing counts are reported explicitly.

## 1. Protocol

| Item | Protocol |
|---|---|
| Questions | RQ1–RQ6 supplied in the request: proximity, modelling/interception, algorithms/evaluation, blocking, RL formulations, and novelty. |
| Coverage | English-language studies published from 1990 through **19 November 2026**. Earlier foundational material is background only. |
| Planned sources | Scopus, Web of Science, IEEE Xplore, ScienceDirect, SpringerLink, ACM Digital Library, Google Scholar, and arXiv. |
| Executable sources | Public-web scholarly discovery; publisher abstracts/full texts; author and institutional repositories; arXiv. |
| Search structure | Resource block A AND moving-job block B AND decision block C; optional method block D; targeted searches for blocking, time-dependent setups, laser processing, masking, and SMDPs. |
| Inclusion | Scheduling/sequencing of moving jobs or motion-induced windows; relevant shared-resource dispatching/RL; conveyor-fed blocking/stoppage optimization. |
| Exclusion | Hardware/vision/control without sequencing; trajectory planning without job selection/order; unrelated generic scheduling RL; duplicates; unavailable abstract and full text. |
| Screening | Verify identity; consolidate publication/preprint versions; screen abstract; inspect accessible full-text material. Abstract-only records remain provisional. |
| Extraction | Citation, application, resources, motion/interception, processing stops, blocking, information/arrivals, objectives, model, methods, bounds, RL design, evaluation, validation, findings, limitations. |
| Appraisal | Five requested 0–2 criteria. Unknown evidence remains unknown; theoretical work is interpreted separately. |
| Synthesis | Narrative taxonomy and feature comparison. No meta-analysis: objectives, robot models, benchmarks, and information assumptions differ substantially. |
| Registration/reviewers | Unregistered protocol, written before searching in this interaction. Single AI-assisted reviewer; no independent duplicate screening. |

PRISMA 2020 is a reporting guideline; using its structure does not compensate for an incomplete search or missing screening records (Page et al., 2021, doi:10.1136/bmj.n71).

**Protocol deviations:** native database access failed; Scholar's specified depth was not achieved; arXiv category-complete searching was not achieved; forward citation chasing was partial; five records could not receive full-text appraisal.

## 2. Search log

### Database access

| Database | Native search specification | Execution/access outcome | Hits |
|---|---|---|---|
| Scopus | `TITLE-ABS-KEY((A) AND (B) AND (C))`; repeat with `AND (D)`; year/language filters | Search page returned 403; not executed | NR |
| Web of Science | `TS=((A) AND (B) AND (C))`; optional `AND TS=(D)` | Search page returned 403 | NR |
| IEEE Xplore | All Metadata: `(A) AND (B) AND (C)`; optional D | Public search inaccessible | NR |
| ScienceDirect | Title/abstract/keywords: expanded A–C | Search page unavailable; individual records sometimes accessible | NR |
| SpringerLink | Expanded A–C; phrase searches | Search page inaccessible; individual articles accessible | NR |
| ACM DL | Advanced search title/abstract: A–C | Inaccessible | NR |
| Google Scholar | Expanded blocks; first 200/query | Inaccessible; 200-result screening not performed | NR |
| arXiv | `(A AND B AND C)` restricted to cs.RO, cs.AI, cs.LG, math.OC | Native search unavailable; individual records accessible | NR |

**NR = not retrievable, not zero.**

### Executed public-web queries (2026-11-19, hits NR for every query)

Q1–Q56 covered: robot picking on moving conveyors; moving-target TSP; "blocking" + "conveyor" + scheduling; robotic laser + conveyor; RL + conveyor/picking; DRL dispatching with action masking / semi-Markov; time-dependent setups on a single machine; laser on-the-fly scheduling; title searches for candidate papers (S02–S16, Stecco 2008, Helvig 2003, Huang–Ontañón, Zhang 2020, Bertsekas 1997, Little 1961, PRISMA 2020); the full Boolean blocks A–C and the additional query strings of the prompt. (The full list of query strings is in the original output kept by the project owner.)

### Citation chasing

Reference sections inspected for seeds S02, S03, S04, S05, S06, S07, S08, S10, S15, S16; forward-discovery attempts Q37–Q46. Partial; no saturation claim.

## 3. PRISMA 2020 flow

| Stage | Number/status |
|---|---:|
| Records identified through native databases | NR |
| Records identified through web discovery/citation chasing | NR |
| Duplicate records removed | NR |
| Unique titles/abstracts screened | NR |
| Records excluded at title/abstract stage | NR |
| Retained comparison records | **21** |
| Retained records with full-text material assessed | **16** |
| Retained records supported only by abstracts | **5** (S01, S09, S12, S14, S18) |
| Full-text exclusions from the retained set | **0** |
| Studies in full-text-supported synthesis | **16** |
| Additional provisional abstract-supported comparisons | **5** |

Exclusion categories: hardware/sensing/motion planning without job sequencing; unrelated warehouse/fleet routing or generic scheduling; manuals, patents, news; duplicates. Counts by reason unavailable.

## 4. Taxonomy and narrative synthesis

### A. Robotic picking from moving conveyors
* **S01 Mattone et al. (2000)**: moving-position-dependent service times, stochastic arrivals, objects lost when they leave the workspace.
* **S02 Han et al. (2020)**: finite-horizon DP and faster approximations; greedy rules need not be time-optimal.
* **S03 Nielsen et al. (2023)**: multiple conveyors, moving item/tray positions, sequence-dependent processing, near-real-time optimization; infeed plus two tray lanes, placement requires a stationary tray lane.
* **S04 Schettini et al. (2023)**: exact formulation and row generation, structural properties, ILS, item holding and controllable conveyor speed.
* **S05 Kokkalis et al. (2026, VARS)**: moving-object availability windows, value-based dispatching and ILS, physical robotic validation; collected utility rather than mandatory service.

*Reviewer judgement:* motion-aware sequencing is established; the most consequential difference of our problem is "unprocessed object stops its belt" instead of "missed object leaves", which changes future positions and admissions.

### B. Moving-target routing and interception
* **S06 Helvig et al. (2003)**: moving-target TSP, exact and approximation algorithms.
* **S07 Stieber & Fügenschuh (2022)**: discrete- vs continuous-time formulations with visibility windows.
* **S08 Philip et al. (2024)**: interception via space-time convex sets, relaxation bounds.

*Reviewer judgement:* interception is established; the distinction is **exogenous trajectories vs trajectories altered by scheduling decisions**.

### C. Single-machine scheduling with time-dependent setups
* **S09 Stecco et al. (2008)**: sequence- and time-dependent setups, branch-and-cut.

*Reviewer judgement:* our setup (travel) can depend on the belt-state vector; $s_{ij}(t)$ may understate the state dependence.

### D. Conveyor blocking and stoppage optimization
* **S10 Briskorn et al. (2025)**: packaging devices blocking each other on an infeed conveyor (no overtaking); complexity, exact DP.
* **S11 Salvia Hornos et al. (2026)**: explicitly minimizes conveyor blockage time (simulation + bounded metaheuristics, industrial case).

*Reviewer judgement:* "optimize blockage time" is not new by itself; the potential contribution is the coupling of boundary blocking, common belt motion, interception and shared-resource dispatching. Classical blocking flow shops are an analogy, not an equivalent formulation.

### E. Shared-resource RL and robotic production control
* **S12 Wang & Usher (2004)**, **S13 Li et al. (2020)**: learning-based online single-machine dispatching.
* **S14 Kim & Lee (2023)**: look-ahead + RL for a robotic flow shop.
* **S15 Zisgen & Hinrichs (2025)**: asynchronous gantry decisions, action restrictions, interference penalties.
* **S16 Drijver et al. (2023)**: PPO for conveyor-speed decisions around a robotic packaging controller.
* **S17 Drakaki & Tzionas (2017)**: timed coloured Petri nets + Q-learning in conveyor-fed order picking.
* **S18 Wei et al. (2024)**: invalid-action masking for AGV dispatching.

### F. Representative scheduling-learning methods
* **S19 Zhang et al. (2020)**: PPO dispatching with graph representations, objective-aligned telescoping reward.
* **S20 van Remmerden et al. (2025)**: masked offline RL trained from CP-generated schedules.
* **S21 Sun et al. (2023)**: event-based dynamic scheduling as an SMDP, telescoping tardiness reward.

## 5. Feature comparison (condensed)

| Study | Resources / conveyors | Moving jobs; interception | Stop-on-processing | Blocking / stoppage | Information |
|---|---|---|---|---|---|
| S01 | picking resource, moving belt | Y; Y | ? | workspace exit / lost objects | online, stochastic |
| S02 | one arm, one input belt | Y; Y | N | objects become inaccessible | horizon/online |
| S03 | multiple arms, infeed + 2 tray lanes | Y; Y | stationary tray for placement | tray advancement | near-real-time |
| S04 | robot line, input/output conveyors | Y; Y | speed-control extension | unpicked items lost; holding extension | offline/horizon |
| S05 | sorting robot | Y; Y | N | unselected objects exit | online |
| S06 | one pursuer | Y; Y | N (exogenous motion) | none | offline |
| S07 | multiple pursuers | Y; Y | N | visibility windows | offline |
| S08 | one agent | Y; Y | N | assigned time windows | offline |
| S09 | one machine | N | — | restricted setup intervals | offline |
| S10 | several packaging devices, one infeed | conveyor-fed, no interception | different | no overtaking | offline |
| S11 | conveyor network, stacker cranes | Y; not central | different | full buffers stop upstream conveyors | offline + simulation |
| S12–S13 | one machine | N | — | none | online |
| S14 | dual-gripper robot | transport; unverified | ? | robotic-flow-shop feasibility | given sequence |
| S15 | several gantries | transport | different | gantry interference | online |
| S16 | robotic packaging | Y (underlying controller) | belt speed controlled | supply/control constraints | online |
| S17 | 4 gantry zones, common conveyor | containers move | constant speed | buffers | batch |
| S18 | AGVs | vehicles move | — | congestion | online |
| S19–S21 | generic shops | N | — | none | offline/online |
| **Our work** | **one XY laser, N equivalent input belts** | **Y; simultaneous-axis interception** | **own belt or all belts** | **boundary-triggered rigid-belt stop; event + duration cost** | **offline/online; deterministic or Poisson; optional pre-positioning** |

Quality appraisal (P, R, B, S, V; 0–2 each), provisional: S02 6, S03 4, S04 7, S05 7, S06 3 (theoretical), S07 6, S08 6, S10 6, S11 7, S13 6, S15 7, S16 6, S17 4, S19 6, S20 7, S21 5; abstract-only S01, S09, S12, S14, S18 not scorable (2–10).

## 6. Answers to RQ1–RQ6 (condensed)

* **RQ1 (closest problems):** S02, S03, S04, S11, S08/S07, S16. No single comparator covers the whole problem; the work sits at their intersection.
* **RQ2 (models, interception):** dynamic service-time models, DP with interception timing, MIGP, integer formulations with row generation, time-window selection, geometric interception, time-expanded / continuous-time MP, space-time conic formulations, time-dependent setup formulations, Petri nets. For simultaneous axes the earliest interception must use the target position *at interception*, including belt stops in between; write $s_{ij}(t,z)$ with a system state $z$ (reviewer deduction).
* **RQ3 (methods, evaluation):** all our method classes exist. A strong evaluation needs: simulator–exact-model agreement, certified optima/bounds for small instances, runtime–quality curves at equal decision budgets, matched arrival streams, several RL training seeds, in- and out-of-distribution tests, ablations; label the information advantage of offline optimizers.
* **RQ4 (blocking):** objects leave (S01, S02, S04, S05), stationary destination lane (S03), no-overtaking obstruction (S10), full buffers stop upstream flow (S11, closest), gantry interference (S15). Blockage-time minimization is known (S11). Define whether overlapping processing and blocking stops are counted once or twice.
* **RQ5 (RL):** single-machine RL (S12–S13), look-ahead robotic RL (S14), asynchronous gantries (S15), PPO conveyor control (S16), masked dispatch (S18), objective-equivalent cumulative reward (S19), learning from expert schedules (S20), SMDP with telescoping reward (S21); SMDP RL (Das et al. 1999) and masking (Huang & Ontañón 2022) are established. The RL state must distinguish states with different transitions (trailing jobs, backlog, commitments).
* **RQ6 (novelty):** potentially yes for the complete combination (shared XY head, several equivalent moving input belts, interception, belt-specific or global processing stops, boundary-triggered rigid-belt blocking, explicit event and duration costs, online arrivals); priority not established. Suggested wording: *"We study a shared-head scheduling problem in which processing and boundary blocking endogenously alter conveyor trajectories. In the literature examined, we found no formulation jointly addressing this mechanism, moving-target interception, online arrivals, and explicit blocking-event and duration costs."* Avoid "first-ever".

## 7. Gap analysis and novelty assessment

| Gap | Our contribution | Evidence needed |
|---|---|---|
| Predetermined target motion ignores schedule-induced belt stops | odometer-based continuous model | formal transition rules, equivalence checks |
| Missed objects leave instead of stopping the belt | boundary blocking / admission coupling | experiments showing different optimal decisions |
| Exact interception and blocking models are separate | integrated MILP/CP-SAT + bounds | validity, scaling, certified gaps |
| Simple rules ignore cross-belt urgency/travel trade-offs | belt-head rule, rollout | strong competing rules, failure cases |
| Anticipation may delay urgent jobs | revocable pre-positioning | precise cancellation semantics, ablations |
| Reward proxies misaligned with cost | exact integrated cost reward | pathwise identity, implementation check |
| Look-ahead too expensive online | rollout-trained BC/PPO | teacher/student runtime–quality comparison |
| No benchmark for this mechanism | public simulator and instances | versioned code, seeds, traces, solver outputs |

| Contribution | Classification | Closest precedent |
|---|---|---|
| Integrated shared-head / interception / endogenous-stop formulation | **NOVEL, provisional** | S02–S04, S07–S11 cover subsets |
| Belt odometer representation | PARTIALLY NOVEL | natural device; novel use for coupled stop/resume |
| MILP / time-indexed formulation | PARTIALLY NOVEL | S03–S04, S07–S09 |
| NP-hardness proof | PARTIALLY NOVEL | reduction must establish this precise variant |
| CP-SAT + valid lower bounds | PARTIALLY NOVEL | model-specific bounds/dominance could contribute |
| Belt-order structural rule | PARTIALLY NOVEL | FIFO/order rules in S01–S02; a dominance theorem would be novel |
| Revocable pre-positioning | PARTIALLY NOVEL; unresolved | look-ahead in S04/S14, rollout |
| Blocking event + duration penalty | PARTIALLY NOVEL | S11 optimizes blockage time |
| SMDP formulation | KNOWN | Das et al. 1999; S15, S21 |
| Return = scheduling objective | KNOWN principle | S19, S21 |
| Maskable PPO | KNOWN | S19, S18, Huang–Ontañón |
| Behaviour cloning from rollout | PARTIALLY NOVEL application | S20 |
| SA/ILS over orders | KNOWN | S04, S05 |
| Rollout policy | KNOWN | Bertsekas et al. 1997; S14 |

Technical points raised (checked in [review_assessment.md](review_assessment.md)):
A. belt order ≠ processing order — head-only actions are a restricted class, measure the restriction gap;
B. NP-hardness needs a realizable reduction (fixed N vs general N, exogenous releases, stop rule);
C. CP-SAT exact only for the discretized time domain;
D. time windows are endogenous;
E. return = objective needs γ = 1, draining, consistent 1/M scaling;
F. distinguish external arrival from belt admission (otherwise delaying entry is rewarded);
G. revocable positioning needs explicit decision semantics.

Recommended additional experiments: FIFO, earliest-boundary / least-slack, nearest job, nearest head, shortest predicted completion, a blocking-aware priority rule; equal-budget SA/ILS and rollout; PPO alone, BC alone, BC→PPO; head-only vs unrestricted actions on small instances; no / irrevocable / revocable pre-positioning; event-only, duration-only and joint blocking costs; own-belt vs all-belts stop; report cost components separately and decision-time tails.

## 8. Recommended venues

| Venue | Fit | Closest precedent |
|---|---|---|
| Journal of Scheduling | formal model, complexity, dominance, exact algorithms, bounds | S04, S10 |
| Robotics and Computer-Integrated Manufacturing | integrated industrial scheduling/control with application evidence | S01, S12 |
| Computers & Operations Research | formulation, bounds, exact/heuristic computation | S09 |
| IEEE Robotics and Automation Letters | robotics/interception with real-time and hardware evidence | S02 |
| IEEE CASE | automation-oriented formulation, simulator, online policy | S16 |

## 9. Limitations of the review

Incomplete database coverage (public-web discovery only); no first-200 Scholar screening; no complete PRISMA identification/deduplication ledger; five abstract-only studies; partial citation chasing; selection and availability bias (English, accessible); provisional quality scores; no inspection of our proofs, models, code or results; no meta-analysis; no non-English studies extracted; novelty remains conditional.

## 10. References

See [references.bib](references.bib) (S01–S21, M01–M05; all identifiers verified to exist on 2026-11-19).
