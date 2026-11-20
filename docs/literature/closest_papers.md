# Closest papers: full-text reading and comparison

**Date:** 2026-11-19
**Purpose:** Read the six works that `review_assessment.md` named as closest (S02, S03, S04, S08, S10, S11) in full where openly available. Compare each one with our problem: one XY laser head, N parallel belts, interception inside a zone [a, b], processing-induced belt stops, boundary blocking with penalty Π = λ_N·N_B + λ_T·T_B, and online arrivals. Then check what `systematic_review_2026-11-19.md` claimed about them.
**Rule:** Every statement below comes from text actually read. Anything else is marked *not verified*. Numbers are quoted from the papers, not recomputed.

## Access

| Paper | Source read | URL used |
|---|---|---|
| S02 Han, Feng, Yu (2020) | Full text, arXiv v1 (preprint; differences from the RA-L version not verified) | https://arxiv.org/html/1912.08009v1 |
| S03 Nielsen et al. (2023) | Full text (publisher PDF, CC BY, from the Aalborg University repository; MDPI HTML returned 403) | https://vbn.aau.dk/ws/files/536324075/algorithms_16_00172_v2.pdf |
| S04 Schettini, Malucelli, Ramalhinho (2023) | Full text (Springer open-access HTML; result tables 3–8 read from the open-access copy at Politecnico di Milano) | https://link.springer.com/article/10.1007/s10951-022-00755-5 ; https://re.public.polimi.it/bitstream/11311/1233551/1/s10951-022-00755-5.pdf |
| S08 Philip, Ren, Rathinam, Choset (2024) | Full text, arXiv v3 (preprint) | https://arxiv.org/html/2403.04917v3 |
| S10 Briskorn, Boysen, Zey (2025) | Full text, Springer open access. Table 2 (complexity summary) and Table 3 (runtimes) did not render, so their contents are *not verified*. | https://link.springer.com/article/10.1007/s10951-024-00826-9 |
| S11 Salvia Hornos et al. (2026) | Full text, Springer open access, Sections 1–7. The annex was not read. | https://link.springer.com/article/10.1007/s10288-026-00620-7 |

No PDF was saved in the repository. The fetch tool cached two PDFs in its own temporary folder; they were read and then deleted. Nothing was paywalled.

---

## S02 Han, Feng, Yu (2020): pick-and-place on a moving conveyor

- **Setting.** One robot arm and one conveyor moving at constant unit speed. The paper expects the conveyor to "run without stopping"; no stops are modelled. Objects that leave the reachable rectangle on the left are lost. There are two arrival models: *one-shot*, where n objects are placed at once, and *continuous*, where objects arrive as a Poisson process at the right boundary. Robot models are a simplified telescoping arm (closed-form pick-and-place time), Delta and SCARA. For Delta and SCARA, times come from inverse kinematics plus a motion library, stored in a precomputed 100×100 lookup table. Pick and drop are instantaneous, and every pick returns to a fixed drop-off point.
- **Objective.** Pick as many objects as possible per unit time. Equivalently for one-shot, minimise the total pick-and-place time of a batch. The paper gives no numbered formal model.
- **Method.**
  - Proposition 3.1: for two objects, the optimal order depends on their horizontal offset, so greedy rules such as FIFO, SPT or nearest-first are not optimal in general. This is shown with a numerical example, not a general bound.
  - OptSeqDP: an exact subset dynamic program over sequences, O(2ⁿ·n).
  - SubOptDP: starts from FIFO and re-optimises sliding windows of length 9 with OptSeqDP.
  - In continuous mode, both are re-invoked on the current set of pickable objects.
- **Results.**
  - About 22 objects (OptSeqDP) and about 100 objects (SubOptDP) are solved within about 1 s.
  - SubOptDP's average gap to OptSeqDP "never exceeds 0.05%". This is empirical; there is no quality guarantee.
  - One-shot: the greedy baselines use about 10–20% more time.
  - Continuous (SCARA, 10,000 objects): SubOptDP beats SPT and Euclidean by about 10% and FIFO by up to about 40%. FIFO is the best baseline at low arrival rates and the worst at high rates.
- **Limitations.** Simulation only, with hardware work stated as in progress. Constant belt speed. One robot. Multi-robot coordination is left as future work.
- **Relation to our problem.**
  - *Same:* single shared resource, moving targets, interception times that depend on position, a Poisson online mode, and greedy dispatching versus look-ahead.
  - *Different:* only one belt; the belt never stops, so processing never changes target trajectories; parts are lost instead of blocking; and the objective is throughput, not flow time.
  - *Reuse:*
    - The `GetPnPTime` abstraction (separating geometry from sequencing) matches our s_ij(t,z) oracle.
    - OptSeqDP is an exact small-instance benchmark. Whether the (set, last) principle of optimality still holds once belt stops make positions history-dependent is *not verified*. It probably needs completion time and per-belt odometer in the state.
    - The finding that FIFO is best at low load and worst at high load is a direct parallel to our head-only result. Cite it as such, not as proof.

## S03 Nielsen et al. (2023): robotic batching with multiple conveyor belts

- **Setting.**
  - A food batching cell. One infeed belt moves "continuously at a fixed speed", with one tray lane on each side. Item weights are random.
  - Several robot arms, each with its own non-overlapping operational area. The simulation uses 2 arms.
  - A tray lane advances incrementally only when the last tray exceeds its target weight.
  - An item "can be placed in the tray only when the tray lane is not moving" (constraint 7).
  - Items arrive stochastically and are unknown until they enter the belt. There is no blocking: items not picked are disposed of at the belt end, per the flow chart.
- **Objective.** Minimise give-away, i.e. the positive deviation of tray weight from its target.
- **Model and method.**
  - The problem is decomposed into one sub-problem per tray, solved sequentially in tray order.
  - Each sub-problem is a binary goal program with variables x_{i,f,n}: item i is picked at belt field f after the n-th tray advancement. The belt is discretised into equal fields, with arrival times AT_{i,f} and arm travel times τ between fields and tray positions.
  - Big-M availability constraints link the sub-problem to schedules already fixed upstream.
  - Each sub-problem is solved with CPLEX. A variant maximises a piecewise-linear "profit" objective (the "Hybrid" model).
- **Results.**
  - Setup: 2 arms; an 18-field buffer zone that items cross in 3.6 s; a decision budget of 1.8 s; exponential inter-arrival times with mean 1.2 s; one 30-minute replication.
  - 92% of 1,116 batch schedules were computed within 1.8 s (median 0.66 s, s.d. 0.69). The Hybrid model reached 98%.
  - A longer buffer zone lowers give-away but raises computation time beyond the budget.
  - The approach is not compared with any baseline dispatching policy.
- **Stated limitations.** Decomposition cannot reach the global optimum. The approach is deterministic. Local search and DRL are suggested as future work.
- **Relation to our problem.**
  - *Same:* multiple belts; moving items; travel times that depend on sequence and position; near-real-time, online decisions under a compute budget.
  - *Different:* several arms with disjoint areas instead of one shared head; no processing-induced stop of the item belt; no blocking; the objective is give-away.
  - The one endogenous-motion element is the tray lane: it moves only when a decision (tray weight) triggers it, and placement needs it to be stationary. This is the closest analogue to our "trajectory changed by scheduling decisions". Here it applies to the destination lane, not the source belt.
  - *Reuse:* field discretisation with AT_{i,f}, which fits our time-indexed model, and reporting the decision-latency distribution against a physical budget.

## S04 Schettini, Malucelli, Ramalhinho (2023): scheduling a pick-and-place packaging line (P4)

- **Setting.**
  - Identical single-grip robots with non-overlapping working areas, arranged along two parallel conveyors moving in the same direction (input and output). "The velocity of the conveyors is fixed."
  - Items arrive in columns. "All items that reach the end of the conveyor without being picked up are lost."
  - Each mission (item column i, container j, robot r) takes a constant time δ.
  - Pick windows [a_i^r, b_i^r] and container windows [α_j^r, β_j^r] are input data; they are not derived from geometry in the model.
  - Cameras see items about 5–10 s ahead. The paper motivates rolling-horizon use, but the experiments are static instances.
- **Objective.** Minimise the number of lost (unpicked) items. Minimising unfilled slots is the alternative.
- **Model and method.**
  - An ILP solved by row generation: a master problem handles availability and capacity, and a per-robot timing-feasibility check adds cuts.
  - Two exchange properties: crossed mission pairs can be "uncrossed" (a 1-irreducible assignment), and the sequence can then be reordered into a dominant order.
  - Extensions:
    - *Item holding:* mission time is δ plus waiting for the container.
    - *Controllable output-conveyor speed:* a discrete speed profile is chosen.
  - An ILS over item and slot counts per robot, with a decoder, tabu list and perturbation.
  - The paper gives no NP-hardness statement.
- **Results.**
  - Instances: 3 four-robot line configurations from Ferrari et al. (2015), with 15–60 item columns; CPLEX with a 250 s limit.
  - Row generation dominates the earlier MILP. For example, config 0 with I = 30, C = 10: the old MILP solved 0 of 11 within 250 s, while row generation solved 11 with an average of 7.6 s (Table 3).
  - ILS on 60-column instances: average lost items versus best lower bound is 4.5/4.2, 36.8/36.1 and 22.4/22.0, in about 1.2–1.4 s (Table 4).
  - The controllable-speed extension helps more than holding. In config 0, average losses drop to 0.0.
- **Stated limitations.** Row generation is only a lower bound if stopped early. "neither ROW nor FERR provide computational times that are compatible with a real-time application". Mixed item types and multi-grip robots are future work.
- **Relation to our problem.**
  - *Same:* items moving on belts, access windows, a sequence per resource, and a control lever on conveyor motion.
  - *Different:* the robots are several and disjoint, not one shared head; mission times are constant and windows exogenous, so there is no geometric interception; unpicked items are lost, not blocking; belt speed is a decision (output conveyor), not a consequence of processing.
  - *Reuse:*
    - The uncrossing/dominance argument is the closest template for a formal belt-order (head-only) dominance result in our setting.
    - Row generation with timing-feasibility cuts.
    - Reporting the ILS value against a lower bound.

## S08 Philip, Ren, Rathinam, Choset (2024): MICP for the moving-target TSP via a graph of convex sets

- **Setting.** One agent with maximum speed v_max starts at a depot, visits each of n targets exactly once within that target's given time window, and returns. Each target "moves along its own line, with fixed speeds". The problem is offline and deterministic.
- **Objective.** Minimise the agent's travel distance.
- **Model and method.**
  - Within its window, each target's trajectory is a segment in (x, y, t), which is a convex set. The MT-TSP then becomes a shortest path in a graph of convex sets with speed (cone) constraints.
  - The paper first writes a biconvex reformulation of the Stieber–Fügenschuh MICP, then replaces the bilinear terms with perspective constraints and removes big-M (MICP-GCS).
  - Theorem 1: MICP-GCS has the same optimal value as the biconvex formulation.
  - The convex relaxation gives lower bounds.
- **Results.**
  - Instances: 80 instances (5, 10, 15 and 20 targets), Gurobi with an 1800 s limit.
  - Runtime is up to two orders of magnitude lower than the baseline MICP, and optimality gaps are up to 60% tighter.
  - The relaxation-to-optimum ratio is 0.95 at 5 targets and falls to about 0.48–0.54 at 20 targets. The baseline MICP relaxation always gave 0.
  - 20 targets remain hard.
- **Limitations.** The method needs linear, constant-speed motion. The bounds are "somewhat crude" for larger instances and wider windows. Multiple agents and piecewise trajectories are future work.
- **Relation to our problem.**
  - *Same:* a single agent intercepting moving targets with continuous-time interception, plus exact formulations and relaxation bounds.
  - *Different:* trajectories are exogenous straight lines. In ours, trajectories are piecewise linear with stops whose timing depends on the schedule. The objective is distance, not flow time. There are no belts, stops, blocking or online arrivals.
  - *Reuse:* perspective/GCS constraints instead of big-M for interception, and the relaxation as a bound. Both apply directly to a no-stop variant. Whether a bound derived from the no-stop variant is valid for our model is *not verified* and would need a proof.

## S10 Briskorn, Boysen, Zey (2025): blocking machines in e-commerce packaging

- **Setting.**
  - A single "blocking machine": several back-to-back packaging devices along one infeed conveyor, each dedicated to one carton size.
  - Goods cannot overtake, so "a good that is assigned to an upstream packaging device will block subsequent goods that require a downstream device".
  - The decision is the infeed sequence. Variants: a given cart sequence (resequencing only within carts), an arbitrary cart sequence, or a fully arbitrary sequence.
  - The infeed has no gaps, and all data are deterministic.
- **Objective.** Minimise makespan, i.e. the number of packaging batches. Blocking loss is measured as empty device slots. Packaging waste is analysed separately, as a trade-off with the number of carton sizes.
- **Results (theory).**
  - With a given cart sequence, a greedy algorithm is optimal (Lemma 1).
  - The problem is polynomial for a fixed number of carts and for a fixed number of carton sizes.
  - The exact DP is O(2^C·C·S̄²), where C is the number of carts. It is strengthened with a lower bound and a beam-search upper bound.
  - The complexity of an arbitrary cart sequence with the number of sizes part of the input is stated as open.
  - Table 2 (the full complexity table) did not render: *not verified*.
- **Results (computation).**
  - 27,000 generated instances and 20 real instances (Olist data, 300 orders each), all solved to proven optimality with runtimes "well below one minute" for C = 20.
  - A fixed infeed sequence raises makespan by more than 200% versus the perfect sequence.
  - A few carton sizes capture most of the waste reduction.
- **Limitations.** One machine; deterministic data; no gaps on the infeed; one carton size per order. Multiple machines and other machine types are future work.
- **Relation to our problem.**
  - *Same:* blocking caused by belt order (no overtaking) is the mechanism behind our head-of-belt blocking, and the complexity analysis distinguishes fixed parameters from parameters that are part of the input.
  - *Different:* there is no moving-target interception, no travel and no shared head. Blocking is between devices on one belt. The objective counts batches, not time.
  - *Reuse:*
    - The variant/parameter complexity grid. Our NP-hardness is for N part of the input, so the case of fixed N is a natural open question to state.
    - "Blocking loss" as lost capacity is an extra reporting metric.

## S11 Salvia Hornos et al. (2026): reducing conveyor blockages in a food warehouse

- **Setting.**
  - An automated warehouse (bonÀrea) with two sub-warehouses served by 8 and 7 stacker cranes.
  - Boxes travel on conveyors to a crane buffer. A blockage occurs when a box is assigned to a crane whose buffer is full and it cannot be redirected. The conveyor then stops upstream until the crane takes a box.
  - Box routing is random among valid cranes; it is not optimised.
- **Decision and objective.**
  - The decision is the article-to-crane/channel assignment map ("Article-Map Relocation Problem"). It must keep the article and workload balance and stay within K = 10 changes of the current map.
  - The objective is the **worst case over random seeds** of cumulative blockage: per-time-step indicators summed over lanes and time.
  - The paper argues that a time-indexed ILP would need at least 10¹² variables.
- **Method.**
  - A Java discrete-event simulator is used as a black-box evaluator, with 5 seeds per candidate.
  - Many metaheuristics are compared: hill climbing, LAHC, SA, ILS, beam search, GA, LNS with a SCIP repair, and a surrogate-assisted variant.
  - Three ways of handling the change budget are compared (discard, ILP repair, and "bounded exploration", which excludes moves that would leave the bound). Bounded exploration dominates.
  - The optimisation is offline.
- **Results.**
  - LNS reduces simulated blockage by about 45% against the hand-tuned baseline. GAs reach about 30%.
  - No lower bound or optimality guarantee is given.
  - The dataset is public (DOI 10.21227/e3b2-q809).
- **Limitations.** One month of data (July 2024); incomplete hyperparameter tuning; the change budget limits the attainable gain.
- **Relation to our problem.**
  - *Same:* the blocking semantics are the closest to ours (the flow stops until a resource clears it), and the objective is explicitly blocked time. Evaluation is simulation-based with seeds.
  - *Different:* it is a strategic, offline assignment decision, not dispatching; there is no interception and no moving-target travel; routing is random; no event count is used.
  - *Reuse:* the risk-averse worst-case-over-seeds objective as a robustness check on our policies, and bounded-neighbourhood search when changes to a baseline are restricted.
  - *Cite with care:* "blocked-time minimisation is known (S11)". The bib entry lists 4 authors, but the Springer page shows an ellipsis in the author list, so completeness is *not verified*.

---

## Check of the earlier review's claims

| Paper | Claim in `systematic_review_2026-11-19.md` | Verdict |
|---|---|---|
| S02 | "finite-horizon DP and faster approximations; greedy rules need not be time-optimal" | **Confirmed.** Greedy suboptimality is shown by example, not a general bound. The "running-time guarantees" are about runtime, not solution quality. |
| S02 | Table: one arm, one belt; moving + interception Y/Y; no stop-on-processing; objects become inaccessible; horizon/online | **Confirmed.** |
| S03 | Multiple conveyors, moving item/tray positions, sequence-dependent processing, near-real-time; placement needs a stationary tray lane | **Confirmed** (constraint 7). The simulation uses only 2 arms. The method is a decomposed goal program (binary MIP) solved with CPLEX. |
| S03 | Table: "stationary tray for placement" under stop-on-processing; "tray advancement" under blocking | **Partly correct.** Tray stops are triggered by the weight threshold, not by processing. The tray lane does not block the item belt: unpicked items leave. |
| S04 | Exact formulation + row generation, structural properties, ILS, item holding, controllable conveyor speed | **Confirmed.** Only the *output* conveyor speed is controllable. |
| S04 | Table: moving jobs + interception "Y; Y"; stop-on-processing "speed-control extension" | **Corrected.** Windows are input data and mission times are a constant δ, so there is no geometric interception. Speed control is a decision, not a stop caused by processing. |
| S04 | "SA/ILS over orders KNOWN (S04)"; "look-ahead in S04" as a precedent for pre-positioning | **Corrected/weak.** The ILS searches over per-robot item/slot counts, not orders. S04 has a 5–10 s information horizon, but no robot pre-positioning. |
| S08 | Interception via space-time convex sets, relaxation bounds; one agent, assigned time windows, offline | **Confirmed.** The objective is travel distance, and the method is limited to straight-line constant-speed targets (≤ 20 targets tested). |
| S10 | Devices block each other on one infeed (no overtaking); complexity; exact DP; offline | **Confirmed**, with nuance: the DP is polynomial in the number of orders only for a fixed number of carton sizes, and exponential in the number of carts. One complexity case is open. |
| S11 | Explicitly minimises conveyor blockage time (simulation + bounded metaheuristics, industrial case); full buffers stop upstream conveyors; offline + simulation | **Confirmed.** The objective is the worst case over 5 seeds of cumulative blocked time steps. |
| S11 | Table: moving jobs + interception "Y; not central"; RQ4 "closest" blocking precedent | **Corrected / confirmed.** There is no interception at all, and the decision is a storage assignment, not dispatching. It is the closest precedent for blocking semantics. |
| all six | Quality scores (S02 6, S03 4, S04 7, S08 6, S10 6, S11 7) | Not re-scored. They are plausible; S03's low score fits the missing baseline comparison and single replication. |

---

## Synthesis

### (a) Feature coverage

| Feature of our problem | S02 | S03 | S04 | S08 | S10 | S11 |
|---|---|---|---|---|---|---|
| One shared resource serving several belts | – (1 belt) | – (disjoint arms) | – (disjoint robots) | single agent, no belts | – | – |
| Several parallel input belts | – | partial (1 infeed + 2 tray lanes) | partial (input + output) | – | – | conveyor network |
| Geometric interception of moving parts | ✓ | partial (fields + τ) | – (given windows, δ) | ✓ (exact, continuous) | – | – |
| Belt stops caused by processing | – | – (tray lane stops by weight rule) | – (speed is a decision) | – | – | – |
| Endogenous accessibility windows | – | partial (tray positions) | partial (via speed decision) | – | – | – |
| Boundary blocking (flow halts until cleared) | – (lost) | – (lost) | – (lost) | – | ✓ (no overtaking) | ✓ (full buffer) |
| Blocking cost: events + duration | – | – | – | – | slots lost | duration only |
| Online stochastic arrivals | ✓ | ✓ | motivated, not tested | – | – | stochastic sim, offline decision |
| Flow time / makespan objective | time per batch | – | – | – | makespan (batches) | – |
| Exact method / lower bound | subset DP | per-tray MIP | row gen. + LB | MICP + relaxation | DP + LB | – |
| Complexity result | – | – | – | – | ✓ | – |
| RL | – | future work | – | – | – | – |

**None of the six covers:** processing-induced stops of the source belt; endogenous interception windows for the part being chased; a single head shared across belts; or an event + duration blocking cost.

### (b) Implications for the novelty claim

- Each ingredient has a precedent:
  - interception: S02, S08;
  - multi-belt robotic scheduling: S03, S04;
  - blocking with time-based cost: S10, S11;
  - conveyor-motion control: S04;
  - decision-dependent lane motion: S03.
- The defensible claim is therefore a **coupled formulation**: one shared head; source belts whose motion is altered by the head's own processing; endogenous windows; and boundary blocking with an event + duration penalty, studied online. Avoid "first" claims.
- S03 and S04 come close on decision-dependent conveyor motion, so the distinction must be stated precisely: in our model, processing stops the *source* belt carrying the targets.

### (c) Ideas worth trying

1. Subset DP (OptSeqDP) as an exact benchmark for small offline instances. First check that the state must include time and belt odometers (S02).
2. Sliding-window DP improvement of the belt-order (head-only) sequence as a cheap look-ahead between nearest_head and rollout (S02).
3. Report the load at which FIFO/head-only stops being competitive. S02 found FIFO best at low rates and worst at high rates, which parallels our bursty-arrival gap (S02).
4. An exchange/"uncrossing" argument to prove dominance of belt-order sequences in restricted cases (S04).
5. Row generation with timing-feasibility cuts for the discretised model, with ILS/rollout values reported against the master's lower bound (S04).
6. Belt discretisation into fields with arrival times AT_{i,f} and travel times τ, for the time-indexed MILP (S03).
7. Report decision-latency histograms against a physical budget, e.g. the time a part takes to reach a (S03).
8. Perspective (GCS) constraints instead of big-M for interception, plus the convex relaxation as a lower bound for the no-stop variant; validity for the stop model must be proved (S08).
9. A complexity grid by which parameters are fixed (N, number of parts per belt), stating fixed-N as an open case (S10).
10. "Blocking loss" (lost head capacity) as an extra reporting metric next to N_B and T_B (S10).
11. A worst-case-over-seeds (risk-averse) evaluation of policies, with common random numbers across policies (S11).
