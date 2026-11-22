# Two papers on a shared head and several conveyors

Research plan received from the owner on 2026-11-22 (translated from Spanish; the
original decisions are kept, the review notes at the end are ours). Titles, sample
sizes and budgets are experimental-design proposals: they will be adjusted in a pilot
and frozen before the confirmatory evaluation runs.

## Decision

**Paper 1.** Formulate the general problem and develop a sequencing algorithm. Fix what
the controller knows and study how to decide better with that information.

**Paper 2.** Measure the value of observing the parts before they enter the working
zone. Fix the cell dynamics and the reference algorithm, and measure which decisions
change, how much the operation improves and when extending the instrumentation is
justified.

Paper 1 comes first. It delivers the formulation, the simulator and the policies that
Paper 2 reuses. The laser is the case that motivates both works. A dimensional
inspection cell with a shared XY head can illustrate the industrial scope if its
conveyors are verified to have the same stopping dynamics.

## Paper 1 · General problem and algorithm

Provisional title: *Scheduling a Shared Mobile Head Across Parallel Conveyors with
Endogenous Blocking*.

Research question: when and by how much does sequencing improve by anticipating that
serving a part stops its conveyor and shifts the availability of the following ones?

### Model studied

A mobile head serves one part at a time in a working zone shared by several parallel
conveyors. Parts advance with their conveyor and arrive during the operation. Serving a
part immobilizes its conveyor for the service time. If an unserved part reaches the end
of the working zone, it blocks its conveyor until it is processed. The head may choose
any reachable part and is assigned only to parts already inside the working zone. The
policies decide when the head is free and there is a candidate. The reference
observation includes the positions of the parts inside the zone, the state of each
conveyor and the position of the head.

For an episode with $n > 0$ parts the main objective is

$$
J=\frac{\sum_{j=1}^{n}(C_j-r_j)+\lambda_N N_B+\lambda_T T_B}{n},
$$

where $r_j$ is the arrival to the system, $C_j$ the end of processing, $N_B$ the number
of blocking events and $T_B$ the aggregate blocked time in conveyor-seconds. Normal
processing stops are recorded separately. After the arrivals close, the simulation
continues until every released part is completed. If the technical limit is reached
before the system empties, the episode is recorded as incomplete and the limit problem
is solved before computing the confirmatory comparisons.

### Proposed contribution

1. **Formulation and properties.** Formalize how the accumulated position of each
   conveyor ties the service sequence to the future service windows. Include the lower
   bounds already derived, the counterexample to always serving the first part of each
   conveyor, and the exact scope of the complexity proofs.
2. **Event-based search with pruning.** Start from the observable state, simulate the
   candidate sequences in FastSim and update the advance, the interception and the
   blocks after each service. Use the current rollout to obtain a feasible solution
   early. Use valid bounds on travel, reachability and service to discard branches that
   cannot improve on it. Prove the validity of the bounds for the objective used and
   enumerate tiny instances before using them. In the sub-problem without future
   arrivals, the blocking cost already incurred plus a lower bound on the remaining
   flow is a starting point when the penalties are non-negative.
3. **A policy that applies only the first decision.** The controller explores sequences
   for the known parts, executes the first action and re-plans at the next decision
   event. The pilot fixes the depth and the computing budget. When the budget runs out,
   the best feasible sequence found is kept. Any certification of the search refers to
   the sub-problem and to the class of actions actually explored.

### Evaluation

| test | comparison | what it shows |
|---|---|---|
| Exactness in the small | Enumeration on tiny cases and CP-SAT on an extended grid for the same penalized objective. To verify the search, solve the same sub-problem without future arrivals and with the same action rules. | Consistency of the objective, solution quality and validity of the pruning. A reference with all future arrivals is labelled an offline reference. |
| Value of modelling the stops | Same search and candidates, with a projection that includes the stops against one that assumes continuous advance. Both policies execute their decisions in FastSim with real stops. | How much of the improvement comes from representing the conveyor dynamics. |
| Value of the bounds | Same search with and without pruning. Measure twice: at equal computing budget, and until the same sub-problem is closed where feasible. | Quality under a time limit, nodes explored and computing saved. |
| Online performance | Nearest part, nearest conveyor head, tuned urgency rule, simple rollout and the proposed method. Fix the information, actions, arrivals and tie-break rules. Compare the searches at the same budget. | How much the method adds over the references, especially the current rollout. |

**Proposed confirmatory bench.** Three cell sizes (2, 3 and 6 conveyors), three offered
processing loads (0.35, 0.60 and 0.80), two travel-to-service ratios, and regular and
bursty arrivals with the same mean rate: 36 configurations. Use 30 paired realizations
per configuration as a starting point and adjust the sample size with the pilot's
variability. Classify the observed congestion separately, since the processing load
excludes travel. Measure the sensitivity to the blocking weights as a separate
analysis.

**Main result.** Paired difference of the cost $J$ against the simple rollout. Also
report seconds per part, blocking events and blocked time, queue per lane, parts
completed, p50 and p95 latencies, and nodes explored. Use the same set of episodes in
every comparison and publish the confidence intervals together with the absolute
differences.

**Criterion to present it as an algorithmic paper.** Show a reproducible improvement
over the rollout at equal budget in identifiable regimes, or show that the bounds solve
sub-problems substantially faster at equal quality. Define with the pilot which
improvement and which latency have operational relevance. If both advantages stay
weak, the formulation, the properties and the characterization of regimes are the
alternative route for the manuscript.

## Paper 2 · Value of information

Provisional title: *The Value of Upstream Visibility in Scheduling Shared-Head
Conveyor Cells*.

Research question: how much is it worth to see the parts upstream, in which
configurations does the performance change, and how does that visibility interact
with the physical ability to pre-position the head?

**Main hypothesis (as first written).** Upstream visibility is worth more when parts
arrive in bursts and the travel between lanes matters. Under low contention, the
working-zone information may be enough. These are hypotheses the experiment must test.

**Reformulation after the pilot (owner, 2026-11-22; 0.7.10, `rl2.md` §8.7).** The
pilot across eight regimes found no gain from widening the view with the same actions
(B vs A within ±0.9 points everywhere), while enabling pre-positioning with that same
view reduced the penalized flow time by 3.0–5.7 % in the four nominal-load regimes. The
paper therefore asks a more concrete engineering question: **how much of the entry
stretch must the camera see for advancing the head to pay off, and which capacity to
act makes that coverage useful?** Its deliverables are:

* B vs A, kept as the main contrast of pure information, now expected to be a null
  result that the confirmatory bench must show to be small enough for the design
  decision (the pilot bounds it at ±0.9 points);
* C vs B, the value of acting on the same view;
* the **coverage curve**: the gain of pre-positioning at each visible length, measured
  as *with* versus *without* pre-positioning **at the same length** (not against the
  whole-zone reference), with the coverage expressed also in seconds of anticipation
  at the belt speed (5 cm are 1.7 s at the nominal 3 cm/s) and plotted against the
  head's travel time across the workspace. The pilot gives 97 % of the pre-positioning
  gain at 5 cm with the nominal head and 53 % with a head half as fast: the required
  coverage grows with the head's travel time.

The bursty/high-travel statement becomes a secondary hypothesis in the opposite
direction: the value of acting early falls with bursts and slow heads (the head is idle
less often, or idle with nothing to advance to), unless other burst settings say
otherwise. The hidden-queue contrast (D) stays as an indication (4.4 points at the
heavy load with a slow head, a regime outside the network's training range) until it
is checked with a model trained in that regime.

### Treatments and controls

In the main comparison both configurations observe the state of the conveyors and the
same positions inside the working zone; what changes is the reach of the observation.
Parts still outside every visible field remain unknown.

| treatment | what it observes | what the head can do |
|---|---|---|
| A · Local visibility | Parts inside the working zone. | Assign itself to a part inside the zone. |
| B · Upstream visibility | The above plus the parts visible in a preceding stretch of defined length. | The same actions as in A. |
| C · Upstream visibility and pre-positioning | Exactly the same as B. | It may also start moving towards a part visible in the preceding stretch. Processing starts inside the zone. |

B versus A measures the value of seeing earlier with the same actions available. C
versus B measures the additional value of acting earlier under the same observation.
C versus A measures the joint effect of the technological upgrade. The main contrast is
B versus A; the other two are secondary. The interruption rules of the pre-positioning
must be defined before the pilot.

### Execution

1. **Fix the planner.** Use the algorithm of Paper 1 with the objective, the budget and
   the parameters frozen. The simulator applies a mask for each visibility level. In A
   and B only the same reachable parts can be chosen for an identical physical state;
   the parts visible upstream in B may influence the future projection. Repeat B versus
   A with the simple rollout as a robustness check.
2. **Pair the realizations.** Each treatment receives the same arrival schedule, service
   durations and other disturbances. Use seeds different from those of Paper 1 and
   record the three treatments per seed. Cross regular or bursty arrivals, low or high
   travel, and the three offered loads. Include the nominal three-conveyor case as an
   anchor. Use 30 paired realizations per configuration as a starting point and
   recompute the size before freezing the design.
3. **Verify that every decision is causal.** Build pairs of states with the same
   observable history and different hidden futures. The features and the decision must
   coincide if the policy is deterministic, or with the same internal seed if it is
   stochastic. Check that the mask is respected when generating forecasts, choosing
   candidates and simulating sequences.
4. **Measure precision and reach separately.** To evaluate point sensors, compare their
   estimated location with a camera's while keeping the visible zone fixed. Evaluate
   noise, latency and detection losses in that secondary analysis. Arrival forecasts can
   be studied as another causal information source, with a measured error and no access
   to the future realization.
5. **Translate the effect into an instrumentation decision.** Report flow seconds saved
   per part, events and conveyor-seconds avoided, parts processed and decision latency.
   With data from a real cell, estimate the avoidable annual cost and the threshold
   above which the instrumentation pays off. The current simulation weights are
   equivalent seconds; the economic analysis must convert them with measured operating
   costs or present explicit valuation scenarios.

**Main result.** Paired saving $J_A - J_B$ per episode, overall and per operating
regime. Also show how much the decision changes on a set of common physical states.
Distinguish that rate of changes from the accumulated improvement over whole episodes,
where earlier decisions may lead to different physical states.

**Criterion for an independent second paper.** Publish it separately if the analysis
identifies when the additional information has an operationally relevant effect, when
its effect is small with sufficient precision, or how its value changes when
pre-positioning is enabled. A comparison on the nominal cell would serve as a
complementary study of Paper 1.

## Work order and available evidence

| step | verifiable deliverable |
|---|---|
| 1. Freeze the base problem | Model, local observation, allowed actions, KPI definition, instance formats and feasibility check. |
| 2. Build and test the search | Version without and with pruning, validity proof and small exact references for the chosen KPI. |
| 3. Close the evaluation of Paper 1 | Frozen configurations and seeds, paired results, ablations and quality-versus-computation tables. |
| 4. Prepare Paper 2 | Observation masks, definition of the upstream field, specified pre-positioning and tests that changing the hidden future leaves the decision intact. |
| 5. Close the evaluation of Paper 2 | A/B/C contrasts, results per regime and, with cost data, the economic instrumentation threshold. |

The simple rollout reduced the penalized flow by about 2.4 % in the nominal
configuration reproduced by the audit. That result serves as the reference to set the
difficulty of Paper 1. The audit also confirmed that an RL2 input read whether arrivals
remained in the simulator's future list. Any use of RL2 as a comparator must correct
that input, describe what the controller knows and evaluate the frozen model on new
seeds. The existing results motivate the experiments of the two papers; the proposed
contributions require the tests described above.

---

## Review notes (2026-11-22, after the 0.7.5–0.7.7 runs)

What the plan asks for and what the repository already has, with the points where the
evidence of the last two days should change the plan.

**Agreements.**

* The objective $J$ is `penalized_flow_time` (`docs/formal_formulation.md` §1.4), the
  decision epochs and the blocking rule are those of the formal model, and the
  "incomplete episode" rule is the simulator's `max_time` safety net.
* Treatments A, B and C are the observation levels and modes already implemented:
  A = `simulation.observation: zone` with `none`; B = `belts` with `none`;
  C = `belts` with `when_idle` (the design mode since 0.7.7, `docs/assumptions.md`
  C4). The masks are enforced in the engine, the fast simulator, the look-ahead
  policies and the `rl2` tokens, and `tests/test_observation.py` already contains the
  causality tests of step 3 (hidden parts, hidden arrival times, hidden future).
* The RL2 leak named by the audit ("arrivals still coming" read from the future list)
  was closed in 0.7.7, together with a second one the audit did not name: the waiting
  time counted from the arrival to the hidden upstream queue. The information set of
  §4.1 was corrected accordingly. Measured effect on the 0.7.6 network: within the
  confidence intervals.
* Paired realizations with common random numbers, frozen seeds and paired confidence
  intervals are how every result of `docs/methods/` is produced.

**Points the evidence argues against, to settle before freezing.**

1. *The known-parts search is the ceiling, not the method.* The plan's algorithm
   searches sequences of the **known** parts. The 0.7.5 runs showed that the look-ahead
   over the known parts saturates at about −5 % (real cell) and that no network of any
   size exceeds its teacher; the 0.7.6 teacher that samples the unknown future (8
   scenarios, common random numbers) reaches −7.5 %, and the network trained on it
   −8.4 % at 4 ms per decision (`docs/methods/rl2.md` §6.4, §8.5). An exact search
   with pruning over the known-parts sub-problem will solve that sub-problem faster and
   with a certificate, which is valuable for the "exactness in the small" test, but it
   optimizes a quantity that is already the limit. Suggestion: make the method of
   Paper 1 the sampled-future search plus the learned policy that amortizes it, and
   keep the pruned search as the verification piece for the sub-problem.
2. *The difficulty reference moved.* The plan sets the bar at the simple rollout
   (−2.4 % on the nominal cell). With the information model of 0.7.3 the references on
   the validation seeds are: simple rollout −4.9 %, network alone −8.4 % (`none`) and
   −13.8 % (`when_idle`). Paper 1 must compare with the sampled-future rollout and with
   the network, not only with the simple rollout, or its claimed gain will be measured
   against a baseline that is no longer the state of the repository.
3. *Bursty arrivals need a definition.* The generators implement deterministic and
   Poisson arrivals. "Bursty with the same mean rate" needs a model (batch Poisson or a
   hyper-exponential inter-arrival) and a parameter for the burstiness; it is a small
   addition to `simulation/arrivals.py` and to the general distribution.
4. *The load grid stops at 0.80.* The heavy example (0.93) is where the pre-positioning
   and the search behave differently (nothing to gain when the head is never idle). A
   0.90 point would cover the saturated regime the hypotheses mention.
5. *The upstream field length is fixed today.* `belts` sees the whole entry zone
   (10 cm, 3.3 s at 3 cm/s). Treatment B as written ("a preceding stretch of defined
   length") needs a visible-length parameter in `simulation/observation.py`, and
   Paper 2 gains a curve (value against visible length) rather than one point.
6. *Economic translation.* The cell does not exist yet (`docs/assumptions.md`, top
   note), so the annual-cost estimate can only be scenario-based; the penalty weights
   λ_N = 10 s and λ_T = 1 s are placeholders.

**Decision (owner, 2026-11-22).** The reference observation of Paper 1 is treatment
B (`belts` with `none`: the camera sees the parts on the belts, the head is assigned
only to parts already in the working zone), which is the level of every result of
`rl2.md` §8.4–§8.5. Paper 2 then measures A (`zone`) against B and C (`belts` with
`when_idle`) against B; the 0.7.9 training mix adds `zone` and `full` cells so one
network serves all treatments.

**First A/B/C/D measurement (0.7.9, `rl2.md` §8.6).** With the network trained on
all levels, on the nominal cells (30 seeds): B vs A −0.2 ±1.3 % (nothing), C vs B
−5.7 ±1.7 % at the design load and +0.9 ±1.6 % at the heavy load, D vs B +0.8 ±0.9 %
(design) and −1.2 ±1.2 % (heavy). The main hypothesis of Paper 2 therefore needs the
bursty/high-travel regimes to find any value of seeing earlier *without* acting
earlier; on the nominal cell the value is in the action, not the information.

**Pilot across regimes (0.7.10, `rl2.md` §8.7).** With the bursty generator and the
visible-length parameter in place, the same network on 8 regimes (real/heavy ×
Poisson/burst × nominal/slow head, 30 seeds): B vs A stays within ±0.9 points
everywhere, so bursts and travel do not create a value of information alone; the
value of the visibility is the pre-positioning it enables (C vs B −5.7 nominal, −3.0
slow, −3.1 burst, −4.0 both, 0 at the heavy load), and it saturates at about half the
entry zone (5 cm ≈ 1.7 s of belt travel). The hidden queue (D) is worth 4.4 points
only at the heavy load with a slow head. Suggested reframing of Paper 2: the value of
upstream visibility is the value of the action it enables, measured as a curve against
the visible length (in seconds of belt travel, against the head's travel time), with
the information-only contrast as a reported null result; the bursty/high-travel
hypothesis should be stated as "the value of acting early falls with bursts and slow
heads" unless the confirmatory bench with other burst settings says otherwise.

**Decisions taken after the pilot (owner, 2026-11-22).**

1. *Retraining.* Retrain the S network once, only if bursty arrivals and slow heads
   are part of the scope to be defended; keep the current network (`rl2_gen_s_d`) as
   the reference and choose between the two on development seeds, never on the
   confirmatory seeds. Extending the training mix changes `GENERAL` (burst parameters,
   laser speeds below 5 cm/s) and is a separate owner decision.
2. *Hypothesis.* Reformulated now (above), with the record that it came from the
   pilot. The confirmatory bench must test the coverage curve and show that, with the
   same actions, the benefit of seeing earlier is small enough for the design decision.
3. *Phase 4.* Launched only after the network, the regimes and the contrasts are
   frozen, once. Paper 1 prioritizes the direct comparison known-parts search →
   sampled-future teacher → network alone; Paper 2 prioritizes B vs A, C vs B and the
   coverage curve.

Reading of the two papers: Paper 1 explains how to decide fast; Paper 2 answers a
different engineering decision, how far it is worth looking and which capacity to act
makes that coverage useful.

**Where the plan's steps already stand.**

| step | status |
|---|---|
| 1 Freeze the base problem | model, KPI, information levels and checks exist; the bursty generator and the visible-length parameter exist since 0.7.10; what remains is to freeze the *paper's* instance formats and the 36-configuration bench (the load grid should add 0.90, review note 4) |
| 2 Build and test the search | not started as a pruned branch-and-bound; the sampled-future rollout and CP-SAT references exist |
| 3 Close the evaluation of Paper 1 | not started; phase 4 of `strong_rl_plan.md` (engine at `dt ≤ 0.001`, seeds 7000–7029, hindsight reference) is the same deliverable |
| 4 Prepare Paper 2 | masks, modes, causality tests, the visible-length parameter (`simulation.visible_length`) and the bursty generator (`arrivals.stochastic.distribution: burst`) exist (0.7.10); `experiments/rl2_information_bench.py` runs the treatments and the visible-length curve across load, arrivals and travel regimes |
| 5 Close the evaluation of Paper 2 | not started; a first A/B/C measurement on the nominal cell with the 0.7.6 network is in `docs/assumptions.md` C4; the pilot across regimes with the 0.7.9 network is in `docs/methods/rl2.md` §8.7 |
