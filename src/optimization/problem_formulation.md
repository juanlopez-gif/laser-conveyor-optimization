# Problem formulation (draft for the optimization phase)

## Classification

The problem combines features of:

* **Scheduling** of jobs under a time-based objective.
* **Single shared resource scheduling**: one "machine" (the laser) for all jobs.
* **Sequence-dependent setup / travel times**: the time before processing `j` depends on
  where the previous job `i` ended (laser travel), and also on *when*, because the part
  moves.
* **Dynamic job arrivals**: jobs are released at `r_j` and are not known in advance
  (online version) or are known (offline / fixed instance).
* **Blocking**: an unprocessed part at the end of the zone stops its conveyor and every
  part behind it.
* **Material handling**: conveyors transport the jobs; position is a function of time
  and of the stops.
* **Hybrid / discrete-event systems**: continuous dynamics (positions) + discrete events
  (start/end of processing, stops).

In approximate Graham notation: `1 | r_j, s_ij(t), blocking, time windows | C_max` (and
variants with `ΣF_j`, `L_max`, multiple objectives). Even `1 | r_j | L_max` and
`1 | s_ij | C_max` (equivalent to a TSP) are NP-hard.

## Data (offline instance)

* `J`: parts; for each `j`: conveyor `k(j)`, release `r_j`, processing `p_j`, due date `d_j`.
* Geometry: `a`, `b`, `y_k`, `v_k`; laser `v_x`, `v_y`, initial position `(x_0, y_0)`.

## Decision variables

See `decision_variables.md`. Core:

* `z_ij ∈ {0,1}`: `j` is processed immediately after `i` (sequence).
* `S_j ≥ 0`: processing start of `j`; `C_j = S_j + p_j`.
* `X_j ∈ [a, b]`: x position where `j` is processed (depends on how far its conveyor moved).
* (optional) `u_k(t) ∈ {0,1}`: run/stop of conveyor `k` if the stopping rule is optimized.

## Constraints

1. **One job at a time** (disjunctive resource): for every pair `i ≠ j`,
   `S_j ≥ C_i + τ_ij − M(1 − z_ij)` and `Σ_j z_ij ≤ 1`, `Σ_i z_ij ≤ 1` (plus dummy
   start/end jobs). In CP-SAT: `NoOverlap` over intervals `[S_j − τ_j, C_j)`.
2. **Travel time included:** `τ_ij = max(|X_j − X_i| / v_x, |y_k(j) − y_k(i)| / v_y)`
   (or the sum, sequential model). Non-linear → linearize with `τ_ij ≥ |X_j−X_i|/v_x`,
   `τ_ij ≥ |Δy|/v_y` and absolute values with two inequalities.
3. **Not before reaching the zone:** `S_j ≥ r_j + a / v_k(j)` (if the conveyor did not
   stop before) and, in general, `X_j ≥ a`.
4. **Physical reach:** `a ≤ X_j ≤ b`, `y_min ≤ y_k(j) ≤ y_max`.
5. **Part position = integral of its conveyor's motion:**
   `X_j = v_k · (S_j − e_j − stopped time of k between e_j and S_j)`, with `e_j` the
   belt entry time. This couples all jobs of a conveyor and is the hardest to model.
6. **The conveyor stops during processing:** `u_k(t) = 0` for `t ∈ [S_j, C_j)` if
   `k = k(j)` (default rule).
7. **Blocking (block mode):** if `X_j` would reach `b` before `S_j`, conveyor `k(j)`
   stays stopped from that moment until `S_j`.
8. **Physical FIFO within a conveyor:** parts do not overtake each other on the belt
   (but the laser may process them in a different order).

## Objectives

`C_max`, `Σ F_j / |J|`, `Σ W_j / |J|`, total laser distance `Σ_ij z_ij · dist_ij`,
`Σ_k T_stop,k`, throughput, `L_max`; or the composite
`α C_max + β D + γ T_stop + δ W̄`.

## Useful simplifications to start with

* **Small fixed instance** (5–15 parts): enumerate sequences / DP over subsets
  (`O(n² 2ⁿ)`) using the **simulator as the exact evaluator** of a given sequence. This
  avoids modelling constraint 5 explicitly.
* **Time discretization** (time-indexed MILP / CP with integer domains in tenths of a
  second): makes positions linear at the cost of size.
* **Rolling horizon**: optimize only the parts visible in the zone and re-optimize at
  every event.

## Candidate methods

MILP (Pyomo + HiGHS/CBC), CP-SAT (OR-Tools; intervals, `NoOverlap`, `AddMaxEquality`),
dynamic programming for small instances, greedy heuristics (4 already implemented),
genetic algorithms, simulated annealing and tabu search over the sequence (with the
simulator as evaluation function), reinforcement learning (the simulator as
environment) and rolling-horizon optimization.
