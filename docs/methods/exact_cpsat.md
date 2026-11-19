# Exact method: time-indexed model $\mathcal P_\Delta$ with CP-SAT

Owner: EXACT agent. Code: `src/optimization/solvers/cpsat.py`; tests: `tests/test_cpsat.py`.

Two benchmark methods are registered:

| method | model | intended use |
|---|---|---|
| `cpsat` | $\mathcal P_\Delta$ exactly as in `docs/formal_formulation.md` §2 (M1–M14) | certified optimum / bound of the formulation |
| `cpsat_ready` | $\mathcal P_\Delta$ + the simulator's dispatching rule (see §5) | best processing order *for the simulator*; model value ≈ replayed value |

```python
from optimization.solvers.cpsat import solve_cpsat
sol = solve_cpsat(inst, "avg_flow_time", time_limit=60, seed=0)          # = method "cpsat"
sol = solve_cpsat(inst, "makespan", 60, 0, ready_dispatch=True)          # = "cpsat_ready"
sol.sequence, sol.lower_bound, sol.proven_optimal, sol.info["model_objective"], sol.info["replay_objective"]
```

Supported objectives: `makespan`, `avg_flow_time`, `avg_waiting_time` (others, and composite
weight dicts, return a skipped `Solution`). Every parameter (geometry, belt speeds, conveyor
positions, laser speeds and home, motion model, spacing $\delta$, stop rule $\sigma$, end-of-zone
rule) is read from the `Config` of `instance_config(inst)`.

---

## 1. Discretisation and scaling

* **Time**: steps of $\Delta = 0.1$ s (`DEFAULT_STEP`, argument `step`). Releases are rounded
  **up** to the grid, $\bar r_j=\lceil r_j/\Delta\rceil$; processing times occupy
  $\bar p_j=\lceil p_j/\Delta\rceil$ steps (exact for the default $p_j=2.5$ s).
* **Positions**: integer *units* chosen as $1/\mathrm{lcm}$ of the denominators of
  $v_k\Delta$, $a$, $b$, $\delta$, $x_0$ and $0.1$ cm. With the default configuration the unit is
  **1 mm**: the belt advances $w_k=v_k\Delta=3$ units per running step, $a=100$, $b=450$,
  $\delta=40$, $x_0=275$.
* **Travel**: the laser covers $v_x\Delta = 10$ units of $x$ per step. Travel times are integer
  numbers of steps, $T_{ij}\ge\lceil |X_i-X_j|/(v_x\Delta)\rceil$ and
  $T_{ij}\ge\lceil|y_{k(i)}-y_{k(j)}|/(v_y\Delta)\rceil$ (simultaneous axes; the sequential model
  uses the sum). Rounding travel up keeps the triangle inequality (ceil is subadditive), so
  pairwise constraints remain valid.
* **Objective**: makespan in units of $\Delta/100$ with the true $p_j$
  ($C_{\max}\ge 100\,S_j+\lceil 100p_j/\Delta\rceil$); flow and waiting time as
  $\sum_j S_j$ plus the exact constant $\sum_j(p_j-r_j)$. Values are converted back to seconds.

## 2. Variables and constraints (mapping to M1–M14)

$H$ = horizon (steps), $t\in\{0,\dots,H-1\}$; $[\mathrm{est}_j,\mathrm{lst}_j]$ = start window of $j$ (§3).

| formulation | CP-SAT implementation |
|---|---|
| (M1) $\sum_t s_{jt}=1$ | `s[j][t]` Booleans on $[\mathrm{est}_j,\mathrm{lst}_j]$ + `AddExactlyOne`; $S_j=\sum_t t\,s_{jt}$ (integer var). Cumulative Booleans $c_{jt}=c_{j,t-1}+s_{jt}$ ("started by $t$") so that "processing at $t$" is the 2-term expression $c_{jt}-c_{j,t-\bar p_j}$. |
| (M2) odometer | `D[k][t]` integers, $D_{k0}=0$, $D_{k,t+1}=D_{kt}+w_k u_{kt}$, `u[k][t]` Booleans. |
| (M3) entry $E_j=\max(D_{k,\bar r_j},E_{\mathrm{pred}(j)}+\delta)$ | `AddMaxEquality` (exact max, so the $\lambda_j$ big-M pair is not needed). |
| (M4) $X_j=D_{k(j)}(S_j)-E_j$ | `AddElement(S_j, D[k], DS_j)`, $X_j=DS_j-E_j$ (CP-SAT expands the element into the indicator form $s_{jt}\Rightarrow DS_j=D_{kt}$, i.e. M4 without big-M). |
| (M5) $a\le X_j\le b$ (window = blocking C4) | domain of `X[j]`. |
| (M6) belt stopped during processing | $u_{kt}+\sum_{j:\sigma_{k,k(j)}=1}(c_{jt}-c_{j,t-\bar p_j})\le 1$; $\sigma$ = own conveyor or all conveyors from `simulation.conveyor_stop_policy`. |
| (M7) $w_{ij}+w_{ji}=1$ | one Boolean `w[i][j]` per pair, $w_{ji}$ is its negation. |
| (M8) $S_j\ge S_i+p_i+\tau_{ij}-M(1-w_{ij})$ | enforced linear constraints (`OnlyEnforceIf`, no big-M), all ordered pairs. |
| (M9) $S_j\ge\tau_{0j}$ | $S_j\ge T_{0j}$, $T_{0j}\ge$ travel from the home $(x_0,y_0)$. |
| travel-time bounds (M10–M11) | `AddAbsEquality` for $\lvert X_i-X_j\rvert$, $\lvert X_j-x_0\rvert$; $T\cdot\mathrm{den}\ge\mathrm{num}\cdot\lvert\Delta X\rvert$ (exact integer form of $\lvert\Delta X\rvert/v_x$), $T\ge\lceil\lvert\Delta y\rvert/(v_y\Delta)\rceil$. |
| (M12) non-idling | $u_{kt}+\sum_{j:\sigma}(\text{processing}_{jt})+\sum_{j\in J_k}\beta_{jt}\ge 1$. |
| (M13) $\beta_{jt}\le 1-\sum_{t'\le t}s_{jt'}$ | $\beta_{jt}\Rightarrow c_{jt}=0$. |
| (M14) $D_{k,t}-E_j\ge b-v_k\Delta-M(1-\beta_{jt})$ | $\beta_{jt}\Rightarrow D_{kt}-E_j\ge b-w_k+1$ (strict version of the slack: a belt may stop only when one more step would carry the part beyond $b$; then it *must* stop because of M5, so the belt dynamics are fully determined by the starts, as in the simulator). $\beta_{jt}$ only exists for $t\in[\bar r_j+\lceil(b-w_k+1)/w_k\rceil,\ \mathrm{lst}_j]$. With `unprocessed_at_zone_end: pass` no $\beta$ is created. |
| `NoOverlap` (redundant) | interval $[S_j,\,S_j+\bar p_j+g_j)$ with $g_j=\min_{i\neq j}g_{ji}$, see §3. |

## 3. Solving strategy (what makes it work)

A first version (plain model, horizon = heuristic makespan + slack, partial hint) did not
even find a feasible solution for $n=6$ in 30 s. The following ingredients changed that.

1. **Discrete decoder** (`decode_order`): for a processing order it computes the earliest
   schedule *of $\mathcal P_\Delta$ itself* (same rounding, belt rules, blocking slack). It is
   the discrete twin of the simulator's fixed-sequence decoder.
2. **Warm start**: the realised orders of the four dispatching policies are decoded, the best
   one is improved by a first-improvement swap descent (10 % of the time limit) and given to
   CP-SAT as a hint. Since CP-SAT drops partial hints that it cannot complete quickly, the hint
   is completed first (`complete_hint`: a solve with the hinted variables fixed, pure
   propagation, < 1 s) and every variable is hinted. CP-SAT then starts from a feasible
   solution in all runs. If CP-SAT ends with status UNKNOWN, the warm start is returned.
3. **Start windows that never cut off an optimum** (`start_windows`), instead of an
   arbitrary horizon: $\mathrm{est}_j=\max(\bar r_j+\lceil a/w_k\rceil,\lceil|y_k-y_0|/(v_y\Delta)\rceil)$;
   for makespan $\mathrm{lst}_j=\bar C^{\rm inc}_{\max}-\bar p_j$; for flow/waiting time
   $\mathrm{lst}_j=\sum_i S^{\rm inc}_i-\sum_{i\ne j}\mathrm{est}_i$. Every solution at least as
   good as the incumbent satisfies these windows, so the optimum and the bound of
   $\mathcal P_\Delta$ are those of the unrestricted model. $H=\max_j(\mathrm{lst}_j+\bar p_j)$.
4. **Valid redundant constraints on laser idle time.** If part $j$ is processed after $i$:
   * different conveyors: $S_j-C_i\ge\lceil|y_{k(i)}-y_{k(j)}|/(v_y\Delta)\rceil$;
   * same conveyor: at $C_i$ the laser is at $X_i$ and part $j$ is at least $\delta$ away
     along the belt (rigid belt, $|E_i-E_j|\ge\delta$, and the belt is stopped during $i$
     because $\sigma_{kk}=1$); laser and part close that distance at most at $v_x+v_k$, so
     $S_j-C_i\ge g_{ij}=\lceil\delta/((v_x+v_k)\Delta)\rceil$ (4 steps by default).
     This is added to the disjunction (M8) for same-belt pairs. The model alone implies it
     only through the odometer chain, which propagation does not see.
   * the redundant `NoOverlap` uses intervals extended by $g_j=\min_i g_{ji}$, which lets the
     energetic reasoning of CP-SAT account for travel. With these two constraints the bound
     of $n=10$, seed 1, makespan went from 45.5 to the optimum 54.5 within 30 s, and of $n=8$,
     seed 2, flow time from 6.09 to 7.87 (optimum ≤ 10.80).
5. **Parameters**: 8 workers, `cp_model_probing_level = 0` — with probing, presolve spent
   up to 10–15 s per solve creating ~600k binary clauses before the search started (2–3x
   slower on $n=5$, no effect on harder instances).

Model size (default config, $\Delta=0.1$): $n=6$: 3 000–9 200 variables; $n=15$: 17 700–68 300
(`info["num_variables"]`, `start_vars`, `beta_vars`, `horizon_steps`; see §6).

## 4. Lower bound: meaning and relation to $\mathcal P$

`Solution.lower_bound` (method `cpsat`) is CP-SAT's proven best objective bound of
$\mathcal P_\Delta$, converted to seconds. It is a valid bound **for $\mathcal P_\Delta$** (start
times on the 0.1 s grid, releases rounded up, travel rounded up to whole steps). With respect to
the continuous model $\mathcal P$:

* $\mathcal P_\Delta$ is *not* a relaxation of $\mathcal P$: rounding $r_j$ and travel up makes it
  slightly pessimistic, so the $\mathcal P_\Delta$ optimum can exceed the $\mathcal P$ optimum by
  a few $\Delta$ per chain of back-to-back operations (each move loses $<\Delta$; a wait for a
  part resets the error);
* conversely, the non-idling slack lets a belt stop up to one step ($v_k\Delta$ = 0.3 cm) before
  $b$, slightly optimistic.

So the bound is valid for $\mathcal P$ only up to $O(\Delta)$. Since the formulation is in addition
a relaxation of the simulator (§5), the `cpsat` bound is usually well below replayed schedules,
but not always: in 1 of 30 runs (n=8, seed 3, makespan) the proven $\mathcal P_\Delta$ optimum
40.0 s exceeded the replay of the same order (39.93 s) by 0.07 s. Consumers that need a strictly
valid bound for simulated schedules should subtract a discretisation margin (a few $\Delta$);
no rigorous margin is derived here. `cpsat_ready` returns `lower_bound = None` (its model is a
*restriction* of $\mathcal P$, so its bound is not a bound for $\mathcal P$); its bound is in
`info["model_bound"]`.

## 5. Discrepancy between the formulation and the simulator (important)

**The simulator's laser cannot anticipate a part that is not yet in the laser zone; the
formulation can.** In the engine, a part becomes a dispatch candidate only when it is READY
($x_j\ge a$); an idle laser stays at its last processing point until then and only then starts
the interception. Constraint (C2)/(M8)–(M9) only require
$S_j\ge C_{\rm prev}+\tau(P_{\rm prev},P_j)$, so in $\mathcal P$ (and $\mathcal P_\Delta$) the laser
may travel *towards the zone entrance* while the part is still upstream and be waiting at
$(a,y_k)$ when it arrives. §3 of `formal_formulation.md` states that the decoder $\Phi$ "is exactly
what the simulator does" with the semantics of $\mathcal P$; this is not the case whenever the
laser is idle before the next part enters the zone.

Example (single part, conveyor 1, $r=0$): $\mathcal P$: $S=a/v=3.33$ s ($\mathcal P_\Delta$: 3.4 s),
$C=5.9$ s. Simulator: the laser leaves the home only at 3.33 s, meets the part at
$x\approx14$ cm after 1.35 s: $C=7.18$ s.

Consequences:

* $\mathcal P$ is a **relaxation** of the set of schedules the simulator can produce (with any
  order). The `cpsat` value is therefore typically *below* its replay
  (replay − model between −0.07 and +1.95 s, mean +0.72 s, §6) and its order is optimal for
  the relaxation, not necessarily for the simulator. Its bound remains a (weaker) bound for all
  simulated methods, up to the $O(\Delta)$ caveat of §4.
* `cpsat_ready` adds the simulator's rule: ready time $R_j$ = first step with
  $D_{k(j),t}-E_j\ge a$ (two `AddElement`s: in the zone at $R_j$, not at $R_j-1$), immediate
  successor literals $z_{ij}$ with `AddCircuit` (node 0 = home/end), $z_{ij}\Rightarrow w_{ij}$ and
  $z_{ij}\Rightarrow S_j\ge R_j+T_{ij}$, $z_{0j}\Rightarrow S_j\ge R_j+T_{0j}$, i.e.
  $S_j\ge\max(C_i,R_j)+\tau_{ij}$ for the *immediate* predecessor (it cannot be imposed pairwise).
  With it, model and replay agree within the discretisation error (§6).
* Remaining semantic difference: the decoder never waits deliberately (semi-active timing),
  while $\mathcal P_\Delta$ may start a part later than its earliest time. In all 30 runs the replay
  of `cpsat_ready` was *better* than its model value by 0.06–0.38 s (pessimistic rounding), so
  deliberate waiting produced no measurable advantage on these instances.

Suggested fix for the formulation (for the integrator): either state in §1/§3 that the laser may
anticipate and that the simulator is a restriction of $\mathcal P$, or add the ready-time
condition $S_{\pi(i+1)}\ge\max(C_{\pi(i)},R_{\pi(i+1)})+\tau(P_{\pi(i)},P_{\pi(i+1)})$ to (C2).

## 6. Validation and scalability (n = 6–15, seeds 1–3, 60 s, 8 workers)

Setup: `generate_instance(n, seed)` (stochastic arrivals, default config, $p_j=2.5$ s),
$n\in\{6,8,10,12,15\}$, seeds 1–3, time limit 60 s per solve (warm start included), 8 CP-SAT
workers, seed 0. Both variants ran concurrently on a 16-thread Windows laptop. Every returned
order was replayed with `evaluate_sequence(inst, seq, objective, dt=0.01, check=True)`.
`nearest_job` = `evaluate_policy(inst, "nearest_job")` (the `policy:nearest_job` method).
Script and raw JSON lines were kept outside the repository (not part of the owned files).

Columns: *optimal* = CP-SAT status OPTIMAL; *gap* = (model value − bound)/model value, in the
model ($\mathcal P_\Delta$, resp. $\mathcal P_\Delta$+ready); *model value* = objective of the
CP-SAT solution in the model; *replayed* = simulator value of the returned order; *replay vs NJ*
= mean relative difference of the replayed value w.r.t. `nearest_job` (negative = better).
Means over the 3 seeds.

| variant | objective | n | optimal | status | time s (mean / max) | gap % (mean / max) | model value | replayed | nearest_job | replay vs NJ % | replay − model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpsat | avg_flow_time | 6 | 3/3 | OPT | 12.7 / 31.2 | 0.00 / 0.00 | 8.751 | 9.427 | 9.373 | +0.47 | +0.676 |
| cpsat | avg_flow_time | 8 | 2/3 | FEAS/OPT | 33.0 / 60.5 | 6.90 / 20.71 | 9.111 | 9.498 | 9.904 | -3.95 | +0.388 |
| cpsat | avg_flow_time | 10 | 0/3 | FEAS | 60.5 / 60.7 | 19.89 / 25.70 | 9.656 | 10.312 | 10.426 | -1.06 | +0.656 |
| cpsat | avg_flow_time | 12 | 0/3 | FEAS | 60.6 / 60.7 | 35.59 / 42.67 | 10.427 | 11.105 | 11.317 | -1.89 | +0.677 |
| cpsat | avg_flow_time | 15 | 0/3 | FEAS | 61.1 / 61.3 | 42.86 / 56.61 | 11.686 | 12.407 | 12.637 | -1.87 | +0.721 |
| cpsat | makespan | 6 | 3/3 | OPT | 3.8 / 6.1 | 0.00 / 0.00 | 34.400 | 35.463 | 35.540 | -0.24 | +1.063 |
| cpsat | makespan | 8 | 3/3 | OPT | 14.0 / 19.9 | 0.00 / 0.00 | 42.033 | 42.370 | 43.713 | -3.21 | +0.337 |
| cpsat | makespan | 10 | 2/3 | FEAS/OPT | 34.6 / 60.2 | 3.04 / 9.12 | 50.300 | 51.063 | 51.733 | -1.40 | +0.763 |
| cpsat | makespan | 12 | 1/3 | FEAS/OPT | 44.1 / 60.5 | 10.84 / 16.42 | 59.000 | 59.903 | 60.620 | -1.18 | +0.903 |
| cpsat | makespan | 15 | 1/3 | FEAS/OPT | 45.1 / 61.1 | 13.06 / 25.18 | 71.567 | 72.590 | 73.413 | -1.16 | +1.023 |
| cpsat_ready | avg_flow_time | 6 | 3/3 | OPT | 14.2 / 27.3 | 0.00 / 0.00 | 9.451 | 9.264 | 9.373 | -1.16 | -0.187 |
| cpsat_ready | avg_flow_time | 8 | 3/3 | OPT | 34.6 / 58.4 | 0.00 / 0.00 | 9.681 | 9.498 | 9.904 | -3.95 | -0.183 |
| cpsat_ready | avg_flow_time | 10 | 0/3 | FEAS | 60.6 / 60.8 | 36.63 / 39.51 | 10.506 | 10.312 | 10.426 | -1.06 | -0.194 |
| cpsat_ready | avg_flow_time | 12 | 0/3 | FEAS | 60.7 / 60.8 | 44.66 / 47.35 | 11.316 | 11.124 | 11.317 | -1.73 | -0.192 |
| cpsat_ready | avg_flow_time | 15 | 0/3 | FEAS | 60.9 / 61.0 | 51.12 / 59.27 | 12.639 | 12.425 | 12.637 | -1.74 | -0.214 |
| cpsat_ready | makespan | 6 | 3/3 | OPT | 4.9 / 7.7 | 0.00 / 0.00 | 35.200 | 35.050 | 35.540 | -1.53 | -0.150 |
| cpsat_ready | makespan | 8 | 3/3 | OPT | 16.8 / 18.8 | 0.00 / 0.00 | 42.567 | 42.370 | 43.713 | -3.21 | -0.197 |
| cpsat_ready | makespan | 10 | 1/3 | FEAS/OPT | 55.5 / 60.5 | 5.97 / 16.69 | 50.967 | 50.787 | 51.733 | -1.89 | -0.180 |
| cpsat_ready | makespan | 12 | 0/3 | FEAS | 60.5 / 60.6 | 12.65 / 19.98 | 59.733 | 59.533 | 60.620 | -1.82 | -0.200 |
| cpsat_ready | makespan | 15 | 0/3 | FEAS | 60.7 / 60.8 | 15.47 / 26.43 | 72.767 | 72.520 | 73.413 | -1.25 | -0.247 |

**Validation outcome**

* All 60 returned orders replay **feasibly** with **0 checker violations**.
* `cpsat_ready`: replay − model $\in[-0.38,-0.06]$ s (mean −0.19 s). The model is slightly
  pessimistic, as expected from rounding releases and every travel up to whole steps of 0.1 s
  (and the ready time $R_j$ to the grid); no case of a model value much better than its replay.
* `cpsat`: replay − model $\in[-0.07,+1.95]$ s (mean +0.72 s). The positive part is the
  anticipation effect of §5 (the model lets the laser pre-position; the simulator cannot), the
  small negative values are discretisation. This is the "model much better than replay" situation
  the brief asked to investigate: it is not a coding error but the formulation/simulator
  discrepancy of §5 (it disappears with `ready_dispatch`, where replay ≈ model).
* Because of the same effect, the `cpsat` order is optimal for the relaxation but can replay
  *worse* than `nearest_job` (n=6, seed 2, flow: 11.504 vs 11.342; n=10, seed 1, makespan:
  56.45 vs 55.90), while `cpsat_ready` never replayed worse than `nearest_job` by more than the
  discretisation (worst: +0.047 s, n=6 seed 2 flow, model-optimal for $\mathcal P_\Delta$+ready).
* Bound vs replays: the `cpsat` bound was ≤ every replayed value (its own, `cpsat_ready`'s and
  `nearest_job`'s) in 29 of 30 runs; the exception (n=8, seed 3, makespan: proven optimum
  40.0 of $\mathcal P_\Delta$ vs replay 39.93 of the same order) shows that the $\mathcal P_\Delta$
  bound can exceed a feasible continuous value by $O(\Delta)$ (§4).

**Scalability observations**

* Proven optimal: all n = 6 and 8 instances with `cpsat_ready` (12/12) and 11/12 with `cpsat`;
  makespan up to n = 15 in some instances (e.g. n=15 seed 2, 13.5 s), flow time never beyond n = 8.
* For flow time the bound is weak for n ≥ 10 (gap 20–60 %), essentially the trivial
  "every part starts at its earliest time" bound: CP-SAT does not see the laser travel early
  enough. Makespan gaps stay at 3–26 %.
* For n ≥ 10 CP-SAT rarely improves the warm start: improvements over the warm start in 7/30
  runs (`cpsat`) and 2/30 (`cpsat_ready`, n=10 seed 3 makespan 49.4 → 47.5 proven optimal;
  n=12 seed 3 makespan 57.7 → 56.5). Most of the gain over `nearest_job` at large n comes from
  the warm start (best policy order + swap descent evaluated with the discrete decoder), not from
  the tree search. The value of the exact method at large n is therefore mainly the bound.
* Model size grows with $n\cdot H$; at n = 15: 17 700–61 200 variables (`cpsat`),
  18 400–68 300 (`cpsat_ready`), $H$ = 680–1 810 steps, 3 200–17 800 $\beta$ variables. At n = 6:
  3 000–9 200 variables.

<details><summary>Per-instance results</summary>

| variant | obj | n | seed | status | time | model | bound | replay | NJ | warm start | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpsat | avg_flow_time | 6 | 1 | OPTIMAL | 1.3 | 6.540 | 6.540 | 7.169 | 7.169 | 6.540 | True (0) |
| cpsat | makespan | 6 | 1 | OPTIMAL | 1.5 | 40.800 | 40.800 | 41.530 | 41.530 | 40.800 | True (0) |
| cpsat | avg_flow_time | 6 | 2 | OPTIMAL | 31.2 | 10.921 | 10.921 | 11.504 | 11.342 | 11.004 | True (0) |
| cpsat | makespan | 6 | 2 | OPTIMAL | 6.1 | 30.200 | 30.200 | 31.410 | 31.640 | 30.300 | True (0) |
| cpsat | avg_flow_time | 6 | 3 | OPTIMAL | 5.8 | 8.793 | 8.793 | 9.608 | 9.608 | 8.793 | True (0) |
| cpsat | makespan | 6 | 3 | OPTIMAL | 3.7 | 32.200 | 32.200 | 33.450 | 33.450 | 32.400 | True (0) |
| cpsat | avg_flow_time | 8 | 1 | OPTIMAL | 5.9 | 7.626 | 7.626 | 8.295 | 8.295 | 7.626 | True (0) |
| cpsat | makespan | 8 | 1 | OPTIMAL | 5.6 | 47.400 | 47.400 | 48.250 | 48.250 | 47.400 | True (0) |
| cpsat | avg_flow_time | 8 | 2 | FEASIBLE | 60.5 | 10.803 | 8.565 | 11.134 | 11.392 | 10.803 | True (0) |
| cpsat | makespan | 8 | 2 | OPTIMAL | 19.9 | 38.700 | 38.700 | 38.930 | 40.240 | 38.700 | True (0) |
| cpsat | avg_flow_time | 8 | 3 | OPTIMAL | 32.6 | 8.903 | 8.903 | 9.066 | 10.027 | 8.903 | True (0) |
| cpsat | makespan | 8 | 3 | OPTIMAL | 16.4 | 40.000 | 40.000 | 39.930 | 42.650 | 40.000 | True (0) |
| cpsat | avg_flow_time | 10 | 1 | FEASIBLE | 60.3 | 9.044 | 7.854 | 9.745 | 9.745 | 9.044 | True (0) |
| cpsat | makespan | 10 | 1 | OPTIMAL | 36.0 | 54.500 | 54.500 | 56.450 | 55.900 | 54.500 | True (0) |
| cpsat | avg_flow_time | 10 | 2 | FEASIBLE | 60.7 | 10.076 | 7.486 | 10.451 | 10.740 | 10.076 | True (0) |
| cpsat | makespan | 10 | 2 | OPTIMAL | 7.5 | 49.000 | 49.000 | 49.410 | 49.870 | 49.000 | True (0) |
| cpsat | avg_flow_time | 10 | 3 | FEASIBLE | 60.5 | 9.848 | 7.798 | 10.739 | 10.793 | 9.848 | True (0) |
| cpsat | makespan | 10 | 3 | FEASIBLE | 60.2 | 47.400 | 43.078 | 47.330 | 49.430 | 48.200 | True (0) |
| cpsat | avg_flow_time | 12 | 1 | FEASIBLE | 60.5 | 10.918 | 6.260 | 11.583 | 11.778 | 10.985 | True (0) |
| cpsat | makespan | 12 | 1 | FEASIBLE | 60.4 | 60.600 | 50.647 | 61.270 | 62.410 | 60.600 | True (0) |
| cpsat | avg_flow_time | 12 | 2 | FEASIBLE | 60.7 | 9.644 | 7.419 | 10.142 | 10.383 | 9.644 | True (0) |
| cpsat | makespan | 12 | 2 | OPTIMAL | 11.2 | 59.900 | 59.900 | 61.010 | 61.010 | 59.900 | True (0) |
| cpsat | avg_flow_time | 12 | 3 | FEASIBLE | 60.7 | 10.719 | 6.319 | 11.589 | 11.789 | 10.719 | True (0) |
| cpsat | makespan | 12 | 3 | FEASIBLE | 60.5 | 56.500 | 47.398 | 57.430 | 58.440 | 56.500 | True (0) |
| cpsat | avg_flow_time | 15 | 1 | FEASIBLE | 61.1 | 13.708 | 5.948 | 14.383 | 14.383 | 13.708 | True (0) |
| cpsat | makespan | 15 | 1 | FEASIBLE | 60.7 | 71.100 | 53.200 | 72.210 | 72.000 | 71.400 | True (0) |
| cpsat | avg_flow_time | 15 | 2 | FEASIBLE | 61.3 | 9.239 | 7.066 | 9.861 | 10.054 | 9.239 | True (0) |
| cpsat | makespan | 15 | 2 | OPTIMAL | 13.5 | 75.800 | 75.800 | 76.940 | 76.940 | 75.800 | True (0) |
| cpsat | avg_flow_time | 15 | 3 | FEASIBLE | 60.8 | 12.111 | 6.245 | 12.977 | 13.476 | 12.171 | True (0) |
| cpsat | makespan | 15 | 3 | FEASIBLE | 61.1 | 67.800 | 58.301 | 68.620 | 71.300 | 67.800 | True (0) |
| cpsat_ready | avg_flow_time | 6 | 1 | OPTIMAL | 2.6 | 7.324 | 7.324 | 7.169 | 7.169 | 7.324 | True (0) |
| cpsat_ready | makespan | 6 | 1 | OPTIMAL | 1.5 | 41.600 | 41.600 | 41.530 | 41.530 | 41.600 | True (0) |
| cpsat_ready | avg_flow_time | 6 | 2 | OPTIMAL | 27.3 | 11.637 | 11.637 | 11.389 | 11.342 | 11.637 | True (0) |
| cpsat_ready | makespan | 6 | 2 | OPTIMAL | 5.6 | 30.700 | 30.700 | 30.470 | 31.640 | 30.700 | True (0) |
| cpsat_ready | avg_flow_time | 6 | 3 | OPTIMAL | 12.7 | 9.393 | 9.393 | 9.235 | 9.608 | 9.393 | True (0) |
| cpsat_ready | makespan | 6 | 3 | OPTIMAL | 7.7 | 33.300 | 33.300 | 33.150 | 33.450 | 33.300 | True (0) |
| cpsat_ready | avg_flow_time | 8 | 1 | OPTIMAL | 22.5 | 8.439 | 8.439 | 8.295 | 8.295 | 8.439 | True (0) |
| cpsat_ready | makespan | 8 | 1 | OPTIMAL | 14.0 | 48.400 | 48.400 | 48.250 | 48.250 | 48.400 | True (0) |
| cpsat_ready | avg_flow_time | 8 | 2 | OPTIMAL | 58.4 | 11.378 | 11.378 | 11.134 | 11.392 | 11.378 | True (0) |
| cpsat_ready | makespan | 8 | 2 | OPTIMAL | 18.8 | 39.200 | 39.200 | 38.930 | 40.240 | 39.200 | True (0) |
| cpsat_ready | avg_flow_time | 8 | 3 | OPTIMAL | 22.9 | 9.228 | 9.228 | 9.066 | 10.027 | 9.228 | True (0) |
| cpsat_ready | makespan | 8 | 3 | OPTIMAL | 17.5 | 40.100 | 40.100 | 39.930 | 42.650 | 40.100 | True (0) |
| cpsat_ready | avg_flow_time | 10 | 1 | FEASIBLE | 60.6 | 9.914 | 6.484 | 9.745 | 9.745 | 9.914 | True (0) |
| cpsat_ready | makespan | 10 | 1 | FEASIBLE | 60.4 | 55.800 | 46.487 | 55.650 | 55.900 | 55.800 | True (0) |
| cpsat_ready | avg_flow_time | 10 | 2 | FEASIBLE | 60.8 | 10.706 | 6.476 | 10.451 | 10.740 | 10.706 | True (0) |
| cpsat_ready | makespan | 10 | 2 | FEASIBLE | 60.5 | 49.600 | 49.000 | 49.380 | 49.870 | 49.600 | True (0) |
| cpsat_ready | avg_flow_time | 10 | 3 | FEASIBLE | 60.4 | 10.898 | 6.998 | 10.739 | 10.793 | 10.898 | True (0) |
| cpsat_ready | makespan | 10 | 3 | OPTIMAL | 45.7 | 47.500 | 47.500 | 47.330 | 49.430 | 49.400 | True (0) |
| cpsat_ready | avg_flow_time | 12 | 1 | FEASIBLE | 60.8 | 11.810 | 6.218 | 11.641 | 11.778 | 11.810 | True (0) |
| cpsat_ready | makespan | 12 | 1 | FEASIBLE | 60.4 | 61.600 | 49.293 | 61.270 | 62.410 | 61.600 | True (0) |
| cpsat_ready | avg_flow_time | 12 | 2 | FEASIBLE | 60.7 | 10.369 | 6.253 | 10.142 | 10.383 | 10.369 | True (0) |
| cpsat_ready | makespan | 12 | 2 | FEASIBLE | 60.5 | 61.100 | 59.901 | 61.010 | 61.010 | 61.100 | True (0) |
| cpsat_ready | avg_flow_time | 12 | 3 | FEASIBLE | 60.6 | 11.769 | 6.244 | 11.589 | 11.789 | 11.769 | True (0) |
| cpsat_ready | makespan | 12 | 3 | FEASIBLE | 60.6 | 56.500 | 47.449 | 56.320 | 58.440 | 57.700 | True (0) |
| cpsat_ready | avg_flow_time | 15 | 1 | FEASIBLE | 61.0 | 14.601 | 5.948 | 14.383 | 14.383 | 14.601 | True (0) |
| cpsat_ready | makespan | 15 | 1 | FEASIBLE | 60.8 | 72.300 | 53.194 | 72.000 | 72.000 | 72.300 | True (0) |
| cpsat_ready | avg_flow_time | 15 | 2 | FEASIBLE | 60.8 | 10.059 | 6.033 | 9.861 | 10.054 | 10.059 | True (0) |
| cpsat_ready | makespan | 15 | 2 | FEASIBLE | 60.8 | 77.000 | 75.800 | 76.940 | 76.940 | 77.000 | True (0) |
| cpsat_ready | avg_flow_time | 15 | 3 | FEASIBLE | 60.8 | 13.258 | 6.091 | 13.031 | 13.476 | 13.258 | True (0) |
| cpsat_ready | makespan | 15 | 3 | FEASIBLE | 60.4 | 69.000 | 56.287 | 68.620 | 71.300 | 69.000 | True (0) |

</details>

## 7. Tests (`tests/test_cpsat.py`, ~50 s)

* hand-made instances with optimum known by reasoning: one part (laser waits at the zone
  entrance, $S=\lceil a/(v\Delta)ceil\Delta$), two independent parts on one belt; values,
  `proven_optimal` and bound checked exactly (expected values derived from the config);
* `ready_dispatch` model vs replay on the single part (within 5Δ);
* a burst on one belt (blocking) with both stop rules: feasible replay, 0 violations,
  bound ≤ replay + 5Δ;
* agreement with exhaustive search: `baselines.exhaustive` on n=4 and the same enumeration
  (both objectives in one pass) on n=5, seeds 1 and 3: `cpsat` model value ≤ exhaustive + 5Δ,
  `cpsat_ready` replay within 5Δ of the exhaustive optimum, model ≈ replay;
* bound ≤ replayed objective + 5Δ, replays feasible; decoder respects $[a,b]$; registration.

## 8. Limitations

* Size grows with $n\times H$: start, cumulative and $\beta$ Booleans per part and step, plus
  $O(n^2)$ pair variables ($O(n^2)$ successor literals for `cpsat_ready`). The flow-time windows
  derived from the incumbent are wide for congested instances, which inflates the model.
* The LP/propagation bound is weak for flow time on congested instances (laser travel is only
  seen through the order-dependent disjunctions), so optimality is proven only for small $n$.
* Discretisation: $\Delta=0.1$ s, releases and travel rounded up; values differ from the
  continuous model by a few $\Delta$ (§4). Smaller $\Delta$ is possible (`step`) but grows $H$.
* Objectives: makespan, mean flow and mean waiting time only (lateness/travel distance/downtime
  are straightforward extensions of §2 but not implemented).
* `ready_dispatch` encodes the fixed-sequence dispatching rule; it does not model online
  information restrictions (it is still clairvoyant).
