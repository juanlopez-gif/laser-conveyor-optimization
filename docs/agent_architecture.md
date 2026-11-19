# Agent architecture for solving the problem

## Can the problem be "solved" mathematically?

There is no closed-form optimal rule: the problem is NP-hard (see
[formal_formulation.md](formal_formulation.md), §5) and, in the cell, online. A rigorous
answer is built from four complementary results, each produced by one specialist agent:

| question | answer we can prove / measure | agent |
|---|---|---|
| What is the optimum of a given instance? | certified optimum (or optimality gap) of the time-indexed model $\mathcal P_\Delta$ | **exact** (CP-SAT) |
| How far can any schedule be from the optimum? | valid lower bounds + theorems for special cases | **theory** |
| Good solutions for instances too large for the exact model? | metaheuristics over the decoder $\Phi(\pi)$ | **metaheuristics** |
| What should the laser do *online*, without knowing the future? | a look-ahead (rollout) policy with measured gap to the clairvoyant optimum | **online** |

## Architecture

```mermaid
flowchart TB
    subgraph F[Shared foundation — built first, read-only for agents]
        INST[instances.py<br/>Instance, generate_instance, instance_config]
        DEC[policies/fixed_sequence.py<br/>decoder Φ(π)]
        EVAL[evaluate.py<br/>evaluate_sequence / evaluate_policy]
        CHK[schedule_check.py<br/>formal-model checker]
        REG[solvers/__init__.py<br/>@register_method, Solution]
        BEN[benchmark.py<br/>instance bank × methods → gaps]
    end
    EX[Agent EXACT<br/>solvers/cpsat.py] --> REG
    MH[Agent METAHEURISTICS<br/>solvers/metaheuristics.py] --> REG
    TH[Agent THEORY<br/>lower_bounds.py + theory.md] --> BEN
    ON[Agent ONLINE<br/>policies/rollout.py + solvers/online.py] --> REG
    EX & MH & ON --> EVAL --> CHK
    BEN --> INT[Integrator<br/>review, merge, full benchmark, report]
```

### Contracts (what makes parallel work safe)

* **One owner per file.** Each agent creates only its own files; shared files are
  read-only for agents. Integration happens afterwards.
* **Method contract:** `method(instance, objective, time_limit, seed) -> Solution`
  registered with `@register_method("name")` in its own module; discovered
  automatically by the benchmark.
* **Single source of truth for feasibility and cost:** every answer is replayed in
  the simulator (`evaluate_sequence` / `evaluate_policy`) at `dt = 0.01` and checked
  against the formal model (`schedule_check`). A solver's internal objective value is
  never trusted on its own.
* **Bounds contract:** `lower_bounds.best_lower_bound(instance, objective)` must be
  valid for *every* feasible schedule of $\mathcal P$; tests compare it against every
  simulated schedule.
* **Online fairness:** online policies may only use information available at decision
  time (parts already arrived), never the future arrival list.

### Agents

| agent | owns | delivers |
|---|---|---|
| EXACT | `src/optimization/solvers/cpsat.py`, `tests/test_cpsat.py`, `docs/methods/exact_cpsat.md` | CP-SAT model of $\mathcal P_\Delta$; optimal sequence, solver bound, optimality certificate for small $n$ |
| METAHEURISTICS | `src/optimization/solvers/metaheuristics.py`, `tests/test_metaheuristics.py`, `docs/methods/metaheuristics.md` | simulated annealing + iterated local search over permutations |
| THEORY | `src/optimization/lower_bounds.py`, `tests/test_lower_bounds.py`, `docs/methods/theory.md` | valid lower bounds; propositions/counterexamples for special cases |
| ONLINE | `src/policies/rollout.py`, `src/optimization/solvers/online.py`, `tests/test_rollout.py`, `docs/methods/online_rollout.md` | rollout policy (one-step look-ahead with a base policy) |
| INTEGRATOR (main session) | shared files, README, final benchmark | review, integration, results |

## Results (integration, 2026-11-19)

All four agents delivered; every method's output is replayed in the simulator and
passes the formal-model checker. Full suite: 120 tests pass. The benchmark below is reproduced with `PYTHONPATH=src python -m optimization.benchmark --n 6 8 10 --seeds 2 --objective avg_flow_time --time-limit 30 --tag integrated_flow` (results are not versioned).

### What can now be stated rigorously

1. **Exact optima (small instances).** CP-SAT proves optimality of the time-indexed
   models for $n\le 8$ (makespan up to some $n=15$ instances). The certificate is for the
   discretized model ($\Delta=0.1$ s, travel rounded up), i.e. optimal up to a few
   tenths of a second: on the integrated benchmark ILS found an order that replays
   0.04 s better than the `cpsat_ready` "optimum" — within discretization error.
2. **Valid lower bounds** (theory agent, L1–L6, proved): gaps between the best bound and
   the best known solution are 1–15 % for makespan and 7–50 % for mean flow time
   (flow bounds are weak for large $n$ and partly reflect the value of pre-positioning).
3. **Theorems for special cases:** light-traffic optimality (LT-P, LT-Φ), a
   counterexample showing belt order is not flow-optimal on a single conveyor, and
   counterexamples where FIFO / nearest job are not optimal.
4. **Model correction:** the simulator only dispatches to parts already in the laser
   zone, so it realises the restricted model $\mathcal P^{\mathrm{sim}}=\mathcal P+$(C7);
   $\mathcal P$ is a relaxation (value of pre-positioning ≈ 1.3 s for an isolated part).
   `docs/formal_formulation.md` and the PDF were corrected.

### Method comparison

Offline instances (`results/benchmark/integrated_flow/`, mean flow time, 30 s per
method, $n\in\lbrace 6,8,10\rbrace$, 2 seeds each): ILS/SA are best on average (gap to the
best bound 12.4 %), then CP-SAT, FIFO and weighted score (12.5 %), nearest job (13.5 %),
round robin (23.2 %). At this size and load the differences are small (< 1 % except round
robin). On larger instances (n = 20–40, 120 s) ILS improves mean flow time by 2–5 % over
nearest job (`docs/methods/metaheuristics.md`).

Online operation (30 stochastic seeds, 300 s, same seeds for every policy;
`docs/methods/online_rollout.md`):

| load | metric | nearest_job | rollout | rollout better on |
|---|---|---|---|---|
| default | mean flow time [s] | 19.95 ± 2.32 | **19.09 ± 2.15** | 28/30 seeds |
| default | makespan [s] | 319.86 ± 3.44 | **317.61 ± 3.31** | 25/30 |
| heavy | mean flow time [s] | 65.56 ± 5.81 | **62.97 ± 5.92** | 25/30 |
| heavy | makespan [s] | 402.90 ± 10.80 | **398.17 ± 10.14** | 25/30 |

### Structural insight (metaheuristics agent)
Good sequences keep **belt order within each conveyor** and only interleave across
conveyors. Nearest job loses time by serving a later part of a conveyor first, which
leaves the earlier part to block the belt. A natural next online rule is
"nearest conveyor, but its first unprocessed part".

### Open items
* Pre-positioning the laser (allowed in $\mathcal P$, not in the simulator) is worth
  ≈ 1–1.5 s per isolated part: a candidate engine extension.
* Flow-time lower bounds for large $n$ are weak (windowed DP ignores cross-window
  contention).
* Engine refactor suggested by the online agent: an explicit `Simulation.fork()` API
  instead of the deepcopy + forced-first trick.
