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
