# Optimization roadmap

## How a new method plugs in

| method type | entry point | example |
|---|---|---|
| online rule / heuristic | new class in `src/policies/` (`@register_policy`) | ATC, look-ahead, learned rule |
| rolling horizon | a policy that solves a sub-problem over the visible parts at every decision | CP-SAT over the next 5–10 parts |
| offline (known instance) | `arrivals.mode: explicit` + solver → sequence → a `fixed_sequence` policy that replays it | MILP, CP-SAT, DP |
| parameter tuning | outer loop over `run_experiments` minimizing `composite_objective` | tuning the `weighted_score` weights with GA / SA |
| reinforcement learning | Gym-style environment wrapping `Simulation.step()` and calling the policy at each decision | PPO / DQN |

## Suggested steps

1. Pin down with the real cell: speeds, processing times, end-of-zone rule
   (`block` / `pass`), part size.
2. Build a bank of small `explicit` instances (5–15 parts) and solve them exactly
   (CP-SAT / DP) to obtain bounds and references.
3. Measure the gap of the simple policies against those references.
4. Design better heuristics (e.g. `weighted_score` with tuned weights, look-ahead).
5. Rolling-horizon CP-SAT for the dynamic case.
6. Multi-objective study (Pareto front: makespan vs travel vs downtime).

## Planned simulator improvements

* Event-driven core (faster), validated against the fixed-step one.
* Laser pre-positioning towards parts still in the entry zone.
* Axis accelerations / trapezoidal profiles.
* Physical part size, several operations per part, several lasers.

Dependencies to add then (not installed yet): `ortools` (CP-SAT), `pyomo` + a MILP
solver (HiGHS/CBC), `scipy`.
