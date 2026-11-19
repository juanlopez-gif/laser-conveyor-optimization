# optimization/

Layer prepared for phase 2. In V1 it contains:

* `objective_functions.py` — single objectives (makespan, flow time, waiting, laser
  travel, conveyor downtime, −throughput, maximum lateness), the configurable composite
  objective (`objective.weights` in the config) and a Pareto filter.
* `problem_formulation.md` — how to formulate the problem (variables, constraints,
  classification, candidate methods).
* `decision_variables.md` — table of decision variables and their counterpart in the
  simulator.

All objectives are **minimized** and take the `summary` dictionary of a run, so any
optimizer only needs `f(summary) -> float`.

No solver has been implemented yet (MILP, CP-SAT, GA, RL...). See
`docs/future_optimization.md`.
