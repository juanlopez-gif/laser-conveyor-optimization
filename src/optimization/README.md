# optimization/

Layer prepared for phase 2. In V1 it contains:

* `objective_functions.py` — single objectives (makespan, flow time, waiting, laser
  travel, conveyor downtime, −throughput, maximum lateness), the configurable composite
  objective (`objective.weights` in the config) and a Pareto filter.
* `problem_formulation.md` — how to formulate the problem (variables, constraints,
  classification, candidate methods).
* `schedule_check.py` — verifies that a simulated schedule satisfies the formal model of
  `docs/formal_formulation.md` (entry coordinates, positions, window, interception,
  belt stops, belt kinematics).
* `decision_variables.md` — table of decision variables and their counterpart in the
  simulator.

All objectives are **minimized** and take the `summary` dictionary of a run, so any
optimizer only needs `f(summary) -> float`.

No solver has been implemented yet (MILP, CP-SAT, GA, RL...). See
`docs/future_optimization.md`.
