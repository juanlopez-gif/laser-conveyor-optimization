# Decision variables

| variable | domain | meaning | in the V1 simulator |
|---|---|---|---|
| `π` (order) | permutation of the parts | order in which the laser processes the parts | decided by the policy at each `select()` |
| `z_ij` | {0,1} | `j` immediately follows `i` | implicit in `laser.history` |
| `a_j(t)` (time assignment) | a part or "none" | which part the laser is assigned to at each instant | `laser.current_part` |
| `S_j` | ℝ⁺ | processing start time | `part.start_processing_time` (result of the interception) |
| `X_j` | [a, b] | x position where the part is processed | meeting point (`motion.find_intercept`) |
| `w_j` (deliberate delay) | ℝ⁺ | intentionally delay the start (e.g. let the part come closer) | not used: the laser always heads to the chosen part as soon as it is free |
| `u_k(t)` | {0,1} | run/stop of conveyor `k` | fixed rule `stop_policy` + blocking |
| `ξ` (stopping policy) | {own conveyor, all, ...} | global stopping rule | `simulation.conveyor_stop_policy` |
| `θ` (policy parameters) | ℝᵈ | e.g. weights `w_wait`, `w_travel`, `w_urgency` | `policy.params` |

Two possible optimization levels:

1. **Optimize the decisions directly** (`π`, `S_j`, `X_j`) for a known instance
   → MILP / CP-SAT / DP / metaheuristics over permutations.
2. **Optimize the rule** (`θ`, `ξ`) that makes online decisions → search over parameters
   evaluated with `run_experiments` (many seeds), or RL.
