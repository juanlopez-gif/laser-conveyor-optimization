# Mathematical model of the simulator (V1)

Notation: conveyors `k = 1..N` at `y_k`; laser zone `[a, b]` (a = 10, b = 45 cm);
total length `L` (55 cm); conveyor speed `v_k`; laser speeds `v_x`, `v_y`.

## 1. Conveyor

State `s_k(t) ∈ {RUN, STOP}`, with `STOP ⇔ R_k(t) ≠ ∅` (set of stop reasons).
For every part `j` on conveyor `k`:

```
dx_j/dt = v_k · 1[s_k(t) = RUN]
```

`block` mode: if `x_j = b` and `j` has not started processing, `blocked ∈ R_k`.

## 2. Laser travel time

From `(x_1, y_1)` to `(x_2, y_2)`, with `dx = |x_2 − x_1|`, `dy = |y_2 − y_1|`:

| model | formula | meaning |
|---|---|---|
| `simultaneous` (default) | `τ = max(dx/v_x, dy/v_y)` | two independent motors moving at the same time; the slower axis dominates |
| `sequential` | `τ = dx/v_x + dy/v_y` | X first, then Y |

**Documented decision:** `simultaneous` is used because it is the normal behaviour of a
Cartesian gantry with one drive per axis. `sequential` is available to quantify how much
is lost if the axes cannot move at the same time.

## 3. Intercepting a moving part

With the laser at `(x_L, y_L)` and the part at `x_j(0)` on a conveyor moving at `v`, its
predicted position is `x_j(t) = min(x_j(0) + v t, b)` in `block` mode. The meeting point
is the **smallest** `t ≥ 0` such that

```
τ( (x_L, y_L) → (x_j(t), y_k) ) ≤ t
```

It is solved with a 64-point scan followed by bisection (`motion.find_intercept`).
In `pass` mode, if no solution exists with `x_j(t) ≤ b`, the part is unreachable.
If `v ≥ v_x` and the part is ahead, a solution only exists because the part stops at `b`.

## 4. Time integration

Fixed step `Δt`: the state at `t + Δt` is obtained by applying, in this order, (1) end
of processing, (2) arrivals, (3) blocking, (4) admission to the belt, (5) policy
decision, (6) laser motion / start of processing, (7) metrics, (8) belt motion,
(9) position-driven state changes. Event-time error ≤ `Δt`.

## 5. Metrics

For part `j`: arrival `r_j`, start `S_j`, completion `C_j = S_j + p_j`, exit `E_j`,
due date `d_j = r_j + Δ_d`.

| metric | definition |
|---|---|
| waiting | `W_j = S_j − r_j` |
| flow time | `F_j = C_j − r_j` |
| lateness | `L_j = C_j − d_j` |
| makespan | `C_max = max_j C_j` |
| throughput | `#completed / T · 60` (parts/min) |
| laser utilization | `(T_move + T_proc) / T` |
| downtime of conveyor k | `∫ 1[s_k = STOP] dt` (split into processing / blocking) |
| queue of conveyor k | `Q_k(t) = #{j on k : r_j ≤ t < S_j}`; mean `∫Q_k dt / T` |

## 6. Composite objective (minimized)

```
J = α·C_max + β·D_laser + γ·Σ_k T_stop,k + δ·W̄    (+ any other registered term)
```

Weights in `objective.weights`; implementation in `src/optimization/objective_functions.py`.
