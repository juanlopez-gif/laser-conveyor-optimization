# Variables de decisión

| variable | dominio | significado | en el simulador V1 |
|---|---|---|---|
| `π` (orden) | permutación de las piezas | orden en que el láser procesa las piezas | lo decide la política en cada `select()` |
| `z_ij` | {0,1} | `j` sigue inmediatamente a `i` | implícito en `laser.history` |
| `a_j(t)` (asignación temporal) | pieza o "ninguna" | a qué pieza está asignado el láser en cada instante | `laser.current_part` |
| `S_j` | ℝ⁺ | instante de inicio del procesado | `part.start_processing_time` (resultado del encuentro) |
| `X_j` | [a, b] | posición x donde se procesa | punto de encuentro (`motion.find_intercept`) |
| `w_j` (espera deliberada) | ℝ⁺ | retrasar a propósito el inicio (p. ej. dejar que la pieza avance) | no usado: el láser va siempre a la pieza elegida en cuanto queda libre |
| `u_k(t)` | {0,1} | marcha/parada de la cinta `k` | regla fija `stop_policy` + bloqueo |
| `ξ` (política de parada) | {solo su cinta, todas, ...} | regla global de parada | `simulation.conveyor_stop_policy` |
| `θ` (parámetros de política) | ℝᵈ | p. ej. pesos `w_wait`, `w_travel`, `w_urgency` | `policy.params` |

Dos niveles de optimización posibles:

1. **Optimizar la decisión directamente** (`π`, `S_j`, `X_j`) para una instancia conocida
   → MILP / CP-SAT / DP / metaheurísticas sobre permutaciones.
2. **Optimizar la regla** (`θ`, `ξ`) que toma decisiones online → búsqueda sobre
   parámetros evaluada con `run_experiments` (muchas semillas), o RL.
