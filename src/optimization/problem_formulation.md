# Formulación del problema (borrador para la fase de optimización)

## Clasificación

El problema combina rasgos de:

* **Scheduling** de trabajos con un objetivo temporal.
* **Single shared resource scheduling**: una sola "máquina" (el láser) para todos los trabajos.
* **Sequence-dependent setup / travel times**: el tiempo antes de procesar `j` depende
  de dónde terminó el trabajo anterior `i` (viaje del láser), y además de *cuándo*,
  porque la pieza se mueve.
* **Dynamic job arrivals**: los trabajos se liberan en `r_j` y no se conocen de antemano
  (versión online) o sí (versión offline / instancia fija).
* **Blocking**: una pieza no procesada al final de la zona detiene su cinta y a todas las
  piezas que la siguen.
* **Material handling**: las cintas transportan los trabajos; la posición es una función
  del tiempo y de las paradas.
* **Hybrid / discrete-event systems**: dinámica continua (posiciones) + eventos discretos
  (inicio/fin de procesado, paradas).

En notación de Graham aproximada: `1 | r_j, s_ij(t), blocking, time windows | C_max` (y
variantes con `ΣF_j`, `L_max`, objetivos múltiples). Incluso `1 | r_j | L_max` y
`1 | s_ij | C_max` (equivalente a un TSP) son NP-difíciles.

## Datos (instancia offline)

* `J`: piezas; para cada `j`: cinta `k(j)`, llegada `r_j`, procesado `p_j`, fecha `d_j`.
* Geometría: `a`, `b`, `y_k`, `v_k`; láser `v_x`, `v_y`, posición inicial `(x_0, y_0)`.

## Variables de decisión

Ver `decision_variables.md`. Núcleo:

* `z_ij ∈ {0,1}`: `j` se procesa inmediatamente después de `i` (secuencia).
* `S_j ≥ 0`: inicio del procesado de `j`; `C_j = S_j + p_j`.
* `X_j ∈ [a, b]`: posición x en la que se procesa `j` (depende de cuánto se ha movido su cinta).
* (opcional) `u_k(t) ∈ {0,1}`: marcha/parada de la cinta `k` si la política de parada se optimiza.

## Restricciones

1. **Un solo trabajo a la vez** (recurso disjuntivo): para todo par `i ≠ j`,
   `S_j ≥ C_i + τ_ij − M(1 − z_ij)` y `Σ_j z_ij ≤ 1`, `Σ_i z_ij ≤ 1` (más secuencia ficticia
   inicial/final). En CP-SAT: `NoOverlap` sobre intervalos `[S_j − τ_j, C_j)`.
2. **Tiempo de viaje incluido:** `τ_ij = max(|X_j − X_i| / v_x, |y_k(j) − y_k(i)| / v_y)`
   (o la suma, modelo secuencial). No lineal → linealizar con `τ_ij ≥ |X_j−X_i|/v_x`,
   `τ_ij ≥ |Δy|/v_y` y los valores absolutos con dos desigualdades.
3. **No antes de llegar a la zona:** `S_j ≥ r_j + a / v_k(j)` (si la cinta no se detuvo
   antes) y, en general, `X_j ≥ a`.
4. **Alcance físico:** `a ≤ X_j ≤ b`, `y_min ≤ y_k(j) ≤ y_max`.
5. **Posición de la pieza = integral de la marcha de su cinta:**
   `X_j = v_k · (S_j − e_j − tiempo parado de k entre e_j y S_j)`, con `e_j` la entrada a
   la cinta. Es la restricción que acopla todos los trabajos de una misma cinta y la más
   difícil de modelar.
6. **La cinta se detiene durante el procesado:** `u_k(t) = 0` para `t ∈ [S_j, C_j)` si
   `k = k(j)` (regla por defecto).
7. **Bloqueo (modo block):** si `X_j` alcanzaría `b` antes de `S_j`, la cinta `k(j)` queda
   parada desde ese instante hasta `S_j`.
8. **Orden FIFO físico dentro de una cinta**: las piezas no se adelantan en la cinta
   (pero el láser sí puede procesarlas en otro orden).

## Objetivos

`C_max`, `Σ F_j / |J|`, `Σ W_j / |J|`, distancia total del láser `Σ_ij z_ij · dist_ij`,
`Σ_k T_stop,k`, throughput, `L_max`; o el compuesto
`α C_max + β D + γ T_stop + δ W̄`.

## Simplificaciones útiles para empezar

* **Instancia pequeña y fija** (5–15 piezas): enumerar secuencias / DP sobre subconjuntos
  (`O(n² 2ⁿ)`) usando el **simulador como evaluador exacto** de una secuencia dada. Evita
  modelar la restricción 5 de forma explícita.
* **Discretización temporal** (time-indexed MILP / CP con dominios enteros en décimas de
  segundo): hace lineales las posiciones a costa de tamaño.
* **Rolling horizon**: optimizar solo las piezas visibles en la zona y re-optimizar en cada
  evento.

## Métodos candidatos

MILP (Pyomo + HiGHS/CBC), CP-SAT (OR-Tools; intervalos, `NoOverlap`, `AddMaxEquality`),
programación dinámica para instancias pequeñas, heurísticas greedy (ya hay 4),
algoritmos genéticos, simulated annealing y tabu search sobre la secuencia (con el
simulador como función de evaluación), aprendizaje por refuerzo (el simulador como
entorno) y optimización rolling-horizon.
