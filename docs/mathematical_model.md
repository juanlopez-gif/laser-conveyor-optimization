# Modelo matemático del simulador (V1)

Notación: cintas `k = 1..N` en `y_k`; zona láser `[a, b]` (a = 10, b = 45 cm);
longitud total `L` (55 cm); velocidad de cinta `v_k`; velocidades del láser `v_x`, `v_y`.

## 1. Cinta

Estado `s_k(t) ∈ {RUN, STOP}`, con `STOP ⇔ R_k(t) ≠ ∅` (conjunto de motivos de parada).
Para toda pieza `j` sobre la cinta `k`:

```
dx_j/dt = v_k · 1[s_k(t) = RUN]
```

Modo `block`: si `x_j = b` y `j` no ha empezado a procesarse, `blocked ∈ R_k`.

## 2. Tiempo de viaje del láser

De `(x_1, y_1)` a `(x_2, y_2)`, con `dx = |x_2 − x_1|`, `dy = |y_2 − y_1|`:

| modelo | fórmula | interpretación |
|---|---|---|
| `simultaneous` (por defecto) | `τ = max(dx/v_x, dy/v_y)` | dos motores independientes moviéndose a la vez; manda el eje más lento |
| `sequential` | `τ = dx/v_x + dy/v_y` | primero X, después Y |

**Decisión documentada:** se usa `simultaneous` porque es el comportamiento normal de un
pórtico cartesiano con un accionamiento por eje. `sequential` queda disponible para
cuantificar cuánto se pierde si los ejes no se pueden mover a la vez.

## 3. Intercepción de una pieza en movimiento

Si el láser está en `(x_L, y_L)` y la pieza en `x_j(0)` sobre una cinta a velocidad `v`,
su posición prevista es `x_j(t) = min(x_j(0) + v t, b)` en modo `block`. El punto de
encuentro es el **menor** `t ≥ 0` tal que

```
τ( (x_L, y_L) → (x_j(t), y_k) ) ≤ t
```

Se resuelve con un barrido de 64 puntos seguido de bisección (`motion.find_intercept`).
En modo `pass`, si no existe solución con `x_j(t) ≤ b`, la pieza no es alcanzable.
Si `v ≥ v_x` y la pieza va por delante, solo existe solución porque la pieza se detiene en `b`.

## 4. Integración temporal

Paso fijo `Δt`: el estado en `t + Δt` se obtiene aplicando, en este orden, (1) fin de
procesado, (2) llegadas, (3) bloqueos, (4) admisión a la cinta, (5) decisión de la
política, (6) movimiento del láser / inicio de procesado, (7) métricas, (8) avance de las
cintas, (9) cambios de estado por posición. Error en los instantes de evento ≤ `Δt`.

## 5. Métricas

Para la pieza `j`: llegada `r_j`, inicio `S_j`, fin `C_j = S_j + p_j`, salida `E_j`,
fecha de entrega `d_j = r_j + Δ_d`.

| métrica | definición |
|---|---|
| espera | `W_j = S_j − r_j` |
| flow time | `F_j = C_j − r_j` |
| lateness | `L_j = C_j − d_j` |
| makespan | `C_max = max_j C_j` |
| throughput | `#completadas / T · 60` (piezas/min) |
| utilización del láser | `(T_mov + T_proc) / T` |
| parada de la cinta k | `∫ 1[s_k = STOP] dt` (desglosado en procesado / bloqueo) |
| cola de la cinta k | `Q_k(t) = #{j en k : r_j ≤ t < S_j}`; media `∫Q_k dt / T` |

## 6. Objetivo compuesto (a minimizar)

```
J = α·C_max + β·D_laser + γ·Σ_k T_stop,k + δ·W̄    (+ otros términos registrados)
```

Pesos en `objective.weights`; implementación en `src/optimization/objective_functions.py`.
