# Hoja de ruta para la optimización

## Cómo se engancha un método nuevo

| tipo de método | punto de entrada | ejemplo |
|---|---|---|
| regla online / heurística | nueva clase en `src/policies/` (`@register_policy`) | ATC, look-ahead, regla aprendida |
| rolling horizon | política que, en cada decisión, resuelve un subproblema con las piezas visibles | CP-SAT sobre las próximas 5–10 piezas |
| offline (instancia conocida) | `arrivals.mode: explicit` + solver → secuencia → política `fixed_sequence` que la reproduce | MILP, CP-SAT, DP |
| ajuste de parámetros | bucle externo sobre `run_experiments` minimizando `composite_objective` | búsqueda de pesos de `weighted_score` con GA / SA |
| aprendizaje por refuerzo | entorno tipo Gym que envuelve `Simulation.step()` y llama a la política en cada decisión | PPO / DQN |

## Pasos sugeridos

1. Fijar con la célula real: velocidades, tiempos de procesado, regla de fin de zona
   (`block` / `pass`), tamaño de pieza.
2. Generar un banco de instancias (`explicit`) pequeñas (5–15 piezas) y resolverlas
   exactamente (CP-SAT / DP) para tener cotas y referencias.
3. Medir el *gap* de las políticas simples frente a esas referencias.
4. Diseñar heurísticas mejores (p. ej. `weighted_score` con pesos optimizados, look-ahead).
5. Rolling horizon con CP-SAT para el caso dinámico.
6. Estudio multiobjetivo (frente de Pareto: makespan vs recorrido vs paradas).

## Mejoras del simulador previstas

* Núcleo dirigido por eventos (más rápido), validado contra el de paso fijo.
* Pre-posicionamiento del láser hacia piezas que aún están en la zona de entrada.
* Aceleraciones / perfiles trapezoidales de los ejes.
* Tamaño físico de piezas, varias operaciones por pieza, varios láseres.

Dependencias que habrá que añadir entonces (no instaladas aún): `ortools` (CP-SAT),
`pyomo` + un solver MILP (HiGHS/CBC), `scipy`.
