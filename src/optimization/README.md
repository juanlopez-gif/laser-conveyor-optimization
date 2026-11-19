# optimization/

Capa preparada para la fase 2. En la V1 contiene:

* `objective_functions.py` — objetivos simples (makespan, flow time, espera, recorrido
  del láser, parada de cintas, −throughput, lateness máxima), el objetivo compuesto
  configurable (`objective.weights` en la config) y un filtro de Pareto.
* `problem_formulation.md` — cómo formular el problema (variables, restricciones,
  clasificación, métodos candidatos).
* `decision_variables.md` — tabla de variables de decisión y su equivalente en el simulador.

Todos los objetivos se **minimizan** y reciben el diccionario `summary` de una ejecución,
de modo que cualquier optimizador solo necesita `f(summary) -> float`.

Aún no se ha implementado ningún solver (MILP, CP-SAT, GA, RL...). Ver
`docs/future_optimization.md`.
