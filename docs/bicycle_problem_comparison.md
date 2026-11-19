# Relación con el "bicycle problem"

## El problema de la bicicleta

Tres personas deben llegar a un destino. Cada una camina a velocidad `w_i` y pedalea a
`c_i > w_i`. Hay **una sola bicicleta** que solo puede usar una persona a la vez. Personas
y bicicleta pueden ir hacia delante y hacia atrás (alguien puede dejar la bicicleta en el
camino para que otro la recoja, o volver con ella). Objetivo: minimizar el instante en que
llega la última persona.

## Por qué existe una analogía

Los dos son casos de **scheduling de un recurso compartido y móvil**:

| | Bicycle problem | Láser + cintas |
|---|---|---|
| agentes / trabajos | 3 personas | piezas en N cintas |
| recurso único | bicicleta | cabezal láser |
| exclusividad | una persona en bici a la vez | una pieza procesada a la vez |
| el recurso debe desplazarse | la bici tiene que llegar a quien la va a usar | el láser viaja hasta la pieza (tiempo de viaje dependiente de la secuencia) |
| decisión | quién usa la bici, cuándo y dónde se deja | qué pieza recibe el láser y cuándo |
| objetivo | minimizar el tiempo de llegada del último | minimizar makespan / espera / viaje / paradas |
| progreso sin recurso | caminar (más lento) | la pieza avanza con su cinta, pero no se procesa |

En ambos, asignar el recurso a un agente beneficia a ese agente y obliga al recurso a
"recolocarse" antes de servir al siguiente; el óptimo equilibra el beneficio inmediato y
el coste de reposicionar el recurso. Las intuiciones (no dejar el recurso ocioso, servir
primero a quien más se retrasa, el coste de reposicionamiento) se trasladan.

## Por qué NO son matemáticamente idénticos

1. **Llegadas continuas.** El bicycle problem tiene un conjunto fijo de 3 agentes
   conocidos desde el principio. Aquí llegan trabajos sin parar (posiblemente infinitos):
   es un problema *online*/dinámico, no una instancia cerrada.
2. **Colas.** Las piezas se acumulan; aparecen longitudes de cola, esperas y estabilidad
   (carga ρ < 1), que no existen en el problema de la bicicleta.
3. **Tiempos de procesado.** El láser no solo "transporta" ventaja: ocupa un tiempo `p_j`
   en cada pieza durante el cual no puede moverse.
4. **Dinámica de las cintas.** Las posiciones de los trabajos cambian solas con el tiempo
   (y dependen de las paradas que provoca el propio láser). En la bicicleta los agentes
   deciden su movimiento; aquí no.
5. **Bloqueo / parada.** Procesar una pieza detiene su cinta y una pieza no procesada al
   final de la zona puede bloquearla. Las decisiones alteran la geometría futura de todos
   los trabajos de esa cinta.
6. **Ventanas de accesibilidad.** Cada pieza solo es alcanzable dentro de la zona láser
   durante un intervalo de tiempo: hay *time windows* implícitas.
7. **Objetivos múltiples.** Makespan, espera, recorrido del láser, paradas de cinta,
   lateness y throughput entran en conflicto; la bicicleta tiene un objetivo único.
8. **Escala.** La bicicleta con 3 personas tiene solución cerrada o casi; aquí el número
   de trabajos crece con el horizonte y el problema es NP-difícil en general.

## Conclusión

El bicycle problem es un **modelo conceptual simplificado** de *shared moving resource
scheduling*: útil para la intuición y para explicar el problema, pero su formulación no
es reutilizable directamente. La formulación adecuada para la célula es la de un
problema de secuenciación de una máquina con tiempos de preparación dependientes de la
secuencia, llegadas dinámicas, ventanas de tiempo y bloqueo (ver
`src/optimization/problem_formulation.md`).
