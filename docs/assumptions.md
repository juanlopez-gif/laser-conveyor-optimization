# Supuestos y decisiones de modelado (V1)

Cada supuesto es configurable salvo que se indique lo contrario. Si alguno no
corresponde a la célula real, se cambia en `config/default_config.yaml` o en el
módulo indicado.

## A. Motor de simulación

**A1. Motor propio de paso fijo (`dt = 0.01 s`) en lugar de SimPy.**
SimPy es excelente cuando el estado solo cambia en eventos discretos. Aquí el estado
cambia de forma *continua* (piezas y láser se mueven) y los eventos dependen de esas
posiciones: cuándo el láser alcanza una pieza que se mueve, cuándo una pieza llega al
final de la zona, cuándo la parada de una cinta cambia la trayectoria prevista. Con
SimPy habría que calcular analíticamente cada uno de esos cruces y cancelar y
reprogramar procesos a mano, lo que añade complejidad sin aportar nada en la V1.
Un integrador de paso fijo, con un orden de operaciones explícito dentro de cada
paso (ver `src/simulation/engine.py`), es simple, determinista y fácil de verificar.
Coste: los instantes de los eventos tienen un error de hasta `dt` y cada ejecución de
300 s tarda unos 2 s. Mejora futura: un núcleo dirigido por eventos con cálculo
analítico de cruces, que puede validarse contra este.

**A2.** Determinismo: semilla maestra → flujos `random.Random("seed/arrivals/i")` y
`"seed/processing/i"` por cinta. Misma semilla ⇒ mismo experimento (testeado), y las
políticas comparten instancias (números aleatorios comunes).

## B. Cintas y piezas

**B1.** Las piezas son puntos con separación mínima `min_part_spacing = 4 cm` en la
cinta (aprox. el tamaño de la pieza). No se modela el tamaño de la pieza ni colisiones.

**B2.** Si una pieza llega cuando su cinta está parada o la entrada está ocupada, espera
en una cola aguas arriba (estado **WAITING**) y entra en cuanto puede.
*Interpretación del estado WAITING del enunciado.*

**B3.** Estados: WAITING (cola de entrada) → MOVING (zona de entrada) → READY (zona láser,
sin procesar) → PROCESSING → COMPLETED (procesada, camino de la salida) → EXITED.

**B4. Pieza sin procesar al final de la zona láser.** El enunciado no dice qué ocurre.
Por defecto (`unprocessed_at_zone_end: block`) **la cinta se detiene** con la pieza en
x = 45 cm hasta que el láser la procese (ninguna pieza se pierde, aparece *blocking*).
Alternativa `pass`: la pieza sigue y sale sin procesar (se cuenta en
`parts_exited_unprocessed`). **Esta es la decisión más importante a validar con la
célula real.**

**B5.** Regla de parada durante el procesado: solo la cinta de la pieza
(`processing_conveyor_only`); alternativa `all_conveyors`. Ampliable en
`src/simulation/stop_policy.py`.

**B6.** Las cintas arrancan y paran instantáneamente (sin rampas).

## C. Láser

**C1.** Velocidad constante por eje, sin aceleración ni tiempo de asentamiento.

**C2.** Movimiento por defecto **simultáneo**: `t = max(dx/vx, dy/vy)`; alternativa
`sequential`: `t = dx/vx + dy/vy`. Para la distancia recorrida se suma la longitud real
del camino (diagonal mientras se mueven ambos ejes y después recto).

**C3. Intercepción.** La pieza sigue moviéndose mientras el láser viaja (su cinta solo
para al empezar el procesado). El láser va al *punto de encuentro* más temprano y, si la
cinta cambia de estado durante el viaje, recalcula. Con las velocidades por defecto
(láser 5 cm/s = cinta 5 cm/s) el láser **no puede alcanzar en X una pieza que va por
delante**; en modo `block` la alcanza cuando la pieza se detiene al final de la zona.

**C4.** El láser solo puede elegir piezas READY (dentro de la zona accesible). No se
pre-posiciona para piezas que todavía están en la zona de entrada (mejora futura).

**C5.** Workspace del láser = zona activa en X (10–45 cm) × [y mín, y máx] de las cintas.
El láser nunca sale de él (testeado).

**C6.** Cuando está ocioso, el láser se queda donde terminó (no vuelve a una posición de
reposo). Posición inicial: centro del workspace.

## D. Llegadas y procesado

**D1.** Modo determinista: primera llegada en `first_arrival` (0, 1, 2 s) y después cada
`intervals` (8, 10, 12 s). Modo estocástico: Poisson (tiempos entre llegadas
exponenciales con medias 8, 10, 12 s).

**D2.** Se generan llegadas hasta `arrival_horizon = 300 s`; después (`drain: true`) se
sigue simulando hasta vaciar el sistema. El makespan es el instante en que termina la
última pieza.

**D3.** Tiempo de procesado constante 3 s (o uniforme / normal truncada por pieza).

**D4.** Fecha de entrega = llegada + 20 s, solo para calcular lateness.

## E. Métricas

**E1.** Cola de una cinta = piezas que han llegado y cuyo procesado no ha empezado
(WAITING + MOVING + READY).

**E2.** Espera = inicio de procesado − llegada. Flow time = fin de procesado − llegada.

**E3.** Utilización del láser = (moviéndose + procesando) / tiempo total. El tiempo que
espera quieto en el punto de encuentro cuenta como ocioso.

## F. Observación sobre la carga del caso por defecto

Con llegadas cada 8, 10 y 12 s y 3 s de procesado, la carga de **solo procesado** es
3·(1/8 + 1/10 + 1/12) ≈ **0.93**. Al añadir los viajes del láser (≈ 1–5 s por pieza) el
sistema queda **sobrecargado**: las colas crecen durante los 300 s de llegadas y las
cintas pasan mucho tiempo bloqueadas. No es un error del simulador, sino una
consecuencia de los parámetros provisionales. Para estudiar régimen estacionario
conviene, por ejemplo, `processing.time: 2.0` o intervalos mayores.
