# Descripción del sistema

## Componentes físicos

| Elemento | Modelo V1 | Parámetros (config) |
|---|---|---|
| Láser | Cabezal cartesiano XY sobre pórtico, Z fijo. Un único trabajo a la vez. | `laser.speed_x`, `laser.speed_y`, `laser.motion_model`, `laser.workspace`, `laser.home` |
| Cintas | N cintas paralelas a X, a alturas `y_i`; todas comparten la geometría longitudinal. | `conveyors.count`, `conveyors.y_positions`, `conveyors.speed` |
| Zonas | entrada `[0, 10)`, zona láser `[10, 45]`, salida `(45, 55]` cm | `geometry.*` |
| Piezas | Puntos sobre la cinta con tiempo de procesado propio. | `processing.*`, `simulation.min_part_spacing` |

Convención: **X** = avance de las cintas (+X), **Y** = transversal entre cintas, **Z** fijo.

## Ciclo de una pieza

```
llegada ──► WAITING (cola de entrada, si la cinta está parada o la entrada ocupada)
        ──► MOVING  (x ∈ [0, 10): el láser no la alcanza)
        ──► READY   (x ∈ [10, 45]: candidata para el láser)
        ──► PROCESSING (el láser está encima; su cinta parada)
        ──► COMPLETED  (la cinta rearranca y la lleva a la salida)
        ──► EXITED     (x ≥ 55)
```

## Ciclo del láser

```
IDLE ──(la política elige una pieza READY)──► MOVING hacia el punto de encuentro
     ──► WAITING_FOR_PART (si llega antes que la pieza)
     ──► PROCESSING (cinta de la pieza STOPPED durante processing_time)
     ──► IDLE (cinta STARTED; nueva decisión en el mismo instante)
```

## Reglas de las cintas

* Una cinta se mueve si su conjunto de *motivos de parada* está vacío.
* Motivo `processing`: lo añade la regla de parada (`stop_policy.py`) al empezar un
  procesado; por defecto solo en la cinta de la pieza.
* Motivo `blocked` (modo `block`): una pieza sin procesar ha llegado a x = 45.
* Al parar una cinta se paran todas sus piezas; las demás cintas siguen.

## Capas del software

```
config (YAML) ─► simulation engine ─► estado/eventos/métricas ─► results/ (CSV, JSON, log)
                        ▲                         │
                policies (select)                 └─► trace.json ─► visualization (solo reproduce)
                                                  └─► optimization/objective_functions
```

El motor no importa nada de la visualización. La visualización no contiene lógica de
simulación. Las políticas solo ven un `DecisionContext` de lectura.
