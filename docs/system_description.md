# System description

## Physical components

| Element | V1 model | Parameters (config) |
|---|---|---|
| Laser | Cartesian XY head on a gantry, Z fixed. One job at a time. | `laser.speed_x`, `laser.speed_y`, `laser.motion_model`, `laser.workspace`, `laser.home` |
| Conveyors | N conveyors parallel to X at heights `y_i`; all share the same longitudinal geometry. | `conveyors.count`, `conveyors.y_positions`, `conveyors.speed` |
| Zones | entry `[0, 10)`, laser zone `[10, 45]`, exit `(45, 55]` cm | `geometry.*` |
| Parts | Points on the belt with their own processing time. | `processing.*`, `simulation.min_part_spacing` |

Convention: **X** = conveyor travel (+X), **Y** = across conveyors, **Z** fixed.

## Part life cycle

```
arrival ──► WAITING (entry queue, if the conveyor is stopped or the entry is occupied)
        ──► MOVING  (x ∈ [0, 10): the laser cannot reach it)
        ──► READY   (x ∈ [10, 45]: candidate for the laser)
        ──► PROCESSING (the laser is on it; its conveyor is stopped)
        ──► COMPLETED  (the conveyor restarts and carries it to the exit)
        ──► EXITED     (x ≥ 55)
```

## Laser cycle

```
IDLE ──(the policy chooses a READY part, or with pre-positioning `when_idle` — the
        design mode since 0.7.6 — an entry-zone part when none is READY)──► MOVING to the meeting point
     ──► WAITING_FOR_PART (if it arrives before the part; a pre-positioning is revoked
        if another part becomes READY meanwhile)
     ──► PROCESSING (the part's conveyor is STOPPED for processing_time)
     ──► IDLE (conveyor STARTED; new decision at the same instant)
```

## Conveyor rules

* A conveyor moves when its set of *stop reasons* is empty.
* Reason `processing`: added by the stopping rule (`stop_policy.py`) when processing
  starts; by default only on the part's conveyor.
* Reason `blocked` (`block` mode): an unprocessed part has reached x = 45.
* When a conveyor stops, all its parts stop; the other conveyors keep moving.

## Software layers

```
config (YAML) ─► simulation engine ─► state/events/metrics ─► results/ (CSV, JSON, log)
                        ▲                         │
                policies (select)                 └─► trace.json ─► visualization (replay only)
                                                  └─► optimization/objective_functions
```

The engine imports nothing from the visualization. The visualization contains no
simulation logic. Policies only see a read-only `DecisionContext`.
