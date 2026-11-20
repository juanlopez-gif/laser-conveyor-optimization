# Modelling assumptions and decisions (V1)

Every assumption is configurable unless stated otherwise. If one does not match the real
cell, change it in `config/default_config.yaml` or in the module mentioned.

## A. Simulation engine

**A1. Custom fixed-step engine (`dt = 0.01 s`) instead of SimPy.**
SimPy is excellent when the state only changes at discrete events. Here the state
changes *continuously* (parts and laser move) and events depend on those positions: when
the laser reaches a moving part, when a part reaches the end of the zone, when a
conveyor stop changes the predicted trajectory. With SimPy each of those crossings would
have to be computed analytically and processes cancelled and rescheduled by hand, which
adds complexity without benefit in V1. A fixed-step integrator with an explicit order of
operations inside each step (see `src/simulation/engine.py`) is simple, deterministic and
easy to verify. Cost: event times have an error of up to `dt` and a 300 s run takes
about 1–2 s. Future improvement: an event-driven core with analytic crossing times,
validated against this one.

**A2.** Determinism: master seed → streams `random.Random("seed/arrivals/i")` and
`"seed/processing/i"` per conveyor. Same seed ⇒ same experiment (tested), and policies
share instances (common random numbers).

## B. Conveyors and parts

**B1.** Parts are points with a minimum spacing `min_part_spacing = 4 cm` on the belt
(roughly the part size). Part size and collisions are not modelled.

**B2.** If a part arrives while its conveyor is stopped or the entry is occupied, it
waits in an upstream queue (state **WAITING**) and enters as soon as possible.
*This is how the WAITING state of the specification was interpreted.*

**B3.** States: WAITING (entry queue) → MOVING (entry zone) → READY (laser zone, not
processed) → PROCESSING → COMPLETED (processed, travelling to the exit) → EXITED.

**B4. Unprocessed part at the end of the laser zone — CONFIRMED: the conveyor stops.**
With `unprocessed_at_zone_end: block` (default) the conveyor stops with the part at
x = 45 cm until the laser processes it (no part is lost; *blocking* appears). This was
confirmed for the real cell. The alternative `pass` (the part exits unprocessed and is
counted in `parts_exited_unprocessed`) is kept only for what-if studies.

**B5.** Stopping rule during processing: only the part's conveyor
(`processing_conveyor_only`); alternative `all_conveyors`. Extensible in
`src/simulation/stop_policy.py`.

**B6.** Conveyors start and stop instantaneously (no ramps).

## C. Laser

**C1.** Constant speed per axis, no acceleration or settling time.

**C2.** Default motion is **simultaneous**: `t = max(dx/vx, dy/vy)`; alternative
`sequential`: `t = dx/vx + dy/vy`. The travelled distance is the real path length
(diagonal while both axes move, then straight).

**C3. Interception.** The part keeps moving while the laser travels (its conveyor only
stops when processing starts). The laser heads to the earliest *meeting point* and
re-plans if the conveyor changes state during the trip. Because the laser (10 cm/s) is
faster than the conveyors (3 cm/s) it can catch parts ahead of it; with the heavy-load
example (both 5 cm/s) it can only catch a part ahead when the part stops at the zone end.

**C4. Pre-positioning (changed in 0.4.0).** By default (`laser.prepositioning: none`)
the laser can only select READY parts (inside the reachable zone). Since 0.4.0 it can
optionally be sent to a part still in the entry zone: it moves to the meeting point at
x ≥ a and waits (`always`), or does so only when no part is READY and gives the
pre-positioning up as soon as one becomes READY (`when_idle`). The default stays `none`
because pre-positioning needs to know that a part is on the entry zone (an entry
sensor), which has **not been confirmed** for the real cell. The assignment is
committed (as for READY parts) in `always` mode. Rationale and results:
docs/CHANGELOG.md (0.4.0), docs/methods/nearest_head_prepositioning.md.

**C5.** Laser workspace = laser zone in X (10–45 cm) × [min y, max y] of the conveyors.
The laser never leaves it (tested).

**C6.** When idle, the laser stays where it finished (no return to a home position).
Initial position: centre of the workspace.

## D. Arrivals and processing

**D1.** Deterministic mode: first arrival at `first_arrival` (0, 1, 2 s), then every
`intervals` (10, 12, 14 s). Stochastic mode: Poisson (exponential inter-arrival times
with means 10, 12, 14 s).

**D2.** Arrivals are generated up to `arrival_horizon = 300 s`; afterwards
(`drain: true`) the simulation continues until the system is empty. Makespan is the time
at which the last part is completed.

**D3.** Constant processing time 2.5 s (or uniform / truncated normal per part).

**D4.** Due date = arrival + 20 s, used only to compute lateness.

## E. Metrics

**E1.** Queue of a conveyor = parts that have arrived and whose processing has not
started (WAITING + MOVING + READY).

**E2.** Waiting = processing start − arrival. Flow time = processing end − arrival.

**E3.** Laser utilization = (moving + processing) / total time. Time spent standing at
the meeting point waiting for the part counts as idle.

**E4. Blocking penalty (0.4.0).** A *blocking event* is a part that reaches the end of
the laser zone unprocessed and holds its conveyor stopped for a positive time
(S_j > B_j, B_j = time it reached x = b). A part that reaches b exactly when the laser
starts on it is not counted (the belt is stopped by processing anyway). Blocked time
T_B = conveyor-seconds stopped by blocking = Σ_j (S_j − B_j); the block ends at S_j
(from then on the belt is held by processing, counted as processing stop). Penalty
Π = λ_N·N_B + λ_T·T_B with λ_N = 10 s per event and λ_T = 1 s per blocked
conveyor-second. **These weights are placeholders** chosen so that one blocked
conveyor-second costs as much as one second of flow time and every event carries a
fixed extra cost; they must be calibrated with the real cost of a stopped conveyor
(`objective.blocking_penalty`).

**E5. Default objective (0.4.0).** `penalized_flow_time` = mean flow time + Π / n
replaces `makespan` as the default composite objective. With stable load the makespan
is almost fixed by the last arrival and hardly separates the policies, and it does not
see blocking.

## F. Load of the default case

**Status (2026-11-20): these are the real values of the cell.** Asked whether the cell
still had to be measured, the owner answered that the data we have are the real ones.
This covers belt speed, laser speed, processing time and mean inter-arrival times. The
default configuration is therefore a model of the real cell, not a placeholder.
The heavy-load case stays a stress scenario, and the sensitivity sweep
(`docs/methods/sensitivity.md`) stays a what-if study: belt speed and job time can be
adjusted in the cell.

Defaults (updated after the first review): conveyors 3 cm/s, laser 10 cm/s, processing
2.5 s, arrivals every 10 / 12 / 14 s. The processing-only load is
2.5·(1/10 + 1/12 + 1/14) ≈ **0.64**; adding laser travel brings the laser utilization to
roughly 0.9, so the cell is busy but stable, and the dispatching policy matters. With
much lighter loads (e.g. arrivals every 12 / 15 / 18 s and 2 s processing) there is
rarely more than one candidate part and all policies behave identically.

The first provisional values (conveyors 5 cm/s, laser 5 cm/s, processing 3 s, arrivals
every 8 / 10 / 12 s) overload the cell (processing load ≈ 0.93 before travel; queues grow
without bound). They are kept in `config/examples/heavy_load.yaml`.
