# Geometry bench: does the head's crossing time predict the camera coverage?

Status: **pilot** (0.7.18, 2026-11-23). Sections 1–3 were written before the pilot
ran; section 4 reports it. The confirmatory bench (owner's brief of 2026-11-23: 36
regimes, seven coverages, two modes, two controllers) is not designed here; its
protocol will be fixed after the pilot and will need a new reserved seed range from
the owner (7000–7029 are used).

## 1. Questions

Paper 2 (`docs/papers/paper2/`) measured, on one cell (three belts 12 cm apart, head
10 cm/s on both axes, belt 3 cm/s), that the gain of pre-positioning grows with the
anticipation the camera gives and levels off near the head's travel time across the
conveyors, τ_cross = (y_max − y_min) / v_y = 2.4 s. The owner's brief asks:

1. Does the coverage curve move when the belt spacing or the transverse head speed
   change?
2. Do two cells with different spacing and speed but the same τ_cross show the same
   curve and the same sufficient coverage?
3. How many seconds of anticipation must the camera give to keep most of the
   pre-positioning benefit, and what does operating with less coverage cost?

τ_cross is treated as a design hypothesis: if the curves do not align, the discrepancy
is quantified and its explanation sought in the traces.

Definitions (owner's brief): τ_cross = (y_max − y_min) / v_y; τ_vis = ℓ_v / v_belt
(nominal anticipation; the effective anticipation, from the instant a part becomes
visible to the instant it enters the laser zone, is also recorded, since a belt can
stop before the part reaches the zone).

## 2. Hypotheses, written before the pilot (2026-11-23)

**H0 (scale invariance, a property of the model, to be checked exactly).** With
simultaneous axes the head's travel time to a point is max(|Δx| / v_x, |Δy| / v_y),
and the only places where the belts' y coordinates enter the simulator are these
travel times and the head's home (the centre of the workspace). Two cells with the
same v_x, the same x geometry and the same lane travel time spacing / v_y therefore
have identical dynamics for every policy that decides on travel times (the rule) or on
positions scaled consistently. Prediction: the three configurations with τ_cross =
2.4 s (spacing 6 cm at v_y = 5 cm/s, 12 at 10, 18 at 15) give **identical** costs
realization by realization, up to floating point. This is a check of the
implementation, not a scientific test; question 2 of the brief is answered by the
model's structure, and the empirical content of the bench lies in question 1, across
*different* τ_cross. (The learned policy sees absolute positions and speeds through
its tokens, so for it the three cells are different inputs; H0 does not apply to it.)

**H1 (normalization by τ_cross).** For the rule, the pre-positioning gain as a share
of its value at 150 % of τ_cross, plotted against τ_vis / τ_cross, is approximately
common across geometries *where the lane component dominates the pre-positioning
moves*: the first grid point that retains at least 90 % of the gain at 150 % lies at
the same fraction of τ_cross within one grid step (0.25) for those geometries.

**H2 (where normalization fails, and what replaces it).** A pre-positioning move
takes max(t_x, t_y) seconds, where t_x is the along-belt travel from the head's
current position (usually the last processing point, inside the 35 cm laser zone) to
the belt's entry point a, and t_y the lane change. The geometric maximum of t_x is
35 / 10 = 3.5 s; its typical value is what the traces will say. When τ_cross is small
(0.8–1.6 s in the grid) the x component will dominate most moves, and the curve in
seconds will level off near the typical t_x rather than near τ_cross; normalized by
τ_cross those curves will saturate *later* (beyond 100 %, possibly not within 150 %).
Prediction: the quantity that predicts the knee in seconds across all geometries is
the typical duration of the pre-positioning moves actually made at full coverage,
τ_pp = median over moves of max(t_x, t_y), not τ_cross; τ_cross predicts it only
when t_y dominates. The traces record, for every pre-positioning move, t_x and t_y at
the decision, so the share of y-dominated moves is measured, not assumed.

**H3 (physical entry length).** Lengthening the entry zone from 10 to 35 cm with the
same visible lengths [a − ℓ_v, a) adds the free transit of 25 cm at 3 cm/s (8.33 s)
to every part's flow time and changes little else: the gain of pre-positioning in
seconds per part at a given ℓ_v is predicted equal within noise between the two entry
lengths (the camera stretch, the zone and the head are the same; only the upstream
queue is admitted to the belt earlier). Reporting the flow net of the free transit
(J − entry_length / v_belt) keeps the upstream waits and stops, which the brief asks
to retain, and makes the two entry lengths comparable in percent.

**H4 (seconds, not percent).** The primary measure of the benefit is seconds saved
per part, G_s(ℓ) = J_none(ℓ) − J_when_idle(ℓ), paired by realization. Percentages are
reported against the flow net of the free transit.

Minimum useful benefit (to be frozen before the confirmatory bench): a geometry and
regime is worth sizing the coverage for when the gain at 150 % of τ_cross is at least
**1 s per part** with its 95 % interval above 0.5 s (in the reference cell the
whole-zone gain is 2.3 s per part, 7.4 % of 31.5 s). Below that the absolute effects
are reported and the sufficient coverage is left undetermined for that regime.

## 3. Pilot design

* Simulator, information model and treatments of phase 4 (`experiments/rl2_eval.py`,
  `docs/methods/confirmatory.md` §9). Real cell, Poisson arrivals at the nominal
  rates (inter-arrivals 10 / 12 / 14 s), processing 2.5 s, belts 3 cm/s, three belts,
  default blocking penalties.
* **Geometry grid:** belt spacing 6 / 12 / 18 cm × v_y 5 / 10 / 15 cm/s (nine cells);
  v_x fixed at 10 cm/s; τ_cross = 2 · spacing / v_y ∈ {0.8, 1.2, 1.6, 2.4 (×3), 3.6,
  4.8, 7.2} s. `conveyors.y_positions=null` so the belts sit at i · spacing; the head's
  home is the centre of the derived workspace.
* **Entry zone:** 35 cm in every cell of the grid (`geometry.entry_length=35`), long
  enough for 150 % of the largest τ_cross (1.5 × 7.2 s × 3 cm/s = 32.4 cm). The laser
  zone (35 cm) and the exit zone (10 cm) keep their lengths; the zone, the workspace
  and the home move with the entry length by construction (`simulation/config.py`).
* **Coverage grid:** ℓ_v = f · τ_cross · v_belt for f ∈ {0, 0.25, 0.5, 0.75, 1, 1.25,
  1.5}, plus the common absolute lengths {1, 2.5, 5, 7.5, 10} cm so that the curves
  in seconds share points across geometries. Each length with `none` and
  `when_idle`, same arrivals and processing times (common random numbers).
* **Physical-length check (H3):** the original cell (spacing 12, v_y 10, entry 10 cm)
  at ℓ_v ∈ {0, 1, 2.5, 5, 7.5, 10} cm, against the 35 cm entry at the same lengths.
  Arrivals depend only on the seed and the rates (`simulation/arrivals.py`), so the
  pairing at the feed point holds across entry lengths and geometries.
* **Controllers:** `nearest_job` (primary; geometry-agnostic, decides on interception
  times) and the frozen network `rl2_gen_s_b` (secondary; the 35 cm entry is outside
  its training range of 5–15 cm, so its results also measure that extrapolation).
* **Seeds:** 3000–3029 (development seeds of the earlier pilots, `confirmatory.md`
  §§7–8); never 7000–7029.
* **Trace statistics per episode** (added to `fastsim.summary()` in 0.7.18):
  number of pre-positioning moves; their t_x and t_y at the decision (means and the
  share with t_y > t_x); revocations; head idle time and time waiting at the meeting
  point; mean effective anticipation (t_ready − t_visible) over the visible parts.
* **Outputs:** per-episode rows in `results/experiments/geometry_pilot/bench.json`
  (not versioned) and `tables.md`; figures in `docs/methods/figures/`.
* **Checks before the run:** (i) the original cell reproduces the phase-4
  configuration (same cost from the pilot script and from `rl2_eval.run` on the same
  seed); (ii) at ℓ_v = 0, `when_idle` equals `none` realization by realization; (iii)
  the visible stretch is respected (no part becomes a candidate before entering
  [a − ℓ_v, a)); (iv) H0 holds exactly for the 2.4 s triplet.

Analysis: paired G_s(ℓ) with 95 % CI; share of G_s at f = 1.5; knee = first grid
fraction with G_s(f) − 0.9 · G_s(1.5) ≥ 0 (point estimate) and the same with the
lower CI bound; curves in seconds and in τ_vis / τ_cross; the triplet; the
physical-length check; the trace statistics against the knee in seconds (H2).

Reproduction:

    python experiments/geometry_pilot.py --tag geometry_pilot --seed-base 3000 --seeds 30
    python experiments/geometry_pilot.py --tag geometry_pilot --analyse-only

## 4. Results of the pilot (2026-11-23; 14 400 episodes, 45 s on 40 workers)

Full tables: `results/experiments/geometry_pilot/tables.md` (not versioned; regenerated
by the command above). Figures: `docs/methods/figures/geometry_pilot_nearest_job.pdf`
and `geometry_pilot_net.pdf` (gain in seconds, gain normalized by τ_cross, and
pre-positioning moves per episode, against the anticipation).

**Checks.** (i) The original cell run by the pilot script equals `rl2_eval.run` on
seeds 3000–3002 at every length and mode, for both controllers (max |Δ| = 0). (ii) At
ℓ_v = 0, `when_idle` equals `none` exactly for the rule; for the network the two
differ (up to 3.8 s in one episode, −0.28…+0.17 s/part on average) because the
pre-positioning mode is a one-hot in its context token (`rl2/features.py`), so the
network conditions on the mode even when the candidate sets are identical. (iii) No
candidate ever lay upstream of the visible stretch (0 violations in 14 400 episodes).
(iv) **H0 holds exactly**: the three cells with τ_cross = 2.4 s give identical costs
realization by realization for the rule (max |ΔJ| = 1.1 · 10⁻¹³ s over 720 paired
episodes). For the network they differ by up to 2.2 s per episode, as expected from
its absolute-position tokens, with the same mean curve within noise (gain at 150 %:
1.58 ±0.28, 1.32 ±0.25, 1.34 ±0.21 s/part). Question 2 of the brief is therefore
answered by the model's structure, not by the data: with simultaneous axes the
geometry enters only through spacing / v_y.

**Caveat on the seeds.** Seeds 3000–3029 are a lighter draw than the confirmatory
seeds: 70.5 parts per episode against 77.5, and `nearest_job` in the original cell at
17.2 s/part against 21.8 (phase 4). Gains in seconds are correspondingly smaller
(1.3 s/part at 10 cm in the original cell against ≈1.6 in phase 4).

### 4.1 The rule: the knee in seconds barely moves with the geometry

| spacing | v_y | τ_cross | G at 150 % τ_cross [s/part] | of which flow / penalty | share at 5 / 7.5 / 10 cm | knee [s] point / conservative | knee / τ_cross | head idle (`none`) | moves per episode | mean move max(t_x, t_y) [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 15 | 0.8 | +0.92 ±0.35 (at 10 cm) | 0.83 / 0.10 | 93 / 99 / 100 % | 1.67 / 2.50 | 2.1 / 3.1 | 31 % | 27.7 | 0.67 |
| 6 | 10 | 1.2 | +1.07 ±0.28 (at 10 cm) | 0.96 / 0.11 | 97 / 100 / 100 % | 1.20 / 1.50 | 1.0 / 1.25 | 28 % | 25.9 | 0.83 |
| 12 | 15 | 1.6 | +1.43 ±0.42 (at 10 cm) | 1.14 / 0.29 | 100 / 100 / 100 % | 1.60 / 1.60 | 1.0 / 1.0 | 25 % | 23.2 | 0.98 |
| 12 | 10 (and 6 / 5, 18 / 15) | 2.4 | +1.64 ±0.46 | 1.36 / 0.28 | 86 / 91 / 100 % | 2.40 / 3.00 | 1.0 / 1.25 | 21 % | 18.7 | 1.32 |
| 18 | 10 | 3.6 | +1.48 ±0.44 | 1.26 / 0.21 | 81 / 95 / 102 % | 2.50 / 3.33 | 0.69 / 0.93 | 15 % | 12.7 | 1.84 |
| 12 | 5 | 4.8 | +1.46 ±0.58 | 1.26 / 0.19 | 101 / 105 / 109 % | 1.67 / 4.80 | 0.35 / 1.0 | 12 % | 8.2 | 2.44 |
| 18 | 5 | 7.2 | +1.52 ±0.87 | 1.36 / 0.16 | 110 / 95 / 114 % | 1.67 / 3.33 | 0.23 / 0.46 | 8 % | 4.0 | 2.97 |

(G at the largest length of the grid: 150 % τ_cross or 10 cm, whichever is longer.
Share = gain at that length / gain at the largest length. Knee = first point of the
merged grid retaining 90 % of the gain at the largest length, point estimate and with
the lower 95 % bound of the paired contrast G(ℓ) − 0.9 G(ℓ_max) ≥ 0. Moves and move
durations at the largest length, `when_idle`.)

* **Question 1 (does the curve move?).** In seconds, little: the knee lies between
  1.2 and 3.3 s in all nine geometries while τ_cross spans 0.8–7.2 s. The whole-zone
  gain is 0.9–1.6 s/part everywhere (5–11 % of the flow net of the free transit), 83–93 %
  of it flow time and the rest blocking penalty. **At 7.5 cm (2.5 s) the rule retains
  91–105 % of the gain in every geometry; at 5 cm (1.67 s) 81–110 %.** Beyond ≈3 s the
  gain does not change at all in the slow-head cells (identical costs at 14.4, 18 and
  21.6 cm for τ_cross 4.8): no decision ever uses a part that far upstream.
* **H1 (normalization by τ_cross) holds only in the middle of the range.** For τ_cross
  1.2–2.4 s the knee is 1.0–1.25 τ_cross (within one grid step, as H1 requires). For
  τ_cross 0.8 s the knee is 2.1–3.1 τ_cross; for 3.6 s it is 0.7–0.9; for 4.8 and
  7.2 s it is 0.2–0.5 (point) and the gains are too noisy (±0.6–0.9 s) to place it
  better than "between 1.7 and 4.8 s". The normalized curves (figure, middle panel)
  fan out: fast heads saturate late in units of τ_cross, slow heads early.
* **H2 (what replaces it).** The traces reject the mechanism H2 assumed and show a
  different one. The *number* of pre-positioning moves hardly depends on the
  anticipation (for τ_cross 0.8: 24.6 moves per episode at 0.2 s of anticipation,
  27.7 at 3.3 s; for 7.2: 2.7 → 4.0): almost every idle period of the head finds a
  visible part even with a tiny window, so the opportunities are fixed by the head's
  idle time (31 % → 8 % of the time from the fastest to the slowest head), not by the
  camera. What grows with the anticipation is the *gain per move*, and it saturates
  at 1.2–3 s in every geometry: 0.7 → 2.3 s per move for τ_cross 0.8, 2.8 → 6.2 for
  2.4, 8 → 27 for 7.2. These gains per move exceed the move's own duration by a
  factor 2–9 because a second saved on the head's cycle is saved by every part waiting
  behind it in the busy period, which is longer when the head is busier. The move
  durations actually made are 0.7–3.0 s (mean of max(t_x, t_y)), always below τ_cross
  because most moves change one lane, not two, and the x component is 0.5–1.3 s. The
  share of y-dominated moves is 45–58 %, so the along-belt component is not
  negligible in any geometry. Neither τ_cross (knee / τ_cross 0.23–2.1) nor the mean
  move duration (knee / τ_pp 0.6–2.5) predicts the knee across the range; what the
  data support is a knee of 1.5–3 s of anticipation at this load, roughly the move
  duration plus the time for the head to reach the entry point and settle, which is
  of the same order for every geometry tested.
* **Comparison with Paper 2's slow head.** In phase 4 the slow head (*both* axes at
  half speed, τ_cross 4.8 s, v_x 5 cm/s) needed the whole 10 cm zone (36 % of the gain
  at 5 cm, 85 % at 7.5 cm). Here τ_cross 4.8 s with v_x kept at 10 cm/s retains
  101–105 % at 5–7.5 cm. The x speed, which the pilot held fixed, therefore matters
  for the coverage as much as the lane crossing does: the owner's point that τ_cross
  is only the transverse component is confirmed from the other side.
* **Effective anticipation** exceeds the nominal one in every geometry (4.3–4.7 s
  against 3.3–3.6 s at 10–10.8 cm; 16 s against 10.8 s for τ_cross 7.2): the belt
  stops while the part crosses the visible stretch, which lengthens the time between
  being seen and entering the zone. The nominal τ_vis is the design quantity; the
  effective one explains why the curves saturate before the nominal anticipation
  reaches the move duration.

### 4.2 H3: physical entry length

Original cell (spacing 12, v_y 10), entry 10 cm against 35 cm, same visible lengths,
same arrivals at the feed point:

| | Δ `none` (35 − 10) [s/part] | ΔG (35 − 10) at 5 / 7.5 / 10 cm [s/part] |
|---|---|---|
| `nearest_job` | +10.23 ±0.59 | +0.23 ±0.53 / +0.23 ±0.46 / +0.32 ±0.53 |
| network | +9.9…+10.1 ±0.4 | +0.25 ±0.34 / +0.06 ±0.32 / +0.03 ±0.27 |

The longer entry adds 10.0–10.2 s to every part's flow time: the 8.33 s of free
transit plus ≈1.7–1.9 s that the pairing attributes to the installation itself. The
likely mechanism, not verified by trace, is that a part on a 35 cm entry zone is
stopped by every processing and blocking stop of its belt during a longer stretch
than on a 10 cm one. The gain of pre-positioning at a given visible length is the
same within noise for both entry lengths (ΔG within ±0.5 s, intervals including zero),
so H3 holds: the extended bench measures the camera's reach, not the installation,
provided percentages are taken against the flow net of the free transit.

### 4.3 The network (secondary)

Its curves have the same shape and the same whole-zone gain as the rule's in every
geometry (0.72–1.81 s/part; knee 1.2–1.8 s point estimate for τ_cross ≤ 2.4, 2.4–3.3 s
for 3.6–4.8, 10.8 s for 7.2 where the curve is noisy), with the 35 cm entry outside
its training range (5–15 cm) producing no visible degradation (its `none` level is
below the rule's in every geometry, 21.4–47.0 against 22.7–48.1 s/part). Two protocol
points follow: the network's baseline at ℓ_v = 0 should be `when_idle` at ℓ_v = 0
(same mode token, no upstream candidates) rather than `none`, which removes the
−0.3…+0.2 s offset of check (ii); and the network's curves are 1.5–2× less noisy than
the rule's (half-widths 0.11–0.28 against 0.28–0.46 s at τ_cross ≤ 3.6), which the
seed count of the confirmatory bench can take into account.

### 4.4 What the pilot changes in the brief

1. **The design reference is a number of seconds, not a multiple of τ_cross.** At
   the nominal load every geometry tested keeps ≥ 90 % of the pre-positioning gain
   with 2.5 s of anticipation (7.5 cm at 3 cm/s) and ≥ 80 % with 1.67 s. The
   normalization by τ_cross is a hypothesis to report as tested and rejected outside
   1.2–2.4 s; the figure in τ_vis / τ_cross stays as the evidence.
2. **v_x must be a factor.** The pilot fixed it at 10 cm/s and found no need for more
   than 2.5 s even at τ_cross 7.2 s, while phase 4's head at 5 cm/s on both axes
   needed the whole zone. The confirmatory bench should vary v_x (5 and 10 cm/s, or
   the "both axes slow" configuration) or the result will not transfer to the cell
   Paper 2 already describes.
3. **Report idle time and moves per episode with every curve.** The opportunities are
   fixed by the head's idle time; the heavy load of the brief will reduce them further
   (phase 4: ≈0 gain at ρ 0.93), so the "minimum useful benefit" threshold of §2 (1 s
   per part, lower bound 0.5 s) will exclude the heavy regimes and should be kept.
4. **Precision.** With 30 realizations the slow-head gains have ±0.6–0.9 s on ≈1.5 s
   and the conservative knee spans 1.7–4.8 s. Sixty realizations would reduce the
   half-widths by 30 %; the owner decides between more seeds for the slow geometries
   and wider stated intervals.
5. **Keep** the 35 cm common entry, the flow net of the free transit, the seconds per
   part as the primary measure, the `when_idle`-at-0 baseline for the network, the
   exact-equality check of H0 as a regression test, and the development seeds
   3000–3029 for any further pilot. The confirmatory bench needs a new reserved
   range.
