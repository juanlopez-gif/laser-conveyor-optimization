# Geometry bench: does the head's crossing time predict the camera coverage?

Status: **confirmatory bench run** (0.7.21, 2026-11-23); **belt-speed bench designed, not
run** (0.7.27, section 7). Sections 1–3 were written before the pilot ran; section 4 reports it. The confirmatory bench (owner's brief of 2026-11-23: 36
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
  2.4, 8 → 27 for 7.2. "Gain per move" is an aggregate ratio (seconds saved summed over
  all parts, divided by the number of moves), not the saving of one part: it exceeds
  the move's own duration by a factor 2–9. *Interpretation, not yet measured:* a second
  saved on the head's cycle is saved by every part queued behind it in the busy period,
  which is longer when the head is busier. The confirmatory bench records the number
  of unfinished parts at each move (H5, §5) to check how much of the saving is
  transmitted to the waiting parts. The move
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

## 5. Confirmatory protocol (frozen 2026-11-23, before the run)

Owner's instructions after the pilot (2026-11-23): a common grid in seconds; v_x = 5
and 10 cm/s crossed with the nine spacing × v_y cells; `nearest_job` primary, the
frozen network secondary; seconds of penalized flow saved per part, the cost of less
coverage, `when_idle` vs `none` at the same length, and for the network also
`when_idle(ℓ)` vs `when_idle(0)`; 60 paired realizations; seeds 9000–9059 once checked
unused; the grid, the minimum useful benefit and the 90 % criterion frozen before the
run; mechanism separated from interpretation. The final question of the paper: how
much anticipation does the camera need when the head can pre-position, and what
predicts it better, the transverse crossing or the moves the head actually makes?

### 5.1 Seeds

Seeds 9000–9059. Check (2026-11-23): no episode in `results/experiments/**/*.json` and
no bench or pilot in the documentation used them. The only occurrence in the
repository is `tests/test_rl2_search.py`, which calls `general_config(9000 + i)` to
draw *general* cells (random geometry and rates) for two unit tests; those are
different cells, and no policy result on the real cell at these seeds exists. The
owner confirmed the range on that basis. `geometry_bench.py --confirmatory` sets the
range, refuses to overwrite its `bench.json`, and refuses the range without the flag.

### 5.2 Design (`experiments/geometry_bench.py`, constants at the top of the file)

* **Regimes (72):** spacing 6 / 12 / 18 cm × v_y 5 / 10 / 15 cm/s × v_x 5 / 10 cm/s ×
  load nominal / high × Poisson / bursts. Loads by the arrival rates only: nominal =
  mean inter-arrivals 10 / 12 / 14 s (ρ_p 0.64), high = 7 / 8.5 / 10 s (ρ_p 0.90);
  belts 3 cm/s, processing 2.5 s, geometry, penalties and the 35 cm entry zone fixed.
  Bursts: the on/off source of phase 4 (`on_fraction` 0.3, `on_mean` 20 s) at the
  same mean rate per belt. Offered load (ρ_p), observed saturation (blocking events,
  blocked belt-seconds) and the head's idle fraction are reported per regime.
* **Coverage grid, common in seconds:** τ_vis ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3.33, 5, 7.5,
  10} s, i.e. ℓ_v = 3 τ_vis ∈ {0, 1.5, …, 30} cm (within the 35 cm entry). The nominal
  anticipation is the design quantity; the effective one is recorded.
* **Modes:** `none` and `when_idle` at every grid point; same arrivals and processing
  times (common random numbers).
* **Controllers:** `nearest_job` (primary) and the frozen network `rl2_gen_s_b`
  (secondary; the 35 cm entry and v_x = 5 with v_y = 15 lie at or beyond its training
  ranges, which its results also measure).
* **Realizations:** 60 per configuration, seeds 9000–9059. 72 × 10 × 2 × 2 × 60 =
  172 800 episodes, plus the physical-length check below (6 720). Smoke test on
  seeds 3000–3001: 5 984 episodes in 0.44 CPU-hours, so ≈ 13 CPU-hours, ≈ 20 min on
  40 workers.
* **Physical-length check (H3):** the original geometry (12 / 10 / v_x 10) at entry
  10 cm, both loads and both arrival processes, at the grid points ≤ 10 cm, against
  the 35 cm entry at the same points.
* **Traces per episode:** as in the pilot, plus the number of unfinished parts in the
  system at each pre-positioning move (H5).

### 5.3 Measures and criteria (frozen)

* **Primary:** G(τ_vis) = J_none(τ_vis) − J_when_idle(τ_vis) in seconds of penalized
  flow time per part, paired, 95 % CI; also the unpenalized flow, blocking events and
  blocked belt-seconds. Percentages, where given, are against the flow net of the free
  transit (entry length / belt speed).
* **Cost of less coverage:** J_when_idle(τ_vis) − J_when_idle(10 s) at τ_vis = 1.5, 2.5
  and 3.33 s, paired.
* **Network only:** `when_idle(τ_vis)` − `when_idle(0)` (value of widening the view
  with the mode token fixed) and `none(0)` − `none(τ_vis)` (value of seeing with the
  same actions). Two different questions, reported apart.
* **Minimum useful benefit:** a regime is worth sizing the coverage for when G(10 s)
  ≥ 1.0 s/part and its lower 95 % bound ≥ 0.5 s/part. Otherwise the absolute effects
  are reported and the sufficient coverage is left *undetermined* for that regime.
* **Knee (sufficient coverage):** the first grid anticipation τ whose paired contrast
  G(τ) − 0.9 G(10 s) is ≥ 0 (point estimate) and, conservatively, whose lower
  simultaneous 95 % bound is ≥ 0 with a Bonferroni factor over the nine non-zero grid
  points (z = 2.773). Both are reported; the conservative one is the headline.
* **Predictors of the knee** (rule, regimes with a useful benefit): Spearman rank
  correlation of the knee with τ_cross and with τ_pp (the mean duration max(t_x, t_y)
  of the pre-positioning moves at 10 s, `when_idle`), and leave-one-out mean absolute
  error of four models: knee = τ_cross; knee = a τ_cross; knee = b τ_pp; knee =
  constant. The predictor with the lowest leave-one-out error is the paper's answer;
  if the constant wins, the answer is "a number of seconds at this load", as the pilot
  suggests.
* **H0** (same τ_cross, identical cells for the rule) and the checks (ii)–(iii) of §3
  are rerun as regression tests.
* **H5 (mechanism):** the aggregate gain per move divided by the mean number of
  unfinished parts at the moves should be of the order of the saving one pre-positioned
  part can receive (bounded by the move duration plus the chase; within a factor of
  two). If it is much larger, the saving is not transmitted through the queue as
  assumed and the interpretation of §4.1 is dropped.

### 5.4 Outputs

`results/experiments/geometry_bench/{bench.json, tables.md, extract.csv.gz}` (the
gzip extract with the per-episode fields is the versioned data of the paper, copied to
`docs/papers/paper2/data/` after the run), figures
`docs/methods/figures/geometry_bench_{nearest_job,net}_{poisson,burst}.pdf` and
`geometry_bench_{nearest_job,net}_knee.pdf`, and section 6 of this document.

## 6. Results of the confirmatory bench (seeds 9000–9059, run 2026-11-23)

> Corrections 2026-11-23 (Paper 2 draft 0.7, after an external audit; no data changed):
> the Spearman correlations of §6.3 are now computed with average ranks for ties (the
> earlier values broke ties by position); the v_x effect of §6.2 is given as a ratio per
> geometry instead of "roughly doubles"; §6.4 separates the Poisson and bursty high-load
> cells; §6.5 replaces "no visible degradation" by the direct level comparison; the
> cost of 3.33 s is the largest value observed in the 18 cells, not a bound.

179 520 episodes, 1 139 s on 40 workers (12.0 CPU-hours). Tables:
`results/experiments/geometry_bench/tables.md`; the same tables regenerate from the
versioned extract (`python experiments/geometry_bench.py --from-extract
docs/papers/paper2/data/geometry_bench.csv.gz`, checked identical). Figures:
`docs/methods/figures/geometry_bench_nearest_job_{poisson,burst}.pdf` (curves),
`geometry_bench_nearest_job_knee.pdf` (knee against the two predictors), and the same
for `net`. One analysis bug was fixed after the run and before any reading of the
results was written (the regime list was built from the first 20 000 rows; no data
changed).

**Checks.** 0 candidate violations of the visible stretch in 179 520 episodes. H0
exact: the three τ_cross = 2.4 s cells are identical for the rule in all eight v_x ×
load × arrivals blocks (max |ΔJ| ≤ 2.6 · 10⁻¹³ s over 2 400 paired episodes each); for
the rule the bench therefore has **56 distinct cells**, and the predictor analysis
counts the triplet once. For the network they differ by up to 9.6 s in one episode
(absolute-position tokens). At τ_vis = 0, `when_idle` = `none` exactly for the rule;
the network differs by up to 13 s in one episode (mode token). Parts per episode: 76
(nominal) and 109 (high) with Poisson, 78 and 113 with bursts, i.e. the nominal draw
matches phase 4 (77.5).

### 6.1 Where coverage is worth sizing (rule)

The whole-grid gain G(10 s) spans −0.08 to +2.44 s/part over the 56 cells. It passes
the frozen usefulness threshold (≥ 1 s/part, lower bound ≥ 0.5) in **18 of 56**: 10 of
28 at the nominal load and 8 of 28 at the high load; 13 of 28 with Poisson arrivals and
5 of 28 with bursts; 10 of 28 with v_x = 5 and 8 of 28 with v_x = 10. In the other 38
cells the absolute effect is reported and the sufficient coverage is left
undetermined, as pre-registered. Bursts halve the gain at the nominal load (0.4–1.5
against 0.8–2.4 s/part with Poisson), as in Paper 2; at the high load the gains are
0.1–2.0 s/part with half-widths of 0.5–1.6 s, so most of them are undetermined by
precision rather than by size. The gain is 82–96 % unpenalized flow time; the blocking
penalty contributes ≤ 0.44 s/part.

### 6.2 How much anticipation (rule, the 18 useful cells)

* Knee (90 % of G(10 s)), point estimate: 0.5–5.0 s, median 2.5–3.3 s. Conservative
  (simultaneous 95 %): 2.0–7.5 s, undetermined in one cell (12 / 5 / v_x 5, high,
  Poisson, where no grid point is conservatively at 90 %).
* Share of G(10 s) retained: at 2.5 s min 62 %, median 94 %; at 3.33 s min 71 %,
  median 97 %; at 5 s min 94 %, median 100 %.
* Cost of operating with less coverage, `when_idle` at τ_vis against at 10 s: at
  3.33 s at most +0.44 s/part (median +0.05; significant in 5 of 18 cells); at 2.5 s at
  most +0.77 (median +0.09; 5 of 18); at 1.5 s at most +1.50 (median +0.33; 10 of 18).

**The x axis is a first-order factor.** At the nominal load with Poisson arrivals,
halving v_x from 10 to 5 cm/s multiplies the gain of pre-positioning by 1.19–2.36 in the
four geometries (τ_cross ≤ 2.4 s) where both speeds pass the criterion (1.03–1.32 →
1.49–2.44 s/part; by 1.1–1.5 in the three where neither does), lengthens the along-belt
component of the moves from 0.44–1.27 s to 1.57–2.42 s (seven geometries), lowers the
head's idle fraction under `none` from 17–27 % to 12–15 % in the four cells, and moves the knee from 1.5–3.33 s (conservative 2.0–3.33) to 3.33–5 s
(conservative 5–7.5). The pilot's finding that the knee "barely moves" was
conditional on v_x = 10 cm/s; the pilot's recommendation to vary v_x is confirmed.

### 6.3 What predicts the knee (rule, 18 cells, pre-registered comparison)

| predictor | Spearman with the knee | leave-one-out MAE [s] | fitted coefficient |
|---|---|---|---|
| knee = τ_cross | +0.38 | 1.53 | — |
| knee = a · τ_cross | +0.38 | 1.75 | a = 0.86 |
| knee = b · τ_pp (mean planned duration of the moves started) | +0.50 | **1.02** | b = 1.74 |
| knee = b′ · (t_x + t_y) | +0.47 | 1.18 | b′ = 1.33 |
| knee = constant | — | 1.13 | 2.76 s |

(Spearman with average ranks for ties; τ_pp is the planned duration recorded when the move
is decided, before any revocation; b = 1.74 is an empirical fit on these 18 traces.)

By the frozen criterion the answer is **the moves the head starts**: τ_pp
predicts the knee with the lowest error (1.0 s), the transverse crossing time with the
highest (1.5 s, and 1.75 s when rescaled), and a constant of 2.8 s lies in between
(1.1 s). Two qualifications the data impose: (i) no predictor is precise, the errors
are 2–3 grid steps on a 0.5 s grid and the knee / τ_pp ratios range 0.8–2.6 (two cells
at 4.7–5.3); (ii) τ_pp beats the constant by 0.1 s only, so for a designer without
traces "about 3 s of anticipation at these loads" is nearly as good as "about 1.7 times
the mean move". τ_cross is the worst of the candidates because it ignores the x
component, which the v_x factor shows to be of the same order as the lane change
(moves are y-dominated in 7–57 % of the cases, never a majority at v_x = 5).

### 6.4 Mechanism (H5)

With the number of unfinished parts recorded at each move, the implied saving per
part (aggregate gain per move / WIP at the move) is 1.0–2.0 times the move's own
duration in all ten useful nominal-load cells: the pre-registered "within a factor of
two" holds, and the reading of the pilot (a second saved on the head's cycle is saved by
the parts queued behind it) is supported there. At the high load the ratio is 3.5–9.5 in the
five Poisson cells (2.5–6.8 moves per episode, head idle 6–10 % under `none`, aggregate
gain per move 20–87 s) and 2.1–2.5 in the three bursty cells (13–17 moves per episode,
idle 14–20 %, 7–17 s per move): the aggregate gain per move is more than the queue at
the move explains, so for those cells the transmission
mechanism is **not** supported and the interpretation is dropped; what remains is the
measured effect itself (flow time, not penalty, since the penalty part is ≤ 0.2 s/part
there). A plausible alternative, not tested, is that one early service at high load
prevents a cascade of belt stops that would delay parts arriving later than the
WIP count sees.

### 6.5 The network (secondary)

Useful in 17 of 72 regimes (14 nominal, 14 Poisson, 13 with v_x = 5; G(10 s) 1.01–1.68
s/part in those; 0.13–1.68 over the 72 regimes, 0.38–1.68 at the nominal load), curves of the same shape as the rule's but noisier and with slower saturation:
knee point 1–10 s, and the conservative knee is never reached before 10 s in any of the
17 (shares at 3.33 s: min 34 %, median 90 %). No predictor works for it (Spearman
≤ 0.14, the constant of 2.7 s has the lowest error). The value of widening the view
with the mode token fixed, `when_idle(τ)` − `when_idle(0)`, equals G within noise at the
nominal load (e.g. +1.03 ±0.49 against G = +1.13 ±0.58), so the `none` baseline does not
distort the network's curve there. The value of seeing with the same actions,
`none(0)` − `none(τ)`, is within ±0.3 s/part at the nominal load and positive at the high
load with bursts (+0.6 to +1.4 ±0.65 s/part at 10 s), consistent with the one
significant B-vs-A cell of Paper 2 (heavy, bursts). With `when_idle` at 10 s the network's
level lies below the rule's in 71 of the 72 regimes (significantly in 70), by 3.1–11.1
s/part at v_x = 5 and 0.2–4.9 at v_x = 10, and above it in one (18 / 5 / v_x 10, high,
Poisson, +1.26 ±1.07): the extrapolation beyond the training ranges does not show in the
levels.

### 6.6 Physical entry length (H3)

The longer entry adds +10.35 ±0.52 (nominal, Poisson), +11.11 ±0.55 (nominal, bursts),
+13.19 ±0.71 (high, Poisson) and +11.69 ±0.64 s/part (high, bursts) to the rule's
`none` level: the 8.33 s of free transit plus 2.0–4.9 s that grow with the load and are
therefore the belt stops acting on the parts of the longer entry zone. The gain of
pre-positioning at a given anticipation differs between the two entry lengths by at
most ±0.26 s/part (nominal) and ±0.46 (high), never significantly: no difference was detected (no equivalence margin was fixed), and
within that precision the bench measures the camera's reach, not the installation.

### 6.7 Answer to the paper's question

How much anticipation does the camera need when the head can pre-position? In the
regimes where pre-positioning is worth at least 1 s per part (Poisson arrivals, or
bursts with a slow x axis), 2.0–3.3 s with the head at 10 cm/s on both axes and
3.3–5 s with the x axis at 5 cm/s, conservatively; 3.33 s (10 cm at 3 cm/s) cost at
most 0.44 s/part (median 0.05; the largest value observed in the 18 cells) against the
full 10 s of coverage. What predicts
that coverage? The duration of the moves the head actually makes, with both axes,
about 1.7 times its mean (an empirical fit on the 18 traces), and not the transverse
crossing time, which was the worst
of the candidates tested; with the caveat that a flat 3 s does nearly as well as
either, because the knees lie in a narrow band once the gain is large enough to size
for. Same-τ_cross cells are identical by the model's structure, so the question "does
the same crossing time give the same coverage?" is answered yes for the rule and
trivially so; the informative comparison is across different τ_cross and v_x. Every cell runs its
belts at 3 cm/s, so seconds and centimetres are one unit inside the bench; which scale
transfers to another belt speed is untested here; §7–§8 run that experiment (3 vs 5 cm/s).

## 7. Belt-speed bench: protocol (frozen 2026-11-23, before the run; results in §8)

Section 6.7 and the external audit of Paper 2 (draft 0.7) leave one question open:
every cell of the geometry bench runs its belts at 3 cm/s, so seconds of anticipation
and centimetres of visible belt are one variable (ℓ_v = 3 τ_vis) and the data cannot
say which scale carries over to another belt speed. This section fixes the bench that
answers it, in the style of §5: hypotheses, margin, design and criteria written before
any result. Script: `experiments/belt_speed_bench.py` (constants at the top of the
file, the same `Bench` / `tables` / `figures` structure as `geometry_bench.py`, the
analysis of the extract without PyTorch). The confirmatory run waits for three owner
decisions, listed in §7.6; nothing in this section was changed after the smoke test.

### 7.1 Question and hypotheses (written before any result)

At 3 cm/s, 10 cm of visible belt give 3.33 s of anticipation, and §6.2 found that
those 3.33 s cost at most 0.44 s/part (median 0.05) against 10 s in the 18 useful
cells. If the belt runs at 5 cm/s, the same 10 cm give 2.00 s, and 3.33 s need
16.7 cm (exactly 50/3 cm, so that the two anticipations coincide). Does keeping 10 cm
keep the benefit of pre-positioning, must the view grow to 16.7 cm to keep the 3.33 s,
or does neither reproduce the 3 cm/s result because the interception changes? Three
hypotheses, mutually exclusive once the margin δ of §7.2 is fixed, decided **per cell**
by the paired cost C_v(ℓ) = J_when_idle,v(ℓ) − J_when_idle,v(50 cm) at v = 5 cm/s:

* **H_cm (the distance transfers).** C_5(10 cm) ≤ δ. Reading: what the camera must
  cover is a stretch of belt; the specification "10 cm" survives the speed change.
* **H_s (the seconds transfer).** C_5(10 cm) > δ and C_5(16.7 cm) ≤ δ. Reading: the
  anticipation must cover the head's move, whose duration does not depend on v_belt,
  so the curve G(τ_vis) in seconds is the same at both speeds and the specification
  "3.33 s" survives, not "10 cm". This is the reading Paper 2 calls the physical
  argument; it is the hypothesis the author expects, and it is written here so that
  the data, not the expectation, decide.
* **H_neither (the interception changes).** C_5(16.7 cm) > δ. Reading: at 5 cm/s a
  part travels 1.67 times further during the head's move and crosses the 35 cm laser
  zone in 7 s instead of 11.7 s, so the meeting point moves downstream, more parts
  reach the end of the zone, and the anticipation needed in seconds grows (or the
  benefit changes size); neither specification reproduces the 3 cm/s result.
* **Design result, not a hypothesis.** A cell that passes the benefit criterion at
  3 cm/s (all four do) and fails it at 5 cm/s (G_5(50 cm) < 1 s/part or lower bound <
  0.5) is reported as "benefit lost at this speed": pre-positioning is then not worth
  sizing the camera for at 5 cm/s, and the question is moot for that cell.

**Predictions written beforehand, from the 3 cm/s curves of §6** (versioned extract,
35 cm entry, nominal load, Poisson, `nearest_job`; paired costs against 10 s). Under
H_s the cost of 10 cm at 5 cm/s should equal the cost of 2.00 s at 3 cm/s, which was
+0.68 ±0.41, +0.21 ±0.20, +0.54 ±0.53 and +0.03 ±0.03 s/part in the cells 12 / 10 /
v_x 5, 12 / 10 / v_x 10, 6 / 15 / v_x 5 and 6 / 15 / v_x 10, and the cost of 16.7 cm
should equal that of 3.33 s, +0.06, +0.01, −0.05 and +0.00. Under H_cm the cost of
10 cm at 5 cm/s should stay at the 3.33 s values, ≈ 0. Two consequences, stated now so
that they are not discovered after the run: (i) the two cells with v_x = 10 cm/s
answer "10 cm keeps the benefit" under H_cm *and* under H_s (their knee at 3 cm/s is
≤ 2 s), so they cannot separate the two; the discriminating cells are the two with
v_x = 5 cm/s. (ii) In those two cells the costs predicted under H_s lie 0.04–0.18
s/part above the margin of 0.5, and with 60 realizations the half-widths of C are
≈ 0.4–0.5 s/part, so the conservative verdict there is expected to be *undetermined*
between H_cm and H_s whatever the truth; the point verdict and the alignment reading of
§7.4 carry the direction, and the conservative verdict can only establish H_cm (upper
bound ≤ δ) or H_neither (lower bound > δ) cleanly. No affordable number of
realizations removes (ii), because it is the margin, not the noise, that the predicted
costs sit close to.

The 3 cm/s predictions above come from the geometry bench, run before the observable
tie-break key of 0.7.26 (`docs/methods/tiebreak_audit.md`: `nearest_job` paired cost
changed by at most ±0.6 % on the reference cell); the belt-speed bench runs on the
current key, so the C_3(10 cm) column of §7.4 is also the check that the 3 cm/s costs
did not move with it.

### 7.2 The margin δ: "keeps the benefit" (owner's decision)

A shorter view *keeps the benefit* when its paired cost against the full view is at
most δ s/part. Proposed: **δ = 0.5 s/part**. Justification from the coverage results of
§6.2: the largest cost of 3.33 s against 10 s observed in the 18 useful cells was
0.44 s/part (median 0.05), the figure Paper 2 reports as the practical price of the
10 cm camera; and 0.5 s/part is the lower-bound threshold of the benefit criterion
itself (half the minimum useful benefit). "Keeps" therefore means "no worse than the
worst loss already accepted at 3 cm/s". Alternatives the owner may prefer: 0.44 (the
exact largest value) or 0.25 (half of it, stricter). The constant is `MARGIN` in the
script; **the owner fixes it before the run** and it is not changed afterwards. Two
readings are reported for every cost: *conservative* (keeps if the upper 95 % bound
≤ δ, does not keep if the lower bound > δ, otherwise undetermined; the headline) and
*point* (the estimate against δ).

### 7.3 Design (`experiments/belt_speed_bench.py`)

* **Belt speed:** 3 and 5 cm/s (`conveyors.speed`), the factor of the bench.
* **Cells (4):** spacing 12 cm / v_y 10 cm/s (τ_cross 2.4 s) and spacing 6 cm /
  v_y 15 cm/s (τ_cross 0.8 s), each with v_x = 5 and 10 cm/s: a slow and a fast
  longitudinal axis, all four useful at 3 cm/s in §6.1. Nominal load (mean
  inter-arrivals 10 / 12 / 14 s, ρ_p 0.64), Poisson arrivals; processing 2.5 s,
  penalties and geometry fixed.
* **Common physical entry zone of 50 cm** (10 s at 5 cm/s, 16.7 s at 3 cm/s).
  Checked by loading the configuration: `geometry.entry_length = 50` moves the laser
  zone to x ∈ [50, 85], the derived workspace to the same x interval and the head's
  home to x = 67.5 (the centre of the workspace), exactly as the 35 cm entry did in
  §5; `simulation.visible_length` is accepted up to the entry length, so 50 cm is a
  valid grid point. Levels of `none` are reported net of the free transit
  (entry / v_belt); paired differences within one speed are unaffected by it. The
  assumption that a longer entry does not change the gain at a given visible length is
  H3 of §4.2 and §6.6 (10 vs 35 cm: ΔG within ±0.26 s/part at the nominal load).
* **The same arrival realizations at both speeds.** `generate_arrivals` draws the
  arrival instants per belt from `random.Random(f"{seed}/arrivals/{k}")` with the mean
  inter-arrival only, and the processing times from `"{seed}/processing/{k}"`; the belt
  speed enters neither. Checked for seed 3100: 80 parts with identical instants at 3
  and at 5 cm/s. The run records the number of parts and the sum of the arrival
  instants of every episode and the tables check that they agree across speeds,
  lengths, modes and controllers for every (cell, seed). Because the realizations are
  common, differences *between speeds* are paired per seed as well.
* **Grid of visible lengths, common in centimetres (12 points):** ℓ_v ∈ {0, 1.5, 3,
  4.5, 6, 7.5, 10, 15, 16.7, 25, 30, 50} cm. In seconds this is {0, 0.5, 1, 1.5, 2,
  2.5, 3.33, 5, 5.56, 8.33, 10, 16.67} at 3 cm/s and {0, 0.3, 0.6, 0.9, 1.2, 1.5, 2,
  3, 3.33, 5, 6, 10} at 5 cm/s. Five pairs give the same anticipation at both speeds:
  4.5 / 7.5 cm (1.5 s), 6 / 10 cm (2 s), 10 / 16.7 cm (3.33 s), 15 / 25 cm (5 s) and
  30 / 50 cm (10 s). The reference of every cost is the full view, 50 cm.
* **Modes:** `none` and `when_idle` at every length; **controllers:** `nearest_job`
  (primary) and the frozen network `rl2_gen_s_b` (secondary; a 5 cm/s belt lies inside
  its training range of 1.5–6 cm/s, the 50 cm entry far outside 5–15 cm).
* **Realizations:** 60 per configuration proposed (as in §5; §7.1 (ii) explains why
  more would not change the conservative verdict in the discriminating cells). 2 speeds
  × 4 cells × 12 lengths × 2 modes × 60 = 11 520 episodes per controller; from the
  smoke test, ≈ 0.012 CPU-s per episode for the rule and ≈ 0.5 for the network, i.e.
  ≈ 1.7 CPU-hours, ≈ 3 min on 40 workers.
* **Seeds:** a new reserved range, fixed by the owner and used once (`RESERVED_BASE`
  in the script, None until then; `RESERVED_N = 60`). The script refuses
  `--confirmatory` while the range is unset, refuses to overwrite the `bench.json` of a
  confirmatory run, and refuses the used ranges 7000–7029 and 9000–9059 (and the
  reserved one) without the flag. Development pilots use 3100+.
* **Traces per episode:** as in §5 (moves, revocations, t_x, t_y, max, y-dominated
  share, WIP at the move, idle and waiting time, effective anticipation).

### 7.4 Measures and criteria (frozen)

* **Primary:** G_v(ℓ) = J_none,v(ℓ) − J_when_idle,v(ℓ) in seconds of penalized flow
  time per part, paired, 95 % CI, at every length and speed; the paired costs C_v(10 cm)
  and C_v(16.7 cm) against 50 cm at each speed; the benefit criterion of §5.3 applied to
  G_v(50 cm) at each speed (G ≥ 1 s/part, lower bound ≥ 0.5); and the verdict of §7.1
  per cell at 5 cm/s, conservative (headline) and point. C_3(10 cm) is reported next to
  it as the check that the 3 cm/s result of §6 is reproduced with the 50 cm entry.
* **Equal distance and equal anticipation:** G_3(ℓ) − G_5(ℓ) at every grid length,
  and G_3(ℓ₃) − G_5(ℓ₅) at the five pairs with ℓ₃ / 3 = ℓ₅ / 5, both paired per
  realization; and G_3(50 cm) − G_5(50 cm), the level of the full-view benefit across
  speeds. Alignment reading: for each scale, the mean absolute difference between the
  shares of G_v(50 cm) retained at the matched points (the 11 non-zero lengths for the
  distance scale, the 5 pairs for the time scale); the scale with the smaller value is
  reported as the better-aligned one. Descriptive, no test; it is the reading that
  carries the direction when the verdict of §7.1 is undetermined.
* **Knee** (sufficient coverage) at each speed, in cm and converted to s: the first
  grid length whose paired contrast G_v(ℓ) − 0.9 G_v(50 cm) is ≥ 0 (point) and whose
  lower simultaneous 95 % bound is ≥ 0 (conservative, Bonferroni over the 11 non-zero
  lengths, z = 2.838; the geometry bench used 2.773 for its 9). Both are reported; the
  knee is secondary here.
* **Network:** the same tables, secondary; no verdict of the network overrides the
  rule's.
* **Checks:** visible stretch respected (0 candidate violations), common arrivals
  (above), `when_idle` = `none` at ℓ = 0 for the rule (exact) and the network (mode
  token, reported).
* **Interpretation rules kept from §6:** "largest value observed", never "bounded"; a
  cost that is undetermined stays undetermined; the mechanism (traces) is reported apart
  from the verdict.

### 7.5 Outputs and smoke test

`results/experiments/belt_speed_bench/{bench.json, tables.md, extract.csv.gz}` (the
extract, copied to `docs/papers/paper2/data/belt_speed_bench.csv.gz` after the run, is
the versioned data; `--from-extract` regenerates the tables without PyTorch), figures
`docs/methods/figures/belt_speed_bench_{nearest_job,net}_curves.pdf` (every cell drawn
twice, x in cm and x in s, both speeds overlaid, dotted lines at 10 and 16.7 cm and at
3.33 s) and `_cost.pdf` (the two costs at each speed against δ), and section 8 of this
document.

Smoke test (2026-11-23, development seeds, one cell): `--tag _belt_smoke --seed-base
3100 --seeds 2 --cells 0` ran end to end in 12 s (192 episodes, 0.01 CPU-hours, both
controllers), with 0 candidate violations of the visible stretch, identical arrivals at
both speeds in every (cell, seed), and `when_idle` = `none` at ℓ = 0 exactly for the
rule; `--analyse-only` and `--from-extract` regenerate identical tables (the latter
without PyTorch), and the figures render. With two realizations its numbers are noise
and were not read as results; they live under `results/experiments/_belt_smoke/`
(git-ignored, figures included). The guards were exercised: seed bases 9010 and 7020
are refused, and `--confirmatory` is refused while the range is unset.

### 7.6 Owner decisions before the run (taken 2026-11-23)

1. Reserved seed range **10000–10059** (`RESERVED_BASE = 10000`, 60 seeds), checked
   unused as in §5.1 (no seed key in that range in any `results/experiments/**/*.json`;
   the test suites use 100000+). Used once by the run of §8.
2. Margin **δ = 0.5 s/part** (`MARGIN = 0.5`, as proposed in §7.2).
3. **`nearest_job` only** (`--who nearest_job`); the network did not run.
4. The explicit go, after a smoke test on development seeds 3100–3101 (one cell,
   `nearest_job`, 96 episodes: 0 violations, identical arrivals at both speeds, exact
   identity at ℓ = 0). Nothing in §7.1–§7.5 was changed after the smoke test or the run.

## 8. Results of the belt-speed bench (seeds 10000–10059, run 2026-11-23)

`experiments/belt_speed_bench.py --tag belt_speed_bench --confirmatory --who nearest_job`:
11 520 episodes (2 speeds × 4 cells × 12 lengths × 2 modes × 60 seeds), 7 s of wall
time on 40 workers, 0.03 CPU-hours. Outputs: `results/experiments/belt_speed_bench/`
(`bench.json`, `tables.md`, `extract.csv.gz`), the extract versioned as
`docs/papers/paper2/data/belt_speed_bench.csv.gz` (`--from-extract` regenerates
identical tables without PyTorch, checked), figures
`docs/methods/figures/belt_speed_bench_nearest_job_{curves,cost}.{pdf,png}`. All
numbers below are the rule `nearest_job`, mean ± 95 % half-width over the 60 paired
realizations, seconds of penalized flow time per part; the criteria are those frozen in
§7.2 and §7.4 and were not changed after the run.

### 8.1 Checks

0 candidate violations of the visible stretch over 11 520 episodes; the number of
parts and the sum of their arrival instants agree across the two speeds, every length,
mode and controller in all 240 (cell, seed) pairs (the realizations are common, as
designed); `when_idle` = `none` at ℓ = 0 exactly (max |ΔJ| = 0). The C_3(10 cm) check
column reproduces the 3 cm/s costs of §6.2 within the precision: +0.10 ±0.47, +0.18
±0.23, −0.15 ±0.26 and −0.11 ±0.17 s/part in the four cells against the §6 values
+0.06, +0.01, −0.05 and +0.00 predicted in §7.1 (every interval contains its
prediction), so the observable tie-break of 0.7.26 did not move the 3 cm/s result
detectably either.

### 8.2 The frozen question, per cell (headline)

| cell | G_3(50 cm) | useful at 3 | G_5(50 cm) | useful at 5 | C_5(10 cm = 2.00 s) | C_5(16.7 cm = 3.33 s) | verdict (conservative / point) |
|---|---|---|---|---|---|---|---|
| spacing 12 / v_y 10 / v_x 5 (τ_cross 2.4 s) | +1.74 ±0.87 | yes | +0.62 ±0.73 | no | −0.00 ±0.51 | +0.12 ±0.39 | **benefit lost at this speed** / same |
| spacing 12 / v_y 10 / v_x 10 (τ_cross 2.4 s) | +0.99 ±0.38 | no (by 0.01 on the point; lower bound 0.61) | +1.76 ±0.39 | yes | +0.06 ±0.26 (upper 0.32) | −0.02 ±0.24 (upper 0.22) | **H_cm** / H_cm |
| spacing 6 / v_y 15 / v_x 5 (τ_cross 0.8 s) | +1.58 ±0.89 | yes | +0.47 ±0.55 | no | +0.48 ±0.55 | +0.12 ±0.45 | **benefit lost at this speed** / same |
| spacing 6 / v_y 15 / v_x 10 (τ_cross 0.8 s) | +1.16 ±0.34 | yes | +1.22 ±0.37 | yes | +0.05 ±0.18 (upper 0.23) | +0.00 ±0.14 (upper 0.14) | **H_cm** / H_cm |

* **10 cm keep the benefit at 5 cm/s in the two cells with v_x = 10 cm/s** (H_cm,
  conservative: the upper 95 % bounds of C_5(10 cm), 0.32 and 0.23 s/part, are below
  δ = 0.5). In those cells 16.7 cm keep it too (upper bounds 0.22 and 0.14) and add
  nothing detectable over 10 cm (C_5(16.7) − C_5(10) = −0.08 and −0.05 on the point).
  As §7.1 (i) stated before the run, these two cells cannot separate H_cm from H_s
  (their predicted costs under H_s, +0.21 and +0.03, lie inside the observed intervals
  as well), so the result there is practical, "10 cm suffice at 5 cm/s", not
  mechanistic.
* **In the two cells with v_x = 5 cm/s the benefit of pre-positioning is lost at
  5 cm/s** (design result of §7.1): G_5(50 cm) = +0.62 ±0.73 and +0.47 ±0.55 s/part,
  below 1 s/part on the point and with lower bounds −0.11 and −0.08, against +1.74
  ±0.87 and +1.58 ±0.89 at 3 cm/s. The question "10 or 16.7 cm" is moot there: there
  is no benefit worth sizing the camera for, at any length. This is **not H_neither**
  (C_5(16.7 cm) = +0.12 ±0.39 and +0.12 ±0.45 do not exceed δ): the shorter views are
  not costly, the full view is not valuable. The §7.1 predictions under H_s for these
  cells (+0.68 and +0.54) assumed the 3 cm/s benefit level persists, which it does not;
  the first interval (upper bound 0.51) excludes its H_s prediction, the second (+0.48
  ±0.55) contains both predictions, and neither reading is evidence about the scale
  because the premise fails.
* **The scale question (centimetres or seconds) is undetermined**, for a reason
  different from the one §7.1 (ii) anticipated: the discriminating cells lost the
  benefit rather than sitting in the margin-noise overlap, and the non-discriminating
  cells answered H_cm as foreseen under either hypothesis.
* **Premise check.** §7.1 took all four cells as useful at 3 cm/s from the 35 cm bench
  of §6. With the 50 cm entry, three are; spacing 12 / v_y 10 / v_x 10 sits on the
  threshold (point 0.99 against the criterion's 1.0, lower bound 0.61 against 0.5). It
  is reported as it falls; the criterion is not relaxed.

### 8.3 The level of the benefit across speeds and the entry length

G_3(50 cm) − G_5(50 cm), paired per realization: +1.12 ±1.21 (12 / 10 / v_x 5, not
detected), **−0.77 ±0.60** (12 / 10 / v_x 10: more benefit at the faster belt),
**+1.11 ±1.02** (6 / 15 / v_x 5: less) and −0.06 ±0.53 (6 / 15 / v_x 10). The `none`
levels net of the free transit are the same at both speeds within 1.7 s/part (32.7 /
32.3, 20.9 / 22.6, 26.5 / 26.3 and 15.0 / 15.5), so the change is in the gain of
pre-positioning, not in the baseline. Against the 35 cm bench of §6 (3 cm/s, nominal,
Poisson, G at 10 s = 30 cm, different seeds, unpaired): 1.65 ±0.76 vs 2.05 ±0.73,
1.06 ±0.36 vs 1.32 ±0.38, 1.87 ±0.88 vs 2.44 ±0.78 and 1.15 ±0.35 vs 1.03 ±0.26
s/part. No difference is detected (the unpaired half-widths are 0.4–1.2), three of the
four point estimates are lower with the longer entry, and no equivalence margin was
fixed; this is the H3 check of §6.6 at 50 cm and it stays a check, not a result.

Knees at 5 cm/s (secondary, §7.4): 10 cm point / 25 cm conservative (2.0 / 5.0 s) in
12 / 10 / v_x 10, 25 cm point / 50 cm conservative (5 / 10 s) in 6 / 15 / v_x 10,
undetermined conservative in both cells that lost the benefit. The knee (90 % of
G_5(50 cm)) and the margin (δ = 0.5 s/part absolute) are different criteria: with
G_5(50 cm) = 1.22 s/part the missing 10 % share is 0.12 s/part, finer than δ, which is
why 10 cm "keep the benefit" in 6 / 15 / v_x 10 while the knee lies at 25 cm.

Alignment reading (frozen in §7.4, descriptive, no test; mean absolute difference of
the retained shares, distance scale over 11 lengths / time scale over 5 pairs): 30 / 24,
17 / 30, 59 / 52 and 23 / 13 points. The time scale is the better-aligned one in three
cells and the distance scale in 12 / 10 / v_x 10; in the only cell useful at both
speeds (6 / 15 / v_x 10) the time scale is better (13 against 23). Post hoc caveat,
written after the run: the five time-scale pairs sit at ≥ 1.5 s where both curves are
flatter, which favours that scale by construction; restricted to the same five lengths
(7.5–50 cm) the distance reading is 32, 16, 58 and 15 points, no better separated than
the time reading in any cell. The reading does not resolve the scale question.

### 8.4 Mechanism (traces, reported apart from the verdict)

From 3 to 5 cm/s, in the four cells in the order of the table: pre-positioning moves
per episode 8.2 → 6.4, 14.3 → 13.5, 11.6 → 8.5 and 23.5 → 20.4; idle share of the
head under `none` 13 → 9 %, 18 → 14 %, 16 → 11 % and 27 → 20 %; mean planned t_x of
the moves 1.78 → 2.49, 0.72 → 1.03, 1.48 → 2.32 and 0.48 → 0.80 s with t_y unchanged
(0.94 / 0.95, 0.93 / 0.94, 0.32 / 0.33, 0.35 / 0.34 s); mean planned max(t_x, t_y)
2.14 → 2.77, 1.30 → 1.52, 1.62 → 2.43 and 0.68 → 0.97 s; share of y-dominated moves
31 → 23 %, 47 → 39 %, 30 → 22 % and 44 → 36 %. The traces are compatible with the
interception moving downstream at the faster belt (the part travels further during the
move, so the x component grows and dominates more often) and with fewer idle windows in
which to pre-position. In the two cells with v_x = 5 cm/s the x component of the moves
(2.3–2.5 s) exceeds the transverse crossing time and approaches the time the belt
needs to carry a part across the same stretch, which is where the benefit was lost; in
the v_x = 10 cells the moves stay short (≤ 1.5 s) and the benefit stays. This is a
reading of the traces, not a tested hypothesis; the bench was not designed to separate
these two mechanisms.

### 8.5 Answer

At 5 cm/s, **10 cm of visible belt keep the benefit of pre-positioning wherever the
benefit survives the speed change**, which in this bench is the two cells with the x
axis at 10 cm/s (upper bounds of the cost 0.32 and 0.23 s/part against δ = 0.5); 16.7
cm add nothing detectable there. With the x axis at 5 cm/s the benefit itself falls
below the criterion at 5 cm/s (point 0.62 and 0.47 s/part), so no visible length is
worth sizing for in those cells. Whether the specification that transfers across belt
speeds is "centimetres" or "seconds" remains undetermined: the cells that could have
separated the two readings are the ones that lost the benefit. For Paper 2 this
replaces the untested statement of §6.7 by: the 10 cm camera keeps its benefit at
5 cm/s with the fast x axis; with the slow x axis the benefit, not the camera, is what
the faster belt removes; and the scale question is open. The reserved seeds 10000–10059
are used.
