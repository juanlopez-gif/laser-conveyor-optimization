# Event-driven fast simulator (version 0.7.0, strong-RL plan phase 1)

Code: `src/fastsim/` (`core.py`, `policies.py`). Tests: `tests/test_fastsim.py`.
Validation and benchmark: `experiments/fastsim_validation.py`,
`experiments/fastsim_benchmark.py`. Plan: [../plans/strong_rl_plan.md](../plans/strong_rl_plan.md)
§2. Changes and rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.7.0).

## 1. Why

The fixed-step engine advances every `dt` seconds. Learning a strong dispatcher needs
millions of simulated decisions, plus cheap copies of the state for search, and at
`dt = 0.05` the engine gives only about 300 decisions per second per core.

Between two events nothing discrete happens: belts move at constant speed and the
laser moves each axis at constant speed. Every crossing time is therefore known in
closed form, and the simulator can jump from one event to the next.

## 2. Model

The model is identical to the engine (`docs/formal_formulation.md`, `docs/assumptions.md`).

* **Events:**
  * arrival;
  * end of processing;
  * a part entering the laser zone (x = a);
  * an unprocessed part reaching the zone end (x = b, the belt blocks);
  * a part leaving the belt (x = L);
  * the last part on a belt reaching the minimum spacing (the next waiting part enters);
  * the laser reaching its target, and the target part reaching it (contact);
  * the arrival horizon and `max_time`.
* **At each event time** the instantaneous logic runs in the order of one engine step:
  1. finish processing;
  2. arrivals;
  3. statuses and blocking;
  4. admission;
  5. revocation of a `when_idle` pre-positioning;
  6. dispatching;
  7. laser contact and re-planning.
* **Interception** uses the same function as the engine (`simulation.motion.find_intercept`).
  The common case of a stopped part inside the zone is inlined with the identical
  expression.
* **Scope:**
  * `block` mode only (the confirmed rule of the real cell); `pass` raises
    `NotImplementedError`;
  * both conveyor stop policies;
  * all three pre-positioning modes;
  * both travel-time models.
* **API:**
  * `run_until_decision()` (True while a decision is open), `candidates()` (part
    indices), `apply(j)`;
  * `run(policy)` with `policy(sim, candidates)`;
  * `clone()`: flat lists, about 10 µs;
  * `cost_so_far()` returns (WIP integral = Σ F, N_B, T_B);
  * `summary()`, and `result()` for the formal checker.

## 3. Bugs found while building it (both fixed in 0.7.0)

1. **Round-off decided exact ties in the engine.**
   * *Cause.* Many decisions are exact ties in continuous time. Examples: two parts on
     the other belts with the same Y move (12 cm / 10 cm/s = 1.2 s), or several parts
     stopped at b = 45.
   * *What went wrong.* The interception root was found by interpolation and carried
     ~1e-16 s of noise. Positions at b could be 45 − 1e-15. That noise, not the
     documented tie-break (earliest arrival, then id), chose the part.
   * *Fix, in two parts:*
     * `find_intercept` polishes the root to the fixed point t = τ(x(t)), which
       returns the dominant constant travel time bit for bit;
     * `nearest_job`, `nearest_head` and the rollout's candidate ranking compare
       travel times rounded to 1e-9 s (`DecisionContext.travel_key`).
   * *Effect on the engine,* measured on the confirmatory seeds 3000–3029:

     | load / mode | `nearest_job`: same order | `nearest_job`: Δ penalized flow | `nearest_head`: same order | `nearest_head`: Δ |
     |---|---|---|---|---|
     | default / `none` | 4/30 | −2.47 % | 23/30 | −0.57 % |
     | default / `when_idle` | 5/30 | −1.69 % | 20/30 | −0.54 % |
     | heavy / `none` | 18/30 | −0.28 % | 25/30 | −0.12 % |
     | heavy / `when_idle` | 23/30 | −0.23 % | 26/30 | +0.43 % |

     The documented tie-break (earliest arrival first) is clearly better than the noise
     for `nearest_job` at the real load. All earlier results used the old behaviour,
     so the reference `nearest_job` / `none` of every Δ % up to 0.6.2 is about 2.5 %
     weaker than the rule as documented. The confirmatory comparison was rerun with
     the corrected engine ([confirmatory.md](confirmatory.md) §7). At the real load
     with `none`, `rl_rollout` now gains −2.3 ±1.5 % (was −4.4 %).
2. **Blocked time** (found by `test_cost_identities`; it affected only the new
   simulator during development). Processing a part *upstream* of a blocking part
   released the block, so T_B was undercounted. The block is now released only when
   no unprocessed part remains at b, as in the engine.

## 4. Acceptance tests

### 4.1 Agreement with the engine (200 seeds × 2 loads × 3 modes × 2 rules)

The plan asked for the same processing order in ≥ 99 % of runs against the engine at
`dt = 0.01`. That target turned out to be unreachable by construction: **the engine
does not agree with itself** when its step changes.

| comparison | runs | identical order | KPI within 0.5 % | mean abs KPI diff | mean signed KPI diff (95 % CI) |
|---|---|---|---|---|---|
| engine `dt = 0.01` vs engine `dt = 0.001` (baseline) | 240 | 90.8 % | 92.9 % | 0.435 % | – |
| **fastsim vs engine `dt = 0.01`** | 2400 | 84.7 % | 90.7 % | 0.395 % | default −0.09 ±0.08 %, heavy +0.02 ±0.07 % |
| **fastsim vs engine `dt = 0.001`** | 240 | **95.8 %** | **97.9 %** | **0.091 %** | default −0.03 ±0.03 %, heavy −0.14 ±0.17 % |

* **Formal checker:** 0 violations in all 2640 fast schedules, with tolerances
  1e-6 s and 1e-6 cm (the engine needs 3·dt).
* **Agreement grows as the engine's step shrinks.** For the 10 runs that still
  differ at `dt = 0.001`, the engine was rerun at `dt = 0.0002`:
  * 3 of them now agree;
  * the other 7 diverge at the same decision as before.
* **Those 7 are exact coincidences in continuous time:**
  * in 6 of them, 2 to 7 candidates have exactly the same travel time;
  * in the 7th, a part reaches x = a at the very instant of the decision.
  * The engine lags the true trajectory by O(v·dt) (each event is rounded up to its
    grid), so it breaks such coincidences with a systematic bias. The event-driven
    simulator applies the documented rule.
* **Conclusion:**
  * the fast simulator is closer to the continuous-time model than the engine at
    `dt = 0.01` is to itself at `dt = 0.001`;
  * the remaining differences are the engine's discretization;
  * the mean KPI bias is below 0.1 %, which is negligible against policy effects of
    2–5 %.
* Final numbers for reports still come from the engine (plan rule), unless a later
  decision moves evaluation to the exact simulator.

Full tables: `results/experiments/fastsim_val_dt01/summary.md`,
`fastsim_val_dt001/summary.md`, `fastsim_engine_baseline/summary.md`.

### 4.2 Speed (single process, `nearest_job`, 20 episodes per load)

| load | engine `dt = 0.05` | engine `dt = 0.01` | fastsim | vs `dt = 0.05` | vs `dt = 0.01` |
|---|---|---|---|---|---|
| default | 280 decisions/s | 71 decisions/s | **5 770 decisions/s** | **20.6×** | 81× |
| heavy | 307 decisions/s | 71 decisions/s | **5 150 decisions/s** | **16.8×** | 73× |

* **Clone cost:** 10–11 µs.
* **Against the target (≥ 20× vs `dt = 0.05`):** met at the real load; 16.8× at
  heavy load. There most of the time goes into interception for many candidates, which
  any policy needs.
* **More cores.** The machine has 8 cores / 16 threads (Ryzen 7 5700U). Earlier runs
  used only 2 workers, and the validation used 7–8. Data generation therefore gains a
  further ~4×, so the overall gain over the 0.6 pipeline is about 80× at the real
  load.

### 4.3 Other tests (`tests/test_fastsim.py`, 21 tests)

* The formal model holds for 2 loads × 3 modes × 2 rules.
* **Cost identities:** the WIP integral equals Σ F_j, and the blocked integral equals
  Σ (S_j − B_j).
* **Agreement with the engine in its continuous limit:** at `dt = 0.001`, short
  horizon, 3 cases: same order, start times within 0.05 s, KPI within 0.5 %.
* `run()` and the decision API give the same episode, and only candidates can be
  applied.
* **`clone()` is independent:** playing the clone to the end leaves the original
  unchanged, and the original then continues exactly like a fresh run.

### 4.4 General cells (0.7.2)

* **Setup:** 40 general cells (1–6 belts, independent speeds and rates, other laser
  speeds and geometry; `rl2.expert_iteration.general_config`).
* **Result:** 0 checker violations.
* **Agreement with the engine at `dt = 0.01`** is lower than on the real cell:
  * identical order in 67 % of runs;
  * mean |KPI diff| 0.97 %.
* **Cause:** the engine's own lag. Rerunning the worst cases at `dt = 0.001` and
  `0.0002` makes the difference shrink linearly to about 0.01 %.
* **Consequence:** general-cell results need `dt ≤ 0.001` or this simulator. Details in
  [rl2.md](rl2.md) §7.

## 5. Reproduce

```bash
python experiments/fastsim_validation.py --seeds 200 --dt 0.01 --workers 8 --tag fastsim_val_dt01
python experiments/fastsim_validation.py --seeds 20 --dt 0.001 --workers 8 --tag fastsim_val_dt001
python experiments/fastsim_validation.py --seeds 20 --dt 0.01 --workers 8 --engine-baseline \
    --tag fastsim_engine_baseline
python experiments/fastsim_benchmark.py --episodes 20
```
