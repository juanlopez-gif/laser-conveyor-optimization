# Working prompt: a strong learned dispatcher (RL v1.0)

Written 2026-11-19, to be executed after the v0.6 chain (DAgger agents, hybrids,
sensitivity sweep) has finished and been documented. Paste the section **Prompt**
into a new session, or follow it step by step. Everything below the prompt is the
technical specification it refers to.

---

## Prompt

> You are continuing the project in `research_2627/laser-conveyor-optimization`
> (XY laser serving N conveyors; read `README.md`, `docs/formal_formulation.md`,
> `docs/methods/rl.md`, `docs/methods/hybrid.md` if present, `docs/CHANGELOG.md` and
> this file first). Goal: build a **strong** learned dispatcher that (a) alone, at
> millisecond latency, matches or beats the best current online policy
> (`rollout` + pre-positioning `when_idle`, −10.1 % penalized flow time vs
> `nearest_job`/`none` at default load) and (b) combined with a short search, beats it
> significantly, while generalizing across cell parameters. Follow the phases of
> `docs/plans/strong_rl_plan.md` in order; do not start a phase before the acceptance
> test of the previous one passes. Rules: all repository content in English, replies
> to the user in Spanish; log every change and its rationale in `docs/CHANGELOG.md`;
> every new module gets tests; never change the default cell (laser 10 cm/s,
> processing 2.5 s, belts 3 cm/s, arrivals 10/12/14 s, no acceleration — user decision
> 2026-11-19); evaluation seeds 1000–1029 and validation seeds 2000+ are never used for
> training; no policy may see future arrivals; commit locally per phase and **push only
> when the user says "Yes, push"**. Report progress to the user at the end of every
> phase with the acceptance numbers.

---

## 0. Status and adjustments (2026-11-20, start of execution)

Facts learned after this plan was written:

* **Pre-positioning.** The real cell **cannot pre-position** (no detection of parts
  before the laser zone), so `none` is its mode (`docs/assumptions.md` C4). The default
  cell values are the real ones (§F).
* **References for the success targets.** At the real load, on fresh seeds and with
  `none`, they are `rollout@avg_flow_time` and `rl_rollout`
  (`docs/methods/confirmatory.md`). This replaces the `rollout`/`when_idle` −10.1 %
  of §5, which mixed in the pre-positioning effect. Training keeps randomizing the
  mode; evaluation reports `none` first.
* **Seeds.**
  * Already seen: 1000–1029 (exploratory) and 3000–3029 (confirmatory). Used by
    phase 1: 5000+ (simulator validation) and 6000+ (benchmark).
  * The final phase-4 evaluation will use **fresh seeds 7000–7029**.
  * Training uses 100000+ and validation 2000+, as before.
* **Hardware.** AMD Ryzen 7 5700U: 8 cores / 16 threads, 15 GB RAM, no CUDA GPU.
  * Model size **S** (CPU) is the main track.
  * M is possible on CPU only with small batches and long runs.
  * L is out of reach locally.
  * Up to 8 worker processes (0.6 used 2).

| phase | status |
|---|---|
| 1 — fast simulator | **done**, with a revised acceptance (`docs/methods/fastsim.md` §4): exact event-driven simulation agrees with the engine better than the engine agrees with itself across `dt`; ≥ 99 % identical order vs `dt = 0.01` is unreachable by construction. Speed 20.6× (real load) / 16.8× (heavy) vs `dt = 0.05`, plus 4× from cores. Found and fixed: round-off was deciding exact ties in the engine. |
| 2 — representation and model | **done** (0.7.1, `docs/methods/rl2.md`): token features, Transformer pointer over any candidate, value head; permutation invariance and masking pass; S = 0.88 M parameters, 4.1 / 4.7 ms median per decision (default / heavy, 1 thread), p95 5.3–5.8 ms. The network is now the bottleneck (340 decisions/s when it drives an episode), so phase 3 search must batch network calls across rollouts. |
| 3 — expert iteration | **in progress** (0.7.3: tokens and teacher restricted to observable parts, `simulation.observation: belts` for the real cell; training restarted. 0.7.2: `docs/methods/rl2.md` §6–§8). Retargeted by the user to the **general problem** (N belts, per-belt speeds and arrival rates, processing time, laser vx/vy, geometry, penalties; ranges in `rl2.expert_iteration.GENERAL`). Search teacher with lock-step batched look-aheads; DAgger-style expert iteration with gating on real, heavy and general validation cells. Training made portable (GPU, checkpoint/resume, cluster scripts: `docs/methods/remote_training.md`). The engine at `dt = 0.01` has ~1 % error on general cells, so phase 4 must use `dt ≤ 0.001` or the exact simulator there. |
| 4 — evaluation | pending |

## 1. Why the current RL is weak (diagnosis)

| limitation | effect | fix in this plan |
|---|---|---|
| it imitates a fixed teacher (rollout over `nearest_job`) | can never exceed the teacher | self-improving search teacher (expert iteration), §4 |
| MLP on 87 hand-made features, 3 slots per belt | cannot see the whole queue, tied to N = 3 | attention model over all parts, §3 |
| action = conveyor head | restricted class (up to 3.5 % lost with bursts) | pointer over any candidate part, §3 |
| PPO from scratch / fine-tune | degraded the cloned policy | search-generated targets + value regression; PPO optional |
| one parameter set per model | must retrain if belts/job change | domain randomization + parameter context tokens, §3–4 |
| slow fixed-step simulator (~10³ decisions/min on 2 cores) | too little data | fast event-driven simulator, §2 |

## 2. Phase 1 — fast simulator (`src/fastsim/`)

**Goal:** ≥ 20× more decisions per second than the fixed-step engine at `dt = 0.05`,
same model, cheap cloning for search.

* Event-driven core with analytic crossing times: next event = min(arrival, part reaches
  a, part reaches b, laser reaches meeting point, processing ends, revocation check).
  Between events every belt moves linearly (odometers), so interception is solved in
  closed form (`motion.find_intercept` already does it per part).
* Decision-level API: `reset(cfg, seed)`, `candidates()`, `apply(part_id)`,
  `run_until_decision()`, `clone()` (flat state arrays, no `deepcopy`), `cost_so_far()`
  (WIP integral, N_B, T_B), `done`.
* Pre-positioning modes `none` / `when_idle` / `always`, both stop rules, `block` mode.
* Optional speed-up: NumPy structure-of-arrays state; Numba only if profiling demands it.

**Acceptance test (tests/test_fastsim.py):**
1. On 200 seeds × {default, heavy} × 3 modes with `nearest_job` and `nearest_head`:
   identical processing order to the fixed-step engine at `dt = 0.01` in ≥ 99 % of the
   runs, start times within 2·dt, KPIs within 0.5 %. Every schedule passes
   `optimization.schedule_check.check_schedule` / the formal checker.
2. Speed: ≥ 20× decisions/s vs `Simulation` at `dt = 0.05` (benchmark script).
3. `clone()` is independent (mutating the clone never changes the original).

## 3. Phase 2 — representation and model (`src/rl2/model.py`, `features.py`)

* **Tokens** (variable count):
  * one per known unfinished part (buffer, entry zone, laser zone): conveyor one-hot /
    embedding, x/L, in-zone flag, time to b, interception time from the laser,
    waiting time, processing time, blocking flag, rank on its belt, candidate flag;
  * one per conveyor: speed, running, blocked, buffer length;
  * one laser token: x, y, busy, committed target;
  * one **context** token: belt speeds, laser speeds, processing-time mean, arrival
    rates, pre-positioning mode, λ_N, λ_T.
* **Network:** Transformer encoder (3–4 layers, d = 128, 4 heads, no positional
  encoding — the set is unordered), then
  * policy = pointer logits over candidate part tokens (non-candidates masked) — **any
    candidate, not only belt heads**;
  * value = MLP on the pooled embedding → predicted cost-to-go
    (Σ F remaining + Π remaining), normalized by a running scale.
* **Size: a scaling ladder, chosen by experiment, not fixed in advance** (user request
  2026-11-19: "make it truly powerful").

  | size | layers × width × heads | parameters (approx.) | hardware |
  |---|---|---|---|
  | S | 4 × 128 × 4 | ~1 M | CPU |
  | M | 8 × 256 × 8 | ~6 M | GPU recommended |
  | L | 12 × 512 × 8 | ~40 M | GPU required |

  All three share the code (size = config). Train S first to validate the pipeline,
  then M and L **with proportionally more search data** (a larger model trained on the
  same data only overfits). Pick the size by the validation curve (score vs parameters
  and vs data); report the scaling curve. The bigger model is kept only if it wins on
  validation. Latency budget: the network must decide in < 5 ms alone (a 40 M model on a
  GPU, batched, does); the physical slack before a block is seconds.
* Larger capacity pays off only together with a stronger teacher (search depth, more
  scenarios) and more data; the plan therefore scales the three together.

**Acceptance test:** permutation invariance (shuffling tokens does not change the
output), masking (never selects a non-candidate), latency benchmark.

## 4. Phase 3 — expert iteration (`src/rl2/search.py`, `expert_iteration.py`)

Approximate policy iteration with a search teacher that improves with the network:

1. **Iteration 0 (warm start):** behaviour cloning on the existing teacher
   (`rollout`, `when_idle`) with the new model; value head on observed cost-to-go
   (γ = 1, finite drained episodes).
2. **Search at a decision** (teacher for iteration k):
   * candidates = top-K by policy prior (K = 4–6, always including the prior's argmax);
   * for each candidate: clone, apply it, run the **current network** greedily for a
     horizon H (e.g. 30–60 s simulated or until the known parts are done), add the
     **value estimate** of the state at the horizon; average over M sampled
     future-arrival scenarios drawn from the *known rates* (never the real future);
   * target = argmin (hard) and softmin over candidates with temperature T (soft).
   * Later option: Gumbel/sampled MCTS (Danihelka et al. 2022) if truncated rollout
     saturates.
3. **Data generation:** parallel workers play episodes where the search acts
   (ε-mixed with the network to cover its own state distribution), storing
   (tokens, mask, search policy, realized cost-to-go).
4. **Training:** cross-entropy to the search policy + MSE value loss; replay buffer of
   the last few iterations.
5. **Gating:** a new network replaces the old one only if it is better on the
   validation set (seeds 2000+, paired ratio vs `nearest_head`, as in `rl.train`).
6. **Domain randomization** per episode: belt speed U[1.5, 6] cm/s, processing time
   U[1.5, 4] s, arrival factor U[0.6, 1.2], laser 10 cm/s (fixed by decision), modes
   `none`/`when_idle`, loads default/heavy. The context token carries the values.
7. Iterate until the validation score stops improving for 2 iterations.

**Acceptance per iteration:** validation improves or the iteration is rejected; the
search teacher itself must beat the previous network (otherwise H, K or M are wrong).

## 5. Phase 4 — evaluation (`experiments/rl2_eval.py`, `docs/methods/rl2.md`)

* Protocol unchanged: 30 seeds 1000–1029, `dt = 0.01` fixed-step simulator for the
  final numbers (the fast simulator is for training only), both loads, modes
  `none` and `when_idle`, paired Δ with 95 % CI vs `nearest_job`/`none`, plus
  `rollout`/`when_idle` and the v0.6 agents and hybrids as references.
* **Hindsight reference:** for every evaluation seed solve the realized instance
  offline (ILS, 120 s, warm-started by the best policy; CP-SAT bound where feasible):
  the gap online-vs-hindsight shows how much is still attainable.
* **Generalization:** the sensitivity grid of v0.6 (belt 1.5–6, processing 1.5–4,
  arrivals ×0.6–1.2, laser 5–16.7) with one single model.
* **Latency:** network alone; network + search with budgets 10 / 50 / 200 ms per
  decision; report decision-time tails against the physical slack (time until the
  next block).
* **Ablations:** head-only vs any-part action; no value head (pure rollout to the end);
  no domain randomization; K, H, M.

**Success criteria**

| deliverable | target |
|---|---|
| network alone | ≥ `rollout`/`when_idle` (−10 % vs NJ at default load), < 5 ms per decision |
| network + search (≤ 200 ms) | better than `rollout`/`when_idle` with CI excluding 0 |
| generalization | never worse than `nearest_head`/`when_idle` on any cell of the grid |
| hindsight gap | reported for every load |

## 6. Compute and engineering notes

* Current machine: 2 CPU cores. The bottleneck is simulation, not the network: a GPU
  is not needed for a < 1 M-parameter model. Budget per expert-iteration round on
  2 cores after Phase 1: ~30–60 min; 5–10 rounds overnight. More cores scale almost
  linearly (one worker per core).
* Reuse: `rl.env` (SMDP and reward definition), `rl.train` (validation, gating,
  checkpoint selection), `policies.rollout` (fairness rules), the study harness
  (`experiments/blocking_prepositioning_study.py`, `name@k=v@k2=v2` specs).
* New policy `rl2` (parametric) registered in `src/policies/`, usable alone or with
  search (`rl2@search_ms=50`).
* Determinism: all randomness from seeds; record versions, seeds and wall-clock
  budgets in `models/<tag>_training.json`.

## 7. Risks

| risk | mitigation |
|---|---|
| fast simulator disagrees with the reference engine | Phase 1 acceptance test; final numbers always on the reference engine |
| search teacher not better than the network | tune H, K, M; fall back to deeper look-ahead; check value calibration |
| value head mis-scaled across randomized parameters | normalize targets per context; report calibration plots |
| overfitting to the synthetic arrival process | domain randomization; re-validate when real arrival data arrive |
| results depend on provisional parameters | keep the sensitivity grid in every evaluation; recalibrate with real data |
