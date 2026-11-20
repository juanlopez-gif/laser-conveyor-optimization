# Strong RL: representation and model (version 0.7.1, strong-RL plan phase 2)

Code: `src/rl2/features.py` (tokens), `src/rl2/model.py` (network, batching, policy,
save/load). Tests: `tests/test_rl2_model.py`. Latency: `experiments/rl2_latency.py`.
Plan: [../plans/strong_rl_plan.md](../plans/strong_rl_plan.md) §3. Changes and
rationale: [../CHANGELOG.md](../CHANGELOG.md) (0.7.1).

Phase 2 builds the model only. It is **untrained**: training is phase 3 (expert
iteration), and results come in phase 4.

## 1. Why a new representation

The 0.5–0.6 agent sees 87 hand-made features in 3 slots per belt. Its action is
"serve the head of conveyor k". That design has three limits (plan §1):

* it cannot see the whole queue;
* it is tied to N = 3 conveyors;
* it can only pick belt heads. With bursts, this restriction costs up to 3.5 %.

The new model reads the state as an **unordered set of tokens**, one per part, belt,
laser and context. Its action is a pointer to **any candidate part**.

## 2. Tokens (`rl2.features.encode`)

Scales:

* lengths are divided by the belt length L;
* speeds are divided by 3 cm/s, the real belt speed;
* times go through `slog(t) = sign(t)·log(1 + |t|/10 s)`, which keeps long waits at
  heavy load bounded but still distinguishable.

No token holds a list position or a conveyor index. A part is linked to its belt
through the y offset from the laser and a copy of its belt's state.

| token | count | features |
|---|---|---|
| part | one per arrived unfinished part (buffer, belt, being processed) | 23 features: status one-hot (waiting, moving, ready, processing); x/L; distance to b; time to b at belt speed; interception time from the laser and a reachable flag; waiting time; p_j / mean p; remaining processing time; blocking flag; rank on the belt among unstarted parts and a head flag; **candidate flag**; committed flag; x and y offset from the laser; nominal and current belt speed; belt running; belt blocked |
| conveyor | N | 11 features: y offset from the laser; nominal and current speed; running; stopped for processing; blocked; buffer length; parts on the belt; arrival rate; offered load; has a candidate |
| laser | 1 | 9 features: x, y; state one-hot (idle, heading, busy); remaining processing time; committed flag and target offset |
| context | 1 | 17 features: laser speeds; motion model; mean processing time; total offered load; a, b, spacing, workspace height (relative to L); pre-positioning mode one-hot; λ_N, λ_T; arrivals still coming; number of parts and of candidates |

* The context token carries the cell parameters. With it, one model can be trained
  with domain randomization (phase 3) and used on other cells.
* Encoding takes 0.14 ms per decision at default load and 0.19 ms at heavy load.

## 3. Network (`rl2.model.DispatchNet`)

* **Embedding:** one linear embedding per token type to width d, plus a learned type
  vector.
* **Encoder:** Transformer, pre-norm, ReLU feed-forward (4·d), **no positional
  encoding**. Padding is masked.
  * Self-attention without positions is permutation-equivariant.
  * g is the masked mean over all tokens.
* **Policy:** logit_i = MLP([h_i, g]) for each part token. Non-candidates and padding
  get −∞, so the action is **any candidate** of the open decision.
* **Value:** MLP([g, h_ctx, h_laser]) predicts the normalized cost-to-go.
  `RunningScale` keeps the normalization; phase 3 trains this head.
* **`NetPolicy`:** a greedy or sampled dispatcher for `FastSim.run`. When a decision
  has a single candidate, it takes it without calling the network.
* `save` / `load` store the size config together with the weights.

| size | layers × d × heads | parameters |
|---|---|---|
| **S** | 4 × 128 × 4 | **0.88 M** |
| M | 8 × 256 × 8 | 6.66 M |
| L | 12 × 512 × 8 | 39.2 M |

## 4. Acceptance (plan §3)

**Tests** (`tests/test_rl2_model.py`, 10 tests, all passing):

| test | what it checks |
|---|---|
| permutation invariance | Part and conveyor tokens of 30 real decision states are shuffled. Logits permute identically and the value is unchanged (|diff| < 1e-4). |
| masking | 5 random networks on 100+ states: non-candidates and padding get probability exactly 0 and are never the argmax or a sample. |
| batching | A padded batch gives the same output per state as single calls. |
| features | Shapes are right, values are finite and bounded, and the candidate flags equal `sim.candidates()`. |
| any candidate | Non-head candidates appear among the states, so the action space is wider than heads only. |
| valid episode | An untrained network drives a full FastSim episode (default/`none` and heavy/`when_idle`), and the formal checker finds 0 violations. |
| size and save/load | S has ~1 M parameters; weights survive a round trip. |
| latency | S forward pass < 25 ms (a loose bound for a loaded machine; the real numbers are below). |

**Latency** (`experiments/rl2_latency.py`; untrained weights; real decision states
from seeds 6100/6101):

| size | single state, 1 thread: median / p95 [ms], default load | heavy load | batch of 64, 8 threads [states/s], default / heavy |
|---|---|---|---|
| **S** | **4.1 / 5.3** | **4.7 / 5.8** | 2 360 / 1 580 |
| M | 15.4 / 17.8 | 18.9 / 22.7 | 440 / 265 |
| L | 64 / 78 | 81 / 103 | 87 / 53 |

* **Size S meets the plan's budget of < 5 ms per decision on the median at both
  loads. The p95 is 5.3–5.8 ms.**
  * The physical slack before a block is seconds, so this is enough for a
    controller.
  * Most of the time is per-layer overhead on very small tensors: about 1.2 ms fixed
    plus 0.85 ms per layer. It is not the arithmetic.
  * Switching from GELU to ReLU and from `no_grad` to `inference_mode` saved about
    10 %.
* **M and L are 4× and 16× slower** on this CPU. They are only viable as offline
  teachers or with a GPU, which confirms the plan's §0 choice of S.

## 5. Consequence for phase 3

A full FastSim episode driven by S runs at **340 decisions/s** on one thread. The
simulator alone runs at 5 500 decisions/s, so the network, not the simulator, is now
the bottleneck of network-guided search.

* **Per-decision network calls cost about 16× more than one simulation step.** Phase
  3 must therefore **batch network calls**: the search rollouts and scenarios of one
  decision advance in lock-step, and their states are evaluated together.
* A batch of 64 gives 1 600–2 400 states/s, against ~240 for single calls.
* Planned design for `src/rl2/search.py`:
  * K candidates × M scenarios rollouts are advanced in lock-step;
  * one batched forward pass per step;
  * the value head truncates the horizon, which cuts the number of network steps;
  * rollouts may use a cheap rule (`nearest_job`) where the network adds nothing.

## 6. Reproduce

```bash
python -m pytest tests/test_rl2_model.py -q
python experiments/rl2_latency.py --sizes S M L --threads 8
```

The output goes to `results/experiments/rl2_latency/` (`summary.md`, `latency.json`).
