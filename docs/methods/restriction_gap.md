# Restriction gap of belt-order (head-only) policies (version 0.5.2)

**Question** (literature review, point A, [../literature/review_assessment.md](../literature/review_assessment.md)):
`nearest_head` and the RL agent only serve the *head* of a conveyor, i.e. every conveyor
is processed in belt order. Is belt order dominant, and if not, how much does the
restriction cost?

## Method

`experiments/restriction_gap.py`. For small instances, every processing order $\pi$ is
replayed with the decoder $\Phi$ (`fixed_sequence`, pre-positioning `none`):

* **all**: best over all $n!$ permutations;
* **belt**: best over the permutations that keep arrival order inside every conveyor
  (exactly the orders a head-only policy can realise).

Screening at `dt = 0.05`; the best order of each class is re-simulated at `dt = 0.01`
and checked against the formal model. Gap $=(\text{belt}-\text{all})/\text{all}$.
Instances: first $n$ arrivals of the stochastic process with seeds 1–5 ($n=6$) and
1–3 ($n=7$), for two arrival regimes:

* **default**: the default arrival process (mean inter-arrival 10/12/14 s);
* **burst**: mean inter-arrival 3 s on every conveyor, so that the parts compete for the
  laser (with few parts and the default rates there is hardly any contention).

```bash
python experiments/restriction_gap.py --workers 2      # ~20 min on 2 cores
```

## Results

| arrivals | n | objective | instances | mean gap [%] | max gap [%] | gap > 0.1 % | optimum keeps belt order |
|---|---|---|---|---|---|---|---|
| default | 6 | penalized flow | 5 | 0.00 | 0.00 | 0 | 5/5 |
| default | 6 | makespan | 5 | 0.00 | 0.00 | 0 | 5/5 |
| default | 7 | penalized flow | 3 | 0.00 | 0.00 | 0 | 3/3 |
| default | 7 | makespan | 3 | 0.00 | 0.00 | 0 | 3/3 |
| burst | 6 | penalized flow | 5 | 0.66 | 3.28 | 1 | 4/5 |
| burst | 6 | makespan | 5 | 0.28 | 1.61 | 1 | 3/5 |
| burst | 7 | penalized flow | 3 | 1.15 | 3.46 | 1 | 2/3 |
| burst | 7 | mean flow | 3 | 1.45 | 3.46 | 2 | 1/3 |
| burst | 7 | makespan | 3 | 1.08 | 2.43 | 2 | 1/3 |

(The rows for mean flow at default load and for burst $n=6$ coincide with penalized flow.)

Further observations:

* At default load many non-belt orders are **infeasible** for the decoder (e.g. 4 240 of
  5 040 permutations for $n=7$, seed 1): the sequence waits for a part that can never
  reach the laser zone because its belt is blocked by a part placed later in the
  sequence (`InfeasibleSequence`). Under bursts all permutations were feasible except
  in one instance.
* Smallest counterexample (fixed as a test, `tests/test_restriction_gap.py`): burst,
  $n=5$, seed 1. The best order `P001 P003 P005 P004 P002` serves a later part of a
  conveyor before an earlier one and beats every belt-order permutation by > 1 %
  (2.2 % in mean flow time).

## Conclusions

1. **Belt order is not dominant.** There are instances where the optimum (over $\Phi$)
   processes a conveyor out of order. The head-only policies (`nearest_head`, the RL
   action space) are therefore a **restricted policy class**, as the review pointed out;
   the formulation (§5, property 2) and the PDF now say so.
2. **At the default load the restriction cost nothing** on these instances (8/8 optimal
   orders keep belt order). It costs up to ≈ 3.5 % (0.3–1.5 % on average) when arrivals
   come in bursts.
3. Hence the restriction is a good engineering choice for the current cell
   parameters, but an RL agent or a rule for bursty arrivals should be allowed to
   serve non-head parts (e.g. action = conveyor + slot), at the price of a larger
   action space.

## Limitations

Small $n$ (6–7) and few instances; optimality is over the decoder $\Phi$ (non-delay
schedules, no deliberate waiting, no pre-positioning), not over all schedules of
$\mathcal P$; the "burst" regime is synthetic.
