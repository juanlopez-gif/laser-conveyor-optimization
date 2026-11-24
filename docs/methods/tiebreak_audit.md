# Tie-break audit: observable key (0.7.26) against the legacy arrival-time key (rerun 0.7.33)

Rerun of 0.7.33: the known-parts search applies the key under audit inside every look-ahead continuation as well as when ordering its candidates (the 0.7.26 run changed only the ordering).

Seeds 2000–2029 (development), treatments B and C, eight regimes. ``changed`` = decisions (≥ 2 candidates) of the new episode at which the legacy key would have chosen another part; Δ = paired cost ratio of the new policy vs the legacy one [%], 95 % CI, negative = the observable key is cheaper.

| regime | t | policy | decisions | changed | Δ new vs legacy [%] |
|---|---|---|---|---|---|
| real / regular / nominal | B | nearest_job | 1556 | 15 (1.0 %) | -0.60 ±0.73 |
| real / regular / nominal | B | nearest_head | 1567 | 9 (0.6 %) | -0.07 ±0.21 |
| real / regular / nominal | B | search | 1534 | 7 (0.5 %) | -0.08 ±0.29 |
| real / regular / nominal | C | nearest_job | 1565 | 12 (0.8 %) | -0.02 ±0.32 |
| real / regular / nominal | C | nearest_head | 1556 | 6 (0.4 %) | +0.10 ±0.36 |
| real / regular / nominal | C | search | 1532 | 1 (0.1 %) | +0.00 ±0.01 |
| real / regular / slow | B | nearest_job | 2031 | 27 (1.3 %) | +0.61 ±0.64 |
| real / regular / slow | B | nearest_head | 2037 | 23 (1.1 %) | +0.64 ±0.48 |
| real / regular / slow | B | search | 2004 | 9 (0.4 %) | +0.14 ±0.51 |
| real / regular / slow | C | nearest_job | 2014 | 22 (1.1 %) | +0.27 ±0.91 |
| real / regular / slow | C | nearest_head | 2034 | 28 (1.4 %) | -0.20 ±1.02 |
| real / regular / slow | C | search | 1992 | 16 (0.8 %) | +0.44 ±0.68 |
| real / burst / nominal | B | nearest_job | 1474 | 6 (0.4 %) | -0.30 ±0.35 |
| real / burst / nominal | B | nearest_head | 1488 | 6 (0.4 %) | -0.03 ±0.09 |
| real / burst / nominal | B | search | 1431 | 9 (0.6 %) | -0.11 ±0.19 |
| real / burst / nominal | C | nearest_job | 1795 | 7 (0.4 %) | -0.48 ±0.54 |
| real / burst / nominal | C | nearest_head | 1799 | 5 (0.3 %) | -0.08 ±0.21 |
| real / burst / nominal | C | search | 1801 | 23 (1.3 %) | -0.04 ±0.48 |
| real / burst / slow | B | nearest_job | 1654 | 8 (0.5 %) | +0.11 ±0.51 |
| real / burst / slow | B | nearest_head | 1676 | 12 (0.7 %) | +0.12 ±0.22 |
| real / burst / slow | B | search | 1654 | 7 (0.4 %) | -0.13 ±0.18 |
| real / burst / slow | C | nearest_job | 1864 | 9 (0.5 %) | +0.17 ±0.22 |
| real / burst / slow | C | nearest_head | 1882 | 12 (0.6 %) | +0.06 ±0.35 |
| real / burst / slow | C | search | 1862 | 5 (0.3 %) | +0.00 ±0.07 |
| heavy / regular / nominal | B | nearest_job | 2631 | 46 (1.7 %) | +0.92 ±0.61 |
| heavy / regular / nominal | B | nearest_head | 2643 | 63 (2.4 %) | +0.57 ±0.55 |
| heavy / regular / nominal | B | search | 2560 | 30 (1.2 %) | +0.41 ±0.64 |
| heavy / regular / nominal | C | nearest_job | 2631 | 58 (2.2 %) | +1.41 ±1.01 |
| heavy / regular / nominal | C | nearest_head | 2629 | 65 (2.5 %) | +0.44 ±0.58 |
| heavy / regular / nominal | C | search | 2593 | 33 (1.3 %) | +0.47 ±0.96 |
| heavy / regular / slow | B | nearest_job | 2644 | 29 (1.1 %) | +1.27 ±1.01 |
| heavy / regular / slow | B | nearest_head | 2675 | 40 (1.5 %) | +1.51 ±0.72 |
| heavy / regular / slow | B | search | 2615 | 13 (0.5 %) | -0.15 ±0.26 |
| heavy / regular / slow | C | nearest_job | 2641 | 21 (0.8 %) | +1.05 ±1.13 |
| heavy / regular / slow | C | nearest_head | 2665 | 35 (1.3 %) | +0.82 ±0.71 |
| heavy / regular / slow | C | search | 2633 | 21 (0.8 %) | +0.76 ±0.93 |
| heavy / burst / nominal | B | nearest_job | 2345 | 14 (0.6 %) | -0.25 ±0.57 |
| heavy / burst / nominal | B | nearest_head | 2353 | 22 (0.9 %) | +0.10 ±0.20 |
| heavy / burst / nominal | B | search | 2314 | 25 (1.1 %) | -0.03 ±0.19 |
| heavy / burst / nominal | C | nearest_job | 2465 | 10 (0.4 %) | +0.20 ±0.22 |
| heavy / burst / nominal | C | nearest_head | 2487 | 23 (0.9 %) | +0.15 ±0.26 |
| heavy / burst / nominal | C | search | 2485 | 23 (0.9 %) | +0.01 ±0.19 |
| heavy / burst / slow | B | nearest_job | 2473 | 6 (0.2 %) | +0.18 ±0.36 |
| heavy / burst / slow | B | nearest_head | 2491 | 16 (0.6 %) | -0.00 ±0.48 |
| heavy / burst / slow | B | search | 2478 | 13 (0.5 %) | +0.01 ±0.66 |
| heavy / burst / slow | C | nearest_job | 2502 | 7 (0.3 %) | -0.27 ±0.38 |
| heavy / burst / slow | C | nearest_head | 2533 | 13 (0.5 %) | +0.15 ±0.56 |
| heavy / burst / slow | C | search | 2526 | 7 (0.3 %) | -0.20 ±0.35 |

Pooled over the eight regimes: one mean contrast per seed over its eight regimes, interval over the 30 seed means (the eight episodes of one seed share the realization):

| t | policy | Δ new vs legacy [%] (per-seed means) | max |Δ| over regimes [%] |
|---|---|---|---|
| B | nearest_job | +0.24 ±0.20 | 1.27 |
| B | nearest_head | +0.35 ±0.16 | 1.51 |
| B | search | +0.01 ±0.13 | 0.41 |
| C | nearest_job | +0.29 ±0.23 | 1.41 |
| C | nearest_head | +0.18 ±0.20 | 0.82 |
| C | search | +0.18 ±0.19 | 0.76 |
