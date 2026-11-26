# Reproducing the conveyor scheduling audit

This package accompanies repository release 0.7.40, Paper 1 v1.2 and Paper 2 v1.5. It checks how a dispatch tie-break changes the published comparisons and whether the Paper 2 sensor result repeats on fresh simulated arrivals. The archived release key uses the physical release time of a part. The observable key uses the time the part first became visible. Both are applied after the primary dispatch criterion. [Frozen source and archived data](https://github.com/juanlopez-gif/laser-conveyor-optimization/tree/4892d46db10effa185f99245c6519878a705f35d)

## Run the checks

Start PowerShell **in the downloaded folder containing this file**. In the example, the source and new results are siblings of this folder. The clone command is for a fresh copy. If a checkout already exists, set `$repo` to its absolute path and verify the commit. A ZIP extracted from the same revision also works with the Python scripts.

```powershell
$helpers = (Get-Location).Path
$parent = Split-Path $helpers -Parent
$repo = Join-Path $parent 'conveyor-source-4892d46'
$results = Join-Path $parent 'rerun-4892d46'
git clone https://github.com/juanlopez-gif/laser-conveyor-optimization.git $repo
git -C $repo checkout --detach 4892d46db10effa185f99245c6519878a705f35d
git -C $repo rev-parse HEAD
Set-Location -LiteralPath $repo
python -m pip install -r requirements-rl.txt
New-Item -ItemType Directory -Force -Path $results | Out-Null

python -B experiments/recheck_0_7_39.py --study reference |
    Tee-Object -FilePath (Join-Path $results 'p1_nearest_job_reference.txt')
python -B (Join-Path $helpers 'p1_nh_tiebreak.py') --workers 8 --output $results
python -B (Join-Path $helpers 'p1_search_tiebreak.py') --workers 8 --output $results
python -B experiments/recheck_0_7_39.py --study geometry |
    Tee-Object -FilePath (Join-Path $results 'p2_retrospective_geometry.txt')
python -B (Join-Path $helpers 'paper2_prospective.py') --repo $repo --output-dir $results --workers 6
```

Run from the source root because the Paper 1 and retrospective scripts resolve imports and inputs against the working directory. The prospective script accepts absolute or relative paths and sets the workers’ working directory from `--repo`. Use a **fresh results directory outside the source checkout**. The Paper 1 scripts overwrite their named CSV files on repetition. The prospective script refuses to overwrite its three outputs. Worker counts may be reduced, with 1–6 allowed for the prospective run. The checks ran on Windows with Python 3.11.8. The environment included PyYAML 6.0.3, NumPy 2.4.2 and PyTorch 2.10.0. The repository requirements use minimum versions.

## Paper 1 — which comparisons change?

Each comparator is paired with the archived network on the same 30 seeds, 7000–7029, in eight operating regimes under each mode. B disables prepositioning. C allows prepositioning while idle. The table counts regimes in which the network had lower mean relative cost and the individual two-sided normal 95% interval excluded zero. Parentheses give the count after Holm adjustment at 5% across the eight regimes for that comparator, key and mode.

| Comparator and secondary key | B of 8, individual (Holm) | C of 8, individual (Holm) |
|---|---:|---:|
| Nearest job, release | 7 (6) | 7 (7) |
| Nearest job, first observation | 8 (8) | 8 (8) |
| Nearest head, release | 6 (6) | 6 (6) |
| Nearest head, first observation | 6 (6) | 7 (6) |
| Known-parts search, release | 7 (7) | 7 (6) |
| Known-parts search, first observation | 7 (7) | 8 (8) |

The nearest-job script averages `100 × (J_network / J_rule − 1)` per seed, so a negative value favors the network. The nearest-head and search helpers average `100 × (J_comparator / J_network − 1)`, so a positive value favors the network. These percentage magnitudes have different denominators. Known-parts search uses `SearchTeacher(None, k=6)`. Both added helpers replay each legacy seed before evaluating the observable key. Maximum absolute replay discrepancies were 1.42 × 10⁻¹³ s/part for nearest head and 8.53 × 10⁻¹⁴ for search. The network costs come from the published archive. These experiments evaluate a fixed learned policy. Its advantage against each comparator depends on the regime, prepositioning mode and statistical criterion.

**Inputs and outputs.** The repository input is `docs/papers/paper1/data/phase4_search.csv`. Its `net` and `nearest_job` rows give the release-key nearest-job comparison. The repository's `recheck_0_7_39_reference.txt` in that directory gives the observable nearest-job means and interval half-widths. For the two other comparators, the files below preserve the seed identifiers, archived and rerun costs, observable costs, paired percentage contrasts and discrepancies. Each seed file has 480 rows and each summary has 16. The summaries include individual intervals, two-sided normal p-values and within-mode Holm-adjusted p-values for the observable key.

| Comparator | Code | Paired costs | Group statistics |
|---|---|---|---|
| Nearest head | [Run script](p1_nh_tiebreak.py) | [480 seed pairs](p1_nh_paired_seeds.csv) | [16 comparisons](p1_nh_summary.csv) |
| Known-parts search | [Run script](p1_search_tiebreak.py) | [480 seed pairs](p1_search_paired_seeds.csv) | [16 comparisons](p1_search_summary.csv) |

## Paper 2 — do shorter sensor views retain the benefit?

Each cohort screens the same 56 unique geometry and operating combinations at belt speed 3 cm/s. At each view, the gain is the paired difference `G = J_none − J_when_idle`, in seconds per part. At the 10 s view, the screen requires both a mean gain of at least 1.0 and a lower individual 95% bound of at least 0.5. The positive grid is 0.5, 1, 1.5, 2, 2.5, 10/3, 5, 7.5 and 10 s of nominal belt travel. At this speed 10/3 s is 10 cm of visible belt.

“First” is the earliest tested view at which the mean paired contrast `G(view) − 0.9 × G(10 s)` reaches zero or higher. “Sustained” also requires every larger tested view to qualify. The conservative crossing applies the same tests to the lower bound, using the published multiplier 2.773 for the within-cell grid. Counts below record crossings **strictly before 10 s**, among cells passing the screen. Shortening cost is the paired difference `J_when_idle(10/3 s) − J_when_idle(10 s)`. A positive value means the shorter view costs more. This direct economic outcome can stay small even when the 90% retention test has wide uncertainty.

| Key and 60-seed batch | Screened in, of 56 | Point first / sustained before 10 s | Conservative first / sustained before 10 s | Median / largest mean shortening cost, s/part | Individual 95% CI at largest estimate |
|---|---:|---:|---:|---:|---:|
| Release, 9000–9059 | 18 | 18 / 18 | 17 / 17 | 0.050 / 0.443 | [0.113, 0.774] |
| First observation, 9000–9059 | 19 | 17 / 14 | 3 / 3 | 0.037 / 0.443 | [−0.008, 0.894] |
| First observation, 43700–43759 | 14 | 13 / 13 | 1 / 1 | 0.086 / 0.468 | [0.021, 0.914] |

On the historical release key, one cell had no conservative crossing anywhere on the grid. On the same seeds with the observable key, the three early crossings were at 2, 7.5 and 7.5 s, and the other 16 cells first crossed at 10 s. With the fresh seeds, one cell crossed conservatively at 2 s and 13 first crossed at 10 s. The early cell had spacing 6 cm, transverse head speed 15 cm/s, longitudinal head speed 10 cm/s, nominal load and Poisson arrivals. First and sustained **point** crossings differed in 3/18, 7/19 and 2/14 cells, respectively, even where the aggregate counts before 10 s coincide.

**Cohort overlap.** Thirteen cells passed under both keys on the historical seeds, with five release-only and six observable-only. The two observable cohorts shared nine cells, with ten historical-only and five fresh-only. The release cohort and fresh observable cohort shared ten cells. Eight passed all three screens. The fresh seeds generated new arrivals at the same fixed design points. The 56 cases provide no sampling model for a population of factories.

**Direct-cost uncertainty.** The historical release and fresh observable maxima occurred at spacing 18 cm, both head speeds 5 cm/s, nominal load and burst arrivals. The historical observable maximum occurred at spacing 18 cm, transverse speed 10 cm/s, longitudinal speed 5 cm/s, high load and burst arrivals. The intervals in the table are per-cell paired normal intervals (`mean ± 1.96 × SE`). They do not account for choosing the largest estimate or multiple cells. All 14 fresh cell means were below 0.5 s/part, while only 9/14 had an upper individual bound at or below 0.5. A universal cost ceiling or a universal shorter view retaining 90% of the benefit would exceed the evidence. The paper's 5 cm/s, 0.5 s/part noninferiority experiment had a separate scope. Its two passing fast-head cells had longitudinal speed 10 cm/s and met the direct-cost margin at 10 cm. Conservative 90% retention first occurred at 25 cm for spacing 12 cm and transverse speed 10 cm/s, and at 50 cm for spacing 6 cm and transverse speed 15 cm/s.

**Inputs and outputs.** The retrospective script reads `docs/papers/paper2/data/geometry_bench.csv.gz`. Its archived text output is `docs/papers/paper2/data/recheck_0_7_39_geometry.txt`. The prospective script uses that text only to identify the historical observable cohort for overlap. Both modes run on all 60 fresh seeds at 10 s in every cell. The eight lower positive views run only in the 14 cells selected by the prespecified screen, giving `(56 + 8 × 14) × 2 × 60 = 20,160` episode rows.

- [Prospective runner](paper2_prospective.py) fixes both operating modes, observable key, seed range and geometry combinations and calls the archived simulator and crossing functions.
- [Compressed episode CSV](paper2_prospective_episodes.csv.gz) contains the five design fields, nominal coverage in seconds, mode, seed, penalized flow time in s/part and the simulator's mean planned prepositioning time in seconds when available.
- [Cell CSV](paper2_prospective_cells.csv) contains all 56 screen decisions and gains, interval half-widths, prior cohort membership and all crossing and direct-cost statistics for the 14 selected cells. Empty curve fields in screened-out cells indicate unevaluated shorter views.
- [Summary JSON](paper2_prospective_summary.json) records the source commit, reference-script checksum, per-threshold counts, exact overlap identities, maximum-cost cell and interval half-width.

To recover the two historical direct-cost intervals at the cells with the largest estimated means, run the following Python from the checkout root. The prospective JSON gives the third interval as its mean and half-width.

```python
import sys
sys.path.insert(0, "experiments")
import recheck_0_7_39 as ref

for cell, key in [
    ((18, 5, 5, "nominal", "burst"), ref.KEY_ARRIVAL),
    ((18, 10, 5, "high", "burst"), ref.KEY_OBSERVED),
]:
    short = ref.geometry_config(cell, "when_idle", 10 / 3)
    full = ref.geometry_config(cell, "when_idle", 10)
    diffs = [
        ref.run(short, seed, key)["penalized_flow_time"]
        - ref.run(full, seed, key)["penalized_flow_time"]
        for seed in ref.SEEDS
    ]
    average, half = ref.interval(diffs)
    print(cell, average, (average - half, average + half))
```

For run-to-run comparisons, match episode rows by the five design fields, coverage, mode and seed. Parallel workers can finish in a different order, and gzip headers can differ. The SHA-256 for the **delivered compressed episode file** is `cd615d589c6be42262702abd0b018ede03aea9944cf8f8ff2b0c7263192a6b4d`.
