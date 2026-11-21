# Training on another machine (0.7.2)

Expert-iteration training (`docs/methods/rl2.md` §6) is portable. It runs on any
Linux or macOS machine with Python ≥ 3.10, on the CPU or on a GPU, and it can be
stopped and resumed. This page says what machine to use, how to set it up, how to
launch and resume a run, and how to bring the results back.

Scripts: `experiments/remote/setup_env.sh`, `run_train.sh` and `slurm_train.sbatch`.

## 1. Why another machine

The laptop has 8 cores and no CUDA GPU. Its measured costs:

| step | where it runs | on the laptop | what speeds it up |
|---|---|---|---|
| data generation (search teacher, one process per core) | CPU | 1 000 episodes in 84 s with the rule teacher; the network-based teacher is about 5× slower | more cores, almost linearly |
| validation (≈ 120 episodes × 3 policies) | CPU | about 1–3 min | more cores |
| network training | CPU (8 threads) or GPU | ≈ 4–5 min per epoch of 50 000 decisions for size S | **a GPU**: typically 10–50× for this model |

* Network training is the bottleneck of a large run.
* Sizes M and L (6.7 M and 39 M parameters) are only practical with a GPU.

**Recommended machine:**

* **cores:** 16–64, which sets the speed of data generation and validation;
* **GPU:** one NVIDIA card with ≥ 8 GB, enough for S and M; L needs more;
* **memory:** ≥ 32 GB; the replay buffer of a large run holds a few hundred thousand
  decisions in memory.

**Where to get it:**

* a university cluster with Slurm: use `slurm_train.sbatch`;
* a cloud VM with a GPU;
* a PC with an NVIDIA card.

Google Colab is a poor fit: it gives only 2 CPU cores, and simulation needs as many
cores as training needs the GPU.

## 2. What changed in the code to make this possible

* **Device.** `--device auto|cpu|cuda|cuda:N|mps`. Training runs on that device; the
  simulation workers stay on the CPU, where single-state network calls are fast.
  * Models are always saved with CPU weights, so a model trained on a GPU loads
    anywhere.
  * A test checks that GPU and CPU training give the same losses (it is skipped on
    machines without CUDA).
* **Checkpoint and resume.**
  * After each completed iteration the whole run state is written atomically to
    `results/experiments/<tag>/state.pkl`: replay buffer, value scale, seeds, best
    model and the log.
  * `--resume` continues from the next iteration.
  * The experiment arguments are taken from the saved run, so the resumed run is the
    same experiment. Only `--iters`, `--workers`, `--threads`, `--device` and
    `--patience` may change.
  * Every machine that worked on the run is recorded in the log (`machine`: host,
    CPUs, device, GPU name, versions).
  * An interrupted iteration is lost and is redone from its start.
* **Portable paths.** Model paths in the log and in the state are relative to the
  repository.
* **Workers.** The default is all logical CPUs. `run_train.sh` uses all but one,
  leaving one for the trainer.
* **`--horizon S`.** Shorter episodes, for quick checks.

## 3. Steps

1. **Get the code onto the machine.**
   * The repository is public on GitHub. The commits for 0.7.0–0.7.2 are local until
     they are pushed.
   * If you clone, push first. Otherwise copy the folder; `results/` is not needed.
2. **Set up:**
   ```bash
   bash experiments/remote/setup_env.sh
   ```
   If the login shell exports `PYTHONPATH` (ROS does), run every project command with
   `env -u PYTHONPATH …`: otherwise pytest loads plugins from there and fails before
   the project tests start (seen on the first workstation, 0.7.4).
   With a GPU, choose the PyTorch CUDA build that matches the driver, for example:
   ```bash
   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash experiments/remote/setup_env.sh
   ```
   The script then runs the fast tests and a 2-minute end-to-end check.
3. **Launch:**
   * workstation or VM:
     ```bash
     nohup bash experiments/remote/run_train.sh &
     ```
   * Slurm: first fill in the `FILL` lines of `slurm_train.sbatch` (partition, GPU
     syntax, modules), then
     ```bash
     sbatch experiments/remote/slurm_train.sbatch
     ```
   * Variables in the environment choose the run, for example
     `TAG=rl2_gen_m SIZE=M EPISODES=3000`.
4. **Resume:**
   * Run the same command again. `run_train.sh` finds `state.pkl` and resumes.
   * On Slurm, `--requeue` and resubmission both resume.
5. **Follow progress:** `results/experiments/<tag>/train.log`.
6. **Bring the results back:**
   * `models/<tag>.pt` (the best model) and `models/<tag>_training.json` (the full
     log) are the deliverables.
   * `models/<tag>_it*.pt` holds the model of each iteration and is optional.
   * `state.pkl` is only needed to continue training.

## 4. Suggested large runs

These are starting points, to be adjusted with the times measured in the first
iteration.

Measured on a 48-thread workstation with an RTX 3080 (0.7.4, `rl2.md` §8.4): S takes
25 min for 6 iterations and M about 6 h, because the network teacher of iterations
≥ 1 runs on the CPU workers and is 8.5× slower per episode for M than for S.

| run | size | episodes (iteration 0 / later) | iterations | validation seeds per group | note |
|---|---|---|---|---|---|
| `rl2_gen_s_big` (the defaults of `run_train.sh`) | S | 4 000 / 2 000 | ≤ 10 | 50 (general: 100) | 4–5× the data of the laptop pilot |
| `rl2_gen_m` | M | 8 000 / 4 000 | ≤ 10 | 50 | only with a GPU; it needs about twice the data of S, otherwise a larger model only overfits (plan §3) |

**Choosing the size.** Keep the larger model only if it wins on validation (plan
§3), and report score against parameters and data.

**Comparability.** The validation seeds and groups are the same on every machine. The
training seeds continue from 100000, so runs are comparable and reproducible.
