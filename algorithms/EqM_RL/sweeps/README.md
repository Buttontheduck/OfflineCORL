# Sweeps

Each YAML file here is one hyperparameter sweep. `../run_sweep.py` expands it into runs
and trains many of them at once on one GPU.

```bash
cd algorithms/EqM_RL
nvidia-cuda-mps-control -d                                # once per boot (see below)
python run_sweep.py sweeps/temperature_minari.yaml --dry-run   # list the runs first
python run_sweep.py sweeps/temperature_minari.yaml
```

Any Python with PyYAML can start `run_sweep.py` (base, `hawk` or `eagle`). It starts each
training run with the Python of the conda env that the target needs, so no `conda activate`
is needed. On a remote server, run it inside `tmux` or with `nohup ... &`, so it survives
a closed SSH session.

## File format

```yaml
target: aware_minari     # what to train (table below)
parallel: 16             # runs at once (optional, default 16; --parallel overrides it)

fixed:                   # the same for every run
  env_name: mujoco/halfcheetah/simple-v0
  num_train_ops: 120001
  agent.fino: false      # nested keys use dots

grid:                    # every combination is one run: 3 x 2 x 3 = 18 runs
  temperature: [0.3, 1.0, 3.0]
  actor.num_step: [10, 20]
  seed: [1, 2, 3]
```

| target | trains | data | conda env |
|---|---|---|---|
| `aware` | `Aware/main.py` (Hydra config `Aware/config.yaml`) | D4RL | `hawk` |
| `aware_minari` | `Aware_Minari/main.py` (Hydra config `Aware_Minari/config.yaml`) | Minari | `eagle` |
| `awac_minari` | `Awac_Minari.py` (Gaussian AWAC baseline, pyrallis `TrainConfig`) | Minari | `eagle` |

- **Keys** are the names in the target's config. A key that doesn't exist is an error,
  which catches typos (for Hydra targets, write `+key: value` to add a new key).
- **Anything not set** keeps the value from the config file.
- **Several grids:** `grid` can also be a list. The runs are the union of each grid's
  combinations, so not every pair has to be crossed:
  ```yaml
  grid:
    - {agent.flag_threshold: 0.8, agent.scale_gradient: [4.0, 8.0], seed: [1, 2, 3]}
    - {agent.flag_threshold: 0.9, agent.scale_gradient: 8.0, seed: [1, 2, 3]}
  ```
- **A list as one value:** wrap it in another list, e.g. `some_key: [[1, 2], [3, 4]]` is
  two runs.
- **Optional keys:** `conda_env` (use another env than the default), `stagger` (seconds
  between starts, default 2).

## What the runner sets for you

- **W&B:** `group` is the sweep file name unless `fixed` sets it. `name` is the target
  prefix (`Aware` / `AWAC`, or `fixed: name:`) plus the grid values, e.g.
  `Aware_temperature0.3_seed1`. The training script adds the env name and a random id.
- **Before starting:** downloads each dataset once, so parallel runs don't download the
  same file at the same time.
- **Environment of every run:** `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` (otherwise each
  run starts a thread per CPU core). For `aware` it also adds `~/.mujoco/mujoco210/bin` to
  `LD_LIBRARY_PATH`, which mujoco-py needs.

## Output

`sweep_runs/<sweep>_<date-time>/` (ignored by git):

- `sweep.yaml`: a copy of the sweep file.
- `plan.txt`: the full command of every run (to rerun one by hand).
- `runs.csv`: the status of each run (`waiting`, `running`, `done`, `failed`, `stopped`,
  `not started`), its exit code, minutes and log path. It is updated while the sweep runs.
- `<idx>_<tag>/stdout.log`: the run's output. Hydra runs also write `<idx>_<tag>/hydra/`.

Checkpoints still go to the config's `checkpoints_path` (`/workspace/Trained_Models/checkpoints/`),
one folder per run. One Aware checkpoint is about 2.4 MB, and a 120k-step run saves 121 of
them (about 290 MB). Set `checkpoints_path: null` in `fixed` to skip them.

Ctrl-C (or `kill <runner pid>`) stops the sweep: running runs get Ctrl-C too, so they can
close their W&B run, and anything not started yet is marked `not started`.

## Throughput (RTX 4090, Aware, measured 2026-11-03)

One Aware run trains at about 43 steps/s, so a 120k-step run takes about 47 min plus
evaluation. The GPU mostly waits for the CPU to send it many tiny kernels, so parallel runs
are almost free, but only with **CUDA MPS** running (`nvidia-cuda-mps-control -d`, as root,
once per boot; stop with `echo quit | nvidia-cuda-mps-control`). The runner warns if MPS is
off.

| runs at once | per run, steps/s | total, steps/s |
|---|---|---|
| 1 | 43 | 43 |
| 16, no MPS | 14 | 229 |
| 16, MPS | 43 | 690 |
| 24, MPS | 27 | 652 |
| 32, MPS | out of GPU memory (about 0.9 GB per run) | |

So 16 at once is the sweet spot on this GPU.
