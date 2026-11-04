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
| `awac_d4rl` | CORL's `algorithms/offline/awac.py` (Gaussian AWAC, pyrallis) | D4RL | `hawk` |
| `iql_d4rl` | CORL's `algorithms/offline/iql.py` (IQL, pyrallis; its dataset key is `env`, not `env_name`) | D4RL | `hawk` |
| `fit_actor` | `Aware/fit_actor.py` (Gaussian or EqM actor alone, on fixed weights, pyrallis) | D4RL | `hawk` |

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

## Experiment 4: priority 2 and the 1M-step run (310k steps)

`EqM_RL.md`, "Planned next experiments", section 2 and 5c. Every training run uses 310k steps,
except the 1M-step Aware run. Weight files go to `/workspace/Trained_Models/weights/` (not in
git), one per source and seed: `<source>_seed<seed>.npz`, the names `fit_actor.py` looks for.

**Unattended:** `run_exp4.sh` (in `algorithms/EqM_RL/`) runs priority 1 and every step of this
table in order, waits for its inputs and for free GPU memory, and ends by writing
`sweep_runs/exp4_<date-time>/summary.md` (`diagnostics/summarize_exp4.py`). Start it detached:
`setsid nohup bash run_exp4.sh > /dev/null 2>&1 < /dev/null &`. Don't edit the script while it
runs: bash reads it as it goes.

| Order | Command | Runs | Needs |
|---|---|---|---|
| any time | `python run_sweep.py sweeps/exp4_aware_1m.yaml` | 3 | nothing (started 2026-11-04) |
| any time | `python run_sweep.py sweeps/exp4_fit_bc.yaml` | 6 | nothing |
| any time | `python run_sweep.py sweeps/exp4_iql.yaml` | 3 | nothing |
| after `exp4_iql` | `make_weights.py --source iql` (below) | – | the IQL checkpoints |
| then | `python run_sweep.py sweeps/exp4_fit_iql.yaml` | 6 | `iql_seed*.npz` |
| after the 1M runs pass 310k | `make_weights.py --source aware` (below) | – | `checkpoint_310000.pt` |
| then | `python run_sweep.py sweeps/exp4_fit_aware.yaml` | 6 | `aware_seed*.npz` |
| optional | `python run_sweep.py sweeps/exp4_fit_aware_matched.yaml` | 6 | `aware_matched_seed*.npz` |

```bash
cd algorithms/EqM_RL/Aware      # hawk env, with the D4RL exports of diagnostics/README.md
C=/workspace/Trained_Models/checkpoints; W=/workspace/Trained_Models/weights
for s in 1 2 3; do
  python make_weights.py --source iql --run_dir $(ls -d $C/IQL_seed$s-*) --out $W/iql_seed$s.npz
  python make_weights.py --source aware --run_dir $(ls -d $C/Aware_1M_seed$s-*) \
      --checkpoint checkpoint_310000.pt --out $W/aware_seed$s.npz
  python make_weights.py --source aware --run_dir $(ls -d $C/Aware_1M_seed$s-*) \
      --checkpoint checkpoint_310000.pt --match_share $W/iql_seed$s.npz --out $W/aware_matched_seed$s.npz
done
```

- **Same targets for both actors:** `make_weights.py` computes the weights once per seed; the
  Gaussian and the EqM fit of that seed read the same file. For the Aware source, V(s) averages
  32 EqM samples per state with a fixed seed (training used 1). About 1 min per file.
- **Matched concentration:** with the weight cap, the effective batch share is not monotone in
  T, so `--match_share` scans T on a log grid, refines every crossing, takes the one closest to
  T = 1, lists all crossings in the file's `meta`, and saves nothing if the match misses by more
  than 0.005 (`--match_tol`).
- **Both sources at 310k:** the IQL critics train 310k steps, and the Aware weights come from
  the 1M-step runs' `checkpoint_310000.pt`, so both weight sources had the same training length.
- **The fits:** `fit_actor.py` trains only the actor (no critic), evaluates 10 episodes every
  1k steps (score, normalized score, flip rate), and keeps a checkpoint every 10k steps. Both
  kinds of fit load with `diagnostics/common.py` (EqM with `Policy`, Gaussian with `AwacPolicy`),
  so the priority-1 scripts work on them too.
- **Expert half:** `data_filter: expert_half` keeps the second 1,000 of the 2,000 episodes
  (999,000 transitions). Their returns: mean 10,645, minimum 2,046 (a few of the expert's own
  episodes go wrong); the dropped medium half's maximum is 5,304.


## Experiment 5: live Aware at T = 1, 2, 3, 4 (310k steps)

`EqM_RL.md`, "Experiment 4, priorities 2 and 3", "What these results say about the temperature
T": with frozen Aware weights, T = 2.3–3.9 made EqM learn faster than T = 1. This checks whether
that holds when the critic trains alongside the actor. 4 temperatures × 3 seeds = 12 runs at once,
T = 1 rerun as the control in the same batch.

**Unattended:** `run_exp5.sh` runs `sweeps/exp5_temperature.yaml`, then counts flips at
`checkpoint_310000.pt` (100 new episodes per seed, env seeds 1000+, as in Experiment 4) and writes
`sweep_runs/exp5_<date-time>/summary.md` (`diagnostics/summarize_exp5.py`): score at 310k,
learning curves, flips, and the live weights' effective batch share over training (from the
per-step `weighting/*` logs). Start it detached:
`setsid nohup bash run_exp5.sh > /dev/null 2>&1 < /dev/null &`. 12 runs at once: about 40
steps/s each, ≈ 2.5 h, plus ≈ 20 min of flip counts.
