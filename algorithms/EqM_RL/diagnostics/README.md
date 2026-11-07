# Diagnostics

Scripts behind the diagnostics of `EqM_RL.md`: Experiment 2 (Minari) and the sampler-drift
diagnostics of Experiment 3 (D4RL). They load trained runs from their checkpoint folders and
need nothing else, so they don't retrain anything. Run them from `algorithms/EqM_RL/`:
`Aware_Minari` runs in the `eagle` env, `Aware` (D4RL) runs in the `hawk` env.

| script | what it measures |
|---|---|
| `diag.py` | **A:** env return vs sampler steps K. **B:** on 2,048 dataset states × 32 samples each, tracked over K: field norm, spread across one state's samples, error to the dataset action, movement, clipping; plus the field at dataset / uniform actions and the dataset's conditional spread (nearest-neighbour estimate). Minari runs. |
| `rollout.py` | **C:** returns, flips (body angle `obs[1]` beyond 2 rad) and forward speed (`obs[8]`) over 10 episodes, for one K. Minari runs; with `--awac`, an `Awac_Minari.py` (Gaussian AWAC) run, evaluated with its mean action. |
| `dataset_stats.py` | **C:** the same flip, speed and return statistics for the Minari datasets themselves. |
| `summarize.py` | Prints the tables (means over seeds) from the JSON files of `diag.py` and `rollout.py`. |
| `drift.py` | **Sampler drift vs critic** (Experiment 3), D4RL or Minari runs. **A:** rollouts for a grid of step size η × steps K × momentum (default 6 × 11 × 3, 20 episodes each, all side by side): return, flips, speed, actions outside the box, the critic's Q vs the discounted return actually collected, and how far the visited states and executed actions are from the dataset (k nearest dataset states; on medium-expert also whether they lie in the medium or the expert part of the data). **B:** dataset states × samples tracked over K for each (η, momentum): field norm, movement per step, spread, outside the box, distance to the nearest neighbour action, Q(sample) − Q(data action). With `--awac <CORL AWAC checkpoint folder>` also a trained Gaussian AWAC critic as an independent judge, and the AWAC policy itself for reference. |
| `summarize_drift.py` | Prints the tables of `drift.py` (means over seeds). |
| `interventions.py` | **Experiment 4, priority 1:** evaluation-only action interventions on one D4RL Aware run and the AWAC run of the same seed: executed bounds ±1 / ±0.95 / ±0.9 (Aware and AWAC) and the average of 4 or 16 clipped EqM samples. Per episode: return, flip, speed, raw dims outside the box, executed dims at the bound, action change; and the same quantities over the 50 steps before each flip vs the same steps of episodes without a flip. |
| `handoff.py` | **Experiment 4, priority 1 (plan 5a):** saves the full MuJoCo state of AWAC and Aware episodes at steps 100 / 300 / 500 and continues every saved state for 200 steps with each policy (2 × 2: state source × continuation). Flips, return and speed per cell, upright and already-flipped states apart; first checks that every restored state replays the source's recorded actions exactly (20 steps, tolerance 1e-6), and stops otherwise. |
| `summarize_exp4.py` | Markdown tables of Experiment 4, priority 1 and 2: interventions (screening and confirmation), handoff cells, Minari flips, the weight files, and the last-20-eval scores and flip rates of the training sweeps. `run_exp4.sh` runs it at the end. |
| `maze2d_modes.py` | **Maze2D action diagnostics** (Experiments 8 and 9; 2-D actions). On 256 fixed dataset states × 256 samples, at K = 20 / 30 / 100 / 1000 of one sampler trajectory: ‖f‖ at the samples (convergence), spread, number of modes (clusters, radius 0.05) and the mass of the largest, and recall / precision / energy distance / off-support share against the actions of the 128 nearest dataset states (with the data's own half-vs-half values as reference). The Gaussian AWAC mean action of the same dataset and seed as reference. Rollouts (100 episodes) at K = 20 / 30 / 100: score, executed actions at ±1, action change per step. An SVG per checkpoint with data and samples for 6 states. `summarize_maze2d_modes.py` makes the tables; `run_maze2d_modes.sh` runs all 27 Maze2D runs. |
| `common.py` | Loads a run (config, dataset, state normalization, EqM field, optionally the critics; Minari or D4RL from the run's `env_name`) and samples with exactly the sampler of `Actor._implicit` (NAG), also with per-row K, η and momentum. Also loads a Gaussian policy (`AwacPolicy`: a CORL AWAC run, or a Gaussian fit of `Aware/fit_actor.py`; critics only if the checkpoint has them) and saves / restores the MuJoCo state of a D4RL env. |

## Reproducing Experiment 2

```bash
cd algorithms/EqM_RL
conda activate eagle
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=/workspace/Trained_Models/checkpoints

# A + B for all 15 runs, in parallel (start CUDA MPS first; about 6 min each)
for d in $C/Aware_env_name*; do
  python diagnostics/diag.py $d diagnostics/out/diag/$(basename $d | cut -d- -f1-2).json &
done; wait

# C: rollouts with flips
for d in $C/Aware_env_namemedium-v0_scale_gradient*; do
  for K in 20 100; do
    python diagnostics/rollout.py $d diagnostics/out/roll/$(basename $d | cut -d- -f1-2)_K$K.json --K $K &
  done
done
for d in $C/Aware_env_namesimple-v0_scale_gradient8.0*; do
  python diagnostics/rollout.py $d diagnostics/out/roll/$(basename $d | cut -d- -f1-2)_K20.json --K 20 &
done; wait

python diagnostics/dataset_stats.py mujoco/halfcheetah/simple-v0 mujoco/halfcheetah/medium-v0
python diagnostics/summarize.py --diag diagnostics/out/diag --roll diagnostics/out/roll
```

## Reproducing the Experiment 3 drift diagnostics (D4RL medium-expert / medium-replay)

```bash
cd algorithms/EqM_RL
conda activate hawk
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
C=/workspace/Trained_Models/checkpoints; O=diagnostics/out/drift

# One job per Aware checkpoint, with the AWAC run of the same seed as the reference.
# Each job holds 3,960 MuJoCo envs (~20 GB RAM): run at most 6 at once under a 143 GB limit.
for s in 1 2 3; do
  AME=$(ls -d $C/AWAC_env_namehalfcheetah-medium-expert-v2_seed$s-*)
  python diagnostics/drift.py $(ls -d $C/Aware_env_namehalfcheetah-medium-expert-v2_seed$s-*) $O/me_T1_s$s.json --awac $AME &
  python diagnostics/drift.py $(ls -d $C/Aware_temperature0.3333_seed$s-*) $O/me_T0.33_s$s.json --awac $AME &
  python diagnostics/drift.py $(ls -d $C/Aware_weightingbc_seed$s-*) $O/me_bc_s$s.json --awac $AME &
  wait
done
for s in 1 2 3; do
  python diagnostics/drift.py $(ls -d $C/Aware_env_namehalfcheetah-medium-replay-v2_seed$s-*) $O/mr_T1_s$s.json \
    --awac $(ls -d $C/AWAC_env_namehalfcheetah-medium-replay-v2_seed$s-*) &
done; wait

python diagnostics/summarize_drift.py $O
```

## Experiment 4, priority 1 (D4RL medium-expert and Minari medium; evaluation only)

`EqM_RL.md`, "Planned next experiments", section 1 and 5a/5b. Nothing is trained; every job
loads existing checkpoints. Setup as above (`hawk` env and the exports), plus:

```bash
C=/workspace/Trained_Models/checkpoints; O=diagnostics/out
for s in 1 2 3; do
  AME=$(ls -d $C/AWAC_env_namehalfcheetah-medium-expert-v2_seed$s-*)
  # 1a: clipping and averaging, screening (20 episodes per treatment, seeds 69, 70, ...)
  python diagnostics/interventions.py $(ls -d $C/Aware_env_namehalfcheetah-medium-expert-v2_seed$s-*) $O/interv/me_T1_s$s.json --awac $AME --save_steps &
  python diagnostics/interventions.py $(ls -d $C/Aware_temperature0.3333_seed$s-*) $O/interv/me_T0.33_s$s.json --awac $AME --save_steps &
  python diagnostics/interventions.py $(ls -d $C/Aware_weightingbc_seed$s-*) $O/interv/me_bc_s$s.json --awac $AME --save_steps &
  # 1b: handoff (AWAC vs Aware with T = 1 and T = 1/3)
  python diagnostics/handoff.py $(ls -d $C/Aware_env_namehalfcheetah-medium-expert-v2_seed$s-*) $AME $O/handoff/me_T1_s$s.json &
  python diagnostics/handoff.py $(ls -d $C/Aware_temperature0.3333_seed$s-*) $AME $O/handoff/me_T0.33_s$s.json &
done; wait

# 1a confirmation on new episodes, for the treatments the screening selects, e.g.:
#   python diagnostics/interventions.py <run> $O/interv/me_T1_s1_confirm.json --awac <AWAC run> \
#       --n_eps 100 --seed0 1000 --only aware_b1_n1,aware_b0.9_n1,awac_b1,awac_b0.9

# 1c (plan 5b): flips on Minari medium-v0, in the eagle env
conda activate eagle
for s in 1 2 3; do
  python diagnostics/rollout.py $(ls -d $C/AWAC_env_namemedium-v0_seed$s-*) $O/roll/medium_awac_s$s.json --awac --n_eps 20
  python diagnostics/rollout.py $(ls -d $C/Aware_env_namemedium-v0_scale_gradient8.0_weightingbc_seed$s-*) $O/roll/medium_bc_s$s.json --n_eps 20
done
```

- **Memory:** an `interventions.py` screening job holds 160 envs, a `handoff.py` job 600; far
  below the drift jobs, so all 15 D4RL jobs fit at once.
- **Common noise:** in `interventions.py` every Aware treatment draws the same sampler noise
  for the same episode and step, and all treatments use the same env seeds.
- **Pre-flip windows:** a flip needs at least `--window` (50) steps before it to count.
- **Restore:** `handoff.py` puts `(qpos, qvel)` into a freshly reset env. Before the
  continuations it restores every saved state and replays the source's own recorded actions
  for 20 steps; all observations must match within 1e-6 (`--restore_tol`), otherwise it stops
  and saves nothing. HalfCheetah's physics are deterministic; the smoke tests gave ~1e-11, most
  likely from MuJoCo's solver warm start, which `(qpos, qvel)` doesn't include.
- **Fitted actors:** both kinds of `Aware/fit_actor.py` checkpoints load here: an EqM fit as the
  main run (`Policy`), a Gaussian fit as `--awac` / the second argument (`AwacPolicy` rebuilds
  its saved depth and needs no critics). So the confirmation and handoff tests also work on them.

`diagnostics/out/` is ignored by git. The scripts fix their random seeds, so a rerun on the
same checkpoints gives the same numbers: `diag.py` uses episode seeds 69, 70, … like the
training evals, and `rollout.py` uses 1000, 1001, ….

Notes:
- **Checkpoint:** the latest `checkpoint_*.pt` of a run, unless `--checkpoint` names one.
- **Sampler:** only the `nag` sampler (the default) is implemented.
- **Same critic:** changing K at evaluation keeps the critic and the weights that training
  produced with the training K.

## Maze2D action diagnostics (Experiments 8 and 9; evaluation only)

`EqM_RL.md`, "Experiment 9: results", next step. All 27 Maze2D Aware runs (Experiment 9 at 500k and 1M,
Experiment 8's T = 4 at its last checkpoint), each with the Gaussian AWAC run of the same dataset and seed:

```bash
cd algorithms/EqM_RL
setsid nohup bash run_maze2d_modes.sh > /dev/null 2>&1 < /dev/null &   # 9 jobs at a time (PARALLEL=9)
# progress: diagnostics/out/maze2d_modes/orchestrator.log; tables: diagnostics/out/maze2d_modes/summary.md
# one run by hand:
python diagnostics/maze2d_modes.py <Aware checkpoint folder> <out.json> --awac <AWAC checkpoint folder> \
    --checkpoints 500000 last --K 20 30 100 1000 --rollout-K 20 30 100
```
