# Diagnostics

Scripts behind the diagnostics of `EqM_RL.md`: Experiment 2 (Minari) and the sampler-drift
diagnostics of Experiment 3 (D4RL). They load trained runs from their checkpoint folders and
need nothing else, so they don't retrain anything. Run them from `algorithms/EqM_RL/`:
`Aware_Minari` runs in the `eagle` env, `Aware` (D4RL) runs in the `hawk` env.

| script | what it measures |
|---|---|
| `diag.py` | **A:** env return vs sampler steps K. **B:** on 2,048 dataset states × 32 samples each, tracked over K: field norm, spread across one state's samples, error to the dataset action, movement, clipping; plus the field at dataset / uniform actions and the dataset's conditional spread (nearest-neighbour estimate). Minari runs. |
| `rollout.py` | **C:** returns, flips (body angle `obs[1]` beyond 2 rad) and forward speed (`obs[8]`) over 10 episodes, for one K. Minari runs. |
| `dataset_stats.py` | **C:** the same flip, speed and return statistics for the Minari datasets themselves. |
| `summarize.py` | Prints the tables (means over seeds) from the JSON files of `diag.py` and `rollout.py`. |
| `drift.py` | **Sampler drift vs critic** (Experiment 3), D4RL or Minari runs. **A:** rollouts for a grid of step size η × steps K × momentum (default 6 × 11 × 3, 20 episodes each, all side by side): return, flips, speed, actions outside the box, the critic's Q vs the discounted return actually collected, and how far the visited states and executed actions are from the dataset (k nearest dataset states; on medium-expert also whether they lie in the medium or the expert part of the data). **B:** dataset states × samples tracked over K for each (η, momentum): field norm, movement per step, spread, outside the box, distance to the nearest neighbour action, Q(sample) − Q(data action). With `--awac <CORL AWAC checkpoint folder>` also a trained Gaussian AWAC critic as an independent judge, and the AWAC policy itself for reference. |
| `summarize_drift.py` | Prints the tables of `drift.py` (means over seeds). |
| `common.py` | Loads a run (config, dataset, state normalization, EqM field, optionally the critics; Minari or D4RL from the run's `env_name`) and samples with exactly the sampler of `Actor._implicit` (NAG), also with per-row K, η and momentum. |

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

`diagnostics/out/` is ignored by git. The scripts fix their random seeds, so a rerun on the
same checkpoints gives the same numbers: `diag.py` uses episode seeds 69, 70, … like the
training evals, and `rollout.py` uses 1000, 1001, ….

Notes:
- **Checkpoint:** the latest `checkpoint_*.pt` of a run, unless `--checkpoint` names one.
- **Sampler:** only the `nag` sampler (the default) is implemented.
- **Same critic:** changing K at evaluation keeps the critic and the weights that training
  produced with the training K.
