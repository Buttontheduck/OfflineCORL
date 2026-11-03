# Diagnostics

Scripts behind the diagnostics of `EqM_RL.md`, Experiment 2. They load trained
`Aware_Minari` runs from their checkpoint folders and need nothing else, so they don't
retrain anything. Run them from `algorithms/EqM_RL/` in the `eagle` env.

| script | what it measures |
|---|---|
| `diag.py` | **A:** env return vs sampler steps K. **B:** on 2,048 dataset states × 32 samples each, tracked over K: field norm, spread across one state's samples, error to the dataset action, movement, clipping; plus the field at dataset / uniform actions and the dataset's conditional spread (nearest-neighbour estimate). |
| `rollout.py` | **C:** returns, flips (body angle `obs[1]` beyond 2 rad) and forward speed (`obs[8]`) over 10 episodes, for one K. |
| `dataset_stats.py` | **C:** the same flip, speed and return statistics for the Minari datasets themselves. |
| `summarize.py` | Prints the tables (means over seeds) from the JSON files of `diag.py` and `rollout.py`. |
| `common.py` | Loads a run (config, dataset, state normalization, EqM field) and samples with exactly the sampler of `Actor._implicit` (NAG). |

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

`diagnostics/out/` is ignored by git. The scripts fix their random seeds, so a rerun on the
same checkpoints gives the same numbers: `diag.py` uses episode seeds 69, 70, … like the
training evals, and `rollout.py` uses 1000, 1001, ….

Notes:
- **Checkpoint:** the latest `checkpoint_*.pt` of a run, unless `--checkpoint` names one.
- **Sampler:** only the `nag` sampler (the default) is implemented.
- **Same critic:** changing K at evaluation keeps the critic and the weights that training
  produced with the training K.
