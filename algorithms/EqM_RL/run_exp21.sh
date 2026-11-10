#!/bin/bash
# Detached launcher for six IEQM AntMaze-large runs, followed by a local summary.
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
HAWK=/workspace/miniconda3/envs/hawk/bin/python
OGB=/workspace/miniconda3/envs/ogbench/bin/python
R=sweep_runs/exp21_$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$R"
exec > "$R/orchestrator.log" 2>&1
echo "Experiment 21 started at $(date -u --iso-8601=seconds); reports in $R"
status=0
"$HAWK" run_sweep.py sweeps/exp21_ogbench_ieqm_antmaze_large.yaml > "$R/sweep.log" 2>&1 || status=$?
"$OGB" diagnostics/summarize_exp21.py > "$R/summary.md" 2> "$R/summary.err" || status=1
echo "Experiment 21 ended at $(date -u --iso-8601=seconds); launcher status $status"
exit "$status"
