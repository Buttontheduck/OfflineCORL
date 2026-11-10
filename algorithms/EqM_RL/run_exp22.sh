#!/bin/bash
# Launch detached: use a new session with stdin/stdout/stderr redirected.
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
HAWK=/workspace/miniconda3/envs/hawk/bin/python
OGB=/workspace/miniconda3/envs/ogbench/bin/python
R=sweep_runs/exp22_$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$R"
exec > "$R/orchestrator.log" 2>&1
echo "Experiment 22 started at $(date -u --iso-8601=seconds); reports in $R"
status=0
"$HAWK" run_sweep.py sweeps/exp22_ogbench_ieqm_antsoccer.yaml --out "$R" > "$R/sweep.log" 2>&1 || status=$?
found=0
for sweep_dir in "$R"/exp22_ogbench_ieqm_antsoccer_*; do
  if [ -d "$sweep_dir" ]; then
    found=1
    "$OGB" diagnostics/summarize_exp22.py "$sweep_dir" > "$R/summary.md" 2> "$R/summary.err" || status=1
  fi
done
if [ "$found" -eq 0 ]; then
  echo "Sweep folder missing; check $R/sweep.log"
  status=1
fi
echo "Experiment 22 ended at $(date -u --iso-8601=seconds); launcher status $status"
exit "$status"
