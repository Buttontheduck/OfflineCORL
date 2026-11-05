#!/bin/bash
# Experiment 8: Aware (T = 4, own critic) vs CORL's Gaussian AWAC on D4RL Maze2D (2-D actions):
# maze2d-umaze/-medium/-large-v1, 3 seeds, 1M steps. Both sweeps run at the same time
# (sweeps/exp8_maze2d_aware.yaml, sweeps/exp8_maze2d_awac.yaml); each one's gpu_gate starts a
# further run only while the GPU has room. Then the summary.
# Needs no input; start it detached:
#     setsid nohup bash run_exp8.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp8_<date-time>/orchestrator.log (German time) and the sweeps' launcher logs.
# Result: sweep_runs/exp8_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
R=sweep_runs/exp8_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 8 (Maze2D, Aware T = 4 vs Gaussian AWAC) started in $R"
$HAWK run_sweep.py sweeps/exp8_maze2d_aware.yaml > "$R/aware.launcher.log" 2>&1 &
PA=$!
sleep 30   # let the first sweep take its first GPU reading before the second one starts
$HAWK run_sweep.py sweeps/exp8_maze2d_awac.yaml > "$R/awac.launcher.log" 2>&1 &
PB=$!
wait $PA $PB

for s in exp8_maze2d_aware exp8_maze2d_awac; do
  d=$(newest "sweep_runs/${s}_2*/")
  if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "$s FAILED to start (see $R/*.launcher.log)"; continue; fi
  log "$s finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"
done

$HAWK diagnostics/summarize_exp8.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 8 finished"
