#!/bin/bash
# Experiment 10: EqM-BC vs Gaussian BC on D4RL AntMaze (umaze, umaze-diverse, medium-diverse, large-diverse), 3 seeds,
# 500k steps, actor only (sweeps/exp10_antmaze_bc.yaml), then the summary (diagnostics/summarize_exp10.py).
# The sweep runs in the foreground of this script: a sweep started with '&' inherits "ignore SIGINT"
# (run_sweep.py now resets it for each run anyway). gpu_gate starts each further run only while the
# GPU has room. Needs no input; start it detached:
#     setsid nohup bash run_exp10.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM <run_sweep.py PID> (or Ctrl-C if run in a terminal). Each run then gets SIGINT,
# so wandb records it as 'killed'; runs still alive after 180 s are killed.
# Progress: sweep_runs/exp10_<date-time>/orchestrator.log (German time) and its sweep.launcher.log.
# Result: sweep_runs/exp10_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
R=sweep_runs/exp10_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 10 (AntMaze, EqM-BC vs Gaussian BC) started in $R; launcher log: $R/sweep.launcher.log"
$HAWK run_sweep.py sweeps/exp10_antmaze_bc.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp10_antmaze_bc_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"
$HAWK diagnostics/summarize_exp10.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 10 finished"
