#!/bin/bash
# Experiment 13: Aware on OGBench single-task (humanoidmaze-medium, antsoccer-arena, cube-double default tasks; AWAC vs
# IQL critic; seeds 1-3), 18 runs of 1M steps (sweeps/exp13_ogbench_aware.yaml), then the summary
# (diagnostics/summarize_exp13.py). The runs use the ogbench conda env (run_sweep.py picks it for target aware_ogbench).
# The sweep runs in the foreground of this script: a sweep started with '&' inherits "ignore SIGINT"
# (run_sweep.py resets it for each run anyway). gpu_gate starts each further run only while the GPU has room.
# Needs no input; start it detached:
#     setsid nohup bash run_exp13.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM <run_sweep.py PID> (or Ctrl-C if run in a terminal). Each run then gets SIGINT,
# so wandb records it as 'killed'; runs still alive after 180 s are killed.
# Progress: sweep_runs/exp13_<date-time>/orchestrator.log (German time) and its sweep.launcher.log.
# Result: sweep_runs/exp13_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python    # the summary reads wandb files (wandb is in both envs)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
R=sweep_runs/exp13_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 13 (Aware on OGBench, AWAC vs IQL critic) started in $R; launcher log: $R/sweep.launcher.log"
$HAWK run_sweep.py sweeps/exp13_ogbench_aware.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp13_ogbench_aware_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"
$OGB diagnostics/summarize_exp13.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 13 finished"
