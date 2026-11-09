#!/bin/bash
# Experiment 18: Aware on humanoidmaze-medium with V from 16 samples, gamma 0.999, 4 x 512 networks and critic
# LayerNorm (sweeps/exp18_ogbench_aware_v.yaml; 3 seeds, 1M steps).
# Needs no input; start it detached:
#     setsid nohup bash run_exp18.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py process (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp18_<date-time>/orchestrator.log (German time), the runs' stdout.log in the sweep folder.
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
R=sweep_runs/exp18_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }

log "Experiment 18 (Aware on humanoidmaze: V from 16 samples, gamma 0.999, width 512, critic LayerNorm) set up in $R"
$HAWK run_sweep.py sweeps/exp18_ogbench_aware_v.yaml > "$R/sweep.log" 2>&1
d=$(newest "sweep_runs/exp18_ogbench_aware_v_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "the sweep FAILED to start (see $R/sweep.log)"; exit 1; fi
log "training finished ($(status "$d")) in $d"
log "Experiment 18 finished"
