#!/bin/bash
# Experiment 17: Gaussian BC on humanoidmaze-medium (3 seeds, 500k steps; sweeps/exp17_ogbench_gauss_bc.yaml),
# then the two-group check of the Experiment 14 EqM-BC actors against these Gaussian actors
# (diagnostics/ogbench_two_groups.py, eval only, no wandb).
# Needs no input; start it detached:
#     setsid nohup bash run_exp17.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py process (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp17_<date-time>/orchestrator.log (German time) and sweep.bc.log.
# Result: diagnostics/out/two_groups/humanoidmaze-medium-navigate-singletask-v0/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python    # the two-group check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ENV=humanoidmaze-medium-navigate-singletask-v0
R=sweep_runs/exp17_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }

log "Experiment 17 (Gaussian BC on humanoidmaze-medium) set up in $R"
$HAWK run_sweep.py sweeps/exp17_ogbench_gauss_bc.yaml > "$R/sweep.bc.log" 2>&1
d=$(newest "sweep_runs/exp17_ogbench_gauss_bc_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "the sweep FAILED to start (see $R/sweep.bc.log)"; exit 1; fi
log "training finished ($(status "$d")) in $d"

log "two-group check started (EqM: Exp14_BC, Gaussian: Exp17_BC)"
$OGB diagnostics/ogbench_two_groups.py --env $ENV --kinds eqm,gaussian --bc_prefix Exp14_BC --gauss_prefix Exp17_BC \
  > "$R/two_groups.log" 2>&1 \
  && log "two-group check written to diagnostics/out/two_groups/$ENV/summary.md" \
  || log "two-group check FAILED (see $R/two_groups.log)"
log "Experiment 17 finished"
