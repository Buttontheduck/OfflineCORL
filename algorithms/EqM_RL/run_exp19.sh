#!/bin/bash
# Experiment 19: IEQM = the IFQL recipe (IFQL.md) with our EqM actor, on humanoidmaze-medium task 1, 3 seeds, 1M steps
# (sweeps/exp19_ogbench_ieqm.yaml, Aware/train_ieqm.py), then the summary (diagnostics/summarize_exp19.py).
# Each run also evaluates Experiment 14's frozen EqM-BC actor of its seed with the new critic, so those checkpoints
# must exist (/workspace/Trained_Models/checkpoints/Exp14_BC_*humanoidmaze-medium*_seed<1-3>-*/checkpoint_500000.pt).
# About 1 hour per run (2.5 ms per training step with compile, plus 20 evals of about 1 minute); the 3 runs run together.
# Needs no input; start it detached:
#     setsid nohup bash run_exp19.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py process (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp19_<date-time>/orchestrator.log (German time), the runs' stdout.log in the sweep folder.
# Result: sweep_runs/exp19_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python    # the summary needs numpy only
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
R=sweep_runs/exp19_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }

log "Experiment 19 (IEQM: IFQL's critic + the EqM actor, best of 32) set up in $R"
$HAWK run_sweep.py sweeps/exp19_ogbench_ieqm.yaml > "$R/sweep.log" 2>&1
d=$(newest "sweep_runs/exp19_ogbench_ieqm_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "the sweep FAILED to start (see $R/sweep.log)"; exit 1; fi
log "training finished ($(status "$d")) in $d"
$OGB diagnostics/summarize_exp19.py "$d" > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 19 finished"
