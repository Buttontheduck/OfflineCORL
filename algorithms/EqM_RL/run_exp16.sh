#!/bin/bash
# Experiment 16: three pulls toward the dataset actions for QIPO fine-tuning on OGBench (humanoidmaze-medium,
# antsoccer-arena default tasks, seeds 1-3, beta 10, frozen Experiment 14 critics and BC actors):
#   plain QIPO / data_mult 16 / bc_coef 1, and early stopping read off each at 80k and 120k.
#   18 runs of 400k   sweeps/exp16_ogbench_ft_anchor.yaml   (Aware/finetune_actor.py)
# then the summary (diagnostics/summarize_exp16.py).
# Needs no input; start it detached:
#     setsid nohup bash run_exp16.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py process (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp16_<date-time>/orchestrator.log (German time) and sweep.log.
# Result: sweep_runs/exp16_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python    # the summary reads wandb files
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
R=sweep_runs/exp16_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }

log "Experiment 16 (pulls toward the dataset actions for QIPO fine-tuning) set up in $R"
$HAWK run_sweep.py sweeps/exp16_ogbench_ft_anchor.yaml > "$R/sweep.log" 2>&1
d=$(newest "sweep_runs/exp16_ogbench_ft_anchor_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "the sweep FAILED to start (see $R/sweep.log)"; exit 1; fi
log "fine-tuning finished ($(status "$d")) in $d"
$OGB diagnostics/summarize_exp16.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 16 finished"
