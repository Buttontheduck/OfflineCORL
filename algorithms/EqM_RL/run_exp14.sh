#!/bin/bash
# Experiment 14: the QIPO pipeline on OGBench (humanoidmaze-medium, antsoccer-arena, cube-double default tasks, seeds 1-3):
#   phase 1  EqM-BC, 9 runs of 500k        sweeps/exp14_ogbench_bc.yaml        (Aware/fit_actor.py)
#   phase 2  critics, 9 runs of 1M         sweeps/exp14_ogbench_critic.yaml    (Aware/train_critic.py)
#   phase 3  fine-tuning, 27 runs of 400k  sweeps/exp14_ogbench_finetune.yaml  (Aware/finetune_actor.py)
# then the summary (diagnostics/summarize_exp14.py). Phases 1 and 2 start together: each critic waits for its BC
# checkpoint (as in Experiment 11). Phase 3 starts when phase 2 has finished.
# By default the script first waits until Experiment 13 (run_sweep.py sweeps/exp13_...) has finished, because the GPU
# has no room for both; set WAIT_FOR_EXP13=0 to start at once.
# Needs no input; start it detached:
#     setsid nohup bash run_exp14.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py processes (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp14_<date-time>/orchestrator.log (German time) and the sweep.*.log launcher logs.
# Result: sweep_runs/exp14_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python    # the summary reads wandb files
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
R=sweep_runs/exp14_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }

log "Experiment 14 (QIPO pipeline on OGBench) set up in $R"
if [ "${WAIT_FOR_EXP13:-1}" = "1" ]; then
  log "waiting for Experiment 13 to finish"
  while pgrep -f "run_sweep.py sweeps/exp13_" > /dev/null; do sleep 120; done
  log "Experiment 13 has finished"
fi

log "phases 1 (EqM-BC) and 2 (critics, each waits for its BC checkpoint) started"
$HAWK run_sweep.py sweeps/exp14_ogbench_bc.yaml > "$R/sweep.bc.log" 2>&1 &
BC_PID=$!
sleep 60   # BC prefetches the datasets first
$HAWK run_sweep.py sweeps/exp14_ogbench_critic.yaml > "$R/sweep.critic.log" 2>&1
wait $BC_PID
d1=$(newest "sweep_runs/exp14_ogbench_bc_2*/"); d2=$(newest "sweep_runs/exp14_ogbench_critic_2*/")
for d in "$d1" "$d2"; do
  if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "a sweep FAILED to start (see $R/sweep.*.log)"; exit 1; fi
done
log "phase 1 finished ($(status "$d1")) in $d1"
log "phase 2 finished ($(status "$d2")) in $d2"

log "phase 3 (fine-tuning) started"
$HAWK run_sweep.py sweeps/exp14_ogbench_finetune.yaml > "$R/sweep.finetune.log" 2>&1
d3=$(newest "sweep_runs/exp14_ogbench_finetune_2*/")
if [ -z "$d3" ] || [ ! -f "$d3/runs.csv" ]; then log "phase 3 FAILED to start (see $R/sweep.finetune.log)"; exit 1; fi
log "phase 3 finished ($(status "$d3")) in $d3"
$OGB diagnostics/summarize_exp14.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 14 finished"
