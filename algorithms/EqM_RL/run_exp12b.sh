#!/bin/bash
# Experiment 12b: Experiment 12 with the support samples at K = 20 sampler steps instead of 10, 18 runs of 400k steps
# (sweeps/exp12b_antmaze_finetune_K20.yaml), then the summary (diagnostics/summarize_exp12.py --prefix Exp12b_FT),
# followed by Experiment 12's table in the same layout for comparison.
# The sweep runs in the foreground of this script: a sweep started with '&' inherits "ignore SIGINT"
# (run_sweep.py resets it for each run anyway). gpu_gate starts each further run only while the GPU has room.
# Needs no input; start it detached:
#     setsid nohup bash run_exp12b.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM <run_sweep.py PID> (or Ctrl-C if run in a terminal). Each run then gets SIGINT,
# so wandb records it as 'killed'; runs still alive after 180 s are killed.
# Progress: sweep_runs/exp12b_<date-time>/orchestrator.log (German time) and its sweep.launcher.log.
# Result: sweep_runs/exp12b_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
R=sweep_runs/exp12b_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 12b (Exp 12 with support_K = 20) started in $R; launcher log: $R/sweep.launcher.log"
$HAWK run_sweep.py sweeps/exp12b_antmaze_finetune_K20.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp12b_antmaze_finetune_K20_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"
S="--steps 40000,60000,100000,200000,400000 --window 20000"   # single evals: before / just after the first renewal, ...
{ $HAWK diagnostics/summarize_exp12.py --prefix Exp12b_FT $S \
      --title "Experiment 12b: Experiment 12 with support_K = 20 (3 seeds)"
  echo; echo "---"; echo
  $HAWK diagnostics/summarize_exp12.py --prefix Exp12_FT $S \
      --title "For comparison: Experiment 12, support_K = 10 (3 seeds), same layout"
} > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 12b finished"
