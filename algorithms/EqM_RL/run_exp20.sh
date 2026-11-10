#!/bin/bash
# Experiment 20: IEQM (Aware/train_ieqm.py) with expectile 0.7 and 0.8 instead of 0.9, 5 seeds each, and the baseline's
# two missing seeds, on humanoidmaze-medium task 1. Order, 5 runs at once:
#   1. expectile 0.7, seeds 1-5    sweeps/exp20_ogbench_ieqm_expectile.yaml
#   2. expectile 0.8, seeds 1-5    (same sweep; each run starts when one of the first five ends)
#   3. expectile 0.9, seeds 4, 5   sweeps/exp19_ogbench_ieqm_seeds45.yaml (Experiment 19 has seeds 1-3 already)
# Then diagnostics/summarize_exp20.py (scores) and, per setting, part 1 of diagnostics/exp19_critic_drift.py: at every
# checkpoint, does walking uphill on V from the start cell end in the goal cell (IFQL.md, Section 12.7)?
# Needs no input; start it detached:
#     setsid nohup bash run_exp20.sh > /dev/null 2>&1 < /dev/null &
# Stop: kill -TERM the run_sweep.py process (and this script). Each run then gets SIGINT, so wandb records 'killed'.
# Progress: sweep_runs/exp20_<date-time>/orchestrator.log (German time), the runs' stdout.log in the sweep folders.
# Result: sweep_runs/exp20_<date-time>/summary.md and uphill_<setting>.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python      # runs run_sweep.py (any Python with PyYAML)
OGB=/workspace/miniconda3/envs/ogbench/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
R=sweep_runs/exp20_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }
status() { tail -n +2 "$1/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs; }
summary() { $OGB diagnostics/summarize_exp20.py > "$R/summary.md" 2> "$R/summary.err" \
              && log "summary written to $R/summary.md ($1)" || log "summary FAILED (see $R/summary.err)"; }

log "Experiment 20 (IEQM: expectile 0.7, then 0.8, then the baseline's seeds 4 and 5) set up in $R"
$HAWK run_sweep.py sweeps/exp20_ogbench_ieqm_expectile.yaml > "$R/sweep.expectile.log" 2>&1
d=$(newest "sweep_runs/exp20_ogbench_ieqm_expectile_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "the expectile sweep FAILED to start (see $R/sweep.expectile.log)"; exit 1; fi
log "expectile 0.7 and 0.8 finished ($(status "$d")) in $d"
summary "expectile 0.7 and 0.8"
$HAWK run_sweep.py sweeps/exp19_ogbench_ieqm_seeds45.yaml > "$R/sweep.baseline.log" 2>&1
d2=$(newest "sweep_runs/exp19_ogbench_ieqm_seeds45_2*/")
if [ -z "$d2" ] || [ ! -f "$d2/runs.csv" ]; then log "the baseline sweep FAILED to start (see $R/sweep.baseline.log)"; else
  log "baseline seeds 4 and 5 finished ($(status "$d2")) in $d2"; fi
summary "all settings"
for spec in "expectile0.7 Exp20_IEQM _expectile0.7" "expectile0.8 Exp20_IEQM _expectile0.8" "expectile0.9 Exp19_IEQM NONE"; do
  set -- $spec; tag=$3; [ "$tag" = "NONE" ] && tag=""
  $OGB diagnostics/exp19_critic_drift.py --prefix "$2" --tag "$tag" --seeds 1,2,3,4,5 --maps "" --rollouts "" \
      > "$R/uphill_$1.md" 2> "$R/uphill_$1.err" \
    && log "uphill walk for $1 written to $R/uphill_$1.md" || log "uphill walk for $1 FAILED (see $R/uphill_$1.err)"
done
log "Experiment 20 finished"
