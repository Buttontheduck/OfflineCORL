#!/bin/bash
# Experiment 5, follow-up: live Aware at T = 4 for 1M steps on D4RL halfcheetah-medium-expert-v2,
# 3 seeds (sweeps/exp5_aware_1m_T4.yaml), then flip counts at 151k, 310k, 500k and 1M (100 new
# episodes per seed, env seeds 1000+, as for the T = 1 run) and a T = 1 vs T = 4 summary.
# Needs no input; start it detached:
#     setsid nohup bash run_exp5_1m.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp5_1m_<date-time>/orchestrator.log (German time). Result: .../summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
C=/workspace/Trained_Models/checkpoints
O=diagnostics/out/interv
R=sweep_runs/exp5_1m_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R" "$O"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 5, 1M-step T = 4 runs: started in $R"
$HAWK run_sweep.py sweeps/exp5_aware_1m_T4.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp5_aware_1m_T4_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"

J=$R/flip_jobs.txt; : > "$J"
for s in 1 2 3; do
  ck=$(newest "$C/Aware_1M_T4_seed$s-*")
  for step in 151000 310000 500000 1000000; do
    if [ -z "$ck" ] || [ ! -f "$ck/checkpoint_$step.pt" ]; then log "no checkpoint_$step.pt for seed $s"; continue; fi
    echo "$HAWK diagnostics/interventions.py $ck $O/aware1m_T4_ck${step}_s$s.json --checkpoint checkpoint_$step.pt --only aware_b1_n1 --n_eps 100 --seed0 1000 > $R/flip_ck${step}_s$s.log 2>&1 || echo ck${step}_s$s >> $R/flip_failed.txt" >> "$J"
  done
done
log "flip counts: $(wc -l < "$J") jobs, 6 at a time"
xargs -P 6 -d '\n' -I{} bash -c '{}' < "$J"
if [ -s "$R/flip_failed.txt" ]; then log "flip counts done, FAILED: $(xargs < "$R/flip_failed.txt")"; else log "flip counts done"; fi

$HAWK diagnostics/summarize_exp5_1m.py > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "finished"
