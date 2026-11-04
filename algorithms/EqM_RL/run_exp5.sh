#!/bin/bash
# Experiment 5: live Aware at T = 1, 2, 3, 4 on D4RL halfcheetah-medium-expert-v2, 310k steps,
# 3 seeds (sweeps/exp5_temperature.yaml), then flip counts at checkpoint_310000.pt (100 new episodes
# per seed, env seeds 1000+, as Experiment 4) and the summary. Needs no input; start it detached:
#     setsid nohup bash run_exp5.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp5_<date-time>/orchestrator.log (German time). Result: .../summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
C=/workspace/Trained_Models/checkpoints
O=diagnostics/out/interv
R=sweep_runs/exp5_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R" "$O"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 5 started in $R"
log "sweep exp5_temperature: start (12 runs at once)"
$HAWK run_sweep.py sweeps/exp5_temperature.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp5_temperature_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"

J=$R/flip_jobs.txt; : > "$J"
for T in 1.0 2.0 3.0 4.0; do
  for s in 1 2 3; do
    ck=$(newest "$C/Exp5_T_temperature${T}_seed$s-*")
    if [ -z "$ck" ] || [ ! -f "$ck/checkpoint_310000.pt" ]; then log "no checkpoint_310000.pt for T = $T, seed $s"; continue; fi
    echo "$HAWK diagnostics/interventions.py $ck $O/exp5_T${T}_s$s.json --checkpoint checkpoint_310000.pt --only aware_b1_n1 --n_eps 100 --seed0 1000 > $R/flip_T${T}_s$s.log 2>&1 || echo T${T}_s$s >> $R/flip_failed.txt" >> "$J"
  done
done
log "flip counts: $(wc -l < "$J") jobs, 6 at a time"
xargs -P 6 -d '\n' -I{} bash -c '{}' < "$J"
if [ -s "$R/flip_failed.txt" ]; then log "flip counts done, FAILED: $(xargs < "$R/flip_failed.txt")"; else log "flip counts done"; fi

$HAWK diagnostics/summarize_exp5.py --sweep_dir "$d" > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 5 finished"
