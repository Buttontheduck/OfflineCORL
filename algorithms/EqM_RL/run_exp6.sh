#!/bin/bash
# Experiment 6: Aware on D4RL antmaze-large-diverse-v2, T = 1/3, 1, 4, 3 seeds, 1M steps
# (sweeps/exp6_antmaze_ld.yaml), then the summary. The sweep's gpu_gate starts each further run only
# while the GPU has room, so the runs start one by one. Needs no input; start it detached:
#     setsid nohup bash run_exp6.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp6_<date-time>/orchestrator.log (German time) and the sweep's launcher log
# (which run started when). Result: sweep_runs/exp6_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
R=sweep_runs/exp6_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 6 (AntMaze large-diverse) started in $R; launcher log: $R/sweep.launcher.log"
$HAWK run_sweep.py sweeps/exp6_antmaze_ld.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp6_antmaze_ld_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"

$HAWK diagnostics/summarize_exp5_1m.py \
  --title "Aware on D4RL antmaze-large-diverse-v2: T = 1/3, 1, 4 (1M steps)" \
  --run "T = 1/3=AntMaze_LD_temperature0.3333:none" \
  --run "T = 1=AntMaze_LD_temperature1.0:none" \
  --run "T = 4=AntMaze_LD_temperature4.0:none" \
  --steps 100000,250000,500000,750000,1000000 --window 100000 --flip_steps "" \
  --refs "Score = success rate (%) over 100 episodes, evaluated every 20k steps. CORL after 1M steps (README): BC 0.0 ± 0.0, AWAC 0.0 ± 0.0, CQL 20.5 ± 13.2, IQL 30.3 ± 3.6, ReBRAC 64.0 ± 5.4." \
  > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 6 finished"
