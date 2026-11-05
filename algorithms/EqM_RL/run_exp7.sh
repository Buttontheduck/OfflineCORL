#!/bin/bash
# Experiment 7: Aware on four D4RL AntMaze datasets (T = 4, own critic) and Aware with an IQL critic on
# antmaze-medium-diverse-v2 (T = 0.1, expectile 0.9), 3 seeds, 1M steps (sweeps/exp7_antmaze.yaml),
# then the summary. The sweep's gpu_gate starts each further run only while the GPU has room.
# Needs no input; start it detached:
#     setsid nohup bash run_exp7.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp7_<date-time>/orchestrator.log (German time) and the sweep's launcher log.
# Result: sweep_runs/exp7_<date-time>/summary.md
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
R=sweep_runs/exp7_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R"
LOG=$R/orchestrator.log

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

log "Experiment 7 (AntMaze, Aware T = 4 and Aware with IQL critic) started in $R; launcher log: $R/sweep.launcher.log"
$HAWK run_sweep.py sweeps/exp7_antmaze.yaml > "$R/sweep.launcher.log" 2>&1
d=$(newest "sweep_runs/exp7_antmaze_2*/")
if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep FAILED to start (see $R/sweep.launcher.log)"; exit 1; fi
log "sweep finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"

P=Exp7_env_name
$HAWK diagnostics/summarize_exp5_1m.py \
  --title "Experiment 7: Aware on D4RL AntMaze (1M steps, 3 seeds)" \
  --run "umaze, T = 4=${P}antmaze-umaze-v2_temperature4.0:none" \
  --run "umaze-diverse, T = 4=${P}antmaze-umaze-diverse-v2_temperature4.0:none" \
  --run "medium-play, T = 4=${P}antmaze-medium-play-v2_temperature4.0:none" \
  --run "medium-diverse, T = 4=${P}antmaze-medium-diverse-v2_temperature4.0:none" \
  --run "medium-diverse, IQL critic, T = 0.1=${P}antmaze-medium-diverse-v2_criticiql_iql_tau0.9_temperature0.1:none" \
  --steps 100000,250000,500000,750000,1000000 --window 100000 --flip_steps "" \
  --refs "Score = success rate (%) over 100 episodes, evaluated every 20k steps. CORL after 1M steps (README), BC / AWAC / IQL: umaze 55.3 / 56.8 / 77.0; umaze-diverse 47.3 / 54.8 / 54.3; medium-play 0.0 / 0.0 / 65.8; medium-diverse 0.8 / 0.0 / 73.8." \
  > "$R/summary.md" 2> "$R/summary.err" \
  && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
log "Experiment 7 finished"
