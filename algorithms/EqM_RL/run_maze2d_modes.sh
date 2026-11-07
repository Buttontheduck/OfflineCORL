#!/bin/bash
# Maze2D action diagnostics (EqM_RL.md, "Experiment 9: results", next step): diagnostics/maze2d_modes.py
# on every Maze2D Aware run (Experiment 9: T = 0.1 and T = 1, checkpoints 500k and 1M; Experiment 8:
# T = 4, stopped at 440k-540k, its last checkpoint), each with the Gaussian AWAC run of the same
# dataset and seed (Experiment 8) as reference. Samples at K = 20, 30, 100, 1000; rollouts (100
# episodes) at K = 20, 30, 100. Evaluation only. Then the tables (summarize_maze2d_modes.py).
# Where a run has two checkpoint folders (Experiment 9's first start died on the disk quota, the
# AWAC half of Experiment 8 was restarted), the one with the most checkpoints is used.
# Start detached:  setsid nohup bash run_maze2d_modes.sh > /dev/null 2>&1 < /dev/null &
# Progress: diagnostics/out/maze2d_modes/orchestrator.log (German time); result: summary.md there.
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
C=/workspace/Trained_Models/checkpoints
O=diagnostics/out/maze2d_modes
PARALLEL=${PARALLEL:-9}
mkdir -p "$O"
LOG=$O/orchestrator.log
log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
# the folder (of those matching the glob) with the most checkpoint files
fullest() { for d in $1; do echo "$(ls "$d" | grep -c '^checkpoint_') $d"; done | sort -n | tail -1 | cut -d' ' -f2; }

JOBS=$O/jobs.txt
: > "$JOBS"
for ds in umaze medium large; do
  for s in 1 2 3; do
    AWAC=$(fullest "$C/Exp8_AWAC_env_namemaze2d-$ds-v1_seed$s-*")
    for T in 0.1 1; do
      RUN=$(fullest "$C/Exp9_Aware_env_namemaze2d-$ds-v1_temperature${T}_seed$s-*")
      echo "$RUN $O/${ds}_T${T}_s$s.json --awac $AWAC --checkpoints 500000 last" >> "$JOBS"
    done
    RUN=$(fullest "$C/Exp8_Aware-T4_env_namemaze2d-$ds-v1_seed$s-*")
    echo "$RUN $O/${ds}_T4_s$s.json --awac $AWAC --checkpoints last" >> "$JOBS"
  done
done

log "started: $(wc -l < "$JOBS") runs, $PARALLEL at a time (job list: $JOBS)"
# one log per job next to its json
xargs -P "$PARALLEL" -L 1 bash -c '
  out=$1; log=${out%.json}.log
  if '"$HAWK"' diagnostics/maze2d_modes.py "$0" "$@" > "$log" 2>&1; then echo "done   $out"; else echo "FAILED $out (see $log)"; fi
' < "$JOBS" | while read -r line; do log "$line"; done
log "all runs finished: $(ls "$O"/*.json 2>/dev/null | wc -l) json files"
$HAWK diagnostics/summarize_maze2d_modes.py "$O" > "$O/summary.md" 2> "$O/summary.err" \
  && log "summary written to $O/summary.md" || log "summary FAILED (see $O/summary.err)"
