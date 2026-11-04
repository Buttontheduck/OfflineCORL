#!/bin/bash
# Experiment 4, priority 1 and priority 2, unattended (EqM_RL.md, "Planned next experiments").
# Runs everything in the right order and needs no input; start it detached, e.g.
#     setsid nohup bash run_exp4.sh > /dev/null 2>&1 < /dev/null &
# Progress: sweep_runs/exp4_<date-time>/orchestrator.log (times in German time).
#
# Branches, all started at once:
#   P1      evaluation-only jobs on existing checkpoints, 6 at a time: interventions (screening, 20
#           episodes, and confirmation, 100 new episodes, for every treatment, so nothing is
#           selected by hand), handoff, then the Minari flip counts (eagle env).
#   BC      sweep exp4_fit_bc (expert-only BC, Gaussian and EqM).
#   IQL     sweep exp4_iql -> IQL weights -> sweep exp4_fit_iql.
#   AWARE   wait for checkpoint_310000.pt of the 1M-step runs (sweeps/exp4_aware_1m.yaml, already
#           running) -> Aware weights -> sweep exp4_fit_aware; once the IQL weights exist, the
#           matched-concentration weights -> sweep exp4_fit_aware_matched.
# Before each training sweep it waits for enough free GPU memory. A failed step is logged and
# only skips the steps that need its output.
set -u
cd "$(dirname "$0")"
HAWK=/workspace/miniconda3/envs/hawk/bin/python
EAGLE=/workspace/miniconda3/envs/eagle/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}
C=/workspace/Trained_Models/checkpoints
W=/workspace/Trained_Models/weights
O=diagnostics/out
R=sweep_runs/exp4_$(date +%Y%m%d-%H%M%S)
mkdir -p "$R/p1" "$W" "$O/interv" "$O/handoff" "$O/roll"
LOG=$R/orchestrator.log
GPU_FREE_MB=${GPU_FREE_MB:-7000}       # free GPU memory needed before a 6-run sweep starts
WAIT_310K_H=${WAIT_310K_H:-12}         # give up waiting for checkpoint_310000.pt after this many hours

log() { echo "[$(TZ=Europe/Berlin date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
newest() { ls -td $1 2>/dev/null | head -1; }

wait_gpu() {  # wait until the GPU has $1 MB free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$1" ] && return 0
    sleep 60
  done
}

sweep() {  # run sweeps/$1.yaml in the foreground; true if every run finished as 'done'
  wait_gpu "$GPU_FREE_MB"
  log "sweep $1: start"
  $HAWK run_sweep.py "sweeps/$1.yaml" > "$R/$1.launcher.log" 2>&1
  local d; d=$(newest "sweep_runs/$1_2*/")
  if [ -z "$d" ] || [ ! -f "$d/runs.csv" ]; then log "sweep $1: FAILED to start (see $R/$1.launcher.log)"; return 1; fi
  log "sweep $1: finished ($(tail -n +2 "$d/runs.csv" | cut -d, -f3 | sort | uniq -c | xargs)) in $d"
  [ "$(tail -n +2 "$d/runs.csv" | cut -d, -f3 | grep -v -c '^done$')" -eq 0 ]
}

weights() {  # make_weights.py with logging; $1 = output name, rest = arguments
  local name=$1; shift
  if $HAWK Aware/make_weights.py "$@" --out "$W/$name.npz" >> "$R/make_weights.log" 2>&1; then
    log "weights $name: $(grep -A2 "^$name.npz" "$R/make_weights.log" | tail -2 | grep 'effective' | xargs)"
  else
    log "weights $name: FAILED (see $R/make_weights.log)"; return 1
  fi
}

log "Experiment 4 orchestrator started in $R"

# ------------------------------------------------------------------ P1: evaluation only
(
  J=$R/p1/jobs.txt; : > "$J"
  for s in 1 2 3; do
    AME=$(newest "$C/AWAC_env_namehalfcheetah-medium-expert-v2_seed$s-*")
    T1=$(newest "$C/Aware_env_namehalfcheetah-medium-expert-v2_seed$s-*")
    T3=$(newest "$C/Aware_temperature0.3333_seed$s-*")
    BC=$(newest "$C/Aware_weightingbc_seed$s-*")
    for pair in "me_T1:$T1" "me_T0.33:$T3" "me_bc:$BC"; do
      n=${pair%%:*}; d=${pair#*:}
      echo "$HAWK diagnostics/interventions.py $d $O/interv/${n}_s$s.json --awac $AME --save_steps > $R/p1/interv_${n}_s$s.log 2>&1 || echo interv_${n}_s$s >> $R/p1/failed.txt" >> "$J"
      echo "$HAWK diagnostics/interventions.py $d $O/interv/${n}_s${s}_confirm.json --awac $AME --n_eps 100 --seed0 1000 --save_steps > $R/p1/interv_${n}_s${s}_confirm.log 2>&1 || echo interv_${n}_s${s}_confirm >> $R/p1/failed.txt" >> "$J"
    done
    for pair in "me_T1:$T1" "me_T0.33:$T3"; do
      n=${pair%%:*}; d=${pair#*:}
      echo "$HAWK diagnostics/handoff.py $d $AME $O/handoff/${n}_s$s.json > $R/p1/handoff_${n}_s$s.log 2>&1 || echo handoff_${n}_s$s >> $R/p1/failed.txt" >> "$J"
    done
  done
  log "P1: $(wc -l < "$J") D4RL jobs, 6 at a time"
  xargs -P 6 -d '\n' -I{} bash -c '{}' < "$J"
  for s in 1 2 3; do
    $EAGLE diagnostics/rollout.py "$(newest "$C/AWAC_env_namemedium-v0_seed$s-*")" "$O/roll/medium_awac_s$s.json" --awac --n_eps 20 > "$R/p1/roll_awac_s$s.log" 2>&1 || echo "roll_awac_s$s" >> "$R/p1/failed.txt"
    $EAGLE diagnostics/rollout.py "$(newest "$C/Aware_env_namemedium-v0_scale_gradient8.0_weightingbc_seed$s-*")" "$O/roll/medium_bc_s$s.json" --n_eps 20 > "$R/p1/roll_bc_s$s.log" 2>&1 || echo "roll_bc_s$s" >> "$R/p1/failed.txt"
  done
  if [ -s "$R/p1/failed.txt" ]; then log "P1: done, FAILED: $(xargs < "$R/p1/failed.txt")"; else log "P1: done, all jobs succeeded"; fi
) &

# ------------------------------------------------------------------ BC: expert-only BC
( sweep exp4_fit_bc ) &

# ------------------------------------------------------------------ IQL: critics -> weights -> fits
(
  ok=1
  if sweep exp4_iql; then
    for s in 1 2 3; do
      weights "iql_seed$s" --source iql --run_dir "$(newest "$C/IQL_seed$s-*")" || ok=0
    done
  else
    ok=0; log "IQL critics failed: no IQL weights"
  fi
  if [ $ok -eq 1 ]; then touch "$R/iql_weights.done"; sweep exp4_fit_iql; else touch "$R/iql_weights.failed"; fi
) &

# ------------------------------------------------------------------ AWARE: 310k checkpoint -> weights -> fits
(
  deadline=$(( $(date +%s) + WAIT_310K_H * 3600 ))
  ok=1
  for s in 1 2 3; do
    d=$(newest "$C/Aware_1M_seed$s-*")
    # checkpoint_311000.pt exists only after checkpoint_310000.pt has been written completely
    while [ ! -f "$d/checkpoint_311000.pt" ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 60; done
    if [ ! -f "$d/checkpoint_310000.pt" ]; then log "AWARE: no checkpoint_310000.pt for seed $s"; ok=0; continue; fi
    weights "aware_seed$s" --source aware --run_dir "$d" --checkpoint checkpoint_310000.pt || ok=0
  done
  if [ $ok -eq 1 ]; then
    ( sweep exp4_fit_aware ) &
    while [ ! -f "$R/iql_weights.done" ] && [ ! -f "$R/iql_weights.failed" ]; do sleep 60; done
    if [ -f "$R/iql_weights.done" ]; then
      mok=1
      for s in 1 2 3; do
        weights "aware_matched_seed$s" --source aware --run_dir "$(newest "$C/Aware_1M_seed$s-*")" \
          --checkpoint checkpoint_310000.pt --match_share "$W/iql_seed$s.npz" || mok=0
      done
      [ $mok -eq 1 ] && sweep exp4_fit_aware_matched || log "matched refit skipped: a matched weight file failed"
    else
      log "matched refit skipped: no IQL weights"
    fi
    wait
  else
    log "AWARE: skipped exp4_fit_aware and the matched refit"
  fi
) &

wait
if [ -f diagnostics/summarize_exp4.py ]; then
  $HAWK diagnostics/summarize_exp4.py --orchestrator_dir "$R" > "$R/summary.md" 2> "$R/summary.err" \
    && log "summary written to $R/summary.md" || log "summary FAILED (see $R/summary.err)"
fi
log "Experiment 4 orchestrator finished (the 1M-step Aware runs go on until they are done)"
