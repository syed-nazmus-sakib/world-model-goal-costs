#!/usr/bin/env bash
# One command for everything: environment, smoke test, jobs J1-J5 spread over the node's GPUs, one archive at the end.
#
#   bash scripts/hpc/run_all.sh
#
# Result: out/hpc/ALL_results.tgz (logs and result files of every job, a few MB) and a PASS/FAIL table at the end of
# out/hpc/run_all.log. Run the same command again after a time limit or crash: finished jobs are skipped and unfinished
# ones continue where they stopped. If the smoke test fails, the jobs are not started and the archive holds its logs.
#
# Options (environment variables):
#   GPUS="0 1 2 3"   GPUs to use (default: all GPUs that nvidia-smi lists, or CUDA_VISIBLE_DEVICES if set)
#   MAXPAR=2         at most this many jobs at the same time (each needs ~64 GB host RAM; default: one per GPU)
#   JOBS="J1 J2"     subset of jobs (default: J1 J2 J3 J4 J5; the follow-ups J6 J7 J8 run with JOBS="J6 J7 J8")
#   PLAN_ONLY=1      print which GPU runs which job, then exit
#   QUICK=1          plumbing test: a few training steps, 2 episodes per evaluation (written to out/hpc_quick/)
#   plus the options of env_setup.sh and run_job.sh (GWM_ENV, STABLEWM_HOME, CONDA_ROOT, ...)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H="$ROOT/scripts/hpc"
Q="${QUICK:-0}"
if [[ "$Q" == 1 ]]; then export OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc_quick}"; else export OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc}"; fi
mkdir -p "$OUT_ROOT"
L="$OUT_ROOT/run_all.log"
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$L"; }
status() { grep -aoE "JOB $1 (PASS|FAIL)[^)]*\)" "$OUT_ROOT/$1/job.log" 2>/dev/null | tail -1; }
JOBS="${JOBS:-J1 J2 J3 J4 J5}"

pack() {
    log "summary:"
    for j in smoke $JOBS; do st="$(status "$j")"; log "  $j: ${st:-not finished (run the same command again to continue)}"; done
    local items=(run_all.log)
    for j in smoke $JOBS; do [[ -d "$OUT_ROOT/$j" ]] && items+=("$j"); done
    tar czf "$OUT_ROOT/ALL_results.tgz" -C "$OUT_ROOT" --exclude='nodes.npz' --exclude='*.mp4' "${items[@]}" \
        && log "DONE. Please send back this one file: $OUT_ROOT/ALL_results.tgz ($(du -h "$OUT_ROOT/ALL_results.tgz" | cut -f1))"
}

# ---------------------------------------------------------------------------------------------------- GPU plan
if [[ -n "${GPUS:-}" ]]; then read -r -a G <<< "$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=',' read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
else read -r -a G <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"; fi
[[ ${#G[@]} -gt 0 ]] || G=(0)
NL=${#G[@]}; [[ -n "${MAXPAR:-}" && "${MAXPAR}" -lt "$NL" ]] && NL="$MAXPAR"
hours() { case $1 in J1) echo 10 ;; J5) echo 7 ;; J4) echo 5 ;; J2|J3|J6|J7) echo 4 ;; *) echo 3 ;; esac; }
declare -a LANE LOAD
for ((i = 0; i < NL; i++)); do LANE[i]=""; LOAD[i]=0; done
for j in $(for j in $JOBS; do echo "$(hours "$j") $j"; done | sort -rn | awk '{print $2}'); do  # longest job to the least loaded lane
    b=0; for ((i = 1; i < NL; i++)); do [[ ${LOAD[i]} -lt ${LOAD[b]} ]] && b=$i; done
    LANE[b]="${LANE[b]} $j"; LOAD[b]=$(( LOAD[b] + $(hours "$j") ))
done
log "run_all start (QUICK=$Q) on $(hostname): GPU slots [${G[*]}], $NL job(s) at a time"
for ((i = 0; i < NL; i++)); do log "  GPU ${G[i]}:${LANE[i]}  (~${LOAD[i]} A100 h)"; done
[[ "${PLAN_ONLY:-0}" == 1 ]] && exit 0
exec 9> "$OUT_ROOT/.run_all.lock"   # two copies writing the same out/ directory overwrite each other's logs
if command -v flock > /dev/null && ! flock -n 9; then
    log "another run_all.sh is already running for $OUT_ROOT: wait for it to finish (or stop it) before starting again"; exit 1
fi

# --------------------------------------------------------------------------------------------- 1. environment
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)" || { log "Conda not found: set CONDA_ROOT"; pack; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda run -n "${GWM_ENV:-gwm}" python -c "import stable_worldmodel, ogbench, gymnasium_robotics" >/dev/null 2>&1; then
    log "environment ${GWM_ENV:-gwm} exists; checking it"
    bash "$H/env_setup.sh" check >> "$L" 2>&1 || { log "environment check FAILED (see run_all.log)"; pack; exit 1; }
else
    log "creating environment ${GWM_ENV:-gwm} (about 5-10 min)"
    bash "$H/env_setup.sh" >> "$L" 2>&1 || { log "environment setup FAILED (see run_all.log)"; pack; exit 1; }
fi
log "environment OK"

# ---------------------------------------------------------------------------------------------- 2. smoke test
if [[ "$(status smoke)" != *PASS* ]]; then
    log "smoke test (about 15-20 min)"
    CUDA_VISIBLE_DEVICES="${G[0]}" bash "$H/run_job.sh" smoke > "$OUT_ROOT/smoke.out" 2>&1
fi
if [[ "$(status smoke)" != *PASS* ]]; then log "smoke test FAILED: jobs not started"; pack; exit 1; fi
log "smoke test PASS"

# -------------------------------------------------------------------------------------------------- 3. jobs
for ((i = 0; i < NL; i++)); do
    (
        for j in ${LANE[i]}; do
            if [[ "$(status "$j")" == *PASS* ]]; then log "$j already PASS, skipped"; continue; fi
            log "$j start on GPU ${G[i]}"
            CUDA_VISIBLE_DEVICES="${G[i]}" bash "$H/run_job.sh" "$j" > "$OUT_ROOT/$j.out" 2>&1
            log "$j: $(status "$j")"
        done
    ) &
    sleep 60  # stagger the first downloads
done
wait
pack
