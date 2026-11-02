#!/usr/bin/env bash
# Graph-anchor planning experiments, one job per call (see scripts/hpc/env_setup.md).
#
#   bash scripts/hpc/run_job.sh smoke          # ~15-20 min: data, checkpoints, graph, one short evaluation per environment
#   bash scripts/hpc/run_job.sh J1             # PointMaze-Large: train LeWM, then LeWM / min-over-plan / graph anchors / AP
#   bash scripts/hpc/run_job.sh J2             # PointMaze-Medium: GCIVL baseline (standard and stitched goals)
#   bash scripts/hpc/run_job.sh J3             # PointMaze-Medium: PLDM baseline, and graph anchors on PLDM
#   bash scripts/hpc/run_job.sh J4             # third seed (1) for the main table, Two-Room and PointMaze-Medium
#   bash scripts/hpc/run_job.sh J5             # ablations on PointMaze-Medium and Two-Room
#   QUICK=1 bash scripts/hpc/run_job.sh J3     # plumbing test: a few training steps, 2 episodes per evaluation
#
# Each job writes to $OUT_ROOT/<job>/ (default out/hpc/<job>/), ends with a PASS/FAIL line in <job>/job.log and packs
# everything needed back into $OUT_ROOT/<job>_results.tgz (logs and result files, no graph or video files).
# Finished runs are skipped, so after a time limit or a crash the same command simply continues.
#
# Environment: GWM_ENV (conda env, default gwm), STABLEWM_HOME (data/checkpoint cache, default ~/.stable_worldmodel),
# OUT_ROOT, QUICK=1. One GPU per job; jobs can run at the same time on different GPUs (CUDA_VISIBLE_DEVICES).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="${1:?usage: bash scripts/hpc/run_job.sh <smoke|J1|J2|J3|J4|J5>}"
source "$ROOT/env/conda.sh"
CONDA_BASE="$(project_conda_root)" || exit 1
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${GWM_ENV:-gwm}" || { echo "conda env ${GWM_ENV:-gwm} missing: run scripts/hpc/env_setup.sh"; exit 1; }

Q="${QUICK:-0}"
export JEPAWM_HOME="$ROOT" SWM_PY="$(command -v python)"
export PYTHONPATH="$ROOT/code/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" SDL_VIDEODRIVER=dummy
source "$ROOT/scripts/hpc/egl.sh"
if [[ "$Q" == 1 ]]; then OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc_quick}"; else OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc}"; fi
O="$OUT_ROOT/$JOB"; mkdir -p "$O" "$STABLEWM_HOME"
L="$O/job.log"
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$L"; }
R="$ROOT/scripts/pod/run_ga_env.sh"; B="$ROOT/scripts/pod/run_baselines.sh"
SW="$ROOT/code/third_party/stable-worldmodel"; P="$ROOT/code/lewm_explorations/paa"
q() { if [[ "$Q" == 1 ]]; then echo "$1"; else echo "$2"; fi; }  # q <quick value> <full value>
if [[ "$Q" == 1 ]]; then export EVX="eval.num_eval=2" STEPS=30 STEPS_WM=30; else export EVX=""; fi
waitf() { while [[ ! -f "$1" ]]; do [[ -f "$(dirname "$1")/ABORT" ]] && return 1; sleep 30; done; }

finish() {  # summary table, PASS/FAIL line, results archive
    local n f
    n=$(cat "$O"/*/summary.txt 2>/dev/null | grep -c 'success_rate=' || true)
    f=$(cat "$O"/*/queue.log 2>/dev/null | grep -cE 'FAILED|ABORT|rc=[1-9]' || true)
    log "results ($n runs):"
    cat "$O"/*/summary.txt 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^success_rate=/) print "  " $1, $i}' | tee -a "$L"
    if [[ "$n" -gt 0 && "$f" -eq 0 ]]; then log "JOB $JOB PASS ($n runs)"; else log "JOB $JOB FAIL ($n runs, $f failures; see */queue.log)"; fi
    tar czf "$OUT_ROOT/${JOB}_results.tgz" -C "$OUT_ROOT" --exclude='nodes.npz' --exclude='*.mp4' "$JOB" \
        && log "packed $OUT_ROOT/${JOB}_results.tgz  <- please send this file back"
}

log "JOB $JOB start (QUICK=$Q) on $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) | git $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
case "$JOB" in
smoke)  # environment, data, checkpoints, graph and planner on Two-Room and PointMaze-Medium, 2 episodes per evaluation
    bash "$ROOT/scripts/hpc/env_setup.sh" check >> "$L" 2>&1 || { log "env check FAILED (see job.log)"; finish; exit 1; }
    export EVX="eval.num_eval=2"
    OFFS=25 SPANS=6 bash "$R" tworoom 0 "$O/tworoom" all >> "$O/lane.log" 2>&1
    OFFS=50 SPANS=6 bash "$R" pmm 0 "$O/pmm" all >> "$O/lane.log" 2>&1
    OFFS=50 bash "$R" pmm 0 "$O/pmm" ap >> "$O/lane.log" 2>&1
    ;;
J1)  # PointMaze-Large (OGBench, 1,000 x 1,000-step episodes, LeWAM/lewam-pointmaze-large): no public LeWM, so we train one
    D="$O/train"; mkdir -p "$D" "$O/pml"
    cd "$SW"
    python "$P/fetch_lewm_data.py" pml > "$D/fetch.log" 2>&1 || { log "fetch FAILED: $(tail -2 "$D/fetch.log")"; finish; exit 1; }
    python -c "
import h5py, hdf5plugin; f = h5py.File('$STABLEWM_HOME/datasets/pointmaze_large.h5', 'r')
print({k: f[k].shape for k in f.keys()})" > "$D/h5_keys.txt" 2>&1; log "data: $(head -c 300 "$D/h5_keys.txt")"
    python scripts/plan/eval_wm.py --config-name=pointmaze_large policy=random eval.num_eval=2 eval.goal_offset_steps=25 \
        eval.eval_budget=50 > "$D/env_smoke.log" 2>&1
    log "env smoke (random policy) rc=$? $(grep -a '\[summary\]' "$D/env_smoke.log" | grep -oE 'success_rate=[0-9.]+')"
    if [[ ! -f "$D/train.done" ]]; then
        t0=$(date +%s)
        python scripts/train/lewm.py data=pointmaze_large output_model_name=lewm_pml subdir=pml \
            trainer.max_epochs="$(q 1 "${EP:-10}")" +trainer.limit_train_batches="$(q 30 "${NB:-3000}")" \
            +trainer.limit_val_batches="$(q 5 20)" > "$D/train.log" 2>&1
        rc=$?; log "train lewm_pml rc=$rc $(( $(date +%s) - t0 ))s $(grep -aE '^[A-Za-z.]*(Error|Exception)' "$D/train.log" | tail -1)"
        [[ $rc == 0 ]] || { echo "train rc=$rc" >> "$D/queue.log"; finish; exit 1; }
        touch "$D/train.done"
    fi
    cd "$ROOT"
    echo pointmaze_large.h5 > "$O/pml/dataset_name"
    export PML_CKPT=lewm_pml
    S="$(q 0 '0 42')"
    ( OFFS="$(q 50 '50 100 200 400')" SPANS=6 bash "$R" pml "$S" "$O/pml" all >> "$O/laneA.log" 2>&1 ) &
    ( waitf "$O/pml/READY" && OFFS="$(q 50 '50 100 200 400')" bash "$R" pml "$S" "$O/pml" ap >> "$O/laneB.log" 2>&1 ) &
    ( waitf "$O/pml/READY" && sleep 60 && XB="$(q 100 400)" bash "$R" pml "$S" "$O/pml" cross >> "$O/laneC.log" 2>&1 ) &
    wait
    ;;
J2)  # GCIVL on PointMaze-Medium (value then policy training), standard goals d50-200 and stitched goals (budget 400)
    XB="$(q 100 400)" XSEEDS="$(q 0 '0 42')" OFFS="$(q 50 '50 100 150 200')" bash "$B" pmm gcivl "$O/bl_pmm" >> "$O/lane.log" 2>&1
    ;;
J3)  # PLDM on PointMaze-Medium: trained here, stitched goals; then graph anchors on PLDM (d100, d200, stitched)
    XB="$(q 100 400)" XSEEDS="$(q 0 '0 42')" OFFS="" bash "$B" pmm pldm "$O/bl_pmm" >> "$O/laneA.log" 2>&1
    if [[ -f "$O/bl_pmm/train_pldm.done" ]]; then
        mkdir -p "$O/pldm_pmm"; echo d4rl_pointmaze_medium_expert.h5 > "$O/pldm_pmm/dataset_name"
        WM=bl_pmm_pldm OFFS="$(q 100 '100 200')" SPANS=6 bash "$R" pmm "$(q 0 '0 42')" "$O/pldm_pmm" all >> "$O/laneB.log" 2>&1
        WM=bl_pmm_pldm XB="$(q 100 400)" XM=ga bash "$R" pmm "$(q 0 '0 42')" "$O/pldm_pmm" cross >> "$O/laneB.log" 2>&1
    else
        log "PLDM training did not finish (see bl_pmm/train_pldm.log)"
    fi
    ;;
J4)  # seed 1 for every main-table row: LeWM, min-over-plan, graph anchors, AP; standard and stitched goals
    ( OFFS="$(q 25 '25 50 75')" SPANS=6 bash "$R" tworoom 1 "$O/tworoom" all >> "$O/laneA.log" 2>&1
      OFFS="$(q 25 '25 50 75')" bash "$R" tworoom 1 "$O/tworoom" ap >> "$O/laneA.log" 2>&1
      XB="$(q 50 200)" bash "$R" tworoom 1 "$O/tworoom" cross >> "$O/laneA.log" 2>&1 ) &
    ( OFFS="$(q 50 '50 100 150 200')" SPANS=6 bash "$R" pmm 1 "$O/pmm" all >> "$O/laneB.log" 2>&1 ) &
    ( sleep 120; waitf "$O/pmm/READY" && OFFS="$(q 50 '50 100 150 200')" bash "$R" pmm 1 "$O/pmm" ap >> "$O/laneC.log" 2>&1 \
      && XB="$(q 100 400)" bash "$R" pmm 1 "$O/pmm" cross >> "$O/laneC.log" 2>&1 ) &
    wait
    ;;
J5)  # ablations (graph anchors, span 6 unless varied): PointMaze-Medium d200 and Two-Room d75
    PO="$(q 50 200)"; TO="$(q 25 75)"; S2="$(q 0 '0 42')"; XB0="$(q 100 400)"; X0="$EVX"
    abl() {  # tag extra-overrides env offset seeds [spans]
        TAG="_$1" EVX="$X0 $2" OFFS="$4" SPANS="${6:-6}" bash "$R" "$3" "$5" "$O/$3" anchor >> "$O/abl_$1.log" 2>&1
    }
    # graphs first (one setup per environment), then three lanes
    OFFS="$PO" SPANS=6 bash "$R" pmm 0 "$O/pmm" all >> "$O/setup.log" 2>&1
    OFFS="$TO" SPANS=6 bash "$R" tworoom 0 "$O/tworoom" all >> "$O/setup.log" 2>&1
    (   # A1: temporal edges only (no cross-episode k-NN edges): standard and stitched goals
        abl knn0 "+paa_knn=0" pmm "$PO" "$S2"
        TAG=_knn0 EVX="$X0 +paa_knn=0" XB="$XB0" XM=ga bash "$R" pmm "$S2" "$O/pmm" cross >> "$O/abl_knn0.log" 2>&1
        # A3: graph built from a fraction of the episodes
        for f in $(q 0.25 '0.1 0.25 0.5'); do abl "f$f" "+paa_frac=$f" pmm "$PO" "$S2"; done ) &
    (   # A2: anchor span sweep
        abl span "" pmm "$PO" "$S2" "$(q 2 '2 4 8 12')"
        abl span "" tworoom "$TO" "$S2" "$(q 2 '2 4 8 12')" ) &
    (   # A4: k-NN size and edge threshold quantile
        for k in $(q 5 '5 20'); do abl "k$k" "+paa_knn=$k" pmm "$PO" 0; done
        for t in $(q 0.25 '0.25 0.75'); do abl "q$t" "+paa_thr_q=$t" pmm "$PO" 0; done
        # A5: LeWM with 3x the CEM samples (compute-matched baseline), PointMaze d200 and Two-Room d75
        TAG=_cem3x EVX="$X0 solver.num_samples=900" OFFS="$PO" bash "$R" pmm 0 "$O/pmm" base >> "$O/abl_cem3x.log" 2>&1
        TAG=_cem3x EVX="$X0 solver.num_samples=900" OFFS="$TO" bash "$R" tworoom 0 "$O/tworoom" base >> "$O/abl_cem3x.log" 2>&1 ) &
    wait
    ;;
*) echo "unknown job $JOB (smoke | J1 | J2 | J3 | J4 | J5)"; exit 1 ;;
esac
finish
