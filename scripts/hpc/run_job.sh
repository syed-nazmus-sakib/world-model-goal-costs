#!/usr/bin/env bash
# Graph-anchor planning experiments, one job per call (see scripts/hpc/env_setup.md).
#
#   bash scripts/hpc/run_job.sh smoke          # ~15-20 min: data, checkpoints, graph, one short evaluation per environment
#   bash scripts/hpc/run_job.sh J1             # PointMaze-Large: train LeWM, then LeWM / min-over-plan / graph anchors / AP
#   bash scripts/hpc/run_job.sh J2             # PointMaze-Medium: GCIVL baseline (standard and stitched goals)
#   bash scripts/hpc/run_job.sh J3             # PointMaze-Medium: PLDM baseline, and graph anchors on PLDM
#   bash scripts/hpc/run_job.sh J4             # third seed (1) for the main table, Two-Room and PointMaze-Medium
#   bash scripts/hpc/run_job.sh J5             # ablations on PointMaze-Medium and Two-Room
#   bash scripts/hpc/run_job.sh J6             # PointMaze-Medium: graph waypoints given to GCBC (graph + policy, no planner)
#   bash scripts/hpc/run_job.sh J7             # PointMaze-Medium: PLDM with graph anchors AND with AP (J3 + AP)
#   bash scripts/hpc/run_job.sh J8             # PointMaze-Medium: AP with its memory built from random-play data
#   bash scripts/hpc/run_job.sh J9             # 3-seed robustness: target x search, AP told a wrong distance, random data,
#                                              #   graph + GCBC with diagnostics, coverage (PointMaze-Medium, Two-Room)
#   bash scripts/hpc/run_job.sh J10            # PointMaze-Large (needs J1's model): AP told a wrong distance, GCBC (+ graph)
#   bash scripts/hpc/run_job.sh J11            # PointMaze-Large (needs J1's model): graph / AP memory from OGBench stitch data
#   bash scripts/hpc/run_job.sh J12            # PointMaze-Medium: GCBC / GCIVL with 4x and PLDM with 5x the training budget
#   bash scripts/hpc/run_job.sh J13            # planning time per decision: LeWM, min-over-plan, graph anchors, AP
#   QUICK=1 bash scripts/hpc/run_job.sh J3     # plumbing test: a few training steps, 2 episodes per evaluation
#
# Each job writes to $OUT_ROOT/<job>/ (default out/hpc/<job>/), ends with a PASS/FAIL line in <job>/job.log and packs
# everything needed back into $OUT_ROOT/<job>_results.tgz (logs and result files, no graph or video files).
# Finished runs are skipped, so after a time limit or a crash the same command simply continues.
#
# Environment: GWM_ENV (conda env, default gwm), STABLEWM_HOME (data/checkpoint cache, default ~/.stable_worldmodel),
# OUT_ROOT, QUICK=1, GWM_NOCONDA=1 (use the python on PATH instead of the conda env, e.g. on Colab).
# One GPU per job; jobs can run at the same time on different GPUs (CUDA_VISIBLE_DEVICES).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="${1:?usage: bash scripts/hpc/run_job.sh <smoke|J1|...|J13>}"
if [[ "${GWM_NOCONDA:-0}" != 1 ]]; then
    source "$ROOT/env/conda.sh"
    CONDA_BASE="$(project_conda_root)" || exit 1
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${GWM_ENV:-gwm}" || { echo "conda env ${GWM_ENV:-gwm} missing: run scripts/hpc/env_setup.sh"; exit 1; }
fi
command -v python > /dev/null || python() { python3 "$@"; }

Q="${QUICK:-0}"
export JEPAWM_HOME="$ROOT" SWM_PY="$(command -v python || command -v python3)"
export PYTHONPATH="$ROOT/code/third_party/stable-worldmodel${PYTHONPATH:+:$PYTHONPATH}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable_worldmodel}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" SDL_VIDEODRIVER=dummy
source "$ROOT/scripts/hpc/egl.sh"
if [[ "$Q" == 1 ]]; then OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc_quick}"; else OUT_ROOT="${OUT_ROOT:-$ROOT/out/hpc}"; fi
mkdir -p "$OUT_ROOT" "$STABLEWM_HOME"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"; STABLEWM_HOME="$(cd "$STABLEWM_HOME" && pwd)"   # absolute: lanes cd before writing
export OUT_ROOT STABLEWM_HOME
O="$OUT_ROOT/$JOB"; mkdir -p "$O"
L="$O/job.log"
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$L"; }
R="$ROOT/scripts/pod/run_ga_env.sh"; B="$ROOT/scripts/pod/run_baselines.sh"
SW="$ROOT/code/third_party/stable-worldmodel"; P="$ROOT/code/lewm_explorations/paa"
q() { if [[ "$Q" == 1 ]]; then echo "$1"; else echo "$2"; fi; }  # q <quick value> <full value>
if [[ "$Q" == 1 ]]; then export EVX="eval.num_eval=2" STEPS=30 STEPS_WM=30; else export EVX=""; fi
waitf() { while [[ ! -f "$1" ]]; do [[ -f "$(dirname "$1")/ABORT" ]] && return 1; sleep 30; done; }

keep_last() {  # training saves one .pt per epoch; the loader needs exactly one: keep the last, move the others away
    local K="$1"
    if [[ $(ls "$K"/*.pt 2>/dev/null | wc -l) -gt 1 ]]; then
        mkdir -p "$K/older_epochs"; ls "$K"/*.pt | sort -V | head -n -1 | xargs -I{} mv {} "$K/older_epochs/"
        log "checkpoint: kept $(ls "$K"/*.pt), older epochs moved to $K/older_epochs"
    fi
}
pml_model() {  # J10/J11: our PointMaze-Large LeWM is trained by J1; wait for it when J1 runs at the same time (max 12 h)
    # J1 may have run in another output folder (e.g. out/hpc while this batch writes to out/hpc2): any finished J1 counts
    local t=0
    j1_done() { local f; for f in "$OUT_ROOT"/J1/train/train.done "$ROOT"/out/hpc*/J1/train/train.done; do [[ -f "$f" ]] && return 0; done; return 1; }
    until j1_done; do
        (( t++ > 720 )) && { log "no trained PointMaze-Large model after 12 h: run J1 first"; return 1; }
        sleep 60
    done
    keep_last "$STABLEWM_HOME/checkpoints/lewm_pml"
    export PML_CKPT=lewm_pml
}

finish() {  # summary table, PASS/FAIL line, results archive
    local n f
    n=$(cat "$O"/*/summary.txt 2>/dev/null | grep -c 'success_rate=' || true)
    # failures since this start only (a resumed job keeps the lines of earlier, interrupted attempts)
    f=$(cat "$O"/*/queue.log 2>/dev/null | awk -v t="$T0" 'substr($0, 1, 19) >= t' | grep -cE 'FAILED|ABORT|rc=[1-9]' || true)
    log "results ($n runs):"
    cat "$O"/*/summary.txt 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^success_rate=/) print "  " $1, $i}' | tee -a "$L"
    if [[ "$n" -gt 0 && "$f" -eq 0 ]]; then log "JOB $JOB PASS ($n runs)"; else log "JOB $JOB FAIL ($n runs, $f failures; see */queue.log)"; fi
    tar czf "$OUT_ROOT/${JOB}_results.tgz" -C "$OUT_ROOT" --exclude='nodes.npz' --exclude='*.mp4' "$JOB" \
        && log "packed $OUT_ROOT/${JOB}_results.tgz  <- please send this file back"
}

T0="$(date -u +%F' '%T)"
rm -f "$O"/*/ABORT   # a stale abort marker from an earlier attempt would stop the waiting lanes of this one
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
        [[ $rc == 0 ]] || { echo "$(date -u +%F' '%T) train rc=$rc" >> "$D/queue.log"; finish; exit 1; }
        touch "$D/train.done"
    fi
    keep_last "$STABLEWM_HOME/checkpoints/lewm_pml"
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
J6)  # graph + policy baseline (reviewer question "SoRB in latent space?"): GCBC trained here, evaluated plain and with
     # graph waypoints (the anchor 6 hops ahead on the LeWM latent graph as its goal); standard d50-200 + stitched goals.
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/laneA.log" 2>&1   # LeWM nodes for the graph (same as our planner's)
    XB="$(q 100 400)" XSEEDS="$(q 0 '0 42')" OFFS="$(q 50 '50 100 150 200')" GNODES="$O/pmm/nodes.npz" \
        GWM=yjang43/lp2-pointmaze-medium bash "$B" pmm "gcbc gcbcg" "$O/bl_pmm" >> "$O/laneB.log" 2>&1
    ;;
J7)  # PLDM on PointMaze-Medium (trained here): PLDM alone, + graph anchors, + AP; d100/d200 and stitched goals, 2 seeds
    XB="$(q 100 400)" XSEEDS="$(q 0 '0 42')" OFFS="" bash "$B" pmm pldm "$O/bl_pmm" >> "$O/laneA.log" 2>&1
    if [[ -f "$O/bl_pmm/train_pldm.done" ]]; then
        mkdir -p "$O/pldm_pmm"; echo d4rl_pointmaze_medium_expert.h5 > "$O/pldm_pmm/dataset_name"
        S2="$(q 0 '0 42')"; export WM=bl_pmm_pldm
        ( OFFS="$(q 100 '100 200')" SPANS=6 bash "$R" pmm "$S2" "$O/pldm_pmm" all >> "$O/laneB.log" 2>&1 ) &
        ( waitf "$O/pldm_pmm/READY" && OFFS="$(q 100 '100 200')" bash "$R" pmm "$S2" "$O/pldm_pmm" ap >> "$O/laneC.log" 2>&1 \
          && XB="$(q 100 400)" XM="ga ap" bash "$R" pmm "$S2" "$O/pldm_pmm" cross >> "$O/laneC.log" 2>&1 ) &
        wait
    else
        log "PLDM training did not finish (see bl_pmm/train_pldm.log)"
    fi
    ;;
J8)  # AP whose retrieval memory is built from random-play data (graph anchors on the same data: 94/88/92/88, stitched 96/94)
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/lane.log" 2>&1     # expert data, LeWM, expert nodes (offset check)
    D="$O/rand_pmm"; mkdir -p "$D"
    RH="$(ls "$STABLEWM_HOME/datasets/" | grep -i 'pointmaze_medium_random.*h5' | head -1)"
    [[ -n "$RH" ]] || { echo "no random dataset" >> "$D/queue.log"; finish; exit 1; }
    if [[ ! -f "$D/nodes.npz" ]]; then
        (cd "$SW" && python "$P/paa_nodes.py" --out "$D/nodes.npz" --model yjang43/lp2-pointmaze-medium --dataset "$RH" \
            --state_key proprio --workers 8 > "$D/nodes.log" 2>&1) || { echo "random nodes FAILED" >> "$D/queue.log"; finish; exit 1; }
    fi
    echo d4rl_pointmaze_medium_expert.h5 > "$D/dataset_name"; touch "$D/READY"
    export TAG=_rand EVX="$EVX +paa_noexclude=true" OKNODES="$O/pmm/nodes.npz"
    S2="$(q 0 '0 42')"
    ( OFFS="$(q 50 '50 100 150 200')" bash "$R" pmm "$S2" "$D" ap >> "$O/laneA.log" 2>&1 ) &
    ( XB="$(q 100 400)" XM=ap bash "$R" pmm "$S2" "$D" cross >> "$O/laneB.log" 2>&1 ) &
    wait
    ;;
J9)  # Tier-1 robustness, seeds 0/42/1 (PointMaze-Medium d200 / Two-Room d75 unless stated):
     #   target x search 2x2 (adds graph anchors with the released H5 search), AP told 0.5x / 1x / 2x the true distance,
     #   random-play graph and AP memory (seed 1), graph + GCBC with waypoint diagnostics (span 6 and 2),
     #   coverage: every node in one doorway cell of PointMaze-Medium removed (cell row 3 / col 2: x in [-2,-1], y in [0,1])
    S3="$(q 0 '0 42 1')"
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/setup.log" 2>&1
    OFFS=25 bash "$R" tworoom 0 "$O/tworoom" setup >> "$O/setup.log" 2>&1
    (   PD="$(q 50 200)"
        TRACE=1 OFFS="$PD" SPANS=6 bash "$R" pmm "$S3" "$O/pmm" all
        TRACE=1 OFFS="$PD" bash "$R" pmm "$S3" "$O/pmm" fact
        OFFS="$(q 50 '100 200')" APH="0.5 2" bash "$R" pmm "$S3" "$O/pmm" ap
        TAG=_drop32 EVX="$EVX +paa_drop_x0=-2 +paa_drop_x1=-1 +paa_drop_y0=0 +paa_drop_y1=1" OFFS="$(q 50 '100 200')" SPANS=6 \
            bash "$R" pmm "$(q 0 '0 42')" "$O/pmm" anchor ) >> "$O/laneA.log" 2>&1 &
    (   TD="$(q 25 75)"
        OFFS="$TD" SPANS=6 bash "$R" tworoom "$S3" "$O/tworoom" all
        OFFS="$TD" bash "$R" tworoom "$S3" "$O/tworoom" fact
        OFFS="$(q 25 '50 75')" APH="0.5 2" bash "$R" tworoom "$S3" "$O/tworoom" ap ) >> "$O/laneB.log" 2>&1 &
    (   # random-play data, seed 1 (seeds 0 and 42: Colab runs x_jwmrand / J8)
        D="$O/rand_pmm"; mkdir -p "$D"
        RH="$(ls "$STABLEWM_HOME/datasets/" | grep -i 'pointmaze_medium_random.*h5' | head -1)"
        if [[ -n "$RH" ]]; then
            [[ -f "$D/nodes.npz" ]] || (cd "$SW" && python "$P/paa_nodes.py" --out "$D/nodes.npz" --model yjang43/lp2-pointmaze-medium \
                --dataset "$RH" --state_key proprio --workers 8 > "$D/nodes.log" 2>&1) || echo "random nodes FAILED" >> "$D/queue.log"
            echo d4rl_pointmaze_medium_expert.h5 > "$D/dataset_name"; touch "$D/READY"
            S1="$(q 1 1)"
            TAG=_rand EVX="$EVX +paa_noexclude=true" OKNODES="$O/pmm/nodes.npz" OFFS="$(q 50 '50 100 150 200')" SPANS=6 bash "$R" pmm "$S1" "$D" anchor
            TAG=_rand EVX="$EVX +paa_noexclude=true" OKNODES="$O/pmm/nodes.npz" OFFS="$(q 50 '50 100 150 200')" bash "$R" pmm "$S1" "$D" ap
            TAG=_rand EVX="$EVX +paa_noexclude=true" XB="$(q 100 400)" XM="ga ap" bash "$R" pmm "$S1" "$D" cross
        else
            echo "no random dataset" >> "$D/queue.log"
        fi
        # graph + GCBC: one GCBC training, every evaluation at seeds 0/42/1, waypoint diagnostics in each log ([waypoints])
        G="GNODES=$O/pmm/nodes.npz GWM=yjang43/lp2-pointmaze-medium"
        env $G EVSEEDS="$S3" XSEEDS="$S3" XB="$(q 100 400)" OFFS="$(q 50 '100 200')" bash "$B" pmm "gcbc gcbcg" "$O/bl_pmm"
        env $G EVSEEDS="$S3" XSEEDS="$S3" XB="$(q 100 400)" OFFS="$(q 50 '100 200')" GSPAN=2 GTAG=_sp2 bash "$B" pmm gcbcg "$O/bl_pmm"
    ) >> "$O/laneC.log" 2>&1 &
    wait
    ;;
J10)  # PointMaze-Large, second maze for the comparison with AP: AP told 0.5x / 1x / 2x the true distance (graph anchors at
      # the same pairs), and GCBC trained on PointMaze-Large, plain and with our graph's waypoints (standard + stitched)
    pml_model || { finish; exit 1; }
    mkdir -p "$O/pml"; echo pointmaze_large.h5 > "$O/pml/dataset_name"
    OFFS=50 bash "$R" pml 0 "$O/pml" setup >> "$O/setup.log" 2>&1
    S2="$(q 0 '0 42')"
    ( OFFS="$(q 50 '200 400')" APH="0.5 2" bash "$R" pml "$S2" "$O/pml" ap
      OFFS="$(q 50 '200 400')" SPANS=6 bash "$R" pml "$S2" "$O/pml" anchor ) >> "$O/laneA.log" 2>&1 &
    ( GNODES="$O/pml/nodes.npz" GWM=lewm_pml EVSEEDS="$S2" XSEEDS="$S2" XB="$(q 100 400)" OFFS="$(q 50 '50 100 200 400')" \
        bash "$B" pml "gcbc gcbcg" "$O/bl_pml" ) >> "$O/laneB.log" 2>&1 &
    wait
    ;;
J11)  # PointMaze-Large: graph and AP memory built from OGBench pointmaze-large-stitch-v0 (short trajectories, no pixels:
      # rendered with the same environment class as the pixel data, gated by a render check and an encoder check)
      # instead of the navigate data the model was trained on; same model and test pairs.  Expert-graph runs: J1.
    pml_model || { finish; exit 1; }
    mkdir -p "$O/pml" "$O/stitch_pml"; echo pointmaze_large.h5 > "$O/pml/dataset_name"
    OFFS=50 bash "$R" pml 0 "$O/pml" setup >> "$O/setup.log" 2>&1
    D="$O/stitch_pml"
    if [[ ! -f "$D/nodes.npz" ]]; then
        (cd "$SW" && python "$P/stitch_nodes.py" --model lewm_pml --out "$D/nodes.npz" \
            --check_h5 "$STABLEWM_HOME/datasets/pointmaze_large.h5" --check_nodes "$O/pml/nodes.npz" \
            --dataset "${STITCH_DS:-pointmaze-large-stitch-v0}" $(q "--max_eps 40" "") > "$D/nodes.log" 2>&1)
        rc=$?; log "stitch nodes rc=$rc $(grep -a '\[stitch_check\]' "$D/nodes.log" | cut -c1-400)"
        [[ $rc == 0 && -f "$D/nodes.npz" ]] || { echo "$(date -u +%F' '%T) stitch nodes FAILED rc=$rc" >> "$D/queue.log"; finish; exit 1; }
    fi
    echo pointmaze_large.h5 > "$D/dataset_name"
    S2="$(q 0 '0 42')"; XS=(TAG=_stitch "EVX=$EVX +paa_noexclude=true" OKNODES="$O/pml/nodes.npz")
    ( env "${XS[@]}" OFFS=50 bash "$R" pml 0 "$D" setup                      # checkpoint check, READY (nodes exist)
      env "${XS[@]}" OFFS="$(q 50 '100 200 400')" SPANS=6 bash "$R" pml "$S2" "$D" anchor ) >> "$O/laneA.log" 2>&1 &
    ( sleep 60; waitf "$D/READY" && env "${XS[@]}" OFFS="$(q 50 '100 200 400')" bash "$R" pml "$S2" "$D" ap ) >> "$O/laneB.log" 2>&1 &
    ( sleep 120; waitf "$D/READY" && env "${XS[@]}" XB="$(q 100 400)" XM="ga ap" bash "$R" pml "$S2" "$D" cross ) >> "$O/laneC.log" 2>&1 &
    wait
    ;;
J12)  # fair baselines on PointMaze-Medium: GCBC and GCIVL with 4x the training budget (4 epochs of 5,000 batches), PLDM with
      # 5 epochs; the stronger PLDM is then planned plain, with min-over-plan, with graph anchors and with AP
    S2="$(q 0 '0 42')"; E4="$(q 1 4)"; E5="$(q 1 5)"
    # three trainings at once need ~150 GB of host RAM (each caches the dataset); with less, run them one after another
    RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"; PARJ=1; [[ "${RAMGB:-0}" -ge 150 ]] || PARJ=0
    log "J12: ${RAMGB:-?} GB free host RAM -> trainings $([[ $PARJ == 1 ]] && echo 'in parallel' || echo 'one after another')"
    bg() { if [[ $PARJ == 1 ]]; then "$@" & else "$@"; fi; }
    bg bash -c "EPOCHS=$E4 MTAG=_e4 EVSEEDS=0 XSEEDS='$S2' XB='$(q 100 400)' OFFS='$(q 50 '50 100 150 200')' bash '$B' pmm gcbc '$O/bl_gcbc_e4' >> '$O/laneA.log' 2>&1"
    bg bash -c "EPOCHS=$E4 MTAG=_e4 EVSEEDS=0 XSEEDS='$S2' XB='$(q 100 400)' OFFS='$(q 50 '50 100 150 200')' bash '$B' pmm gcivl '$O/bl_gcivl_e4' >> '$O/laneB.log' 2>&1"
    ( EPOCHS_WM=$E5 MTAG=_e5 EVSEEDS="$S2" XSEEDS="$S2" XB="$(q 100 400)" OFFS="$(q 50 '50 100 200')" bash "$B" pmm pldm "$O/bl_pldm_e5"
      if [[ -f "$O/bl_pldm_e5/train_pldm.done" ]]; then
          D="$O/pldm5_pmm"; mkdir -p "$D"; echo d4rl_pointmaze_medium_expert.h5 > "$D/dataset_name"
          WM=bl_pmm_pldm_e5 OFFS="$(q 100 '100 200')" SPANS=6 bash "$R" pmm "$S2" "$D" all
          WM=bl_pmm_pldm_e5 OFFS="$(q 100 '100 200')" bash "$R" pmm "$S2" "$D" ap
          WM=bl_pmm_pldm_e5 XB="$(q 100 400)" XM="ga ap" bash "$R" pmm "$S2" "$D" cross
      else
          log "PLDM (5 epochs) training did not finish (see bl_pldm_e5/train_pldm.log)"
      fi ) >> "$O/laneC.log" 2>&1 &
    wait
    ;;
J13)  # planning time per decision (ms_per_decision in each summary line, CUDA-synchronised) and graph build / search time
      # ([graph_timing]); PointMaze-Medium d100, 10 pairs, the four planners interleaved three times so they share the load
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/setup.log" 2>&1
    N="$(q 2 10)"
    for rep in $(q 1 '1 2 3'); do
        TAG=_t$rep EVX="$EVX eval.num_eval=$N" OFFS=100 SPANS=6 bash "$R" pmm 0 "$O/pmm" all
        TAG=_t$rep EVX="$EVX eval.num_eval=$N" OFFS=100 bash "$R" pmm 0 "$O/pmm" ap
    done >> "$O/lane.log" 2>&1
    log "timing (ms per decision, mean over decisions):"
    grep -a "summary" "$O/pmm/summary.txt" | awk '{n=$1; for(i=1;i<=NF;i++) if($i~/^ms_per_decision=/) print "  " n, $i}' | tee -a "$L"
    ;;
J14)  # more offline goal-conditioned baselines (HILP, GCIQL; frozen DINOv2 features, stable-worldmodel), same pairs and budgets:
      # PointMaze-Medium (d50-200), Two-Room (d25-75) and PointMaze-Large (d50-400, needs J1's data), seeds 0/42/1, stitched 0/42
    S3="$(q 0 '0 42 1')"; S2="$(q 0 '0 42')"
    RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"; PARJ=1; [[ "${RAMGB:-0}" -ge 150 ]] || PARJ=0
    log "J14: ${RAMGB:-?} GB free host RAM -> environments $([[ $PARJ == 1 ]] && echo 'in parallel' || echo 'one after another')"
    bg() { if [[ $PARJ == 1 ]]; then "$@" & else "$@"; fi; }
    bg bash -c "EVSEEDS='$S3' XSEEDS='$S2' XB='$(q 100 400)' OFFS='$(q 50 '50 100 150 200')' bash '$B' pmm 'hilp gciql' '$O/bl_pmm' >> '$O/laneA.log' 2>&1"
    bg bash -c "EVSEEDS='$S3' XSEEDS='$S2' XB='$(q 50 200)' OFFS='$(q 25 '25 50 75')' bash '$B' tworoom 'hilp gciql' '$O/bl_tworoom' >> '$O/laneB.log' 2>&1"
    bg bash -c "EVSEEDS='$S2' XSEEDS='$S2' XB='$(q 100 400)' OFFS='$(q 50 '50 100 200 400')' bash '$B' pml 'hilp gciql' '$O/bl_pml' >> '$O/laneC.log' 2>&1"
    wait
    ;;
J17)  # Push-T baselines at long range: GCBC, GCIVL, GCIQL, HILP (frozen DINOv2 features) and PLDM, trained on the LeWM Push-T
      # data, same pairs and budgets as the graph anchor (d50-150, budget 2d) and stitched goals (budget 300), seeds 0/42.
      # Push-T training needs a lot of host RAM (three at once ran out of memory on a 52 GB Colab VM), so lanes run in
      # parallel only with >= 150 GB free
    S2="$(q 0 '0 42')"
    RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"; PARJ=1; [[ "${RAMGB:-0}" -ge 150 ]] || PARJ=0
    log "J17: ${RAMGB:-?} GB free host RAM -> lanes $([[ $PARJ == 1 ]] && echo 'in parallel' || echo 'one after another')"
    bg() { if [[ $PARJ == 1 ]]; then "$@" & else "$@"; fi; }
    bg bash -c "EVSEEDS='$S2' XSEEDS='$S2' XB='$(q 50 300)' OFFS='$(q 25 '50 75 100 150')' bash '$B' pusht 'gcbc gcivl gciql' '$O/bl_pusht' >> '$O/laneA.log' 2>&1"
    bg bash -c "EVSEEDS='$S2' XSEEDS='$S2' XB='$(q 50 300)' OFFS='$(q 25 '50 75 100 150')' bash '$B' pusht 'hilp pldm' '$O/bl_pusht' >> '$O/laneB.log' 2>&1"
    wait
    ;;
J18)  # (1) released DINO-WM Push-T checkpoint at long range (d50-150, 20 pairs, seed 0). Needs the two files
      #     artifacts/dinowm/{config.yaml,dinowm_noprop_pusht_weights.ckpt} (sent separately; DINO_DIR overrides). CEM plans
      #     DINO_BS episodes at once (default 10; 20 needs > 40 GB); each episode is still planned on its own.
      # (2) graph anchors + gated goal handoff (handoff only when the goal is <= 12 graph hops away), 3 seeds, with the
      #     LeWM baselines where earlier jobs have no third seed: Reacher, Two-Room, PointMaze-M, Push-T (also with lambda=3),
      #     and OGBench-Cube (2 seeds, only with >= 150 GB free host RAM)
    S3="$(q 0 '0 42 1')"; S2="$(q 0 '0 42')"; HO="+paa_handoff=true +paa_handoff_hops=12"
    DD="${DINO_DIR:-$ROOT/artifacts/dinowm}"
    if [[ -f "$DD/config.yaml" && -f "$DD/dinowm_noprop_pusht_weights.ckpt" ]]; then
      DM="$STABLEWM_HOME/checkpoints/dinowm_pusht"; mkdir -p "$DM" "$O/dinowm_pusht"
      [[ -f "$DM/model_object.pt" ]] || ( cd "$SW" && python "$P/convert_lightning_ckpt.py" "$DD/config.yaml" \
          "$DD/dinowm_noprop_pusht_weights.ckpt" "$DM/model_object.pt" ) > "$O/dinowm_pusht/dinowm_convert.log" 2>&1 \
          || log "J18: DINO-WM convert FAILED: $(tail -1 "$O/dinowm_pusht/dinowm_convert.log")"
      for pair in "$(q 25 '150 50')" "$(q 25 '100 75')"; do
        ( for d in $pair; do DOFFS=$d N_DINO="$(q 2 20)" DINO_CFG="$DD/config.yaml" DINO_CKPT="$DD/dinowm_noprop_pusht_weights.ckpt" \
            EVX="$EVX solver.batch_size=${DINO_BS:-10}" bash "$B" pusht dinowm "$O/dinowm_pusht"; done ) >> "$O/laneD${pair// /_}.log" 2>&1 &
      done
    else
      log "J18: no DINO-WM files in $DD (config.yaml, dinowm_noprop_pusht_weights.ckpt): DINO-WM part skipped"
    fi
    bash "$R" reacher 0 "$O/reacher" setup >> "$O/setup.log" 2>&1
    OFFS=25 bash "$R" tworoom 0 "$O/tworoom" setup >> "$O/setup.log" 2>&1
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/setup.log" 2>&1
    OFFS=50 bash "$R" pusht 0 "$O/pusht" setup >> "$O/setup.log" 2>&1
    ( TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 25 '25 50 100 150')" SPANS=6 bash "$R" reacher "$S3" "$O/reacher" anchor
      OFFS="$(q 25 '25 50 100 150')" bash "$R" reacher "$S3" "$O/reacher" base
      OFFS="$(q 25 '25 50 100 150')" SPANS=6 bash "$R" reacher "$S3" "$O/reacher" anchor ) >> "$O/laneA.log" 2>&1 &
    ( TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 25 '50 75')" SPANS=6 bash "$R" tworoom "$S3" "$O/tworoom" anchor
      TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 50 '100 200')" SPANS=6 bash "$R" pmm "$S3" "$O/pmm" anchor ) >> "$O/laneB.log" 2>&1 &
    ( TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 50 '100 150')" SPANS=6 bash "$R" pusht "$S3" "$O/pusht" anchor
      TAG=_sw3hg12 EVX="$EVX +paa_switch=3 $HO" OFFS="$(q 50 '100 150')" SPANS=6 bash "$R" pusht "$S3" "$O/pusht" anchor
      TAG=_sw3 EVX="$EVX +paa_switch=3" OFFS="$(q 50 '100 150')" SPANS=6 bash "$R" pusht "$S3" "$O/pusht" anchor ) >> "$O/laneC.log" 2>&1 &
    RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"
    if [[ "${RAMGB:-0}" -ge 150 ]]; then
      ( bash "$R" cube 0 "$O/cube" setup
        TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 50 '50 100 150')" SPANS=6 bash "$R" cube "$S2" "$O/cube" anchor
        OFFS="$(q 50 '50 100 150')" bash "$R" cube "$S2" "$O/cube" base
        OFFS="$(q 50 '50 100 150')" SPANS=6 bash "$R" cube "$S2" "$O/cube" anchor ) >> "$O/laneE.log" 2>&1 &
    else
      log "J18: ${RAMGB:-?} GB free host RAM < 150: OGBench-Cube part skipped"
    fi
    wait
    ;;
J19)  # fills the remaining cells of the main table:
      # (1) Reacher baselines: GCBC, GCIVL, GCIQL, HILP, PLDM (one epoch, as on the other environments) at d25-150, seeds 0/42,
      #     and DINO-WM trained on the Reacher data (5 epochs, no proprio column there) at d100/150, 20 pairs, seed 0
      # (2) PointMaze-Large (needs J1's data and model): GCIVL and PLDM at d50-400 + stitched (seeds 0/42), DINO-WM trained on
      #     its data at d200/400 + stitched (20 pairs, seed 0), and graph anchors + gated goal handoff (G=12) at d400 + stitched
      # Lanes run in parallel only with >= 200 GB free host RAM (PointMaze-Large training caches a lot), else one after another.
    S2="$(q 0 '0 42')"; HO="+paa_handoff=true +paa_handoff_hops=12"
    RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}')"; PARJ=1; [[ "${RAMGB:-0}" -ge 200 ]] || PARJ=0
    log "J19: ${RAMGB:-?} GB free host RAM -> lanes $([[ $PARJ == 1 ]] && echo 'in parallel' || echo 'one after another')"
    bg() { if [[ $PARJ == 1 ]]; then "$@" & else "$@"; fi; }
    bg bash -c "EVSEEDS='$S2' OFFS='$(q 25 '25 50 100 150')' bash '$B' reacher 'gcbc gcivl gciql hilp pldm' '$O/bl_reacher' >> '$O/laneA.log' 2>&1"
    bg bash -c "EPOCHS_DWM='$(q 1 5)' STEPS_DWM='$(q 20 5000)' EVX='$EVX eval.num_eval=$(q 2 20) solver.batch_size=${DINO_BS:-10}' EVSEEDS=0 OFFS='$(q 25 '100 150')' \
        bash '$B' reacher dwm '$O/dwm_reacher' >> '$O/laneB.log' 2>&1"
    if pml_model; then
      bg bash -c "EVSEEDS='$S2' XSEEDS='$S2' XB='$(q 100 400)' OFFS='$(q 50 '50 100 200 400')' bash '$B' pml 'gcivl pldm' '$O/bl_pml' >> '$O/laneC.log' 2>&1"
      bg bash -c "EPOCHS_DWM='$(q 1 5)' STEPS_DWM='$(q 20 5000)' EVX='$EVX eval.num_eval=$(q 2 20) solver.batch_size=${DINO_BS:-10}' EVSEEDS=0 XSEEDS=0 XB='$(q 100 400)' \
          OFFS='$(q 50 '200 400')' bash '$B' pml dwm '$O/dwm_pml' >> '$O/laneD.log' 2>&1"
      mkdir -p "$O/pml"; echo pointmaze_large.h5 > "$O/pml/dataset_name"
      OFFS=50 bash "$R" pml 0 "$O/pml" setup >> "$O/setup.log" 2>&1
      ( TAG=_hg12 EVX="$EVX $HO" OFFS="$(q 50 400)" SPANS=6 bash "$R" pml "$S2" "$O/pml" anchor
        TAG=_hg12 EVX="$EVX $HO" XB="$(q 100 400)" XM=ga bash "$R" pml "$S2" "$O/pml" cross ) >> "$O/laneE.log" 2>&1 &
    fi
    wait
    ;;
J15)  # DINO-WM (frozen DINOv2 patch features + learned predictor, scripts/train/prejepa.py) trained on PointMaze-Medium and
      # Two-Room, planned with the same CEM and the patch-MSE goal cost; long range and stitched goals, seed 0 (planning is slow)
    # planning costs ~11-16 s per step on an L4 (patch tokens), so 20 pairs and only the long offsets + stitched goals
    export EPOCHS_DWM="$(q 1 5)" STEPS_DWM="$(q 20 5000)" EVX="eval.num_eval=$(q 2 20)"
    ( EVSEEDS=0 XSEEDS=0 XB="$(q 100 400)" OFFS="$(q 50 '100 200')" bash "$B" pmm dwm "$O/dwm_pmm" ) >> "$O/laneA.log" 2>&1 &
    ( EVSEEDS=0 XSEEDS=0 XB="$(q 50 200)" OFFS="$(q 25 '50 75')" bash "$B" tworoom dwm "$O/dwm_tworoom" ) >> "$O/laneB.log" 2>&1 &
    wait
    ;;
J16)  # episode-switch cost lambda = 3 (neighbour edges cost 3, anchor = 6 recorded moves ahead) on the navigation tasks, to check
      # that the Push-T fix does not hurt where the plain graph already works: PointMaze-M d100/200 + stitched, Two-Room d50/75 +
      # stitched, PointMaze-L d200/400 + stitched (needs J1), seeds 0/42
    S2="$(q 0 '0 42')"
    OFFS=50 bash "$R" pmm 0 "$O/pmm" setup >> "$O/setup.log" 2>&1
    OFFS=25 bash "$R" tworoom 0 "$O/tworoom" setup >> "$O/setup.log" 2>&1
    ( TAG=_sw3 EVX="$EVX +paa_switch=3" OFFS="$(q 50 '100 200')" SPANS=6 bash "$R" pmm "$S2" "$O/pmm" anchor
      TAG=_sw3 EVX="$EVX +paa_switch=3" XB="$(q 100 400)" XM=ga bash "$R" pmm "$S2" "$O/pmm" cross ) >> "$O/laneA.log" 2>&1 &
    ( TAG=_sw3 EVX="$EVX +paa_switch=3" OFFS="$(q 25 '50 75')" SPANS=6 bash "$R" tworoom "$S2" "$O/tworoom" anchor
      TAG=_sw3 EVX="$EVX +paa_switch=3" XB="$(q 50 200)" XM=ga bash "$R" tworoom "$S2" "$O/tworoom" cross ) >> "$O/laneB.log" 2>&1 &
    if pml_model; then
      mkdir -p "$O/pml"; echo pointmaze_large.h5 > "$O/pml/dataset_name"
      OFFS=50 bash "$R" pml 0 "$O/pml" setup >> "$O/setup.log" 2>&1
      ( TAG=_sw3 EVX="$EVX +paa_switch=3" OFFS="$(q 50 '200 400')" SPANS=6 bash "$R" pml "$S2" "$O/pml" anchor
        TAG=_sw3 EVX="$EVX +paa_switch=3" XB="$(q 100 400)" XM=ga bash "$R" pml "$S2" "$O/pml" cross ) >> "$O/laneC.log" 2>&1 &
    fi
    wait
    ;;
*) echo "unknown job $JOB (smoke | J1 ... J19)"; exit 1 ;;
esac
finish
