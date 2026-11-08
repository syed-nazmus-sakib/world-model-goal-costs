#!/bin/bash
# Graph anchor on other LeWM envs (2026-10-31). Same protocol as PushT: goal offset d, budget 2d, 50 pairs, seed = pair sampler.
#   fetch : dataset from the HF collection quentinll/lewm (fetch_lewm_data.py)
#   nodes : encode the dataset's 5-step grid with the frozen encoder
#   eval  : released h5, relmin h10 (released + min-over-plan), graph anchor span 6 and 8 (min, h10) at each offset
# Offsets: OFFS (default "50 100 150"); an offset is skipped when fewer than 60 episodes are longer than it.
# usage: bash scripts/pod/run_ga_env.sh <tworoom|cube|reacher|pmm> [seeds="0"] [out=/root/out/ga_<env>] [lane=all|base|anchor|ap]
# lane ap: AP-CEM (APH="0.5 2" adds runs with the goal offset misstated by those factors)
# OKNODES: nodes file for the offset check when the graph comes from another dataset
# EVX: extra Hydra overrides for every evaluation; TAG: run-name suffix (ablations)
# lane setup: data, checkpoint and nodes only
# lane match: graph anchors under AP's planner (MSPANS); lane cross: stitched goals (XB budgets, XM methods); WM: other world model
# lane fact: graph anchors with the RELEASED search (H=5, last-step cost): the missing cell of target x search
# lane ngc: NGC baseline (NGCM="ngc gangc": + the hybrid, our anchor scored with NGC's graph distance); cross: XM="ngc ..."
# HA: horizon of the min-over-plan and anchor runs (default 10)
# TRACE=1: save a per-run trace (agent state, anchors) to $OUT/trace_<run>.npz for the failure analysis
# TRACE_EVERY: trace stride in steps (default 5); VID=1: keep each run's rollout videos in $OUT/vid_<run>; SK_MIN=1: skip the min-over-plan run
set -u
ENV=$1; SEEDS=${2:-"0"}; OUT=${3:-/root/out/ga_$ENV}; LANE=${4:-all}; mkdir -p "$OUT"
R=${JEPAWM_HOME:-/root/jepa-wms-project}
SWM=$R/code/third_party/stable-worldmodel; P=$R/code/lewm_explorations/paa; PY=${SWM_PY:-/root/venv_swm/bin/python}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Colab registers only Mesa's EGL vendor; MuJoCo's EGL context then segfaults once CUDA is up. Use NVIDIA's EGL instead.
if [ -f /usr/lib64-nvidia/libEGL_nvidia.so.0 ]; then
  V=/usr/share/glvnd/egl_vendor.d/10_nvidia.json; mkdir -p "$(dirname $V)"
  [ -f $V ] || printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0"}}\n' > $V
  export __EGL_VENDOR_LIBRARY_FILENAMES=$V LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
fi
case $ENV in
  tworoom) CFG=tworoom; M=quentinll/lewm-tworooms; KEY=pos_agent ;;
  pusht)   CFG=pusht;   M=quentinll/lewm-pusht;    KEY=state; DS=${DS:-galilai-group/lewm-pusht} ;;
  cube)    CFG=cube;    M=quentinll/lewm-cube;     KEY=none ;;
  reacher) CFG=reacher; M=quentinll/lewm-reacher;  KEY=none ;;
  pmm)     CFG=pointmaze_medium; M=yjang43/lp2-pointmaze-medium; KEY=proprio; DSN=d4rl_pointmaze_medium_expert.h5 ;;
  pml)     CFG=pointmaze_large;  M=${PML_CKPT:?set PML_CKPT to the trained LeWM}; KEY=state; DSN=pointmaze_large.h5 ;;
  *) echo "unknown env $ENV"; exit 1 ;;
esac
M=${WM:-$M}  # WM: another world model (e.g. bl_tworoom_pldm, our own PLDM checkpoint) planned with the same anchors
DS=${DS:-$(cat "$OUT/dataset_name" 2>/dev/null)}
NODES=$OUT/nodes.npz
log() { echo "$(date -u +%F' '%T) $*" | tee -a "$OUT/queue.log"; }
abort() { log "$1"; touch "$OUT/ABORT"; exit 1; }
# why <log>: the line that says why a run died (last Python error, CUDA out of memory, or kill), for queue.log and job.log
why() { grep -aE 'OutOfMemoryError|Error:|Error\b|Killed|Traceback' "$1" | grep -v '^ *File ' | tail -1 | cut -c1-200; }
cd "$SWM"
if [ "$LANE" != anchor ]; then
  if [ -z "$DS" ]; then
    $PY $P/fetch_lewm_data.py $ENV > "$OUT/fetch.log" 2>&1 || abort "fetch FAILED"
    DS=$(grep -a '^H5 ' "$OUT/fetch.log" | awk '{print $2}' | grep -x "${DSN:-.*}" | grep -iv test | head -1)
    [ -n "$DS" ] || abort "no dataset found"
    echo "$DS" > "$OUT/dataset_name"; log "dataset $DS"
  fi
  [ -f "$OUT/dataset_name" ] || echo "$DS" > "$OUT/dataset_name"
  case $M in quentinll/*|/*|bl_*|lewm_*) ;; *)  # third-party checkpoint: its config may only build stable_worldmodel / torch classes
    $PY -c "
import json, sys; from huggingface_hub import hf_hub_download
c = json.load(open(hf_hub_download('$M', 'config.json'))); t = []
def walk(x):
    if isinstance(x, dict):
        [t.append(v) for k, v in x.items() if k == '_target_']; [walk(v) for v in x.values()]
    elif isinstance(x, list): [walk(v) for v in x]
walk(c); bad = [v for v in t if not str(v).startswith(('stable_worldmodel.', 'stable_pretraining.', 'torch.nn.'))]
print('targets', sorted(set(t))); sys.exit(1 if bad else 0)" > "$OUT/ckpt_check.log" 2>&1 || abort "checkpoint config has unexpected targets" ;;
  esac
  $PY -c "import stable_worldmodel as swm; swm.wm.utils.load_pretrained('$M')" > "$OUT/ckpt.log" 2>&1 || abort "checkpoint FAILED"
  [ -f $NODES ] || { log "nodes start"; $PY $P/paa_nodes.py --out $NODES --model $M --dataset $DS --state_key $KEY --workers 8 \
      > $OUT/nodes.log 2>&1 || abort "nodes FAILED"; log "nodes: $(tail -1 $OUT/nodes.log)"; }
  $PY -c "
import numpy as np; d=np.load('$NODES'); ep,st=d['ep'],d['step']; u=np.unique(ep)
ln=np.array([st[ep==e].max() for e in u]); print('episodes',len(u),'nodes',len(ep),'len median',int(np.median(ln)),
{o:int((ln>o).sum()) for o in (25,50,75,100,150,200)})" | tee -a "$OUT/queue.log"
  touch "$OUT/READY"
fi
while [ ! -f "$OUT/READY" ]; do [ -f "$OUT/ABORT" ] && { log "anchor lane: setup aborted"; exit 1; }; sleep 30; done   # anchor lane waits for the nodes
[ "$LANE" = setup ] && { log "setup done (data, checkpoint, nodes)"; exit 0; }  # lane setup: only prepare $OUT
DS=$(cat "$OUT/dataset_name")
ok_off() { $PY -c "
import numpy as np; d=np.load('${OKNODES:-$NODES}'); ep,st=d['ep'],d['step']
print(int(sum(st[ep==e].max()>$1 for e in np.unique(ep))>=60))"; }
ev() {  # name seed offset extra...
  local name=$1 seed=$2 off=$3; shift 3
  [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && return
  log "$name start"
  $PY scripts/plan/eval_wm.py --config-name=$CFG policy=$M seed=$seed eval.dataset_name=$DS \
    eval.goal_offset_steps=$off eval.eval_budget=$((2 * off)) "$@" ${EVX:-} \
    ${TRACE:+"+trace_npz=$OUT/trace_$name.npz" +trace_every=${TRACE_EVERY:-5}} ${VID:+"+video_dir=$OUT/vid_$name"} > "$OUT/$name.log" 2>&1 || log "$name FAILED: $(why "$OUT/$name.log")"
  grep -aq '\[summary\]' "$OUT/$name.log" && echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
  log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
}
if [ "$LANE" = cross ]; then  # E3 stitching test: goals from other episodes, fixed budgets XB; AP gets half the budget as its offset
  for seed in $SEEDS; do for bud in ${XB:-200 400}; do
    X="eval.goal_offset_steps=5 eval.eval_budget=$bud +cross_goal=true"
    for m in ${XM:-released relmin ga ap}; do
      name=x${m}${GCAL:+c}${TAG:-}_b${bud}_s$seed
      [ -f "$OUT/$name.log" ] && grep -q '\[summary\]' "$OUT/$name.log" && continue
      log "$name start"
      case $m in
        released) A="" ;;
        relmin)   A="+metric_head=released +metric_agg=min plan_config.horizon=10" ;;
        ga)       A="+paa_nodes=$NODES +paa_mode=fixed +paa_span=6 +metric_agg=min plan_config.horizon=10" ;;
        ap)       A="+paa_nodes=$NODES +paa_mode=ap +metric_agg=last plan_config.horizon=1 plan_config.receding_horizon=1 +paa_ap_offset=$((bud / 2))" ;;
        ngc)      A="+ngc_nodes=$NODES +metric_agg=last" ;;
        gangc)    A="+ngc_nodes=$NODES +ngc_target=anchor +paa_span=6 +metric_agg=min plan_config.horizon=10" ;;
        gap)      A="+paa_nodes=$NODES +paa_mode=gap +paa_span=${GSPAN:-6} +metric_agg=last plan_config.horizon=1 plan_config.receding_horizon=1 ${GCAL:++gap_cal=true}" ;;
      esac
      $PY scripts/plan/eval_wm.py --config-name=$CFG policy=$M seed=$seed eval.dataset_name=$DS $X $A \
        +pairs_npz=$OUT/pairs_$name.npz ${EVX:-} > "$OUT/$name.log" 2>&1 || log "$name FAILED: $(why "$OUT/$name.log")"
      grep -aq '\[summary\]' "$OUT/$name.log" && echo "$name $(grep -a '\[summary\]' "$OUT/$name.log")" >> "$OUT/summary.txt"
      log "$name: $(grep -a '\[summary\]' "$OUT/$name.log" | grep -oE 'success_rate=[0-9.]+')"
    done
  done; done
  log "GA_LANE_DONE $ENV cross"; exit 0
fi
for seed in $SEEDS; do for off in ${OFFS:-50 100 150}; do
  [ "$(ok_off $off)" = 1 ] || { log "skip d$off: fewer than 60 episodes longer than $off"; continue; }
  if [ "$LANE" = all ] || [ "$LANE" = base ]; then
    [ -n "${SK_REL:-}" ] || ev released${TAG:-}_h5_d${off}_s$seed $seed $off   # SK_REL: only the min-over-plan run
    [ -n "${SK_MIN:-}" ] || ev relmin${TAG:-}_h${HA:-10}_d${off}_s$seed $seed $off +metric_head=released +metric_agg=min plan_config.horizon=${HA:-10}
  fi
  if [ "$LANE" = ap ]; then  # Anchored Planning (arXiv 2609.30036) re-implemented: one-block CEM toward the retrieved successor
    ev apcem${TAG:-}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=ap +metric_agg=last plan_config.horizon=1 plan_config.receding_horizon=1
    for f in ${APH:-}; do  # AP told a wrong goal offset: f x the true one (AP needs the offset, graph anchors do not)
      ev apcem_H${f}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=ap +metric_agg=last plan_config.horizon=1 \
         plan_config.receding_horizon=1 +paa_ap_offset=$(python3 -c "print(int(round($off * $f)))")
    done
    continue
  fi
  if [ "$LANE" = fact ]; then  # 2x2 target {goal, anchor} x search {released H5 last, H10 min}: this is (anchor, released)
    ev gah5${TAG:-}_sp6_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=fixed +paa_span=6 +metric_agg=last
    continue
  fi
  if [ "$LANE" = ngc ]; then  # NGC (arXiv 2610.25013, concurrent work) and the hybrid: our anchor + NGC's graph distance
    for m in ${NGCM:-ngc}; do
      case $m in
        ngc)   ev ngc${TAG:-}_d${off}_s$seed $seed $off +ngc_nodes=$NODES +metric_agg=last ;;
        gangc) ev gangc${TAG:-}_sp6_d${off}_s$seed $seed $off +ngc_nodes=$NODES +ngc_target=anchor +paa_span=6 \
                  +metric_agg=min plan_config.horizon=10 ;;
      esac
    done
    continue
  fi
  if [ "$LANE" = gap ]; then  # graph-guided retrieval: graph waypoint + hop count replace AP's goal and offset
    for sp in ${GSPANS:-6}; do
      ev gap${GCAL:+c}_sp${sp}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=gap +paa_span=$sp +metric_agg=last \
         plan_config.horizon=1 plan_config.receding_horizon=1 ${GCAL:++gap_cal=true}
    done
    continue
  fi
  if [ "$LANE" = match ]; then  # graph anchors with AP's planner (one-block CEM, replan every block): isolates the anchor choice
    for sp in ${MSPANS:-1 2}; do
      ev gam_sp${sp}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=fixed +paa_span=$sp +metric_agg=last \
         plan_config.horizon=1 plan_config.receding_horizon=1
    done
    continue
  fi
  if [ "$LANE" != base ]; then
    for sp in ${SPANS:-6 8}; do
      ev ga${TAG:-}_sp${sp}_d${off}_s$seed $seed $off +paa_nodes=$NODES +paa_mode=fixed +paa_span=$sp +metric_agg=min plan_config.horizon=${HA:-10}
    done
  fi
done; done
log "GA_LANE_DONE $ENV $LANE"
