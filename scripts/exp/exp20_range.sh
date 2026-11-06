#!/bin/bash
# exp-20: range of the goal cost across world models, and local reach.  One environment per call.
#   nodes  : LeWM and PLDM (paa_nodes.py through run_ga_env.sh setup), frozen DINOv2, DINO-WM when bl_<env>_dwm exists
#   range  : exp_range.py, rho(h) and Spearman against a graph built from true positions only, plus one cost map
#   reach  : success of released + min-over-plan planners at d=30 on two seeds, for each world model
# Needs JEPAWM_HOME, SWM_PY; checkpoints bl_<env>_pldm in $CK.  QUICK=1: 3000 nodes, 5 goals, 2 pairs.
# usage: bash scripts/exp/exp20_range.sh <pmm|tworoom> <out dir>
set -u
ENV=$1; D=$2; Q=${QUICK:-0}; mkdir -p "$D"
REPO=${JEPAWM_HOME:?}; PY=${SWM_PY:-python}; R=$REPO/scripts/pod/run_ga_env.sh
P=$REPO/code/lewm_explorations/paa; SW=$REPO/code/third_party/stable-worldmodel
CK=${CK:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints}
case $ENV in pmm) KEY=proprio ;; tworoom) KEY=pos_agent ;; *) echo "unknown env $ENV"; exit 1 ;; esac
FR=${EP_FRAC:-$([ $ENV = pmm ] && echo 0.2 || echo 1.0)}   # PointMaze-M has 199k nodes, a fifth of its episodes is enough here
MAXN=$([ "$Q" = 1 ] && echo 3000 || echo 0); NG=$([ "$Q" = 1 ] && echo 5 || echo 200)
EV=$([ "$Q" = 1 ] && echo eval.num_eval=2 || echo "")
OFFS=25 bash $R $ENV 0 $D/lewm setup
DS=$(cat $D/lewm/dataset_name)
L=""
cd $SW
$PY $P/exp_encode.py --out $D/lewm_e.npz --model $( [ $ENV = pmm ] && echo yjang43/lp2-pointmaze-medium || echo quentinll/lewm-tworooms ) \
  --dataset $DS --state_key $KEY --max_nodes $MAXN --ep_frac $FR || echo "encode lewm FAILED"
[ -f $D/lewm_e.npz ] && L="$L lewm=$D/lewm_e.npz"
$PY $P/exp_encode.py --out $D/dinov2_e.npz --model dinov2 --dataset $DS --state_key $KEY --max_nodes $MAXN --ep_frac $FR || echo "encode dinov2 FAILED"
[ -f $D/dinov2_e.npz ] && L="$L dinov2=$D/dinov2_e.npz"
for m in pldm dwm; do
  n=bl_${ENV}_$m
  [ -d $CK/$n ] || { echo "no checkpoint $n, skipped"; continue; }
  $PY $P/exp_encode.py --out $D/${m}_e.npz --model $n --dataset $DS --state_key $KEY --max_nodes $MAXN --ep_frac $FR || echo "encode $m FAILED"
  [ -f $D/${m}_e.npz ] && L="$L $m=$D/${m}_e.npz"
done
$PY $P/exp_range.py --out $D/range --n_goals $NG $L
cd $REPO
EVX="$EV" OFFS=30 bash $R $ENV "0 42" $D/lewm base
if [ -d $CK/bl_${ENV}_pldm ]; then
  WM=bl_${ENV}_pldm OFFS=25 bash $R $ENV 0 $D/pldm setup
  WM=bl_${ENV}_pldm EVX="$EV" OFFS=30 bash $R $ENV "0 42" $D/pldm base
fi
echo "exp20 $ENV done"
