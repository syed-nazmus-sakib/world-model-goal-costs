#!/bin/bash
# exp-29: DINO-WM reached 95-100% on PointMaze-Large in J19, far above its 15% on PointMaze-Medium, and its goal cost has a proprio
# term that the LeWM cost has not.  Same trained model, 20 pairs at d400:
#   full cost, seeds 42 and 1 | pixel-only cost, seed 0 | proprio-only cost, seed 0
# Reading: pixel-only stays high -> the result is DINO-WM's own; pixel-only falls -> it comes from the position in the cost.
# Also the Reacher DINO-WM evaluations that ran out of memory in J19.  Uses the checkpoints J19 left on disk, else trains again.
# usage: bash scripts/exp/exp29_dinowm_cost.sh <out dir>
set -u
D=$1; REPO=${JEPAWM_HOME:?}; B=$REPO/scripts/pod/run_baselines.sh
CK=${CK:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints}; Q=${QUICK:-0}
EV="eval.num_eval=$([ "$Q" = 1 ] && echo 2 || echo 20) solver.batch_size=${DINO_BS:-10}"
export EPOCHS_DWM=$([ "$Q" = 1 ] && echo 1 || echo 5) STEPS_DWM=$([ "$Q" = 1 ] && echo 20 || echo 5000)
mkdir -p $D/pml $D/reacher
[ -n "$(find "$CK" -path '*bl_pml_dwm*' -name model_object.pt 2>/dev/null)" ] && touch $D/pml/train_dwm.done
[ -n "$(find "$CK" -path '*bl_reacher_dwm*' -name model_object.pt 2>/dev/null)" ] && touch $D/reacher/train_dwm.done
OF=$([ "$Q" = 1 ] && echo 50 || echo 400)
EVX="$EV" EVSEEDS=$([ "$Q" = 1 ] && echo 0 || echo "42 1") OFFS=$OF bash $B pml dwm $D/pml
EVX="$EV" EVSEEDS=0 OFFS=$OF DWM_OBJ=objective=goal_mse_pixels GTAG=_pix bash $B pml dwm $D/pml
EVX="$EV" EVSEEDS=0 OFFS=$OF DWM_OBJ=objective=goal_mse_proprio GTAG=_prop bash $B pml dwm $D/pml
EVX="$EV" EVSEEDS=0 OFFS="$([ "$Q" = 1 ] && echo 25 || echo '100 150')" bash $B reacher dwm $D/reacher
echo "exp29 done"
