#!/bin/bash
# Reduced "Experiment A": one sequential lane on a cheap pod. Resumable: every stage skips finished outputs.
# Stage 1 task search -> Stage 2 encoder generality (offline L1-L3, exact dynamics) -> Stage 3 task generality (L4, exact
# dynamics, DINOv2) -> Stage 4 real world-model planner on drawer-close (last: cut if the budget runs out).
R=/root/jepa-wms-project
cd $R && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
G=$R/artifacts/raw_logs/failure_attribution/generality; mkdir -p $G/search $G/enc $G/l4 $G/wm
ln -sfn $R/configs $R/artifacts/configs   # check_goal_variants.py looks for configs next to the logs root
log() { echo "$(date -u +%F' '%T) $*" >> $G/queue.log; }
log "queue start"

# ---- Stage 1: task search (12 episodes, both oracle planners) ----
for t in window-close drawer-open faucet-open door-close handle-press-side plate-slide; do
  if [ ! -s $G/search/${t}_summary.txt ]; then
    timeout 1500 python -m evals.failure_attribution.check_goal_variants $t > $G/search/${t}_variants.log 2>&1
    timeout 7200 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_$t.yaml --out_dir $G/search/$t \
        --episodes 12 --planners sim_latent sim_true > $G/search/$t.log 2>&1
    python -m evals.failure_attribution.summarize_closed --run_dir $G/search/$t > $G/search/${t}_summary.txt 2>&1
    log "search $t: $(grep -E 'sim_latent +[0-9]+/' $G/search/${t}_summary.txt | head -1 | cut -c1-40) | $(grep -E 'sim_true +[0-9]+/' $G/search/${t}_summary.txt | head -1 | cut -c1-40)"
  fi
done
python $R/scripts/pod/pick_tasks.py $G/search > $G/search/selected.txt 2> $G/search/pick.log
SEL=$(cat $G/search/selected.txt | tr '\n' ' ')
log "selected tasks: $SEL"

# ---- Stage 2: encoder generality (offline levels, exact dynamics, alpha 0, no CEM) ----
for enc in siglip2 clip dinov2_b vjepa2; do
  for t in drawer-close reach-wall $SEL; do
    o=$G/enc/${t}_$enc
    [ -s $o/summary.json ] && continue
    V=4; [ "$t" = reach-wall ] && V=0
    timeout 7200 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_jepa-wm_$t.yaml --out_dir $o \
        --episodes 24 --variants $V --alpha 0 --no_cem --encoder $enc --num_workers 8 > $o.log 2>&1
    log "enc $enc $t: $(python -c "import json;d=json.load(open('$o/summary.json'));print(d.get('L2_ordering_accuracy_vs_true_cost_per_variant(clear pairs)'))" 2>/dev/null || echo FAILED)"
  done
done
# the DINOv2 reference for reach-wall and the selected tasks (the drawer-close reference is the Mac backup podA/drawer_alpha0/goal_equiv_a0, same protocol)
for t in reach-wall $SEL; do
  o=$G/enc/${t}_dinov2
  [ -s $o/summary.json ] && continue
  timeout 7200 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_jepa-wm_$t.yaml --out_dir $o \
      --episodes 24 --variants 4 --alpha 0 --no_cem --num_workers 8 > $o.log 2>&1
  log "enc dinov2 $t done"
done

# ---- Stage 3: task generality, closed loop L4 (exact dynamics, DINOv2, alpha 0) ----
for t in $SEL; do
  for v in 0 3; do
    o=$G/l4/${t}_v$v
    [ -s ${o}_summary.txt ] && continue
    timeout 9000 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_$t.yaml --out_dir $o \
        --episodes 24 --planners sim_latent --goal_variant $v --alpha 0 > $o.log 2>&1
    python -m evals.failure_attribution.summarize_closed --run_dir $o > ${o}_summary.txt 2>&1
    log "l4 $t v$v: $(grep -E 'sim_latent +[0-9]+/' ${o}_summary.txt | head -1 | cut -c1-40)"
  done
done

# ---- Stage 4: the real world-model planner on drawer-close (smoke first; skipped if it does not fit or is too slow) ----
o=$G/wm/drawer-close_smoke
if [ ! -s $o/ep_000.npz ]; then
  timeout 2400 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $o \
      --episodes 1 --planners wm --alpha 0 > $o.log 2>&1
  log "wm smoke: $(grep 'done:' $o.log | tail -1 | cut -c1-80) $(grep -c 'OutOfMemory' $o.log) oom-lines"
fi
if [ -s $o/ep_000.npz ]; then
  for v in 0 3; do
    o=$G/wm/drawer-close_v$v
    [ -s ${o}_summary.txt ] && continue
    timeout 16000 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $o \
        --episodes 24 --planners wm --goal_variant $v --alpha 0 > $o.log 2>&1
    python -m evals.failure_attribution.summarize_closed --run_dir $o > ${o}_summary.txt 2>&1
    log "wm v$v: $(grep -E 'wm +[0-9]+/' ${o}_summary.txt | head -1 | cut -c1-40)"
  done
fi
log "queue done"
echo QUEUE_DONE > $G/QUEUE_DONE
