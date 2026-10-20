R=/root/jepa-wms-project; B=$R/artifacts/raw_logs/failure_attribution
until [ -f $B/drawer_alpha0/EXT_DONE ]; do sleep 30; done
cd $R && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
mkdir -p $B/dino
( timeout 5400 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $B/drawer_alpha0/goal_equiv_a0 --episodes 24 --alpha 0 --no_cem --num_workers 8 > $B/drawer_alpha0/goal_equiv_a0.log 2>&1
  timeout 5400 python -m evals.failure_attribution.goal_equiv_mw --fname $R/configs/mw_dino-wm_drawer-close.yaml --out_dir $B/dino/goal_equiv_a0 --episodes 24 --alpha 0 --no_cem --num_workers 8 > $B/dino/goal_equiv_a0.log 2>&1 ) &
( for v in 0 3; do
    timeout 7200 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_dino-wm_drawer-close.yaml --out_dir $B/dino/drawer_a0_v$v --episodes 24 --planners sim_latent --goal_variant $v --alpha 0 > $B/dino/drawer_a0_v$v.log 2>&1
    python -m evals.failure_attribution.summarize_closed --run_dir $B/dino/drawer_a0_v$v > $B/dino/drawer_a0_v${v}_summary.txt 2>&1
  done ) &
wait; echo NEXT_DONE > $B/dino/NEXT_DONE
