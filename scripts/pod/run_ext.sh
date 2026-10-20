cd /root/jepa-wms-project && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
R=/root/jepa-wms-project; D=$R/artifacts/raw_logs/failure_attribution/drawer_alpha0
for v in 0 3; do
 ( timeout 9000 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $D/v$v --episodes 48 --planners sim_latent --goal_variant $v --alpha 0 >> $D/v$v.log 2>&1
   python -m evals.failure_attribution.summarize_closed --run_dir $D/v$v > $D/v${v}_summary48.txt 2>&1 ) &
done; wait; echo EXT_DONE > $D/EXT_DONE
