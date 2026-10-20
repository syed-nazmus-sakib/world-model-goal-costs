cd /root/jepa-wms-project && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
R=/root/jepa-wms-project; D=$R/artifacts/raw_logs/failure_attribution/drawer_alpha0
for o in 2,3,4 0,1,2,3,4; do
  n=orbit_$(echo $o | tr -d ,)
  timeout 7200 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $D/$n --episodes 24 --planners sim_latent --orbit $o --alpha 0 > $D/$n.log 2>&1
  python -m evals.failure_attribution.summarize_closed --run_dir $D/$n > $D/${n}_summary.txt 2>&1
done
echo ORBIT_DONE > $D/ORBIT_DONE
