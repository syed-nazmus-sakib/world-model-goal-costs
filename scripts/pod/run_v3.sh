D=/root/jepa-wms-project/artifacts/raw_logs/failure_attribution/drawer_alpha0
until [ -f $D/DONE ]; do sleep 20; done
cd /root/jepa-wms-project && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
R=/root/jepa-wms-project
rm -rf $D/v3
timeout 6000 python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $D/v3 --episodes 24 --planners sim_latent --goal_variant 3 --alpha 0 > $D/v3.log 2>&1
python -m evals.failure_attribution.summarize_closed --run_dir $D/v3 > $D/v3_summary.txt 2>&1
echo V3_DONE > $D/V3DONE
