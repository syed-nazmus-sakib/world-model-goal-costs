# Sourced by every chain on the Colab VM: paths, runners and a log helper (chain log: /content/out/chain.log).
export JEPAWM_HOME=/content/jepa-wms-project SWM_PY=python3
export PYTHONPATH=/content/jepa-wms-project/code/third_party/stable-worldmodel   # our stable-worldmodel tree, not a stale install
cd $JEPAWM_HOME; R=scripts/pod/run_ga_env.sh; B=scripts/pod/run_baselines.sh; O=/content/out; mkdir -p $O
L=$O/chain.log; log() { echo "$(date -u +%F' '%T) $*" >> $L; }
CK=/root/.stable_worldmodel/checkpoints; mkdir -p $CK
