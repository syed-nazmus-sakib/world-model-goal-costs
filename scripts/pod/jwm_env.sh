# Source this on a pod: JEPA-WM paths and thread limits for the uv venv made by setup_jwm_venv.sh (replaces env/env.sh, which needs conda).
# The project checkout is expected at JEPAWM_HOME (default /root/jepa-wms-project).
export JEPAWM_HOME=${JEPAWM_HOME:-/root/jepa-wms-project}
export JEPAWM_DSET=$JEPAWM_HOME/artifacts/datasets JEPAWM_LOGS=$JEPAWM_HOME/artifacts/raw_logs JEPAWM_CKPT=$JEPAWM_HOME/artifacts/checkpoints
# simulator workers are CPU processes: unbounded torch / BLAS threads in each of them thrash a pod's CPU quota
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PATH=/root/venv_jwm/bin:$PATH
cd $JEPAWM_HOME/code/jepa-wms
