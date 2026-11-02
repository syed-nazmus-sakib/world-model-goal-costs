# Sourced by env_setup.sh and run_job.sh. Headless MuJoCo rendering uses EGL; on some machines only Mesa's EGL vendor is
# registered and MuJoCo's EGL context then crashes once CUDA is up. If NVIDIA's EGL library exists, register it for this
# process through a user-level vendor file (no root needed). Set GWM_EGL_FIX=0 to skip.
if [[ "${GWM_EGL_FIX:-1}" == 1 && "${MUJOCO_GL:-egl}" == egl && -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    for _lib in /usr/lib64-nvidia/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib64/libEGL_nvidia.so.0; do
        if [[ -f "$_lib" ]]; then
            _vd="${STABLEWM_HOME:-$HOME/.stable_worldmodel}/egl"; mkdir -p "$_vd"
            printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' "$_lib" > "$_vd/10_nvidia.json"
            export __EGL_VENDOR_LIBRARY_FILENAMES="$_vd/10_nvidia.json" LD_LIBRARY_PATH="$(dirname "$_lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            break
        fi
    done
    unset _lib _vd
fi
