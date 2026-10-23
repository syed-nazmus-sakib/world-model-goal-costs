#!/bin/bash
# Run first on a new pod. MuJoCo renders through EGL; if the host does not mount the NVIDIA EGL driver
# (no 10_nvidia.json / libEGL_nvidia), rendering silently falls back to the CPU: about 30 min instead of 2 min per
# Metaworld episode (pod F, 2026-10-23). Installing the matching libnvidia-gl package did not fix it there: use another host.
# usage: bash scripts/pod/check_egl.sh        prints EGL_OK or EGL_MISSING, then a render timing if the venv exists
ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | grep -q nvidia && ls /usr/lib/x86_64-linux-gnu/ | grep -q libEGL_nvidia && echo EGL_OK || echo EGL_MISSING
[ -x /root/venv_jwm/bin/python ] || exit 0
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /root/venv_jwm/bin/python - <<'P' 2>/dev/null
import os, time
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
d, r = mujoco.MjData(m), mujoco.Renderer(m, 224, 224)
t = time.time()
for _ in range(200):
    r.update_scene(d)
    r.render()
print(f"render {1000 * (time.time() - t) / 200:.2f} ms per 224x224 frame", flush=True)
os._exit(0)  # the EGL context's destructor can raise at exit; the timing is what matters
P
