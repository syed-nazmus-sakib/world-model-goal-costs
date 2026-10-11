# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Exact snapshot / restore of a Metaworld episode, for closed-loop failure attribution.

A Metaworld state is the MuJoCo integration state (qpos, qvel, act, warmstart, ctrl, mocap, time, ...)
plus four Python attributes that `SawyerXYZEnv.step` mutates: `curr_path_length`, `_prev_obs` (half of the
39-d observation is the previous step's), `_last_stable_obs` and `_did_see_sim_exception`. Everything set
at reset (object / target placement, init positions used by the reward) is reproduced in a worker by a
reset with the same frozen `rand_vec`.

No mj_forward after restoring: the next mj_step recomputes everything it needs, and the frame rendered
after the last step then uses the same (pre-integration) kinematics as the env stack of the eval.
tests/test_fast_rollout_mw.py checks state and pixel equality against the eval's env stack.
"""

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
PY_ATTRS = ("curr_path_length", "_prev_obs", "_last_stable_obs", "_did_see_sim_exception")

# distance (m) under which Metaworld v3 reports success, per task (evaluate_state); used to scale the
# true-state cost so that cost < 1 <=> success for tasks whose success is a single distance threshold
SUCCESS_THRESHOLD = {"mw-reach": 0.05, "mw-reach-wall": 0.05, "mw-push": 0.05, "mw-push-wall": 0.05,
                     "mw-drawer-close": 0.055, "mw-door-open": 0.03, "mw-button-press-topdown": 0.024}


def capture(mw):
    """mw: the unwrapped SawyerXYZEnv. Returns a picklable snapshot."""
    s = np.empty(mujoco.mj_stateSize(mw.model, STATE_SPEC))
    mujoco.mj_getState(mw.model, mw.data, s, STATE_SPEC)
    py = {}
    for k in PY_ATTRS:
        v = getattr(mw, k)
        py[k] = np.array(v).copy() if isinstance(v, np.ndarray) else v
    return {"mj": s, "py": py, "rand_vec": np.array(mw._last_rand_vec).copy()}


def restore(mw, snap):
    mujoco.mj_setState(mw.model, mw.data, snap["mj"], STATE_SPEC)
    for k, v in snap["py"].items():
        setattr(mw, k, v.copy() if isinstance(v, np.ndarray) else v)


def robot_geoms(model):
    """Geoms of the robot: every body in the subtree of 'base' (pedestal, torso, head, arm, hand, claws) + mocap."""
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    mocap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
    robot = {base, mocap}
    for b in range(model.nbody):  # bodies are ordered so that parents come before children
        if model.body_parentid[b] in robot and b != 0:
            robot.add(b)
    return np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in robot])


def robot_bodies(model):
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    mocap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
    robot = {base, mocap}
    for b in range(model.nbody):
        if model.body_parentid[b] in robot and b != 0:
            robot.add(b)
    return robot


def object_joints(model):
    """Joints that are not part of the robot (puck free joint, drawer slide, door hinge, button slide, ...)."""
    robot = robot_bodies(model)
    return [j for j in range(model.njnt) if model.jnt_bodyid[j] not in robot]


def object_qpos(mw):
    m = mw.model
    return np.concatenate([mw.data.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + _QPOS_N[m.jnt_type[j]]] for j in object_joints(m)])


_QPOS_N = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4, mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}


def render_hidden(wrapper, geoms):
    """Render with the given geoms made fully transparent."""
    m = wrapper.unwrapped.model
    alpha = m.geom_rgba[geoms, 3].copy()
    m.geom_rgba[geoms, 3] = 0.0
    try:
        return wrapper.render().copy()
    finally:
        m.geom_rgba[geoms, 3] = alpha


# ------------------------------------------------------------------ the eval's env stack (main process)
def capture_env(env):
    """env: the PixelWrapper stack of the eval. Also saves the wrappers' counters and histories."""
    tl = _find(env, "_elapsed_steps")
    return {
        "mw": capture(env.unwrapped),
        "elapsed": tl._elapsed_steps,
        "frames": [f.copy() for f in env._frames],
        "proprios": [p.clone() if hasattr(p, "clone") else np.array(p).copy() for p in env._proprios],
    }


def restore_env(env, snap):
    restore(env.unwrapped, snap["mw"])
    _find(env, "_elapsed_steps")._elapsed_steps = snap["elapsed"]
    env._frames.clear()
    env._frames.extend(f.copy() for f in snap["frames"])
    env._proprios.clear()
    env._proprios.extend(p.clone() if hasattr(p, "clone") else np.array(p).copy() for p in snap["proprios"])


def _find(env, attr):
    e = env
    while e is not None:
        if attr in vars(e):
            return e
        e = getattr(e, "env", None)
    raise AttributeError(attr)


# ------------------------------------------------------------------ worker side
class WorkerEnv:
    """A MetaWorldWrapper per task, reset to the placement of the episode being planned."""

    def __init__(self, img_size=224):
        self.img_size, self.envs, self.cur = img_size, {}, {}

    def get(self, task, rand_vec):
        from evals.mw_metric_conflict.collect import make_cfg
        from evals.simu_env_planning.envs.metaworld import make_env

        if task not in self.envs:
            self.envs[task] = make_env(make_cfg(task, self.img_size, 0)).env  # MetaWorldWrapper (no TimeLimit)
        w = self.envs[task]
        key = tuple(np.round(rand_vec, 12))
        if self.cur.get(task) != key:
            mw = w.unwrapped
            mw._last_rand_vec = np.array(rand_vec).copy()
            mw._freeze_rand_vec = True
            w.reset()
            self.cur[task] = key
        return w


def mw_fast_rollout(wenv, spec, actions, render):
    """
    spec:    (task, snapshot from capture()) or (task, snapshot, hide_robot)
    actions: (T, 4) float32 env actions (already denormalized)
    Returns state (41,) float32 = [39-d obs, success, obj_to_target] and, if render, the final frame
    (rendered without the robot if hide_robot: oracle).
    """
    task, snap = spec[0], spec[1]
    hide = len(spec) > 2 and spec[2]
    w = wenv.get(task, snap["rand_vec"])
    restore(w.unwrapped, snap)
    info = {}
    for a in actions:
        obs, _, _, _, info = w.step(np.asarray(a, dtype=np.float32))
    state = np.concatenate([obs, [float(info["success"]), float(info.get("obj_to_target", np.nan))]])
    frame = None
    if render:
        frame = render_hidden(w, robot_geoms(w.unwrapped.model)) if hide else w.render()
    return state.astype(np.float32), frame
