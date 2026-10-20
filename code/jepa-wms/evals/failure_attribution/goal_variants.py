# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Task-equivalent goal variants for Metaworld: the same object goal, the robot arm moved elsewhere.

The released eval's goal is the last frame of a scripted expert episode run in a temporary env
(PlanEvaluator.get_goal_state_mw). We capture that env's final simulator state, then move only the arm
(through the mocap target, first up, then sideways) to other poses that do not touch the object, check
that the object did not move, and render each variant through the same MetaWorldWrapper render path.
"""

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from evals.failure_attribution.mw_sim import capture, restore


def hook_expert_capture(evaluator):
    """After the next set_episode, evaluator.expert_final_snap holds the expert env's final state."""
    orig = evaluator.unroll_expert

    def wrapped(env, *a, **k):
        evaluator.expert_init_snap = capture(env.unwrapped)
        out = orig(env, *a, **k)
        evaluator.expert_final_snap = capture(env.unwrapped)
        evaluator.expert_env = env
        return out

    evaluator.unroll_expert = wrapped


def find_mw_wrapper(env):
    e = env
    while e is not None:
        if type(e).__name__ == "MetaWorldWrapper":
            return e
        e = getattr(e, "env", None)
    raise RuntimeError("no MetaWorldWrapper in the env stack")


def _obj_pos(mw):
    return np.array(mw._get_pos_objects()[:3])


def _obj_qpos_adr(mw):
    """qpos / qvel addresses of the object's free joint (the first free joint that is not the arm)."""
    m = mw.model
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return m.jnt_qposadr[j], m.jnt_dofadr[j]
    raise RuntimeError("no free joint")


def make_arm_variants(env, init_snap, goal_snap, k, seed, min_obj_dist=0.12, max_steps=120, tol=0.002):
    """
    Variant 0: the eval's goal state (expert's final arm pose).
    Variants 1..k: the episode's initial state with every object joint set to its goal value (free objects:
    goal x-y, resting flat; articulated objects: goal joint value), velocities zeroed and settled, then the
    arm driven (mocap) to a pose away from the object; variant 1 keeps the initial arm pose. Variants 1..k
    share the exact object state; only the arm differs.
    Returns [(snapshot, frame (H,W,3) uint8, obs (39,), obj_pos (3,))].
    """
    from evals.failure_attribution.mw_sim import _QPOS_N, object_joints, object_qpos

    w = find_mw_wrapper(env)
    mw = w.unwrapped
    m = mw.model
    rng = np.random.default_rng(seed)
    restore(mw, goal_snap)
    goal_q = {j: mw.data.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + _QPOS_N[m.jnt_type[j]]].copy() for j in object_joints(m)}
    out = [(goal_snap, w.render().copy(), np.asarray(mw._get_obs(), dtype=np.float32), _obj_pos(mw))]
    free = any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in goal_q)
    if not free:
        # articulated objects (drawer, door, button): start from the goal state itself, release the gripper and
        # hold the arm still until the object is at rest (the initial arm pose can collide with e.g. an open door)
        restore(mw, goal_snap)
        for _ in range(60):
            w.step(np.array([0, 0, 0, -1], dtype=np.float32))
        base_snap = capture(mw)
        obj_q0, obj0 = object_qpos(mw), _obj_pos(mw)
        out.append((base_snap, w.render().copy(), np.asarray(mw._get_obs(), dtype=np.float32), obj0.copy()))
    else:
        restore(mw, init_snap)
    for j, q in (goal_q.items() if free else []):
        a = m.jnt_qposadr[j]
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:  # may be lifted / tilted by the gripper: rest it flat
            mw.data.qpos[a:a + 2] = q[:2]
        else:
            mw.data.qpos[a:a + len(q)] = q
        mw.data.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + (6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1)] = 0
    if free:
        mujoco.mj_forward(m, mw.data)
        mw._prev_obs = mw._get_curr_obs_combined_no_goal()
        for _ in range(10):  # settle
            w.step(np.array([0, 0, 0, -1], dtype=np.float32))
        base_snap = capture(mw)
        obj_q0, obj0 = object_qpos(mw), _obj_pos(mw)
        out.append((base_snap, w.render().copy(), np.asarray(mw._get_obs(), dtype=np.float32), obj0.copy()))
    # tolerance ~1/10 of the success margin: slide joints 5 mm, hinges 0.015 rad (~3 mm at a door handle)
    jt = np.concatenate([[m.jnt_type[j]] * _QPOS_N[m.jnt_type[j]] for j in object_joints(m)])
    tols = np.where(jt == mujoco.mjtJoint.mjJNT_SLIDE, 0.005, np.where(jt == mujoco.mjtJoint.mjJNT_HINGE, 0.015, tol))
    moved = lambda: bool((np.abs(object_qpos(mw) - obj_q0) > tols).any())
    obj_js = object_joints(m)

    def hold():
        """Articulated objects: keep their joints at the base values while the arm is being repositioned."""
        for j in obj_js:
            a, n = m.jnt_qposadr[j], _QPOS_N[m.jnt_type[j]]
            mw.data.qpos[a:a + n] = base_q[j]
            mw.data.qvel[m.jnt_dofadr[j]] = 0
        mujoco.mj_forward(m, mw.data)

    from evals.failure_attribution.mw_sim import robot_bodies

    robot_b = robot_bodies(m)
    obj_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] not in robot_b and m.geom_bodyid[g] != 0])
    ct0, ca0 = m.geom_contype[obj_geoms].copy(), m.geom_conaffinity[obj_geoms].copy()

    def ghost(on):
        """Articulated objects: make the object geoms non-colliding while the arm is repositioned."""
        if free:
            return
        m.geom_contype[obj_geoms] = 0 if on else ct0
        m.geom_conaffinity[obj_geoms] = 0 if on else ca0

    def robot_object_contact():
        for c in range(mw.data.ncon):
            b1, b2 = m.geom_bodyid[mw.data.contact[c].geom1], m.geom_bodyid[mw.data.contact[c].geom2]
            if (b1 in robot_b) != (b2 in robot_b) and 0 not in (b1, b2):
                return True
        return False

    restore(mw, base_snap)
    base_q = {j: mw.data.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + _QPOS_N[m.jnt_type[j]]].copy() for j in obj_js}
    tries = 0
    reasons = {}
    rej = lambda r: reasons.__setitem__(r, reasons.get(r, 0) + 1)
    make_arm_variants.reasons = reasons
    while len(out) < k + 1 and tries < 50 * k:
        tries += 1
        restore(mw, base_snap)
        lo, hi = mw.hand_low, mw.hand_high
        target = rng.uniform(lo, hi)
        target[2] = rng.uniform(max(lo[2], 0.12), min(hi[2], 0.35))
        if np.linalg.norm(target[:2] - obj0[:2]) < min_obj_dist:
            rej("target_near_object")
            continue
        ok = True
        ghost(True)
        for phase in ("up", "move"):
            goal = np.array([mw.tcp_center[0], mw.tcp_center[1], max(target[2], 0.2)]) if phase == "up" else target
            for _ in range(max_steps):
                delta = goal - mw.tcp_center
                if np.linalg.norm(delta) < 0.01:
                    break
                w.step(np.concatenate([np.clip(delta * 100, -0.5, 0.5), [-1.0]]).astype(np.float32))
                if not free:
                    hold()
                elif moved():
                    ok = False
                    break
            if not ok:
                break
        ghost(False)
        if not free:
            hold()
            if robot_object_contact():
                rej("contact_after_unghost")
                continue
        for _ in range(10):  # released: the object must stay at rest on its own
            w.step(np.array([0, 0, 0, -1], dtype=np.float32))
        if not ok or moved() or np.linalg.norm(mw.tcp_center - target) > 0.05:
            rej("not_ok" if not ok else ("object_moved" if moved() else f"target_missed"))
            continue
        make_arm_variants.drift = max(getattr(make_arm_variants, "drift", 0.0), float(np.abs(object_qpos(mw) - obj_q0).max()))
        out.append((capture(mw), w.render().copy(), np.asarray(mw._get_obs(), dtype=np.float32), _obj_pos(mw)))
    return out


def robot_free_frame(env, snap):
    """Render the snapshot with the robot hidden (oracle: goal image without the embodiment)."""
    from evals.failure_attribution.mw_sim import render_hidden, robot_geoms

    w = find_mw_wrapper(env)
    restore(w.unwrapped, snap)
    # restore() sets qpos/qvel only; the renderer reads geom poses, which are updated by mj_forward. Without it the
    # render shows the geoms where the last forward/step left them (found 2026-10-17: start and goal renders identical).
    mujoco.mj_forward(w.unwrapped.model, w.unwrapped.data)
    return render_hidden(w, robot_geoms(w.unwrapped.model))


def goal_td(frame, obs):
    """Goal TensorDict in the format of the eval's goal_obs (one frame)."""
    return TensorDict(
        {
            "visual": torch.from_numpy(frame.transpose(2, 0, 1).copy()).unsqueeze(0),
            "proprio": torch.from_numpy(np.asarray(obs[:4], dtype=np.float32)).unsqueeze(0),
        },
        batch_size=[],
    )
