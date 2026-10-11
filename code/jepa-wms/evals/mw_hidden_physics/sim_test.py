# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Simulator-side test of hidden-physics ambiguity in Metaworld push. No world model is involved.

From a base state on an expert trajectory we build worlds that render to the same image but differ
in a hidden quantity (velocity, friction or object mass), apply the same open-loop action sequences
in each, and record how far the futures and the task costs of the candidate sequences diverge from
the nominal world.

Usage:
    python -m evals.mw_hidden_physics.sim_test --out $JEPAWM_LOGS/mw_hidden_physics/part0.npz \
        --episode_start 0 --num_episodes 5
"""

import argparse
import json

import mujoco
import numpy as np

from evals.mw_metric_conflict.collect import make_cfg
from evals.simu_env_planning.envs.metaworld import make_env, task_name_to_policy

TASK = "mw-push"
HORIZON = 30  # env steps = planning horizon 6 x frameskip 5 of the released Metaworld eval
N_BRANCH_TIMES = 4
# open-loop candidate action sequences: (kind, noise std, count)
CANDIDATES = [("expert", 0.0, 1), ("expert_noise", 0.3, 7), ("expert_noise", 1.0, 8), ("random_hold", 0.0, 8)]
OBJ_DOF = slice(9, 15)  # free joint of the puck in qvel
ARM_DOF = slice(0, 7)
VARIANTS = (
    [("nominal", "none", 0.0), ("vel_all_zero", "vel_zero", 0.0)]
    + [(f"vel_obj_+{v}mps", "vel_obj", v) for v in (0.1, 0.3, 1.0)]
    + [(f"vel_arm_+{v}radps", "vel_arm", v) for v in (0.5, 2.0)]
    + [(f"friction_x{s}", "friction", s) for s in (0.1, 0.3, 3.0, 10.0)]
    + [(f"obj_mass_x{s}", "mass", s) for s in (0.1, 0.3, 3.0, 10.0, 100.0)]
)


def snapshot(mw):
    d = mw.data
    return {
        k: np.array(getattr(d, k)).copy()
        for k in ("qpos", "qvel", "mocap_pos", "mocap_quat", "qacc_warmstart", "ctrl")
    } | {"time": float(d.time)}


def restore(mw, snap, qvel=None):
    d = mw.data
    d.qpos[:] = snap["qpos"]
    d.qvel[:] = snap["qvel"] if qvel is None else qvel
    d.mocap_pos[:] = snap["mocap_pos"]
    d.mocap_quat[:] = snap["mocap_quat"]
    d.qacc_warmstart[:] = snap["qacc_warmstart"]
    d.ctrl[:] = snap["ctrl"]
    d.time = snap["time"]
    mujoco.mj_forward(mw.model, d)
    mw.curr_path_length = 0


class Physics:
    """Applies a hidden-physics variant to the model and undoes it."""

    def __init__(self, mw):
        self.m = mw.model
        self.obj = self.m.body("obj").id
        self.friction0 = self.m.geom_friction.copy()
        self.mass0 = self.m.body_mass.copy()
        self.inertia0 = self.m.body_inertia.copy()

    def apply(self, kind, value):
        self.reset()
        if kind == "friction":
            # contact friction is the max over the two geoms, so scale the sliding friction of all of them
            self.m.geom_friction[:, 0] = self.friction0[:, 0] * value
        elif kind == "mass":
            self.m.body_mass[self.obj] = self.mass0[self.obj] * value
            self.m.body_inertia[self.obj] = self.inertia0[self.obj] * value

    def reset(self):
        self.m.geom_friction[:] = self.friction0
        self.m.body_mass[:] = self.mass0
        self.m.body_inertia[:] = self.inertia0


def variant_qvel(kind, value, snap, obj_dir, arm_dir):
    qvel = snap["qvel"].copy()
    if kind == "vel_zero":
        qvel[:] = 0
    elif kind == "vel_obj":
        qvel[OBJ_DOF.start : OBJ_DOF.start + 2] += value * obj_dir
    elif kind == "vel_arm":
        qvel[ARM_DOF] += value * arm_dir
    return qvel


def rollout(env, mw, snap, qvel, actions):
    """Returns obj (H+1, 3), tcp (H+1, 3), cost (H+1,) for one open-loop action sequence."""
    restore(mw, snap, qvel)
    obj, tcp = [np.array(mw._get_pos_objects())], [np.array(mw.tcp_center)]
    for a in actions:
        env.step(a.astype(np.float32))
        obj.append(np.array(mw._get_pos_objects()))
        tcp.append(np.array(mw.tcp_center))
    obj, tcp = np.array(obj), np.array(tcp)
    return obj, tcp, np.linalg.norm(obj - np.array(mw._target_pos), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    args = parser.parse_args()

    env = make_env(make_cfg(TASK, 224, 0)).env
    mw = env.unwrapped
    physics = Physics(mw)
    policy = task_name_to_policy(TASK)()
    K = sum(c[2] for c in CANDIDATES)
    V = len(VARIANTS)
    out = {k: [] for k in ("obj", "tcp", "cost", "pixel_diff", "base_qvel_obj_speed", "base_t", "episode", "actions")}

    for e in range(args.episode_start, args.episode_start + args.num_episodes):
        rs = np.random.RandomState(1000 + e)
        mw._freeze_rand_vec, mw.seeded_rand_vec = False, True
        obs, _ = env.reset(seed=e)
        mw._freeze_rand_vec = True
        # nominal expert rollout, with a snapshot of the simulator before every step
        snaps, expert_actions, success = [], [], []
        for _ in range(args.max_episode_steps + HORIZON):
            snaps.append(snapshot(mw))
            a = np.clip(policy.get_action(obs), -1, 1)
            obs, _, _, _, info = env.step(a.astype(np.float32))
            expert_actions.append(a)
            success.append(info["success"])
        expert_actions = np.array(expert_actions)
        first_success = int(np.argmax(success)) if any(success) else args.max_episode_steps
        branch_times = np.linspace(0, min(first_success, args.max_episode_steps), N_BRANCH_TIMES).round().astype(int)

        for t_b in branch_times:
            snap = snaps[t_b]
            # the same open-loop candidate action sequences are applied in every world
            cands = []
            for kind, std, n in CANDIDATES:
                for _ in range(n):
                    if kind == "random_hold":
                        cands.append(np.tile(rs.uniform(-1, 1, size=4), (HORIZON, 1)))
                    else:
                        cands.append(expert_actions[t_b : t_b + HORIZON] + std * rs.randn(HORIZON, 4))
            cands = np.clip(np.array(cands), -1, 1)
            phi = rs.uniform(0, 2 * np.pi)
            obj_dir = np.array([np.cos(phi), np.sin(phi)])
            arm_dir = rs.randn(7)
            arm_dir /= np.linalg.norm(arm_dir)

            obj = np.zeros((V, K, HORIZON + 1, 3))
            tcp = np.zeros((V, K, HORIZON + 1, 3))
            cost = np.zeros((V, K, HORIZON + 1))
            pixel_diff = np.zeros(V)
            image0 = None
            for v, (_, kind, value) in enumerate(VARIANTS):
                physics.apply(kind, value)
                qvel = variant_qvel(kind, value, snap, obj_dir, arm_dir)
                # the starting observation of every variant must be the nominal image
                restore(mw, snap, qvel)
                image = env.render().astype(np.int16)
                image0 = image if image0 is None else image0
                pixel_diff[v] = np.abs(image - image0).max()
                for k in range(K):
                    obj[v, k], tcp[v, k], cost[v, k] = rollout(env, mw, snap, qvel, cands[k])
            physics.reset()
            out["obj"].append(obj)
            out["tcp"].append(tcp)
            out["cost"].append(cost)
            out["pixel_diff"].append(pixel_diff)
            out["base_qvel_obj_speed"].append(np.linalg.norm(snap["qvel"][OBJ_DOF][:3]))
            out["base_t"].append(t_b)
            out["episode"].append(e)
            out["actions"].append(cands)
            print(f"episode {e} t_branch {t_b}: max pixel diff over variants {pixel_diff.max():.0f}", flush=True)

    np.savez_compressed(
        args.out,
        variants=json.dumps(VARIANTS),
        candidates=json.dumps(CANDIDATES),
        **{k: np.array(v) for k, v in out.items()},
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
