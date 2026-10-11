# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Simulator ground truth for action cycles in Metaworld: a forward action sequence followed by the
same sequence reversed in time with the hand displacement negated. No world model is involved.

A free-space cycle should bring the whole scene back. A cycle that pushes the object should bring
the hand back but leave the object displaced. For every cycle we save the initial, midpoint and
final simulator states and renders, and the hand / object closure errors.

Usage:
    python -m evals.mw_cycles.sim_cycles --task mw-push --out $JEPAWM_LOGS/mw_cycles/push_part0.npz \
        --episode_start 0 --num_episodes 5
"""

import argparse
import json

import mujoco
import numpy as np

from evals.mw_hidden_physics.sim_test import restore, snapshot
from evals.mw_metric_conflict.collect import make_cfg
from evals.simu_env_planning.envs.metaworld import make_env, task_name_to_policy

# The hand lags its commanded target by a few steps, so a bare forward + reversed sequence ends before the
# hand is back. Each half is therefore FWD moving steps followed by HOLD steps with zero hand displacement,
# and the cycle starts from a state settled for PRE_SETTLE zero-displacement steps.
# The full cycle (2 * HALF env steps) is the released planning horizon (6 model steps x frameskip 5).
FWD, HOLD, PRE_SETTLE = 10, 5, 10
HALF = FWD + HOLD
N_BRANCH_TIMES = 4
# forward programs per base state: (kind, parameter, count)
PROGRAMS = (
    [("random_hold", 1.0, 3), ("random_hold", 0.5, 3), ("expert", 0.0, 1), ("expert_noise", 0.5, 5)]
    # constant direction towards the object, aimed with a lateral offset (m) to get glancing contacts
    + [("to_object", off, 1) for off in (0.0, 0.02, 0.04, 0.06)]
)


def reverse_program(forward):
    """Time-reversed sequence with the hand displacement negated, same gripper effort."""
    rev = forward[::-1].copy()
    rev[:, :3] *= -1
    return rev


def hold_program(gripper, n):
    return np.concatenate([np.zeros((n, 3)), np.full((n, 1), gripper)], axis=1)


def cycle_program(forward):
    hold = hold_program(forward[0, 3], HOLD)
    return np.concatenate([forward, hold, reverse_program(forward), hold])


def make_programs(rs, expert_actions, t_b, tcp, obj, gripper):
    progs, kinds = [], []
    for kind, par, n in PROGRAMS:
        for _ in range(n):
            if kind == "random_hold":
                d = rs.randn(3)
                a = np.tile(par * d / np.linalg.norm(d), (FWD, 1))
            elif kind == "expert":
                a = expert_actions[t_b : t_b + FWD, :3].copy()
            elif kind == "expert_noise":
                a = expert_actions[t_b : t_b + FWD, :3] + par * rs.randn(FWD, 3)
            else:
                to_obj = obj - tcp
                lateral = np.array([-to_obj[1], to_obj[0], 0.0])
                lateral = lateral / max(np.linalg.norm(lateral), 1e-9) * par * rs.choice([-1, 1])
                d = to_obj + lateral
                a = np.tile(d / max(np.linalg.norm(d), 1e-9), (FWD, 1))
            a = np.clip(a, -1, 1)
            progs.append(np.concatenate([a, np.full((FWD, 1), gripper)], axis=1))
            kinds.append(kind if kind != "to_object" else f"to_object_{par}")
    return np.array(progs), kinds


class ContactMeter:
    """Contacts between the task object and the robot."""

    def __init__(self, mw):
        self.m, self.d = mw.model, mw.data
        self.obj = self.m.body("obj").id
        self.robot_root = self.m.body_rootid[self.m.body("hand").id]

    def robot_object_force(self):
        """Largest normal force of a robot-object contact at the current step, 0 if none."""
        force, f6 = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            other = b2 if b1 == self.obj else b1 if b2 == self.obj else None
            if other is not None and self.m.body_rootid[other] == self.robot_root:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                force = max(force, abs(f6[0]), 1e-12)
        return force


def run_cycle(env, mw, meter, snap, forward):
    """Returns per-step tcp, obj, mocap, contact force over the 2 * HALF steps, and renders / states."""
    restore(mw, snap)
    rec = {"tcp": [np.array(mw.tcp_center)], "obj": [np.array(mw._get_pos_objects())], "force": []}
    rec["mocap"] = [mw.data.mocap_pos[0].copy()]
    images, states = [env.render()], [snapshot(mw)]
    for a in cycle_program(forward):
        env.step(a.astype(np.float32))
        rec["tcp"].append(np.array(mw.tcp_center))
        rec["obj"].append(np.array(mw._get_pos_objects()))
        rec["mocap"].append(mw.data.mocap_pos[0].copy())
        rec["force"].append(meter.robot_object_force())
        if len(rec["force"]) in (HALF, 2 * HALF):
            images.append(env.render())
            states.append(snapshot(mw))
    return {k: np.array(v) for k, v in rec.items()}, np.array(images), states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["mw-reach", "mw-push"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    args = parser.parse_args()

    env = make_env(make_cfg(args.task, 224, 0)).env
    mw = env.unwrapped
    meter = ContactMeter(mw)
    policy = task_name_to_policy(args.task)()
    out = {k: [] for k in ("tcp", "obj", "mocap", "force", "images", "qpos", "qvel", "actions", "episode", "t_branch")}
    kinds_all = []

    for e in range(args.episode_start, args.episode_start + args.num_episodes):
        rs = np.random.RandomState(2000 + e)
        mw._freeze_rand_vec, mw.seeded_rand_vec = False, True
        obs, _ = env.reset(seed=e)
        mw._freeze_rand_vec = True
        snaps, expert_actions, success = [], [], []
        for _ in range(args.max_episode_steps + FWD):
            snaps.append(snapshot(mw))
            a = np.clip(policy.get_action(obs), -1, 1)
            obs, _, _, _, info = env.step(a.astype(np.float32))
            expert_actions.append(a)
            success.append(info["success"])
        expert_actions = np.array(expert_actions)
        first_success = int(np.argmax(success)) if any(success) else args.max_episode_steps
        branch_times = np.linspace(0, min(first_success, args.max_episode_steps), N_BRANCH_TIMES).round().astype(int)
        for t_b in branch_times:
            # settle the hand on its commanded target before the cycle starts
            restore(mw, snaps[t_b])
            for a in hold_program(expert_actions[t_b, 3], PRE_SETTLE):
                env.step(a.astype(np.float32))
            snap = snapshot(mw)
            progs, kinds = make_programs(
                rs,
                expert_actions,
                t_b,
                np.array(mw.tcp_center),
                np.array(mw._get_pos_objects()),
                expert_actions[t_b, 3],
            )
            for forward in progs:
                rec, images, states = run_cycle(env, mw, meter, snap, forward)
                for k in ("tcp", "obj", "mocap", "force"):
                    out[k].append(rec[k])
                out["images"].append(images)
                out["qpos"].append(np.array([s["qpos"] for s in states]))
                out["qvel"].append(np.array([s["qvel"] for s in states]))
                out["actions"].append(forward)
                out["episode"].append(e)
                out["t_branch"].append(t_b)
            kinds_all.extend(kinds)
        print(f"{args.task} episode {e}: {len(branch_times) * len(progs)} cycles", flush=True)

    np.savez_compressed(
        args.out,
        kinds=json.dumps(kinds_all),
        half=HALF,
        fwd=FWD,
        hold=HOLD,
        pre_settle=PRE_SETTLE,
        **{k: np.array(v) for k, v in out.items()},
    )
    print(f"Saved {len(kinds_all)} cycles to {args.out}")


if __name__ == "__main__":
    main()
