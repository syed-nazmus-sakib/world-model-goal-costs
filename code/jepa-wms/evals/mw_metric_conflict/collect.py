# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Collects candidate states with simulator task costs around expert trajectories in Metaworld.

For each episode (one random object / target placement) of a task:
    - the expert policy is rolled out for `max_episode_steps`, its last frame is the goal image,
      as in the planning eval with goal_source=expert;
    - the episode is replayed from the same reset up to several branch times along the expert
      trajectory; the state at the branch time is one candidate, and each branch is continued for a
      few steps with perturbed actions to give further candidates.
Every candidate is rendered with the same env wrapper as the planning eval and stored with the
simulator quantities that define the task's success.

Usage:
    python -m evals.mw_metric_conflict.collect --task mw-push --out_dir $JEPAWM_LOGS/mw_metric_conflict/data
"""

import argparse
import json
import os

import mujoco
import numpy as np
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.metaworld import make_env, task_name_to_policy

# The quantity each task thresholds for success (metaworld v3, evaluate_state), lower is better.
TASK_COST = {
    "mw-reach": "||tcp_center - target|| , success <= 0.05",
    "mw-push": "||obj - target|| , success <= 0.05",
    "mw-drawer-open": "||handle - target|| , success <= 0.03",
    "mw-door-open": "|handle_x - target_x| , success <= 0.08",
}
N_BRANCH_TIMES = 8
# (kind, number of perturbed steps)
BRANCHES = [("random_hold", 10), ("random_hold", 20), ("noisy_expert", 10), ("hand_to_goal", 150)]
NOISY_EXPERT_STD = 0.5
# hand_to_goal: the hand is lifted to this height, moved above its position in the goal image, then lowered,
# so the arm matches the goal image whether or not the task object was moved
HAND_TO_GOAL_CLEARANCE_Z = 0.3


def hand_to_goal_action(tcp, goal_tcp, phase, gripper):
    """Returns (action, phase). Metaworld moves the hand by action[:3] * 0.01 m per step."""
    lift_z = max(HAND_TO_GOAL_CLEARANCE_Z, goal_tcp[2])
    if phase == 0 and (abs(tcp[2] - lift_z) < 0.01 or np.linalg.norm(tcp[:2] - goal_tcp[:2]) < 0.01):
        phase = 1
    if phase == 1 and np.linalg.norm(tcp[:2] - goal_tcp[:2]) < 0.01:
        phase = 2
    if phase == 0:
        target = np.array([tcp[0], tcp[1], lift_z])
    elif phase == 1:
        target = np.array([goal_tcp[0], goal_tcp[1], tcp[2]])
    else:
        target = goal_tcp
    return np.concatenate([np.clip(100 * (target - tcp), -1, 1), [gripper]]), phase


def task_cost(task, mw):
    """Task cost of the current simulator state, and the variables it is made of."""
    # the observation returned by reset can be stale (body moved without a forward pass), so read the simulator
    mujoco.mj_forward(mw.model, mw.data)
    tcp = np.array(mw.tcp_center, dtype=np.float64)
    obj = np.array(mw._get_curr_obs_combined_no_goal()[4:7], dtype=np.float64)
    target = np.array(mw._target_pos, dtype=np.float64)
    if task == "mw-reach":
        cost = np.linalg.norm(tcp - target)
    elif task == "mw-door-open":
        cost = abs(obj[0] - target[0])
    else:  # push, drawer-open
        cost = np.linalg.norm(obj - target)
    return float(cost), tcp, obj, target


def make_cfg(task, img_size, seed):
    return OmegaConf.create(
        {
            "meta": {"seed": seed},
            "task_specification": {
                "task": task,
                "img_size": img_size,
                "max_episode_steps": 500,
                "env": {"freeze_rand_vec": False},
            },
        }
    )


class Episode:
    """One object / target placement, replayable from the same reset."""

    def __init__(self, env, task, seed):
        self.env, self.task = env, task
        self.mw = env.unwrapped
        self.mw._freeze_rand_vec = False
        self.mw.seeded_rand_vec = True
        obs, _ = env.reset(seed=seed)
        self.rand_vec = np.array(self.mw._last_rand_vec).copy()
        # every later reset of this episode reuses the same placement
        self.mw._freeze_rand_vec = True
        self.obs0 = obs

    def reset(self):
        obs, _ = self.env.reset()
        assert np.allclose(self.mw._last_rand_vec, self.rand_vec), "reset changed the placement"
        return obs

    def step(self, action):
        obs, _, _, _, info = self.env.step(np.clip(action, -1, 1).astype(np.float32))
        return obs, info

    def record(self, obs, info=None):
        cost, tcp, obj, target = task_cost(self.task, self.mw)
        return {
            "image": self.env.render(),
            "obs": np.array(obs, dtype=np.float32),
            "cost": cost,
            "tcp": tcp,
            "obj": obj,
            "target": target,
            "success": float(info["success"]) if info is not None else 0.0,
        }


def collect_episode(env, task, seed, rs, max_episode_steps):
    ep = Episode(env, task, seed)
    policy = task_name_to_policy(task)()

    # ---- expert rollout
    obs = ep.reset()
    expert_actions, expert_success = [], []
    for _ in range(max_episode_steps):
        a = policy.get_action(obs)
        obs, info = ep.step(a)
        expert_actions.append(np.clip(a, -1, 1))
        expert_success.append(info["success"])
    goal = ep.record(obs, info)
    # replaying the same actions from a fresh reset must give the same simulator state
    obs_replay = ep.reset()
    for a in expert_actions:
        obs_replay, _ = ep.step(a)
    assert np.allclose(obs_replay, obs, atol=1e-6), "replay from reset is not reproducible"
    first_success = int(np.argmax(expert_success)) if any(expert_success) else max_episode_steps

    # ---- candidates: branch points spread from the start up to shortly after the expert first succeeds
    t_end = min(first_success + 5, max_episode_steps)
    branch_times = np.unique(np.linspace(0, t_end, N_BRANCH_TIMES).round().astype(int))
    cands, cand_meta = [], []
    for t_b in branch_times:
        for b, (kind, n_steps) in enumerate(BRANCHES):
            obs = ep.reset()
            info = None
            for t in range(t_b):
                obs, info = ep.step(expert_actions[t])
            if b == 0:
                cands.append(ep.record(obs, info))
                cand_meta.append({"kind": "expert", "t_branch": int(t_b), "n_steps": 0})
            hold = rs.uniform(-1, 1, size=4)
            phase = 0
            for _ in range(n_steps):
                if kind == "random_hold":
                    a = hold
                elif kind == "noisy_expert":
                    a = policy.get_action(obs) + NOISY_EXPERT_STD * rs.randn(4)
                else:
                    tcp = np.array(ep.mw.tcp_center, dtype=np.float64)
                    if phase == 2 and np.linalg.norm(tcp - goal["tcp"]) < 0.005:
                        break
                    a, phase = hand_to_goal_action(tcp, goal["tcp"], phase, expert_actions[-1][3])
                obs, info = ep.step(a)
            cands.append(ep.record(obs, info))
            cand_meta.append({"kind": kind, "t_branch": int(t_b), "n_steps": n_steps})
    return goal, cands, cand_meta, {
        "seed": seed,
        "rand_vec": ep.rand_vec.tolist(),
        "expert_success": bool(any(expert_success)),
        "expert_first_success_step": first_success,
        "expert_actions": np.array(expert_actions).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_COST))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(make_cfg(args.task, args.img_size, args.seed)).env  # MetaWorldWrapper, without the TimeLimit
    rs = np.random.RandomState(args.seed)
    keys = ["image", "obs", "cost", "tcp", "obj", "target", "success"]
    out = {f"cand_{k}": [] for k in keys}
    out.update({f"goal_{k}": [] for k in keys})
    cand_episode, episodes, cand_metas = [], [], []
    e, seed = 0, args.seed * 100003
    while e < args.num_episodes:
        goal, cands, cand_meta, ep_meta = collect_episode(env, args.task, seed, rs, args.max_episode_steps)
        seed += 1
        if not ep_meta["expert_success"]:
            # the goal image must show a solved task
            print(f"seed {ep_meta['seed']}: expert failed, skipping", flush=True)
            continue
        for k in keys:
            out[f"goal_{k}"].append(goal[k])
            out[f"cand_{k}"].extend(c[k] for c in cands)
        cand_episode.extend([e] * len(cands))
        cand_metas.extend(cand_meta)
        episodes.append(ep_meta)
        costs = np.array([c["cost"] for c in cands])
        print(
            f"{args.task} ep {e:03d} seed {ep_meta['seed']}: {len(cands)} candidates, "
            f"cost range [{costs.min():.3f}, {costs.max():.3f}], goal cost {goal['cost']:.3f}",
            flush=True,
        )
        e += 1

    name = args.task.replace("mw-", "")
    np.savez_compressed(
        os.path.join(args.out_dir, f"{name}.npz"),
        cand_episode=np.array(cand_episode),
        **{k: np.array(v) for k, v in out.items()},
    )
    with open(os.path.join(args.out_dir, f"{name}_meta.json"), "w") as f:
        json.dump(
            {
                "task": args.task,
                "task_cost_definition": TASK_COST[args.task],
                "seed": args.seed,
                "img_size": args.img_size,
                "max_episode_steps": args.max_episode_steps,
                "branches": BRANCHES,
                "noisy_expert_std": NOISY_EXPERT_STD,
                "candidates": cand_metas,
                "episodes": episodes,
            },
            f,
        )
    print(f"Saved {len(cand_episode)} candidates from {len(episodes)} episodes of {args.task}")


if __name__ == "__main__":
    main()
