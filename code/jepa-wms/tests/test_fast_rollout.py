# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
The simulator-in-the-loop planners of evals/failure_attribution rely on the fast rollouts giving exactly
the states and frames of the env stack used by the planning eval.

Run with: python -m tests.test_fast_rollout
"""

import time

import numpy as np
import torch
from omegaconf import OmegaConf

from evals.failure_attribution.adapters import get_adapter
from evals.failure_attribution.sim_worker import FAST_ROLLOUT, SimPool, make_worker_env
from evals.simu_env_planning.envs.init import make_env


def make_eval_env(task):
    cfg = OmegaConf.create(
        {
            "frameskip": 5,
            "meta": {"seed": 1},
            "planner": {"horizon": 6},
            "task_specification": {
                "task": task,
                "obs": "rgb_state",
                "obs_concat_channels": False,
                "goal_source": "random_state",
                "goal_H": 6,
                "num_frames": 1,
                "num_proprios": 1,
                "img_size": 224,
                "multitask": False,
                "env": {"with_target": True, "with_velocity": True, "freeze_rand_vec": False},
            },
        }
    )
    return make_env(cfg)


def check(task, n_trials=20):
    adapter = get_adapter(task)
    env = make_eval_env(task)
    env.reset()
    lean = make_worker_env(adapter.kind)
    rs = np.random.RandomState(0)
    worst_state, worst_pixel = 0.0, 0
    for trial in range(n_trials):
        init_state, _ = env.sample_random_init_goal_states(trial)
        if adapter.kind == "pusht":
            init_state = np.concatenate([init_state[:5], rs.randn(2) * 50])
            env_info = {"shape": "T"}
            env.update_env(env_info)
            actions = (rs.randn(30, 2) * 0.3).astype(np.float32)
        else:
            env_info = None
            # large steps so that the dot runs into the walls and through the door
            actions = (rs.randn(30, 2) * 1.5 + np.array([rs.choice([-1.0, 1.0]), 0.0])).astype(np.float32)
        prepared = {"seed": trial, "init_state": np.array(init_state), "env_info": env_info}
        obses, infos = env.rollout(trial, init_state, torch.from_numpy(actions), env_info)
        spec = adapter.spec(prepared, env)
        ref_state, ref_frame = np.asarray(infos[-1]["state"]), obses[-1][0].numpy()
        state, frame = FAST_ROLLOUT[adapter.kind](lean, spec, actions, True)
        worst_state = max(worst_state, float(np.abs(state - ref_state).max()))
        worst_pixel = max(worst_pixel, int(np.abs(frame.transpose(2, 0, 1).astype(int) - ref_frame.astype(int)).max()))
    print(f"{task}: max |state diff| fast vs env rollout over {n_trials} trials: {worst_state}, max pixel diff: {worst_pixel}")
    assert worst_state == 0.0 and worst_pixel == 0

    pool = SimPool(adapter.kind, 12)
    A = np.tile(actions, (300, 1, 1)) + (rs.randn(300, 30, 2) * 0.1).astype(np.float32)
    pool.rollout(spec, A, True)
    t = time.time()
    S, F = pool.rollout(spec, A, True)
    t_render = time.time() - t
    t = time.time()
    S2, _ = pool.rollout(spec, A, False)
    t_plain = time.time() - t
    serial, _ = FAST_ROLLOUT[adapter.kind](lean, spec, A[7], False)
    print(f"{task}: pool, 300 rollouts of 30 steps: {t_render:.2f}s with render, {t_plain:.2f}s without; {S.shape} {F.shape}")
    assert np.array_equal(S[7], serial) and np.array_equal(S, S2)
    pool.close()


def main():
    for task in ("pusht-base", "wall-base"):
        check(task)
    print("OK")


if __name__ == "__main__":
    main()
