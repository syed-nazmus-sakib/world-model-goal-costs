# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Checks that the Metaworld snapshot / restore and the SimPool fast rollout reproduce the env stack of the
planning eval exactly (state and pixels), from a mid-episode state.

    python -m tests.test_fast_rollout_mw [--tasks mw-reach-wall mw-push] [--trials 10]
"""

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from evals.failure_attribution.mw_sim import capture_env, restore_env
from evals.failure_attribution.sim_worker import SimPool


def make_eval_env(task, img_size=224):
    """The same wrapper stack as evals.simu_env_planning.envs.init.make_env for an rgb_state Metaworld eval."""
    from evals.simu_env_planning.envs.metaworld import make_env
    from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
    from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper

    cfg = OmegaConf.create(
        {
            "meta": {"seed": 0},
            "task_specification": {
                "task": task, "img_size": img_size, "max_episode_steps": 100, "num_frames": 1, "num_proprios": 1,
                "obs_concat_channels": False, "env": {"freeze_rand_vec": False},
            },
        }
    )
    return PixelWrapper(cfg, TensorWrapper(make_env(cfg)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["mw-reach-wall", "mw-push"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()
    pool = SimPool("mw", 8)
    rng = np.random.default_rng(0)
    for task in args.tasks:
        env = make_eval_env(task)
        mw = env.unwrapped
        worst = {"restore_state": 0.0, "restore_pixel": 0, "fast_state": 0.0, "fast_pixel": 0, "succ_mismatch": 0}
        for trial in range(args.trials):
            mw._freeze_rand_vec, mw.seeded_rand_vec = False, True
            env.reset(seed=1000 + trial)
            mw._freeze_rand_vec = True
            for _ in range(int(rng.integers(5, 40))):  # move somewhere mid-episode
                env.step(torch.from_numpy(rng.uniform(-1, 1, 4).astype(np.float32)))
            snap = capture_env(env)
            acts = rng.uniform(-1, 1, (args.horizon, 4)).astype(np.float32)

            obses, _, _, infos = env.step_multiple(torch.from_numpy(acts))
            s1, f1, ok1 = np.asarray(infos[-1]["state"]), obses[-1][0].numpy().transpose(1, 2, 0), infos[-1]["success"]
            restore_env(env, snap)
            obses, _, _, infos = env.step_multiple(torch.from_numpy(acts))
            s2, f2 = np.asarray(infos[-1]["state"]), obses[-1][0].numpy().transpose(1, 2, 0)
            st, fr = pool.rollout((task, snap["mw"]), acts[None], True)

            worst["restore_state"] = max(worst["restore_state"], float(np.abs(s1 - s2).max()))
            worst["restore_pixel"] = max(worst["restore_pixel"], int(np.abs(f1.astype(int) - f2).max()))
            worst["fast_state"] = max(worst["fast_state"], float(np.abs(st[0, :39] - s1).max()))
            worst["fast_pixel"] = max(worst["fast_pixel"], int(np.abs(fr[0].astype(int) - f1).max()))
            worst["succ_mismatch"] += int(bool(st[0, 39]) != bool(ok1))
        print(f"{task}: over {args.trials} trials, max diffs {worst}", flush=True)
        assert worst["restore_state"] == 0 and worst["restore_pixel"] == 0, "restore is not exact"
        assert worst["fast_state"] == 0 and worst["fast_pixel"] == 0 and worst["succ_mismatch"] == 0, "fast rollout differs"

        acts = rng.uniform(-1, 1, (300, args.horizon, 4)).astype(np.float32)
        for render in (True, False):
            t0 = time.time()
            pool.rollout((task, snap["mw"]), acts, render)
            print(f"{task}: 300 rollouts of {args.horizon} steps, render={render}: {time.time() - t0:.2f}s", flush=True)
    pool.close()
    print("OK")


if __name__ == "__main__":
    main()
