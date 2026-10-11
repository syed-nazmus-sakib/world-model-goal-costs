# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
The simulator-in-the-loop planner relies on fast_rollout giving exactly the states and frames of the
env stack used by the planning eval.

Run with: python -m tests.test_pusht_fast_rollout
"""

import time

import numpy as np
import torch

from evals.pusht_failure_attribution.sim_worker import SimPool, fast_rollout
from evals.pusht_goal_decomp.harness import PushTStateHarness
from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv


def main():
    env = PushTStateHarness().env
    lean = PushTEnv(with_velocity=True, with_target=True)
    rs = np.random.RandomState(0)
    worst_state, worst_pixel = 0.0, 0
    for trial in range(20):
        s0 = np.array(
            [
                rs.uniform(100, 400),
                rs.uniform(100, 400),
                rs.uniform(150, 350),
                rs.uniform(150, 350),
                rs.uniform(0, 6.28),
                rs.randn() * 50,
                rs.randn() * 50,
            ]
        )
        actions = torch.from_numpy((rs.randn(30, 2) * 0.3).astype(np.float32))
        # the path the planning eval uses: update_env, prepare, step_multiple
        env.update_env({"shape": "T"})
        obses, infos = env.rollout(trial, s0, actions, {"shape": "T"})
        ref_state, ref_frame = np.asarray(infos[-1]["state"]), obses[-1][0].numpy()
        state, frame = fast_rollout(lean, trial, s0, "T", actions.numpy(), True)
        worst_state = max(worst_state, float(np.abs(state - ref_state).max()))
        worst_pixel = max(worst_pixel, int(np.abs(frame.transpose(2, 0, 1).astype(int) - ref_frame.astype(int)).max()))
    print(f"max |state diff| fast vs env rollout over 20 trials: {worst_state}, max pixel diff: {worst_pixel}")
    assert worst_state == 0.0 and worst_pixel == 0

    pool = SimPool(12)
    A = (rs.randn(300, 30, 2) * 0.3).astype(np.float32)
    pool.rollout(0, s0, "T", A, True)
    t = time.time()
    S, F = pool.rollout(0, s0, "T", A, True)
    t_render = time.time() - t
    t = time.time()
    S2, _ = pool.rollout(0, s0, "T", A, False)
    t_plain = time.time() - t
    serial, _ = fast_rollout(lean, 0, s0, "T", A[7], False)
    print(f"pool, 300 rollouts of 30 steps: {t_render:.2f}s with render, {t_plain:.2f}s without; {S.shape} {F.shape}")
    assert np.array_equal(S[7], serial) and np.array_equal(S, S2)
    pool.close()
    print("OK")


if __name__ == "__main__":
    main()
