# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Batched ground-truth Push-T rollouts in worker processes.

fast_rollout reproduces PushTWrapper.prepare followed by PushTEnv.step for every action, without the
per-step rendering and reward computation, so that the simulator can replace the world model inside
the planner. This module must stay free of torch imports: it is loaded by spawned workers.
"""

import multiprocessing as mp

import numpy as np

from evals.failure_attribution.pool_utils import rollout_map
from pymunk.vec2d import Vec2d

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

_ENV = None


def _init_worker():
    global _ENV
    _ENV = PushTEnv(with_velocity=True, with_target=True)


def fast_rollout(env, seed, init_state, shape, actions, render):
    """
    env:        PushTEnv
    init_state: (7,) simulator state the episode is prepared with
    actions:    (T, 2) float32 env actions (already denormalized)
    Returns the final state (7,) float32 and, if render, the final frame (H, W, 3) uint8.
    """
    # PushTWrapper.update_env + PushTWrapper.prepare, minus the frame rendered by PushTEnv.reset
    env.shape = shape
    env.seed(seed)
    env.reset_to_state = init_state
    env._setup()
    env._set_state(init_state)
    # PushTEnv.step, minus reward and rendering
    dt = 1.0 / env.sim_hz
    n_steps = env.sim_hz // env.control_hz
    for action in actions:
        action = np.array(action) * env.action_scale
        if env.relative:
            action = env.agent.position + action
        for _ in range(n_steps):
            acceleration = env.k_p * (action - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acceleration * dt
            env.space.step(dt)
    state = env._get_obs()
    frame = env._render_frame("rgb_array") if render else None
    return state, frame


def _rollout_chunk(args):
    seed, init_state, shape, actions, render = args
    states, frames = [], []
    for a in actions:
        s, f = fast_rollout(_ENV, seed, init_state, shape, a, render)
        states.append(s)
        frames.append(f)
    return np.array(states), (np.array(frames) if render else None)


class SimPool:
    """Rolls out batches of action sequences from one prepared state, in parallel."""

    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.pool = mp.get_context("spawn").Pool(num_workers, initializer=_init_worker)

    def rollout(self, seed, init_state, shape, actions, render):
        """
        actions: (B, T, 2) float32
        Returns states (B, 7) float32 and frames (B, H, W, 3) uint8 or None.
        """
        chunks = np.array_split(np.arange(len(actions)), min(self.num_workers, len(actions)))
        jobs = [(seed, init_state, shape, actions[c], render) for c in chunks if len(c)]
        out = rollout_map(self.pool, _rollout_chunk, jobs)
        states = np.concatenate([o[0] for o in out])
        frames = np.concatenate([o[1] for o in out]) if render else None
        return states, frames

    def close(self):
        self.pool.close()
        self.pool.join()
