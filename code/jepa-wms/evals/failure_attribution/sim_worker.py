# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Batched ground-truth rollouts in worker processes, for Push-T, Wall and Metaworld.

Each fast rollout reproduces what the env stack of the planning eval does between env.prepare and the
last env.step of an action sequence, without the per-step rendering, so that the simulator can replace
the world model inside the planner. tests/test_fast_rollout.py checks the equivalence.
"""

import multiprocessing as mp

import numpy as np

from evals.failure_attribution.pool_utils import rollout_map

_ENV = None
_KIND = None


def make_worker_env(kind):
    if kind == "pusht":
        from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv

        return PushTEnv(with_velocity=True, with_target=True)
    if kind == "wall":
        import torch
        from omegaconf import OmegaConf

        from evals.simu_env_planning.envs.wall_gym_wrap import make_env

        torch.set_num_threads(1)
        cfg = OmegaConf.create({"task_specification": {"task": "wall-base", "max_episode_steps": 30}})
        return make_env(cfg).env  # WallEnvWrapper around DotWall
    if kind == "mw":
        from evals.failure_attribution.mw_sim import WorkerEnv

        return WorkerEnv()
    raise ValueError(kind)


def _init_worker(kind):
    global _ENV, _KIND
    _ENV, _KIND = make_worker_env(kind), kind


def pusht_fast_rollout(env, spec, actions, render):
    """
    spec:    (seed, init_state (7,), shape)
    actions: (T, 2) float32 env actions (already denormalized)
    Returns the final state (7,) float32 and, if render, the final frame (H, W, 3) uint8.
    """
    from pymunk.vec2d import Vec2d

    seed, init_state, shape = spec
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


def wall_fast_rollout(wrapper, spec, actions, render):
    """
    spec:    (seed, init_state (2,), wall_x, hole_y)
    actions: (T, 2) float32 env actions (already denormalized)
    Returns the final dot position (2,) float32 and, if render, the final frame (H, W, 3) uint8.
    """
    import torch

    seed, init_state, wall_x, hole_y = spec
    env = wrapper.env  # DotWall
    if float(env.wall_x) != wall_x or float(env.hole_y) != hole_y or not hasattr(env, "wall_img"):
        env.wall_x = torch.tensor(wall_x, device=env.device).to(env.wall_x.dtype)
        env.hole_y = torch.tensor(hole_y, device=env.device).to(env.hole_y.dtype)
        env.wall_img = env._render_walls(env.wall_x, env.hole_y)
    # WallEnvWrapper.prepare, minus the frame rendered by DotWall.reset
    env.seed(seed)
    env.set_init_state(init_state)
    env.dot_position = env.reset_to_state.to(env.device)
    # DotWall.step, minus rendering
    for action in actions:
        env.dot_position = env._calculate_next_position(action)
    state = env.dot_position.float().numpy()
    frame = None
    if render:
        env.dot_img = env._render_dot(env.dot_position)
        frame = wrapper.render()
    return state, frame


def _mw_fast_rollout(wenv, spec, actions, render):
    from evals.failure_attribution.mw_sim import mw_fast_rollout

    return mw_fast_rollout(wenv, spec, actions, render)


FAST_ROLLOUT = {"pusht": pusht_fast_rollout, "wall": wall_fast_rollout, "mw": _mw_fast_rollout}


def _rollout_chunk(args):
    spec, actions, render = args
    states, frames = [], []
    for a in actions:
        s, f = FAST_ROLLOUT[_KIND](_ENV, spec, a, render)
        states.append(s)
        frames.append(f)
    return np.array(states), (np.array(frames) if render else None)


class SimPool:
    """Rolls out batches of action sequences from one prepared state, in parallel."""

    def __init__(self, kind, num_workers):
        self.kind, self.num_workers = kind, num_workers
        self.pool = mp.get_context("spawn").Pool(num_workers, initializer=_init_worker, initargs=(kind,))

    def rollout(self, spec, actions, render):
        """
        actions: (B, T, action_dim) float32
        Returns states (B, state_dim) float32 and frames (B, H, W, 3) uint8 or None.
        """
        chunks = np.array_split(np.arange(len(actions)), min(self.num_workers, len(actions)))
        jobs = [(spec, actions[c], render) for c in chunks if len(c)]
        out = rollout_map(self.pool, _rollout_chunk, jobs)
        states = np.concatenate([o[0] for o in out])
        frames = np.concatenate([o[1] for o in out]) if render else None
        return states, frames

    def close(self):
        self.pool.close()
        self.pool.join()
