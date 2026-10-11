# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Exact-state reset / render harness for Push-T, and the decomposition of the
state error between two Push-T states into official, block-only and pusher-only parts.

State layout (with_velocity=True): [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]
"""

import numpy as np
import shapely.geometry as sg
import torch
from omegaconf import OmegaConf

from evals.simu_env_planning.envs.pusht_env.pusht_env import pymunk_to_shapely
from evals.simu_env_planning.envs.pusht_gym_wrap import make_env as make_pusht_env
from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper

AGENT_RADIUS = 15.0
# Walls are segments of radius 2 at 5 and 506
WALL_LOW, WALL_HIGH = 7.0, 504.0


def angle_diff(a, b):
    """Absolute wrapped angle difference, same convention as PushTWrapper.eval_state."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


class PushTStateHarness:
    """
    Wraps the same env stack as the planning eval (PushTEnv -> PushTWrapper -> TimeLimit
    -> TensorWrapper -> PixelWrapper) and exposes exact-state resets through env.prepare(seed, state).
    """

    def __init__(self, img_size=224, with_target=True, max_episode_steps=30):
        self.cfg = OmegaConf.create(
            {
                "task_specification": {
                    "task": "pusht-base",
                    "obs": "rgb_state",
                    "obs_concat_channels": False,
                    "num_frames": 1,
                    "num_proprios": 1,
                    "img_size": img_size,
                    "max_episode_steps": max_episode_steps,
                    "env": {"with_target": with_target, "with_velocity": True},
                }
            }
        )
        env = make_pusht_env(self.cfg)
        env = TensorWrapper(env)
        self.env = PixelWrapper(self.cfg, env)
        self.env.reset()
        self._block_local_kps = None

    @property
    def pusht(self):
        return self.env.unwrapped

    # ------------------------------------------------------------------ #
    # Geometry (no physics step involved)
    # ------------------------------------------------------------------ #
    def _pose_block(self, block_pose):
        """Places the live block body at block_pose = (x, y, angle) without stepping physics."""
        block = self.pusht.block
        # same assignment order as PushTEnv._set_state (non-legacy)
        block.angle = float(block_pose[2])
        block.position = (float(block_pose[0]), float(block_pose[1]))
        return block

    def block_geom(self, block_pose):
        block = self._pose_block(block_pose)
        return pymunk_to_shapely(block, block.shapes)

    def block_keypoints(self, block_pose):
        """World coordinates of the T polygon vertices, shape (n_vertices, 2)."""
        block = self._pose_block(block_pose)
        kps = [block.local_to_world(v) for shape in block.shapes for v in shape.get_vertices()]
        return np.array([[p.x, p.y] for p in kps], dtype=np.float64)

    def is_valid(self, state, margin=2.0):
        """
        A state is valid if agent and block are inside the walls and the pusher disc
        is at least `margin` px away from the block (no contact, so the physics step
        in PushTEnv._set_state cannot move the block).
        """
        state = np.asarray(state, dtype=np.float64)
        agent = state[:2]
        lo, hi = WALL_LOW + AGENT_RADIUS + margin, WALL_HIGH - AGENT_RADIUS - margin
        if np.any(agent < lo) or np.any(agent > hi):
            return False
        kps = self.block_keypoints(state[2:5])
        if np.any(kps < WALL_LOW + margin) or np.any(kps > WALL_HIGH - margin):
            return False
        geom = self.block_geom(state[2:5])
        return geom.distance(sg.Point(agent[0], agent[1])) >= AGENT_RADIUS + margin

    # ------------------------------------------------------------------ #
    # Exact-state reset
    # ------------------------------------------------------------------ #
    def reset_to(self, state, seed=0):
        """
        Returns dict with
            visual:   uint8 tensor (1, 3, H, W), as fed to EncPredWM.encode
            proprio:  float tensor (1, 4)
            realized: np.ndarray (7,), the simulator state after the reset
        """
        state = np.asarray(state, dtype=np.float64)
        obs, info = self.env.prepare(seed, state)
        return {
            "visual": obs.clone(),
            "proprio": info["proprio"].clone().float(),
            "realized": np.asarray(info["state"], dtype=np.float64),
        }

    @staticmethod
    def reset_error(requested, realized):
        """Discrepancy between requested and realized state, per component."""
        requested = np.asarray(requested, dtype=np.float64)
        realized = np.asarray(realized, dtype=np.float64)
        return {
            "block_pos": float(np.linalg.norm(requested[2:4] - realized[2:4])),
            "block_ang": float(angle_diff(requested[4], realized[4])),
            "pusher_pos": float(np.linalg.norm(requested[:2] - realized[:2])),
            "pusher_vel": float(np.linalg.norm(requested[5:] - realized[5:])),
        }

    # ------------------------------------------------------------------ #
    # Error decomposition
    # ------------------------------------------------------------------ #
    def decompose(self, goal_state, cur_state):
        """
        A. official: PushTWrapper.eval_state (success, state_dist) + its two ingredients
        B. block-only: block xy error, block angle error, RMS displacement of T vertices
        C. pusher-only: pusher xy error, pusher velocity error
        """
        goal_state = np.asarray(goal_state, dtype=np.float64)
        cur_state = np.asarray(cur_state, dtype=np.float64)
        official = self.env.eval_state(goal_state, cur_state)
        kp_goal = self.block_keypoints(goal_state[2:5])
        kp_cur = self.block_keypoints(cur_state[2:5])
        return {
            # A
            "official_success": bool(official["success"]),
            "official_state_dist": float(official["state_dist"]),
            "official_pos_diff": float(np.linalg.norm(goal_state[:4] - cur_state[:4])),
            # B
            "block_pos_err": float(np.linalg.norm(goal_state[2:4] - cur_state[2:4])),
            "block_ang_err": float(angle_diff(goal_state[4], cur_state[4])),
            "block_kp_err": float(np.sqrt(((kp_goal - kp_cur) ** 2).sum(-1).mean())),
            # C
            "pusher_pos_err": float(np.linalg.norm(goal_state[:2] - cur_state[:2])),
            "pusher_vel_err": float(np.linalg.norm(goal_state[5:] - cur_state[5:])),
        }


def latent_goal_costs(enc, goal_enc):
    """
    Planning cost terms of ReprTargetDistMPCObjective for encoded terminal states.

    enc:      TensorDict with visual (B, 1, V, H, W, D) and proprio (B, 1, N, P)
    goal_enc: TensorDict with visual (1, 1, V, H, W, D) and proprio (1, 1, N, P)
    Returns (d_visual, d_proprio), each of shape (B,). The planner minimizes
    d_visual + alpha * d_proprio.
    """
    d_vis = (goal_enc["visual"] - enc["visual"]).pow(2).flatten(1).mean(1)
    d_prop = (goal_enc["proprio"] - enc["proprio"]).pow(2).flatten(1).mean(1)
    return d_vis, d_prop


def to_td(visual, proprio):
    """Stacks per-state (1, 3, H, W) / (1, 4) tensors into the batch layout EncPredWM.encode expects."""
    from tensordict import TensorDict

    return TensorDict({"visual": torch.stack(visual), "proprio": torch.stack(proprio)}, batch_size=[len(visual)])
