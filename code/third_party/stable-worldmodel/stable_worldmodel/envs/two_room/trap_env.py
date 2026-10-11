"""TwoRoom with a thin-ice patch (MeanTrap project).

Identical to TwoRoomEnv except:

- A fixed ice patch (pixel columns ICE_X, rows ICE_Y) is drawn in the left
  room, in every frame.
- After each step, if the agent's center is on the ice, the agent falls
  through with probability ``hazard``. Falling is irreversible: the agent
  stops moving for the rest of the episode, is drawn black, and can no
  longer succeed.
- ``fall_prob`` is the probability of falling when crossing the ice in a
  straight line at the default speed (CROSSING_STEPS steps on the ice):
  hazard = 1 - (1 - fall_prob) ** (1 / CROSSING_STEPS).
  fall_prob=0 is the deterministic twin: same pixels, never falls.

Importing this module registers ``swm/TwoRoomTrap-v0``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .env import TwoRoomEnv


class TwoRoomTrapEnv(TwoRoomEnv):
    ICE_X = (40, 80)  # pixel columns [40, 80)
    ICE_Y = (100, 140)  # pixel rows [100, 140)
    ICE_COLOR = (173, 216, 230)  # LightBlue
    FALLEN_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    CROSSING_STEPS = 8  # 40 px of ice / 5 px per step (default speed)

    def __init__(self, fall_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= fall_prob <= 1.0:
            raise ValueError(f'fall_prob must be in [0, 1], got {fall_prob}')
        self.fall_prob = float(fall_prob)
        self.hazard = 1.0 - (1.0 - self.fall_prob) ** (
            1.0 / self.CROSSING_STEPS
        )
        self.fallen = False
        self.ice_mask = (
            (self.grid_x >= self.ICE_X[0])
            & (self.grid_x < self.ICE_X[1])
            & (self.grid_y >= self.ICE_Y[0])
            & (self.grid_y < self.ICE_Y[1])
        )

    def on_ice(self, pos_xy) -> bool:
        x, y = float(pos_xy[0]), float(pos_xy[1])
        return (
            self.ICE_X[0] <= x < self.ICE_X[1]
            and self.ICE_Y[0] <= y < self.ICE_Y[1]
        )

    def reset(self, seed=None, options=None):
        # Clear before the parent renders the target image in reset().
        self.fallen = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.fallen:
            # Irreversible: no movement, no success.
            obs = self._get_obs()
            info = self._get_info()
            info['distance_to_target'] = float(
                torch.norm(self.agent_position - self.target_position)
            )
            return obs, 0.0, False, False, info

        obs, reward, terminated, truncated, info = super().step(action)
        if self.hazard > 0.0 and self.on_ice(self.agent_position):
            if self.np_random.random() < self.hazard:
                self.fallen = True
                terminated = False
        info['fallen'] = np.array(self.fallen)
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        info = super()._get_info()
        info['fallen'] = np.array(self.fallen)
        return info

    def _render_frame(self, agent_pos: torch.Tensor):
        # Same as TwoRoomEnv._render_frame, plus the ice patch (drawn under
        # the target and the agent) and a black agent once it has fallen.
        H = W = self.IMG_SIZE

        bg = self.variation_space['background']['color'].value
        img = torch.empty((3, H, W), dtype=torch.uint8)
        img[0].fill_(int(bg[0]))
        img[1].fill_(int(bg[1]))
        img[2].fill_(int(bg[2]))

        wall_mask, door_mask = self._wall_and_door_masks()

        door_color = self.variation_space['door']['color'].value
        if door_mask.any():
            img[0, door_mask] = int(door_color[0])
            img[1, door_mask] = int(door_color[1])
            img[2, door_mask] = int(door_color[2])

        wall_color = self.variation_space['wall']['color'].value
        if wall_mask.any():
            img[0, wall_mask] = int(wall_color[0])
            img[1, wall_mask] = int(wall_color[1])
            img[2, wall_mask] = int(wall_color[2])

        # ice (MeanTrap)
        img[0, self.ice_mask] = self.ICE_COLOR[0]
        img[1, self.ice_mask] = self.ICE_COLOR[1]
        img[2, self.ice_mask] = self.ICE_COLOR[2]

        render_target = (
            bool(self.variation_space['rendering']['render_target'].value)
            or self.render_target_flag
        )
        if render_target:
            tgt_color = self.variation_space['target']['color'].value
            tgt_r = float(
                self.variation_space['target']['radius'].value.item()
            )
            tgt_dot = self._gaussian_dot(self.target_position, tgt_r)
            img = self._alpha_blend(img, tgt_dot, tgt_color)

        # agent (black once it has fallen; MeanTrap)
        if self.fallen:
            agent_color = self.FALLEN_COLOR
        else:
            agent_color = self.variation_space['agent']['color'].value
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        agent_dot = self._gaussian_dot(agent_pos, agent_r)
        img = self._alpha_blend(img, agent_dot, agent_color)

        return img


if 'swm/TwoRoomTrap-v0' not in gym.registry:
    gym.register(
        id='swm/TwoRoomTrap-v0',
        entry_point='stable_worldmodel.envs.two_room.trap_env:TwoRoomTrapEnv',
    )
