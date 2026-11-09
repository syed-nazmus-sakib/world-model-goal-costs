"""Goal-conditioned OGBench AntMaze from pixels, `swm/AntMaze-v0`.

The quadruped of OGBench's antmaze tasks in the maze picked by `task` ('medium' / 'large' / 'giant'), seen as in OGBench's own
visual-antmaze data: the camera follows the robot from above and behind, the floor is coloured so that the position can be
read from the image, and one leg is white so that the heading can. Only the image size differs (224 px instead of 64).
The state is the full simulator state [qpos (15), qvel (14)]; replaying it is exact, so the training images are rendered
through this same class (code/lewm_explorations/paa/antmaze_data.py). Success is the torso within `goal_threshold`
(OGBench's 0.5 by default) of the goal frame's position. No goal marker is drawn.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ogbench.locomaze.maze import make_maze_env

from stable_worldmodel.envs.ogbench.pointmaze_env import PointMazeEnv

ENV_ID = "swm/AntMaze-v0"


class AntMazeEnv(PointMazeEnv):
    def __init__(self, task: str = "large", resolution: int = 224, goal_threshold: float | None = None, **kwargs):
        """
        Args:
            task (str): OGBench maze type ('medium', 'large' or 'giant'; 'teleport' is not supported)
            resolution (int): square render size, in pixels
            goal_threshold (float | None): success distance to the goal; None = OGBench's
        """
        gym.Env.__init__(self)
        assert task != "teleport", "teleport mazes need OGBench's own step()"
        self.task = str(task)
        self.res = int(resolution)
        # built at OGBench's 64 px (the model's offscreen buffer is too small for more), then re-sized below
        self.env = make_maze_env("ant", "maze", maze_type=self.task, ob_type="pixels", render_mode="rgb_array",
                                 width=64, height=64, camera_name="back")
        self.u = self.env.unwrapped
        self.nq, self.nv = int(self.u.model.nq), int(self.u.model.nv)
        if getattr(self.u, "custom_renderer", None) is not None:
            self.u.custom_renderer.close()
        self._configure_renderer()
        self.goal_threshold = float(getattr(self.u, "_goal_tol", 0.5) if goal_threshold is None else goal_threshold)
        self._goal_state = None
        self._goal_xy = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(int(self.u.model.nu),), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
        })
        self.render_mode = "rgb_array"

    def step(self, action):
        # the simulator step of OGBench's AntEnv.step, without the image it renders as its observation on every call
        self.u.do_simulation(np.clip(np.asarray(action, np.float64), -1.0, 1.0), self.u.frame_skip)
        dist = self._goal_distance()
        terminated = self._goal_xy is not None and dist < self.goal_threshold
        reward = 0.0 if self._goal_xy is None else -dist
        return self._obs(), float(reward), bool(terminated), False, {"goal_distance": dist}


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=AntMazeEnv)
