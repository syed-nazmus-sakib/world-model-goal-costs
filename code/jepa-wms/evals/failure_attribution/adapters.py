# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""What the failure-attribution run needs to know about each environment."""

import numpy as np


def angle_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


class PushTAdapter:
    kind = "pusht"
    action_dim = 2
    has_expert = True  # the goal is the end of a dataset segment, whose actions reach it

    def spec(self, prepared, env):
        shape = prepared["env_info"]["shape"] if prepared["env_info"] is not None else "T"
        return (prepared["seed"], prepared["init_state"], shape)

    def spec_to_save(self, spec):
        return {"shape": spec[2]}

    def proprio(self, states):
        return states[:, [0, 1, 5, 6]]

    def true_cost(self, states, goal):
        """< 1 exactly when PushTWrapper.eval_state reports success."""
        pos = np.linalg.norm(states[:, :4] - goal[:4], axis=1)
        return np.maximum(pos / 20.0, angle_diff(states[:, 4], goal[4]) / (np.pi / 9))

    def errors(self, state, goal):
        return {
            "agent_err": float(np.linalg.norm(state[:2] - goal[:2])),
            "object_err": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "object_ang_err": float(angle_diff(state[4], goal[4])),
        }

    def describe(self, init, goal, spec):
        return (
            f"block must move {np.linalg.norm(init[2:4] - goal[2:4]):.0f} px and rotate "
            f"{np.degrees(angle_diff(init[4], goal[4])):.0f} deg, pusher {np.linalg.norm(init[:2] - goal[:2]):.0f} px"
        )


class WallAdapter:
    kind = "wall"
    action_dim = 2
    has_expert = False  # init and goal are random states on opposite sides of the wall

    def spec(self, prepared, env):
        dot_wall = env.unwrapped
        return (prepared["seed"], prepared["init_state"], float(dot_wall.wall_x), float(dot_wall.hole_y))

    def spec_to_save(self, spec):
        return {"wall_x": spec[2], "hole_y": spec[3]}

    def proprio(self, states):
        return states

    def true_cost(self, states, goal):
        """< 1 exactly when WallEnvWrapper.eval_state reports success."""
        return np.linalg.norm(states[:, :2] - goal[:2], axis=1) / 4.5

    def errors(self, state, goal):
        return {"agent_err": float(np.linalg.norm(state[:2] - goal[:2]))}

    def describe(self, init, goal, spec):
        return (
            f"dot must travel {np.linalg.norm(init[:2] - goal[:2]):.1f} px through the door at y={spec[3]:.0f} "
            f"of the wall at x={spec[2]:.0f}"
        )


class MetaworldAdapter:
    """States from the mw SimPool are [39-d obs, success, obj_to_target] (see mw_sim.mw_fast_rollout)."""

    kind = "mw"
    action_dim = 4
    has_expert = False

    def __init__(self, task):
        from evals.failure_attribution.mw_sim import SUCCESS_THRESHOLD

        self.task = task
        self.threshold = SUCCESS_THRESHOLD.get(task, 0.05)

    def proprio(self, states):
        return states[:, :4]

    def true_cost(self, states, goal):
        """obj_to_target / success threshold: < 1 exactly when Metaworld reports success (distance tasks).
        Door tasks report obj_to_target = 0 always (found 2026-10-23); their success test is |handle x - target x| <= 0.08,
        with the target x in obs[36] (the goal field of the observation), so the cost uses that instead."""
        if self.task in ("mw-door-open", "mw-door-close", "mw-door-lock", "mw-door-unlock"):
            return np.abs(states[:, 4] - states[:, 36]) / 0.08
        return states[:, 40] / self.threshold

    def errors(self, state, goal):
        return {
            "hand_err": float(np.linalg.norm(state[:3] - goal[:3])),
            "obj_err": float(np.linalg.norm(state[4:7] - goal[4:7])),
            "obj_to_target": float(state[40]) if len(state) > 40 else float("nan"),
        }

    def describe(self, init, goal, spec=None):
        return f"hand must move {np.linalg.norm(init[:3] - goal[:3]):.3f} m, object {np.linalg.norm(init[4:7] - goal[4:7]):.3f} m"


def get_adapter(task):
    if task.startswith("mw-"):
        return MetaworldAdapter(task)
    if task.startswith("pusht"):
        return PushTAdapter()
    if task.startswith("wall"):
        return WallAdapter()
    raise ValueError(f"no failure-attribution adapter for task {task}")
