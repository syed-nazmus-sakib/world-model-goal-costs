"""MeanTrap sanity checks for swm/TwoRoomTrap-v0 (CPU only, about a minute).

1. Straight line through the ice: the fall rate should be close to fall_prob.
2. Detour around the ice: never falls, always reaches the goal.
3. A fallen agent never moves again and never succeeds.
4. Saves a picture: normal agent above the ice | fallen (black) agent on it.

Usage:
    python scripts/data/check_tworoom_trap.py --fall_prob 0.5
"""

import argparse

import numpy as np
from PIL import Image

from stable_worldmodel.envs.two_room.trap_env import TwoRoomTrapEnv

START = (60.0, 80.0)  # above the ice
GOAL = (60.0, 160.0)  # below the ice
STRAIGHT = [GOAL]
DETOUR = [(30.0, 95.0), (30.0, 145.0), GOAL]  # left of the ice


def run_episode(env, seed, waypoints, max_steps=50):
    """Follow waypoints at full speed.

    Returns (success, fallen, moved_after_fall).
    """
    env.reset(seed=seed, options={'state': START, 'target_state': GOAL})
    waypoints = list(waypoints)
    fall_pos = None
    for _ in range(max_steps):
        pos = env.agent_position.numpy().copy()
        if len(waypoints) > 1 and np.linalg.norm(waypoints[0] - pos) < 5.0:
            waypoints.pop(0)
        direction = np.asarray(waypoints[0], dtype=np.float32) - pos
        norm = float(np.linalg.norm(direction))
        action = direction / norm if norm > 1e-8 else np.zeros(2)
        _, _, terminated, _, _ = env.step(action.astype(np.float32))
        if env.fallen and fall_pos is None:
            fall_pos = env.agent_position.numpy().copy()
        if terminated:
            return True, env.fallen, False
    moved = fall_pos is not None and not np.array_equal(
        fall_pos, env.agent_position.numpy()
    )
    return False, env.fallen, moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fall_prob', type=float, required=True)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--out', default='tworoom_trap_check.png')
    args = parser.parse_args()

    env = TwoRoomTrapEnv(fall_prob=args.fall_prob)
    print(f'fall_prob={env.fall_prob} -> per-step hazard={env.hazard:.4f}')

    for name, waypoints in (('straight', STRAIGHT), ('detour', DETOUR)):
        results = [
            run_episode(env, seed, [np.array(w) for w in waypoints])
            for seed in range(args.episodes)
        ]
        success, fallen, moved = (np.array(r) for r in zip(*results))
        print(
            f'{name:8s}: success {success.mean():.3f}  fall rate '
            f'{fallen.mean():.3f}  fallen-then-moved {int(moved.sum())}'
        )

    env.reset(seed=0, options={'state': START, 'target_state': GOAL})
    normal = env.render()
    env.agent_position[:] = env.agent_position.new_tensor([60.0, 120.0])
    env.fallen = True
    fallen_img = env.render()
    Image.fromarray(np.concatenate([normal, fallen_img], axis=1)).save(
        args.out
    )
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
