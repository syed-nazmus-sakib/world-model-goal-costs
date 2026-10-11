"""MeanTrap G2: checks for the oracle in eval_meantrap_oracle.py (CPU).

1. move() + fall_check() replay TwoRoomTrapEnv.step exactly: the same
   positions (bit for bit) and the same falls, over random action sequences
   (including actions outside [-1, 1]) from starts all over the map, near the
   door and around the ice. The env's own random draws are replayed.
2. on_ice() equals env.on_ice on a 0.5 px grid around the ice.
3. OracleCost on a hand-made task (start (60, 75), goal (60, 165)): a
   straight plan crosses the ice for 8 steps; a detour goes around it. Both
   end exactly at the goal if nothing falls.
   - fall_prob 0.5: O1 (expected) must match the analytic expected cost of
     the straight plan, and give the detour cost 0; O3 (mean) must give the
     straight plan only a small cost (it is slowed, never falls).
   - fall_prob 0: both oracles give both plans cost 0.

Usage:
    python scripts/plan/check_meantrap_oracle.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_meantrap_oracle import (  # noqa: E402
    OracleCost,
    dynamics_params,
    fall_check,
    move,
    on_ice,
)
from stable_worldmodel.envs.two_room.trap_env import TwoRoomTrapEnv  # noqa: E402

START = np.array([60.0, 75.0], dtype=np.float32)
GOAL = np.array([60.0, 165.0], dtype=np.float32)
STRAIGHT = [(0, 1)] * 18 + [(0, 0)] * 7
DETOUR = [(-1, 1)] * 5 + [(0, 1)] * 8 + [(1, 1)] * 5 + [(0, 0)] * 7


def fail(msg):
    raise SystemExit(f'[oracle-check] FAILED: {msg}')


def check_replay(n_eps, steps, seed):
    rng = np.random.default_rng(seed)
    env = TwoRoomTrapEnv(fall_prob=0.5)
    count = dict(steps=0, mismatches=0, wall_stops=0, room_changes=0,
                 border_stops=0, falls=0, frozen_steps=0)
    for ep in range(n_eps):
        kind = ep % 3
        if kind == 0:
            start = rng.uniform(21, 203, 2)
        elif kind == 1:  # near the wall and the door
            start = (rng.uniform(95, 129), rng.uniform(25, 75))
        else:  # around the ice
            start = (rng.uniform(30, 90), rng.uniform(85, 155))
        env.reset(seed=ep, options={
            'state': np.asarray(start, dtype=np.float32),
            'target_state': np.array([200.0, 30.0], dtype=np.float32)})
        p = dynamics_params(env)
        draws = copy.deepcopy(env.np_random)  # replays the env's draws
        pos = env.agent_position.numpy().copy()
        fallen = np.array(False)
        for _ in range(steps):
            a = rng.normal(0.0, 1.5, 2).astype(np.float32)
            if kind == 1:
                a[0] = np.float32(rng.choice([-2.0, 2.0]))
            env.step(a)
            if fallen:
                count['frozen_steps'] += 1
            else:
                new = move(pos, a, p)
                u = draws.random() if (p['hazard'] > 0 and on_ice(new)) else 1.0
                was_fallen = bool(fallen)
                fallen = fall_check(fallen, new, u, p['hazard'])
                count['falls'] += int(bool(fallen) and not was_fallen)
                count['wall_stops'] += int(
                    new[0] in (np.float32(p['eff_left'] - 0.5),
                               np.float32(p['eff_right'] + 0.5)))
                count['room_changes'] += int(
                    (pos[0] < p['center']) != (new[0] < p['center']))
                count['border_stops'] += int(
                    np.any(np.isin(new, np.float32([p['lo'], p['hi']]))))
                pos = new
            count['steps'] += 1
            same = (np.array_equal(pos, env.agent_position.numpy())
                    and bool(fallen) == bool(env.fallen))
            count['mismatches'] += int(not same)
    print(f'[oracle-check] 1. replay vs env: {count}')
    if count['mismatches']:
        fail('vectorized dynamics differ from the env')
    if min(v for key, v in count.items() if key != 'mismatches') == 0:
        fail('a case was never exercised')


def check_on_ice():
    env = TwoRoomTrapEnv(fall_prob=0.5)
    xs, ys = np.arange(30, 90, 0.5), np.arange(90, 150, 0.5)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    ours = on_ice(grid.astype(np.float32))
    ref = np.array([env.on_ice(q) for q in grid])
    print(f'[oracle-check] 2. on_ice on {len(grid)} grid points: '
          f'{int((ours != ref).sum())} differences, {int(ref.sum())} on ice')
    if (ours != ref).any():
        fail('on_ice differs from the env')


def plan_costs(fall_prob, oracle, k=4096):
    env = TwoRoomTrapEnv(fall_prob=fall_prob)
    env.reset(seed=0, options={'state': START, 'target_state': GOAL})
    plans = np.array([STRAIGHT, DETOUR], dtype=np.float32)  # env actions
    # identity normalization: mean 0, std 1
    scaler = StandardScaler().fit(np.array([[-1, -1], [1, 1]], np.float32))
    S = len(plans)
    info = {
        'state': torch.as_tensor(START).view(1, 1, 1, 2).expand(1, S, 1, 2),
        'goal_state': torch.as_tensor(GOAL).view(1, 1, 1, 2).expand(
            1, S, 1, 2),
        'fallen': torch.zeros(1, S, 1, dtype=torch.bool),
    }
    candidates = torch.as_tensor(plans.reshape(1, S, 5, 10))
    cost = OracleCost(oracle, [env], scaler, k=k, seed=0)
    return cost.get_cost(info, candidates).numpy()[0], env.hazard


def check_costs():
    (o1_straight, o1_detour), h = plan_costs(0.5, 'expected')
    # analytic: falls on the j-th ice step (y = 100 + 5 j) w.p. h (1-h)^j
    j = np.arange(8)
    analytic = float(np.sum(h * (1 - h) ** j * (65.0 - 5.0 * j) ** 2))
    (o3_straight, o3_detour), _ = plan_costs(0.5, 'mean')
    print(f'[oracle-check] 3. fall_prob 0.5: O1 straight {o1_straight:.1f} '
          f'(analytic {analytic:.1f}), O1 detour {o1_detour:.1f}; '
          f'O3 straight {o3_straight:.2f}, O3 detour {o3_detour:.2f}')
    if abs(o1_straight - analytic) > 100 or o1_detour != 0:
        fail('O1 costs are wrong')
    if not (0 < o3_straight < 25) or o3_detour != 0:
        fail('O3 costs are wrong')
    zero = [plan_costs(0.0, o)[0] for o in ('expected', 'mean')]
    print(f'[oracle-check]    fall_prob 0: O1 {zero[0].tolist()}, '
          f'O3 {zero[1].tolist()}')
    if any(np.any(z != 0) for z in zero):
        fail('with fall_prob 0 both plans must cost 0')


def main():
    check_replay(n_eps=1500, steps=50, seed=0)
    check_on_ice()
    check_costs()
    print('[oracle-check] ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
