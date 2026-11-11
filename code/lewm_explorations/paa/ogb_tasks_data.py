"""The official OGBench evaluation goals of a maze, as small datasets the evaluation script can read.

OGBench scores a maze on 5 fixed tasks: a start cell and a goal cell each, with uniform noise on both positions
(ogbench/locomaze/maze.py: set_tasks, add_noise), 50 episodes per task, 1,000 steps, success when the agent is within the
environment's goal tolerance.  This script takes the tasks and the noise from the installed OGBench itself (nothing is copied
by hand) and writes, for task k, one file with E episodes of two frames each: frame 0 = start, frame 1 = goal, both rendered by
the same class the evaluation uses (zero velocity, as after OGBench's own reset).  eval_wm.py then runs them like any other
start-goal pair:
  eval.dataset_name=<out>_s<seed>_t<k>.h5  eval.goal_offset_steps=1  eval.num_eval=E  eval.eval_budget=1000
  +stats_dataset=<training data>.h5        (action and state normalisation from the training data, not from these files)
Output: $STABLEWM_HOME/datasets/<out>_s<seed>_t<k>.h5 for k = 1..5 and <out>_s<seed>_tasks.json (cells, positions, tolerance).

usage: python ogb_tasks_data.py [--task large] [--out pointmaze_large_ogb] [--episodes 50] [--seed 0] [--preview goals.png]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='large')
    ap.add_argument('--out', default='pointmaze_large_ogb')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0, help='seed of the position noise; one set of files per seed')
    ap.add_argument('--res', type=int, default=224)
    ap.add_argument('--preview', default=None, help='PNG: start (top) and goal (bottom) of the first episode of every task')
    a = ap.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import h5py
    import hdf5plugin
    from stable_worldmodel.envs.ogbench.pointmaze_env import PointMazeEnv

    home = os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel'))
    dd = os.path.join(home, 'datasets'); os.makedirs(dd, exist_ok=True)
    env = PointMazeEnv(task=a.task, resolution=a.res); env.reset(seed=0); u = env.u
    tasks = list(u.task_infos); nq, nv = env.nq, env.nv; noisy_goal = bool(getattr(u, '_add_noise_to_goal', True))
    meta = dict(maze=a.task, seed=a.seed, episodes=a.episodes, goal_tolerance=float(env.goal_threshold), noise=float(u._noise),
                noise_on_goal=noisy_goal, tasks=[])
    print(f'[tasks] maze {a.task}: {len(tasks)} official tasks, position noise +-{u._noise}, goal tolerance {env.goal_threshold} '
          f'(this OGBench version), {a.episodes} episodes each, noise seed {a.seed}', flush=True)

    def frame(xy):
        s = np.zeros(nq + nv, np.float64); s[:2] = xy       # the point: position, zero velocity
        env.set_state(s)
        return s.astype(np.float32), env.render()

    prev = []
    for k, t in enumerate(tasks, 1):
        np.random.seed(1000 * a.seed + k)                   # OGBench draws its noise from numpy's global generator
        st, px, xy = [], [], []
        for _ in range(a.episodes):
            i_xy = np.asarray(u.add_noise(u.ij_to_xy(t['init_ij'])), np.float64)
            g_xy = np.asarray(u.ij_to_xy(t['goal_ij']), np.float64)
            g_xy = np.asarray(u.add_noise(g_xy), np.float64) if noisy_goal else g_xy
            for p in (i_xy, g_xy):
                s, f = frame(p); st.append(s); px.append(f)
            xy.append([*i_xy, *g_xy])
        st, px = np.stack(st), np.stack(px); N = len(st); E = a.episodes
        path = os.path.join(dd, f'{a.out}_s{a.seed}_t{k}.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('pixels', data=px, chunks=(1, a.res, a.res, 3), **hdf5plugin.Zstd(clevel=3))
            f.create_dataset('action', data=np.zeros((N, int(u.model.nu)), np.float32))   # unused: statistics come from +stats_dataset
            f.create_dataset('proprio', data=st); f.create_dataset('state', data=st)
            f.create_dataset('ep_idx', data=np.repeat(np.arange(E), 2).astype(np.int32))
            f.create_dataset('step_idx', data=np.tile(np.arange(2), E).astype(np.int32))
            f.create_dataset('id', data=np.arange(N, dtype=np.int64))
            f.create_dataset('ep_len', data=np.full(E, 2, np.int32)); f.create_dataset('ep_offset', data=(2 * np.arange(E)).astype(np.int64))
            f.attrs['source'] = f'OGBench {a.task} maze, official task {k}'
        d = np.linalg.norm(np.asarray(xy)[:, :2] - np.asarray(xy)[:, 2:], axis=1)
        meta['tasks'].append(dict(task=k, init_ij=list(map(int, t['init_ij'])), goal_ij=list(map(int, t['goal_ij'])), file=os.path.basename(path),
                                  straight_line_distance=float(d.mean()), xy=np.asarray(xy).round(4).tolist()))
        print(f'[tasks] task {k}: start cell {tuple(t["init_ij"])} -> goal cell {tuple(t["goal_ij"])} | straight-line distance '
              f'{d.mean():.1f} | {path}', flush=True)
        prev.append(np.concatenate([px[0], px[1]], 0))
    json.dump(meta, open(os.path.join(dd, f'{a.out}_s{a.seed}_tasks.json'), 'w'), indent=1)
    if a.preview:
        try:
            from PIL import Image
            Image.fromarray(np.concatenate(prev, 1)).save(a.preview); print('[tasks] preview', a.preview, flush=True)
        except Exception as x:  # noqa: BLE001
            print('[tasks] no preview:', x, flush=True)
    print('TASKS_DONE', len(tasks), flush=True)


if __name__ == '__main__':
    main()
