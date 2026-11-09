"""OGBench AntMaze as an image dataset for LeWM: download the official state dataset and render every frame at 224 px.

Source: https://rail.eecs.berkeley.edu/datasets/ogbench/<dataset>.npz (default antmaze-large-navigate-v0, 1,000 episodes of
1,001 steps, 243 MB: observations, actions, terminals, qpos, qvel).  OGBench's own image version is 64 px; LeWM is trained at
224 px, so each recorded simulator state is replayed through swm/AntMaze-v0 (the same class the evaluation uses, same camera,
coloured floor and white leg as OGBench's visual-antmaze) and rendered.  Output, in the layout of the other LeWM datasets:
  $STABLEWM_HOME/datasets/<out>.h5   pixels (N,224,224,3) uint8 | action (N,8) | proprio = state = [qpos, qvel] (N,29) |
                                     ep_idx, step_idx, id (N,) | ep_len, ep_offset (episodes,)
Progress is printed every few seconds with the speed and the time left.  The run can be stopped and started again: the
download resumes, and rendering continues after the last finished episode (<out>.h5.part until it is complete).

usage: python antmaze_data.py [--dataset antmaze-large-navigate-v0] [--task large] [--out antmaze_large]
                              [--episodes 0 (all)] [--workers 8] [--preview frames.png]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import urllib.request

import numpy as np

URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench/{}.npz'


def hms(s):
    s = int(max(s, 0)); return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'


def download(name, ddir):
    """the .npz, resumed from a partial file; a progress line every 5 s"""
    os.makedirs(ddir, exist_ok=True)
    path, part, url = os.path.join(ddir, f'{name}.npz'), os.path.join(ddir, f'{name}.npz.part'), URL.format(name)
    if os.path.exists(path):
        print(f'[download] {name}.npz already here ({os.path.getsize(path) >> 20} MB)', flush=True); return path
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib', **({'Range': f'bytes={have}-'} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:   # the server ignored the range: start again
            have = 0
        total = have + int(r.headers.get('Content-Length', 0)); t0 = last = time.time(); got = 0
        print(f'[download] {url} -> {path} ({total >> 20} MB{", resuming at " + str(have >> 20) + " MB" if have else ""})', flush=True)
        with open(part, 'ab' if have else 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); got += len(b)
                if time.time() - last > 5:
                    last = time.time(); sp = got / (last - t0)
                    print(f'[download] {100 * (have + got) / max(total, 1):.0f}% | {(have + got) >> 20}/{total >> 20} MB | '
                          f'{sp / 2 ** 20:.1f} MB/s | left {hms((total - have - got) / max(sp, 1))}', flush=True)
    if total and os.path.getsize(part) != total:
        sys.exit(f'[download] incomplete ({os.path.getsize(part)} of {total} bytes): run the script again to resume')
    os.replace(part, path)
    print(f'[download] done in {hms(time.time() - t0)}', flush=True)
    return path


_env = None


def _init(task, res):
    global _env
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from stable_worldmodel.envs.ogbench.antmaze_env import AntMazeEnv
    _env = AntMazeEnv(task=task, resolution=res); _env.reset(seed=0)


def _render(states):
    out = np.empty((len(states), _env.res, _env.res, 3), np.uint8)
    for i, s in enumerate(states):
        _env.set_state(s); out[i] = _env.render()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='antmaze-large-navigate-v0')
    ap.add_argument('--task', default='large')
    ap.add_argument('--out', default='antmaze_large')
    ap.add_argument('--episodes', type=int, default=0, help='only the first N episodes (0 = all); for a quick test')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--res', type=int, default=224)
    ap.add_argument('--preview', default=None, help='also save a strip of 8 frames of the first episode to this PNG')
    ap.add_argument('--download_only', action='store_true')
    a = ap.parse_args()
    home = os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel'))
    npz = download(a.dataset, os.path.join(home, 'ogbench'))
    if a.download_only:
        return
    import h5py
    import hdf5plugin

    d = np.load(npz)
    print('[data]', {k: d[k].shape for k in d.files}, flush=True)
    term = np.asarray(d['terminals']).astype(bool); ends = np.nonzero(term)[0]
    if len(ends) == 0 or ends[-1] != len(term) - 1:
        ends = np.r_[ends, len(term) - 1]
    starts = np.r_[0, ends[:-1] + 1]; E = len(ends) if a.episodes <= 0 else min(a.episodes, len(ends))
    state = np.concatenate([d['qpos'], d['qvel']], 1).astype(np.float32); act = np.asarray(d['actions'], np.float32)
    lens = (ends - starts + 1)[:E]; N = int(lens.sum())
    print(f'[data] {len(ends)} episodes in the file, using {E} ({N} frames, length {lens.min()}-{lens.max()}); state {state.shape[1]}-D, '
          f'action {act.shape[1]}-D', flush=True)

    dd = os.path.join(home, 'datasets'); os.makedirs(dd, exist_ok=True)
    final, part = os.path.join(dd, f'{a.out}.h5'), os.path.join(dd, f'{a.out}.h5.part')
    if os.path.exists(final):
        with h5py.File(final, 'r') as f:
            print(f'[render] {final} is already complete: {f["pixels"].shape[0]} frames, {len(f["ep_len"])} episodes. '
                  f'Delete it to render again.', flush=True)
        print('DATA_DONE', final, flush=True); return
    f = h5py.File(part, 'a')
    if 'pixels' not in f:
        f.create_dataset('pixels', shape=(N, a.res, a.res, 3), dtype=np.uint8, chunks=(1, a.res, a.res, 3), **hdf5plugin.Zstd(clevel=3))
        f.create_dataset('action', shape=(N, act.shape[1]), dtype=np.float32)
        for k in ('proprio', 'state'):
            f.create_dataset(k, shape=(N, state.shape[1]), dtype=np.float32)
        for k, t in (('ep_idx', np.int32), ('step_idx', np.int32), ('id', np.int64)):
            f.create_dataset(k, shape=(N,), dtype=t)
        f.create_dataset('ep_len', data=lens.astype(np.int32)); f.create_dataset('ep_offset', data=np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64))
        f.attrs['episodes_done'] = 0; f.attrs['source'] = a.dataset
    elif f['pixels'].shape[0] != N:
        sys.exit(f'[render] {part} was started with another --episodes value ({f["pixels"].shape[0]} frames): delete it or use the same value')
    done = int(f.attrs['episodes_done']); off = np.r_[0, np.cumsum(lens)]
    print(f'[render] {E} episodes at {a.res} px with {a.workers} workers -> {final}' + (f' (continuing after episode {done})' if done else ''), flush=True)

    jobs = [state[starts[e]: ends[e] + 1] for e in range(done, E)]
    t0 = last = time.time(); fr = 0
    with mp.get_context('spawn').Pool(a.workers, initializer=_init, initargs=(a.task, a.res)) as pool:
        for k, px in enumerate(pool.imap(_render, jobs)):
            e = done + k; s, t = int(off[e]), int(off[e + 1]); sl = slice(starts[e], ends[e] + 1)
            f['pixels'][s:t] = px; f['action'][s:t] = act[sl]; f['proprio'][s:t] = state[sl]; f['state'][s:t] = state[sl]
            f['ep_idx'][s:t] = e; f['step_idx'][s:t] = np.arange(t - s); f['id'][s:t] = np.arange(s, t)
            f.attrs['episodes_done'] = e + 1; f.flush(); fr += t - s
            if a.preview and e == 0:
                try:
                    from PIL import Image
                    Image.fromarray(np.concatenate(list(px[np.linspace(0, len(px) - 1, 8).astype(int)]), 1)).save(a.preview)
                    print(f'[render] preview of episode 0 saved to {a.preview}', flush=True)
                except Exception as x:  # noqa: BLE001
                    print('[render] no preview:', x, flush=True)
            if time.time() - last > 10 or e + 1 == E:
                last = time.time(); sp = fr / (last - t0); gb = os.path.getsize(part) / 2 ** 30
                print(f'[render] episode {e + 1}/{E} ({100 * (e + 1) / E:.0f}%) | {sp:.0f} frames/s | elapsed {hms(last - t0)} | '
                      f'left {hms((N - off[e + 1]) / max(sp, 1e-9))} | file {gb:.1f} GB (about {gb * N / off[e + 1]:.0f} GB when done)', flush=True)
    # check: stored frames are what the environment renders from the stored state
    _init(a.task, a.res); idx = np.sort(np.random.default_rng(0).choice(N, size=min(16, N), replace=False))
    mad = [float(np.abs(_render(f['state'][int(i)][None])[0].astype(np.float32) - f['pixels'][int(i)].astype(np.float32)).mean()) for i in idx]
    print(f'[check] re-rendered {len(idx)} stored frames: mean abs pixel difference max {max(mad):.3f} (0 = identical)', flush=True)
    f.close()
    if max(mad) > 1.0:
        sys.exit('[check] FAILED: stored frames differ from a fresh render; the file is kept as .part')
    os.replace(part, final)
    print(f'[render] done in {hms(time.time() - t0)}: {final} ({os.path.getsize(final) / 2 ** 30:.1f} GB)', flush=True)
    print('DATA_DONE', final, flush=True)


if __name__ == '__main__':
    main()
