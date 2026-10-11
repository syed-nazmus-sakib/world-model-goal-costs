"""GoalShift G2: copy the PushT expert dataset with randomized colors (CPU only).

Every frame is re-drawn from its stored state with the PushT renderer, the same
path G1 uses for goal images (GoalRenderer in scripts/plan/eval_goalshift.py).
In each frame, the background, the agent and the block are each recolored with
probability --prob (default 0.5). Random colors are uniform RGB, redrawn until
every drawn shade of every object is at least --min_dist (default 60, Euclidean
RGB) from every shade of every other object. Agent and block are drawn in two
shades: the color c (outline) and the fill min(1.2 c, 255) (light_color in
envs/utils.py). The green goal-T keeps its default color. Colors are seeded by
the frame index, so the output doesn't depend on how the work is split.

Every other column is copied unchanged. Three uint8 columns record the colors
used: color_background, color_agent, color_block.

Before converting, 1,000 random frames are re-drawn with the default colors and
compared with the recorded pixels (G1 measured a mean absolute error of 0.42).
The script stops if the error is above 1.0. The output is written to
<dst>.partial and renamed only when complete.

Usage (STABLEWM_HOME/datasets/pusht_expert_train.h5 -> pusht_colorrand.h5):
    python scripts/data/rerender_pusht_colors.py --workers 12
Quick test on the first 40 episodes:
    python scripts/data/rerender_pusht_colors.py --episodes 40 --dst /tmp/t.h5
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402
import numpy as np  # noqa: E402

from stable_worldmodel.data.utils import get_cache_dir  # noqa: E402
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

FACTORS = ('background', 'agent', 'block')
# PushT defaults (checked against the env in main()).
DEFAULTS = {
    'background': np.array([255, 255, 255], dtype=np.uint8),
    'agent': np.array([65, 105, 225], dtype=np.uint8),  # RoyalBlue
    'block': np.array([119, 136, 153], dtype=np.uint8),  # LightSlateGray
}
GOAL_COLOR = np.array([144, 238, 144], dtype=np.uint8)  # LightGreen, unchanged
BLOSC_CODECS = {0: 'blosclz', 1: 'lz4', 2: 'lz4hc', 3: 'snappy', 4: 'zlib',
                5: 'zstd'}


def shades(color):
    """Colors the renderer actually draws for an agent/block of this color."""
    c = np.asarray(color, dtype=np.float32)
    light = np.minimum(np.float32(1.2) * c, np.float32(255))
    return [c.astype(np.int64), light.astype(np.int64)]


def valid(colors, min_dist):
    """True if every drawn shade is >= min_dist from every other object's."""
    groups = [
        [np.asarray(colors['background'], dtype=np.int64)],
        [GOAL_COLOR.astype(np.int64)],
        shades(colors['agent']),
        shades(colors['block']),
    ]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for a in groups[i]:
                for b in groups[j]:
                    if np.linalg.norm(a - b) < min_dist:
                        return False
    return True


def sample_colors(index, seed, prob, min_dist):
    """Colors for frame `index` (deterministic in seed and index)."""
    rng = np.random.default_rng([seed, index])
    recolor = rng.random(len(FACTORS)) < prob
    for _ in range(1000):
        colors = {
            f: rng.integers(0, 256, size=3).astype(np.uint8) if r
            else DEFAULTS[f]
            for f, r in zip(FACTORS, recolor)
        }
        if valid(colors, min_dist):
            return colors
    raise RuntimeError(f'no valid colors for frame {index} after 1000 draws')


def render(env, state, colors=None):
    """Re-draw one frame from its state (copied from G1's GoalRenderer)."""
    s = np.array(state, dtype=np.float64)  # copy
    # PushT._set_state() runs one physics step: zero the agent velocity so
    # the agent is drawn at its recorded position.
    s[5:7] = 0.0
    options = {'state': s, 'goal_state': s}
    if colors is not None:
        options['variation_values'] = {
            f'{f}.color': np.array(colors[f], dtype=np.uint8) for f in FACTORS
        }
    _, info = env.reset(seed=0, options=options)
    return info['goal']


def filters(ds):
    """[(filter id, cd_values)] of an h5py dataset."""
    plist = ds.id.get_create_plist()
    return [plist.get_filter(i)[:3:2] for i in range(plist.get_nfilters())]


_ENV = None


def _init_worker(resolution):
    global _ENV
    import cv2

    cv2.setNumThreads(1)
    _ENV = PushT(resolution=resolution)


def _render_chunk(task):
    start, states, seed, prob, min_dist = task
    res = _ENV.render_size
    pixels = np.empty((len(states), res, res, 3), dtype=np.uint8)
    colors = np.empty((len(states), len(FACTORS), 3), dtype=np.uint8)
    for k, state in enumerate(states):
        c = sample_colors(start + k, seed, prob, min_dist)
        pixels[k] = render(_ENV, state, c)
        colors[k] = [c[f] for f in FACTORS]
    return start, pixels, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='pusht_expert_train.h5')
    parser.add_argument('--dst', default='pusht_colorrand.h5')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--episodes', type=int, default=None,
                        help='convert only the first N episodes (tests)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prob', type=float, default=0.5)
    parser.add_argument('--min_dist', type=float, default=60.0)
    parser.add_argument('--identity_frames', type=int, default=1000)
    args = parser.parse_args()

    datasets = get_cache_dir(sub_folder='datasets')
    src_path = Path(datasets, args.src)
    dst_path = Path(datasets, args.dst)  # an absolute --dst is kept as is
    tmp_path = dst_path.with_name(dst_path.name + '.partial')
    if dst_path.exists():
        raise SystemExit(f'{dst_path} already exists; delete it first.')
    if tmp_path.exists():
        print(f'[colorrand] removing leftover {tmp_path}')
        tmp_path.unlink()

    with h5py.File(src_path, 'r') as src:
        lengths, offsets = src['ep_len'][:], src['ep_offset'][:]
        total = int(lengths.sum())
        if offsets[0] != 0 or np.any(offsets[1:] != offsets[:-1] + lengths[:-1]):
            raise SystemExit('episodes are not stored contiguously')
        n_eps = len(lengths) if args.episodes is None else min(
            args.episodes, len(lengths))
        n = int(offsets[n_eps - 1] + lengths[n_eps - 1])
        pix = src['pixels']
        res = pix.shape[1]
        assert pix.shape == (total, res, res, 3) and pix.dtype == np.uint8
        (fid, cd), = filters(pix)  # exactly one filter: Blosc
        assert fid == 32001, f'pixels filter {fid} is not Blosc'
        cd = [int(v) for v in cd]
        blosc = hdf5plugin.Blosc(cname=BLOSC_CODECS[cd[6]], clevel=cd[4],
                                 shuffle=cd[5])
        states = src['state'][:n]
        print(f'[colorrand] {src_path} -> {dst_path}: {n_eps} episodes, '
              f'{n} frames, pixels Blosc {BLOSC_CODECS[cd[6]]} level {cd[4]} '
              f'shuffle {cd[5]}, chunks {pix.chunks}')

        env = PushT(resolution=res)
        env.reset(seed=0)
        for f in FACTORS:
            assert np.array_equal(env.variation_space[f]['color'].value,
                                  DEFAULTS[f]), f
        assert np.array_equal(env.variation_space['goal']['color'].value,
                              GOAL_COLOR)
        assert valid(DEFAULTS, args.min_dist), 'defaults violate min_dist'

        idx = np.sort(np.random.default_rng(12345).choice(
            n, size=min(args.identity_frames, n), replace=False))
        recorded = pix[idx]
        errors = np.array([
            np.abs(render(env, states[i]).astype(np.int16)
                   - recorded[k].astype(np.int16)).mean()
            for k, i in enumerate(idx)
        ])
        mae = errors.mean()
        print(f'[colorrand] identity check on {len(idx)} random frames: '
              f'MAE {mae:.3f} (worst frame {errors.max():.3f}; G1 measured '
              f'0.42)')
        if mae > 1.0:
            raise SystemExit('re-drawn frames do not match the recorded ones')

        step = pix.chunks[0]
        tasks = ((s, states[s:s + step], args.seed, args.prob, args.min_dist)
                 for s in range(0, n, step))
        t0 = time.time()
        done, next_report = 0, 0.0
        with h5py.File(tmp_path, 'w', libver='latest') as dst:
            for key in src:
                if key == 'pixels':
                    continue
                d = src[key]
                if filters(d):
                    raise SystemExit(f'{key}: unexpected filters {filters(d)}')
                rows = {total: n, len(lengths): n_eps}[d.shape[0]]
                dst.create_dataset(key, data=d[:rows], chunks=d.chunks,
                                   maxshape=d.maxshape)
            out_pix = dst.create_dataset(
                'pixels', shape=(n, res, res, 3), dtype=np.uint8,
                chunks=pix.chunks, maxshape=pix.maxshape, **blosc)
            out_col = {
                f: dst.create_dataset(f'color_{f}', shape=(n, 3),
                                      dtype=np.uint8, chunks=(1000, 3),
                                      maxshape=(None, 3))
                for f in FACTORS
            }
            dst.attrs.update(source=args.src, seed=args.seed, prob=args.prob,
                             min_dist=args.min_dist, episodes=n_eps)
            ctx = mp.get_context('spawn')
            with ctx.Pool(args.workers, initializer=_init_worker,
                          initargs=(res,)) as pool:
                for start, pixels, colors in pool.imap(_render_chunk, tasks):
                    end = start + len(pixels)
                    out_pix[start:end] = pixels
                    for j, f in enumerate(FACTORS):
                        out_col[f][start:end] = colors[:, j]
                    done += len(pixels)
                    if done / n >= next_report or done == n:
                        rate = done / (time.time() - t0)
                        print(f'[colorrand] {done}/{n} frames ({done / n:.0%}),'
                              f' {rate:.0f} frames/s, ETA '
                              f'{(n - done) / rate / 60:.1f} min', flush=True)
                        next_report = done / n + 0.02
    os.replace(tmp_path, dst_path)
    print(f'[colorrand] done in {(time.time() - t0) / 60:.1f} min: {dst_path} '
          f'({dst_path.stat().st_size / 1e9:.1f} GB)')


if __name__ == '__main__':
    main()
