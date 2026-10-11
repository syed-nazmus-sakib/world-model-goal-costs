"""GoalShift G2: checks for the random-color PushT dataset (CPU only).

1. Every column except the pixels and the colors equals the source.
2. Each part (background, agent, block) is recolored in about --prob of the
   frames, and every frame meets the contrast rule.
3. Random frames re-drawn with their stored colors equal the stored pixels
   exactly, and the stored colors equal a fresh draw from the seed.
4. The library loads the file the way training does, and its samples match
   the source dataset in every column except pixels.
5. Saves a grid of 32 random frames to look at.

Usage:
    python scripts/data/check_pusht_colors.py
    python scripts/data/check_pusht_colors.py --dst /tmp/t.h5 --out /tmp/g.png
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.utils import get_cache_dir  # noqa: E402
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerender_pusht_colors import (  # noqa: E402
    DEFAULTS,
    FACTORS,
    GOAL_COLOR,
    render,
    sample_colors,
    shades,
)

# As training loads PushT: scripts/train/config/data/pusht.yaml with lewm.yaml
# (history 3 + 1 prediction = 4 steps, frameskip 5).
LOAD_KWARGS = dict(num_steps=4, frameskip=5,
                   keys_to_load=['pixels', 'action', 'proprio', 'state'],
                   keys_to_cache=['action', 'proprio', 'state'])


def fail(msg):
    raise SystemExit(f'[colorrand-check] FAILED: {msg}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='pusht_expert_train.h5')
    parser.add_argument('--dst', default='pusht_colorrand.h5')
    parser.add_argument('--frames', type=int, default=200)
    parser.add_argument('--out', default='pusht_colorrand_grid.png')
    args = parser.parse_args()

    datasets = get_cache_dir(sub_folder='datasets')
    src_path, dst_path = Path(datasets, args.src), Path(datasets, args.dst)
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'r') as dst:
        seed, prob = int(dst.attrs['seed']), float(dst.attrs['prob'])
        min_dist = float(dst.attrs['min_dist'])
        n_eps, n = int(dst.attrs['episodes']), dst['pixels'].shape[0]

        # 1. Unchanged columns.
        expected = set(src) | {f'color_{f}' for f in FACTORS}
        if set(dst) != expected:
            fail(f'columns {sorted(dst)} != {sorted(expected)}')
        for key in src:
            if key == 'pixels':
                continue
            rows = n_eps if key in ('ep_len', 'ep_offset') else n
            a, b = src[key][:rows], dst[key][:]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(
                    a, b):
                fail(f'column {key} differs from the source')
        if int(dst['ep_len'][:].sum()) != n:
            fail('ep_len does not add up to the number of frames')
        print(f'[colorrand-check] 1. {n_eps} episodes, {n} frames; columns '
              f'equal to the source: {sorted(k for k in src if k != "pixels")}')

        # 2. Recolor rates and contrast.
        cols = {f: dst[f'color_{f}'][:] for f in FACTORS}
        rates = {f: float(np.any(cols[f] != DEFAULTS[f], axis=1).mean())
                 for f in FACTORS}
        groups = [[cols['background'].astype(np.int64)],
                  [np.broadcast_to(GOAL_COLOR.astype(np.int64), (n, 3))],
                  shades(cols['agent']), shades(cols['block'])]
        closest = np.full(n, np.inf)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                for a in groups[i]:
                    for b in groups[j]:
                        closest = np.minimum(
                            closest, np.linalg.norm(a - b, axis=1))
        print(f'[colorrand-check] 2. recolored fraction {rates} (target '
              f'{prob}); closest shades in any frame: {closest.min():.1f} '
              f'(rule: >= {min_dist})')
        if closest.min() < min_dist:
            fail('a frame breaks the contrast rule')
        if any(abs(r - prob) > 0.05 for r in rates.values()):
            fail('recolor rates are far from the target')

        # 3. Exact re-draw from the stored colors and the seed.
        rng = np.random.default_rng(7)
        idx = np.sort(rng.choice(n, size=min(args.frames, n), replace=False))
        env = PushT(resolution=dst['pixels'].shape[1])
        stored = dst['pixels'][idx]
        for k, i in enumerate(idx):
            c = {f: cols[f][i] for f in FACTORS}
            fresh = sample_colors(int(i), seed, prob, min_dist)
            if not all(np.array_equal(fresh[f], c[f]) for f in FACTORS):
                fail(f'frame {i}: stored colors differ from the seeded draw')
            if not np.array_equal(render(env, dst['state'][i], c), stored[k]):
                fail(f'frame {i}: pixels differ from a re-draw')
        print(f'[colorrand-check] 3. {len(idx)} random frames: re-draw equals '
              f'the stored pixels exactly; colors match the seed')

        grid_idx = np.sort(rng.choice(n, size=min(32, n), replace=False))
        grid = dst['pixels'][grid_idx]

    # 4. Load like training.
    ds = swm.data.load_dataset(str(dst_path), **LOAD_KWARGS)
    ref = swm.data.load_dataset(str(src_path), **LOAD_KWARGS)
    if n_eps == len(ref.lengths) and len(ds) != len(ref):
        fail(f'{len(ds)} samples, source has {len(ref)}')
    for i in (0, len(ds) // 2, len(ds) - 1):
        s, r = ds[i], ref[i]
        for key in ('action', 'proprio', 'state'):
            if not torch.equal(s[key], r[key]):
                fail(f'sample {i}: {key} differs from the source')
        if s['pixels'].shape != r['pixels'].shape:
            fail(f'sample {i}: pixels shape {tuple(s["pixels"].shape)}')
    print(f'[colorrand-check] 4. library loads it: {len(ds)} samples, pixels '
          f'{tuple(s["pixels"].shape)} {s["pixels"].dtype}, action '
          f'{tuple(s["action"].shape)}; non-pixel columns match the source')

    # 5. Grid to look at.
    rows = [np.concatenate(list(grid[r:r + 8]), axis=1)
            for r in range(0, len(grid) - len(grid) % 8, 8)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f'[colorrand-check] 5. saved {args.out}')
    print('[colorrand-check] ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
