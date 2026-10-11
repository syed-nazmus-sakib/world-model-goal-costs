"""Idea 3, step 2: fixed-budget datasets for retraining LeWM.

A training window = 4 frames at 5-env-step spacing + 4 action blocks (10-d, the
4th unused by the loss), i.e. exactly one LeWM training sample (history 3, 1 pred).
Every dataset has the same number of windows (= env interaction budget).

  EXP   B=1  anchors = random expert-dataset states, actions = the expert's own
  RND1  B=1  anchors as EXP, action sequence copied from a random OTHER expert window
  RND4  B=4  N/4 anchors, 4 independent marginal action sequences each
  RND8  B=8  N/8 anchors, 8 ...
Marginal action sequences come from the same pool for all RND*, so p(a) matches
and only p(a | s) (conditional coverage) differs.

Output: results/exp5/data/<name>.npz with frames uint8 (N,4,R,R,3), actions
float32 normalised (N,4,10), anchor int (N,), state0 (N,7).
"""
import argparse, os
from multiprocessing import Pool

import h5py
import numpy as np

from ctrlgeo import sim
from ctrlgeo.wm import H5

A_MEAN = np.array([-0.00752025, 0.00777495]); A_STD = np.array([0.2097516, 0.20697816])


def rollout(args):
    s7, raw, res = args  # raw: (W,20,2) action sequences from the same anchor
    e = sim.make_env(resolution=res)
    e._set_state(s7)
    snap = sim.snapshot(e)
    out = []
    for seq in raw:
        sim.restore(e, snap)
        fr = [e.render()]
        for blk in range(3):
            for a in seq[5 * blk:5 * blk + 5]:
                sim.step_raw(e, a)
            fr.append(e.render())
        out.append(np.stack(fr))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--res', type=int, default=112)
    ap.add_argument('--names', nargs='+', default=['EXP', 'RND1', 'RND4', 'RND8'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/exp5/data')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with h5py.File(H5, 'r') as f:
        state = f['state'][:]; act = np.nan_to_num(f['action'][:])
        ep = f['episode_idx'][:]; step = f['step_idx'][:]; L = f['ep_len'][:]
    valid = np.where(step <= L[ep] - 21)[0]          # 20 env steps of actions available
    # hold out episodes used for evaluation (anchors never come from them)
    test_eps = set(rng.choice(np.unique(ep), 1000, replace=False).tolist())
    np.save(os.path.join(args.out, 'test_episodes.npy'), np.array(sorted(test_eps)))
    valid = valid[~np.isin(ep[valid], list(test_eps))]
    seq_of = lambda i: act[i:i + 20]                 # (20,2) raw
    pool_marg = rng.choice(valid, 4 * args.n, replace=True)  # shared marginal pool

    for name in args.names:
        B = 1 if name in ('EXP', 'RND1') else int(name[3:])
        n_anchor = args.n // B
        anchors = rng.choice(valid, n_anchor, replace=False)
        jobs, anch_ids, acts = [], [], []
        mp = 0
        for k, a in enumerate(anchors):
            if name == 'EXP':
                seqs = seq_of(a)[None]
            else:
                idx = pool_marg[mp:mp + B]; mp += B
                seqs = np.stack([seq_of(i) for i in idx])
            jobs.append((state[a], seqs, args.res))
            anch_ids += [k] * B
            acts.append(seqs)
        with Pool(30) as p:
            frames = np.concatenate(p.map(rollout, jobs, chunksize=64))
        acts = np.concatenate(acts)                   # (N,20,2) raw
        an = ((acts - A_MEAN) / A_STD).reshape(len(acts), 4, 10).astype(np.float32)
        np.savez(os.path.join(args.out, f'{name}.npz'), frames=frames, actions=an,
                 anchor=np.array(anch_ids), state0=state[np.repeat(anchors, B)])
        print(name, frames.shape, f'marginal |a| mean={np.abs(an).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
