"""Graph nodes from an OGBench *stitch* dataset (e.g. pointmaze-large-stitch-v0) for a LeWM trained on the pixel version of
the OGBench navigate data (LeWAM/lewam-pointmaze-large).

OGBench stitch datasets hold short trajectories (a few maze cells each) and no pixels.  LeWAM rendered its pixel dataset by
replaying the simulator state through swm/PointMaze-v0 (stable_worldmodel/envs/ogbench/pointmaze_env.py), so we render the
stitch states the same way, encode every fifth frame with the frozen LeWM encoder and write a nodes file in the format of
paa_nodes.py (emb fp16, act, state, ep, step).

Two checks run first and must pass, otherwise nothing is written and the exit code is 2:
  1. render check : states from the pixel dataset (--check_h5), re-rendered here, must reproduce its stored pixels
                    (median mean-absolute pixel difference < --max_px)
  2. encoder check: our encoding of stored pixels must match the nodes file paa_nodes.py wrote for the same frames
                    (--check_nodes; median squared difference < --max_rel x the median one-block latent step)
usage: python stitch_nodes.py --model lewm_pml --out stitch_nodes.npz --check_h5 $STABLEWM_HOME/datasets/pointmaze_large.h5 \
           --check_nodes <expert nodes.npz> [--dataset pointmaze-large-stitch-v0] [--task large] [--max_eps N]
"""
import argparse
import json
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
import stable_pretraining as spt
from stable_worldmodel.envs.ogbench.pointmaze_env import PointMazeEnv


def tf():
    return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                               transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224)])


@torch.no_grad()
def encode(model, frames, T, batch=256):
    out = []
    for s in range(0, len(frames), batch):
        x = torch.stack([T(np.ascontiguousarray(f)) for f in frames[s:s + batch]]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model.encode({'pixels': x[:, None]})['emb'][:, 0].float().cpu())
    return torch.cat(out)


def load_ogbench(name, ddir):
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, f'{name}.npz')
    if not os.path.exists(path):
        import ogbench
        try:
            ogbench.download_datasets([name], dataset_dir=ddir)
        except AttributeError:  # older ogbench: download through make_env_and_datasets
            from ogbench.utils import make_env_and_datasets
            make_env_and_datasets(name, dataset_dir=ddir)
    d = np.load(path)
    print('ogbench', path, {k: d[k].shape for k in d.files}, flush=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default='pointmaze-large-stitch-v0')
    ap.add_argument('--task', default='large')
    ap.add_argument('--check_h5', required=True)
    ap.add_argument('--check_nodes', default=None)
    ap.add_argument('--ogbench_dir', default=os.path.join(os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel')), 'ogbench'))
    ap.add_argument('--max_eps', type=int, default=0, help='only the first N episodes (plumbing test)')
    ap.add_argument('--n_check', type=int, default=64)
    ap.add_argument('--max_px', type=float, default=3.0)
    ap.add_argument('--max_rel', type=float, default=0.1)
    args = ap.parse_args()
    report = {}

    env = PointMazeEnv(task=args.task, resolution=224)
    env.reset(seed=0)

    def render(state):
        env.set_state(state)
        return env.render()

    model = swm.wm.utils.load_pretrained(args.model).cuda().eval().requires_grad_(False)
    T = tf()

    # ------------------------------------------------------------------ check 1: re-rendering reproduces stored pixels
    import h5py
    import hdf5plugin  # noqa: F401  (compressed pixels)
    with h5py.File(args.check_h5, 'r') as f:
        n = f['pixels'].shape[0]
        ep_h = np.asarray(f['ep_idx'][:]) if 'ep_idx' in f else None
        step_h = np.asarray(f['step_idx'][:]) if 'step_idx' in f else None
        pool = np.nonzero(step_h % 5 == 0)[0] if step_h is not None else np.arange(n)  # frames that are graph nodes
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(pool, size=min(args.n_check, len(pool)), replace=False))
        px = np.stack([f['pixels'][int(i)] for i in idx])
        st = np.stack([f['state'][int(i)] for i in idx])
    ren = np.stack([render(s) for s in st])
    mad = np.abs(ren.astype(np.float32) - px.astype(np.float32)).mean(axis=(1, 2, 3))
    report['render_mad_median'], report['render_mad_max'] = float(np.median(mad)), float(mad.max())
    ok1 = report['render_mad_median'] < args.max_px
    zs, zr = encode(model, px, T), encode(model, ren, T)
    report['render_latent_sq'] = float(np.median(((zs - zr) ** 2).sum(-1).numpy()))

    # ------------------------------------------------------------------ check 2: our encoding == paa_nodes.py encoding
    ok2 = True
    if args.check_nodes:
        nd = np.load(args.check_nodes)
        emb, ep_n, st_n = nd['emb'].astype(np.float32), nd['ep'], nd['step']
        key = {(int(e), int(s)): k for k, (e, s) in enumerate(zip(ep_n, st_n))}
        pairs = [(j, key[(int(ep_h[i]), int(step_h[i]))]) for j, i in enumerate(idx)
                 if ep_h is not None and (int(ep_h[i]), int(step_h[i])) in key]
        o = np.lexsort((st_n, ep_n))
        nxt = (ep_n[o][1:] == ep_n[o][:-1]) & (st_n[o][1:] - st_n[o][:-1] == 5)
        one = float(np.median(((emb[o][1:][nxt] - emb[o][:-1][nxt]) ** 2).sum(-1)))
        report['one_block_sq'] = one
        if pairs:
            d = np.array([((zs[j].numpy() - emb[k]) ** 2).sum() for j, k in pairs])
            report['encoder_rel'] = float(np.median(d) / one)
            report['encoder_pairs'] = len(pairs)
            ok2 = report['encoder_rel'] < args.max_rel
        else:  # the random check frames missed the 5-step grid: compare rendered vs stored in units of the block step
            report['encoder_pairs'] = 0
        report['render_rel'] = report['render_latent_sq'] / one
        ok1 = ok1 and report['render_rel'] < args.max_rel
    report['checks_pass'] = bool(ok1 and ok2)
    print('[stitch_check] ' + json.dumps(report), flush=True)
    json.dump(report, open(args.out + '.check.json', 'w'), indent=1)
    if not report['checks_pass']:
        print('STITCH_CHECK_FAIL: rendering or encoding does not match the pixel dataset; no nodes written', flush=True)
        raise SystemExit(2)

    # ------------------------------------------------------------------ render + encode the stitch dataset's 5-step grid
    d = load_ogbench(args.dataset, args.ogbench_dir)
    qpos = d['qpos'] if 'qpos' in d else d['observations'][:, :2]
    qvel = d['qvel'] if 'qvel' in d else np.zeros_like(qpos)
    term = d['terminals'].ravel().astype(bool)
    ends = np.nonzero(term)[0]
    starts = np.r_[0, ends[:-1] + 1]
    if args.max_eps:
        starts, ends = starts[:args.max_eps], ends[:args.max_eps]
    ep, step, rows = [], [], []
    for e, (a, b) in enumerate(zip(starts, ends)):
        for t in range(0, b - a + 1, 5):
            ep.append(e); step.append(t); rows.append(a + t)
    rows = np.asarray(rows)
    state = np.concatenate([qpos[rows], qvel[rows]], 1).astype(np.float32)
    print(f'{len(starts)} episodes (median length {int(np.median(ends - starts + 1))}), {len(rows)} grid frames', flush=True)
    embs = []
    B = 1024
    for s in range(0, len(rows), B):
        frames = [render(x) for x in state[s:s + B]]
        embs.append(encode(model, frames, T).half())
        if (s // B) % 20 == 0:
            print(f'{s + len(frames)} / {len(rows)}', flush=True)
    np.savez(args.out, emb=torch.cat(embs).numpy(), act=np.zeros((len(rows), 2), np.float32), state=state,
             ep=np.asarray(ep), step=np.asarray(step))
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
