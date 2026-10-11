"""Cube (small object that is NOT the robot): does an area-invariant aggregation of frozen DINOv2 patch
mismatches plan better than the patch mean used by DINO-WM-style planners?

Simulator dynamics (exact, ctrlgeo/cube_sim.py), so only the goal cost changes. Tasks as exp6: held-out
expert start, goal = expert state 25 steps later with the cube moved >= 5 cm; success = cube within 4 cm of
its goal position at any step of a 50-step budget. CEM: horizon 5 blocks x 5 steps, 300 samples, 30 iters,
top-30 elites, replan every 25 steps.

Costs on DINOv2 ViT-S/14 patch tokens of the candidate's final frame vs the goal frame:
  mean        mean over patches of the per-patch squared feature difference (DINO-WM cost)
  mean+a*p    plus alpha * squared distance of the effector position (proprio), as DINO-WM / JEPA-WMs
  topk:K      mean of the K largest per-patch mismatches (ours)
  true        cube distance to its goal position (oracle reference)

  MUJOCO_GL=egl python scripts/exp7_cube_cost.py --costs mean topk:8 true --episodes 50
"""
import argparse, importlib.util, json, os, time
from multiprocessing import get_context

import numpy as np
import torch

OUT = 'results/exp7_cube_cost'
RES = 224
HUB = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')

_ENV = None


def _w_init():
    global _ENV
    from ctrlgeo import cube_sim as cs
    _ENV = cs.make_env(RES)


def _w_roll(args):
    """snap, actions (N,T,5), render -> final frames (N,H,W,3) | None, cube (N,3), effector (N,3)."""
    from ctrlgeo import cube_sim as cs
    snap, acts, render = args
    u = _ENV.unwrapped
    frames, cubes, effs = [], [], []
    for seq in acts:
        cs.restore(_ENV, snap)
        for a in seq:
            cs.step(_ENV, a)
        cubes.append(cs.cube_pos(_ENV))
        effs.append(u._data.site_xpos[u._pinch_site_id].copy())
        if render:
            frames.append(cs.render(_ENV))
    return (np.stack(frames) if render else None), np.stack(cubes), np.stack(effs)


class Pool:
    def __init__(self, n=30):
        self.n, self.p = n, get_context('spawn').Pool(n, initializer=_w_init)

    def rollout(self, snap, acts, render):
        chunks = np.array_split(np.arange(len(acts)), self.n)
        res = self.p.map(_w_roll, [(snap, acts[c], render) for c in chunks if len(c)])
        f = np.concatenate([r[0] for r in res]) if render else None
        return f, np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res])


class Encoder:
    def __init__(self):
        self.m = torch.hub.load(HUB, 'dinov2_vits14', source='local').cuda().eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs, bs=500):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.as_tensor(imgs[i:i + bs], device='cuda').permute(0, 3, 1, 2).float() / 255
            x = ((x - self.mean) / self.std).half()
            out.append(self.m.forward_features(x)['x_norm_patchtokens'].float())
        return torch.cat(out)  # (N, 256, 384)


def score(kind, feats, gfeat, eff, geff, cube, gcube):
    if kind == 'true':
        return torch.as_tensor(np.linalg.norm(cube - gcube, axis=1), device='cuda')
    d = (feats - gfeat[None]).pow(2).mean(-1)  # (N, 256)
    name, _, par = kind.partition(':')
    if name == 'mean':
        c = d.mean(-1)
    elif name == 'topk':
        c = d.topk(int(par), dim=-1).values.mean(-1)
    else:
        raise ValueError(kind)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--costs', nargs='+', default=['mean', 'topk:8', 'true'])
    ap.add_argument('--alpha', type=float, default=0.0, help='weight of the effector-position (proprio) term')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--iters', type=int, default=30)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    spec = importlib.util.spec_from_file_location('e6', 'scripts/exp6_cube.py')
    e6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e6)
    tasks, E = e6.test_tasks(a.episodes, a.seed)
    S, A, C = E['snaps'], E['actions'], E['cubes']
    st = json.load(open('results/exp6/action_stats.json')); mu_a, sd_a = np.array(st['a_mean']), np.array(st['a_std'])
    pool, enc = Pool(), Encoder()
    from ctrlgeo import cube_sim as cs
    env = cs.make_env(RES)
    u = env.unwrapped
    H, BLK, AD = 5, 5, 5
    for kind in a.costs:
        tag = f"{kind.replace(':', '')}_a{a.alpha:g}_s{a.seed}"
        path = f'{OUT}/{tag}.json'
        if os.path.exists(path):
            print('done already', tag); continue
        rng = torch.Generator(device='cuda').manual_seed(a.seed)
        succ, final_d, t0 = [], [], time.time()
        for ep, (i, t) in enumerate(tasks):
            cs.restore(env, S[i, t + 25]); gimg = cs.render(env); gcube = C[i, t + 25].copy()
            geff = u._data.site_xpos[u._pinch_site_id].copy()
            gfeat = enc(gimg[None])[0]
            cs.restore(env, S[i, t])
            ok, steps = False, 0
            while steps < 50:
                snap = cs.snapshot(env)
                m = torch.zeros(H, BLK * AD, device='cuda'); sd = torch.ones_like(m)
                for it in range(a.iters):
                    u_n = m + sd * torch.randn(a.samples, H, BLK * AD, device='cuda', generator=rng)
                    u_n[0] = m
                    raw = (u_n.view(a.samples, H * BLK, AD).cpu().numpy() * sd_a + mu_a).clip(-1, 1)
                    frames, cube, eff = pool.rollout(snap, raw, kind != 'true')
                    feats = enc(frames) if kind != 'true' else None
                    c = score(kind, feats, gfeat, eff, geff, cube, gcube)
                    if a.alpha > 0 and kind != 'true':
                        c = c + a.alpha * torch.as_tensor(((eff - geff) ** 2).sum(1), device='cuda', dtype=c.dtype)
                    el = u_n[c.topk(30, largest=False).indices]
                    m, sd = el.mean(0), el.std(0)
                plan = (m.view(H * BLK, AD).cpu().numpy() * sd_a + mu_a).clip(-1, 1)
                for k in range(H * BLK):
                    if steps >= 50:
                        break
                    cs.step(env, plan[k]); steps += 1
                    ok |= np.linalg.norm(cs.cube_pos(env) - gcube) <= 0.04
            succ.append(bool(ok)); final_d.append(float(np.linalg.norm(cs.cube_pos(env) - gcube)))
            print(f'{tag} ep {ep:02d}: {"S" if ok else "F"} final cube dist {final_d[-1]:.3f} | '
                  f'{sum(succ)}/{len(succ)} | {(time.time() - t0) / len(succ):.0f}s/ep', flush=True)
        json.dump(dict(cost=kind, alpha=a.alpha, seed=a.seed, success=succ, final_dist=final_d,
                       tasks=[list(map(int, x)) for x in tasks]), open(path, 'w'))
        print(f'== {tag}: success {100 * np.mean(succ):.1f}%', flush=True)


if __name__ == '__main__':
    main()
