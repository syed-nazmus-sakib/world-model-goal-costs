"""Exp 8 (2026-10-19): does the goal metric weight the scene by image area? PushT, exact renders, no dynamics.

Two measurements on the same rendered states, for three goal metrics:
  pixel   mean squared pixel difference (area-weighted by construction; the reference)
  dinov2  frozen DINOv2 ViT-S/14 patch tokens, mean over patches of ||z_p - z'_p||^2 (the DINO-WM / JEPA-WM metric)
  lewm    LeWM PushT encoder (trained jointly with its predictor), ||z - z'||^2 on the 192-d projected CLS embedding

Part 1, exchange rate. For N base states, perturb only the agent (translate d px), only the block (translate d px),
  or only the block angle (theta deg), and record the metric. rho(d) = cost(agent moved d) / cost(block moved d).
  Repeated for agent / block sizes (variation 'agent.scale', 'block.scale'); LeWM was trained only at the default
  sizes, so its non-default rows are out of distribution and reported separately.
Part 2, task-equivalent goals. Per group: goal state g, start = g with the block displaced; candidates have the block
  between start and goal (random fraction + noise) and the agent near the block. Task progress = block-only
  distance (pos/20 px + angle/(pi/9)). Goal variants with the block unchanged: the original goal, the agent moved by
  ~80 px (3 variants), the agent removed. Ordering accuracy on clear pairs (|dD| > 1) for each metric and variant.

    python scripts/exp8_area_weighting.py --out results/exp8 --n 300 --groups 200
"""
import argparse
import json
import os

import numpy as np
import torch

from stable_worldmodel.envs.pusht.env import PushT

LO, HI = 40.0, 472.0  # keep bodies inside the 512 px arena (walls at 5 / 506)


# ------------------------------------------------------------------ rendering
class Renderer:
    def __init__(self, agent_scale=40.0, block_scale=30.0, res=224):
        self.env = PushT(resolution=res)
        vv = {'agent.scale': np.float32(agent_scale), 'block.scale': np.float32(block_scale)}
        s0 = np.array([100, 100, 256, 256, 0, 0, 0], dtype=np.float64)
        self.env.reset(seed=0, options={'state': s0, 'goal_state': s0, 'variation_values': vv})
        e = self.env
        self.agent_r = max(s.radius for s in e.agent.shapes)
        self.agent_area = np.pi * self.agent_r ** 2
        self.block_area = sum(s.area for s in e.block.shapes)

    def place(self, state):
        """state: (agent x, agent y, block x, block y, angle). Sets bodies exactly (no physics step)."""
        e = self.env
        e.agent.position = tuple(state[0:2]); e.agent.velocity = (0.0, 0.0)
        e.block.position = tuple(state[2:4]); e.block.angle = float(state[4])
        e.block.velocity = (0.0, 0.0); e.block.angular_velocity = 0.0
        e.space.reindex_shapes_for_body(e.agent)
        e.space.reindex_shapes_for_body(e.block)

    def gap(self, state):
        """Agent-surface to block-surface distance (px); inf if the agent is off-canvas."""
        if state[0] < 0:
            return np.inf
        self.place(state)
        p = self.env.agent.position
        return min(s.point_query(p).distance for s in self.env.block.shapes) - self.agent_r

    def block_inside(self, state):
        self.place(state)
        bb = [s.cache_bb() for s in self.env.block.shapes]
        return min(b.left for b in bb) > 10 and min(b.bottom for b in bb) > 10 and \
            max(b.right for b in bb) < 501 and max(b.top for b in bb) < 501

    def render(self, state):
        self.place(state)
        return self.env.render().copy()


# ------------------------------------------------------------------ metrics
class Metrics:
    def __init__(self, device, lewm_name):
        import stable_worldmodel as swm
        from stable_pretraining.data import dataset_stats
        self.dev = device
        st = dataset_stats.ImageNet
        self.mean = torch.tensor(st['mean'], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(st['std'], device=device).view(1, 3, 1, 1)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        m = swm.wm.utils.load_pretrained(lewm_name)
        m.interpolate_pos_encoding = True
        self.lewm = m.to(device).eval().requires_grad_(False)

    def _x(self, imgs):
        x = torch.as_tensor(imgs, device=self.dev).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def embed(self, imgs, bs=128):
        """imgs uint8 (N,224,224,3) -> dict of features (cpu float32)."""
        out = {'pixel': [], 'dinov2': [], 'lewm': []}
        for i in range(0, len(imgs), bs):
            raw = imgs[i:i + bs]
            x = self._x(raw)
            out['pixel'].append(torch.as_tensor(raw).float().div(255.0).reshape(len(raw), -1))
            out['dinov2'].append(self.dino.forward_features(x)['x_norm_patchtokens'].float().cpu())
            o = self.lewm.encoder(x, interpolate_pos_encoding=True)
            out['lewm'].append(self.lewm.projector(o.last_hidden_state[:, 0]).float().cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    @staticmethod
    def cost(fa, fb):
        """Squared distance per metric, averaged over pixels / patches (matching the planning costs)."""
        d = (fa - fb).pow(2)
        if d.dim() == 3:  # (N, P, D) patch tokens
            return d.sum(-1).mean(-1)
        return d.mean(-1) if d.shape[-1] > 1000 else d.sum(-1)


# ------------------------------------------------------------------ sampling
def rand_dir(rng):
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a), np.sin(a)])


def sample_base(R, rng, min_gap=15.0):
    while True:
        s = np.array([*rng.uniform(LO, HI, 2), *rng.uniform(150, 362, 2), rng.uniform(0, 2 * np.pi)])
        if R.block_inside(s) and R.gap(s) > min_gap:
            return s


def move_agent(R, s, d, rng, min_gap=10.0, tries=200):
    for _ in range(tries):
        t = s.copy(); t[0:2] = s[0:2] + d * rand_dir(rng)
        if LO <= t[0] <= HI and LO <= t[1] <= HI and R.gap(t) > min_gap:
            return t
    return None


def move_block(R, s, d, rng, dtheta=0.0, min_gap=10.0, tries=200):
    for _ in range(tries):
        t = s.copy()
        if d > 0:
            t[2:4] = s[2:4] + d * rand_dir(rng)
        t[4] = s[4] + dtheta * rng.choice([-1, 1])
        if R.block_inside(t) and R.gap(t) > min_gap:
            return t
    return None


# ------------------------------------------------------------------ part 1
def part1(M, args, rng):
    configs = [(40, 30), (20, 30), (60, 30), (40, 20), (40, 45)]
    D = [5, 10, 20, 40, 80]
    TH = [5, 10, 20, 40]
    rows = []
    for (asc, bsc) in configs:
        R = Renderer(asc, bsc)
        recs = {f'agent_{d}': [] for d in D} | {f'block_{d}': [] for d in D} | {f'rot_{t}': [] for t in TH}
        imgs, keys, base_idx = [], [], []
        for i in range(args.n):
            s = sample_base(R, rng)
            imgs.append(R.render(s)); keys.append('base'); base_idx.append(i)
            for d in D:
                for name, t in ((f'agent_{d}', move_agent(R, s, d, rng)), (f'block_{d}', move_block(R, s, d, rng))):
                    if t is not None:
                        imgs.append(R.render(t)); keys.append(name); base_idx.append(i)
            for th in TH:
                t = move_block(R, s, 0.0, rng, dtheta=np.deg2rad(th))
                if t is not None:
                    imgs.append(R.render(t)); keys.append(f'rot_{th}'); base_idx.append(i)
        F = M.embed(np.stack(imgs))
        base_pos = {b: j for j, (k, b) in enumerate(zip(keys, base_idx)) if k == 'base'}
        for metric, f in F.items():
            per = {k: [] for k in recs}
            for j, (k, b) in enumerate(zip(keys, base_idx)):
                if k != 'base':
                    per[k].append(float(Metrics.cost(f[j:j + 1], f[base_pos[b]:base_pos[b] + 1])[0]))
            med = {k: float(np.median(v)) for k, v in per.items() if v}
            row = dict(agent_scale=asc, block_scale=bsc, metric=metric, area_ratio=R.agent_area / R.block_area,
                       in_distribution=(asc, bsc) == (40, 30), median_cost=med,
                       n={k: len(v) for k, v in per.items()},
                       rho={d: med[f'agent_{d}'] / med[f'block_{d}'] for d in D})
            rows.append(row)
            print(f"scale a{asc} b{bsc} {metric:7s} area a/b {row['area_ratio']:.3f}  rho(d) " +
                  ' '.join(f"{d}:{row['rho'][d]:.3f}" for d in D), flush=True)
    return rows


# ------------------------------------------------------------------ part 2
def block_dist(s, g):
    dp = np.linalg.norm(s[..., 2:4] - g[..., 2:4], axis=-1) / 20.0
    da = np.abs(s[..., 4] - g[..., 4]) % (2 * np.pi)
    da = np.minimum(da, 2 * np.pi - da) / (np.pi / 9)
    return dp + da


def part2(M, args, rng):
    R = Renderer()
    res = {}
    for gi in range(args.groups):
        g = sample_base(R, rng, min_gap=20.0)
        start = None
        while start is None:
            start = move_block(R, g, rng.uniform(50, 110), rng, dtheta=np.deg2rad(rng.uniform(0, 45)), min_gap=0.0)
        goals = [('original', g)]
        for k in range(3):
            v = move_agent(R, g, 80.0, rng, min_gap=15.0)
            if v is not None:
                goals.append((f'agent_moved_{k}', v))
        nag = g.copy(); nag[0:2] = -1000.0
        goals.append(('agent_removed', nag))
        cands = []
        while len(cands) < args.cands:
            f = rng.uniform(0, 1)
            c = start + f * (g - start)
            c[4] = start[4] + f * (((g[4] - start[4] + np.pi) % (2 * np.pi)) - np.pi)
            c[2:4] += rng.normal(0, 5, 2); c[4] += np.deg2rad(rng.normal(0, 5))
            if not R.block_inside(c):
                continue
            ok = False
            for _ in range(100):  # agent near the block, as in pushing
                c[0:2] = c[2:4] + rng.uniform(30, 110) * rand_dir(rng)
                if LO <= c[0] <= HI and LO <= c[1] <= HI and 3.0 < R.gap(c) < 40.0:
                    ok = True; break
            if ok:
                cands.append(c.copy())
        cands = np.stack(cands)
        imgs = [R.render(s) for s in cands] + [R.render(s) for _, s in goals]
        F = M.embed(np.stack(imgs))
        Dtrue = block_dist(cands, g)
        i, j = np.triu_indices(len(cands), 1)
        clear = np.abs(Dtrue[i] - Dtrue[j]) > 1.0
        for metric, f in F.items():
            fc, fg = f[:len(cands)], f[len(cands):]
            for gk, (name, _) in enumerate(goals):
                c = Metrics.cost(fc, fg[gk:gk + 1].expand_as(fc) if fc.dim() == 2 else fg[gk:gk + 1]).numpy()
                acc = float(np.mean(np.sign(c[i] - c[j])[clear] == np.sign(Dtrue[i] - Dtrue[j])[clear]))
                key = name if not name.startswith('agent_moved') else 'agent_moved'
                res.setdefault(metric, {}).setdefault(key, []).append(acc)
        if gi % 20 == 0:
            print(f'group {gi}: ' + ' | '.join(f"{m} " + ' '.join(f"{k} {np.mean(v):.3f}" for k, v in d.items())
                                                for m, d in res.items()), flush=True)
    return res


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x); r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/exp8')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--groups', type=int, default=200)
    p.add_argument('--cands', type=int, default=24)
    p.add_argument('--lewm', default='quentinll/lewm-pusht')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--parts', default='1,2')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = Metrics(dev, args.lewm)
    rng = np.random.default_rng(args.seed)
    if '1' in args.parts:
        rows = part1(M, args, rng)
        json.dump(rows, open(os.path.join(args.out, 'part1_exchange.json'), 'w'), indent=1)
    if '2' in args.parts:
        res = part2(M, args, rng)
        summ = {m: {k: boot_ci(v) for k, v in d.items()} for m, d in res.items()}
        json.dump({'per_group': res, 'summary': summ}, open(os.path.join(args.out, 'part2_ordering.json'), 'w'))
        print('PART2 ordering accuracy on clear block-progress pairs (mean, 95% CI over groups)')
        for m, d in summ.items():
            print(f'  {m:7s} ' + ' | '.join(f'{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]' for k, v in d.items()))
    print('EXP8_DONE')


if __name__ == '__main__':
    main()
