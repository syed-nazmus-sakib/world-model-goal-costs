"""Exp 9 (2026-10-19): arm-versus-object weighting in a robot-arm scene, frozen DINOv2 vs a trained LeWM encoder.

OGBench cube-single through stable-worldmodel (swm/OGBCube-v0, 224 px, the env the public quentinll/lewm-cube
checkpoint was trained on). Kinematic only: set qpos, mj_forward, render. Per reset placement:
  cube moved on the table by r and 2r (r = 4 cm, the OGBench success distance)
  arm moved so that the pinch site moves 2, 5, 10 cm in a random direction with the minimum joint change (damped
  pseudo-inverse of the effector Jacobian, as IK would), without touching the cube
Metrics: pixel MSE, DINOv2 ViT-S/14 patch-token cost, LeWM cube encoder (192-d projected CLS) squared distance.
Reports cost(arm d) / cost(cube r) and d* = the arm displacement whose cost equals that of the cube off by r.
Then a linear probe (ridge, 5-fold CV) of the cube position from each representation, arm in random poses.

    MUJOCO_GL=egl python scripts/exp9_cube_weighting.py --out results/exp9 --episodes 60 --probe_n 600
"""
import argparse
import json
import os

import gymnasium as gym
import mujoco
import numpy as np
import torch

import stable_worldmodel  # noqa: F401  (registers swm/ envs)

R = 0.04
ARM_D = [0.02, 0.05, 0.10]


class Scene:
    def __init__(self):
        self.env = gym.make('swm/OGBCube-v0', env_type='single', ob_type='pixels', width=224, height=224,
                            visualize_info=False, terminate_at_goal=False)
        self.u = self.env.unwrapped
        self.env.reset(seed=0)
        m = self.u._model
        self.cube_g = set(self.u._cube_geom_ids_list[0])
        self.cube_adr = m.joint('object_joint_0').qposadr[0]
        self.arm_adr = np.asarray(self.u._arm_joint_ids)

    def reset(self, seed):
        self.env.reset(seed=seed)
        self.q0 = self.u._data.qpos.copy()

    def set_q(self, q):
        d = self.u._data
        d.qpos[:] = q; d.qvel[:] = 0
        mujoco.mj_forward(self.u._model, d)

    def cube(self):
        return self.u._data.qpos[self.cube_adr:self.cube_adr + 3].copy()

    def eff(self):
        return self.u._data.site_xpos[self.u._pinch_site_id].copy()

    def cube_touched(self):
        d = self.u._data
        for i in range(d.ncon):
            a, b = d.contact[i].geom1, d.contact[i].geom2
            if (a in self.cube_g) != (b in self.cube_g) and d.contact[i].dist < 0:
                other = b if a in self.cube_g else a
                if self.u._model.geom_bodyid[other] != 0:
                    return True
        return False

    def render(self):
        return self.u.render().copy()

    def move_cube(self, dist, rng, tries=50):
        for _ in range(tries):
            a = rng.uniform(0, 2 * np.pi)
            q = self.q0.copy(); q[self.cube_adr:self.cube_adr + 2] += dist * np.array([np.cos(a), np.sin(a)])
            self.set_q(q)
            if not self.cube_touched():
                return q, dist
        return None

    def _jac(self, q, eps=1e-4):
        self.set_q(q); e = self.eff(); J = np.zeros((3, len(self.arm_adr)))
        for k, a in enumerate(self.arm_adr):
            qq = q.copy(); qq[a] += eps; self.set_q(qq); J[:, k] = (self.eff() - e) / eps
        return J, e

    def move_arm(self, dist, rng, q_base=None, tries=40):
        """Effector moved `dist` in a random direction with the minimum joint change (damped pseudo-inverse of the
        effector Jacobian, a few Newton steps), as the env's IK controller would."""
        q_base = self.q0 if q_base is None else q_base
        self.set_q(q_base); e0 = self.eff()
        for _ in range(tries):
            u = rng.standard_normal(3); u[2] = abs(u[2]) * 0.5; u /= np.linalg.norm(u)
            goal = e0 + dist * u
            q = q_base.copy()
            for _ in range(6):
                J, e = self._jac(q)
                q[self.arm_adr] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), goal - e)
            self.set_q(q)
            got = np.linalg.norm(self.eff() - e0)
            if np.linalg.norm(self.eff() - goal) < 0.1 * dist and self.eff()[2] > 0.02 and not self.cube_touched():
                return q, got
        return None


class Metrics:
    def __init__(self, device, lewm_name):
        import stable_worldmodel as swm
        self.dev = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
        m = swm.wm.utils.load_pretrained(lewm_name)
        m.interpolate_pos_encoding = True
        self.lewm = m.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def embed(self, imgs, bs=64):
        out = {'pixel': [], 'dinov2': [], 'lewm': []}
        for i in range(0, len(imgs), bs):
            raw = torch.as_tensor(np.stack(imgs[i:i + bs]))
            x = (raw.to(self.dev).permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            out['pixel'].append(raw.float().div(255.0).reshape(len(raw), -1))
            out['dinov2'].append(self.dino.forward_features(x)['x_norm_patchtokens'].float().cpu())
            o = self.lewm.encoder(x, interpolate_pos_encoding=True)
            out['lewm'].append(self.lewm.projector(o.last_hidden_state[:, 0]).float().cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    @staticmethod
    def cost(a, b):
        d = (a - b).pow(2)
        if d.dim() == 3:
            return d.sum(-1).mean(-1)
        return d.mean(-1) if d.shape[-1] > 1000 else d.sum(-1)


def ridge_cv(X, Y, lams=(1e-2, 1e-1, 1, 10, 100, 1e3), folds=5, seed=0):
    """Dual ridge, features standardised inside each fold, 5-fold CV over lambda. On the GPU: one Gram matrix and
    one eigendecomposition per fold serve every lambda. Returns (rmse of the 3-d position, lambda, R^2)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64, device=dev)
    N, P = X.shape
    parts = np.array_split(np.random.default_rng(seed).permutation(N), folds)
    errs = {lam: torch.zeros_like(Yt) for lam in lams}
    for k in range(folds):
        te = torch.as_tensor(parts[k], device=dev)
        tr = torch.as_tensor(np.concatenate([parts[i] for i in range(folds) if i != k]), device=dev)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        K, Kte = (Xtr @ Xtr.T).double() / P, (Xte @ Xtr.T).double() / P
        s, V = torch.linalg.eigh(K)
        ym = Yt[tr].mean(0)
        R = V.T @ (Yt[tr] - ym)
        for lam in lams:
            errs[lam][te] = Kte @ (V @ (R / (s + lam)[:, None])) + ym - Yt[te]
    best = None
    Yn = np.asarray(Y)
    for lam in lams:
        e = errs[lam].cpu().numpy()
        rmse = float(np.sqrt((e ** 2).sum(1).mean()))
        if best is None or rmse < best[0]:
            best = (rmse, lam, float(1 - (e ** 2).sum() / ((Yn - Yn.mean(0)) ** 2).sum()))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/exp9')
    p.add_argument('--episodes', type=int, default=60)
    p.add_argument('--probe_n', type=int, default=600)
    p.add_argument('--lewm', default='quentinll/lewm-cube')
    p.add_argument('--parts', default='1,2')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    S = Scene()
    M = Metrics('cuda' if torch.cuda.is_available() else 'cpu', args.lewm)

    # part 1: exchange rate
    rec = []
    for ep in range(args.episodes if '1' in args.parts else 0):
        S.reset(1000 + ep); S.set_q(S.q0)
        imgs, tags, disp = [S.render()], ['base'], [0.0]
        for k in (1, 2):
            o = S.move_cube(k * R, rng)
            if o is not None:
                S.set_q(o[0]); imgs.append(S.render()); tags.append(f'obj_{k}r'); disp.append(o[1])
        for d in ARM_D:
            a = S.move_arm(d, rng)
            if a is not None:
                S.set_q(a[0]); imgs.append(S.render()); tags.append(f'arm_{int(d * 100)}cm'); disp.append(a[1])
        if ep == 0:
            np.save(os.path.join(args.out, 'example_frames.npy'), np.stack(imgs))
        F = M.embed(imgs)
        row = {'ep': ep, 'disp': dict(zip(tags, disp))}
        for metric, f in F.items():
            row[metric] = dict(zip(tags[1:], Metrics.cost(f[1:], f[:1]).numpy().tolist()))
        rec.append(row)
    summ = {}
    for metric in (('pixel', 'dinov2', 'lewm') if rec else ()):
        o1 = [r[metric]['obj_1r'] for r in rec if 'obj_1r' in r[metric]]
        s = {'obj_1r': float(np.median(o1)), 'n_obj': len(o1)}
        for d in ARM_D:
            t = f'arm_{int(d * 100)}cm'
            v = [r[metric][t] for r in rec if t in r[metric]]
            s[t] = float(np.median(v)) if v else None
            s[f'ratio_{t}_obj1r'] = s[t] / s['obj_1r'] if v else None
        dstar = []
        for r in rec:
            if 'obj_1r' not in r[metric]:
                continue
            ts = [f'arm_{int(d * 100)}cm' for d in ARM_D if f'arm_{int(d * 100)}cm' in r[metric]]
            if len(ts) < 2:
                continue
            xs = [0.0] + [r['disp'][t] for t in ts]; ys = np.maximum.accumulate([0.0] + [r[metric][t] for t in ts])
            c = r[metric]['obj_1r']
            dstar.append(float(np.interp(c, ys, xs)) if c <= ys[-1] else np.inf)
        s['dstar_cm_median'] = float(np.median(dstar)) * 100 if dstar else None
        s['dstar_n'] = len(dstar)
        summ[metric] = s
        print(f"CUBE {metric:7s} arm2/5/10cm over cube r: " +
              ' '.join(f"{s[f'ratio_arm_{int(d * 100)}cm_obj1r']:.2f}" for d in ARM_D if s[f'ratio_arm_{int(d * 100)}cm_obj1r'] is not None) +
              f"  d* = {s['dstar_cm_median']:.2f} cm (n {s['dstar_n']})", flush=True)
    if rec:
        json.dump({'summary': summ, 'records': rec}, open(os.path.join(args.out, 'part1_exchange.json'), 'w'))

    # part 2: linear probe of the cube position, arm in random poses
    imgs, Y = [], []
    while len(Y) < args.probe_n:
        S.reset(5000 + len(Y))
        o = S.move_cube(rng.uniform(0, 3 * R), rng)
        if o is None:
            continue
        a = S.move_arm(rng.uniform(0.0, 0.15), rng, q_base=o[0])
        if a is None:
            continue
        S.set_q(a[0]); imgs.append(S.render()); Y.append(S.cube())
    Y = np.stack(Y)
    F = M.embed(imgs)
    pr = {'n': len(Y), 'r_cm': R * 100,
          'predict_mean_rmse_cm': float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()) * 100)}
    for metric, f in F.items():
        rmse, lam, r2 = ridge_cv(f.reshape(len(Y), -1).double().numpy(), Y)
        pr[metric] = {'rmse_cm': rmse * 100, 'r2': r2, 'lam': lam}
    print('CUBE PROBE ' + json.dumps(pr), flush=True)
    json.dump(pr, open(os.path.join(args.out, 'part2_probe.json'), 'w'), indent=1)
    print('EXP9_DONE')


if __name__ == '__main__':
    main()
