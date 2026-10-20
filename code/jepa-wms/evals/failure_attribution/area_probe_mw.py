# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Arm-versus-object weighting of the frozen goal metric, and whether the object is in the features (2026-10-19).

Kinematic only (qpos set, mj_forward, render; no physics step), through the eval's MetaWorldWrapper render path
(corner2 camera, 224 px). Metrics: frozen DINOv2 ViT-S/14 patch tokens (the JEPA-WM / DINO-WM Metaworld encoder),
mean over patches of the squared token distance; and raw pixel MSE (area-weighted by construction).

Part 1, exchange rate. Per episode (a reset placement): move only the object along its joint(s) so that the task
  position (_get_pos_objects) moves by r and 2r, r = the task's success distance; move only the arm so that the hand
  (tcp_center) moves by 2, 5, 10 cm in a random direction with the minimum joint change (damped pseudo-inverse of
  the hand Jacobian, as IK would), without contact with non-robot geoms. Reports
  cost(arm d) / cost(object r) and d* = the arm displacement whose cost equals that of the object off by r.
Part 2, linear probe. Random object joint values over a task-scale range and random arm poses; ridge regression
  (dual form, 5-fold CV) from the patch tokens to the 3-d object position, with the robot visible and with the robot
  hidden. Reports RMSE in cm against r: RMSE < r means the features resolve the object at the task's precision.

    python -m evals.failure_attribution.area_probe_mw --out <dir> --episodes 30 --probe_n 600
"""

import argparse
import json
import os

import mujoco
import numpy as np
import torch

from evals.failure_attribution.mw_sim import SUCCESS_THRESHOLD, object_joints, render_hidden, robot_bodies, robot_geoms

RELEASED = ["mw-reach", "mw-reach-wall", "mw-button-press", "mw-door-lock", "mw-drawer-close", "mw-faucet-close",
            "mw-faucet-open", "mw-box-close"]
EXTRA = ["mw-drawer-open", "mw-plate-slide", "mw-push"]
R_DEFAULT = 0.05
ARM_D = [0.02, 0.05, 0.10]
FREE, SLIDE, HINGE = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE


def make_wrapper(task, img_size=224):
    from evals.mw_metric_conflict.collect import make_cfg
    from evals.simu_env_planning.envs.metaworld import make_env

    return make_env(make_cfg(task, img_size, 0)).env  # MetaWorldWrapper


class Scene:
    def __init__(self, task):
        self.task = task
        self.w = make_wrapper(task)
        self.mw = self.w.unwrapped
        m = self.mw.model
        self.r = SUCCESS_THRESHOLD.get(task, R_DEFAULT)
        rb = robot_bodies(m)
        self.robot_g = robot_geoms(m)
        self.robot_gset = set(self.robot_g.tolist())
        floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.arm_j = [j for j in range(m.njnt) if m.jnt_bodyid[j] in rb and m.jnt_type[j] == HINGE
                      and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("right_j")]
        self.obj_j = [j for j in object_joints(m) if m.jnt_type[j] in (FREE, SLIDE, HINGE)]
        self.floor = floor

    def reset(self, seed):
        self.mw._freeze_rand_vec = False
        self.mw.seeded_rand_vec = True
        self.w.reset(seed=seed)
        self.q0 = self.mw.data.qpos.copy()

    def set_q(self, q):
        d = self.mw.data
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(self.mw.model, d)

    def obj_pos(self):
        return np.array(self.mw._get_pos_objects()[:3])

    def hand(self):
        return np.array(self.mw.tcp_center)

    def robot_touches_scene(self):
        d, m = self.mw.data, self.mw.model
        for i in range(d.ncon):
            a, b = d.contact[i].geom1, d.contact[i].geom2
            ra, rb_ = a in self.robot_gset, b in self.robot_gset
            if ra != rb_ and d.contact[i].dist < 0.0:
                other = b if ra else a
                if m.geom_bodyid[other] != 0:  # touching anything but the world body (table top / floor) counts
                    return True
        return False

    def render(self, hide_robot=False):
        return render_hidden(self.w, self.robot_g) if hide_robot else self.w.render().copy()

    # -------------------------------------------------------------- perturbations
    def _obj_dirs(self):
        m = self.mw.model
        dirs = []
        for j in self.obj_j:
            a = m.jnt_qposadr[j]
            if m.jnt_type[j] == FREE:
                dirs += [(a, 0), (a + 1, 0)]  # x, y translation of the body
            else:
                dirs.append((a, j))
        return dirs

    def move_object(self, target, rng, tries=40):
        """qpos with the object's task position moved by ~target metres (and the robot untouched)."""
        m = self.mw.model
        dirs = self._obj_dirs()
        if not dirs:
            return None
        self.set_q(self.q0); p0 = self.obj_pos()
        for _ in range(tries):
            v = rng.standard_normal(len(dirs)); v /= np.linalg.norm(v)
            q = self.q0.copy()
            for (adr, _), vi in zip(dirs, v):
                q[adr] += 1e-3 * vi
            self.set_q(q); g = np.linalg.norm(self.obj_pos() - p0) / 1e-3
            if g < 1e-3:
                continue
            q = self.q0.copy()
            for (adr, j), vi in zip(dirs, v):
                q[adr] += target / g * vi
                if j and m.jnt_limited[j]:
                    q[adr] = np.clip(q[adr], *m.jnt_range[j])
            self.set_q(q)
            dist = np.linalg.norm(self.obj_pos() - p0)
            if 0.7 * target < dist < 1.3 * target and not self.robot_touches_scene():
                return q, dist
        return None

    def _hand_jac(self, q, adr, eps=1e-4):
        self.set_q(q); h = self.hand(); J = np.zeros((3, len(adr)))
        for k, a in enumerate(adr):
            qq = q.copy(); qq[a] += eps; self.set_q(qq); J[:, k] = (self.hand() - h) / eps
        return J, h

    def move_arm(self, target, rng, q_base=None, tries=40):
        """Move the hand by `target` metres in a random direction with the smallest joint change (damped
        pseudo-inverse of the hand Jacobian, a few Newton steps), as an IK / mocap controller would. A random
        joint-space direction would also swing the elbow through its null space and inflate the arm's cost."""
        m = self.mw.model
        q_base = self.q0 if q_base is None else q_base
        adr = [m.jnt_qposadr[j] for j in self.arm_j]
        self.set_q(q_base); h0 = self.hand()
        for _ in range(tries):
            u = rng.standard_normal(3); u[2] = abs(u[2]) * 0.5; u /= np.linalg.norm(u)  # not into the table
            goal = h0 + target * u
            q = q_base.copy()
            for _ in range(6):
                J, h = self._hand_jac(q, adr)
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), goal - h)
                q[adr] += dq
                for j, a in zip(self.arm_j, adr):
                    if m.jnt_limited[j]:
                        q[a] = np.clip(q[a], *m.jnt_range[j])
            self.set_q(q)
            dist = np.linalg.norm(self.hand() - h0)
            if np.linalg.norm(self.hand() - goal) < 0.1 * target and not self.robot_touches_scene():
                return q, dist
        return None


class Metrics:
    def __init__(self, device):
        self.dev = device
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, imgs, bs=64):
        out = {"pixel": [], "dinov2": []}
        for i in range(0, len(imgs), bs):
            raw = torch.as_tensor(np.stack(imgs[i:i + bs]))
            x = raw.to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            out["pixel"].append(raw.float().div(255.0).reshape(len(raw), -1))
            out["dinov2"].append(self.dino.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"].float().cpu())
        return {k: torch.cat(v) for k, v in out.items()}

    @staticmethod
    def cost(a, b):
        d = (a - b).pow(2)
        return d.sum(-1).mean(-1) if d.dim() == 3 else d.mean(-1)


def part1(M, task, args, rng):
    S = Scene(task)
    rec = []
    for ep in range(args.episodes):
        S.reset(seed=1000 + ep)
        S.set_q(S.q0)
        imgs, tags, disp = [S.render()], ["base"], [0.0]
        for k in (1, 2):
            o = S.move_object(k * S.r, rng)
            if o is not None:
                S.set_q(o[0]); imgs.append(S.render()); tags.append(f"obj_{k}r"); disp.append(o[1])
        for d in ARM_D:
            a = S.move_arm(d, rng)
            if a is not None:
                S.set_q(a[0]); imgs.append(S.render()); tags.append(f"arm_{int(d * 100)}cm"); disp.append(a[1])
        F = M.embed(imgs)
        row = {"ep": ep, "disp": dict(zip(tags, disp))}
        for metric, f in F.items():
            c = Metrics.cost(f[1:], f[:1]).numpy()
            row[metric] = dict(zip(tags[1:], c.tolist()))
        rec.append(row)
    summ = {"task": task, "r": S.r, "n_obj_joints": len(S.obj_j), "episodes": len(rec)}
    for metric in ("pixel", "dinov2"):
        def med(t):
            v = [r[metric][t] for r in rec if t in r[metric]]
            return (float(np.median(v)) if v else None), len(v)
        o1, n1 = med("obj_1r")
        s = {"obj_1r": o1, "n_obj": n1}
        dstar = []
        for r in rec:
            if "obj_1r" not in r[metric]:
                continue
            xs = [0.0] + [r["disp"][f"arm_{int(d * 100)}cm"] for d in ARM_D if f"arm_{int(d * 100)}cm" in r[metric]]
            ys = [0.0] + [r[metric][f"arm_{int(d * 100)}cm"] for d in ARM_D if f"arm_{int(d * 100)}cm" in r[metric]]
            if len(xs) < 3:
                continue
            ys_m = np.maximum.accumulate(ys)
            t = r[metric]["obj_1r"]
            dstar.append(float(np.interp(t, ys_m, xs)) if t <= ys_m[-1] else np.inf)
        for d in ARM_D:
            a, _ = med(f"arm_{int(d * 100)}cm")
            s[f"arm_{int(d * 100)}cm"] = a
            s[f"ratio_arm{int(d * 100)}cm_obj1r"] = (a / o1) if (a is not None and o1) else None
        s["dstar_cm_median"] = float(np.median(dstar)) * 100 if dstar else None
        s["dstar_n"] = len(dstar)
        summ[metric] = s
    return summ, rec


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


def part2(M, task, args, rng):
    S = Scene(task)
    vis, hid, Y = [], [], []
    while len(Y) < args.probe_n:
        S.reset(seed=5000 + len(Y))
        o = S.move_object(rng.uniform(0, 4 * S.r), rng)
        if o is None:
            continue
        a = S.move_arm(rng.uniform(0.01, 0.15), rng, q_base=o[0])
        if a is None:
            continue
        S.set_q(a[0])
        vis.append(S.render()); hid.append(S.render(hide_robot=True)); Y.append(S.obj_pos())
    Y = np.stack(Y)
    out = {"task": task, "r_cm": S.r * 100, "n": len(Y), "obj_pos_std_cm": (Y.std(0) * 100).tolist()}
    for name, imgs in (("robot_visible", vis), ("robot_hidden", hid)):
        F = M.embed(imgs)
        for metric in ("dinov2", "pixel"):
            X = F[metric].reshape(len(Y), -1).double().numpy()
            rmse, lam, r2 = ridge_cv(X, Y)
            out[f"{name}/{metric}"] = {"rmse_cm": rmse * 100, "r2": float(r2), "lam": lam}
        Xm = F["dinov2"].mean(1).double().numpy()
        rmse, lam, r2 = ridge_cv(Xm, Y)
        out[f"{name}/dinov2_meanpool"] = {"rmse_cm": rmse * 100, "r2": float(r2), "lam": lam}
    base = np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()) * 100
    out["predict_mean_rmse_cm"] = float(base)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--probe_n", type=int, default=600)
    p.add_argument("--tasks", default=",".join(RELEASED + EXTRA))
    p.add_argument("--probe_tasks", default="mw-drawer-open,mw-plate-slide,mw-faucet-open,mw-drawer-close,mw-push,mw-button-press")
    p.add_argument("--parts", default="1,2")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    M = Metrics("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    if "1" in args.parts:
        allsum = []
        for task in args.tasks.split(","):
            summ, rec = part1(M, task, args, rng)
            allsum.append(summ)
            json.dump(rec, open(os.path.join(args.out, f"part1_{task}.json"), "w"))
            d, px = summ["dinov2"], summ["pixel"]
            print(f"{task:18s} r={summ['r']*100:.1f}cm n_obj={d['n_obj']:2d} | dinov2 arm5cm/obj_r "
                  f"{d['ratio_arm5cm_obj1r'] if d['ratio_arm5cm_obj1r'] is None else round(d['ratio_arm5cm_obj1r'], 2)}"
                  f" d*={d['dstar_cm_median']} cm | pixel arm5cm/obj_r "
                  f"{px['ratio_arm5cm_obj1r'] if px['ratio_arm5cm_obj1r'] is None else round(px['ratio_arm5cm_obj1r'], 2)}"
                  f" d*={px['dstar_cm_median']} cm", flush=True)
        json.dump(allsum, open(os.path.join(args.out, "part1_summary.json"), "w"), indent=1)
    if "2" in args.parts:
        res = []
        for task in args.probe_tasks.split(","):
            o = part2(M, task, args, rng)
            res.append(o)
            print(f"PROBE {task:18s} r={o['r_cm']:.1f}cm mean-pred {o['predict_mean_rmse_cm']:.1f}cm | " +
                  " | ".join(f"{k} {v['rmse_cm']:.2f}cm R2 {v['r2']:.2f}" for k, v in o.items() if isinstance(v, dict)),
                  flush=True)
        json.dump(res, open(os.path.join(args.out, "part2_probe.json"), "w"), indent=1)
    print("AREA_PROBE_DONE")


if __name__ == "__main__":
    main()
