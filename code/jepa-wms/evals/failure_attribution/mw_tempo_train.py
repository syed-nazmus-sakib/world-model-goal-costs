# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
TEMPO baseline for JEPA-WM Metaworld (Moukheiber, Xue, Chen, arXiv 2610.04988; code github.com/lamawmouk/tempo), 2026-10-24.

Re-implements TEMPO's temporal map on our frozen encodings so the comparison runs inside our planner:
  input   [patch-mean visual embedding, proprio embedding] (TEMPO's patch-grid convention: pool_grid + extra embeddings)
  F_phi   MLP d_in -> 256 -> 256 -> 64 (SiLU), trained so that ||F(z_t) - F(z_{t+k})|| = k / kappa for same-episode pairs with
          k <= max_steps (smooth L1), and >= max_steps / kappa for pairs farther apart (hinge, weight 0.5)
  cost    J = (1 - w) * own + w * d_H^2 * ||F(z) - F(G)||^2, w = 0.5, d_H^2 = median squared own-cost displacement over
          H-step pairs (H = one plan in env steps), so both terms share units (costs.TempoMixObjective)
Training data: the expert frames (kind 0, every 2nd step) and the eval goals of the pilot-data episodes of all given tasks.
Not reproduced: TEMPO's online rounds on the planner's own episodes (offline map only; stated in the write-up).

    python -m evals.failure_attribution.mw_tempo_train --dirs <pilot train dirs> --out <map.pt> [--kappa 25 --max_steps 75 --plan_steps 30]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TemporalMap(nn.Module):
    def __init__(self, d_in, out_dim=64, hidden=256, kappa=25.0):
        super().__init__()
        self.kappa = float(kappa)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def pooled(vis, prop):
    """(N, P, D) fp16 patches + (N, Dp) proprio embedding -> (N, D + Dp) float: TEMPO's input for patch-grid models"""
    return torch.cat([torch.from_numpy(vis).float().mean(1), torch.from_numpy(prop).float()], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, default=25.0)
    ap.add_argument("--max_steps", type=int, default=75)
    ap.add_argument("--plan_steps", type=int, default=30, help="env steps of one plan (horizon 6 x frameskip 5)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    X, T, E, OWN = [], [], [], []
    n_ep = 0
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep_*.npz"))):
            z = np.load(f)
            keep = np.isin(z["kind"], [0, 1]) & ((z["kind"] == 0) | (z["var"] == 0))  # expert frames + the eval goal
            vis, prop, t = z["vis"][keep], z["prop"][keep], z["t"][keep]
            order = np.argsort(t)
            X.append(pooled(vis[order], prop[order])), T.append(t[order]), E.append(np.full(keep.sum(), n_ep))
            v = torch.from_numpy(vis[order]).float()  # own cost (mean squared patch difference) over H-step pairs
            tt = t[order]
            for i in range(len(tt)):
                j = np.searchsorted(tt, tt[i] + args.plan_steps)
                if j < len(tt) and abs(tt[j] - tt[i] - args.plan_steps) <= 2:
                    OWN.append(float((v[i] - v[j]).pow(2).mean()))
            n_ep += 1
    X, T, E = torch.cat(X).cuda(), np.concatenate(T), np.concatenate(E)
    d_H2 = float(np.median(OWN))
    print(f"{len(X)} frames, {n_ep} episodes, d_H^2 (median own cost over {args.plan_steps}-step pairs) = {d_H2:.4g}", flush=True)
    eps = np.unique(E)
    val = np.isin(E, rng.choice(eps, size=max(1, len(eps) // 10), replace=False))
    first = {e: np.nonzero(E == e)[0] for e in eps}

    def pairs(mask, n, far):
        a = []
        pool = [e for e in eps if (np.isin(first[e], np.nonzero(mask)[0]).any())]
        for e in rng.choice(pool, size=n):
            ix = first[e]
            i = rng.integers(len(ix))
            later = ix[T[ix] > T[ix[i]] + (args.max_steps if far else 0)]
            later = later[T[later] - T[ix[i]] <= args.max_steps] if not far else later
            if len(later) == 0:
                continue
            j = rng.choice(later)
            a.append((ix[i], j, T[j] - T[ix[i]]))
        if not a:
            return None
        a = np.array(a)
        return a[:, 0], a[:, 1], torch.from_numpy(a[:, 2].astype(np.float32)).cuda()

    fmap = TemporalMap(X.shape[1], kappa=args.kappa).cuda()
    opt = torch.optim.Adam(fmap.parameters(), lr=1e-3)
    for step in range(args.steps):
        i, j, k = pairs(~val, 256, far=False)
        loss = F.smooth_l1_loss((fmap(X[i]) - fmap(X[j])).norm(dim=-1), k / fmap.kappa)
        far = pairs(~val, 256, far=True)
        if far is not None:
            fi, fj, _ = far
            loss = loss + 0.5 * F.relu(args.max_steps / fmap.kappa - (fmap(X[fi]) - fmap(X[fj])).norm(dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    fmap.eval()
    with torch.no_grad():
        i, j, k = pairs(val, 2000, far=False)
        pred = (fmap(X[i]) - fmap(X[j])).norm(dim=-1) * fmap.kappa
        mae = float((pred - k).abs().mean())
    rep = {"val_mae_steps": mae, "d_H2": d_H2, "n_frames": int(len(X)), "n_episodes": int(n_ep), "args": vars(args)}
    print(json.dumps(rep), flush=True)
    torch.save({"state_dict": fmap.state_dict(), "d_in": X.shape[1], "kappa": args.kappa, "d_H2": d_H2, "report": rep}, args.out)


if __name__ == "__main__":
    main()
