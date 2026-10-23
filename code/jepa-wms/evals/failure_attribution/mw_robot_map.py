# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Robot map for the reach-then-change cost (2026-10-23): which DINOv2 patches show the robot?

Two per-patch heads on the frozen patch features (mw_pilot_data --robot_mask encodings), trained on all given tasks:
  mask     logistic regression per patch, labels = the render mask (robot hidden vs shown); needs the robot's appearance
           at training time, no task labels. Reported: accuracy, IoU, AUC on held-out episodes.
  proprio  self-supervised: a small MLP predicts the hand position from one patch; robot patches predict it well,
           background and objects do not. Robot score = exp(-err^2 / s^2) with s fitted on the training set.
           Reported: AUC against the render mask on held-out episodes (the mask is only used to evaluate).

    python -m evals.failure_attribution.mw_robot_map --dirs <pilot dir> [<pilot dir> ...] --out_dir <dir>
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from torch import nn


class PatchRobotHead(nn.Module):
    """Per-patch head: (N, P, D) features -> (N, P) robot logit (mask mode) or (N, P, 3) hand position (proprio mode)."""

    def __init__(self, d_in, mode, width=256):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Linear(d_in, 1) if mode == "mask" else nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, 3))
        self.err_scale = nn.Parameter(torch.tensor(0.05), requires_grad=False)  # proprio mode: s in exp(-err^2 / s^2)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1) if self.mode == "mask" else self.net(self.norm(x))

    @torch.no_grad()
    def robot_prob(self, x, hand=None):
        """(N, P, D) [+ hand (N, 3)] -> (N, P) probability that the patch shows the robot."""
        if self.mode == "mask":
            return torch.sigmoid(self(x))
        err = (self(x) - hand[:, None, :]).pow(2).sum(-1)
        return torch.exp(-err / self.err_scale.pow(2))


def load(dirs):
    X, M, H, E = [], [], [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep_*.npz"))):
            z = np.load(f)
            keep = z["rmask"][:, 0] != 255
            X.append(z["vis"][keep]), M.append(z["rmask"][keep]), H.append(z["state"][keep, :3])
            E.append(np.full(keep.sum(), len(E)))
    return np.concatenate(X), np.concatenate(M).astype(np.float32), np.concatenate(H).astype(np.float32), np.concatenate(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--modes", default="mask,proprio")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    X, M, H, E = load(args.dirs)
    eps = np.unique(E)
    val = np.isin(E, rng.choice(eps, size=max(1, len(eps) // 10), replace=False))
    print(f"{len(X)} frames from {len(eps)} episodes, robot fraction {M.mean():.3f}, {val.sum()} held-out frames", flush=True)
    Xt, Mt, Ht = torch.from_numpy(X).cuda(), torch.from_numpy(M).cuda(), torch.from_numpy(H).cuda()
    tr, va = np.nonzero(~val)[0], np.nonzero(val)[0]
    report = {}
    for mode in args.modes.split(","):
        head = PatchRobotHead(X.shape[2], mode).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(args.steps):
            i = rng.choice(tr, size=64)
            x = Xt[i].float()
            if mode == "mask":
                loss = F.binary_cross_entropy_with_logits(head(x), Mt[i])
            else:
                loss = (head(x) - Ht[i][:, None, :]).pow(2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            if mode == "proprio":  # s = median per-patch error of robot patches on the training set
                i = tr[:2000]
                err = (head(Xt[i].float()) - Ht[i][:, None, :]).pow(2).sum(-1).sqrt()
                head.err_scale.fill_(float(err[Mt[i] > 0.5].median()) * 2)
            p = torch.cat([head.robot_prob(Xt[va[s : s + 256]].float(), Ht[va[s : s + 256]]) for s in range(0, len(va), 256)])
        p, m = p.cpu().numpy().ravel(), M[va].ravel()
        pred = p > 0.5
        pos, neg = p[m > 0.5], p[m <= 0.5]
        auc = float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))) if len(pos) and len(neg) else float("nan")
        rep = {"auc": auc, "acc": float((pred == (m > 0.5)).mean()),
               "iou": float((pred & (m > 0.5)).sum() / max(1, (pred | (m > 0.5)).sum())), "final_loss": float(loss)}
        if mode == "proprio":
            rep["err_scale"] = float(head.err_scale)
        report[mode] = rep
        print(mode, json.dumps(rep), flush=True)
        torch.save({"state_dict": head.state_dict(), "d_in": X.shape[2], "mode": mode, "report": rep, "dirs": args.dirs},
                   os.path.join(args.out_dir, f"robot_{mode}.pt"))
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
