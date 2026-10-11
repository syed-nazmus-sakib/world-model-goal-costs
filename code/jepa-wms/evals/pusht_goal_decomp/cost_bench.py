# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Offline bench of planning costs on the Push-T goal grids of pusht_goal_decomp (64 goals x 23 block variants
x 19 pusher variants, patch embeddings of the released encoders already saved by encode.py).

The released cost averages squared feature differences over the 256 patches, so a region contributes in
proportion to its area: the small pusher weighs little next to the block. We compare it with aggregations of
the per-patch mismatch that do not scale with area (power means, top-k, max), and with the proprio term.

Scores, per cost, over all non-goal cells of each goal grid:
  rank_acc_official   pairwise ranking agreement with the official state distance
  rank_acc_pusher     ... with the pusher position error only (cells sharing the block variant)
  auc_success         AUC of the cost for separating officially successful cells from failed ones

    python -m evals.pusht_goal_decomp.cost_bench --enc_dir $JEPAWM_LOGS/pusht_goal_decomp/jepa_wm
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata


def pair_acc(cost, ref, mask=None):
    """Fraction of cell pairs (with distinct ref) ordered the same way by cost and ref."""
    n = len(cost)
    iu, ju = np.triu_indices(n, k=1)
    if mask is not None:
        keep = mask[iu, ju]
        iu, ju = iu[keep], ju[keep]
    dr, dc = ref[iu] - ref[ju], cost[iu] - cost[ju]
    k = np.abs(dr) > 1e-9
    return float((np.sign(dc[k]) == np.sign(dr[k])).sum()), int(k.sum())


def auc(score_fail, score_succ):
    """P(cost of a failed cell > cost of a successful cell)."""
    if len(score_fail) == 0 or len(score_succ) == 0:
        return np.nan, 0
    r = rankdata(np.concatenate([score_succ, score_fail]))
    n1, n0 = len(score_fail), len(score_succ)
    u = r[n0:].sum() - n1 * (n1 + 1) / 2
    return float(u), n1 * n0


def costs_for(vis, goal, prop, goal_prop):
    """vis: (N, P, D) cells; goal: (P, D). Returns dict name -> (N,) cost."""
    d_patch = ((vis - goal[None]) ** 2).mean(-1)  # (N, P) per-patch mismatch
    d_prop = ((prop - goal_prop[None]) ** 2).mean(-1)
    out = {"L2 (released visual)": d_patch.mean(1)}
    for a in (0.1, 1.0, 10.0):
        out[f"L2 + {a:g} proprio"] = d_patch.mean(1) + a * d_prop
    for p in (2, 4, 8):
        out[f"power mean p={p}"] = (d_patch ** p).mean(1) ** (1 / p)
    for k in (4, 8, 16):
        out[f"top-{k} patches"] = np.sort(d_patch, 1)[:, -k:].mean(1)
    out["max patch"] = d_patch.max(1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc_dir", required=True)
    parser.add_argument("--sets_dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    sets_dir = args.sets_dir or os.path.join(os.path.dirname(args.enc_dir.rstrip("/")), "sets")
    acc = {}
    for f in sorted(glob.glob(os.path.join(args.enc_dir, "goal_*.npz"))):
        e = np.load(f)
        s = np.load(os.path.join(sets_dir, os.path.basename(f)))
        V = e["visual_emb"].astype(np.float32)  # (B, Pv, 256, 384)
        Pr = e["proprio_emb"].astype(np.float32)  # (B, Pv, 16)
        B, Pv = V.shape[:2]
        vis, prop = V.reshape(B * Pv, *V.shape[2:]), Pr.reshape(B * Pv, -1)
        cells = np.arange(1, B * Pv)  # cell 0 = the goal itself
        official = s["cost_official_state_dist"].reshape(-1)[cells]
        pusher = s["cost_pusher_pos_err"].reshape(-1)[cells]
        succ = s["cost_official_success"].reshape(-1)[cells] > 0.5
        block_id = np.repeat(np.arange(B), Pv)[cells]
        same_block = block_id[:, None] == block_id[None, :]
        for name, c in costs_for(vis[cells], vis[0], prop[cells], prop[0]).items():
            a = acc.setdefault(name, np.zeros(6))
            a[0:2] += pair_acc(c, official)
            a[2:4] += pair_acc(c, pusher, same_block)
            a[4:6] += auc(c[~succ], c[succ])
    res = {name: {"rank_acc_official": a[0] / a[1], "rank_acc_pusher_same_block": a[2] / a[3],
                  "auc_success": a[4] / a[5] if a[5] else float("nan")} for name, a in acc.items()}
    print(f"{args.enc_dir}")
    print(f"{'cost':24s} {'rank vs official':>16s} {'rank vs pusher':>15s} {'AUC success':>12s}")
    for name, r in res.items():
        print(f"{name:24s} {r['rank_acc_official']:16.3f} {r['rank_acc_pusher_same_block']:15.3f} {r['auc_success']:12.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
