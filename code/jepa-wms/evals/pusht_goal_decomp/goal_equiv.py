# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Goal-equivalence test, Push-T side (the CONTROL case: here the pusher pose is part of success).

For each of the 64 goal grids, the original goal G0 (block and pusher at goal) is replaced by goal variants Gj
with the block at the same pose and the pusher displaced by 10 / 40 / 80 px. For a fixed set of candidate
final states (all 437 cells of the grid) we measure, between G0 and Gj:
  - cost sensitivity: relative change of the latent cost of the same candidate states
  - ranking flips: fraction of candidate pairs whose order changes, for the latent cost and for the official
    success distance (the reference: in Push-T a correct cost SHOULD flip as often as the official one)

    python -m evals.pusht_goal_decomp.goal_equiv jepa_wm dino_wm
"""

import glob
import os
import sys

import numpy as np

N_P = 19
VARIANTS = {10: [4, 5, 6], 40: [10, 11, 12], 80: [13, 14, 15]}


def official(states, goal):
    pos = np.linalg.norm(states[:, :4] - goal[:4], axis=1)
    d = np.abs(states[:, 4] - goal[4]) % (2 * np.pi)
    return pos + 100 * np.minimum(d, 2 * np.pi - d)  # lexicographic-ish: angle error counts heavily


def flip_rate(c0, c1, rng, n_pairs=20000):
    i, j = rng.integers(0, len(c0), (2, n_pairs))
    k = (i != j) & (np.abs(c0[i] - c0[j]) > 1e-12) & (np.abs(c1[i] - c1[j]) > 1e-12)
    return float((np.sign(c0[i] - c0[j]) != np.sign(c1[i] - c1[j]))[k].mean())


def main():
    root = os.path.join(os.environ["JEPAWM_LOGS"], "pusht_goal_decomp")
    rng = np.random.default_rng(0)
    for model in sys.argv[1:] or ["jepa_wm"]:
        res = {m: {"sens": [], "flip_latent": [], "flip_official": []} for m in VARIANTS}
        for f in sorted(glob.glob(os.path.join(root, model, "goal_*.npz"))):
            e = np.load(f)
            s = np.load(os.path.join(root, "sets", os.path.basename(f)))
            V = e["visual_emb"].astype(np.float32).reshape(-1, 256, 384)
            P = e["proprio_emb"].astype(np.float32).reshape(V.shape[0], -1)
            st = s["realized"].reshape(-1, 7)
            cost = lambda g: ((V - V[g]) ** 2).mean((1, 2)) + 0.1 * ((P - P[g]) ** 2).mean(1)
            c0, o0 = cost(0), official(st, st[0])
            for mag, ps in VARIANTS.items():
                for p in ps:
                    g = p  # cell (block 0, pusher p)
                    c1, o1 = cost(g), official(st, st[g])
                    res[mag]["sens"].append(float(np.median(np.abs(c1 - c0) / (c0 + 1e-9))))
                    res[mag]["flip_latent"].append(flip_rate(c0, c1, rng))
                    res[mag]["flip_official"].append(flip_rate(o0, o1, rng))
        print(f"== {model}: goal variants with the same block pose, pusher displaced (pusher pose IS part of success)")
        print(f"{'pusher shift':>12s} {'median rel. cost change':>24s} {'ranking flips: latent':>22s} {'official':>9s}")
        for mag, r in res.items():
            print(f"{mag:10d}px {np.mean(r['sens']):24.2f} {np.mean(r['flip_latent']):21.3f} {np.mean(r['flip_official']):9.3f}")


if __name__ == "__main__":
    main()
