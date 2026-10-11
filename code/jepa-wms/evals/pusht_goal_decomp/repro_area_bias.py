# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Reproduces the area bias of the released latent goal cost on Push-T (released JEPA-WM / DINO-WM encoders,
64 goals from pusht_goal_decomp).

For every goal we compare two states:
    S  block exactly at the goal, pusher displaced by P px     -> official FAILURE  (P >= 40)
    T  pusher exactly at the goal, block displaced by B px     -> official SUCCESS  (B <= 10)
A cost suited to the task must rank T as closer to the goal than S. We count how often each cost prefers the
failing state S, and how many of the 256 patches each displacement changes.

    python -m evals.pusht_goal_decomp.repro_area_bias --model jepa_wm --figure out.png
"""

import argparse
import glob
import os

import numpy as np

N_PUSHER = 19
PUSHER = {40: [10, 11, 12], 80: [13, 14, 15], 160: [16, 17, 18]}  # pusher variant ids by displacement (px)
BLOCK = {5: [3, 4], 10: [5, 6]}  # block variant ids by displacement (px)


def cell(b, p):
    return b * N_PUSHER + p


def patch_mismatch(vis, goal):
    """(..., 256, 384) vs (256, 384) -> (..., 256) mean squared feature difference per patch."""
    return ((vis - goal) ** 2).mean(-1)


def costs(d):
    return {"mean (released)": d.mean(-1), "top-8 patches": np.sort(d, -1)[..., -8:].mean(-1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jepa_wm")
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()
    root = os.path.join(os.environ["JEPAWM_LOGS"], "pusht_goal_decomp")
    files = sorted(glob.glob(os.path.join(root, args.model, "goal_*.npz")))
    prefer_fail = {}
    n_changed = {"pusher": [], "block": []}
    example = None
    for f in files:
        e = np.load(f)
        s = np.load(os.path.join(root, "sets", os.path.basename(f)))
        V = e["visual_emb"].astype(np.float32).reshape(-1, 256, 384)
        succ = s["cost_official_success"].reshape(-1) > 0.5
        d = patch_mismatch(V, V[0])  # (437, 256); cell 0 is the goal itself
        thr = 0.1 * d.max()
        for P, pids in PUSHER.items():
            for p in pids:
                S = cell(0, p)
                assert not succ[S]
                n_changed["pusher"].append(int((d[S] > thr).sum()))
                for B, bids in BLOCK.items():
                    for b in bids:
                        T = cell(b, 0)
                        if not succ[T]:
                            continue
                        for name, c in costs(d[[S, T]]).items():
                            k = prefer_fail.setdefault((name, P, B), [0, 0])
                            k[0] += int(c[0] < c[1])  # cost prefers the failing state S
                            k[1] += 1
        for b in BLOCK[10]:
            n_changed["block"].append(int((d[cell(b, 0)] > thr).sum()))
        if example is None:
            S, T = cell(0, 13), cell(5, 0)  # pusher +80 px  vs  block +10 px
            example = dict(goal=s["visual"][0, 0], S=s["visual"][0, 13], T=s["visual"][5, 0], dS=d[S], dT=d[T],
                           cS={k: float(v) for k, v in zip(costs(d[S:S + 1]).keys(), [x[0] for x in costs(d[S:S + 1]).values()])},
                           cT={k: float(v[0]) for k, v in costs(d[T:T + 1]).items()})

    print(f"== {args.model}: {len(files)} goals")
    print("patches changed (mismatch > 10% of the grid max): "
          f"pusher displaced 40-160 px -> median {np.median(n_changed['pusher']):.0f} of 256 | "
          f"block displaced 10 px -> median {np.median(n_changed['block']):.0f} of 256")
    print("\nhow often the cost prefers S (block perfect, pusher off -> FAILURE) over T (pusher perfect, block slightly off -> SUCCESS)")
    print(f"{'pusher off':>10s} {'block off':>9s} | " + " | ".join(f"{n:>16s}" for n in ("mean (released)", "top-8 patches")))
    for P in PUSHER:
        for B in BLOCK:
            row = [prefer_fail[(n, P, B)] for n in ("mean (released)", "top-8 patches")]
            print(f"{P:8d}px {B:7d}px | " + " | ".join(f"{100 * a / t:15.1f}%" for a, t in row) + f"   (n={row[0][1]} pairs)")

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ex = example
        fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
        img = lambda x: np.transpose(x, (1, 2, 0))
        vmax = max(ex["dS"].max(), ex["dT"].max())
        ax[0].imshow(img(ex["goal"])); ax[0].set_title("goal")
        ax[1].imshow(img(ex["S"])); ax[1].set_title("S: block at goal, pusher +80 px\nofficial: FAIL")
        ax[2].imshow(ex["dS"].reshape(16, 16), cmap="magma", vmin=0, vmax=vmax)
        ax[2].set_title(f"S patch mismatch\nmean {ex['cS']['mean (released)']:.4f} | top-8 {ex['cS']['top-8 patches']:.3f}")
        ax[3].imshow(img(ex["T"])); ax[3].set_title("T: pusher at goal, block +10 px\nofficial: SUCCESS")
        ax[4].imshow(ex["dT"].reshape(16, 16), cmap="magma", vmin=0, vmax=vmax)
        ax[4].set_title(f"T patch mismatch\nmean {ex['cT']['mean (released)']:.4f} | top-8 {ex['cT']['top-8 patches']:.3f}")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=110)
        print(f"\nfigure: {args.figure}")


if __name__ == "__main__":
    main()
