# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Summary of a closed-loop attribution run (run_closed.py): success ladder with Wilson intervals, attribution
of the world-model failures, and the replan at which each planner first reaches success.

    python -m evals.failure_attribution.summarize_closed --run_dir <dir>
"""

import argparse
import json
import os

import numpy as np

from evals.failure_attribution.analysis import failure_masks
from evals.failure_attribution.io import save_json


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    files = sorted(f for f in os.listdir(args.run_dir) if f.startswith("ep_") and f.endswith(".npz"))
    eps = [np.load(os.path.join(args.run_dir, f)) for f in files]
    if not eps:
        parser.error("No completed ep_*.npz files found in --run_dir")
    names = [n for n in ("wm", "sim_latent", "sim_true") if f"{n}_success" in eps[0].files]
    n = len(eps)
    S = {k: np.array([bool(e[f"{k}_success"]) for e in eps]) for k in names}
    A = {k: np.array([bool(e[f"{k}_success_any"]) for e in eps]) for k in names}
    print(f"{n} episodes | expert success {np.mean([bool(e['expert_success']) for e in eps]) * 100:.1f}%")
    print("\n== success ladder (same start, goal, CEM budget and replanning; success = final step, as the eval)")
    for k in names:
        lo, hi = wilson(S[k].sum(), n)
        print(f"  {k:11s} {S[k].sum():3d}/{n} = {100 * S[k].mean():5.1f}%  [{lo:.1f}, {hi:.1f}]"
              f"   | success at any replan {100 * A[k].mean():5.1f}%")
    out = {"episodes": n, "success": {k: int(S[k].sum()) for k in names}}
    if set(names) == {"wm", "sim_latent", "sim_true"}:
        f = ~S["wm"]
        masks = failure_masks(S["wm"], S["sim_latent"], S["sim_true"])
        dyn, cost, search = (masks[k] for k in ("dynamics", "cost", "search"))
        nf = int(f.sum())
        print(f"\n== attribution of the {nf} world-model failures")
        for lab, m in (("dynamics (sim_latent succeeds)", dyn), ("cost landscape (sim_latent fails, sim_true succeeds)", cost),
                       ("search (both simulator planners fail)", search)):
            print(f"  {lab:55s} {int(m.sum()):3d}  ({100 * m.sum() / max(nf, 1):.0f}% of failures)")
        print("\n== cross-tab (rows wm, cols sim_latent / sim_true; S = success)")
        for w in (True, False):
            r = S["wm"] == w
            cells = [int((r & (S["sim_latent"] == a) & (S["sim_true"] == b)).sum())
                     for a, b in ((True, True), (True, False), (False, True), (False, False))]
            print(f"  wm={'S' if w else 'F'}:  latent=S,true=S {cells[0]:3d} | latent=S,true=F {cells[1]:3d} | "
                  f"latent=F,true=S {cells[2]:3d} | latent=F,true=F {cells[3]:3d}")
        out["attribution"] = {"failures": nf, "dynamics": int(dyn.sum()), "cost": int(cost.sum()), "search": int(search.sum())}
    print("\n== first replan with success (median over successful episodes)")
    for k in names:
        first = [int(np.argmax(e[f"{k}_chunk_success"])) for e in eps if e[f"{k}_chunk_success"].any()]
        print(f"  {k:11s} {np.median(first) if first else float('nan'):.1f}  (n={len(first)})")
    save_json(os.path.join(args.run_dir, "summary.json"), out, indent=1)


if __name__ == "__main__":
    main()
