# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Summarizes sim_test.py: for each hidden-physics variant and horizon, how far do the futures of the
same action sequences diverge from the nominal world, and do the candidate rankings change.

Usage:
    python -m evals.mw_hidden_physics.summarize --parts "$JEPAWM_LOGS/mw_hidden_physics/part*.npz" --out <summary.json>
"""

import argparse
import glob
import json

import numpy as np
from scipy import stats

HORIZONS = [1, 5, 10, 20, 30]
COST_MARGIN = 0.01  # m, pairs of candidates closer than this in nominal cost are not counted
SUCCESS_RADIUS = 0.05  # m, Metaworld push


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(args.parts))
    parts = [np.load(f) for f in files]
    variants = [v[0] for v in json.loads(str(parts[0]["variants"]))]
    obj = np.concatenate([p["obj"] for p in parts])  # (B, V, K, H+1, 3)
    tcp = np.concatenate([p["tcp"] for p in parts])
    cost = np.concatenate([p["cost"] for p in parts])  # (B, V, K, H+1)
    pixel_diff = np.concatenate([p["pixel_diff"] for p in parts])
    B, V, K, _ = cost.shape
    iu, ju = np.triu_indices(K, k=1)

    q = lambda x: {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(np.max(x))}
    summary = {
        "n_base_states": int(B),
        "n_candidates": int(K),
        "variants": variants,
        "max_pixel_diff_vs_nominal": float(pixel_diff.max()),
        "frac_base_states_with_any_pixel_diff": float((pixel_diff.max(1) > 0).mean()),
        "nominal": {},
        "by_variant": {},
    }
    for h in HORIZONS:
        c0 = cost[:, 0, :, h]
        summary["nominal"][f"h{h}"] = {
            "obj_displacement_from_start": q(np.linalg.norm(obj[:, 0, :, h] - obj[:, 0, :, 0], axis=-1)),
            "cost_spread_across_candidates_std": float(np.median(c0.std(1))),
            "best_minus_median_candidate_cost": float(np.median(np.median(c0, 1) - c0.min(1))),
        }
    for v, name in enumerate(variants[1:], start=1):
        res = {}
        for h in HORIZONS:
            cA, cB = cost[:, 0, :, h], cost[:, v, :, h]
            rho = [stats.spearmanr(cA[b], cB[b])[0] for b in range(B) if cA[b].std() > 1e-9 and cB[b].std() > 1e-9]
            dA, dB = cA[:, iu] - cA[:, ju], cB[:, iu] - cB[:, ju]
            labelled = np.abs(dA) > COST_MARGIN
            best_A, best_B = cA.argmin(1), cB.argmin(1)
            # cost paid in world B by executing the candidate that is best in the nominal world
            regret = cB[np.arange(B), best_A] - cB.min(1)
            succ_A_plan_in_B = cB[np.arange(B), best_A] <= SUCCESS_RADIUS
            succ_B_plan_in_B = cB.min(1) <= SUCCESS_RADIUS
            res[f"h{h}"] = {
                "obj_divergence": q(np.linalg.norm(obj[:, v, :, h] - obj[:, 0, :, h], axis=-1)),
                "hand_divergence": q(np.linalg.norm(tcp[:, v, :, h] - tcp[:, 0, :, h], axis=-1)),
                "task_cost_divergence": q(np.abs(cB - cA)),
                "ranking_spearman_median": float(np.median(rho)),
                "pair_reversal_rate": float((np.sign(dA[labelled]) != np.sign(dB[labelled])).mean()),
                "best_candidate_changes_rate": float((best_A != best_B).mean()),
                "regret_of_nominal_best": q(regret),
                "frac_regret_over_1cm": float((regret > 0.01).mean()),
                "frac_regret_over_success_radius": float((regret > SUCCESS_RADIUS).mean()),
                "success_flip_rate": float((succ_A_plan_in_B != succ_B_plan_in_B).mean()),
            }
        summary["by_variant"][name] = res
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"{B} base states x {K} candidates, max pixel diff vs nominal: {pixel_diff.max():.0f}")
    for h in HORIZONS:
        n = summary["nominal"][f"h{h}"]
        print(f"nominal h={h}: obj moves median {100 * n['obj_displacement_from_start']['median']:.2f} cm, "
              f"p90 {100 * n['obj_displacement_from_start']['p90']:.2f} cm; "
              f"best-vs-median candidate cost gap {100 * n['best_minus_median_candidate_cost']:.2f} cm")
    for h in (10, 30):
        print(f"\n--- horizon {h} env steps (all distances in cm)")
        print(f"{'variant':20s} {'obj div med/p90':>16s} {'hand div med/p90':>17s} {'cost div med/p90':>17s} "
              f"{'spearman':>8s} {'pair rev':>8s} {'best chg':>8s} {'regret med/p90/max':>19s} {'>1cm':>5s} {'>5cm':>5s}")
        for name, res in summary["by_variant"].items():
            r = res[f"h{h}"]
            print(f"{name:20s} {100 * r['obj_divergence']['median']:7.2f}/{100 * r['obj_divergence']['p90']:6.2f} "
                  f"{100 * r['hand_divergence']['median']:9.2f}/{100 * r['hand_divergence']['p90']:6.2f} "
                  f"{100 * r['task_cost_divergence']['median']:9.2f}/{100 * r['task_cost_divergence']['p90']:6.2f} "
                  f"{r['ranking_spearman_median']:8.3f} {r['pair_reversal_rate']:8.3f} {r['best_candidate_changes_rate']:8.2f} "
                  f"{100 * r['regret_of_nominal_best']['median']:6.2f}/{100 * r['regret_of_nominal_best']['p90']:5.2f}/"
                  f"{100 * r['regret_of_nominal_best']['max']:5.2f} {r['frac_regret_over_1cm']:5.2f} "
                  f"{r['frac_regret_over_success_radius']:5.2f}")


if __name__ == "__main__":
    main()
