# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Decomposes the latent goal cost of encoded Push-T states into block and pusher contributions.

Usage:
    python -m evals.pusht_goal_decomp.analyze --sets_dir <sets> --enc_dir <encode out_dir> --out <results.json>
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy import stats

ALPHAS = [0.0, 0.1]
BLOCK_GAP_BINS = [0, 5, 10, 20, 40, np.inf]
PUSHER_GAP_BINS = [0, 10, 20, 40, 80, np.inf]


def load(sets_dir, enc_dir):
    meta = json.load(open(os.path.join(sets_dir, "meta.json")))
    costs = {k: [] for k in meta["cost_keys"]}
    d_vis, d_prop, enc_files = [], [], []
    for f in sorted(glob.glob(os.path.join(sets_dir, "goal_*.npz"))):
        s = np.load(f)
        for k in costs:
            costs[k].append(s[f"cost_{k}"])
        enc_file = os.path.join(enc_dir, os.path.basename(f))
        e = np.load(enc_file)
        d_vis.append(e["d_visual"])
        d_prop.append(e["d_proprio"])
        enc_files.append(enc_file)
    costs = {k: np.stack(v).astype(np.float64) for k, v in costs.items()}
    return meta, costs, np.stack(d_vis).astype(np.float64), np.stack(d_prop).astype(np.float64), enc_files


def med_iqr(x):
    x = np.asarray(x).ravel()
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))}


def variant_levels(specs):
    """Maps (kind, magnitude) -> list of variant indices."""
    levels = {}
    for idx, s in enumerate(specs):
        levels.setdefault((s["kind"], s["magnitude"]), []).append(idx)
    return levels


def partial_corr(x, y, z):
    """Pearson correlation of x and y after regressing z out of both."""
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var()), beta


def regression(cost, block, pusher):
    r2_full, beta = r2([block, pusher], cost)
    r2_b, _ = r2([block], cost)
    r2_p, _ = r2([pusher], cost)
    rb, rp, rc = stats.rankdata(block), stats.rankdata(pusher), stats.rankdata(cost)
    return {
        "pearson_block": float(np.corrcoef(cost, block)[0, 1]),
        "pearson_pusher": float(np.corrcoef(cost, pusher)[0, 1]),
        "spearman_block": float(np.corrcoef(rc, rb)[0, 1]),
        "spearman_pusher": float(np.corrcoef(rc, rp)[0, 1]),
        "partial_pearson_block_given_pusher": partial_corr(cost, block, pusher),
        "partial_pearson_pusher_given_block": partial_corr(cost, pusher, block),
        "partial_spearman_block_given_pusher": partial_corr(rc, rb, rp),
        "partial_spearman_pusher_given_block": partial_corr(rc, rp, rb),
        "ols_slope_block_per_px": float(beta[1]),
        "ols_slope_pusher_per_px": float(beta[2]),
        "ols_std_beta_block": float(beta[1] * block.std() / cost.std()),
        "ols_std_beta_pusher": float(beta[2] * pusher.std() / cost.std()),
        "r2_full": r2_full,
        "r2_block_only": r2_b,
        "r2_pusher_only": r2_p,
        "unique_r2_block": r2_full - r2_p,
        "unique_r2_pusher": r2_full - r2_b,
        # LMG / Shapley share of R2 with two predictors
        "lmg_block": 0.5 * (r2_b + r2_full - r2_p),
        "lmg_pusher": 0.5 * (r2_p + r2_full - r2_b),
    }


def pairwise_stats(cost, block, pusher, official_sd, official_pd):
    """
    All unordered pairs of non-goal cells within each goal grid.
    cost etc. have shape (G, N). Returns ranking accuracies and matched-pair (Set D) inversion rates.
    """
    G, N = cost.shape
    iu, ju = np.triu_indices(N, k=1)
    agree = {"block": [0, 0], "official_state_dist": [0, 0], "official_pos_diff": [0, 0]}
    nb, npg = len(BLOCK_GAP_BINS) - 1, len(PUSHER_GAP_BINS) - 1
    inv_block = np.zeros((nb, npg))
    inv_official = np.zeros((nb, npg))
    official_vs_block = np.zeros((nb, npg))
    count = np.zeros((nb, npg))
    for g in range(G):
        dc = cost[g, iu] - cost[g, ju]
        for name, ref in (("block", block), ("official_state_dist", official_sd), ("official_pos_diff", official_pd)):
            dr = ref[g, iu] - ref[g, ju]
            keep = (np.abs(dr) > 1e-9) & (np.abs(dc) > 0)
            agree[name][0] += int((np.sign(dc[keep]) == np.sign(dr[keep])).sum())
            agree[name][1] += int(keep.sum())
        # Set D: orient each pair so that s1 has the lower block error; keep pairs where s1 has the higher pusher error
        db = block[g, iu] - block[g, ju]
        dp = pusher[g, iu] - pusher[g, ju]
        sign = -np.sign(db)  # +1 if cell iu is s1
        matched = (np.abs(db) > 1e-9) & (sign * dp > 1e-9)
        block_gap = np.abs(db[matched])
        pusher_gap = np.abs(dp[matched])
        cost_s1_minus_s2 = (sign * dc)[matched]
        off_s1_minus_s2 = (sign * (official_sd[g, iu] - official_sd[g, ju]))[matched]
        bi = np.digitize(block_gap, BLOCK_GAP_BINS) - 1
        pi = np.digitize(pusher_gap, PUSHER_GAP_BINS) - 1
        np.add.at(count, (bi, pi), 1)
        # latent cost prefers s2, the state with the worse block error
        np.add.at(inv_block, (bi, pi), cost_s1_minus_s2 > 0)
        # the official joint cost itself prefers s2
        np.add.at(official_vs_block, (bi, pi), off_s1_minus_s2 > 0)
        # latent cost disagrees with the official joint cost
        np.add.at(inv_official, (bi, pi), np.sign(cost_s1_minus_s2) != np.sign(off_s1_minus_s2))
    safe = np.maximum(count, 1)
    return {
        "ranking_accuracy_vs_block_only": agree["block"][0] / agree["block"][1],
        "ranking_accuracy_vs_official_state_dist": agree["official_state_dist"][0] / agree["official_state_dist"][1],
        "ranking_accuracy_vs_official_pos_diff": agree["official_pos_diff"][0] / agree["official_pos_diff"][1],
        "n_pairs_block": agree["block"][1],
        "matched_pairs": {
            "block_gap_bins_px": [str(b) for b in BLOCK_GAP_BINS],
            "pusher_gap_bins_px": [str(b) for b in PUSHER_GAP_BINS],
            "count": count.tolist(),
            "inversion_rate_vs_block_only": (inv_block / safe).tolist(),
            "inversion_rate_vs_official_state_dist": (inv_official / safe).tolist(),
            "official_prefers_worse_block_rate": (official_vs_block / safe).tolist(),
            "overall_inversion_rate_vs_block_only": float(inv_block.sum() / count.sum()),
            "overall_inversion_rate_vs_official_state_dist": float(inv_official.sum() / count.sum()),
            "overall_official_prefers_worse_block_rate": float(official_vs_block.sum() / count.sum()),
        },
    }


def retrieval(enc_files, alpha, b_levels, device):
    """
    Nearest-neighbour retrieval of block-pose identity inside each goal grid: the query is a cell (i, j),
    the gallery is every other cell; a hit means the nearest neighbour has the same block variant i.
    Also restricted to sub-grids {identity, variants at one level} x all pusher variants.
    """
    hits_block, hits_pusher, n = 0, 0, 0
    per_level = {f"{k[0]}_{k[1]:g}": [0, 0, 0] for k in b_levels if k[0] != "identity"}
    for f in enc_files:
        e = np.load(f)
        vis = torch.from_numpy(e["visual_emb"].astype(np.float32)).to(device)
        prop = torch.from_numpy(e["proprio_emb"]).to(device)
        NB, NP = vis.shape[:2]
        v = vis.reshape(NB * NP, -1)
        p = prop.reshape(NB * NP, -1)
        d = torch.cdist(v, v).pow(2) / v.shape[1] + alpha * torch.cdist(p, p).pow(2) / p.shape[1]
        d.fill_diagonal_(float("inf"))
        bi = torch.arange(NB, device=device).repeat_interleave(NP)
        pi = torch.arange(NP, device=device).repeat(NB)
        nn = d.argmin(1)
        hits_block += int((bi[nn] == bi).sum())
        hits_pusher += int((pi[nn] == pi).sum())
        n += NB * NP
        for k, idxs in b_levels.items():
            if k[0] == "identity":
                continue
            keep = torch.isin(bi, torch.tensor([0] + idxs, device=device))
            sub = d[keep][:, keep]
            nn_sub = sub.argmin(1)
            rec = per_level[f"{k[0]}_{k[1]:g}"]
            rec[0] += int((bi[keep][nn_sub] == bi[keep]).sum())
            rec[1] += int((pi[keep][nn_sub] == pi[keep]).sum())
            rec[2] += int(keep.sum())
    return {
        "block_identity_accuracy": hits_block / n,
        "nn_shares_pusher_rate": hits_pusher / n,
        "by_block_separation": {
            k: {"block_identity_accuracy": r[0] / r[2], "nn_shares_pusher_rate": r[1] / r[2]}
            for k, r in per_level.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets_dir", required=True)
    parser.add_argument("--enc_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta, costs, d_vis, d_prop, enc_files = load(args.sets_dir, args.enc_dir)
    G, NB, NP = d_vis.shape
    b_levels = variant_levels(meta["block_variants"])
    p_levels = variant_levels(meta["pusher_variants"])
    results = {"num_goals": G, "grid": [NB, NP], "max_reset_err": meta["max_reset_err"]}

    # ---- between-goal reference distances (unrelated block and pusher poses)
    between = np.load(os.path.join(args.enc_dir, "between_goals.npz"))
    off_diag = ~np.eye(G, dtype=bool)
    results["between_unrelated_goals"] = {
        "d_visual": med_iqr(between["d_visual"][off_diag]),
        "d_proprio": med_iqr(between["d_proprio"][off_diag]),
    }

    # ---- Set A: same block pose, varying pusher. Set B: same pusher, varying block pose.
    set_a = {}
    for (kind, mag), idxs in p_levels.items():
        if kind == "identity":
            continue
        set_a[f"pusher_{mag:g}px"] = {
            "d_visual": med_iqr(d_vis[:, 0, idxs]),
            "d_proprio": med_iqr(d_prop[:, 0, idxs]),
            "official_success_rate": float(costs["official_success"][:, 0, idxs].mean()),
        }
    set_b = {}
    for (kind, mag), idxs in b_levels.items():
        if kind == "identity":
            continue
        unit = "px" if kind == "translate" else "deg"
        set_b[f"block_{kind}_{mag:g}{unit}"] = {
            "d_visual": med_iqr(d_vis[:, idxs, 0]),
            "block_kp_err": med_iqr(costs["block_kp_err"][:, idxs, 0]),
            "official_success_rate": float(costs["official_success"][:, idxs, 0].mean()),
        }
    results["set_A_same_block_vary_pusher"] = set_a
    results["set_B_same_pusher_vary_block"] = set_b

    # ---- Equivalence violation rate: P[d(g, g+) > d(g, g-)], g+ in Set A, g- in Set B
    for alpha in ALPHAS:
        cost = d_vis + alpha * d_prop
        evr = {}
        for (pk, pm), pidx in p_levels.items():
            if pk == "identity":
                continue
            row = {}
            for (bk, bm), bidx in b_levels.items():
                if bk == "identity":
                    continue
                pos = cost[:, 0, pidx][:, :, None]  # (G, n_p, 1)
                neg = cost[:, bidx, 0][:, None, :]  # (G, 1, n_b)
                row[f"block_{bk}_{bm:g}"] = float((pos > neg).mean())
            # against the base goals of unrelated grids
            unrelated = between["d_visual"] + alpha * between["d_proprio"]
            viol = [
                (cost[g, 0, pidx][:, None] > unrelated[g, off_diag[g]][None, :]).mean() for g in range(G)
            ]
            row["unrelated_goal"] = float(np.mean(viol))
            evr[f"pusher_{pm:g}px"] = row
        results[f"equivalence_violation_rate_alpha{alpha:g}"] = evr

    # ---- Set C / Set D: pooled grid cells without the goal cell itself
    flat = lambda x: x.reshape(G, NB * NP)[:, 1:]
    block, pusher = flat(costs["block_kp_err"]), flat(costs["pusher_pos_err"])
    off_sd, off_pd = flat(costs["official_state_dist"]), flat(costs["official_pos_diff"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    have_emb = "visual_emb" in np.load(enc_files[0]).files
    for alpha in ALPHAS:
        cost = flat(d_vis + alpha * d_prop)
        res = {"regression_set_C": regression(cost.ravel(), block.ravel(), pusher.ravel())}
        res.update(pairwise_stats(cost, block, pusher, off_sd, off_pd))
        if have_emb:
            res["nn_retrieval"] = retrieval(enc_files, alpha, b_levels, device)
        results[f"alpha{alpha:g}"] = res
    # the explicit proprio term alone, and the physical costs themselves as references
    results["proprio_term_only"] = {"regression_set_C": regression(flat(d_prop).ravel(), block.ravel(), pusher.ravel())}
    results["official_state_dist_as_cost"] = pairwise_stats(off_sd, block, pusher, off_sd, off_pd)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
