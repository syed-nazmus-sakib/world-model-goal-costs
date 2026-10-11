# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
How much of the visual latent goal distance of encoded Push-T states is explained by a
three-variable surrogate
    d_visual ~ b0 + b_block * e_block_pos + b_angle * e_block_ang + b_pusher * e_pusher_pos
and what is left of the block-only ranking errors once the pusher term is removed.

The surrogate is fit on one set of goals and evaluated on a held-out set.

Usage:
    python -m evals.pusht_goal_decomp.surrogate --fit_sets <sets> --fit_enc <enc> \
        --test_sets <sets_seed1> --test_enc <enc_seed1> --out <surrogate.json>
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

KEYS = ["block_pos_err", "block_ang_err", "pusher_pos_err", "block_kp_err"]


def load(sets_dir, enc_dir):
    meta = json.load(open(os.path.join(sets_dir, "meta.json")))
    x = {k: [] for k in KEYS}
    d_vis = []
    for f in sorted(glob.glob(os.path.join(sets_dir, "goal_*.npz"))):
        s = np.load(f)
        for k in KEYS:
            x[k].append(s[f"cost_{k}"])
        d_vis.append(np.load(os.path.join(enc_dir, os.path.basename(f)))["d_visual"])
    x = {k: np.stack(v).astype(np.float64) for k, v in x.items()}
    x["block_ang_err"] = np.degrees(x["block_ang_err"])
    return meta, x, np.stack(d_vis).astype(np.float64)


def flat(a):
    """(G, NB, NP) -> (G, NB * NP - 1), dropping the goal cell."""
    return a.reshape(a.shape[0], -1)[:, 1:]


def design(x, transform=None):
    cols = [flat(x["block_pos_err"]), flat(x["block_ang_err"]), flat(x["pusher_pos_err"])]
    if transform is not None:
        cols = [transform(c) for c in cols]
    return cols


def fit(cols, y):
    X = np.column_stack([np.ones(y.size)] + [c.ravel() for c in cols])
    beta, *_ = np.linalg.lstsq(X, y.ravel(), rcond=None)
    return beta


def predict(cols, beta):
    return beta[0] + sum(b * c for b, c in zip(beta[1:], cols))


def r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pair_agreement(a, b):
    """Fraction of within-goal pairs ranked in the same order by a and b, shape (G, N)."""
    iu, ju = np.triu_indices(a.shape[1], k=1)
    da, db = a[:, iu] - a[:, ju], b[:, iu] - b[:, ju]
    keep = (np.abs(da) > 1e-12) & (np.abs(db) > 1e-12)
    return float((np.sign(da[keep]) == np.sign(db[keep])).mean())


def matched_inversion(cost, block, pusher):
    """Pairs where s1 has the lower block error and the higher pusher error; inversion = cost prefers s2."""
    iu, ju = np.triu_indices(cost.shape[1], k=1)
    db, dp, dc = block[:, iu] - block[:, ju], pusher[:, iu] - pusher[:, ju], cost[:, iu] - cost[:, ju]
    sign = -np.sign(db)
    matched = (np.abs(db) > 1e-9) & (sign * dp > 1e-9)
    return float(((sign * dc)[matched] > 0).mean()), int(matched.sum())


def additive_levels_fit(meta, d_vis_fit, d_vis_test):
    """
    Additive but otherwise unconstrained model d = mu + f(block variant level) + g(pusher level),
    with one free value per (kind, magnitude) level. What it cannot explain is block-pusher interaction
    plus goal-to-goal and direction-to-direction variation.
    """
    b_lvl = [f"{s['kind']}_{s['magnitude']}" for s in meta["block_variants"]]
    p_lvl = [f"{s['kind']}_{s['magnitude']}" for s in meta["pusher_variants"]]
    bu, pu = sorted(set(b_lvl)), sorted(set(p_lvl))
    bi = np.array([bu.index(v) for v in b_lvl])
    pi = np.array([pu.index(v) for v in p_lvl])
    NB, NP = len(b_lvl), len(p_lvl)
    X = np.zeros((NB, NP, len(bu) + len(pu)))
    for i in range(NB):
        for j in range(NP):
            X[i, j, bi[i]] = 1
            X[i, j, len(bu) + pi[j]] = 1
    Xf = np.tile(X.reshape(1, NB * NP, -1), (d_vis_fit.shape[0], 1, 1)).reshape(-1, X.shape[-1])
    beta, *_ = np.linalg.lstsq(Xf, d_vis_fit.reshape(-1), rcond=None)
    pred = (X.reshape(NB * NP, -1) @ beta).reshape(1, NB, NP)
    return np.broadcast_to(pred, d_vis_test.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_sets", required=True)
    parser.add_argument("--fit_enc", required=True)
    parser.add_argument("--test_sets", required=True)
    parser.add_argument("--test_enc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta, xf, df = load(args.fit_sets, args.fit_enc)
    _, xt, dt = load(args.test_sets, args.test_enc)
    y_fit, y = flat(df), flat(dt)
    block_kp, pusher = flat(xt["block_kp_err"]), flat(xt["pusher_pos_err"])
    res = {"n_fit_goals": int(df.shape[0]), "n_test_goals": int(dt.shape[0])}

    # ---- three-variable surrogates, fit on the fit goals, evaluated on held-out goals
    for name, tr in (("linear", None), ("sqrt", np.sqrt)):
        beta = fit(design(xf, tr), y_fit)
        pred = predict(design(xt, tr), beta)
        res[f"surrogate_{name}"] = {
            "intercept": float(beta[0]),
            "beta_block_pos_per_px": float(beta[1]),
            "beta_block_ang_per_deg": float(beta[2]),
            "beta_pusher_per_px": float(beta[3]),
            "pusher_to_block_ratio": float(beta[3] / beta[1]),
            "deg_of_rotation_equal_to_1px_block": float(beta[1] / beta[2]),
            "heldout_r2": r2(y.ravel(), pred.ravel()),
            "heldout_spearman": float(stats.spearmanr(y.ravel(), pred.ravel())[0]),
            "heldout_pair_agreement_with_latent": pair_agreement(y, pred),
        }
    pred_add = flat(additive_levels_fit(meta, df, dt))
    res["surrogate_additive_levels"] = {
        "heldout_r2": r2(y.ravel(), pred_add.ravel()),
        "heldout_spearman": float(stats.spearmanr(y.ravel(), pred_add.ravel())[0]),
        "heldout_pair_agreement_with_latent": pair_agreement(y, pred_add),
    }

    # ---- block-only ranking errors before / after removing the fitted pusher term
    beta = fit(design(xf), y_fit)
    corrected = y - beta[3] * pusher
    oracle_block = predict(design(xt), np.array([0.0, beta[1], beta[2], 0.0]))
    rows = {
        "latent": y,
        "latent_minus_linear_pusher_term": corrected,
        "surrogate_block_terms_only": oracle_block,
    }
    res["block_only_ranking"] = {}
    for k, c in rows.items():
        inv, n = matched_inversion(c, block_kp, pusher)
        res["block_only_ranking"][k] = {
            "pair_accuracy_vs_block_kp_err": pair_agreement(c, block_kp),
            "matched_pair_inversion_rate": inv,
            "n_matched_pairs": n,
        }

    # ---- pure block ranking at a fixed pusher: Set B (pusher identical to the goal's)
    setb = dt[:, 1:, 0]
    setb_kp = xt["block_kp_err"][:, 1:, 0]
    res["set_B_pair_accuracy_vs_block_kp_err"] = pair_agreement(setb, setb_kp)
    # ---- and across every column: same pusher in both states, any pusher offset from the goal
    acc = [pair_agreement(dt[:, :, j], xt["block_kp_err"][:, :, j]) for j in range(1, dt.shape[2])]
    res["same_pusher_pair_accuracy_vs_block_kp_err_by_pusher_variant"] = {
        f"{meta['pusher_variants'][j]['magnitude']:g}px_dir{meta['pusher_variants'][j]['dir_idx']}": a
        for j, a in zip(range(1, dt.shape[2]), acc)
    }

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
