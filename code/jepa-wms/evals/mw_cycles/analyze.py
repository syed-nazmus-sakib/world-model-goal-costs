# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Ground-truth closure statistics of the action cycles of sim_cycles.py.

Usage:
    python -m evals.mw_cycles.analyze --dir $JEPAWM_LOGS/mw_cycles --out $JEPAWM_LOGS/mw_cycles/summary.json
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

# forward object displacement separating glancing from strong contact (m)
STRONG_DISP = 0.02
# a cycle counts as leaving the object displaced above this residual (m)
RESIDUAL_THR = 0.01


def load(dir_, task):
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(dir_, f"{task}_part*.npz")))]
    half = int(parts[0]["half"])
    cat = lambda k: np.concatenate([p[k] for p in parts])
    kinds = sum((json.loads(str(p["kinds"])) for p in parts), [])
    tcp, obj, mocap, force = cat("tcp"), cat("obj"), cat("mocap"), cat("force")
    d = {
        "kind": np.array(kinds),
        "base": cat("episode") * 1000 + cat("t_branch"),
        "hand_mid": np.linalg.norm(tcp[:, half] - tcp[:, 0], axis=1),
        "hand_closure": np.linalg.norm(tcp[:, -1] - tcp[:, 0], axis=1),
        "mocap_closure": np.linalg.norm(mocap[:, -1] - mocap[:, 0], axis=1),
        "obj_forward": np.linalg.norm(obj[:, half] - obj[:, 0], axis=1),
        "obj_residual": np.linalg.norm(obj[:, -1] - obj[:, 0], axis=1),
        "contact_steps_fwd": (force[:, :half] > 0).sum(1),
        "contact_steps_rev": (force[:, half:] > 0).sum(1),
        "peak_force_fwd": force[:, :half].max(1),
    }
    contact = (d["contact_steps_fwd"] + d["contact_steps_rev"]) > 0
    d["category"] = np.where(~contact, "no_contact", np.where(d["obj_forward"] >= STRONG_DISP, "strong", "glancing"))
    return d


def q(x):
    if len(x) == 0:
        return None
    return {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(np.max(x))}


def describe(d, sel):
    n = int(sel.sum())
    if n == 0:
        return {"n_cycles": 0}
    return {
        "n_cycles": n,
        "n_base_states": int(len(np.unique(d["base"][sel]))),
        "hand_mid_displacement": q(d["hand_mid"][sel]),
        "hand_closure": q(d["hand_closure"][sel]),
        "hand_closure_normalized": q(d["hand_closure"][sel] / np.maximum(d["hand_mid"][sel], 1e-6)),
        "mocap_target_closure": q(d["mocap_closure"][sel]),
        "obj_forward_displacement": q(d["obj_forward"][sel]),
        "obj_residual_displacement": q(d["obj_residual"][sel]),
        "frac_obj_residual_over_1cm": float((d["obj_residual"][sel] > RESIDUAL_THR).mean()),
        "contact_steps_forward": q(d["contact_steps_fwd"][sel]),
        "contact_steps_reverse": q(d["contact_steps_rev"][sel]),
        "peak_contact_force_forward_N": q(d["peak_force_fwd"][sel]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = {"strong_contact_min_forward_disp_m": STRONG_DISP}
    for task in ("reach", "push"):
        d = load(args.dir, task)
        res = {"all": describe(d, np.ones(len(d["kind"]), dtype=bool))}
        for c in ("no_contact", "glancing", "strong"):
            res[c] = describe(d, d["category"] == c)
        # hand closure when the commanded target itself returned (no clipping at the workspace limits)
        res["hand_closure_when_target_returns"] = q(d["hand_closure"][d["mocap_closure"] < 1e-6])
        res["frac_cycles_target_clipped"] = float((d["mocap_closure"] >= 1e-6).mean())
        # does the residual grow with the contact-induced forward displacement
        con = d["category"] != "no_contact"
        if con.sum() > 10:
            x, y = d["obj_forward"][con], d["obj_residual"][con]
            slope, intercept = np.polyfit(x, y, 1)
            res["residual_vs_forward_contact_cycles"] = {
                "n": int(con.sum()),
                "spearman": float(stats.spearmanr(x, y)[0]),
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "ols_slope": float(slope),
                "ols_intercept_m": float(intercept),
                "persistence_ratio_median": float(np.median(y[x > 0.005] / x[x > 0.005])),
                "by_forward_disp_bin_cm": {
                    f"{lo}-{hi}": {
                        "n": int(((x >= lo / 100) & (x < hi / 100)).sum()),
                        "residual_median_cm": float(100 * np.median(y[(x >= lo / 100) & (x < hi / 100)]))
                        if ((x >= lo / 100) & (x < hi / 100)).any()
                        else None,
                    }
                    for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 30))
                },
            }
        # systematic: per base state, is there a strong-contact cycle that leaves the object displaced
        bases = np.unique(d["base"])
        strong = d["category"] == "strong"
        res["frac_base_states_with_strong_cycle"] = float(np.mean([strong[d["base"] == b].any() for b in bases]))
        res["frac_base_states_with_persistent_strong_cycle"] = float(
            np.mean([(strong & (d["obj_residual"] > RESIDUAL_THR))[d["base"] == b].any() for b in bases])
        )
        # matched hand travel: contact vs no-contact cycles with similar midpoint hand displacement
        matched = {}
        for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.30)):
            inbin = (d["hand_mid"] >= lo) & (d["hand_mid"] < hi)
            matched[f"hand_mid_{int(100 * lo)}-{int(100 * hi)}cm"] = {
                c: {
                    "n": int((inbin & (d["category"] == c)).sum()),
                    "hand_closure_median_cm": float(100 * np.median(d["hand_closure"][inbin & (d["category"] == c)]))
                    if (inbin & (d["category"] == c)).any()
                    else None,
                    "obj_residual_median_cm": float(100 * np.median(d["obj_residual"][inbin & (d["category"] == c)]))
                    if (inbin & (d["category"] == c)).any()
                    else None,
                }
                for c in ("no_contact", "glancing", "strong")
            }
        res["matched_by_hand_travel"] = matched
        res["by_program_kind"] = {
            k: {
                "n": int((d["kind"] == k).sum()),
                "categories": {c: int(((d["kind"] == k) & (d["category"] == c)).sum()) for c in ("no_contact", "glancing", "strong")},
            }
            for k in sorted(set(d["kind"]))
        }
        summary[task] = res

        print(f"\n===== {task}: {len(d['kind'])} cycles, {len(bases)} base states; "
              f"commanded target clipped in {100 * res['frac_cycles_target_clipped']:.1f}% of cycles")
        print(f"{'category':11s} {'n':>5s} {'bases':>5s} {'hand mid':>9s} {'hand closure med/p90':>21s} {'norm.':>6s} "
              f"{'obj fwd med':>11s} {'obj resid med/p90':>18s} {'>1cm':>5s} {'contact fwd/rev':>15s} {'peak N':>7s}")
        for c in ("no_contact", "glancing", "strong"):
            r = res[c]
            if r["n_cycles"] == 0:
                print(f"{c:11s} {0:5d}")
                continue
            print(f"{c:11s} {r['n_cycles']:5d} {r['n_base_states']:5d} {100 * r['hand_mid_displacement']['median']:8.2f}  "
                  f"{100 * r['hand_closure']['median']:9.2f}/{100 * r['hand_closure']['p90']:6.2f}     "
                  f"{r['hand_closure_normalized']['median']:6.3f} {100 * r['obj_forward_displacement']['median']:10.2f}  "
                  f"{100 * r['obj_residual_displacement']['median']:8.2f}/{100 * r['obj_residual_displacement']['p90']:6.2f}   "
                  f"{r['frac_obj_residual_over_1cm']:5.2f} {r['contact_steps_forward']['median']:8.0f}/{r['contact_steps_reverse']['median']:3.0f}    "
                  f"{r['peak_contact_force_forward_N']['median']:7.1f}")
        print("  (distances in cm)")
        if "residual_vs_forward_contact_cycles" in res:
            rv = res["residual_vs_forward_contact_cycles"]
            print(f"  residual vs forward displacement (contact cycles, n={rv['n']}): spearman {rv['spearman']:.2f}, "
                  f"slope {rv['ols_slope']:.2f}, persistence ratio median {rv['persistence_ratio_median']:.2f}")
            print("  residual median by forward-displacement bin (cm):",
                  {k: (v["n"], None if v["residual_median_cm"] is None else round(v["residual_median_cm"], 2)) for k, v in rv["by_forward_disp_bin_cm"].items()})
        print(f"  base states with a strong-contact cycle: {100 * res['frac_base_states_with_strong_cycle']:.0f}%, "
              f"with a strong cycle leaving >1 cm residual: {100 * res['frac_base_states_with_persistent_strong_cycle']:.0f}%")
        print("  matched by hand travel:", json.dumps(matched))
        print("  program kinds:", json.dumps(res["by_program_kind"]))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
