# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Builds the matched counterfactual Push-T state sets around a set of base goal states.

For each base goal g0 = (pusher p0, block pose b0) we render the full factorial grid
    block variants  B = {b0} + translations of b0 + rotations of b0
    pusher variants P = {p0} + displacements of p0
so that, relative to g0:
    Set A = grid[0, :]   same block pose, varying pusher
    Set B = grid[:, 0]   same pusher, varying block pose
    Set C = grid[:, :]   both varied independently
    Set D = pairs of cells of Set C trading block error against pusher error (built at analysis time)

Usage:
    python -m evals.pusht_goal_decomp.build_sets --out_dir $JEPAWM_LOGS/pusht_goal_decomp/sets --num_goals 64
"""

import argparse
import json
import os

import numpy as np

from evals.pusht_goal_decomp.harness import PushTStateHarness

BLOCK_TRANS_PX = [2, 5, 10, 20, 40, 80]
BLOCK_TRANS_DIRS = 2
BLOCK_ROT_DEG = [5, 10, 20, 45, 90]  # each applied with both signs
PUSHER_PX = [5, 10, 20, 40, 80, 160]
PUSHER_DIRS = 3

# requested vs realized block pose after env.prepare
TOL_BLOCK_POS = 1e-3  # px
TOL_BLOCK_ANG = 1e-4  # rad
MARGIN = 2.0  # px of clearance between pusher disc and block
MAX_TRIES = 300

COST_KEYS = [
    "official_success",
    "official_state_dist",
    "official_pos_diff",
    "block_pos_err",
    "block_ang_err",
    "block_kp_err",
    "pusher_pos_err",
    "pusher_vel_err",
]


def block_variant_specs():
    specs = [{"kind": "identity", "magnitude": 0.0}]
    for m in BLOCK_TRANS_PX:
        for d in range(BLOCK_TRANS_DIRS):
            specs.append({"kind": "translate", "magnitude": float(m), "dir_idx": d})
    for deg in BLOCK_ROT_DEG:
        for sign in (1, -1):
            specs.append({"kind": "rotate", "magnitude": float(deg), "sign": sign})
    return specs


def pusher_variant_specs():
    specs = [{"kind": "identity", "magnitude": 0.0}]
    for m in PUSHER_PX:
        for d in range(PUSHER_DIRS):
            specs.append({"kind": "translate", "magnitude": float(m), "dir_idx": d})
    return specs


def sample_base_goal(rs):
    """Same ranges as PushTEnv.reset, with a uniform block angle and zero pusher velocity."""
    return np.array(
        [
            rs.uniform(50, 450),
            rs.uniform(50, 450),
            rs.uniform(100, 400),
            rs.uniform(100, 400),
            rs.uniform(0, 2 * np.pi),
            0.0,
            0.0,
        ]
    )


def make_state(pusher_xy, block_pose):
    return np.array([pusher_xy[0], pusher_xy[1], block_pose[0], block_pose[1], block_pose[2] % (2 * np.pi), 0.0, 0.0])


def sample_goal_grid(harness, rs, b_specs, p_specs):
    """Returns (block_poses (NB, 3), pusher_xys (NP, 2)) or None if the base goal must be resampled."""
    g0 = sample_base_goal(rs)
    if not harness.is_valid(g0, MARGIN):
        return None
    p0, b0 = g0[:2], g0[2:5]

    block_poses = []
    for spec in b_specs:
        if spec["kind"] == "identity":
            block_poses.append(b0.copy())
        elif spec["kind"] == "rotate":
            b = b0.copy()
            b[2] = (b0[2] + spec["sign"] * np.deg2rad(spec["magnitude"])) % (2 * np.pi)
            if not harness.is_valid(make_state(p0, b), MARGIN):
                return None
            block_poses.append(b)
        else:
            for _ in range(MAX_TRIES):
                phi = rs.uniform(0, 2 * np.pi)
                b = b0.copy()
                b[:2] = b0[:2] + spec["magnitude"] * np.array([np.cos(phi), np.sin(phi)])
                if harness.is_valid(make_state(p0, b), MARGIN):
                    break
            else:
                return None
            block_poses.append(b)

    pusher_xys = []
    for spec in p_specs:
        if spec["kind"] == "identity":
            pusher_xys.append(p0.copy())
            continue
        # the pusher variant must be collision-free with every block variant, so the grid is complete
        for _ in range(MAX_TRIES):
            phi = rs.uniform(0, 2 * np.pi)
            p = p0 + spec["magnitude"] * np.array([np.cos(phi), np.sin(phi)])
            if all(harness.is_valid(make_state(p, b), MARGIN) for b in block_poses):
                break
        else:
            return None
        pusher_xys.append(p)
    return np.array(block_poses), np.array(pusher_xys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_goals", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    harness = PushTStateHarness(img_size=args.img_size)
    b_specs, p_specs = block_variant_specs(), pusher_variant_specs()
    NB, NP = len(b_specs), len(p_specs)
    rs = np.random.RandomState(args.seed)

    n_rejected_goals = 0
    max_err = {"block_pos": 0.0, "block_ang": 0.0, "pusher_pos": 0.0, "pusher_vel": 0.0}
    for g in range(args.num_goals):
        grid = None
        while grid is None:
            grid = sample_goal_grid(harness, rs, b_specs, p_specs)
            n_rejected_goals += grid is None
        block_poses, pusher_xys = grid
        env_seed = args.seed * 100003 + g

        requested = np.zeros((NB, NP, 7))
        realized = np.zeros((NB, NP, 7))
        visual = np.zeros((NB, NP, 3, args.img_size, args.img_size), dtype=np.uint8)
        proprio = np.zeros((NB, NP, 4), dtype=np.float32)
        reset_err = {k: np.zeros((NB, NP)) for k in max_err}
        for i in range(NB):
            for j in range(NP):
                state = make_state(pusher_xys[j], block_poses[i])
                out = harness.reset_to(state, seed=env_seed)
                err = harness.reset_error(state, out["realized"])
                assert (
                    err["block_pos"] < TOL_BLOCK_POS and err["block_ang"] < TOL_BLOCK_ANG
                ), f"goal {g} cell ({i},{j}): realized block pose differs from requested: {err}"
                requested[i, j] = state
                realized[i, j] = out["realized"]
                visual[i, j] = out["visual"][0].numpy()
                proprio[i, j] = out["proprio"][0].numpy()
                for k in err:
                    reset_err[k][i, j] = err[k]
                    max_err[k] = max(max_err[k], err[k])

        # costs of every cell w.r.t. the base goal g0 = cell (0, 0), on realized states
        costs = {k: np.zeros((NB, NP)) for k in COST_KEYS}
        for i in range(NB):
            for j in range(NP):
                dec = harness.decompose(realized[0, 0], realized[i, j])
                for k in COST_KEYS:
                    costs[k][i, j] = dec[k]

        np.savez_compressed(
            os.path.join(args.out_dir, f"goal_{g:03d}.npz"),
            requested=requested,
            realized=realized,
            visual=visual,
            proprio=proprio,
            env_seed=env_seed,
            **{f"reset_err_{k}": v for k, v in reset_err.items()},
            **{f"cost_{k}": v for k, v in costs.items()},
        )
        print(f"goal {g:03d}: g0={np.round(realized[0, 0], 2).tolist()} max reset err so far {max_err}", flush=True)

    meta = {
        "seed": args.seed,
        "num_goals": args.num_goals,
        "img_size": args.img_size,
        "margin_px": MARGIN,
        "tol_block_pos_px": TOL_BLOCK_POS,
        "tol_block_ang_rad": TOL_BLOCK_ANG,
        "block_variants": b_specs,
        "pusher_variants": p_specs,
        "cost_keys": COST_KEYS,
        "n_rejected_base_goals": int(n_rejected_goals),
        "max_reset_err": max_err,
        "state_layout": ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {args.num_goals} goals x {NB} block variants x {NP} pusher variants to {args.out_dir}")


if __name__ == "__main__":
    main()
