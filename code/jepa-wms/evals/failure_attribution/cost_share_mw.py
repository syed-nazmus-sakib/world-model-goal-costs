# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
How much of the goal cost is the robot? (mechanism check across tasks, 2026-10-17)

For each eval episode: the start image o_0, the eval goal image g, and both rendered again with the robot hidden
(simulator oracle). Two exact measures of the released patch cost C(o_0, g) = sum_p ||z_p(o_0) - z_p(g)||^2:
  share_robot_patches   fraction of C carried by patches that the robot occupies in o_0 or in g (pixel mask from the
                        robot-hidden render, max-pooled to the 16x16 patch grid)
  share_removed_by_mask 1 - C(o_0^rf, g^rf) / C(o_0, g): how much of the start-to-goal latent distance disappears when
                        the robot is removed from both images (what is left is the scene change the task asks for)
plus the fraction of patches the robot covers. 'reach-wall' should be near 1 on both shares (the arm is the task);
object tasks reveal how little of the cost is the object.

    python -m evals.failure_attribution.cost_share_mw --fname <mw config> --out <json> --episodes 12
"""

import argparse
import json
import os

import numpy as np
import torch

from evals.failure_attribution.goal_variants import goal_td, hook_expert_capture, make_arm_variants, robot_free_frame
from evals.failure_attribution.run import build, load_args


def patch_mask(frame, rf_frame, patch=14, thresh=12):
    """(H,W,3) uint8 pair -> (h,w) bool: patches where the two renders differ (the robot was there)."""
    d = np.abs(frame.astype(np.int16) - rf_frame.astype(np.int16)).max(-1) > thresh
    H, W = d.shape
    return d.reshape(H // patch, patch, W // patch, patch).max(axis=(1, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    args = parser.parse_args()
    from evals.utils import prepare_obs

    args_eval = load_args(args.fname)
    task = args_eval["task_specification"]["task"]
    cfg, agent, env, evaluator = build(args_eval)
    hook_expert_capture(evaluator)
    model, device = agent.model, agent.device
    obs_type = cfg.task_specification.obs

    def enc(frame, obs):
        with torch.no_grad():
            z = model.encode(prepare_obs(obs_type, goal_td(frame, obs)).unsqueeze(0).to(device), act=False)["visual"]
        return z[0, -1, 0].float().cpu().numpy()  # (h, w, D)

    rows = []
    for ep in range(args.episodes):
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        env.reset(seed=ep_seed, task_idx=-1)
        u = env.proprio_env.unwrapped
        u._freeze_rand_vec, u.seeded_rand_vec = False, True
        env.seed(ep_seed)
        init_obs, goal_obs, _, _ = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=-1)
        xenv = evaluator.expert_env
        var = make_arm_variants(xenv, evaluator.expert_init_snap, evaluator.expert_final_snap, 0, seed=ep)
        g_frame, g_obs = var[0][1], var[0][2]
        g_rf = robot_free_frame(xenv, var[0][0])
        s_frame = np.asarray(init_obs["visual"][-1]).transpose(1, 2, 0).copy()  # last context frame (H, W, 3)
        s_obs = np.asarray(init_obs["proprio"][-1], dtype=np.float32)
        s_rf = robot_free_frame(xenv, evaluator.expert_init_snap)
        try:
            xenv.close()
        except Exception:
            pass
        evaluator.expert_env = None
        if s_rf.shape != s_frame.shape:  # different render size: resize the hidden render to the observation
            import cv2

            s_rf = cv2.resize(s_rf, (s_frame.shape[1], s_frame.shape[0]), interpolation=cv2.INTER_AREA)
            g_rf = cv2.resize(g_rf, (g_frame.shape[1], g_frame.shape[0]), interpolation=cv2.INTER_AREA)
        if ep == 0 and os.environ.get("COST_SHARE_DUMP"):  # visual check of the four renders
            import cv2

            cv2.imwrite(os.environ["COST_SHARE_DUMP"], np.concatenate([s_frame, s_rf, g_frame, g_rf], 1)[..., ::-1])
        zs, zg, zs_rf, zg_rf = enc(s_frame, s_obs), enc(g_frame, g_obs), enc(s_rf, s_obs), enc(g_rf, g_obs)
        h = zs.shape[0]
        ps = s_frame.shape[0] // h
        m = patch_mask(s_frame, s_rf, ps) | patch_mask(g_frame, g_rf, ps)
        per_patch = ((zs - zg) ** 2).sum(-1)  # (h, w)
        c_full, c_rf = float(per_patch.sum()), float(((zs_rf - zg_rf) ** 2).sum())
        rows.append({"episode": ep, "robot_patch_frac": float(m.mean()),
                     "share_robot_patches": float(per_patch[m].sum() / max(per_patch.sum(), 1e-12)),
                     "share_removed_by_mask": float(1.0 - c_rf / max(c_full, 1e-12)),
                     "cost_full": c_full, "cost_robot_free": c_rf})
        print(f"ep {ep:02d}: robot covers {100 * m.mean():.0f}% of patches, carries {100 * rows[-1]['share_robot_patches']:.0f}% of the cost; "
              f"removing it removes {100 * rows[-1]['share_removed_by_mask']:.0f}% of the start-goal distance", flush=True)
    summ = {k: [round(float(np.mean([r[k] for r in rows])), 3), round(float(np.median([r[k] for r in rows])), 3)]
            for k in ("robot_patch_frac", "share_robot_patches", "share_removed_by_mask")}
    out = {"task": task, "episodes": len(rows), "mean_median": summ, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"task": task, **summ}))


if __name__ == "__main__":
    main()
