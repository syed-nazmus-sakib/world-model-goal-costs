# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Encodes the counterfactual state sets of build_sets.py with a pretrained world model and
logs the two terms of the planning cost w.r.t. the base goal of each grid:
    d_visual  : visual latent goal distance (the alpha=0 planning cost)
    d_proprio : explicit proprio term, planning cost = d_visual + alpha * d_proprio

Usage:
    python -m evals.pusht_goal_decomp.encode --config <eval_config.yaml> --model_name jepa_wm_pusht \
        --sets_dir $JEPAWM_LOGS/pusht_goal_decomp/sets --out_dir $JEPAWM_LOGS/pusht_goal_decomp/jepa_wm
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from tensordict import TensorDict

from evals.pusht_goal_decomp.harness import latent_goal_costs
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from hubconf import _load_model_with_config


@torch.no_grad()
def encode_batched(model, visual, proprio, batch_size):
    """visual (N, 3, H, W) uint8, proprio (N, 4) -> visual embs (N, h*w, D), proprio embs (N, P)"""
    vis_embs, prop_embs = [], []
    for s in range(0, len(visual), batch_size):
        td = TensorDict(
            {
                "visual": torch.from_numpy(visual[s : s + batch_size]).unsqueeze(1),
                "proprio": torch.from_numpy(proprio[s : s + batch_size]).unsqueeze(1),
            },
            batch_size=[len(visual[s : s + batch_size])],
        )
        enc = model.encode(td)
        vis_embs.append(enc["visual"].flatten(1, 4).float().cpu())  # b t v h w d -> b (h w) d
        # the proprio embedding is repeated over the patch grid, keep one copy
        prop = enc["proprio"]
        assert torch.allclose(prop[:, :, :1], prop[:, :, -1:]), "proprio embedding is not constant over patches"
        prop_embs.append(prop[:, 0, 0].float().cpu())
    return torch.cat(vis_embs), torch.cat(prop_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True, choices=["jepa_wm_pusht", "dino_wm_pusht"])
    parser.add_argument("--sets_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_embeddings", type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, _ = _load_model_with_config(args.config, model_name=args.model_name, pretrained=False)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.sets_dir, "goal_*.npz")))
    goal_vis, goal_prop = [], []
    max_objective_gap = 0.0
    for f in files:
        data = np.load(f)
        NB, NP = data["visual"].shape[:2]
        visual = data["visual"].reshape(NB * NP, *data["visual"].shape[2:])
        proprio = data["proprio"].reshape(NB * NP, -1)
        vis, prop = encode_batched(model, visual, proprio, args.batch_size)

        # cell (0, 0) is the base goal
        d_vis = (vis - vis[:1]).pow(2).flatten(1).mean(1)
        d_prop = (prop - prop[:1]).pow(2).flatten(1).mean(1)

        # check against the planning objective of the repo, in its (T, B, ...) layout
        enc = TensorDict({"visual": vis[None], "proprio": prop[None]}, batch_size=[1, len(vis)])
        goal_enc = TensorDict({"visual": vis[:1][None], "proprio": prop[:1][None]}, batch_size=[1, 1])
        for alpha in (0.0, 0.1):
            ref = ReprTargetDistMPCObjective(None, target_enc=goal_enc, alpha=alpha)(enc, None)
            max_objective_gap = max(max_objective_gap, (ref - (d_vis + alpha * d_prop)).abs().max().item())
        ref_vis, ref_prop = latent_goal_costs(
            TensorDict({"visual": vis, "proprio": prop}), TensorDict({"visual": vis[:1], "proprio": prop[:1]})
        )
        assert torch.allclose(ref_vis, d_vis) and torch.allclose(ref_prop, d_prop)

        out = {
            "d_visual": d_vis.reshape(NB, NP).numpy(),
            "d_proprio": d_prop.reshape(NB, NP).numpy(),
            "proprio_emb": prop.reshape(NB, NP, -1).numpy(),
        }
        if args.save_embeddings:
            out["visual_emb"] = vis.reshape(NB, NP, *vis.shape[1:]).numpy().astype(np.float16)
        np.savez(os.path.join(args.out_dir, os.path.basename(f)), **out)
        goal_vis.append(vis[0])
        goal_prop.append(prop[0])
        print(f"{os.path.basename(f)}: d_visual max {d_vis.max():.4f}, d_proprio max {d_prop.max():.4f}", flush=True)

    # distances between the base goals of different grids (unrelated block and pusher poses)
    goal_vis, goal_prop = torch.stack(goal_vis), torch.stack(goal_prop)
    between_vis = (goal_vis[:, None] - goal_vis[None]).pow(2).flatten(2).mean(2)
    between_prop = (goal_prop[:, None] - goal_prop[None]).pow(2).flatten(2).mean(2)
    np.savez(
        os.path.join(args.out_dir, "between_goals.npz"),
        d_visual=between_vis.numpy(),
        d_proprio=between_prop.numpy(),
    )
    with open(os.path.join(args.out_dir, "encode_meta.json"), "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "config": os.path.abspath(args.config),
                "sets_dir": os.path.abspath(args.sets_dir),
                "max_abs_gap_vs_repo_objective": max_objective_gap,
                "visual_emb_shape": list(goal_vis.shape[1:]),
                "proprio_emb_dim": int(goal_prop.shape[-1]),
            },
            f,
            indent=2,
        )
    print(f"max |cost - ReprTargetDistMPCObjective| = {max_objective_gap:.3e}")


if __name__ == "__main__":
    main()
