# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Encodes the candidate and goal images of collect.py with the frozen visual encoder of a
pretrained Metaworld world model.

Usage:
    python -m evals.mw_metric_conflict.encode --config <eval_config.yaml> --model_name jepa_wm_metaworld \
        --data_dir $JEPAWM_LOGS/mw_metric_conflict/data --out_dir $JEPAWM_LOGS/mw_metric_conflict/emb_jepa_wm
"""

import argparse
import glob
import os

import numpy as np
import torch

from hubconf import _load_model_with_config


@torch.no_grad()
def encode(model, images, batch_size):
    """images (N, H, W, 3) uint8 -> (N, h*w, D) float16"""
    out = []
    for s in range(0, len(images), batch_size):
        x = torch.from_numpy(images[s : s + batch_size]).permute(0, 3, 1, 2).unsqueeze(1)  # b t c h w
        out.append(model.encode(x).flatten(1, 4).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True, choices=["jepa_wm_metaworld", "dino_wm_metaworld"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, _ = _load_model_with_config(args.config, model_name=args.model_name, pretrained=False)
    model.eval()
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*.npz"))):
        d = np.load(f)
        np.savez(
            os.path.join(args.out_dir, os.path.basename(f)),
            cand=encode(model, d["cand_image"], args.batch_size),
            goal=encode(model, d["goal_image"], args.batch_size),
        )
        print(f"encoded {os.path.basename(f)}", flush=True)


if __name__ == "__main__":
    main()
