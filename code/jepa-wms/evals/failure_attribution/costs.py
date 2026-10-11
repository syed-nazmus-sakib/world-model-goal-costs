# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Area-invariant latent goal costs.

The released cost (ReprTargetDistMPCObjective) averages squared feature differences over all patches, so a
region of the image contributes in proportion to its area and small task-critical entities (the Push-T
pusher, a cube, a gripper) weigh little. Here the squared difference is averaged over features only, giving
a per-patch mismatch, which is then aggregated over patches without scaling with area:

    topk:K     mean of the K largest patch mismatches
    pmean:P    power mean (mean_p d_p^P)^(1/P)
    l2         the released aggregation (mean over patches), for reference

The proprio term is kept as in the released cost (alpha * mean squared proprio difference).
"""

from typing import Union

import torch
from tensordict import TensorDict


def parse_cost(spec):
    """'l2', 'topk:8', 'pmean:4' -> (mode, param)"""
    mode, _, val = spec.partition(":")
    assert mode in ("l2", "topk", "pmean"), spec
    return mode, (float(val) if val else None)


def patch_mismatch(target, enc):
    """(T, B, *patch_dims, D) -> (T, B, n_patches) mean squared feature difference per patch."""
    d = (target - enc).pow(2).mean(-1)
    return d.flatten(2)


def aggregate(d, mode, param):
    if mode == "l2":
        return d.mean(-1)
    if mode == "topk":
        k = min(int(param), d.shape[-1])
        return d.topk(k, dim=-1).values.mean(-1)
    if mode == "pmean":
        return d.pow(param).mean(-1).pow(1.0 / param)
    raise ValueError(mode)


class PatchAggMPCObjective:
    def __init__(self, cfg, target_enc: Union[torch.Tensor, TensorDict], cost: str = "topk:8", alpha: float = 0.1,
                 sum_all_diffs: bool = False, **kwargs):
        self.cfg, self.target_enc, self.alpha, self.sum_all_diffs = cfg, target_enc, alpha, sum_all_diffs
        self.mode, self.param = parse_cost(cost)

    def __call__(self, encodings, actions, keepdims: bool = False):
        if isinstance(encodings, TensorDict):
            diff = aggregate(patch_mismatch(self.target_enc["visual"], encodings["visual"]), self.mode, self.param)
            diff_p = (self.target_enc["proprio"] - encodings["proprio"]).pow(2)
            diff = diff + self.alpha * diff_p.mean(dim=tuple(range(2, diff_p.ndim)))
        else:
            diff = aggregate(patch_mismatch(self.target_enc, encodings), self.mode, self.param)
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff.cumsum(0).flip(0) if self.sum_all_diffs else diff
