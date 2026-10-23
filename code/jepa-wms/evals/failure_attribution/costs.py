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
    head:PATH  method C (2026-10-22): ||f(z) - f(g)||^2 with a learned head f (mw_head_train.py) on the patch
               tokens and proprio embedding; no proprio weight

The proprio term is kept as in the released cost (alpha * mean squared proprio difference).
"""

from typing import Union

import numpy as np
import torch
from tensordict import TensorDict


def parse_cost(spec):
    """'l2', 'topk:8', 'pmean:4' -> (mode, param)"""
    mode, _, val = spec.partition(":")
    assert mode in ("l2", "topk", "pmean", "head"), spec
    if mode == "head":
        return mode, val
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


class ReachThenChangeOracle:
    """Stage-1 oracle of the reach-then-change cost (2026-10-23): simulator states give the reach term and the gate.

    A Metaworld reward is written as "bring the hand to the object, then bring the object to its target". This cost
    imposes that order on any goal cost `inner` (the change term: the released latent cost, or the true object cost):
      - candidates that have moved the object (|obj - obj_start| > eps) are ranked by `inner`;
      - candidates that have not are ranked below all of them, by the hand-to-object distance (the reach term);
      - if the goal moves no object (|obj_goal - obj_start| <= eps, e.g. reach-wall), `inner` is used unchanged.
    States (B, 41) = [39-d obs, success, obj_to_target] come from the SimUnroll that produced the encodings
    (unroll.last_states): hand = obs[:3], object = obs[4:7]. The image-based method replaces both with maps
    computed from the frozen encoder; this oracle tells whether the structure itself solves the approach phase.
    """

    def __init__(self, inner, unroll, obj_start, obj_goal, eps: float = 0.005):
        self.inner, self.unroll, self.eps = inner, unroll, eps
        self.obj_start = np.asarray(obj_start, dtype=np.float64)
        self.active = bool(np.linalg.norm(np.asarray(obj_goal, dtype=np.float64) - self.obj_start) > eps)

    def __call__(self, encodings, actions, keepdims: bool = False):
        c = self.inner(encodings, actions, keepdims)
        if not self.active or keepdims:
            return c
        st = np.asarray(self.unroll.last_states, dtype=np.float64)
        moved = torch.from_numpy(np.linalg.norm(st[:, 4:7] - self.obj_start, axis=1) > self.eps).to(c.device)
        reach = torch.from_numpy(np.linalg.norm(st[:, :3] - st[:, 4:7], axis=1)).to(c.device, c.dtype)
        return torch.where(moved, c, c.max().detach() + reach)  # every non-mover ranks below every mover


class LearnedHeadMPCObjective:
    """Method C: squared distance between learned embeddings of the candidate and the goal (see mw_head_train.py)."""

    def __init__(self, cfg, target_enc: TensorDict, head_path: str, sum_all_diffs: bool = False, **kwargs):
        from evals.failure_attribution.mw_head_train import PatchHead

        ck = torch.load(head_path, map_location="cpu")
        self.head = PatchHead(ck["n_patches"], ck["d_in"], ck["d_prop"])
        self.head.load_state_dict(ck["state_dict"])
        self.head = self.head.to(target_enc["visual"].device).eval().requires_grad_(False)
        self.cfg, self.sum_all_diffs = cfg, sum_all_diffs
        self.f_goal = self.embed(target_enc)  # (T_goal, B_goal, d)

    @torch.no_grad()
    def embed(self, enc):
        """visual (*lead, V, H, W, D) + proprio (*lead, Dp) -> (*lead, d)"""
        vis, prop = enc["visual"], enc["proprio"]
        lead = vis.shape[:-4]
        n = int(np.prod(lead))
        f = self.head(vis.reshape(n, -1, vis.shape[-1]).float(), prop.reshape(n, -1).float())
        return f.view(*lead, -1)

    def __call__(self, encodings, actions, keepdims: bool = False):
        diff = (self.embed(encodings) - self.f_goal).pow(2).sum(-1)
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff.cumsum(0).flip(0) if self.sum_all_diffs else diff
