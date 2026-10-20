# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
World-model goal set: the goal latent is averaged with the latents the world model predicts after K sustained,
random free-motion action sequences started from the goal (one random xyz direction per sequence, gripper neutral).
Squared distance to the mean latent equals the mean squared distance to the set (up to a constant), so planning
cost is unchanged. Optional reversibility gate: keep the fraction `gate_q` of rollouts whose reversed motion
returns closest to the goal latent under the model (a motion that changed the task outcome does not come back).
"""

import torch
from tensordict import TensorDict


@torch.no_grad()
def build_goalset_target(agent, g, K=16, H=8, mag=1.0, gate_q=0.0, seed=0):
    """g: goal encoding TensorDict (visual [1,1,1,h,w,D], proprio [1,1,P,D]). Returns (target TensorDict, cycle errors)."""
    model, device, cfg = agent.model, agent.device, agent.cfg
    fs = cfg.frameskip
    cgen = torch.Generator().manual_seed(seed)

    def norm_seq(raw):  # raw (K, H*fs, 4) -> normalized (H, K, fs*4)
        n = torch.stack([agent.preprocessor.normalize_actions(raw[k]) for k in range(raw.shape[0])])
        return n.reshape(raw.shape[0], H, -1).permute(1, 0, 2).to(device).float()

    d = torch.randn(K, 3, generator=cgen)
    d = d / d.norm(dim=1, keepdim=True)
    raw = torch.zeros(K, H * fs, 4)
    raw[..., :3] = mag * d[:, None, :]
    out = model.unroll(g, act_suffix=norm_seq(raw))
    fin = {k: out[k][-1].unsqueeze(1) for k in ("visual", "proprio")}  # (K, 1, ...)
    z2 = TensorDict({"visual": fin["visual"], "proprio": fin["proprio"]}, batch_size=[K])
    out2 = model.unroll(z2, act_suffix=norm_seq(-raw.flip(1)))
    err = (out2["visual"][-1] - g["visual"][0]).pow(2).flatten(1).mean(1).cpu()
    keep = torch.arange(K)
    if gate_q > 0:
        keep = torch.argsort(err)[: max(1, int(round(gate_q * K)))]
    tgt = g.clone()
    for k in ("visual", "proprio"):
        tgt[k] = torch.cat([g[k], fin[k][keep.to(device)]], 0).mean(0, keepdim=True)
    return tgt, err


@torch.no_grad()
def free_motion_finals(agent, z, K=32, H=16, mag=0.7, seed=0):
    """Final visual latents (K, 1, h, w, D) after K sustained random free-motion rollouts from context z (a TensorDict)."""
    model, device, cfg = agent.model, agent.device, agent.cfg
    fs = cfg.frameskip
    cgen = torch.Generator().manual_seed(seed)
    d = torch.randn(K, 3, generator=cgen)
    d = d / d.norm(dim=1, keepdim=True)
    raw = torch.zeros(K, H * fs, 4)
    raw[..., :3] = mag * d[:, None, :]
    n = torch.stack([agent.preprocessor.normalize_actions(raw[k]) for k in range(K)])
    A = n.reshape(K, H, -1).permute(1, 0, 2).to(device).float()
    return model.unroll(z, act_suffix=A)["visual"][-1]


@torch.no_grad()
def nuisance_projection(fin, gv, z0):
    """Change-conditioned nuisance subspace (method candidate 1, 2026-10-17).

    fin: (K, P) predicted latents after free-motion rollouts from the goal; gv: (P,) goal latent; z0: (P,) start latent.
    U = fin - g spans "the arm can move here" directions at the goal. d = g - z0 is the start-to-goal change: what the task
    asks for. Returns orthonormal bases (rows) and the explainability rho = ||proj_U d||^2 / ||d||^2:
      Q_full   span(U)                       (pure goal-set subspace; ignores the task; expected to break reach tasks)
      Q_dperp  span(U - <U, d_hat> d_hat)    (nuisance directions with the task direction removed: never discards progress along d)
    """
    d = gv - z0
    dn = d / d.norm().clamp_min(1e-12)
    U = fin - gv[None]
    Q_full = _row_space(U)  # (r, P), orthonormal rows, r = numerical rank
    rho = float((Q_full @ d).pow(2).sum() / d.pow(2).sum().clamp_min(1e-12))
    Up = U - (U @ dn)[:, None] * dn[None]
    Q_dperp = _row_space(Up)
    # invariant: the protected direction d is never in the removed subspace (audit 2026-10-18 found that the unthresholded
    # SVD returned null-space rows for rank-deficient U, which can contain d)
    assert Q_dperp.shape[0] == 0 or float((Q_dperp @ dn).abs().max()) < 1e-3, "Q_dperp not orthogonal to d"
    return Q_full, Q_dperp, rho


def _row_space(U, rel_tol=1e-4):
    """Orthonormal basis (rows) of the row space of U, keeping only singular vectors whose singular value is above
    rel_tol * s_max. The unthresholded torch.linalg.svd(..., full_matrices=False).Vh returns min(K, P) rows regardless of
    rank; for a rank-deficient U the extra rows are arbitrary null-space directions and must not be projected out."""
    s, Vh = torch.linalg.svd(U, full_matrices=False)[1:]
    if s.numel() == 0 or float(s.max()) <= 0:
        return Vh[:0]
    keep = s > rel_tol * s.max()
    return Vh[keep]


@torch.no_grad()
def projected_costs(X, gv, Q):
    """Mean squared residual to the goal after removing span(Q): X (M, P), gv (P,), Q (r, P) orthonormal rows -> (M,).
    Same scale as the released visual cost (mean over patches and channels of the squared difference)."""
    r = X - gv[None]
    c = r @ Q.T
    return ((r * r).sum(1) - (c * c).sum(1)).clamp_min(0) / X.shape[1]


@torch.no_grad()
def projection_bases(agent, z_start, g, K=32, H=16, mag=0.7, seed=0):
    """Free-motion rollouts from the goal under the world model, then nuisance_projection. Returns (Q_full, Q_dperp, rho)."""
    fin = free_motion_finals(agent, g, K, H, mag, seed).flatten(1)
    gv = g["visual"].flatten()
    z0 = z_start["visual"][:, -1].flatten()
    return nuisance_projection(fin, gv, z0)


@torch.no_grad()
def explain_ratio(agent, z_start, g, K=32, H=16, mag=0.7, seed=0):
    """How well the goal latent is explained by arm-only motion from the start:
    min_k ||z_goal - z_start_k||^2 / ||z_goal - z_start||^2 (small: the arm alone reaches the goal; about 1 or more: something else changed)."""
    fin = free_motion_finals(agent, z_start, K, H, mag, seed).flatten(1)
    gv = g["visual"].flatten()
    z0 = z_start["visual"][:, -1].flatten()
    dg = (gv - z0).pow(2).mean()
    dm = (gv[None] - fin).pow(2).mean(1).min()
    return float(dm / dg.clamp_min(1e-12)), float(dg)
