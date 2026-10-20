# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Swap the frozen visual encoder of a released planner for another pretrained backbone (offline tests only).

`install_alt_encoder(agent, name)` replaces `agent.model.encode` so that the "visual" entry comes from the chosen
backbone's patch tokens (B, T, 1, h, w, D) while the "proprio" entry still uses the released proprio encoder. The
released predictor cannot be used with the new features, so this is only valid for exact-simulator protocols
(goal_equiv_mw.py --no_cem, run_closed.py sim_latent) where the encoder is the only learned component in the cost.

Backbones (all downloaded on first use):
    dinov2      the released DINOv2 ViT-S/14 (no change; for reference)
    dinov2_b    DINOv2 ViT-B/14 (torch.hub facebookresearch/dinov2), 16x16 patches at 224, 768-d
    siglip2     SigLIP 2 ViT-B/16 (timm vit_base_patch16_siglip_224.v2_webli), language-supervised, 14x14, 768-d
    clip        CLIP ViT-B/16 (timm vit_base_patch16_clip_224.openai), 14x14, 768-d
    vjepa2      V-JEPA 2 ViT-L/16 (torch.hub facebookresearch/vjepa2), video JEPA, frame duplicated to a 2-frame clip,
                256x256 -> 16x16, 1024-d
    dinov3      DINOv3 ViT-S/16 (torch.hub facebookresearch/dinov3); gated weights: set DINOV3_WEIGHTS to the URL/path
"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _drop_patch_kwargs(f):
    """Adapt a V-JEPA 2 block forward(x, mask, attn_mask, T, H, W) to callers that pass H_patches/W_patches."""

    def g(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, **kw):
        return f(x, mask=mask, attn_mask=attn_mask, T=T, H=H_patches, W=W_patches, **kw)

    return g


class AltBackbone:
    def __init__(self, name, device):
        self.name, self.device = name, device
        self.mean, self.std = IMAGENET
        self.res, self.video = 224, False
        if name == "dinov2_b":
            self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
            self.fwd = lambda x: self.net.forward_features(x)["x_norm_patchtokens"]
        elif name == "dinov3":
            w = os.environ.get("DINOV3_WEIGHTS")
            assert w, "set DINOV3_WEIGHTS (gated release) to use dinov3"
            self.net = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", weights=w).to(device).eval()
            self.fwd = lambda x: self.net.forward_features(x)["x_norm_patchtokens"]
        elif name in ("siglip2", "clip"):
            import timm

            tag = {"siglip2": "vit_base_patch16_siglip_224.v2_webli", "clip": "vit_base_patch16_clip_224.openai"}[name]
            self.net = timm.create_model(tag, pretrained=True, num_classes=0).to(device).eval()
            cfg = timm.data.resolve_data_config({}, model=self.net)
            self.mean, self.std, self.res = tuple(cfg["mean"]), tuple(cfg["std"]), cfg["input_size"][-1]
            npre = getattr(self.net, "num_prefix_tokens", 0)
            self.fwd = lambda x: self.net.forward_features(x)[:, npre:]
        elif name == "vjepa2":
            # The hub `main` (checked 2026-10-17) points its checkpoint URL at localhost and passes H_patches/W_patches to
            # blocks whose forward takes H/W. Pre-download https://dl.fbaipublicfiles.com/vjepa2/vitl.pt to
            # ~/.cache/torch/hub/checkpoints/vitl.pt; the kwarg mismatch is patched here when a dummy forward fails.
            enc, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
            enc = enc.to(device).eval()
            try:
                with torch.no_grad():
                    enc(torch.zeros(1, 3, 2, 256, 256, device=device))
            except TypeError:
                for blk in enc.blocks:
                    blk.forward = _drop_patch_kwargs(blk.forward)
            self.net, self.res, self.video = enc, 256, True
            self.fwd = self._vjepa_fwd
        else:
            raise ValueError(name)
        self.mean_t = torch.tensor(self.mean, device=device).view(1, 3, 1, 1)
        self.std_t = torch.tensor(self.std, device=device).view(1, 3, 1, 1)

    def _vjepa_fwd(self, x):  # x (N, 3, 256, 256) -> clip (N, 3, 2, 256, 256) -> tokens (N, 16*16, D)
        clip = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        return self.net(clip)

    @torch.no_grad()
    def __call__(self, x):
        """x (N, 3, H, W) float in [0, 1] -> (N, n_patches, D)"""
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        x = (x - self.mean_t) / self.std_t
        out = []
        for s in range(0, len(x), 64):
            out.append(self.fwd(x[s : s + 64]).float())
        return torch.cat(out)


def install_alt_encoder(agent, name):
    """Monkeypatch agent.model.encode to use the named backbone for the visual features (proprio unchanged)."""
    if name == "dinov2":
        return None
    model, device = agent.model, agent.device
    bb = AltBackbone(name, device)

    @torch.no_grad()
    def encode(obs, act=True):
        if isinstance(obs, (TensorDict, dict)):
            visual = obs["visual"]
            tp = model.preprocessor.normalize_proprios(obs["proprio"].cpu()).to(device, dtype=torch.float32)
            proprio_emb = model.encode_proprio(tp)
        else:
            visual, proprio_emb = obs, None
        b, t = visual.shape[:2]
        x = rearrange(visual.to(device, dtype=torch.float32) / 255.0, "b t c h w -> (b t) c h w")
        tok = bb(x)
        g = int(round(tok.shape[1] ** 0.5))
        feats = rearrange(tok, "(b t) (h w) d -> b t 1 h w d", b=b, t=t, h=g, w=g)
        if proprio_emb is None:
            return feats
        return TensorDict({"visual": feats, "proprio": proprio_emb}, device=device)

    model.encode = encode
    agent.alt_encoder = name
    return bb
