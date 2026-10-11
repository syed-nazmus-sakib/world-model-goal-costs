"""Frozen LeWM wrapper: encode images, roll out Phi_K(z, u), action Jacobians.

All actions here are in the *normalised* space the planner uses (z-scored raw
2-d actions, blocked by 5 env steps -> 10-d per model step).
"""
import os

import h5py
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import stable_worldmodel as swm
from stable_pretraining.data import dataset_stats

from ctrlgeo.paths import H5
ACTION_BLOCK = 5


class FrozenLeWM:
    def __init__(self, name='quentinll/lewm-pusht', device='cuda', path=None, a_mean=None, a_std=None):
        """name: stable-worldmodel checkpoint id; or path: folder with config.json +
        weights.pt written by ctrlgeo.train."""
        if path is not None:
            import json, hydra
            cfg = json.load(open(os.path.join(path, 'config.json')))
            m = hydra.utils.instantiate(cfg)
            m.load_state_dict(torch.load(os.path.join(path, 'weights.pt'), map_location='cpu'))
        else:
            m = swm.wm.utils.load_pretrained(name)
        self.res = m.encoder.config.image_size
        m.interpolate_pos_encoding = True
        self.m = m.to(device).eval().requires_grad_(False)
        self.dev = device
        self.hist = getattr(self.m.predictor, 'num_frames', 3)
        st = dataset_stats.ImageNet
        self.img_mean = torch.tensor(st['mean'], device=device).view(1, 3, 1, 1)
        self.img_std = torch.tensor(st['std'], device=device).view(1, 3, 1, 1)
        if a_mean is not None:  # e.g. Cube: stats stored next to the dataset
            self.a_mean, self.a_std = np.asarray(a_mean), np.asarray(a_std)
        else:
            with h5py.File(H5, 'r') as f:
                a = f['action'][::10]
            a = a[~np.isnan(a).any(1)]
            self.a_mean, self.a_std = a.mean(0), a.std(0)
        self.act_dim = len(self.a_mean)                 # per env step
        self.block_dim = ACTION_BLOCK * self.act_dim    # per model step

    # ---- actions -------------------------------------------------------
    def norm_actions(self, raw):
        """raw (..., T*5, 2) env actions -> normalised (..., T, 10)."""
        raw = np.asarray(raw)
        n = (raw - self.a_mean) / self.a_std
        return n.reshape(*n.shape[:-2], -1, self.block_dim)

    def denorm_actions(self, u):
        """normalised (..., T, 10) -> raw (..., T*5, 2)."""
        u = np.asarray(u)
        r = u.reshape(*u.shape[:-2], -1, self.act_dim)
        return r * self.a_std + self.a_mean

    # ---- encoder -------------------------------------------------------
    @torch.no_grad()
    def encode(self, imgs, bs=256):
        """imgs: uint8 (N,224,224,3) -> (N,192) float32."""
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.as_tensor(imgs[i:i + bs], device=self.dev)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.img_mean) / self.img_std
            o = self.m.encoder(x, interpolate_pos_encoding=True)
            out.append(self.m.projector(o.last_hidden_state[:, 0]))
        return torch.cat(out).float()

    # ---- dynamics ------------------------------------------------------
    def rollout(self, z, u):
        """z: (N,D) current latent; u: (N,K,10) normalised actions.
        Returns (N,K+1,D) — index 0 is z itself. Mirrors LeWM.rollout with a
        single context frame (history_len=1, as in the PushT eval config)."""
        act = self.m.action_encoder(u)  # (N,K,A)
        embs = [z]
        for t in range(u.shape[1]):
            lo = max(0, 1 + t - self.hist)
            e = torch.stack(embs[lo:], 1)
            a = act[:, lo:1 + t]
            embs.append(self.m.predict(e, a)[:, -1])
        return torch.stack(embs, 1)

    def phi(self, z, u):
        return self.rollout(z, u)[:, -1]

    def jacobian(self, z, u, chunk=64):
        """d Phi_K / d u at (z,u): returns (N, D, K*10) via forward-mode AD."""
        N, K, A = u.shape

        def f(zz, uu):
            return self.phi(zz[None], uu.view(1, K, A))[0]

        jf = torch.func.vmap(torch.func.jacfwd(f, argnums=1))
        out = []
        # forward-mode AD is only implemented for the math SDPA kernel
        with sdpa_kernel(SDPBackend.MATH):
            for i in range(0, N, chunk):
                out.append(jf(z[i:i + chunk], u[i:i + chunk].reshape(-1, K * A)))
        return torch.cat(out)

    def jacobian_fd(self, z, u, eps=1e-2):
        """Central finite differences (fallback / check). (N,D,K*10)."""
        N, K, A = u.shape
        P = K * A
        I = torch.eye(P, device=self.dev).view(P, K, A) * eps
        up = (u[:, None] + I[None]).reshape(N * P, K, A)
        um = (u[:, None] - I[None]).reshape(N * P, K, A)
        zz = z.repeat_interleave(P, 0)
        with torch.no_grad():
            d = (self.phi(zz, up) - self.phi(zz, um)) / (2 * eps)
        return d.view(N, P, -1).transpose(1, 2)
