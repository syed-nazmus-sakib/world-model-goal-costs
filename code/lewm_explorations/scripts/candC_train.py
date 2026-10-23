"""Method C (2026-10-22), step 2: train a task-aligned goal cost head on frozen LeWM latents.

cost(z, g) = || f(z) - f(g) ||^2, f a small MLP. The head is trained so that || f(z) - f(g) || regresses
y = log(1 + d), where d is the distance to success between the states of z and g:
    --target state : d = |agent+block position error| / 20 px + |block angle error| / (pi/9)
                     (the PushT success test in units of its thresholds; uses simulator states at training only)
    --target time  : d = steps between z and g in the same episode / 5 (self-supervised, no state labels)
z is the predictor's rollout latent (what CEM scores) or a real frame's latent; g is always a real frame later in the
same episode (plus 20% goals from other episodes, state target only). log1p makes the cost most precise near the goal.
Episodes are split 90/10 for validation (Spearman of cost vs true d on held-out episodes).

Usage: python candC_train.py --data /root/out/C/latents.npz --target state --out /root/out/C/head_state.pt
"""

import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


class Head(nn.Module):
    def __init__(self, d_in=192, width=512, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, d_out),
        )

    def forward(self, z):
        return self.net(z)


def success_distance(s, g, env='pusht'):
    if env == 'tworoom':  # success: agent within 16 px of the target (straight line; ignores the wall)
        return np.linalg.norm(s[..., :2] - g[..., :2], axis=-1) / 16.0
    if env == 'cube':  # success: block within 0.04 m of the target (state = privileged block xyz)
        return np.linalg.norm(s[..., :3] - g[..., :3], axis=-1) / 0.04
    if env == 'reacher':  # success: every joint within 0.05 rad of the target qpos
        return np.abs(s - g).max(-1) / 0.05
    pos = np.linalg.norm(s[..., :4] - g[..., :4], axis=-1) / 20.0
    ang = np.abs(s[..., 4] - g[..., 4]) % (2 * np.pi)
    ang = np.minimum(ang, 2 * np.pi - ang) / (np.pi / 9)
    return pos + ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', choices=['state', 'time'], default='state')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--p_pred', type=float, default=0.7, help='fraction of anchors that are predictor latents')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--env', choices=['pusht', 'tworoom', 'cube', 'reacher'], default='pusht')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    D = np.load(args.data)
    emb, pred, state, ep, start = D['emb'], D['pred'], D['state'], D['ep'], D['start']
    N = len(ep)
    # goal bank: every real frame, indexed by (episode, step)
    bank_z = emb.reshape(N * 6, -1)
    bank_s = state.reshape(N * 6, -1)
    bank_ep = np.repeat(ep, 6)
    bank_t = (start[:, None] + 5 * np.arange(6)[None]).reshape(-1)
    order = np.lexsort((bank_t, bank_ep))
    bank_z, bank_s, bank_ep, bank_t = bank_z[order], bank_s[order], bank_ep[order], bank_t[order]
    BIG = 10**7
    bank_key = bank_ep.astype(np.int64) * BIG + bank_t
    eps = np.unique(ep)
    val_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
    is_val = np.array([e in val_eps for e in ep])
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]

    def sample(idx, n):
        i = rng.choice(idx, size=n)
        k = rng.integers(0, 5, size=n)  # anchor = frame k+1 of the clip
        use_pred = rng.random(n) < args.p_pred
        z = np.where(use_pred[:, None], pred[i, k], emb[i, k + 1]).astype(np.float32)
        s = state[i, k + 1]
        t = start[i] + 5 * (k + 1)
        # goal: a real frame of the same episode at or after the anchor (bank rows are sorted by step)
        lo = np.searchsorted(bank_key, ep[i].astype(np.int64) * BIG + t)  # first bank frame of this episode at or after t
        hi = np.searchsorted(bank_key, (ep[i].astype(np.int64) + 1) * BIG)  # end of the episode
        gi = lo + (rng.random(n) * np.maximum(hi - lo, 1)).astype(np.int64)
        gi = np.minimum(gi, len(bank_key) - 1)
        if args.target == 'state':
            other = rng.random(n) < 0.2
            gi[other] = rng.integers(0, len(bank_ep), size=other.sum())
            d = success_distance(s, bank_s[gi], args.env)
        else:
            d = np.abs(bank_t[gi] - t) / 5.0
        return (torch.from_numpy(z), torch.from_numpy(bank_z[gi].astype(np.float32)),
                torch.from_numpy(np.log1p(d).astype(np.float32)), d)

    head = Head(d_in=emb.shape[-1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    log = []
    for step in range(1, args.steps + 1):
        z, g, y, _ = sample(tr_idx, args.batch)
        z, g, y = z.cuda(), g.cuda(), y.cuda()
        dist = (head(z) - head(g)).norm(dim=-1)
        loss = nn.functional.smooth_l1_loss(dist, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps:
            head.eval()
            with torch.no_grad():
                z, g, y, d = sample(va_idx, 8192)
                c = (head(z.cuda()) - head(g.cuda())).norm(dim=-1).cpu().numpy()
                base = ((z - g) ** 2).sum(-1).numpy()  # released LeWM cost on the same pairs
            rec = {'step': step, 'loss': float(loss), 'val_spearman_head': float(spearmanr(c, d).correlation),
                   'val_spearman_released': float(spearmanr(base, d).correlation)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            head.train()
    torch.save({'state_dict': head.state_dict(), 'd_in': emb.shape[-1], 'args': vars(args), 'log': log}, args.out)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
