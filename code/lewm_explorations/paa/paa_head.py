"""PAA attainability head: P(the deployed planner reaches anchor z_a from z_t when aiming at it for hops+2 blocks).

Training data: harvest dumps written by eval_wm.py (dump_npz=...) on training-episode pairs (start, start + 5*hops),
joined with the graph nodes (paa_nodes.py) for the start and anchor latents.  Label (default, no state labels):
the encoder latent of the real final frame is within eps of the anchor latent, eps = the median one-block latent step.
--label success uses the simulator's success test instead (privileged ablation).
usage: python paa_head.py --nodes nodes.npz --dumps 'harvest/*.npz' --out head.pt [--label latent|success]
"""
import argparse
import glob

import numpy as np
import torch
from torch import nn


class AttainNet(nn.Module):
    def __init__(self, d, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d + 2, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 1))

        self.register_buffer('mu', torch.zeros(d)); self.register_buffer('sd', torch.ones(d))

    def forward(self, z0, za, hops):
        z0, za = (z0 - self.mu) / self.sd, (za - self.mu) / self.sd
        dz = za - z0
        x = torch.cat([z0, za, dz, dz.pow(2).sum(-1, keepdim=True).sqrt(), hops[:, None] / 8.0], -1)
        return self.net(x)[:, 0]


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def forward(self, z0, za, hops):
        p = torch.stack([torch.sigmoid(m(z0.float(), za.float(), hops.float())) for m in self.members])
        return p.mean(0), p.std(0)


def load_head(path):
    ck = torch.load(path, map_location='cpu')
    ens = Ensemble([AttainNet(ck['d']) for _ in ck['state_dicts']])
    for m, sd in zip(ens.members, ck['state_dicts']):
        m.load_state_dict(sd, strict=False)  # heads saved before input standardization have no mu/sd (identity)
    return ens.cuda().eval()


def build(nodes, dumps, label):
    d = np.load(nodes)
    key = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(d['ep'], d['step']))}
    emb = d['emb'].astype(np.float32)
    order = np.lexsort((d['step'], d['ep']))
    a, b = order[:-1], order[1:]
    nxt = (d['ep'][a] == d['ep'][b]) & (d['step'][b] - d['step'][a] == 5)
    eps = float(np.median(((emb[a[nxt]] - emb[b[nxt]]) ** 2).sum(-1)))
    Z0, ZA, HP, Y, YS = [], [], [], [], []
    for f in sorted(glob.glob(dumps)):
        h = np.load(f)
        off = int(h['offset'])
        for i, (e, s) in enumerate(zip(h['ep'], h['start'])):
            s0 = 5 * int(round(int(s) / 5))  # starts are not grid-aligned: snap to the nearest grid frame
            i0, ia = key.get((int(e), s0)), key.get((int(e), s0 + off))
            if i0 is None or ia is None:
                s0 = 5 * (int(s) // 5)
                i0, ia = key.get((int(e), s0)), key.get((int(e), s0 + off))
            if i0 is None or ia is None:
                continue
            Z0.append(emb[i0]); ZA.append(emb[ia]); HP.append(off / 5.0)
            Y.append(float(((h['final_emb'][i] - emb[ia]) ** 2).sum() < eps)); YS.append(float(h['success'][i]))
    Z0, ZA, HP = map(lambda x: torch.tensor(np.stack(x) if isinstance(x[0], np.ndarray) else np.array(x)), (Z0, ZA, HP))
    y = torch.tensor(Y if label == 'latent' else YS)
    print(f'{len(y)} attempts, eps={eps:.3f}, attained (latent) {np.mean(Y):.3f}, success {np.mean(YS):.3f}, '
          f'agreement {np.mean(np.array(Y) == np.array(YS)):.3f}', flush=True)
    return Z0.float(), ZA.float(), HP.float(), y.float(), np.array(Y), np.array(YS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', required=True); ap.add_argument('--dumps', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--label', default='latent', choices=['latent', 'success'])
    ap.add_argument('--n_ens', type=int, default=3); ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    Z0, ZA, HP, y, _, _ = build(args.nodes, args.dumps, args.label)
    n = len(y); perm = rng.permutation(n); va, tr = perm[: n // 10], perm[n // 10:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z0, ZA, HP, y = Z0.to(dev), ZA.to(dev), HP.to(dev), y.to(dev)
    members = []
    for k in range(args.n_ens):
        m = AttainNet(Z0.shape[1]).to(dev)
        m.mu.copy_(Z0.mean(0)); m.sd.copy_(Z0.std(0) + 1e-6)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-2)
        bi = rng.choice(tr, size=len(tr), replace=True)  # bootstrap per member; out-of-bag rows pick the stopping step
        boot, oob = torch.as_tensor(bi, device=dev), torch.as_tensor(np.setdiff1d(tr, bi), device=dev)
        best, best_sd = float('inf'), None
        for s in range(args.steps):
            i = boot[torch.randint(len(boot), (512,), device=dev)]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Z0[i], ZA[i], HP[i]), y[i])
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 10 == 9:
                with torch.no_grad():
                    lo = nn.functional.binary_cross_entropy_with_logits(m(Z0[oob], ZA[oob], HP[oob]), y[oob]).item()
                if lo < best:
                    best, best_sd = lo, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_sd)
        print(f'member {k}: oob bce {best:.3f}', flush=True)
        members.append(m.eval())
    ens = Ensemble(members)
    with torch.no_grad():
        vi = torch.as_tensor(va, device=dev)
        p, sd = ens(Z0[vi], ZA[vi], HP[vi])
        yv = y[vi]
        acc = ((p > 0.5).float() == yv).float().mean().item()
        bins = torch.linspace(0, 1, 6, device=dev)
        rel = [(float(p[(p >= lo) & (p < hi)].mean()), float(yv[(p >= lo) & (p < hi)].mean()), int(((p >= lo) & (p < hi)).sum()))
               for lo, hi in zip(bins[:-1], bins[1:]) if ((p >= lo) & (p < hi)).any()]
        pos, neg = p[yv == 1], p[yv == 0]
        auc = (pos[:, None] > neg[None]).float().mean().item()
        hv = HP[vi]  # reference: ranking by hop count alone
        auc_h = ((hv[yv == 1][:, None] < hv[yv == 0][None]).float() + 0.5 * (hv[yv == 1][:, None] == hv[yv == 0][None]).float()).mean().item()
    print(f'val acc {acc:.3f} auc {auc:.3f} (hops-only auc {auc_h:.3f}) base rate {yv.mean().item():.3f}; reliability (pred, obs, n): {rel}', flush=True)
    torch.save({'d': Z0.shape[1], 'label': args.label, 'state_dicts': [m.state_dict() for m in members]}, args.out)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
