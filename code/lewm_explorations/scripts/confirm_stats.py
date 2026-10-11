"""Paired analysis of the pre-registered confirmation run."""
import json, sys
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'results/exp2/confirm_prereg.jsonl'
R = {}
for l in open(path):
    r = json.loads(l); R[(r['method'], r['seed'])] = np.array(r['ep_success'])
names = {'geometry': 'cf_ridge:K1:M8:rho0.1', 'L2': 'l2', 'bestprobe': 'probemin:K1:M8'}
seeds = sorted({s for (_, s) in R})
X = {k: np.stack([R[(m, s)] for s in seeds]) * 100.0 for k, m in names.items()}  # (S,E)
rng = np.random.default_rng(0)
for k in names:
    per = X[k].mean(1)
    print(f'{k:10s} SR={per.mean():5.1f}  per-seed={per.round(0).tolist()}')
for other in ('L2', 'bestprobe'):
    d = X['geometry'] - X[other]  # paired per episode
    ds = d.mean(1)
    cb = [ds[rng.integers(0, len(ds), len(ds))].mean() for _ in range(10000)]
    flat = d.ravel()
    eb = [flat[rng.integers(0, len(flat), len(flat))].mean() for _ in range(10000)]
    wins = int((ds > 0).sum())
    print(f'Δ vs {other:9s} = {ds.mean():+5.1f}  seed-cluster CI[{np.percentile(cb,2.5):+.1f},{np.percentile(cb,97.5):+.1f}]'
          f'  episode CI[{np.percentile(eb,2.5):+.1f},{np.percentile(eb,97.5):+.1f}]  seeds won {wins}/{len(ds)}'
          f'  discordant eps: geo-only {int((d>0).sum())}, {other}-only {int((d<0).sum())}')
