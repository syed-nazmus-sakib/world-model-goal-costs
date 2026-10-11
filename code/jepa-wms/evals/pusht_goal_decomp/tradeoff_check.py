"""Exchange rate (pusher px vs block px) and trade-off disagreement with the official criterion, per cost."""
import glob, os, sys
import numpy as np

root = os.path.join(os.environ["JEPAWM_LOGS"], "pusht_goal_decomp")
for model in sys.argv[1:]:
    rows = {}
    files = sorted(glob.glob(os.path.join(root, model, "goal_*.npz")))
    X, Y = [], {}
    inv = {}
    for f in files:
        e = np.load(f); s = np.load(os.path.join(root, "sets", os.path.basename(f)))
        V = e["visual_emb"].astype(np.float32).reshape(-1, 256, 384)
        d = ((V - V[0]) ** 2).mean(-1)
        cs = {"mean (released)": d.mean(-1), "top-8 patches": np.sort(d, -1)[:, -8:].mean(-1),
              "top-4 patches": np.sort(d, -1)[:, -4:].mean(-1)}
        blk = s["cost_block_pos_err"].reshape(-1); pus = s["cost_pusher_pos_err"].reshape(-1)
        ang = s["cost_block_ang_err"].reshape(-1); off = s["cost_official_state_dist"].reshape(-1)
        sel = ang < 1e-6  # translation-only cells, so position errors fully describe the state
        X.append(np.stack([blk[sel], pus[sel], np.ones(sel.sum())], 1))
        iu, ju = np.triu_indices(len(blk), 1)
        trade = (np.sign(blk[iu] - blk[ju]) * np.sign(pus[iu] - pus[ju]) < 0) & (np.abs(off[iu] - off[ju]) > 1e-6)
        for n, c in cs.items():
            Y.setdefault(n, []).append(c[sel])
            a = inv.setdefault(n, [0, 0])
            a[0] += int((np.sign(c[iu] - c[ju]) != np.sign(off[iu] - off[ju]))[trade].sum()); a[1] += int(trade.sum())
    X = np.concatenate(X)
    print(f"== {model}")
    for n in Y:
        y = np.concatenate(Y[n]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"  {n:16s} pusher/block weight per px = {beta[1] / beta[0]:.2f} (task criterion: 1.00) | "
              f"disagrees with official on trade-off pairs: {100 * inv[n][0] / inv[n][1]:.1f}%  (n={inv[n][1]})")
