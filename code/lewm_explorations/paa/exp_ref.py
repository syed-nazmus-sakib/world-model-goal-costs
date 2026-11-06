"""Reference graph from true positions only (exp-20, exp-28): temporal edges + edges between nodes closer than the median
recorded five-step move (10 nearest in position).  Unit edge length, so distances are hops of about five steps."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def ref_graph(xy, ep, step):
    n = len(xy)
    order = np.lexsort((step, ep)); a, b = order[:-1], order[1:]
    nx = (ep[a] == ep[b]) & (step[b] - step[a] == 5); ta, tb = a[nx], b[nx]
    r = float(np.median(np.linalg.norm(xy[ta] - xy[tb], axis=1)))
    dist, idx = NearestNeighbors(n_neighbors=11).fit(xy).kneighbors(xy)
    ok = dist[:, 1:] < r; ri = np.repeat(np.arange(n)[:, None], 10, 1)
    rows = np.concatenate([ta, ri[ok], idx[:, 1:][ok]]); cols = np.concatenate([tb, idx[:, 1:][ok], ri[ok]])
    G = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)); G.data[:] = 1.0
    print(f'reference graph: {n} nodes, {G.nnz} edges, radius {r:.3f}', flush=True)
    return G
