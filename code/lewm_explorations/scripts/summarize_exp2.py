import sys, json, collections
import numpy as np
d = collections.defaultdict(list); meta = {}
for path in sys.argv[1:]:
    for l in open(path):
        r = json.loads(l)
        k = (r.get('offset', 25), r['budget'], r['method'])
        d[k].append(r['success']); meta[k] = r['n_samples']
for k, v in sorted(d.items()):
    print(f'offset={k[0]:3d} budget={k[1]:5d} {k[2]:32s} n={meta[k]:4d} '
          f'success={np.mean(v):5.1f} ± {np.std(v)/max(len(v)-1,1)**.5:4.1f}  {v}')
