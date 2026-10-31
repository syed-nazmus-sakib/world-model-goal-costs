"""Fetch a LeWM dataset (HF collection quentinll/lewm) into $STABLEWM_HOME/datasets and print what it holds.
Lists the quentinll / galilai-group dataset repos, downloads the ones whose id contains the env key, extracts any
.tar.zst archive, links every .h5 into the datasets folder and prints its top-level keys and episode-length summary.
usage: python fetch_lewm_data.py <tworoom|cube|reacher>
"""
import glob
import os
import subprocess
import sys

import h5py
import numpy as np
from huggingface_hub import HfApi, snapshot_download

env = sys.argv[1]
key = {'tworoom': 'tworoom', 'cube': 'cube', 'reacher': 'reacher'}[env]
home = os.environ.get('STABLEWM_HOME', os.path.expanduser('~/.stable_worldmodel'))
dd = os.path.join(home, 'datasets')
os.makedirs(dd, exist_ok=True)
api = HfApi()
ids = []
for author in ('quentinll', 'galilai-group'):
    try:
        ids += [d.id for d in api.list_datasets(author=author, limit=500)]
    except Exception as e:  # noqa: BLE001
        print('list failed', author, e)
print('dataset repos:', [i for i in ids if 'lewm' in i.lower() or key in i.lower()], flush=True)
pick = [i for i in ids if key in i.lower() and i.startswith('quentinll/')] or [i for i in ids if key in i.lower()]
print('downloading', pick, flush=True)
for rid in pick:
    loc = os.path.join(dd, '_raw', rid.replace('/', '--'))
    snapshot_download(rid, repo_type='dataset', local_dir=loc)
    for z in glob.glob(f'{loc}/**/*.tar.zst', recursive=True):
        subprocess.run(['tar', '--zstd', '-xf', z, '-C', os.path.dirname(z)], check=True)
    for h in glob.glob(f'{loc}/**/*.h5', recursive=True):
        dst = os.path.join(dd, os.path.basename(h))
        if not os.path.exists(dst):
            os.symlink(h, dst)
for h in sorted(glob.glob(f'{dd}/*.h5')):
    if key not in os.path.basename(h).lower():
        continue
    with h5py.File(h, 'r') as f:
        info = {k: (f[k].shape if hasattr(f[k], 'shape') else 'group') for k in f.keys()}
        print('H5', os.path.basename(h), os.path.getsize(h) >> 20, 'MB', info, flush=True)
        for lk in ('ep_len', 'episode_len', 'ep_lens'):
            if lk in f:
                ln = np.asarray(f[lk])
                print('  episodes', len(ln), 'len min/median/max', ln.min(), int(np.median(ln)), ln.max(),
                      {o: int((ln > o).sum()) for o in (25, 50, 75, 100, 150, 200)}, flush=True)
print('FETCH_DONE', env, flush=True)
