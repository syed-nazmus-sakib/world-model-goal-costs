"""PAA step 1: graph nodes.  Encode every frame on the 5-step grid of the LeWM PushT dataset with the frozen LeWM encoder.
Stores emb (N, D) fp16, act (N, A) the action block leaving the frame (normalized, solver space), state (N, S), ep, step.
usage (from code/third_party/stable-worldmodel):  python <path>/paa_nodes.py --out /root/out/paa/nodes.npz
"""
import argparse

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='quentinll/lewm-pusht')
    ap.add_argument('--dataset', default='galilai-group/lewm-pusht')
    ap.add_argument('--state_key', default='state')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ds = swm.data.load_dataset(args.dataset, num_steps=1, frameskip=5, keys_to_load=['pixels', 'action', args.state_key])
    stats = dt.dataset_stats.ImageNet
    ds.transform = spt.data.transforms.Compose(
        dt.transforms.Compose(
            dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
            dt.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        column_normalizer(ds, 'action', 'action'),
    )
    clips = np.asarray(ds.clip_indices)
    pick = np.nonzero(clips[:, 1] % 5 == 0)[0]
    print(f'{len(clips)} clips, {len(pick)} on the 5-step grid, {len(np.unique(clips[:, 0]))} episodes', flush=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, pick), batch_size=args.batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True, prefetch_factor=2)
    model = swm.wm.utils.load_pretrained(args.model).cuda().eval().requires_grad_(False)
    embs, acts, states = [], [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, b in enumerate(loader):
            emb = model.encode({'pixels': b['pixels'].cuda(non_blocking=True)})['emb']  # (B, 1, D)
            embs.append(emb[:, 0].float().cpu().half())
            acts.append(torch.nan_to_num(b['action'][:, 0].float(), 0.0))
            states.append(b[args.state_key][:, 0].float())
            if i % 50 == 0:
                print(f'{(i + 1) * args.batch} / {len(pick)}', flush=True)
    np.savez(args.out, emb=torch.cat(embs).numpy(), act=torch.cat(acts).numpy(), state=torch.cat(states).numpy(),
             ep=clips[pick, 0], step=clips[pick, 1])
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
