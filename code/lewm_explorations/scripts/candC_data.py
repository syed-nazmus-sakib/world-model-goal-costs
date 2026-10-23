"""Method C (2026-10-22), step 1: latents for training a task-aligned goal cost on LeWM PushT.

For a random subset of 6-frame clips (frameskip 5, clips starting on the 5-step grid) store
  emb   (N, 6, D)  frozen LeWM embeddings of the 6 real frames,
  pred  (N, 5, D)  the LeWM predictor's autoregressive rollout from frame 0 with the logged actions
                   (exactly what CEM scores: one context frame, history 3, 5 blocks),
  state (N, 6, 7)  simulator states of the 6 frames (agent xy, block xy, block angle, agent velocity),
  ep, start        episode index and start step.
Usage (from code/third_party/stable-worldmodel):
    python <path>/candC_data.py --out /root/out/C/latents.npz --clips 120000
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
    ap.add_argument('--clips', type=int, default=120000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--state_key', default='state', help="'state' (PushT) or 'pos_agent' (TwoRoom)")
    args = ap.parse_args()

    ds = swm.data.load_dataset(
        args.dataset, num_steps=6, frameskip=5, keys_to_load=['pixels', 'action', args.state_key],
    )
    stats = dt.dataset_stats.ImageNet
    ds.transform = spt.data.transforms.Compose(
        dt.transforms.Compose(
            dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
            dt.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        column_normalizer(ds, 'action', 'action'),
    )
    clips = np.asarray(ds.clip_indices)
    on_grid = np.nonzero(clips[:, 1] % 5 == 0)[0]
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(on_grid, size=min(args.clips, len(on_grid)), replace=False))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, pick), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, prefetch_factor=2,
    )

    model = swm.wm.utils.load_pretrained(args.model).cuda().eval().requires_grad_(False)
    HS = getattr(model.predictor, 'num_frames', 3)
    embs, preds, states = [], [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, b in enumerate(loader):
            px = b['pixels'].cuda(non_blocking=True)
            act = torch.nan_to_num(b['action'].cuda(non_blocking=True), 0.0)
            emb = model.encode({'pixels': px})['emb']  # (B, 6, D)
            act_emb = model.action_encoder(act)  # (B, 6, D); index k = block leaving frame k
            hist = [emb[:, 0]]
            for t in range(5):  # same indexing as LeWM.rollout with one context frame
                lo = max(0, 1 + t - HS)
                hist.append(model.predict(torch.stack(hist[lo:], 1), act_emb[:, lo : 1 + t])[:, -1])
            embs.append(emb.float().cpu().half())
            preds.append(torch.stack(hist[1:], 1).float().cpu().half())
            states.append(b[args.state_key].float())
            if i % 50 == 0:
                print(f'{(i + 1) * args.batch} / {len(pick)}', flush=True)
    np.savez(
        args.out, emb=torch.cat(embs).numpy(), pred=torch.cat(preds).numpy(),
        state=torch.cat(states).numpy(), ep=clips[pick, 0], start=clips[pick, 1],
    )
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
