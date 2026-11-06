"""exp-20: encode the 5-step grid of a dataset with any world model, for the cross-model range diagnostic.
--model <checkpoint name>   stable-worldmodel checkpoint (LeWM, PLDM, DINO-WM ...); patch latents are mean-pooled
--model dinov2              frozen DINOv2-S patch tokens pooled to 4x4 (the features DINO-WM and the policies start from)
Stores emb (N, D) fp16, state (N, S), ep, step.  --max_nodes limits the node count (quick test).
usage (from code/third_party/stable-worldmodel): python <path>/exp_encode.py --out x.npz --model NAME --dataset DS --state_key KEY"""
import argparse

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--model', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--state_key', default='state'); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max_nodes', type=int, default=0)
ap.add_argument('--ep_frac', type=float, default=1.0, help='keep this seeded fraction of episodes, the same frames for every model')
args = ap.parse_args()
ds = swm.data.load_dataset(args.dataset, num_steps=1, frameskip=5, keys_to_load=['pixels', args.state_key])
st = dt.dataset_stats.ImageNet
ds.transform = spt.data.transforms.Compose(dt.transforms.Compose(dt.transforms.ToImage(**st, source='pixels', target='pixels'),
                                                                 dt.transforms.Resize(224, source='pixels', target='pixels')))
clips = np.asarray(ds.clip_indices)
pick = np.nonzero(clips[:, 1] % 5 == 0)[0]
if args.ep_frac < 1:
    eps = np.unique(clips[pick, 0])
    keep = np.random.default_rng(0).choice(eps, int(len(eps) * args.ep_frac), replace=False)
    pick = pick[np.isin(clips[pick, 0], keep)]
if args.max_nodes:
    pick = pick[:args.max_nodes]
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, pick), batch_size=args.batch, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
if args.model == 'dinov2':
    from transformers import AutoModel
    net = AutoModel.from_pretrained('facebook/dinov2-small').cuda().eval().requires_grad_(False)
    def enc(x):  # 16x16 patch tokens pooled to a 4x4 grid, so the squared distance stays a spatial, patch-like cost
        t = net(pixel_values=x[:, 0]).last_hidden_state[:, 1:]
        t = t.reshape(len(t), 16, 16, -1).permute(0, 3, 1, 2)
        return torch.nn.functional.avg_pool2d(t, 4).flatten(1)
else:
    model = (torch.load(args.model, weights_only=False) if args.model.endswith('.pt') else swm.wm.utils.load_pretrained(args.model)
             ).cuda().eval().requires_grad_(False)
    def enc(x):
        info = model.encode({'pixels': x}, emb_keys=[]) if hasattr(model, 'extra_encoders') else model.encode({'pixels': x})
        e = (info['pixels_emb'] if 'pixels_emb' in info else info['emb'])[:, 0]
        if e.dim() == 3:                                     # patches on a square grid -> 4x4 grid, as in paa_cost._pool4
            n = int(e.shape[1] ** 0.5)
            e = torch.nn.functional.adaptive_avg_pool2d(e.reshape(-1, n, n, e.shape[-1]).permute(0, 3, 1, 2), 4).flatten(1)
        return e
embs, states = [], []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for i, b in enumerate(loader):
        embs.append(enc(b['pixels'].cuda(non_blocking=True)).float().cpu().half())
        states.append(b[args.state_key][:, 0].float())
        if i % 100 == 0:
            print(f'{(i + 1) * args.batch} / {len(pick)}', flush=True)
np.savez(args.out, emb=torch.cat(embs).numpy(), state=torch.cat(states).numpy(), ep=clips[pick, 0], step=clips[pick, 1])
print('saved', args.out, torch.cat(embs).shape, flush=True)
