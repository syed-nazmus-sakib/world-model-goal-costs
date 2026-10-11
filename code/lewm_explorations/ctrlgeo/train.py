"""Minimal LeWM trainer (same architecture + loss as stable-worldmodel's
scripts/train/lewm.py) over our fixed-budget window datasets.

  python -m ctrlgeo.train --data results/exp5/data/RND4.npz --out results/exp5/models/RND4_s0 --seed 0
"""
import argparse, faulthandler, json, os, time

import hydra
import numpy as np
import torch
from stable_pretraining.data import dataset_stats
from stable_worldmodel.wm.loss import SIGReg


def model_config(img_size, act_in=10):
    return {
        '_target_': 'stable_worldmodel.wm.lewm.LeWM',
        'encoder': {'_target_': 'stable_pretraining.backbone.utils.vit_hf', 'size': 'tiny',
                    'patch_size': 14, 'image_size': img_size, 'pretrained': False,
                    'use_mask_token': False},
        'predictor': {'_target_': 'stable_worldmodel.wm.lewm.module.Predictor', 'num_frames': 3,
                      'input_dim': 192, 'hidden_dim': 192, 'output_dim': 192, 'depth': 6,
                      'heads': 16, 'mlp_dim': 2048, 'dim_head': 64, 'dropout': 0.1,
                      'emb_dropout': 0.0},
        'action_encoder': {'_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
                           'input_dim': act_in, 'emb_dim': 192},
        'projector': {'_target_': 'stable_worldmodel.wm.lewm.module.MLP', 'input_dim': 192,
                      'output_dim': 192, 'hidden_dim': 2048,
                      'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True}},
        'pred_proj': {'_target_': 'stable_worldmodel.wm.lewm.module.MLP', 'input_dim': 192,
                      'output_dim': 192, 'hidden_dim': 2048,
                      'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True}},
    }


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--sigreg', type=float, default=0.09)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val_frac', type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = 'cuda'

    d = np.load(args.data)
    frames = torch.from_numpy(d['frames']).to(dev)            # (N,4,R,R,3) uint8 on GPU
    actions = torch.from_numpy(d['actions']).to(dev)          # (N,4,10)
    anchor = d['anchor']
    R = frames.shape[2]
    # group split by anchor so branches of one state never straddle train/val
    rng = np.random.default_rng(1234)
    ua = np.unique(anchor)
    val_a = set(rng.choice(ua, int(len(ua) * args.val_frac), replace=False).tolist())
    is_val = np.isin(anchor, list(val_a))
    tr_idx = torch.tensor(np.where(~is_val)[0], device=dev)
    va_idx = torch.tensor(np.where(is_val)[0], device=dev)

    cfg = model_config(R, actions.shape[-1])
    model = hydra.utils.instantiate(cfg).to(dev)
    sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05, anneal_strategy='cos')
    st = dataset_stats.ImageNet
    mean = torch.tensor(st['mean'], device=dev).view(1, 1, 3, 1, 1)
    std = torch.tensor(st['std'], device=dev).view(1, 1, 3, 1, 1)
    g = torch.Generator(device=dev).manual_seed(args.seed)

    def batch(idx):
        x = frames[idx].permute(0, 1, 4, 2, 3).float().div_(255)
        return {'pixels': (x - mean) / std, 'action': actions[idx]}

    def loss_fn(b):
        out = model.encode(b)
        emb, act = out['emb'], out['act_emb']
        pred = model.predict(emb[:, :3], act[:, :3])
        pl = (pred - emb[:, 1:]).pow(2).mean()
        sr = sigreg(emb.transpose(0, 1))
        return pl, sr

    log, start = [], 1
    ck = os.path.join(args.out, 'resume.pt')
    if os.path.exists(ck):  # resume after a crash
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt'])
        sched.load_state_dict(c['sched']); log = c['log']; start = c['step'] + 1
        g.manual_seed(args.seed * 1000003 + start)
        print(f'resumed from step {c["step"]}', flush=True)
    t0 = time.time()
    model.train()
    for it in range(start, args.steps + 1):
        idx = tr_idx[torch.randint(len(tr_idx), (args.bs,), device=dev, generator=g)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pl, sr = loss_fn(batch(idx))
            loss = pl + args.sigreg * sr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 500 == 0 or it == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                vl = [loss_fn(batch(va_idx[i:i + 256])) for i in range(0, min(len(va_idx), 2048), 256)]
            model.train()
            rec = dict(step=it, pred=pl.item(), sigreg=sr.item(),
                       val_pred=float(np.mean([v[0].item() for v in vl])),
                       val_sigreg=float(np.mean([v[1].item() for v in vl])),
                       lr=sched.get_last_lr()[0], time=time.time() - t0)
            log.append(rec)
            print(json.dumps(rec), flush=True)
        if it % 2000 == 0 and it < args.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            step=it, log=log), ck + '.tmp')
            os.replace(ck + '.tmp', ck)
    model.eval()
    torch.save(model.state_dict(), os.path.join(args.out, 'weights.pt'))
    json.dump(cfg, open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
    json.dump(dict(vars(args), log=log), open(os.path.join(args.out, 'train_log.json'), 'w'))
    if os.path.exists(ck):
        os.remove(ck)


if __name__ == '__main__':
    main()
