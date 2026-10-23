"""Method candidate A (2026-10-22): give LeWM a learned null action, for action guidance at planning time.

Starts from the released LeWM PushT checkpoint. The encoder and projector are frozen (the latent space and the goal
cost stay exactly as released); the predictor, its projection head and the action encoder are fine-tuned on the same
expert data with the LeWM prediction loss. With probability p each action token is replaced by a learned null
embedding, so the predictor also learns "what happens next whatever the action" (classifier-free guidance training).
p = 0 is the control: same fine-tuning, no null.

Output: <STABLEWM_HOME>/checkpoints/<run>/weights.pt + config.json (loadable with swm.wm.utils.load_pretrained)
and null_emb.pth (the learned null embedding; absent for p = 0).

Usage (from code/third_party/stable-worldmodel):
    python <path>/candA_finetune.py --run candA/p0.15 --p_null 0.15 --steps 20000
"""

import argparse
import json
import time

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.utils import _resolve, get_cache_dir, save_pretrained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--base', default='quentinll/lewm-pusht')
    ap.add_argument('--dataset', default='galilai-group/lewm-pusht')
    ap.add_argument('--p_null', type=float, default=0.15)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # data: identical pipeline to scripts/train/lewm.py (history 3, 1 prediction, frameskip 5)
    dataset = swm.data.load_dataset(
        args.dataset, transform=None, num_steps=4, frameskip=5,
        keys_to_load=['pixels', 'action'], keys_to_cache=['action'],
    )
    stats = dt.dataset_stats.ImageNet
    transform = spt.data.transforms.Compose(
        dt.transforms.Compose(
            dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
            dt.transforms.Resize(224, source='pixels', target='pixels'),
        ),
        column_normalizer(dataset, 'action', 'action'),
    )
    dataset.transform = transform
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    model = swm.wm.utils.load_pretrained(args.base).cuda()
    _, config = _resolve(args.base, get_cache_dir(sub_folder='checkpoints'))
    for m in (model.encoder, model.projector):
        m.eval().requires_grad_(False)
    train_mods = [model.predictor, model.pred_proj, model.action_encoder]
    for m in train_mods:
        m.train()
    D = model.predictor.output_dim if hasattr(model.predictor, 'output_dim') else 192
    null_emb = torch.nn.Parameter(torch.zeros(D, device='cuda'))
    params = [q for m in train_mods for q in m.parameters()]
    if args.p_null > 0:
        params.append(null_emb)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))
    )

    log = []
    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in loader:
            pixels = batch['pixels'].cuda(non_blocking=True)
            action = torch.nan_to_num(batch['action'].cuda(non_blocking=True), 0.0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    emb = model.encode({'pixels': pixels})['emb']  # (B, 4, D), frozen
                act_emb = model.action_encoder(action)  # (B, 4, D)
                if args.p_null > 0:
                    drop = torch.rand(act_emb.shape[:2], device='cuda') < args.p_null
                    act_emb = torch.where(drop[..., None], null_emb.to(act_emb.dtype), act_emb)
                pred = model.predict(emb[:, :3], act_emb[:, :3])
                loss = (pred.float() - emb[:, 1:].float()).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                rec = {'step': step, 'loss': float(loss), 'sec': round(time.time() - t0, 1)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step >= args.steps:
                break

    model.eval()
    save_pretrained(model, run_name=args.run, config=config, filename='weights.pt')
    out = get_cache_dir(sub_folder='checkpoints') / args.run
    if args.p_null > 0:
        torch.save(null_emb.detach().float().cpu(), out / 'null_emb.pth')
    with open(out / 'train_log.json', 'w') as f:
        json.dump({'args': vars(args), 'log': log}, f, indent=1)
    print(f'saved to {out}', flush=True)


if __name__ == '__main__':
    main()
