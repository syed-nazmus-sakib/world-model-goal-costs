"""Turn a Lightning checkpoint plus its training config.yaml (cfg.model is the instantiable model) into a whole-module
file that eval_wm.py loads with +policy_object=<out>. Used for the released DINO-WM PushT checkpoint
(dinowm_noprop_pusht_weights.ckpt).
usage (from code/third_party/stable-worldmodel): python convert_lightning_ckpt.py <config.yaml> <weights.ckpt> <out.pt>
"""
import sys

import hydra
import torch
from omegaconf import OmegaConf

import stable_worldmodel  # noqa: F401  (registers the classes the config targets)

cfg_path, ckpt_path, out = sys.argv[1:4]
if not OmegaConf.has_resolver('eval'):
    OmegaConf.register_new_resolver('eval', eval)
cfg = OmegaConf.load(cfg_path)
model = hydra.utils.instantiate(cfg.model)
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'missing {len(missing)} {missing[:6]}  unexpected {len(unexpected)} {unexpected[:6]}', flush=True)
if len(missing) > 0:
    sys.exit(f'refusing to save: {len(missing)} weights missing')
torch.save(model, out)
print('saved', out, flush=True)
