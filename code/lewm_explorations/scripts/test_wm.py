import time, numpy as np, torch, h5py
from ctrlgeo.wm import FrozenLeWM, H5
from ctrlgeo import sim
wm = FrozenLeWM()
print('a_mean/std', wm.a_mean, wm.a_std)
f = h5py.File(H5,'r'); off=f['ep_offset'][:]; o=off[5]
px = f['pixels'][o:o+60]; act=f['action'][o:o+60]
e = sim.make_env(); e._set_state(f['state'][o]); r = e.render()
print('render vs dataset pixel absdiff mean', np.abs(r.astype(float)-px[0]).mean())
z = wm.encode(px)
u = torch.tensor(wm.norm_actions(act[:50]), device='cuda').float()[None]  # (1,10,10)
with torch.no_grad(): roll = wm.rollout(z[:1], u)
tgt = z[0:51:5]
err = (roll[0]-tgt).pow(2).sum(-1).sqrt().cpu().numpy(); base=(tgt-tgt[0]).pow(2).sum(-1).sqrt().cpu().numpy()
print('pred err per model step', err.round(2)); print('dist from start   ', base.round(2))
zz = z[:8:1]; uu = torch.randn(8,2,10,device='cuda')
t0=time.time(); J = wm.jacobian(zz, uu); torch.cuda.synchronize(); print('jacfwd', J.shape, time.time()-t0)
Jf = wm.jacobian_fd(zz, uu, 1e-3)
print('jac rel diff vs FD', ((J-Jf).norm()/J.norm()).item())
S = torch.linalg.svdvals(J[0]); print('sing vals', S.cpu().numpy().round(3))
N=4096; zb=z[:1].repeat(N,1); ub=torch.randn(N,5,10,device='cuda')
with torch.no_grad():
    wm.phi(zb,ub); torch.cuda.synchronize(); t0=time.time(); wm.phi(zb,ub); torch.cuda.synchronize()
print('transitions/s', N*5/(time.time()-t0))
