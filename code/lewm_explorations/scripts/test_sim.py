import time, numpy as np, h5py
from ctrlgeo import sim
from ctrlgeo.paths import H5
e = sim.make_env()
f = h5py.File(H5, 'r')
off, L = f['ep_offset'][:], f['ep_len'][:]
st = f['state'][off[3]:off[3]+L[3]]; act = f['action'][off[3]:off[3]+L[3]]
e._set_state(st[0])
# replay expert actions, compare to dataset states
for t in range(40): sim.step_raw(e, act[t])
print('replay err vs dataset at t=40:', np.abs(sim.snap_to_state7(sim.snapshot(e))[:5]-st[40][:5]).max())
s0 = sim.snapshot(e); rng=np.random.default_rng(0); A = rng.normal(0,.3,(30,2))
def roll():
    sim.restore(e,s0)
    for a in A: sim.step_raw(e,a)
    return sim.snapshot(e)
r1=roll(); _=[sim.step_raw(e,a) for a in A]; r2=roll()
print('restore determinism err:', np.abs(r1-r2).max())
t0=time.time(); n=0
for k in range(20):
    sim.restore(e,s0)
    for a in A: sim.step_raw(e,a); n+=1
print('env steps/s:', n/(time.time()-t0))
t0=time.time(); [sim.render_state(e,s0) for _ in range(50)]; print('renders/s', 50/(time.time()-t0))
gaps=[]
for t in range(0,L[3]):
    e._set_state(st[t]); gaps.append(sim.agent_block_gap(e))
print('gap along ep:', np.round(gaps[::8],1)); print('agent radius', [s.radius for s in e.agent.shapes])
