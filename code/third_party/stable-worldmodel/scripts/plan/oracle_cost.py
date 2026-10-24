"""Oracle swaps for LeWM PushT planning (diagnosis D2, 2026-10-22).

Replaces the world model inside CEM with the PushT simulator, so a planning failure can be attributed:

    released     LeWM predicts, LeWM latent goal cost            (model + cost + search)
    sim_latent   simulator rolls out, LeWM latent goal cost       (cost + search)
    sim_true     simulator rolls out, true task distance           (search only)

released -> sim_latent removes model error; sim_latent -> sim_true removes cost error.

Each candidate is replayed from the env's current state in a worker process (one PushT env per worker).
The stored state has no block velocity (PushT._set_state zeroes it), so replays are a close, not exact,
copy of the real dynamics; check_fidelity() measures the gap.
"""

import multiprocessing as mp

import numpy as np
import torch
from torchvision import tv_tensors

from stable_worldmodel.planning.evaluator import ShootingCostEvaluator

_ENV = None


def _init_worker():
    global _ENV
    from stable_worldmodel.envs.pusht.env import PushT

    _ENV = PushT(resolution=224)
    _ENV.reset(seed=0)


def _rollout(args):
    """Replay raw action sequences from `state`; return final states (and final frames)."""
    state, goal_state, actions, render = args
    finals, frames = [], []
    for seq in actions:
        _ENV._set_state(state)
        _ENV._set_goal_state(goal_state)
        for a in seq:
            obs, *_ = _ENV.step(a)
        finals.append(obs['state'])
        if render:
            frames.append(_ENV.render())
    return np.array(finals), (np.array(frames) if render else None)


def _render_state(args):
    state, goal_state = args
    _ENV._set_state(state)
    _ENV._set_goal_state(goal_state)
    return _ENV.render()


def true_distance(goal, cur):
    """Distance aligned with PushT success: position error (agent + block) in units of the 20 px
    threshold plus block angle error in units of the pi/9 threshold (T block: no symmetry)."""
    pos = np.linalg.norm(goal[:4] - cur[:, :4], axis=1) / 20.0
    ang = np.abs(goal[4] - cur[:, 4]) % (2 * np.pi)
    ang = np.minimum(ang, 2 * np.pi - ang) / (np.pi / 9)
    return pos + ang


class OracleCost(ShootingCostEvaluator):
    def __init__(self, model, objective, mode, process, pixel_transform, workers=14, reach_r0=40.0, reach_eps=2.0):
        super().__init__(model, objective)
        assert mode in ('sim_latent', 'sim_true', 'sim_latent_reach', 'sim_true_reach')
        # *_reach (2026-10-24): the reach-then-change oracle of the Metaworld study on PushT. Candidates that moved the block
        # (> reach_eps px from its current position) are ranked by the inner cost; the others rank below all of them, by the
        # agent-to-block distance saturating below reach_r0 px (agent radius 15 + half the T's bar).
        self.reach = mode.endswith('_reach')
        self.mode = mode.replace('_reach', '')
        self.reach_r0, self.reach_eps = float(reach_r0), float(reach_eps)
        self.process = process
        self.pixel_transform = pixel_transform
        self.workers = workers
        self.pool = mp.get_context('spawn').Pool(workers, initializer=_init_worker)
        self.checked = False

    def _raw(self, key, info):
        v = info[key]
        v = v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
        v = v[0, 0].reshape(-1, v.shape[-1])[-1:]
        return self.process[key].inverse_transform(v)[0]

    def get_cost(self, info_dict, action_candidates):
        B, S, H, A = action_candidates.shape
        assert B == 1, 'oracle cost expects solver batch_size 1'
        state = self._raw('state', info_dict)
        goal = self._raw('goal_state', info_dict)
        cand = action_candidates[0].detach().float().cpu().numpy().reshape(S * H * A // 2, 2)
        raw = self.process['action'].inverse_transform(cand).reshape(S, H * A // 2, 2)
        render = self.mode == 'sim_latent'
        if not self.checked:  # the worker env must render the current state like the eval env does
            f = self.pool.apply(_render_state, ((state, goal),))
            mine = self.pixel_transform(tv_tensors.Image(torch.from_numpy(f).permute(2, 0, 1))).float()
            seen = info_dict['pixels'][0, 0].reshape(-1, *mine.shape)[-1].float().cpu()
            print(f'[oracle] mode={self.mode} render check: mean |worker - eval| = {float((mine - seen).abs().mean()):.4f}', flush=True)
            self.checked = True
        chunks = np.array_split(np.arange(S), self.workers)
        res = self.pool.map(_rollout, [(state, goal, raw[c], render) for c in chunks if len(c)])
        finals = np.concatenate([r[0] for r in res])

        def wrap(cost):  # reach-then-change on the inner cost (torch (S,))
            if not self.reach:
                return cost
            moved = torch.as_tensor(np.linalg.norm(finals[:, 2:4] - state[2:4], axis=1) > self.reach_eps, device=cost.device)
            reach = np.maximum(np.linalg.norm(finals[:, :2] - finals[:, 2:4], axis=1) - self.reach_r0, 0.0)
            reach = torch.as_tensor(reach, dtype=cost.dtype, device=cost.device)
            return torch.where(moved, cost, cost.max().detach() + reach)

        if self.mode == 'sim_true':
            cost = torch.as_tensor(true_distance(goal, finals), dtype=torch.float32, device=action_candidates.device)
            return wrap(cost)[None]
        frames = np.concatenate([r[1] for r in res])  # (S, 224, 224, 3) uint8
        px = torch.stack(
            [self.pixel_transform(tv_tensors.Image(torch.from_numpy(f).permute(2, 0, 1))) for f in frames]
        )
        dev = next(self.model.parameters()).device
        if 'goal_emb' not in info_dict:
            info_dict['goal_emb'] = self.encode_goal(self.model, info_dict)
        g = info_dict['goal_emb']
        g = g.reshape(-1, g.shape[-1])[-1]
        with torch.no_grad():
            emb = self.model.encode({'pixels': px.to(dev, next(self.model.parameters()).dtype)[:, None]})['emb'][:, -1]
        cost = (emb.float() - g.float()).pow(2).sum(-1)
        return wrap(cost)[None].to(action_candidates.device)


def check_fidelity(process, state, goal, raw_actions, real_final):
    """Replay one executed action sequence in a fresh env and compare with the real final state."""
    _init_worker()
    fin, _ = _rollout((state, goal, raw_actions[None], False))
    return np.abs(fin[0] - real_final)
