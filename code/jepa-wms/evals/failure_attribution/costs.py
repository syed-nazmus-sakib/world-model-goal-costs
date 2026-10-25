# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Area-invariant latent goal costs.

The released cost (ReprTargetDistMPCObjective) averages squared feature differences over all patches, so a
region of the image contributes in proportion to its area and small task-critical entities (the Push-T
pusher, a cube, a gripper) weigh little. Here the squared difference is averaged over features only, giving
a per-patch mismatch, which is then aggregated over patches without scaling with area:

    topk:K     mean of the K largest patch mismatches
    pmean:P    power mean (mean_p d_p^P)^(1/P)
    l2         the released aggregation (mean over patches), for reference
    head:PATH  method C (2026-10-22): ||f(z) - f(g)||^2 with a learned head f (mw_head_train.py) on the patch
               tokens and proprio embedding; no proprio weight

The proprio term is kept as in the released cost (alpha * mean squared proprio difference).
"""

from typing import Union

import numpy as np
import torch
from tensordict import TensorDict


def parse_cost(spec):
    """'l2', 'topk:8', 'pmean:4' -> (mode, param)"""
    mode, _, val = spec.partition(":")
    assert mode in ("l2", "topk", "pmean", "head"), spec
    if mode == "head":
        return mode, val
    return mode, (float(val) if val else None)


def patch_mismatch(target, enc):
    """(T, B, *patch_dims, D) -> (T, B, n_patches) mean squared feature difference per patch."""
    d = (target - enc).pow(2).mean(-1)
    return d.flatten(2)


def aggregate(d, mode, param):
    if mode == "l2":
        return d.mean(-1)
    if mode == "topk":
        k = min(int(param), d.shape[-1])
        return d.topk(k, dim=-1).values.mean(-1)
    if mode == "pmean":
        return d.pow(param).mean(-1).pow(1.0 / param)
    raise ValueError(mode)


class PatchAggMPCObjective:
    def __init__(self, cfg, target_enc: Union[torch.Tensor, TensorDict], cost: str = "topk:8", alpha: float = 0.1,
                 sum_all_diffs: bool = False, **kwargs):
        self.cfg, self.target_enc, self.alpha, self.sum_all_diffs = cfg, target_enc, alpha, sum_all_diffs
        self.mode, self.param = parse_cost(cost)

    def __call__(self, encodings, actions, keepdims: bool = False):
        if isinstance(encodings, TensorDict):
            diff = aggregate(patch_mismatch(self.target_enc["visual"], encodings["visual"]), self.mode, self.param)
            diff_p = (self.target_enc["proprio"] - encodings["proprio"]).pow(2)
            diff = diff + self.alpha * diff_p.mean(dim=tuple(range(2, diff_p.ndim)))
        else:
            diff = aggregate(patch_mismatch(self.target_enc, encodings), self.mode, self.param)
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff.cumsum(0).flip(0) if self.sum_all_diffs else diff


class ReachThenChangeOracle:
    """Stage-1 oracle of the reach-then-change cost (2026-10-23): simulator states give the reach term and the gate.

    A Metaworld reward is written as "bring the hand to the object, then bring the object to its target". This cost
    imposes that order on any goal cost `inner` (the change term: the released latent cost, or the true object cost):
      - candidates that have moved the object (|obj - obj_start| > eps) are ranked by `inner`;
      - candidates that have not are ranked below all of them, by the hand-to-object distance (the reach term),
        which saturates below r0: every candidate with the hand within r0 of the object ties, so the search keeps its
        spread there and can find the grasp-and-pull (a dense reach term collapsed the search onto "hold still at
        the handle" in the first smoke episode: hand 1.4 cm from the drawer handle, drawer nudged 3-4 mm, never pulled);
      - if the goal moves no object (|obj_goal - obj_start| <= eps, e.g. reach-wall), `inner` is used unchanged.
    States (B, 41) = [39-d obs, success, obj_to_target] come from the SimUnroll that produced the encodings
    (unroll.last_states): hand = obs[:3], object = obs[4:7]. The image-based method replaces both with maps
    computed from the frozen encoder; this oracle tells whether the structure itself solves the approach phase.
    Defaults (2026-10-23, set after that smoke episode, before the 12-episode runs): eps 2 mm, r0 2 cm.
    """

    def __init__(self, inner, unroll, obj_start, obj_goal, eps: float = 0.002, r0: float = 0.02):
        self.inner, self.unroll, self.eps, self.r0 = inner, unroll, eps, r0
        self.obj_start = np.asarray(obj_start, dtype=np.float64)
        self.active = bool(np.linalg.norm(np.asarray(obj_goal, dtype=np.float64) - self.obj_start) > max(eps, 0.005))

    def __call__(self, encodings, actions, keepdims: bool = False):
        c = self.inner(encodings, actions, keepdims)
        if not self.active or keepdims:
            return c
        st = np.asarray(self.unroll.last_states, dtype=np.float64)
        moved = torch.from_numpy(np.linalg.norm(st[:, 4:7] - self.obj_start, axis=1) > self.eps).to(c.device)
        reach = np.maximum(np.linalg.norm(st[:, :3] - st[:, 4:7], axis=1) - self.r0, 0.0)
        reach = torch.from_numpy(reach).to(c.device, c.dtype)
        return torch.where(moved, c, c.max().detach() + reach)  # every non-mover ranks below every mover


class ReachThenChangeImage:
    """Stage 2 (2026-10-23): the reach-then-change cost from images, no task labels, no simulator state.

    From the episode's start encoding z0 and the goal encoding g (set_start / __init__), on the 16x16 patch grid:
      robot map   r_p = P(patch p shows the robot), from mw_robot_map.PatchRobotHead on the frozen patch features
      change map  c_p = 1 for patches that are not robot in z0 nor in g and differ between them by more than a robust
                  threshold (median + 4 MAD of the non-robot patch differences, and at least 1% of the largest one):
                  what the task has to change. Empty change map (reach-wall): fall back to the released cost.
    Per candidate z (its predicted or simulated final frame):
      change term  mean over change patches of ||z_p - g_p||^2 (the released patch cost restricted to the change region)
      moved        the change region has moved away from the start: mean_c ||z_p - z0_p||^2 > eps_c, eps_c = 10% of the
                   start-to-goal change on those patches
      reach term   grid distance from the nearest robot patch of z to the change region's centroid, saturating below r0
      cost = change term for movers; non-movers rank below every mover, by the reach term.
    In 'proprio' robot-map mode the hand position comes from the robot's own sensor (unroll.last_states[:, :3]).
    """

    def __init__(self, cfg, target_enc: TensorDict, head_path: str, unroll=None, r0: float = 1.0, mad_k: float = 4.0, **kwargs):
        from evals.failure_attribution.mw_robot_map import PatchRobotHead
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

        ck = torch.load(head_path, map_location="cpu")
        self.head = PatchRobotHead(ck["d_in"], ck["mode"])
        self.head.load_state_dict(ck["state_dict"])
        self.device = target_enc["visual"].device
        self.head = self.head.to(self.device).eval().requires_grad_(False)
        self.unroll, self.r0, self.mad_k = unroll, r0, mad_k
        self.rf = bool(kwargs.get("rf", False))  # robot-free-mask change term (whole image minus the robot in goal and candidate)
        # v2 (2026-10-24): (a) a detected object cluster is trusted only within `radius_gate` of the goal hand (an object's start
        # is near where the hand ends; 6 of 9 false button detections were farther), (b) closed-loop verification: if the hand
        # has been at the detected target for `verify_replans` replans and the change term has not moved 10% toward the goal,
        # the detection was wrong and the target falls back to the goal hand (static object).
        self.radius_gate = float(kwargs.get("radius_gate", 0.0))  # 0 = off
        self.verify_replans = int(kwargs.get("verify_replans", 0))  # 0 = off
        # v3b (2026-10-24, from the v3 12-episode button run): (a) change patches next to the start arm (one-patch dilation of the
        # start robot mask) are its shadow / vacated region, never the object: 6 of 12 button episodes detected exactly that;
        # (b) an object cluster must change the features at least `rel_min` times as much as the robot's own appearance change
        # (start-arm shadows measure 0.2-0.6 x, objects 0.9-1.8 x); (c) in the goal-hand fallback the stand-off is 6 cm and the
        # second phase is the distance to the goal proprio (hand and gripper), so the gripper closes like the expert's.
        self.v3b = bool(kwargs.get("v3b", False))
        self.rel_min = float(kwargs.get("rel_min", 0.5))  # 0.5: keeps faucet's top patches (p90 0.89), ~1 shadow patch in 5 passes
        # (d) the far cluster (object at its start) is lifted at table height when the goal hand is low (the hand hovers 4-7 cm
        # above an object it pushes along the table: v3 push / plate-slide estimates were 5-14 cm toward the camera, 1.5 cm of
        # horizontal error per cm of height), and at goal-hand height otherwise (a raised handle is grasped at its own height)
        self.table_z = float(kwargs.get("table_z", 0.02))
        self.low_hand = float(kwargs.get("low_hand", 0.10))
        self.gp = None
        self._at_target, self._fallback_done = 0, False
        self.released = ReprTargetDistMPCObjective(cfg, target_enc=target_enc, alpha=cfg.planner.planning_objective.alpha)
        self.g = self._flat(target_enc)[0]  # (P, D)
        P = self.g.shape[0]
        side = int(round(P ** 0.5))
        yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        self.grid = torch.stack([yy, xx], -1).reshape(P, 2).float().to(self.device)
        self.active, self.info = False, {}

    @staticmethod
    def _flat(enc):
        """visual (*lead, V, H, W, D) -> (n, V*H*W, D); the grid below assumes V = 1 (one frame per encoding)"""
        vis = enc["visual"]
        return vis.reshape(int(np.prod(vis.shape[:-4])), -1, vis.shape[-1]).float()

    def _hand(self, n):
        if self.head.mode != "proprio":
            return None
        return torch.from_numpy(np.asarray(self.unroll.last_states)[:, :3]).to(self.device).float()[:n]

    def set_camera(self, pos, mat, fovy, img_size, patch=14):
        """Calibrated render camera (MuJoCo corner2): pos (3,), mat (3, 3) world<-camera, fovy in degrees. With it the reach
        term is the 3D hand-to-object distance: the hand from the robot's own sensor, the object as the change-map centroid
        lifted to the goal's hand height. Convention validated on the robot masks (projected hand on a robot patch in 100%
        of training frames): x = W/2 + f px/(-pz), row = H/2 + f py/(-pz) with p = mat^T (X - pos)."""
        self.cam = {"pos": np.asarray(pos, dtype=np.float64), "mat": np.asarray(mat, dtype=np.float64).reshape(3, 3),
                    "f": (img_size / 2) / np.tan(np.deg2rad(fovy) / 2), "hw": img_size / 2, "patch": patch}

    def project(self, X):
        """world point (3,) -> patch-grid (row, col) floats (inverse of lift)"""
        c = self.cam
        q = c["mat"].T @ (np.asarray(X, dtype=np.float64) - c["pos"])
        return np.array([c["hw"] + c["f"] * q[1] / -q[2], c["hw"] + c["f"] * q[0] / -q[2]]) / c["patch"] - 0.5

    def lift(self, row_col, z):
        """patch-grid (row, col) -> the 3D point on the camera ray through that patch centre at height z (world)."""
        c = self.cam
        row, col = (np.asarray(row_col, dtype=np.float64) + 0.5) * c["patch"]
        d = c["mat"] @ np.array([(col - c["hw"]) / c["f"], (row - c["hw"]) / c["f"], -1.0])
        t = (z - c["pos"][2]) / d[2] if abs(d[2]) > 1e-6 else 0.0
        return c["pos"] + max(t, 0.0) * d

    @torch.no_grad()
    def set_start(self, z0_enc, hand0=None, goal_hand_z=None, goal_hand=None, goal_prop=None):
        self.gp = None if goal_prop is None else np.asarray(goal_prop, dtype=np.float64).reshape(-1)[:4]
        z0 = self._flat(z0_enc)[0]
        r0 = self.head.robot_prob(z0[None], None if hand0 is None else hand0[None].to(self.device).float())[0]
        rg = self.head.robot_prob(self.g[None], None if hand0 is None else hand0[None].to(self.device).float())[0]
        nonrobot = (r0 < 0.5) & (rg < 0.5)
        diff = (z0 - self.g).pow(2).sum(-1)
        d = diff[nonrobot]
        med = d.median()
        mad = (d - med).abs().median() + 1e-8
        change = nonrobot & (diff > med + self.mad_k * mad) & (diff > 0.01 * diff.max())
        self.z0, self.change, self.robot0, self.disc = z0, change, r0 > 0.5, False
        self.active = bool(change.sum() >= 1)
        self.info = {"n_change": int(change.sum()), "n_robot_start": int((r0 > 0.5).sum()), "n_robot_goal": int((rg > 0.5).sum()),
                     "active": self.active}
        self.obj3d = None
        if self.active:
            self.cent = self.grid[change].mean(0)
            self.eps_c = 0.1 * (z0[change] - self.g[change]).pow(2).sum(-1).mean()
            self.info["change_centroid"] = self.cent.tolist()
            if getattr(self, "cam", None) is not None and goal_hand_z is not None:
                idx = change.nonzero().squeeze(-1)
                pts = np.stack([self.lift(rc, float(goal_hand_z)) for rc in self.grid[idx].cpu().numpy()])  # (n_change, 3)
                gh = np.asarray(goal_hand, dtype=np.float64) if goal_hand is not None else None
                h0 = hand0.cpu().numpy().astype(np.float64) if hand0 is not None else None
                # workspace bounds (Sawyer table): lifted change points outside them are not objects (arm shadows on walls)
                inb = (np.abs(pts[:, 0]) < 0.6) & (pts[:, 1] > 0.3) & (pts[:, 1] < 1.1) & (pts[:, 2] > -0.05) & (pts[:, 2] < 0.5)
                if h0 is not None:
                    inb &= np.linalg.norm(pts - h0, axis=1) < 0.9
                self.info["n_change_in_workspace"] = int(inb.sum())
                if self.v3b:  # object patches: in the workspace, not next to the start arm, strong relative to the robot's own change
                    arm_app = rg & ~(r0 > 0.5)  # patches where the arm appears over background in the goal image
                    robot_scale = float(diff[arm_app].mean()) if arm_app.any() else float(diff[rg].mean())
                    rel = (diff[idx] / max(robot_scale, 1e-8)).cpu().numpy()
                    r0_pts = self.grid[r0 > 0.5]
                    near_r0 = (torch.cdist(self.grid[idx], r0_pts).min(-1).values <= 1.5).cpu().numpy() if len(r0_pts) else np.zeros(len(idx), bool)
                    inb = inb & (rel >= self.rel_min) & ~near_r0
                    self.info.update(robot_scale=round(robot_scale, 4), n_change_strong=int(inb.sum()),
                                     n_change_next_to_start_arm=int(near_r0.sum()), rel_strength=[round(float(v), 2) for v in sorted(rel)[-6:]])
                if self.radius_gate > 0 and gh is not None and inb.sum() >= 3:
                    far0 = inb & (np.linalg.norm(pts - gh, axis=1) > 0.10)
                    cand = pts[far0].mean(0) if far0.sum() >= 3 else pts[inb].mean(0)
                    if np.linalg.norm(cand - gh) > self.radius_gate:
                        self.info["detection_rejected_by_radius"] = round(float(np.linalg.norm(cand - gh)), 3)
                if inb.sum() >= 3 and not self.info.get("detection_rejected_by_radius"):  # a real detection
                    keep = torch.zeros_like(change)
                    keep[idx[torch.from_numpy(inb).to(idx.device)]] = True
                    if gh is not None:  # plus the goal's contact disc: the object's final state sits under the hand there
                        keep |= (self.grid - torch.from_numpy(self.project(gh)).to(self.device).float()).norm(dim=-1) <= 1.5
                    self.change = change = keep
                    far = inb & (np.linalg.norm(pts - gh, axis=1) > 0.10) if gh is not None else np.zeros_like(inb)
                    # an object that travels leaves a change cluster away from the goal hand: that is where it is now
                    obj = pts[far].mean(0) if far.sum() >= 3 else pts[inb].mean(0)
                    if self.v3b and far.sum() >= 3:  # (d) lift the object at its own height, not the hand's
                        lift_z = self.table_z if float(goal_hand_z) < self.low_hand else float(goal_hand_z)
                        obj = np.stack([self.lift(rc, lift_z) for rc in self.grid[idx[torch.from_numpy(far).to(idx.device)]].cpu().numpy()]).mean(0)
                        self.info["far_cluster_lift_z"] = round(lift_z, 3)
                    self.info["n_change_far_from_goal_hand"] = int(far.sum())
                    self.info["n_change_region"] = int(change.sum())
                    self.eps_c = 0.1 * (z0[change] - self.g[change]).pow(2).sum(-1).mean()
                else:  # nothing detected at the object (its own change is too small, e.g. a button): the object is where the
                    # goal hand presses, 5 cm above it; the change region is the patch disc around that point (goal: object in
                    # its final state under the hand)
                    obj = gh + np.array([0.0, 0.0, 0.05])
                    rc = torch.from_numpy(self.project(obj)).to(self.device).float()
                    change = (self.grid - rc).norm(dim=-1) <= 1.5
                    self.change, self.disc = change, True
                    self.eps_c = 0.1 * (z0[change] - self.g[change]).pow(2).sum(-1).mean()
                    self.info["change_disc_patches"] = int(change.sum())
                self.obj3d = torch.from_numpy(obj).to(self.device).float()
                self.info["object_3d_estimate"] = [round(float(v), 3) for v in obj]
            self.gh = None if goal_hand is None else np.asarray(goal_hand, dtype=np.float64)
            self._at_target, self._fallback_done = 0, False
            self.c0 = (z0[self.change] - self.g[self.change]).pow(2).sum(-1).mean()  # the start's change term
            self.rg = rg > 0.5
            if self.rf:
                m = (~self.robot0) & (~self.rg)
                self.c0 = ((z0 - self.g).pow(2).sum(-1) * m.float()).sum() / m.float().sum().clamp_min(1.0)

    @torch.no_grad()
    def update(self, z_now_enc, hand_now):
        """Closed-loop verification of the detected target (v2). Called once per replan with the current observation."""
        if not (self.active and self.verify_replans > 0 and self.obj3d is not None and self.gh is not None) or self._fallback_done:
            return
        if self.info.get("change_disc_patches"):  # already on the goal-hand fallback
            return
        hand = np.asarray(hand_now, dtype=np.float64)
        rn = 0.04 if self.rf else 0.06
        z = self._flat(z_now_enc)[0]
        r = self.head.robot_prob(z[None])[0] > 0.5
        m = ((~r) & (~self.rg) & (~self.robot0)).float()
        now = ((z - self.g).pow(2).sum(-1) * m).sum() / m.sum().clamp_min(1.0)
        c0 = ((self.z0 - self.g).pow(2).sum(-1) * m).sum() / m.sum().clamp_min(1.0)
        at_target = np.linalg.norm(hand - self.obj3d.cpu().numpy()) < rn + 0.03
        self._at_target = self._at_target + 1 if at_target else 0
        if self._at_target >= self.verify_replans and now > 0.9 * c0:  # reached the target, nothing changed: wrong detection
            obj = self.gh + np.array([0.0, 0.0, 0.05])
            rc = torch.from_numpy(self.project(obj)).to(self.device).float()
            self.change = (self.grid - rc).norm(dim=-1) <= 1.5
            self.disc, self.obj3d, self._fallback_done = True, torch.from_numpy(obj).to(self.device).float(), True
            self.info["verification_fallback_at_replan"] = int(self.info.get("replans", 0))
            print(f"  [reach-img] target verification failed after {self._at_target} replans at the target: falling back to the goal hand", flush=True)
        self.info["replans"] = int(self.info.get("replans", 0)) + 1

    def __call__(self, encodings, actions, keepdims: bool = False):
        if not self.active or keepdims:
            return self.released(encodings, actions, keepdims)
        z = self._flat(encodings)  # (B, P, D)
        c = self.change
        robot = self.head.robot_prob(z, self._hand(len(z))) > 0.5  # (B, P)
        change_term = (z[:, c] - self.g[c]).pow(2).sum(-1).mean(-1)
        if self.rf:  # robot in neither image: the oracle's robot-free cost, with the regions under the arm unknown
            m = ((~robot) & (~self.rg[None]) & (~self.robot0[None])).float()  # robot in no image: start, goal, candidate
            change_term = ((z - self.g[None]).pow(2).sum(-1) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
            c0_b = ((self.z0 - self.g)[None].pow(2).sum(-1) * m).sum(-1) / m.sum(-1).clamp_min(1.0)  # start, same mask
        # moved: judged on the change patches the robot does not cover in the candidate; all covered = the arm is on the object
        visible = (~robot)[:, c].float()
        d0 = (z[:, c] - self.z0[c]).pow(2).sum(-1)
        d0v = (d0 * visible).sum(-1) / visible.sum(-1).clamp_min(1.0)
        # a candidate has moved the object if its visible change patches departed from the start by more than the absolute
        # floor eps_c AND stand out from the other candidates (3x the batch median: arm shadows and render variation move
        # every candidate a little, a real object motion moves a few a lot)
        thr = torch.maximum(self.eps_c, 3.0 * d0v.median())
        # progress gate: the change region is at least 10% closer to the goal's appearance than it was at the start
        moved = (change_term < 0.9 * self.c0) | (d0v > thr) | (visible.sum(-1) == 0)
        if self.rf and self.obj3d is None:
            moved = change_term < 0.9 * c0_b
        if self.disc:  # disc region (object change too small to detect): the hand covering half the disc = on the object
            moved = moved | (robot[:, c].float().mean(-1) >= 0.5)
        # reach: the arm's new footprint (robot in the candidate, not robot at the episode start), so the static base cancels
        # and the nearest new patch to the change region tracks the moving arm (2026-10-23: the nearest *robot* patch was
        # always the base near the top of the image, saturated from the start, and the hand never approached)
        new = robot & ~self.robot0[None]
        if self.obj3d is not None:  # calibrated camera: 3D distance from the hand (own sensor) to the lifted change centroid
            hand = torch.from_numpy(np.asarray(self.unroll.last_states)[:len(z), :3]).to(self.device).float()
            raw = (hand - self.obj3d).norm(dim=-1)
            rn = 0.04 if self.rf else 0.06  # stand-off at the object estimate (4 cm: within one planning step of the object)
            if self.v3b and self.disc:
                rn = 0.06  # the fallback point is 5 cm above the goal hand: 4 cm never engaged the second phase (v3 smoke ep 1)
            reach = (raw - rn).clamp_min(0.0)
            near = raw < rn  # the hand is at the object: the change term decides from here on
            moved = near if self.rf else (moved | near)  # image progress gates are fooled by arm shadows: rf uses geometry only
            if self.disc and self.verify_replans > 0 and getattr(self, "gh", None) is not None:
                # v3: in the goal-hand fallback nothing but the robot's contact region changes between start and goal, so the goal
                # hand pose is the task; the second phase is the distance to it (the latent term on the contact disc barely
                # separates "pressed" from "touching": v2 smoke 0/3 on button after reaching it)
                change_term = (hand - torch.from_numpy(self.gh).to(self.device).float()).norm(dim=-1)
                if self.v3b and self.gp is not None:
                    # v3b: distance to the goal proprio, hand and gripper (gripper opening in [0, 1] weighted as 5 cm): in v3, 3 of the
                    # 6 button episodes that reached the goal hand to 1-2 cm hit the housing instead of the button
                    prop = torch.from_numpy(np.asarray(self.unroll.last_states)[:len(z), :4]).to(self.device).float()
                    w = torch.tensor([1.0, 1.0, 1.0, 0.05], device=self.device)
                    change_term = ((prop - torch.from_numpy(self.gp).to(self.device).float()) * w).norm(dim=-1)
        else:  # image only: the arm's new footprint on the patch grid
            dist = (self.grid[None] - self.cent).norm(dim=-1).expand(len(z), -1)
            reach = torch.where(new, dist, torch.full_like(dist, float(self.grid.shape[0]))).min(-1).values
            reach = (reach - self.r0).clamp_min(0.0)
        if not getattr(self, "_dbg", False):  # one line per episode, to check that the two phases behave
            self._dbg = True
            print(f"  [reach-img] B={len(z)} moved={moved.float().mean():.2f} eps_c={float(self.eps_c):.3g} thr={float(thr):.3g} "
                  f"d0v med={float(d0v.median()):.3g} max={float(d0v.max()):.3g} | reach min={float(reach.min()):.1f} "
                  f"med={float(reach.median()):.1f} new-footprint patches med={float(new.float().sum(-1).median()):.0f}", flush=True)
        return torch.where(moved, change_term, change_term.max().detach() + reach)


class TempoMixObjective:
    """TEMPO baseline (arXiv 2610.04988) inside our planner: J = (1 - w) * own + w * d_H^2 * ||F(z) - F(G)||^2, with the
    released patch cost as `own`, F from mw_tempo_train.py on [patch-mean visual, proprio] and d_H^2 from its checkpoint.
    w = 0 is the released planner; w = 1 the pure temporal map."""

    def __init__(self, cfg, target_enc: TensorDict, map_path: str, w: float = 0.5, **kwargs):
        from evals.failure_attribution.mw_tempo_train import TemporalMap
        from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective

        ck = torch.load(map_path, map_location="cpu")
        self.fmap = TemporalMap(ck["d_in"], kappa=ck["kappa"]).to(target_enc["visual"].device).eval().requires_grad_(False)
        self.fmap.load_state_dict(ck["state_dict"])
        self.w, self.scale = float(w), float(ck["d_H2"])
        self.own = ReprTargetDistMPCObjective(cfg, target_enc=target_enc, alpha=cfg.planner.planning_objective.alpha)
        self.fg = self.embed(target_enc)[0]

    @torch.no_grad()
    def embed(self, enc):
        vis, prop = enc["visual"], enc["proprio"]
        n = int(np.prod(vis.shape[:-4]))
        x = torch.cat([vis.reshape(n, -1, vis.shape[-1]).float().mean(1), prop.reshape(n, -1).float()], -1)
        return self.fmap(x)

    def __call__(self, encodings, actions, keepdims: bool = False):
        own = self.own(encodings, actions, keepdims)
        if keepdims:
            return own
        tmp = (self.embed(encodings) - self.fg).pow(2).sum(-1)
        return (1.0 - self.w) * own + self.w * self.scale * tmp


class LearnedHeadMPCObjective:
    """Method C: squared distance between learned embeddings of the candidate and the goal (see mw_head_train.py)."""

    def __init__(self, cfg, target_enc: TensorDict, head_path: str, sum_all_diffs: bool = False, beta: float = 0.0, **kwargs):
        """head_path: mw_head_train / mw_pilot_heads checkpoint; with several members ('state_dicts', mw_pilot_heads --n_ens)
        the cost is mean_k + beta * std_k of the member costs (pessimism where the members disagree, 2026-10-24)."""
        from evals.failure_attribution.mw_head_train import PatchHead

        ck = torch.load(head_path, map_location="cpu")
        sds = ck.get("state_dicts", [ck["state_dict"]])
        members = []
        for sd in sds:
            h = PatchHead(ck["n_patches"], ck["d_in"], ck["d_prop"])
            h.load_state_dict(sd)
            members.append(h)
        self.head = torch.nn.ModuleList(members).to(target_enc["visual"].device).eval().requires_grad_(False)
        self.cfg, self.sum_all_diffs, self.beta = cfg, sum_all_diffs, float(beta)
        self.f_goal = self.embed(target_enc)  # (E, T_goal, B_goal, d)

    @torch.no_grad()
    def embed(self, enc):
        """visual (*lead, V, H, W, D) + proprio (*lead, Dp) -> (E, *lead, d)"""
        vis, prop = enc["visual"], enc["proprio"]
        lead = vis.shape[:-4]
        n = int(np.prod(lead))
        v, p = vis.reshape(n, -1, vis.shape[-1]).float(), prop.reshape(n, -1).float()
        return torch.stack([h(v, p).view(*lead, -1) for h in self.head])

    def __call__(self, encodings, actions, keepdims: bool = False):
        self.last_enc = encodings  # the population the planner evaluated last (run_closed --dump_dir saves a sample of it)
        if encodings["visual"].dim() == 6 and encodings["visual"].shape[1] > 1:
            self.last_pop = (encodings, actions)  # last CEM population (not the final single mean rollout): predicted latents + actions
        d = (self.embed(encodings) - self.f_goal).pow(2).sum(-1)  # (E, T, B)
        diff = d.mean(0)
        if self.beta and d.shape[0] > 1:
            diff = diff + self.beta * d.std(0, unbiased=False)
        if not keepdims:
            return diff.sum(0) if self.sum_all_diffs else diff[-1]
        return diff.cumsum(0).flip(0) if self.sum_all_diffs else diff


class HeadChangeObjective:
    """Approach-aware head + robot-free change term (2026-10-24, from the H3 closed loops).

    The ensemble rtc head (walks + pessimism) brought the hand to the object in all 24 button and push episodes but, like the
    released cost, is arm-dominated once there: button b1 ep 006 ends 1 cm from the goal hand with the button unpressed at
    cost 0.031, below the 0.047 of a pressed state (ep 002); push ends with the hand at the goal hand and the puck 5-10 cm
    off. The change term of the reach-then-change oracle (Stage 1: button 12/12, push 7/12 with oracle reach) resolves the
    object state: squared feature difference to the goal over the patches that show the robot in neither the start, the goal
    nor the candidate frame (PatchRobotHead mask), divided by the same quantity at the episode start (same mask), so it runs
    from ~1 (nothing changed) to ~0 (goal appearance).

        cost(z) = head(z) + lam * rf(z) / rf(z0)

    The head dominates the approach phase (0.3-8 against lam * ~1) and the change term decides where the head is flat (< 0.1
    near the goal hand). No detection, no localisation and no switch: the two terms are simply added."""

    def __init__(self, cfg, target_enc: TensorDict, head_path: str, robot_path: str, beta: float = 1.0, lam: float = 0.5, **kwargs):
        from evals.failure_attribution.mw_robot_map import PatchRobotHead

        self.head = LearnedHeadMPCObjective(cfg, target_enc, head_path, beta=beta)
        ck = torch.load(robot_path, map_location="cpu")
        self.rhead = PatchRobotHead(ck["d_in"], ck["mode"])
        self.rhead.load_state_dict(ck["state_dict"])
        self.device = target_enc["visual"].device
        self.rhead = self.rhead.to(self.device).eval().requires_grad_(False)
        self.lam = float(lam)
        self.g = ReachThenChangeImage._flat(target_enc)[0]  # (P, D)
        self.rg = self.rhead.robot_prob(self.g[None])[0] > 0.5
        self.z0, self.robot0, self.info, self._dbg = None, None, {}, False

    def set_camera(self, *args, **kwargs):  # no geometry here (run_closed calls it before set_start)
        pass

    @torch.no_grad()
    def set_start(self, z0_enc, hand0=None, **kwargs):
        self.z0 = ReachThenChangeImage._flat(z0_enc)[0]
        self.robot0 = self.rhead.robot_prob(self.z0[None])[0] > 0.5
        m = ~(self.rg | self.robot0)
        self.info = {"n_robot_start": int(self.robot0.sum()), "n_robot_goal": int(self.rg.sum()),
                     "rf0": float((self.z0 - self.g).pow(2).sum(-1)[m].mean()) if m.any() else 0.0, "lam": self.lam}
        self._dbg = False

    @torch.no_grad()
    def rf(self, z):
        """(B, P, D) -> (B,): robot-free squared difference to the goal, relative to the start under the same mask"""
        robot = self.rhead.robot_prob(z) > 0.5
        m = ((~robot) & (~self.rg[None]) & (~self.robot0[None])).float()
        cur = ((z - self.g[None]).pow(2).sum(-1) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        ref = ((self.z0 - self.g)[None].pow(2).sum(-1) * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        return cur / ref.clamp_min(1e-8)

    def __call__(self, encodings, actions, keepdims: bool = False):
        self.last_enc = encodings
        if encodings["visual"].dim() == 6 and encodings["visual"].shape[1] > 1:
            self.last_pop = (encodings, actions)  # see LearnedHeadMPCObjective: run_closed --planners wm --dump_dir
        h = self.head(encodings, actions, keepdims)
        if keepdims or self.robot0 is None:
            return h
        vis = encodings["visual"]
        last = vis[-1] if vis.dim() == 6 else vis  # (T, B, V, H, W, D) from the predictor: the change term on the final frame
        z = last.reshape(last.shape[0], -1, last.shape[-1]).float()
        if len(z) != h.shape[-1]:
            return h
        r = self.rf(z)
        if not self._dbg:
            self._dbg = True
            print(f"  [head+rf] B={len(z)} head min={float(h.min()):.3g} med={float(h.median()):.3g} max={float(h.max()):.3g} | "
                  f"rf min={float(r.min()):.3g} med={float(r.median()):.3g} max={float(r.max()):.3g}", flush=True)
        return h + self.lam * r
