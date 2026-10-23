# Reach-then-change, Stage 1 oracle results (pod J, 2026-10-23, 07:46-11:30 UTC)

JEPA-WM encoder, exact simulator dynamics (sim_latent), the eval's CEM, 12 episodes per task (episodes 0-11, same for every planner).
Reach term and gate from simulator hand/object positions (`costs.ReachThenChangeOracle`, eps 2 mm, r0 2 cm); change term varies.
Per-episode success strings are in episode order.

| task | released (known) | released + reach | robot-free latent + reach (label-free) | true object cost alone | true + reach |
|---|---|---|---|---|---|
| button-press-topdown | 0/12 | 0/12 `............` | **12/12** `SSSSSSSSSSSS` (first success replan 3) | 0/12 `............` | 12/12 `SSSSSSSSSSSS` |
| push | 0/12 | 3/12 `.....SS.S...` | **7/12** `.SS...SSS.SS` | 11/12 `.SSSSSSSSSSS` | 12/12 |
| drawer-open | 0/12 | 1/12 `..........S.` | **6/12** `.....SS.SSSS` | 10/12 `SSSSS.SSSSS.` | 6/12 `SSSS..S...S.` |
| door-open | 0/12 | 0/12 | 0/12 | invalid (env reports obj_to_target = 0; success = \|handle x - target x\| <= 0.08) | invalid |
| reach-wall (fallback, no object change) | 23/24 | 12/12 | - | - | - |

Median final obj_to_target: button robot-free+reach 0.004 (threshold 0.024); push 0.034 (0.05); drawer-open 0.027 (0.03).
Released + reach fails after contact: the released cost drags the arm toward the goal arm pose once the object is nudged.
Pre-stated Stage 1 bar (>= 6/12 on >= 2 of 4 object tasks, fallback >= 10/12): met with the robot-free change term (3 of 4).

Stage 1 smoke history (drawer-open, 1 episode): eps 5 mm, dense reach -> hand parks 1.4 cm from the handle, drawer nudged 3-4 mm, never pulled;
after eps 2 mm + saturation at 2 cm -> true + reach opens the drawer (1/1); released + reach reaches the handle then drifts 24 cm away.

## Stage 2 (image-based cost) state when pod J dropped (12:00 UTC)
- Robot map (mask-supervised per-patch logistic head on DINOv2 features, 4 tasks, 24 episodes each): held-out AUC 0.998, IoU 0.94. Self-supervised proprio version: AUC 0.48 (useless).
- Change map = non-robot patches differing between start and goal (median + 4 MAD): drawer-open 24 patches, button 18 patches.
- Image-only reach (nearest robot patch to the change centroid): hand never approaches (the robot base is near the object in 2D). New-footprint variant: approaches slowly (0.39 -> 0.21 m over an episode), no press.
- Calibrated camera (corner2: pos (0.75, 0.075, 0.70), fovy 60, f = 112/tan 30 deg; row = H/2 + f py/(-pz), col = W/2 + f px/(-pz), p = mat^T (X - pos); validated: projected hand on a robot patch in 100% of frames; lift(project(X)) == X).
- Button's change patches sit 5 patches away from the button (arm shadows); the button itself changes too little. Fix written but not yet run: keep only change patches whose lifted point is within 0.3 m of the goal hand; if none, use the goal hand position.
- Patches to port into the repo: scratchpad podj_patch_smoke.sh, podj_patch_smoke2.sh, podj_patch_cam.sh, podj_patch_cam2.sh (exact string replacements on costs.py / run_closed.py); door-task true cost fix on adapters.py.
