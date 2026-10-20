# Handoff: RunPod pod, what is running, what comes next

> **Update 2026-10-16:** there are now two pods (A = the one described below, B = `underground_coffee_hummingbird`). Access details, prices and what runs where are in `09_PODS_ACCESS.md`; new results are in `07_RESTART_LOG.md`.

Written 2026-10-15 (pod clock about 17:30 UTC). Read this first, then `07_RESTART_LOG.md` (findings and decision rules) and
`06_LITERATURE_REVIEW_v2.md` (novelty map). Older docs (`01`-`05`) contain facts that were corrected later; see section 1 of `06`.

Goal: a CVPR 2027 paper (registration 2026-11-10, paper 2026-11-16, supplementary 2026-11-23, AoE). The user wants a credible paper with a real
idea, method and novelty, not a borderline one. Do not suggest other venues.

## 0. Rules that apply to every agent
- **Never write secrets to disk or to these docs**: no passwords (the user pasted a sudo password into the chat; it is not needed and must not be
  stored), no RunPod account login or API key. SSH uses a key pair only.
- Ask before spending money in a new way (a bigger GPU, a second pod). The user has about **$24 of RunPod credit** in total.
- Stop the pod when it is idle. Back results up to the Mac (section 7) before stopping or resetting the pod.
- Report negative results plainly with numbers. Kill directions by the pre-stated tests in `07_RESTART_LOG.md`.

## 1. The pod
| Item | Value |
|---|---|
| Name | `due_amber_cardinal` (RunPod pod id `i9ayugl1jd6ic3`) |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24 GB (sm_120), driver 580 |
| CPU | host shows 48 cores but the container cgroup caps it at about 10.2 cores (`/sys/fs/cgroup/cpu.max`) |
| RAM / disk | 125 GB visible, container disk 50 GB (about 15 GB used); there is no persistent `/workspace` volume |
| Price | about $0.57/hr (listed "RTX PRO 4000") |
| Direct SSH | `ssh -i ~/.ssh/id_ed25519 root@213.173.104.47 -p 22324` (IP and port change if the pod is restarted: read them from the pod's Connect tab) |
| Proxy SSH | `ssh -tt -i ~/.ssh/id_ed25519 i9ayugl1jd6ic3-64410cda@ssh.runpod.io` (needs a PTY; no scp or sftp) |

The Mac's public key (`~/.ssh/id_ed25519.pub`, an ed25519 key) is installed in the container's `/root/.ssh/authorized_keys`. The RunPod
account itself had no SSH key configured. If direct SSH says `Permission denied (publickey)` after a pod restart, connect through the proxy
with a PTY, then run:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '<contents of the Mac id_ed25519.pub>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
service ssh start
```

Driving the proxy non-interactively works by feeding commands on stdin: `(printf 'cmd\nexit\n'; sleep 15) | ssh -tt ... i9ayugl1jd6ic3-...@ssh.runpod.io`.
Use direct SSH for everything else (rsync works over it).

The first time a host key prompt appears use `-o StrictHostKeyChecking=accept-new`; do not disable checking globally.

## 2. Pod layout
```
/root/miniconda3                       conda; env `jepa-wms` (python 3.10, torch 2.7.0+cu128)
/root/jepa-wms-project/                copy of the project (same layout as the Mac folder)
    code/jepa-wms/                     repo with our evals; macros.py rewritten to /root/jepa-wms-project
    configs/*.yaml                     absolute paths rewritten to /root/jepa-wms-project
    artifacts/checkpoints/             only mw_jepa-wm.pth.tar and mw_dino-wm.pth.tar (Push-T and Wall are NOT uploaded)
    artifacts/datasets/Metaworld/      705 MB (Push-T 7 GB and Wall 55 GB are not uploaded)
    artifacts/raw_logs/failure_attribution/{drawer_alpha0,dino}/   all runs of this restart
    env/env.sh                         activates the env, sets paths, and limits BLAS threads (see pitfalls)
/root/run_*.sh, /root/mkenv.sh         launch scripts (copies in scripts/pod/ on the Mac)
```
Configs added on the pod: `configs/mw_dino-wm_drawer-close.yaml` and `configs/mw_dino-wm_push.yaml` (copies of `mw_dino-wm.yaml` with the
task line changed). They are not yet in the Mac folder.

## 3. How to run things
```bash
ssh -i ~/.ssh/id_ed25519 root@213.173.104.47 -p 22324
cd /root/jepa-wms-project && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
python -m tests.test_fast_rollout_mw --tasks mw-push --trials 2          # must end with OK, 0.0 / 0 diffs
```
Closed-loop drawer-close (exact simulator dynamics, JEPA-WM encoder), one planner, resumable (re-run with a larger `--episodes` to extend):
```bash
R=/root/jepa-wms-project; D=$R/artifacts/raw_logs/failure_attribution/<new_dir>
python -m evals.failure_attribution.run_closed --fname $R/configs/mw_jepa-wm_drawer-close.yaml --out_dir $D \
    --episodes 24 --planners sim_latent --goal_variant 3 --alpha 0          # alpha 0 removes the proprio term
python -m evals.failure_attribution.summarize_closed --run_dir $D           # success ladder
```
Options of `run_closed.py`: `--planners wm sim_latent sim_true`, `--goal_variant k` (arm-moved equivalent goal k), `--orbit 2,3,4` (mean encoding of
those goal variants as the goal latent, the oracle goal-set test), `--robot_free`, `--cost {l2,topk:K,pmean:P}`, `--alpha`.
Offline ranking tests: `python -m evals.failure_attribution.goal_equiv_mw --fname <cfg> --out_dir <dir> --episodes 24 --alpha 0 --no_cem [--wm_orbit 16]`.

**Always launch long jobs detached, from a script file, with a timeout:**
```bash
setsid nohup bash /root/myjob.sh > /root/myjob.log 2>&1 < /dev/null & disown
```
Timing: about 142 s per episode alone; two jobs in parallel about 106 s per episode each effective; the bottleneck is CPU (MuJoCo rendering and simulator
workers), so more than 2 jobs do not add throughput. A real world-model planner (`--planners wm`) is much slower (5-10 min per episode, see `04`).

## 4. What is running now (check before launching anything)
| Job | Script / output | Purpose | State at writing |
|---|---|---|---|
| Oracle goal-set, closed loop | `/root/run_orbit.sh`, `drawer_alpha0/orbit_234` then `orbit_01234` (+ `_summary.txt`, flag `ORBIT_DONE`) | does the mean over the equivalent goal set {2,3,4} then {0..4} restore variant-3 success at alpha 0? | episode 2 of 24, about 70 min left, then the second set (another 85 min) |
| DINO-WM closed loop | `/root/run_next.sh`, `dino/drawer_a0_v0`, then `dino/drawer_a0_v3` | replicate the drawer effect on the second model family | v0 at about episode 3 of 24; v3 starts after it |
| Offline alpha 0 ranking test | `/root/run_next.sh`, `drawer_alpha0/goal_equiv_a0` (JEPA-WM, finished or finishing) and `dino/goal_equiv_a0` | ordering accuracy per goal variant at alpha 0 | files written, summary not yet read |
| World-model goal-set smoke test | `drawer_alpha0/wm_orbit_smoke` | check `--wm_orbit` code path | died on a 25 min timeout under CPU contention without finishing an episode; **not yet validated**, re-run when the CPU is free |

Check progress: `grep -c "done:" <log>`; `grep "done:" <log> | tail -1`; `ps -eo pid,etime,pcpu,args | grep -E "run_closed|goal_equiv"`.

## 5. Results so far (details in `07_RESTART_LOG.md`)
- Code check: the variant goal's proprio is the variant's hand pose (`goal_td(frame, obs)` uses `obs[:4]`), so the released alpha = 0.1 also pulled the arm to the
  variant pose. Control = `--alpha 0`.
- Drawer-close, alpha 0, exact dynamics, JEPA-WM encoder: eval goal **48/48**; variant 3 **36/48** (12 lost, 0 gained, exact p about 0.0005); variant 2 22/24
  (not significant, 24 episodes). The image-side effect is real and depends on the arm pose (variant 4 was fine at alpha 0.1).
- Offline (saved earlier candidates, alpha 0.1, 24 episodes): ordering accuracy on clearly different candidate pairs 0.894 (eval goal) vs 0.66-0.71 (variants 2-4).
  Averaging the five equivalent goals restores 0.892; min over them gives 0.813; soft-min (T = 0.5) 0.880. Averaging is the viable aggregation.
- Not yet established: whether the oracle goal-set works in closed loop; whether a world-model-built set (no simulator) works; DINO-WM replication; other tasks.

## 6. Plan (in order, with the pre-stated decision rules)
1. **Read the oracle goal-set result** (`orbit_234`, `orbit_01234`). Rule: variant 3 must reach about 22/24 or better; otherwise drop the mean-over-a-set mechanism.
2. Read `goal_equiv_a0/summary.json` (JEPA-WM) and `dino/goal_equiv_a0/summary.json`: key `L2_ordering_accuracy_vs_true_cost_per_variant(clear pairs)`.
3. DINO-WM closed loop (v0 vs v3, alpha 0): does the drop replicate? Rule from `04`: if not, report it and investigate encoder normalisation.
4. If 1 holds: validate and run the **world-model-built goal set** (`goal_equiv_mw --wm_orbit 16`, then a closed-loop `--orbit_wm` option still to be written).
   Rule: ordering accuracy must stay within 0.03 of the eval goal on drawer-close.
5. Add the **reversibility gate** (accept a nuisance rollout only if reversing it returns to the goal under the model, or a progress probe is unchanged). It targets the
   leniency failure and the spring-button case. Not implemented yet.
6. **Baselines** to implement: forward-kinematics or render-difference arm mask (Know Thyself style), proprio-residual cost, CF-JEPA-style controllable split, temporal-distance
   cost, alpha tuning, oracle true-state cost. All on the same protocol.
7. **Breadth**: more tasks (drawer-close, push, button-press-topdown as counter-example, Push-T as control, one new object task with 30-80% base success), 48+ episodes for key
   comparisons, 3 seeds where the planner is stochastic, DINO-WM and ideally LeWM/PLDM, MPPI as a second planner, a real-world-model (`--planners wm`) run on drawer-close.
8. Go/no-go about 2026-10-24: if no method beats the arm mask and keeps Push-T and the button correct, say so and fall back to the diagnosis and protocol paper, flagged as
   below the bar. Writing starts by about 2026-11-07; register 2026-11-10.

Novelty position (from `06` and the two prior-art checks): goals as sets is not new (GS-HER); a hand-built or learned agent mask is Know Thyself; dynamics-built set with a
reversibility gate, scoped to embodiment pose in goal images, is the least-covered variant; "min over a reachable set" is lenient and should be avoided.

## 7. Backing up and moving results
```bash
cd /Volumes/ns_external/Shared/jepa-wms-project
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p 22324" root@213.173.104.47:/root/jepa-wms-project/artifacts/raw_logs/failure_attribution/ artifacts/raw_logs/pod_runs/failure_attribution/
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p 22324" root@213.173.104.47:'/root/run_*.sh /root/mkenv.sh' scripts/pod/
# push edited code back:
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p 22324" code/jepa-wms/evals/failure_attribution/ root@213.173.104.47:/root/jepa-wms-project/code/jepa-wms/evals/failure_attribution/
```
Mac copy of results so far: `artifacts/raw_logs/pod_runs/failure_attribution/`; scripts: `scripts/pod/`. Our code edits live in the Mac folder `code/jepa-wms/evals/failure_attribution/`
(`goal_equiv_mw.py`, `run_closed.py`) and were rsynced to the pod. The Mac has no numpy: analysis scripts ran in a venv at the session scratchpad; recreate with
`python3 -m venv v && v/bin/pip install numpy scipy` (the two analysis snippets are described in `07`, section 3-4).

## 8. Pitfalls already hit
| Pitfall | Fix |
|---|---|
| Workers crash with `OpenBLAS pthread_create failed ... Resource temporarily unavailable`, then `KeyboardInterrupt` in `scipy`/`cv2` imports, and the run hangs at 1% CPU | The host reports 48 CPUs, so each process starts 48 BLAS threads and the container thread limit is exhausted. `env/env.sh` now exports `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`. Kill hung runs by PID and relaunch |
| `pkill -f <name>` killed my own SSH shell (exit 255) | find PIDs with `ps`/`pgrep` and `kill <pid>`; put kill commands in a script file |
| `sleep N && ...` is blocked in the harness | use `until <check>; do sleep 20; done` in a background Bash call |
| A job's `timeout` kill prints `KeyboardInterrupt` tracebacks from workers | normal when the parent is terminated; check whether episodes were saved |
| A SimPool hang leaves a 0-byte `ep_XXX.npz` | delete it before resuming |
| Jobs launched from an SSH session that then closes | use `setsid nohup ... < /dev/null & disown` from a script file |
| Container disk is lost if the pod is terminated | rsync results back regularly (section 7) |
| rsync on macOS is `openrsync` (protocol 29); `du` flags differ from GNU | keep rsync flags simple (`-az`); avoid `du -h --exclude` |

## 9. Cost
About 5.5 pod hours so far, roughly $3 of the $24 credit (estimate, check the RunPod billing page). Each drawer-close episode costs about $0.02; the full plan above is
estimated at well under $15 if the CPU-bound runs are kept to two parallel jobs.
