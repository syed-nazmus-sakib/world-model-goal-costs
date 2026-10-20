# Pods and machines: access sheet

Written 2026-10-16. For any agent that needs to reach the compute. **No secret values are stored here.** The RunPod API key lives only in
`/Volumes/ns_external/Shared/jepa-wms-project/.env` (variable `runpod_api`, one line); `.env` is listed in `.gitignore`. Never print, copy, log or commit it.
SSH uses the Mac's key pair `~/.ssh/id_ed25519` (public part: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFQuBpFB3ycfDpQsyZMACoUmMLNL+pbp7K2v4DJQeSl0 nazmus-sakib@Mac`).
The user's sudo password was pasted into a chat once; it is not needed, must not be stored anywhere, and should be rotated.

Account state (API, 2026-10-16): balance about $19.03, spend about $0.864/hr, spend limit $80. Check with the GraphQL query in section 4.
Budget rule: the user has little credit; ask before creating more pods or a bigger GPU, and stop or delete pods that are idle.

## 1. RunPod pods (both on the community cloud, container disk only, no persistent volume)
| | Pod A `due_amber_cardinal` | Pod B `underground_coffee_hummingbird` |
|---|---|---|
| Pod id | `i9ayugl1jd6ic3` | `21es5ky1u8rg35` |
| GPU | RTX PRO 4000 Blackwell, 24 GB (driver 580) | RTX 4000 Ada, 20 GB (driver 580) |
| vCPU / RAM | 12 listed, cgroup cap about 10.2 cores / 31 GB | 16 listed, cgroup cap about 13.6 cores / 62 GB |
| Price | **$0.57/hr** | **$0.28/hr** (not $0.20: that was the lowest listing) |
| Disk | 50 GB container disk | 50 GB container disk |
| Direct SSH (SCP/rsync work) | `ssh -i ~/.ssh/id_ed25519 root@213.173.104.47 -p 22324` (stopped 2026-10-16) | `ssh -i ~/.ssh/id_ed25519 root@157.157.221.29 -p 22014` (port changes on every restart: 22144 -> 22058 -> 22014; read it from the Connect tab or the REST API) |
| Proxy SSH (needs `-tt`, no scp) | `ssh -tt -i ~/.ssh/id_ed25519 i9ayugl1jd6ic3-64410cda@ssh.runpod.io` | `ssh -tt -i ~/.ssh/id_ed25519 21es5ky1u8rg35-64410eca@ssh.runpod.io` |
| Project root | `/root/jepa-wms-project` | `/root/jepa-wms-project` (copied from pod A) |
| Conda | `/root/miniconda3`, env `jepa-wms` | same (copied; torch 2.7.0+cu128, verified on the 4000 Ada) |
| Launch scripts | `/root/run_*.sh` | `/root/run_b.sh`, `/root/pull.sh` |

IPs and ports change if a pod is restarted: read them from the pod's Connect tab or from the REST API. Pod B had no `raw_logs` copied; its results are in
`/root/jepa-wms-project/artifacts/raw_logs/failure_attribution/`. Pod A holds the earlier results (Mac backup in `artifacts/raw_logs/pod_runs/`).

Activate on either pod:
```bash
cd /root/jepa-wms-project && export CONDA_ROOT=/root/miniconda3 && source env/env.sh && cd code/jepa-wms
```
`env.sh` limits BLAS threads to 1. Without that, workers crash with `OpenBLAS pthread_create failed` and runs hang.
Use `setsid nohup bash script.sh > log 2>&1 < /dev/null &` from a script file for long jobs. Read `08_HANDOFF_runpod.md` for run commands and pitfalls.

Pod B authorisation notes: the Mac key is installed via the pod's `PUBLIC_KEY` setting. A temporary key pair (`podB-temp`) was used for the pod-to-pod copy and has been removed
from pod A's `authorized_keys`; delete `/root/.ssh/id_a*` on pod B if it is still there.

**Confirmed 2026-10-17: stopping a pod wipes its container disk** (`/root` was empty after `start`). Pod B was rebuilt from the Mac (upload + `mkenv.sh`,
about 40 min). Pod A is stopped and empty. Always back up before stopping; prefer finishing a queue over stop/start cycles.

## 2. What runs where (2026-10-17: pod B only)
- Pod B runs `/root/jepa-wms-project/scripts/pod/queue_generality.sh` (one sequential lane): task search (6 Metaworld tasks, 12 episodes, sim_latent + sim_true),
  encoder generality (offline L1-L3 with siglip2, clip, dinov2_b, vjepa2 via `alt_encoders.py`), task generality (L4 on the selected tasks), then the real
  world-model planner on drawer-close. Outputs under `artifacts/raw_logs/failure_attribution/generality/`, progress in `generality/queue.log`, flag `QUEUE_DONE`.
  A watcher on the Mac (`artifacts/raw_logs/pod_runs/watch_b.log`) backs up and stops the pod when the flag appears.

Pitfalls met on 2026-10-17 (pod B):
- Configs on the pod were rewritten by `mkenv.sh` (checkpoint path -> pod path). A config copied from the Mac points at the lab path
  `/run/media/rmedu2025/...` and fails with FileNotFoundError. Make new task configs on the pod from the pod's `mw_jepa-wm.yaml` (`task:` line only).
- V-JEPA 2 via torch.hub `main` is broken: the checkpoint URL is `http://localhost:8300/vitl.pt` and the ViT passes `H_patches` to blocks that take `H`.
  Fix: `curl -L -o /root/.cache/torch/hub/checkpoints/vitl.pt https://dl.fbaipublicfiles.com/vjepa2/vitl.pt` (5.1 GB, 20 s on the pod), plus the kwarg adapter
  already in `alt_encoders.py`. Loads in 13 s, 1.3 GB of VRAM.
- `pgrep -f rerun_rw.sh` inside an ssh command matches the ssh command itself (the name is in its argv), so "already running" checks must live in a script on the
  pod (`/root/launch_reruns.sh`) or use a bracket pattern (`'rerun_r[w]'`).
- The Mac's download link was about 10-30 KB/s on 2026-10-17 (upload 1.5 MB/s; the pod itself uploads at 22 MB/s), so pod-to-Mac copies of the `ep_*.npz`
  arrays (0.5 MB each, 160 MB for the encoder stage) take hours. The watcher was replaced by `watch_b2.sh` (scratchpad): small files first (`--exclude 'ep_*.npz'`),
  then a bounded slice for the arrays, and the pod is stopped only after a complete full rsync. Mac `rsync` is openrsync: `--exclude`, `--partial`, `--timeout` work.
- `scp` to the pod stalls; use `ssh ... "cat > remote" < local` and `ssh ... "cat remote" > local`.
- Measured 03:30 UTC: pod -> Mac 1.2 KB/s (3 MB in 41 min), Mac -> pod 44 KB/s, pod -> internet 22 MB/s, lab-6 -> internet 52 MB/s. So the **full backup of
  pod B goes to lab-6**: lab-6's own key (`~/.ssh/id_ed25519.pub`, comment `runpod-dpcs-rmedu-06`) was added to pod B's `authorized_keys`, and `~/podB_pull.sh`
  on lab-6 (started with `bash ~/start_pull.sh`) rsyncs `failure_attribution/` into `~/jepa_wms_podB_backup/failure_attribution/` every 5 min, then after
  `QUEUE_DONE` does a final pull and touches `~/jepa_wms_podB_backup/LAB_BACKUP_DONE`. It uses no GPU. The Mac watcher (`watch_b3.sh`, scratchpad) mirrors only the
  small files (`--exclude 'ep_*.npz'`) into `artifacts/raw_logs/pod_runs/podB/` and stops the pod only when `QUEUE_DONE`, no jobs, and `LAB_BACKUP_DONE` all hold.
  Since 13:10 UTC both loops wait for `generality/ALL_DONE` (written by `/root/stage5.sh`, method candidate 1), not `QUEUE_DONE` (written by stage 4).
  To get the arrays onto the Mac later: `rsync -az lab-6:~/jepa_wms_podB_backup/failure_attribution/ artifacts/raw_logs/pod_runs/podB/` once the Mac's link recovers.

## 2a. What ran on 2026-10-16 (both pods, now stopped; results in `artifacts/raw_logs/pod_runs/`)
- Pod A: mask baselines `/root/run_mask.sh A|B` (reach-wall and button-press-topdown, released cost vs oracle robot mask), outputs in `artifacts/raw_logs/failure_attribution/mask_baselines/`.
- Pod B: `/root/run_b.sh 1` (offline world-model goal set, K = 32, H = 16, output `drawer_alpha0/wm_dir_h16k32`) and `/root/run_b.sh 2` (closed-loop world-model goal set on drawer-close variant 3
  then the eval goal, outputs `drawer_alpha0/wmset_h8_v3` and `wmset_h8_v0`).
Check with `ps -eo pid,etime,args | grep run_closed` and `grep -c "done:" <log>`.

## 3. Lab machines (not RunPod; the user's lab)
Defined in `~/.ssh/config`: `lab-5` (user `rmedu-4090`, RTX 4090 24 GB, 28 cores, 62 GB RAM, disk 96% full) and `lab-6` (user `rmedu`, RTX 3090 Ti 24 GB, 24 cores, 30 GB RAM), plus `lab-5-lan` and `lab-6-lan`.
At 2026-10-16 02:14 their CPUs were nearly idle but **both GPUs were full of other people's training jobs** (lab-5: `train_depthctrl.py`, 21.7 GB; lab-6: `phaseS1_train.py`, 23.7 GB,
running 13 hours). Do **not** start GPU jobs there without the user's explicit OK: an out-of-memory error could kill their training. Read-only checks are fine.

## 4. Managing pods from the Mac (RunPod REST API; the key stays in `.env`)
```bash
cd /Volumes/ns_external/Shared/jepa-wms-project
KEY=$(sed -E 's/^[^=]*=[[:space:]]*//; s/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//' .env)   # never echo $KEY
curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods                                   # list pods (id, status, costPerHr)
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" https://api.runpod.io/graphql \
     -d '{"query":"query { myself { clientBalance currentSpendPerHr } }"}'                                   # balance
curl -s -X POST   -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/<POD_ID>/stop            # stop (container disk may be lost)
curl -s -X DELETE -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/<POD_ID>                 # terminate
```
Notes: Python's default `urllib` user agent is blocked by the CDN (error 1010): use `curl`. Creating a pod with `POST /v1/pods` returned "no instances currently available" for the GPU types tried
(4090, 3090, A40); capacity changes minute to minute, so the user created pod B in the web UI. Community-cloud prices, per GPU, with minimum vCPU (GraphQL `gpuTypes.lowestPrice`) on 2026-10-16:
RTX 3080 Ti 12 GB $0.18 (16 vCPU), RTX 4000 Ada 20 GB $0.20 (16), RTX A4500 20 GB $0.19 (12), RTX A4000 16 GB $0.17 (12), RTX 3090 24 GB $0.22 (8), L40 48 GB $0.69 (24 vCPU). The price actually
charged can be higher than the listing (pod B: $0.28), so check `costPerHr` right after creation. Our jobs are CPU-bound and need only about 3 GB of VRAM, so pick by vCPU per dollar, not by GPU.

`runpodctl` is not installed on the Mac (the user said they installed it, but it was not on PATH when checked). macOS has no `setsid`; use `nohup ... &` locally.

## 5. Backing up results
```bash
cd /Volumes/ns_external/Shared/jepa-wms-project
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p 22324" root@213.173.104.47:/root/jepa-wms-project/artifacts/raw_logs/failure_attribution/ artifacts/raw_logs/pod_runs/podA/
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p 22144" root@157.157.221.29:/root/jepa-wms-project/artifacts/raw_logs/failure_attribution/ artifacts/raw_logs/pod_runs/podB/
```
Container disks are lost when a pod is terminated; always back up before stopping a pod. The Mac's `rsync` is `openrsync` (keep flags to `-az`).
