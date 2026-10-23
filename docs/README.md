# Documents

Read in numerical order. 01 is the self-contained summary; 02-04 give the full detail.

| File | What it is |
|---|---|
| [`01_ASSESSMENT.md`](01_ASSESSMENT.md) | **Start here.** Each failure case found, judged on whether it is real, causally isolated, large/general and still open; which can carry a method. |
| [`02_REPORT_progress.md`](02_REPORT_progress.md) | Full results report: oracle attribution, cost studies, task-equivalence experiments (levels L1-L4), literature position. |
| [`03_REPORT_area_cost.md`](03_REPORT_area_cost.md) | The area-invariant (top-k) patch cost: why it was proposed, what it fixed offline, why it failed in closed loop. A closed direction, kept for the record. |
| [`04_NEXT_STEPS.md`](04_NEXT_STEPS.md) | Runbook for continuing: planned experiments E1-E6 with exact commands, decision rules and pitfalls. |
| [`05_LITERATURE_REVIEW.md`](05_LITERATURE_REVIEW.md) | Literature review from the earlier LeWM phase. |
| [`06_LITERATURE_REVIEW_v2.md`](06_LITERATURE_REVIEW_v2.md) | **Current literature review** (2026-10-15): verified map, novelty verdict per finding, CVPR calibration, candidate directions. Raw reports in `literature_raw/`. |
| [`07_RESTART_LOG.md`](07_RESTART_LOG.md) | **Restart log**: findings since 2026-10-15 (proprio confound, drawer alpha 0 results, oracle goal set), code changes, pre-stated decision rules. |
| [`08_HANDOFF_runpod.md`](08_HANDOFF_runpod.md) | **Handoff**: how to connect to and run on the RunPod pod, what is running, pitfalls, next plan. Start here to continue the work. |
| [`09_PODS_ACCESS.md`](09_PODS_ACCESS.md) | **Access sheet**: both RunPod pods (ids, IPs, ports, prices), lab machines, API usage, what runs where. No secret values. |
| [`10_METHOD_C_D_RESULTS.md`](10_METHOD_C_D_RESULTS.md) | **Learned goal costs** (2026-10-21 to 10-23): Methods A, C, D and the state head on LeWM and Metaworld; tables, findings, related work, infrastructure lessons. |
| [`11_DIRECTIONS.md`](11_DIRECTIONS.md) | **Current directions**: the paper, open and closed directions with decision rules, plan to the CVPR deadline. Start here for what to do next. |
| [`setup/HANDOFF_from_previous_machine.md`](setup/HANDOFF_from_previous_machine.md) | State of the project when it moved from the first machine (RTX 3090 Ti): Push-T attribution and earlier Metaworld diagnostics. |
| [`setup/MOVING.md`](setup/MOVING.md) | Portable setup after moving or cloning. |
| [`setup/GITHUB.md`](setup/GITHUB.md) | Local checks, commit review, and GitHub upload commands. |

## Glossary

| Term | Meaning |
|---|---|
| `wm` | released planner: learned world model + latent cost + CEM |
| `sim_latent` | exact simulator dynamics + the same latent cost + CEM (removes model error) |
| `sim_true` | exact simulator dynamics + true-state cost + CEM (removes model and cost error) |
| latent cost | mean over the 256 DINOv2 patches of the squared feature difference to the goal image, + α · proprio distance (released α = 0.1) |
| task-equivalent goal | goal image with the same task-relevant state (object pose / joint) but the robot arm elsewhere |
| L1 / L2 / L3 / L4 | cost change / candidate-ranking flips / best-candidate agreement / closed-loop success, between equivalent goals |

## Note on dates and paths

02-04 were written during the work (2026-10-06 to 10-08) on the second machine (RTX 5090); 01 on 2026-10-11.
Current run commands use the checkout and `JEPAWM_*` variables. Historical run metadata and the handoff
retain their original provenance. Setup commands in `setup/MOVING.md` apply to the current folder layout.
