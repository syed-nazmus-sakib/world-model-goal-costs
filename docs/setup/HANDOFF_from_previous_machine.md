# Handoff: failure attribution for latent world-model planning

Historical handoff from before the folder reorganization. Paths and commands
below describe the earlier layout; use [MOVING.md](MOVING.md) for current setup.


Written 2026-10-05. Read `MOVING.md` first if the project has been relocated.

## Goal and constraints

- A CVPR 2027 submission on robot world models, started 2026-10-04 with about six weeks.
- One RTX 3090 Ti (24 GB), no real robot, no large-model training. Everything so far uses frozen released models.
- Working agreement: code is written and smoke-tested first; the long runs are launched by the user from a
  single command.

## Where things are

| What | Where |
|---|---|
| Repo (facebookresearch/jepa-wms, commit `13cf1d9`), all new work uncommitted | `jepa-wms/` |
| Current experiment code | `jepa-wms/evals/failure_attribution/` |
| Earlier Push-T-only version of the same experiment (kept because the alpha sweep uses it) | `jepa-wms/evals/pusht_failure_attribution/` |
| Equivalence tests for the fast simulators | `jepa-wms/tests/test_fast_rollout.py` |
| Configs, checkpoints, data, run scripts, results | `jepa-wms-work/` |
| Environment activation and paths | `source jepa-wms-work/env.sh` (conda env `jepa-wms`) |

The environment is installed for Push-T, Wall and Metaworld only. PointMaze dependencies were skipped.

## The current direction

A latent planner has three parts: a predictor (the world model), a cost (latent distance to the goal image,
plus a small proprio term weighted by `alpha`), and a search (CEM). The experiment re-plans every episode of
a released eval with parts swapped for an oracle, keeping the start, goal and CEM budget fixed:

| Planner | Dynamics | Cost |
|---|---|---|
| `wm` (the released eval) | world model | latent |
| `sim_latent` | simulator | latent |
| `sim_true` | simulator | true state distance |

A failure is attributed to dynamics if `sim_latent` succeeds, to the cost landscape if `sim_latent` fails but
`sim_true` succeeds, and to search if `sim_true` fails.

### Result on Push-T (96 released episodes, complete)

| Planner | JEPA-WM | DINO-WM |
|---|---|---|
| `wm` | 62/96 = 64.6% | 63/96 = 65.6% |
| `sim_latent` | 62/96 = 64.6% | 59/96 = 61.5% |
| `sim_true` | 87/96 = 90.6% | 87/96 = 90.6% |

- Replacing the world model with the simulator does not raise success.
- Attribution of the world-model failures (JEPA-WM / DINO-WM): dynamics 5 / 6, cost landscape 20 / 18,
  search 9 / 9.
- The failures are mostly the pusher, not the block. Block-only success of `wm` is 84.4% / 87.5%, and in
  19 / 20 episodes the block is within tolerance while the pusher is not. Failed episodes end with the
  pusher a median 96 / 87 px from its goal position.
- In 88% / 82% of failures the world model scores the dataset's expert actions better than the plan CEM
  returned.
- The proprio term is under 1% of the final cost in failed episodes at the released `alpha = 0.1`.

Files: `logs/pusht_failure_attribution/{jepa_wm,dino_wm}/` (one `ep_XXX.npz` per episode, `summary.json`) and
`*_summary.txt` next to them.

### Runs in progress when this was written

1. Alpha sweep on Push-T with the simulator as dynamics (`alpha` = 0, 0.1, 1, 10, 100; JEPA-WM encoding).
   Done: 0 and 0.1, both 62/96 = 64.6% official. `alpha = 1` was at 35/50. Finished values are skipped on
   rerun. Output: `logs/pusht_failure_attribution/jepa_wm/alpha_sweep/` and `alpha_sweep_jepa.log`.
2. The same attribution on Wall, JEPA-WM then DINO-WM, 96 episodes each. JEPA-WM was at 37/96 with the
   world model succeeding on 28. Output: `logs/failure_attribution/wall_{jepa,dino}_wm/` and
   `wall_*_summary.txt`.

Commands (after `source jepa-wms-work/env.sh`, from `jepa-wms/`):

```bash
# resume or restart the Wall run
nohup bash <root>/jepa-wms-work/run_attribution_wall.sh > <root>/jepa-wms-work/logs/failure_attribution/wall_driver.log 2>&1 &

# resume the alpha sweep
python -m evals.pusht_failure_attribution.alpha_sweep --fname <root>/jepa-wms-work/configs/pt_jepa-wm_repro.yaml \
  --run_dir <root>/jepa-wms-work/logs/pusht_failure_attribution/jepa_wm --alphas 0 0.1 1 10 100

# summarise any attribution run
python -m evals.failure_attribution.summarize --run_dir <run dir>

# run the attribution on another single-plan eval config (Push-T or Wall)
python -m evals.failure_attribution.run --fname <eval config> --out_dir <dir> --episodes 96 [--alpha A] [--no_oracles]
```

Each episode takes about 2.3 minutes on the 3090 Ti, almost all of it the world-model CEM; the two oracle
planners add about 15 seconds.

## Next steps, in order

1. Read the alpha sweep. If a larger `alpha` brings the pusher home with the simulator, rerun the real
   world-model eval at that value (`run.py --alpha`). If not, find out why CEM leaves the pusher behind.
2. Read the Wall result: does the simulator fail to help there too.
3. Read "Planning Limits of Latent World Models" (September 2026). It reportedly contains an
   oracle-dynamics ablation and has not been checked. This decides how much of the finding is new.
4. Extend to Metaworld. Its eval replans every 15 steps over 100-step episodes, so the released eval alone
   is roughly 11 hours per model on this GPU. `run.py` asserts single-plan evals and needs a closed-loop
   version; `evals/mw_hidden_physics/sim_test.py` has a tested MuJoCo state snapshot / restore to build on.
5. A paper needs the pattern on several tasks plus a fix that raises success with frozen models. For scale,
   D-JEPA's abstract reports 87.9% on Push-T with a learned component.

## Things that are easy to get wrong

- The released Push-T, Wall and Metaworld checkpoints of JEPA-WM and DINO-WM contain only the predictor and
  the proprio / action encoders. Both use the same frozen stock DINOv2 ViT-S/14, with different input
  normalisation. They are not independent evidence for anything about the encoder or the latent cost.
- Push-T success requires the pusher position as well as the block (`PushTWrapper.eval_state`). Results
  should be reported under both the official and a block-only criterion.
- These 96-episode numbers are for one released checkpoint and one seed. The paper reports 70.2 +/- 2.8
  (JEPA-WM) and 66.0 +/- 4.7 (DINO-WM) averaged over seeds and epochs.
- The Push-T and Wall datasets came from DINO-WM's public OSF mirror, which the repo README describes as the
  same unmodified data. The Hugging Face dataset repo `facebook/jepa-wms` is gated.
- A Hugging Face token was pasted into the chat on 2026-10-04 and should be revoked. No token is stored on
  disk.
- Decoder heads and plots are disabled in the local configs (`configs/*_repro.yaml`). This does not affect
  planning.
- `sim_worker.py` is loaded by spawned worker processes; scripts that use `SimPool` must be run from a file
  or with `python -m`, not from stdin.

## Directions tried and dropped

Each was closed by a cheap test. Code is under `jepa-wms/evals/`, results under `jepa-wms-work/logs/`.

| Direction | Outcome | Code / results |
|---|---|---|
| Pusher entanglement in the Push-T latent cost | A linear model in block position, block angle and pusher position reproduces about 91% of the latent's pairwise rankings; the pusher weighs about 0.4 of a block pixel. No surprising residual. | `pusht_goal_decomp` |
| One global planning metric cannot serve several Metaworld tasks | A global learned metric reaches 94.5% ranking accuracy against 96.9% per-task; with a higher-capacity global metric the gap falls to about 0.5 points. | `mw_metric_conflict` |
| Identical pixels, different hidden physics (Metaworld push) | Within plausible ranges of velocity, friction and mass, futures and preferred actions barely change; the hand is position-controlled. | `mw_hidden_physics` |
| Partial reversibility of action cycles (Metaworld) | In push the puck is gripped and returns with the hand, so contact cycles are reversible. | `mw_cycles` |
| More search makes control worse; calibrated deferral for video world-model evaluators | Not run. The first was superseded by recent papers; for the second, the proposed MiraBench test labels every case as a true failure and the GigaBrain evaluator split appears to ship no outcome labels. | none |

Several of the plans behind these directions cited papers and datasets that could not be verified (for
example a 473 MB MiraBench dataset on Hugging Face). Check a resource exists before building on it.
