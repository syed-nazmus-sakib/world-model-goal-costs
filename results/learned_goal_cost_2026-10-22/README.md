# Learned goal costs: Methods A, C, D and the state head (2026-10-21 to 10-23)

Small summary files behind `docs/10_METHOD_C_D_RESULTS.md`. Per-episode files and the large encodings stay in
`artifacts/lewm_explorations_raw/` (not on GitHub). Run scripts: `scripts/pod/`.

| Folder | What | Script |
|---|---|---|
| `lewm_methodC/pusht_action_guidance/` | Method A: action guidance, LeWM PushT (killed) | `eval_goalshift.py`, `candA_finetune.py` |
| `lewm_methodC/pusht_heads/` | Method C heads (state, time) on PushT, offsets 25/50/75, seed 0; head training logs | `run_lewm_methodC.sh pusht` |
| `lewm_methodC/pusht_seeds/`, `pusht_h10_seeds/` | PushT seeds 0/1/42, last-step scoring, horizon 5 and 10 | same |
| `lewm_methodC/pusht_head_min/`, `pusht_released_min/` | PushT min-over-plan (h10): learned head vs released cost (the fair control) | same |
| `lewm_methodC/pusht_horizon/` | PushT horizon sweep | same |
| `lewm_methodC/tworoom/`, `tworoom_controls/` | TwoRoom: heads, released, released + min h10, seed 42 | `run_lewm_methodC.sh tworoom` |
| `lewm_methodC/cube/` | OGBench Cube: heads, released, released + min h10, seeds 0 / 42 (empty lines: OOM-killed, rerun later in the file) | `run_lewm_methodC.sh cube` |
| `mw_methodC_time/` | Metaworld time head (8-task dataset): drawer-open closed loop vs released; training log | `run_mw_time_head.sh` |
| `mw_methodD_pilot/` | Method D pilot: `offline.json` (ranking, regret, decoy) and closed-loop summaries per head | `run_mw_methodD_pilot.sh` |
| `mw_state_head/` | state head on five more Metaworld tasks: offline tests (closed loop: see docs/10, added when finished) | `run_mw_state_head.sh` |

Line format of the LeWM summaries: `<run name> [summary] env=... seed=... goal_offset=... horizon=... metric=<head> agg=<last|min> success_rate=<%> n=<goals>`.
