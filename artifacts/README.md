# Artifacts (large files, not for git)

About 170 GB. Everything here can be regenerated from `code/` and `scripts/` except the downloaded datasets and
checkpoints, which come from the upstream projects.

| Folder | Content | Size |
|---|---|---|
| `datasets/` | Push-T (`pusht_noise`), Wall (`wall_single`), each unzipped and as the original zip; Metaworld parquet files | ~67 GB |
| `checkpoints/` | released JEPA-WM and DINO-WM predictors for Push-T, Wall, Metaworld (`env/env.sh` points `JEPAWM_CKPT` here) | 1.4 GB |
| `raw_logs/` | per-episode outputs of every jepa-wms run (`JEPAWM_LOGS` points here; curated copies in `results/jepa_wms/`) | ~20 GB |
| `lewm_checkpoints/` | stable-worldmodel cache: LeWM PushT / TwoRoom checkpoints, trained variants, eval videos | 5.8 GB |
| `lewm_explorations_raw/` | full outputs of the LeWM study, incl. window datasets and trained models (`exp5/`, `exp6/`) | ~53 GB |
| `original_archives/` | the two zips the project arrived in from the first machine (kept unmodified) | 26 GB |
| `local_git/` | preserved Git metadata from the two bundled upstream snapshots | local only |
| `maintenance_backup_2026-10-12/` | original derived summaries before the attribution repair | small |
| `exports/` | source archives created by `scripts/legacy/make_archives.sh` | generated |

Only this README is included in Git. Artifact paths can be overridden through the `JEPAWM_*` environment
variables. Existing datasets and checkpoints were preserved during preparation for GitHub.
