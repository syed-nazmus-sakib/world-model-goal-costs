# Preparing and uploading to GitHub

The workspace is one Git repository on branch `main`. The upstream source
snapshots are included as ordinary folders; provenance and license boundaries
are recorded in [PROVENANCE.md](../../PROVENANCE.md).

Before committing:

```bash
python3 -m pip install numpy PyYAML
python3 scripts/check_repo.py
python3 -m unittest discover -s tests -v
git status --short
git diff --cached --stat
```

The lightweight checks need no GPU, Conda, datasets or checkpoints. To also
check local datasets and checkpoint locations, run
`python3 scripts/check_repo.py --check-assets`.

The local history is split into 18 commits with short, descriptive messages.
Review it with `git log --oneline --reverse`, then connect the empty GitHub
repository you create:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Datasets, checkpoints, raw logs, nested Git backups, environment caches, and
generated compatibility macros are excluded. Small arrays in `results/` and
the upstream source licenses are included. Checkpoints are downloaded separately
following the upstream README; see [MOVING.md](MOVING.md) for setup after cloning.
