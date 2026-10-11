#!/usr/bin/env bash
# Export source and curated results selected by Git. Large artifacts stay local.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 - "$PROJECT_ROOT" "${1:-$PROJECT_ROOT/artifacts/exports/source_snapshot.zip}" <<'PYCODE'
import subprocess
import sys
import zipfile
from pathlib import Path

root, target = map(Path, sys.argv[1:])
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
if not any(files):
    raise SystemExit('Select the source files with git add before making an archive.')
target.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        if name and (root / name).is_file():
            archive.write(root / name, arcname=name)
print(f'Source archive written to {target}')
PYCODE
