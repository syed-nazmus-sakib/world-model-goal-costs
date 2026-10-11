#!/usr/bin/env python3
"""Check source syntax, portable configs, curated results, and Git packaging."""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-assets", action="store_true", help="require local datasets and checkpoints")
    args = parser.parse_args()
    errors = []
    python_files = [p for folder in ("code", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")]
    for path in python_files:
        try:
            ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    shell_files = [p for folder in ("scripts", "env", "code/lewm_explorations") for p in (ROOT / folder).rglob("*.sh")]
    for path in shell_files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")

    paths = {
        name: os.environ.get(name, str(ROOT / "artifacts" / suffix))
        for name, suffix in (("JEPAWM_LOGS", "raw_logs"), ("JEPAWM_CKPT", "checkpoints"), ("JEPAWM_DSET", "datasets"))
    }
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    for path in configs:
        try:
            data = yaml.safe_load(path.read_text())
            for value in (data["folder"], data["model_kwargs"]["checkpoint"]):
                if not re.match(r"^\$\{JEPAWM_(?:LOGS|CKPT)\}/", value):
                    raise ValueError(f"artifact path must use an environment placeholder: {value}")
            value = data["model_kwargs"]["checkpoint"]
            resolved = re.sub(r"\$\{([^}]+)\}", lambda match: paths[match[1]], value)
            if args.check_assets and not Path(resolved).is_file():
                raise ValueError(f"checkpoint missing: {resolved}")
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    if args.check_assets:
        for name in ("pusht_noise", "wall_single", "Metaworld"):
            if not (Path(paths["JEPAWM_DSET"]) / name).is_dir():
                errors.append(f"dataset missing: {name} under {paths['JEPAWM_DSET']}")

    result_files = list((ROOT / "results").rglob("*.json")) + list((ROOT / "results").rglob("*.jsonl"))
    for path in result_files:
        try:
            text = path.read_text()
            if not text.strip():
                raise ValueError("empty result file")
            if path.suffix == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        json.loads(line)
            else:
                data = json.loads(text)
                attribution = data.get("attribution") if isinstance(data, dict) else None
                if attribution and "dynamics (sim_latent succeeds)" in attribution:
                    groups = [set(value["episodes"]) for value in attribution.values()]
                    if sum(map(len, groups)) != len(set().union(*groups)):
                        raise ValueError("failure attribution categories overlap")
        except (ValueError, TypeError, KeyError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")

    staged = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=ROOT, capture_output=True, text=True)
    if staged.returncode == 0:
        for entry in staged.stdout.split("\0"):
            if not entry:
                continue
            metadata, filename = entry.split("\t", 1)
            if metadata.startswith("160000"):
                errors.append(f"nested Git link instead of bundled source: {filename}")
            if filename.startswith("artifacts/") and filename != "artifacts/README.md":
                errors.append(f"large artifact staged: {filename}")
            if (ROOT / filename).is_file() and (ROOT / filename).stat().st_size >= 100_000_000:
                errors.append(f"file exceeds GitHub's regular-file limit: {filename}")
    if errors:
        raise SystemExit("Repository checks failed:\n" + "\n".join(f"  {error}" for error in errors))
    print(f"OK: {len(python_files)} Python files, {len(shell_files)} shell scripts, "
          f"{len(configs)} portable configs, {len(result_files)} result files; Git packaging checked.")


if __name__ == "__main__":
    main()
