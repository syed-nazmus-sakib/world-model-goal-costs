"""Regressions for portable setup and failure-attribution maintenance."""

import itertools
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JEPA = ROOT / "code" / "jepa-wms"
sys.path.insert(0, str(JEPA))

from evals.failure_attribution.analysis import failure_masks
from evals.failure_attribution.check_goal_variants import goal_check_paths
from evals.failure_attribution.io import save_episode, save_json
from evals.failure_attribution.pool_utils import rollout_map


class AttributionTests(unittest.TestCase):
    def test_all_outcomes_partition_world_model_failures(self):
        outcomes = np.array(list(itertools.product((False, True), repeat=3)))
        masks = failure_masks(*outcomes.T)
        total = sum(mask.astype(int) for mask in masks.values())
        np.testing.assert_array_equal(total, (~outcomes[:, 0]).astype(int))
        # A true-cost search failure must not duplicate an exact-dynamics rescue.
        row = np.where((outcomes == [False, True, False]).all(axis=1))[0][0]
        self.assertTrue(masks["dynamics"][row])
        self.assertFalse(masks["search"][row])

    def test_closed_summary_cli_with_nonmonotonic_oracles(self):
        with tempfile.TemporaryDirectory() as folder:
            for ep, (wm, latent, true) in enumerate(itertools.product((False, True), repeat=3)):
                data = {"expert_success": True}
                for name, success in zip(("wm", "sim_latent", "sim_true"), (wm, latent, true)):
                    data.update({f"{name}_success": success, f"{name}_success_any": success,
                                 f"{name}_chunk_success": np.array([success])})
                save_episode(Path(folder) / f"ep_{ep:03d}.npz", **data)
            result = subprocess.run([sys.executable, "-m", "evals.failure_attribution.summarize_closed",
                                     "--run_dir", folder], cwd=JEPA, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(folder) / "summary.json").read_text())
            self.assertEqual(report["attribution"], {"failures": 4, "dynamics": 2, "cost": 1, "search": 1})

    def test_empty_summary_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-m", "evals.failure_attribution.summarize_closed",
                                     "--run_dir", folder], cwd=JEPA, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No completed ep_*.npz", result.stderr)


class OutputTests(unittest.TestCase):
    def test_failed_write_preserves_completed_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ep_000.npz"
            save_episode(path, success=True)
            before = path.read_bytes()
            with patch("evals.failure_attribution.io.np.savez", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    save_episode(path, success=False)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            with np.load(path) as episode:
                self.assertTrue(episode["success"])

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            save_json(path, {"episodes": 23}, indent=1)
            self.assertEqual(json.loads(path.read_text()), {"episodes": 23})


class PoolTests(unittest.TestCase):
    def test_worker_timeout_terminates_pool(self):
        pool = Mock()
        pool.map_async.return_value.get.side_effect = mp.TimeoutError
        with self.assertRaisesRegex(RuntimeError, "Simulator workers did not return"):
            rollout_map(pool, str, [1], timeout=0.1)
        pool.terminate.assert_called_once()
        pool.join.assert_called_once()

    def test_timeout_is_configurable_and_success_preserved(self):
        pool = Mock()
        pool.map_async.return_value.get.return_value = ["result"]
        with patch.dict(os.environ, {"JEPAWM_ROLLOUT_TIMEOUT": "42"}):
            self.assertEqual(rollout_map(pool, str, [1]), ["result"])
        pool.map_async.return_value.get.assert_called_once_with(timeout=42.0)
        pool.terminate.assert_not_called()


class PathTests(unittest.TestCase):
    def test_goal_configs_are_independent_of_log_location(self):
        with patch.dict(os.environ, {"JEPAWM_HOME": "/checkout with spaces", "JEPAWM_LOGS": "/external logs"}):
            config, output = goal_check_paths("drawer-close")
        self.assertEqual(config, Path("/checkout with spaces/configs/mw_jepa-wm_drawer-close.yaml"))
        self.assertEqual(output, Path("/external logs/failure_attribution/goal_variants_check_drawer-close.png"))

    def test_activation_with_custom_conda_directory(self):
        with tempfile.TemporaryDirectory(prefix="conda with spaces ") as folder:
            root = Path(folder)
            (root / "etc/profile.d").mkdir(parents=True)
            (root / "etc/profile.d/conda.sh").write_text(
                'conda() { [[ "$1" == activate && "$2" == jepa-wms ]]; }\n'
            )
            environment = {k: v for k, v in os.environ.items() if not k.startswith("JEPAWM_")}
            environment["CONDA_ROOT"] = str(root)
            result = subprocess.run(["bash", "-c", 'source "$1/env/env.sh" && printf "%s\\n" "$JEPAWM_CKPT"',
                                     "bash", str(ROOT)], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(ROOT / "artifacts/checkpoints"))

    def test_environment_paths_follow_moved_checkout_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="checkout with spaces ") as folder:
            root = Path(folder)
            (root / "env").mkdir()
            shutil.copy(ROOT / "env/paths.sh", root / "env/paths.sh")
            environment = {k: v for k, v in os.environ.items() if not k.startswith("JEPAWM_")}
            result = subprocess.run(["bash", "-c", 'source "$1/env/paths.sh"; printf "%s\\n" "$JEPAWM_CKPT"',
                                     "bash", str(root)], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(root / "artifacts/checkpoints"))

    def test_legacy_config_conversion_is_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="checkout with spaces ") as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            script = root / "scripts/relocate_configs.sh"
            shutil.copy(ROOT / "scripts/relocate_configs.sh", script)
            config = root / "configs/test.yaml"
            config.write_text("folder: /old checkout/artifacts/raw_logs/run\n"
                              "model_kwargs:\n  checkpoint: /old checkout/artifacts/checkpoints/model.pth.tar\n")
            for _ in range(2):
                result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(config.read_text(), "folder: ${JEPAWM_LOGS}/run\n"
                             "model_kwargs:\n  checkpoint: ${JEPAWM_CKPT}/model.pth.tar\n")


if __name__ == "__main__":
    unittest.main()
