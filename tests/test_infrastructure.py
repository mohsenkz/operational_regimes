import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class InfrastructureTests(unittest.TestCase):
    def test_entry_points_import_without_dependencies_or_outputs(self):
        names = [
            p.name.removesuffix(".py.txt")
            for p in (ROOT / "experiments/_legacy").glob("*.py.txt")
        ]
        names = [
            n for n in names if n not in {"tier1_families", "tier2_mm", "plot_fig1"}
        ]
        # -S disables site-packages: wrappers must not import scientific packages.
        modules = []
        for name in names:
            paths = [
                p
                for p in (ROOT / "experiments").rglob(name + ".py")
                if "_legacy" not in p.parts
            ]
            self.assertEqual(len(paths), 1, name)
            modules.append(".".join(paths[0].relative_to(ROOT).with_suffix("").parts))
        code = (
            "import sys; sys.path.insert(0, "
            + repr(str(ROOT))
            + "); "
            + "; ".join("import " + n for n in modules)
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-S", "-c", code],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_configuration_matches_archived_defaults(self):
        import ast

        for config_path in (ROOT / "configs").glob("*.json"):
            config = json.loads(config_path.read_text())
            source = ROOT / "experiments/_legacy" / (config["experiment"] + ".py.txt")
            tree = ast.parse(source.read_text())
            defaults = {}
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                ):
                    try:
                        defaults[node.targets[0].id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            for key, value in config["parameters"].items():
                self.assertEqual(json.loads(json.dumps(defaults[key])), value)
            compile(tree, str(source), "exec")

    def test_unknown_parameter_rejected_before_execution(self):
        from experiments.experiment_runtime import run_script

        with self.assertRaisesRegex(ValueError, "Unknown parameters"):
            run_script(
                "exp_admm", {"experiment": "exp_admm", "parameters": {"typo": 1}}
            )

    def test_missing_original_drift_requires_opt_in(self):
        from experiments.experiment_runtime import run_script

        with self.assertRaisesRegex(ValueError, "Missing original"):
            run_script("make_fig_drift")

    def test_cli_list(self):
        result = subprocess.run(
            [sys.executable, "-S", str(ROOT / "main.py"), "list"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("exp_mimo_v3", result.stdout)


if __name__ == "__main__":
    unittest.main()
