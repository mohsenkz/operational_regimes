"""Import-safe execution of archived reproduction scripts.

Compatibility layer: preserves script-level globals and star imports. Not thread-safe
because legacy scripts use process-wide argv, cwd and sys.path.
"""

import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_config(name, path=None):
    path = Path(path) if path else ROOT / "configs" / (name + ".json")
    data = json.loads(path.read_text())
    if data.get("experiment") != name:
        raise ValueError("Config experiment does not match requested experiment")
    return data


def run_script(name, config=None, argv=None):
    defaults = load_config(name)
    cfg = copy.deepcopy(defaults if config is None else config)

    if cfg.get("experiment") != name:
        raise ValueError("Config experiment does not match requested experiment")
    if name == "make_fig_drift" and not cfg.get("allow_reconstructed_drift", False):
        raise ValueError(
            "Missing original drift module: explicitly opt in to reconstructed drift support"
        )

    overrides = cfg.get("parameters", {})
    unknown = set(overrides) - set(defaults["parameters"])
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")

    effective = copy.deepcopy(defaults)
    effective["parameters"].update(overrides)
    effective["allow_reconstructed_drift"] = bool(
        cfg.get("allow_reconstructed_drift", False)
    )
    effective["argv"] = (
        cfg.get("argv", defaults.get("argv", [])) if argv is None else list(argv)
    )
    source = ROOT / "experiments" / "_legacy" / (name + ".py.txt")
    tree = ast.parse(source.read_text(), filename=str(source))
    # Replace only explicitly selected, top-level literal defaults. All other
    # scientific code, ordering, seed calls and embedded loop grids are unchanged.

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            key = node.targets[0].id
            if key in effective["parameters"]:
                old = ast.literal_eval(node.value)
                value = effective["parameters"][key]
                if isinstance(old, tuple):
                    value = tuple(value)
                node.value = ast.copy_location(
                    ast.parse(repr(value), mode="eval").body, node.value
                )

    # Prior commit 892621a used the pre-package src.* module paths. Translate
    # import paths only; archived source bytes remain unchanged and hashable.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for old, new in (
                ("src.optimization", "operational_regimes.optimization"),
                ("src.regimes", "operational_regimes.regimes"),
            ):
                if node.module == old or node.module.startswith(old + "."):
                    node.module = new + node.module[len(old) :]

    ast.fix_missing_locations(tree)
    Path("figs").mkdir(exist_ok=True)
    logs = Path("run_metadata")
    logs.mkdir(exist_ok=True)
    # Save before execution, so failed runs retain the effective configuration.
    import datetime

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    import hashlib
    import platform

    effective["provenance"] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "output_directory": str(Path.cwd()),
        "status": "configuration saved before execution; not a success certificate",
    }

    (logs / f"{name}_{stamp}.json").write_text(json.dumps(effective, indent=2))
    previous_argv, previous_path = sys.argv[:], sys.path[:]

    try:
        sys.argv = [str(source), *effective["argv"]]
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "src"))
        namespace = {
            "__name__": "__main__",
            "__file__": str(source),
            "__package__": None,
        }
        exec(compile(tree, str(source), "exec"), namespace)
    finally:
        sys.argv[:] = previous_argv
        sys.path[:] = previous_path
