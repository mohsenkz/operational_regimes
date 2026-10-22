"""Compatibility import for relocated entry-point wrappers; no experiments run."""
from experiments.experiment_runtime import load_config, run_script

__all__ = ["load_config", "run_script"]
