# Review snapshot: first-stage execution refactoring

This is a locally modified review snapshot, not an official upstream release or
verified reproduction of the paper. No scientific conclusions have been updated.

## Usage

```sh
python -m pip install -r requirements.txt
python main.py list
python main.py run plot_fig1 --output-dir results/figure1
python main.py run exp_mimo_v2 --output-dir results/minpower
python main.py run exp_mimo_v2 --output-dir results/maxmin --script-args maxmin
python main.py run exp_admm --config configs/exp_admm.json --output-dir results/admm
python -m unittest discover -s tests -v
```

Commands other than `list` can be costly. `--script-args` must come last, and
forwards all subsequent tokens to the original script. The CLI defaults to
`minpower` for exp_mimo_v2; the original direct entry point still requires its
positional argument. Use separate output directories for independent runs:
several legacy scripts resume existing JSON files without checking configuration
compatibility. The CLI changes cwd to the output directory, so analysis/plotting
commands must point to the directory containing their input results.

## What changed

1. Fifteen formerly executable-at-import scripts now expose `run(config=None,
   argv=None)` and guarded direct entry points. Importing these entry points does
   not import scientific dependencies, solve problems or write files.
2. Their original scientific bodies are retained in `experiments/_legacy/*.py.txt`
   and executed only on explicit invocation. This compatibility boundary preserves
   module-global semantics and wildcard imports. It is deliberately not presented
   as the final clean functional architecture.
3. Eighteen CLI commands and per-command JSON configuration files are available.
   The already guarded tier1_families, tier2_mm and plot_fig1 libraries remain
   unchanged; archived copies are used only for CLI execution.
4. A conservative subset of top-level literal parameters was extracted without
   changing defaults. Tuple defaults are restored to tuples on execution. Unknown
   parameter keys are rejected. Seeds, NT/NC tuple assignments, formulas, nested
   grids and some thresholds are still embedded in source: these configurations
   are NOT exhaustive experiment manifests. Only documented `parameters` and
   `argv` fields should be edited.
5. Each invocation saves effective extracted settings, argv, Python version,
   source SHA-256 and output directory under `run_metadata/` before executing.
   This is not a full environment lock, dependency manifest or success record.
6. `make_fig_drift` now handles an empty transition-distance array when printing
   summary statistics. Its missing support module was reconstructed earlier and
   is marked as such in `src/drift_experiment.py`. Execution is blocked unless
   explicitly authorized:

   ```sh
   python main.py run make_fig_drift --output-dir results/drift --allow-reconstructed-drift
   ```

   This opt-in does NOT certify original SPEEDS/BURN settings or the paper figure.
   Source archive provenance: all retained bodies are copied from the inspected
   snapshot, except the two empty-array summary guards in make_fig_drift.

## Validation performed

- Five infrastructure unit tests passed: dependency-free and output-free imports
  of all 15 wrappers; literal-default consistency and compilation of all 18
  retained bodies; rejection of unknown parameter names; missing-drift opt-in;
  CLI command listing.
- A real `plot_fig1` CLI smoke run generated PNG, PDF and SVG outputs. The figure's
  thresholds remain explicitly illustrative, not extracted paper settings.
- Full Monte Carlo, MIMO, ADMM and drift experiments were NOT rerun. Numerical
  equivalence to the paper or upstream outputs has NOT been established.

## Remaining priorities

- Replace the compatibility executor, one experiment at a time, with explicit
  functions that receive typed configurations and return structured results.
- Move reusable kernels to a real installable `operational_regimes` package;
  repair imports and remove wildcard imports only with numerical regression tests.
- Extract all seeds, sample counts, stopping criteria and thresholds; add schema
  validation, environment locking and completion/failure manifests.
- Validate resumed-result compatibility rather than trusting existing JSON.
- Recover the original drift support code/data; reconcile sample counts, the
  parametric path, ADMM tables and MIMO stopping/regularity diagnostics.
- This executor modifies process-wide argv/sys.path and is NOT thread-safe;
  execute parallel experiments in separate processes. Scientific dependencies
  and full tests are not enforced by the lightweight infrastructure suite.
