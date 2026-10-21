"""Central command-line entry point for reproduction scripts."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runtime import load_config, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List supported scripts')
    run = sub.add_parser('run', help='Run one script (may be expensive)')
    run.add_argument('experiment')
    run.add_argument('--config', type=Path)
    run.add_argument('--output-dir', type=Path, default=Path('results'))
    run.add_argument('--script-args', nargs=argparse.REMAINDER, default=None,
                     help='Arguments for the legacy script; place this option last')
    run.add_argument('--allow-reconstructed-drift', action='store_true')
    args = parser.parse_args()
    names = sorted(p.stem for p in (ROOT / 'configs').glob('*.json'))
    if args.command == 'list':
        print('\n'.join(names))
        return
    if args.experiment not in names:
        parser.error('Unknown experiment; use: python main.py list')
    cfg = load_config(args.experiment, args.config)
    if args.experiment == 'make_fig_drift' and not args.allow_reconstructed_drift:
        parser.error('Original drift_experiment.py is missing. Reconstructed support requires explicit --allow-reconstructed-drift; exact paper reproduction is NOT guaranteed.')
    cfg['allow_reconstructed_drift'] = args.allow_reconstructed_drift
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(out)
        run_script(args.experiment, cfg, argv=args.script_args)
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    main()
