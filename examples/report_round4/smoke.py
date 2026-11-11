"""Prepare pinned ETTh1 train/validation tables and run the nested YAML example."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from benchmark_suite import prepare_dataset  # noqa: E402

from timestral_ft.cli import main as cli  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    frame, _, manifest = prepare_dataset('ETTh1', 'data/validation')
    output = Path('runs/report-round4-example-data')
    output.mkdir(parents=True, exist_ok=True)
    a, b = manifest['train'][1], manifest['validation'][1]
    frame.iloc[:a].to_parquet(output / 'train.parquet', index=False)
    frame.iloc[a:b].to_parquet(output / 'validation.parquet', index=False)
    if not args.prepare_only:
        cli(['train', '--config', 'examples/report_round4/smoke.yaml'])


if __name__ == '__main__':
    main()
