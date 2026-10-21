"""Give an idle CPU worker more frozen GIFT tasks without interrupting MPS.

The MPS worker already traverses all 97 tasks and honors each task lock. It
therefore waits for tasks claimed here and still completes any unclaimed work.
Only scheduling changes; predictions use the existing paired-test function.
"""

import argparse
import fcntl
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import parallel_gift209_test as parallel

runner = parallel.runner
ROOT = runner.ROOT


def cpu_queue(args, inventory):
    eligible = []
    for task in inventory:
        if parallel.complete(args, task['key']):
            continue
        shape = runner.read(runner.task_directory(args, task['key']) / 'training_summary.json')['data']
        short_taxi = task['key'].startswith('sz_taxi/15T/') and shape['train_length_max'] <= 2048
        if shape['series'] != 1 and not short_taxi:
            continue
        metadata = runner.read(ROOT / '.cache/gift-eval-data' / task['dataset'] / 'dataset_info.json')
        features = metadata['features']
        rows = features['target'].get('length', 1)
        covariates = features.get('past_feat_dynamic_real')
        if covariates:
            rows += covariates.get('length', 1)
        cost = min(32768, shape['train_length_max'] + 21*shape['horizon']) * rows * shape['series']
        eligible.append(dict(task=task['key'], priority=0 if short_taxi else 1,
                             estimated_input_work=cost, total_input_rows=rows,
                             training_shape=shape))
    return sorted(eligible, key=lambda x: (x['priority'], x['estimated_input_work'], x['task']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/gift-eval-209m-search-v2')
    p.add_argument('--results-dir', type=Path, default=ROOT / 'results/gift_eval_209m_lora_v2')
    p.add_argument('--threads', type=int, default=4)
    options = p.parse_args()
    import pandas as pd
    import torch

    from timestral_ft.artifacts import file_sha256

    args = parallel.experiment_args(options)
    torch.set_num_threads(options.threads)
    with (args.run_dir / '.cpu-extension.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = runner.read(args.results_dir / 'progress.json')
        if status['phase'] != 'frozen_paired_test':
            raise ValueError('CPU extension requires an active frozen paired test')
        if any(w['name'] == 'test_cpu' and w['returncode'] is None for w in status['workers']):
            raise ValueError('Original CPU worker is still running; refusing a second CPU worker')
        _, inventory, _ = runner.open_protocol(args)
        frozen_sha = file_sha256(args.run_dir / 'frozen_selection.json')
        frozen = runner.read(args.run_dir / 'frozen_selection.json')
        if frozen['count'] != 97 or len(frozen['tasks']) != 97:
            raise ValueError('All 97 adapter selections must be frozen')
        sources = {name: file_sha256(ROOT/name) for name in
                   ['scripts/extend_gift209_cpu.py', 'scripts/parallel_gift209_test.py',
                    'scripts/run_gift209_search.py']}
        manifest_path = args.results_dir / 'cpu_extension_plan.json'
        if manifest_path.exists():
            manifest = runner.read(manifest_path)
            if (manifest['frozen_selection_sha256'] != frozen_sha or manifest['source_sha256'] != sources
                    or manifest['threads'] != options.threads):
                raise ValueError('CPU extension identity/source changed; preserve the previous execution manifest')
        else:
            manifest = dict(queue=cpu_queue(args, inventory), frozen_selection_sha256=frozen_sha,
                            source_sha256=sources, threads=options.threads,
                            routing='Remaining single-series multivariate tasks plus short-context SZ taxi; shape-only ordering, no test errors used',
                            official_forecast_settings_unchanged=True)
            parallel.controller.atomic_json(manifest_path, manifest)
        for name in sources:
            snapshot = args.run_dir / 'cpu_extension_source_snapshot' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                snapshot.write_bytes((ROOT/name).read_bytes())
        state = dict(pid=os.getpid(), phase='running', device='cpu', threads=options.threads,
                     started_at_utc=datetime.now(timezone.utc).isoformat(), total_queue=len(manifest['queue']),
                     completed_by_this_worker=[], current_task=None, skipped_busy=[])

        def record():
            state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
            for path in [args.results_dir/'cpu_extension_status.json', args.run_dir/'cpu_extension_status.json']:
                parallel.controller.atomic_json(path, state)

        def terminate(signum, frame):
            raise SystemExit(f'CPU extension received signal {signum}; completed receipts permit resumption')

        signal.signal(signal.SIGTERM, terminate)

        try:
            record()
            args.device = 'cpu'
            namespace = runner.test_namespace()
            reference = pd.read_csv(ROOT / '.cache/gift-eval/results/Timestral-209M/all_results.csv').set_index('dataset')
            lookup = {task['key']: task for task in inventory}
            for item in manifest['queue']:
                key = item['task']
                if parallel.complete(args, key):
                    continue
                state['current_task'] = key
                record()
                started = time.perf_counter()
                if runner.test_task(args, lookup[key], namespace, reference, nonblocking=True):
                    state['completed_by_this_worker'].append(dict(task=key, seconds=time.perf_counter()-started))
                else:
                    state['skipped_busy'].append(key)
                record()
            state.update(phase='complete', current_task=None)
            record()
        except BaseException as error:
            state.update(phase='failed', error=str(error))
            record()
            raise


if __name__ == '__main__':
    main()
