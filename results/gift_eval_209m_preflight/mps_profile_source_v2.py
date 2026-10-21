"""Measure equal-work one/two-process MPS throughput on frozen GIFT workloads.

The production evaluator is suspended during measurement and always resumed.
ABBA trials exclude model loading/warmup, retain official forecast settings,
and compare nine-quantile predictions across every trial and process.
"""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.cache/upstream/src'), str(ROOT / '.cache/gift-eval/src')]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / '.cache/gift209/mps_parallel_profile')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--case', action='append', choices=['electricity/15T/short', 'ett1/15T/long',
                                                      'bizitobs_l2c/5T/long', 'bizitobs_application/10S/long'])
    p.add_argument('--pause-pid', type=int)
    p.add_argument('--pause-pending-pid', type=int)
    p.add_argument('--worker-input', type=Path)
    p.add_argument('--worker-output', type=Path)
    p.add_argument('--start-file', type=Path)
    p.add_argument('--worker-jobs', type=int)
    p.add_argument('--task')
    return p.parse_args()


def command(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                          capture_output=True, text=True, check=False).stdout.strip()


def swap_usage():
    return subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True, text=True, check=False).stdout.strip()


def worker(args):
    import numpy as np
    import torch

    from timestral_ft.compat import PortableForecaster
    from timestral_ft.gift import load_gift_adapter

    torch.set_num_threads(args.threads)
    inputs = np.load(args.worker_input)
    context = inputs['context']
    covariates = inputs['past'] if 'past' in inputs else None
    horizon = int(inputs['horizon'])
    fc = PortableForecaster(load_gift_adapter(
        ROOT / 'runs/gift-eval-209m-search-v2/tasks' / args.task.replace('/', '__'), 'mps'), max_context=32768)
    indices = [next(i for i, level in enumerate(fc.quantile_levels) if abs(level-q/10) < 1e-9)
               for q in range(1, 10)]

    def forecast():
        result = fc.forecast(context, horizon, past_covariates=covariates)
        prediction = result.quantiles[..., indices].float().cpu().numpy()
        torch.mps.synchronize()
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Non-finite forecast')
        return prediction

    forecast()  # Warm Metal kernels and allocator before the timed barrier.
    write(args.worker_output / 'ready.json', dict(pid=os.getpid(), driver_bytes=torch.mps.driver_allocated_memory()))
    while not args.start_file.exists():
        time.sleep(.01)
    peak_driver = torch.mps.driver_allocated_memory()
    started = time.perf_counter()
    first = None
    for i in range(args.worker_jobs):
        prediction = forecast()
        if i == 0:
            first = prediction
        if not np.allclose(first, prediction, rtol=1e-4, atol=1e-5):
            raise ValueError('Within-worker predictions changed across repeats')
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
    elapsed = time.perf_counter()-started
    np.save(args.worker_output / 'prediction.npy', first)
    write(args.worker_output / 'done.json', dict(pid=os.getpid(), jobs=args.worker_jobs, seconds=elapsed,
                                               peak_driver_bytes=peak_driver,
                                               allocated_tensor_bytes=torch.mps.current_allocated_memory()))


def prepare(args):
    import itertools

    import numpy as np
    from gift_eval.data import Dataset

    from timestral_ft.artifacts import file_sha256

    os.environ['GIFT_EVAL'] = str(ROOT / '.cache/gift-eval-data')
    run = ROOT / 'runs/gift-eval-209m-search-v2'
    frozen = read(run / 'frozen_selection.json')
    if frozen['count'] != 97 or len(frozen['tasks']) != 97:
        raise ValueError('Complete frozen selection required before execution probes')
    hashes = {r['task']: r['adapter_sha256'] for r in frozen['tasks']}
    cases = []
    for key, name, batch in [('electricity/15T/short', 'electricity/15T', 3),
                              ('ett1/15T/long', 'ett1/15T', 1),
                              ('bizitobs_l2c/5T/long', 'bizitobs_l2c/5T', 1),
                              ('bizitobs_application/10S/long', 'bizitobs_application', 1)]:
        if args.case and key not in args.case:
            continue
        if file_sha256(run / 'tasks' / key.replace('/', '__') / 'adapter.safetensors') != hashes[key]:
            raise ValueError('Frozen adapter changed')
        term = key.rsplit('/', 1)[1]
        ds = Dataset(name, term=term, to_univariate=False)
        entries = list(itertools.islice(ds.test_data.input, batch))
        targets, covariates = [], []
        for entry in entries:
            t = np.asarray(entry['target'], dtype=np.float32)
            targets.append((t[None] if t.ndim == 1 else t)[:, -32768:])
            c = entry.get('past_feat_dynamic_real')
            if c is not None:
                c = np.asarray(c, dtype=np.float32)
                covariates.append((c[None] if c.ndim == 1 else c)[:, -32768:])
        length = max(t.shape[-1] for t in targets)

        def padded(arrays):
            return np.stack([np.pad(a, ((0, 0), (length-a.shape[-1], 0)), constant_values=np.nan)
                             for a in arrays])

        data = dict(context=padded(targets), horizon=np.array(ds.prediction_length))
        if covariates:
            data['past'] = padded(covariates)
        rows = data['context'].shape[1] + (data['past'].shape[1] if 'past' in data else 0)
        if batch > 1 and (batch*rows > 16 or batch*rows*(length+ds.prediction_length) > 2**17):
            raise ValueError('Probe batch exceeds original official predictor budget')
        path = args.output / f'{key.replace("/", "__")}.npz'
        np.savez(path, **data)
        cases.append(dict(task=key, input_path=str(path), input_sha256=file_sha256(path),
                          adapter_sha256=hashes[key], batch=batch, targets=data['context'].shape[1],
                          past_covariates=data['past'].shape[1] if 'past' in data else 0,
                          context=length, horizon=ds.prediction_length))
    return cases


def trial(args, case, workers, index):
    import numpy as np

    trial_dir = args.output / f'{case["task"].replace("/", "__")}_trial{index}_workers{workers}'
    trial_dir.mkdir()
    start_file = trial_dir / 'start.json'
    processes, handles = [], []
    launched = time.perf_counter()
    last_pressure_check = 0.
    try:
        for i in range(workers):
            directory = trial_dir / f'worker{i}'
            directory.mkdir()
            handle = (directory / 'output.log').open('w')
            handles.append(handle)
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  '--worker-input', case['input_path'], '--task', case['task'],
                                  '--worker-output', str(directory), '--start-file', str(start_file),
                                  '--worker-jobs', str(args.jobs//workers), '--threads', str(args.threads)],
                                 cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, HF_HUB_OFFLINE='1'))
            processes.append((directory, p))

        def healthy():
            nonlocal last_pressure_check
            for directory, p in processes:
                if p.poll() not in [None, 0]:
                    raise RuntimeError(f'Worker failed: {(directory / "output.log").read_text()[-2000:]}')
            if time.perf_counter()-launched > 1200:
                raise TimeoutError('Execution trial exceeded 20-minute safety deadline')
            if time.perf_counter()-last_pressure_check > 2:
                pressure = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True,
                                          check=False).stdout
                match = re.search(r'free percentage:\s*(\d+)%', pressure)
                if match and int(match[1]) < 10:
                    raise MemoryError('System memory headroom below 10%; aborting trial')
                last_pressure_check = time.perf_counter()

        while not all((directory / 'ready.json').exists() for directory, _ in processes):
            healthy()
            time.sleep(.05)
        before_swap = swap_usage()
        started = time.perf_counter()
        write(start_file, dict(started=True))
        while not all((directory / 'done.json').exists() for directory, _ in processes):
            healthy()
            time.sleep(.02)
        elapsed = time.perf_counter()-started
        receipts = [read(directory / 'done.json') for directory, _ in processes]
        predictions = [np.load(directory / 'prediction.npy') for directory, _ in processes]
        record = dict(workers=workers, jobs=args.jobs, wall_seconds=elapsed,
                      forecasts_per_second=args.jobs/elapsed,
                      sum_worker_peak_driver_bytes=sum(r['peak_driver_bytes'] for r in receipts),
                      swap_before=before_swap, swap_after=swap_usage(), worker_receipts=receipts,
                      startup_and_run_seconds=time.perf_counter()-launched)
        write(trial_dir / 'result.json', record)
        return record, predictions
    finally:
        for _, p in processes:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=20)
        for handle in handles:
            handle.close()


def main():
    args = arguments()
    if args.worker_input:
        worker(args)
        return
    if args.jobs < 4 or args.jobs % 2:
        raise ValueError('Use an even number of at least four equal-work jobs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    paused = []

    def interrupted(signum, frame):
        raise InterruptedError(f'Execution benchmark interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    report = dict(method='ABBA equal-work trials (1,2,2,1 processes), warmed barrier, full official forecast defaults',
                  jobs_per_trial=args.jobs, threads_per_process=args.threads,
                  production_model_selection_unchanged=True,
                  paused_production_model_memory_retained=True, cases=[], status='running')
    from timestral_ft.artifacts import file_sha256
    report.update(frozen_selection_sha256=file_sha256(ROOT / 'runs/gift-eval-209m-search-v2/frozen_selection.json'),
                  source_sha256=file_sha256(Path(__file__)),
                  full_dataset_evaluation=False, batch_repeated_with_identical_inputs=True,
                  official_max_rows=16, official_max_tokens=2**17, prediction_rtol=1e-4, prediction_atol=1e-5,
                  hardware=subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                          text=True, check=False).stdout.strip())
    try:
        # Pause the pending replacement first, so it cannot take over while the
        # production evaluator is suspended. The original supervisor stays live.
        for pid, fragment in [(args.pause_pending_pid, 'parallel_gift209_test.py'),
                              (args.pause_pid, 'run_gift209_search.py')]:
            if pid:
                if fragment not in command(pid):
                    raise ValueError(f'Unexpected process identity for {pid}')
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
        time.sleep(2)  # Let already submitted production GPU work drain.
        cases = prepare(args)
        write(args.output / 'inputs.json', cases)
        for case in cases:
            records, baseline, matches = [], None, True
            print(f'START CASE {case["task"]} rows={case["targets"]+case["past_covariates"]} context={case["context"]}', flush=True)
            for index, count in enumerate([1, 2, 2, 1]):
                record, predictions = trial(args, case, count, index)
                if baseline is None:
                    baseline = predictions[0]
                import numpy as np
                close = all(np.allclose(baseline, p, rtol=1e-4, atol=1e-5) for p in predictions)
                matches = matches and close
                record.update(predictions_match=close,
                              max_absolute_prediction_difference=max(float(np.abs(baseline-p).max()) for p in predictions))
                records.append(record)
                print(json.dumps(dict(task=case['task'], **record)), flush=True)
                write(args.output / 'progress.json', dict(case=case, completed_trials=records))
            single = statistics.median(r['wall_seconds'] for r in records if r['workers'] == 1)
            dual = statistics.median(r['wall_seconds'] for r in records if r['workers'] == 2)
            row = dict(**case, trials=records, single_median_seconds=single, dual_median_seconds=dual,
                       throughput_speedup=single/dual, all_predictions_match=matches)
            report['cases'].append(row)
            write(args.output / 'report.json', report)
            print('CASE RESULT', case['task'], 'speedup', round(single/dual, 3), 'parity', matches, flush=True)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        for pid in reversed(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        report['production_processes_resumed'] = paused
        write(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
