"""Equal-work Bitbrains batch-size sweep with frozen weights and official prediction.

Production workers are suspended temporarily and resumed in finally. No test
labels, adapters, forecast horizons, or postprocessing settings are changed.
"""

import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import run_gift209_search as runner

ROOT = runner.ROOT


def command(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True,
                          text=True, check=False).stdout.strip()


def utc():
    return datetime.now(timezone.utc).isoformat()


def profile(args):
    import numpy as np
    import torch
    from gift_eval.data import Dataset

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.compat import PortableForecaster
    from timestral_ft.gift import load_gift_adapter

    torch.set_num_threads(4)
    frozen_file = ROOT / 'runs/gift-eval-209m-search-v2/frozen_selection.json'
    frozen = runner.read(frozen_file)
    if frozen['count'] != 97 or len(frozen['tasks']) != 97:
        raise ValueError('Complete frozen selection required')
    task = 'bitbrains_fast_storage/5T/short'
    directory = ROOT / 'runs/gift-eval-209m-search-v2/tasks' / task.replace('/', '__')
    expected = next(r['adapter_sha256'] for r in frozen['tasks'] if r['task'] == task)
    if file_sha256(directory/'adapter.safetensors') != expected:
        raise ValueError('Frozen adapter changed')
    namespace = runner.test_namespace()
    ds = Dataset('bitbrains_fast_storage/5T', term='short', to_univariate=False)
    indices = sorted(np.random.default_rng(42).choice(len(ds.hf_dataset)*ds.windows,
                                                     size=args.samples, replace=False).tolist())
    selected = set(indices)
    entries = []
    for index, entry in enumerate(ds.test_data.input):
        if index in selected:
            entries.append(entry)
        if len(entries) == args.samples:
            break
    if len(entries) != args.samples:
        raise ValueError('Insufficient real input windows')
    rows = ds.target_dim + ds.past_feat_dynamic_real_dim
    max_length = max(e['target'].shape[-1] for e in entries)
    input_hash = hashlib.sha256()
    for entry in entries:
        for field in ['target', 'past_feat_dynamic_real']:
            input_hash.update(np.asarray(entry[field], dtype=np.float32).tobytes())
    fc = PortableForecaster(load_gift_adapter(directory, 'mps'), max_context=32768)
    predictor = namespace['TimestralPredictor'](fc, ds.prediction_length, True)
    original_run = predictor._run
    observed = dict(peak_driver_bytes=0, calls=0, largest_batch=0)

    def measured_run(indices, targets, covariates):
        result = original_run(indices, targets, covariates)
        allocated = torch.mps.driver_allocated_memory()
        observed.update(peak_driver_bytes=max(observed['peak_driver_bytes'], allocated),
                        calls=observed['calls']+1,
                        largest_batch=max(observed['largest_batch'], len(indices)))
        if allocated > args.memory_cap_gib*2**30:
            raise MemoryError('Benchmark Metal allocation exceeded the configured safety cap')
        return result

    predictor._run = measured_run
    report = dict(task=task, samples=args.samples, targets=ds.target_dim,
                  past_covariates=ds.past_feat_dynamic_real_dim,
                  context_min=min(e['target'].shape[-1] for e in entries), context_max=max_length,
                  window_sampling='seed 42 uniform input-origin indices, fixed before profiling',
                  input_origin_indices=indices, input_values_sha256=input_hash.hexdigest(),
                  horizon=ds.prediction_length, weights='validation-frozen LoRA adapter',
                  adapter_sha256=expected, frozen_selection_sha256=file_sha256(frozen_file),
                  original_budget=dict(max_rows=16, max_tokens=2**17),
                  memory_cap_gib=args.memory_cap_gib,
                  prediction_rtol=1e-4, prediction_atol=1e-5, trials=[], summary=[],
                  source_sha256=file_sha256(Path(__file__)), torch=str(torch.__version__),
                  method='Same real windows through pinned official predictor; forward/reverse batch order; initialization and warmup excluded',
                  status='running')
    baseline = None
    for index, batch in enumerate([2, 4, 8, 16, 16, 8, 4, 2]):
        gc.collect()
        torch.mps.empty_cache()
        namespace['MAX_ROWS'] = 16 if batch == 2 else batch*rows
        namespace['MAX_TOKENS'] = 2**17 if batch == 2 else batch*rows*(max_length+ds.prediction_length)
        record = dict(index=index, batch=batch, max_rows=namespace['MAX_ROWS'],
                      max_tokens=namespace['MAX_TOKENS'])
        try:
            predictor.predict(entries[:batch])
            torch.mps.synchronize()
            observed.update(peak_driver_bytes=torch.mps.driver_allocated_memory(), calls=0, largest_batch=0)
            started = time.perf_counter()
            forecasts = predictor.predict(entries)
            torch.mps.synchronize()
            seconds = time.perf_counter()-started
            predictions = np.stack([f.forecast_array for f in forecasts])
            if not np.isfinite(predictions).all():
                raise FloatingPointError('Non-finite predictions')
            if baseline is None:
                baseline = predictions.copy()
                np.save(args.output/'baseline_predictions.npy', baseline)
            close = bool(np.allclose(baseline, predictions, rtol=1e-4, atol=1e-5))
            record.update(status='complete', seconds=seconds, windows_per_second=args.samples/seconds,
                          predictions_match=close,
                          max_absolute_difference=float(np.abs(predictions-baseline).max()),
                          max_relative_difference=float((np.abs(predictions-baseline)/np.maximum(np.abs(baseline), 1e-5)).max()),
                          **observed)
            np.save(args.output/f'predictions_trial{index}_batch{batch}.npy', predictions)
            print(json.dumps(record), flush=True)
            del forecasts, predictions
        except (RuntimeError, MemoryError) as error:
            record.update(status='failed', error=str(error), **observed)
            print(json.dumps(record), flush=True)
            if batch == 2:
                raise
            gc.collect()
            torch.mps.empty_cache()
        report['trials'].append(record)
        for size in [2, 4, 8, 16]:
            trials = [t for t in report['trials'] if t['batch'] == size and t['status'] == 'complete']
            if not trials:
                continue
            seconds_median = float(np.median([t['seconds'] for t in trials]))
            row = dict(batch=size, repeats=len(trials), median_seconds=seconds_median,
                       windows_per_second=args.samples/seconds_median,
                       predictions_match=all(t['predictions_match'] for t in trials),
                       peak_driver_bytes=max(t['peak_driver_bytes'] for t in trials))
            report['summary'] = [r for r in report['summary'] if r['batch'] != size] + [row]
        from timestral_ft.gift import atomic_json
        atomic_json(args.output/'progress.json', report)
    reference = next(r for r in report['summary'] if r['batch'] == 2)
    for row in report['summary']:
        row['speedup'] = row['windows_per_second']/reference['windows_per_second']
    report['status'] = 'complete'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/gift209/batch_profile')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--memory-cap-gib', type=float, default=12)
    parser.add_argument('--pause-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.samples < 16 or args.samples % 16:
        raise ValueError('Samples must be a positive multiple of 16')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    paused = []
    started_at = utc()
    report = dict(status='starting', started_at_utc=started_at)
    started = time.perf_counter()

    def terminate(signum, frame):
        raise InterruptedError(f'Batch sweep interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for pid in args.pause_pid:
            identity = command(pid)
            if not identity:
                continue
            if not any(name in identity for name in ['parallel_gift209_test.py', 'extend_gift209_cpu.py']):
                raise ValueError(f'Unexpected production process identity: {pid}')
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        pause_started = utc()
        time.sleep(2)
        report = profile(args)
        report['production_pause_started_at_utc'] = pause_started
    except BaseException as error:
        progress = args.output/'progress.json'
        if progress.exists():
            report.update(runner.read(progress))
        report.update(status='failed', error=str(error))
        raise
    finally:
        gc.collect()
        for pid in reversed(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        report.update(production_processes_resumed=paused, started_at_utc=started_at, ended_at_utc=utc(),
                      total_profile_seconds=time.perf_counter()-started)
        from timestral_ft.gift import atomic_json
        atomic_json(args.output/'report.json', report)


if __name__ == '__main__':
    main()
