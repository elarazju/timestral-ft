"""Frozen, bounded single-window versus native-DHF adaptation experiment.

Train every adapter and freeze its checksum before any held-out forecasting.
This is an equal-optimizer-step comparison, not an equal-compute experiment.
"""

import argparse
import gc
import json
import logging
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from benchmark_suite import MODEL_ID, MODEL_REVISION, prepare_dataset, windows
from run_report_round1 import digest_json, immutable_json, score_arrays, seasonal_scales, sparse_origins
from timestral.loading import resolve_checkpoint

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import PortableForecaster, load_model
from timestral_ft.data import DataProcessor, DHFSequenceDataset, WindowDataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ETTh1', 'ETTh2')
STRATEGIES = ('single_window', 'dhf')
SEEDS = (42, 43, 44)
HORIZONS = (96, 192)
METRICS = ('masked_wql21', 'crps9_approx', 'mae', 'mse', 'mase24', 'coverage80', 'width80', 'training_seconds')
CONTEXT, TRAIN_HORIZON = 256, 192


def source_identity():
    names = ['scripts/run_report_round2.py', 'scripts/audit_report_round2.py',
             'scripts/run_report_round1.py', 'scripts/audit_report_round1.py', 'scripts/benchmark_suite.py']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'src/timestral_ft').glob('*.py'))]
    return {name: file_sha256(ROOT / name) for name in names}


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def release(device):
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def config(strategy, seed, steps, settings):
    return TrainConfig(mode='lora', loss_space='normalized', validation_loss_space='history_scaled',
                       training_strategy='dhf' if strategy.startswith('dhf') else 'window',
                       context_length=CONTEXT, prediction_length=TRAIN_HORIZON, max_steps=steps,
                       batch_size=4, gradient_accumulation=4, learning_rate=1e-4,
                       lora_rank=8, lora_alpha=16, eval_every=50, patience=3,
                       dhf_min_context_patches=1, dhf_max_block_patches=6,
                       seed=seed, device=settings.device, num_threads=settings.threads)


def dataset_identity(frame, manifest, cfg):
    a, b = manifest['train'][1], manifest['validation'][1]
    processor = DataProcessor(DataConfig(time_column='date', target_columns=manifest['columns'], frequency='h')).fit(frame.iloc[:a])
    return processor, {
        'train': (DHFSequenceDataset(processor, frame.iloc[:a], CONTEXT, TRAIN_HORIZON, min_context_steps=32)
                  if cfg.training_strategy == 'dhf' else WindowDataset(processor, frame.iloc[:a], CONTEXT, TRAIN_HORIZON)).fingerprint(),
        'validation': WindowDataset(processor, frame.iloc[a:b], CONTEXT, TRAIN_HORIZON,
                                    history=frame.iloc[:a], stride=TRAIN_HORIZON).fingerprint(),
    }


class MemorySampler:
    """Sample allocator counters; sampling can miss brief peaks, RSS is process-wide."""

    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            if self.device == 'mps':
                self.samples.append((torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()))
            self.stop.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def record(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'sampled_mps_live_bytes_max': max((a for a, _ in self.samples), default=None),
                'sampled_mps_driver_bytes_max': max((b for _, b in self.samples), default=None),
                'process_cumulative_max_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024),
                'memory_scope': 'MPS allocator counters sampled every100ms across this train invocation; allocator peaks may be missed; RSS cumulative process high-water mark, not per-model incremental RSS'}


def run_training(frame, manifest, cfg, directory, stop=None):
    directory = Path(directory)
    a, b = manifest['train'][1], manifest['validation'][1]
    exported = directory / 'model/finetune.json'
    if exported.exists() and stop is None:
        metadata = json.loads(exported.read_text())
        processor, fingerprints = dataset_identity(frame, manifest, cfg)
        expected = {'train_config': cfg.to_dict(), 'base_model': MODEL_ID, 'base_revision': MODEL_REVISION,
                    'preprocessing': processor.to_dict(), 'fingerprints': fingerprints, 'completed': True}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'Completed-run identity mismatch: {key} in {directory}')
        return FineTunedModel.from_pretrained(directory / 'model', device=cfg.device)
    with MemorySampler(cfg.device) as sampler:
        artifact = finetune(frame.iloc[:a], frame.iloc[a:b], model_id=MODEL_ID, revision=MODEL_REVISION,
                            data_config=DataConfig(time_column='date', target_columns=manifest['columns'], frequency='h'),
                            train_config=cfg, output_dir=directory,
                            resume=directory / 'checkpoint.pt' if (directory / 'checkpoint.pt').exists() else None,
                            stop_after_steps=stop)
    memory = sampler.record()
    memory_path = directory / 'execution_memory.json'
    if memory_path.exists():
        previous = json.loads(memory_path.read_text())
        for key in ('sampled_mps_live_bytes_max', 'sampled_mps_driver_bytes_max', 'process_cumulative_max_rss_bytes'):
            if previous[key] is not None:
                memory[key] = max(previous[key], memory[key] or 0)
    write_json(memory_path, memory)
    return artifact


def preflight(settings):
    path = Path(settings.output) / 'preflight.json'
    identity = {'sources': source_identity(), 'device': settings.device, 'threads': settings.threads,
                'requested_steps': settings.steps, 'budget_hours': settings.budget_hours,
                'supplementary_requested': settings.supplementary}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['identity'] != identity:
            raise ValueError('Preflight identity mismatch; use a fresh output/results pair')
        return existing
    frame, _, manifest = prepare_dataset('ETTh1', settings.data)
    measurements = []
    for strategy in STRATEGIES:
        started = time.perf_counter()
        artifact = run_training(frame, manifest, config(strategy, 42, settings.steps, settings),
                                Path(settings.output) / 'preflight' / strategy, stop=10)
        synchronize(settings.device)
        seconds = max(time.perf_counter() - started, artifact.metadata['training_seconds'])
        measurements.append({'strategy': strategy, 'optimizer_steps': 10, 'wall_seconds': seconds,
                             'conservative_seconds_per_step': seconds / 10})
        del artifact
        release(settings.device)
    rates = {x['strategy']: x['conservative_seconds_per_step'] for x in measurements}
    chosen = None
    for steps in [x for x in (200, 100, 50) if x <= settings.steps]:
        projected = 6 * steps * sum(rates.values()) * 1.4
        if projected <= settings.budget_hours * 3600:
            chosen = steps
            break
    if chosen is None:
        raise RuntimeError('50-step main matrix exceeds budget; revise before scoring test')
    supplementary_steps = max(1, int(chosen * TRAIN_HORIZON / (CONTEXT + TRAIN_HORIZON - 32)))
    extra = 6 * supplementary_steps * rates['dhf'] * 1.4
    include = bool(settings.supplementary and projected + extra <= settings.budget_hours * 3600)
    result = {'identity': identity, 'measurements': measurements, 'selected_steps': chosen,
              'supplementary_included': include, 'supplementary_steps': supplementary_steps if include else None,
              'projected_seconds_with_40pct_overhead': projected + (extra if include else 0),
              'selection_rule': 'largest common 200/100/50 main budget; add fixed floor(steps*192/416) DHF control only if requested and fitting',
              'supplementary_limitation': 'approximate matched labels, different optimizer schedule and actual compute; no causal attribution to sharing alone'}
    write_json(path, result)
    return result


def training_configs(settings, timing):
    result = {s: config(s, 42, timing['selected_steps'], settings).to_dict() for s in STRATEGIES}
    if timing['supplementary_included']:
        result['dhf_label_control'] = config('dhf_label_control', 42, timing['supplementary_steps'], settings).to_dict()
    return result


def expected_labels(protocol):
    return {f'{d}_{s}_s{seed}' for d in DATASETS for s in protocol['training'] for seed in SEEDS}


def expected_rows(protocol):
    return {f'{label}_h{h}' for label in expected_labels(protocol) for h in HORIZONS} | {
        f'{d}_base_h{h}' for d in DATASETS for h in HORIZONS}


def freeze(settings, timing, manifests):
    path = Path(settings.results) / 'protocol.json'
    old = json.loads(path.read_text()) if path.exists() else None
    checkpoint = resolve_checkpoint(MODEL_ID, revision=MODEL_REVISION)
    git_head = old['environment']['git_head_before_round2'] if old else subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protocol = {
        'version': 2, 'paper': 'https://arxiv.org/pdf/2610.37585', 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'model_config_sha256': file_sha256(checkpoint / 'config.json'), 'sources': source_identity(),
        'datasets': manifests, 'training': training_configs(settings, timing), 'seeds': list(SEEDS),
        'context': CONTEXT, 'training_horizon': TRAIN_HORIZON, 'evaluation_horizons': list(HORIZONS),
        'validation_selection_horizon': TRAIN_HORIZON, 'validation_selection_stride': TRAIN_HORIZON,
        'selection': 'minimum common history_scaled single-origin validation loss, every50 plus final; all adapter checksums frozen before any test forecast',
        'evaluation': 'nonoverlapping horizons, stride equal to evaluation horizon; context256; 30 origins H96,15 origins H192 each split',
        'dhf_layout': 'sequence448, prefix32, remaining416 supervised; uniform block length1..6 patches, final block truncated; layout shared across all variables in an item; separate checkpointed layout RNG seed=seed XOR0x444846 leaves window-index RNG unchanged',
        'comparison': 'same optimizer steps and sampled448point windows; single_window labels192, DHF labels416 per variable/item; differing origins, context and lead distributions; unequal work',
        'postprocessing': {'symmetric_averaging': True, 'clip_nonnegative': False},
        'metrics': 'global WQL21, separate CRPS9 approximation, median MAE/MSE/MASE24, coverage80/width80; train-only standardized units',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'device': settings.device,
                        'threads': settings.threads, 'git_head_before_round2': git_head},
        'preflight': timing,
        'budget_scope': '8-hour planning envelope using ten-step timing plus40pct allowance; no hard wall-clock cutoff is enforced by this script; actual elapsed cost is reported',
        'limitations': ['short adaptation is not official pretraining reproduction', 'ETT pretraining overlap unknown',
                        'sparse evaluation is not stride1 leaderboard', 'three seeds, no significance claim',
                        'DHF differing supervision/context/lead distribution confounds effect attribution',
                        'sampling distribution chosen by this project; paper does not publish exact recipe'],
    }
    immutable_json(path, protocol)
    immutable_json(Path(settings.output) / 'protocol.json', protocol)
    snapshot = Path(settings.output) / 'source_snapshot'
    for name, checksum in protocol['sources'].items():
        destination = snapshot / name
        if destination.exists():
            if file_sha256(destination) != checksum:
                raise ValueError(f'Source snapshot changed: {name}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    return protocol


def predict_split(model, values, manifest, split, horizon, path, device):
    indices = sparse_origins(*manifest[split], horizon)
    forecaster = PortableForecaster(model.eval(), max_context=CONTEXT)
    predictions, actuals = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), 4):
            x, y = windows(values, indices[offset:offset + 4], CONTEXT, horizon)
            q = forecaster.forecast(x, horizon, symmetric_averaging=True, clip_nonnegative=False).quantiles
            predictions.append(q.detach().cpu().numpy())
            actuals.append(y)
    synchronize(device)
    prediction, actual = np.concatenate(predictions), np.concatenate(actuals)
    score = score_arrays(prediction, actual, seasonal_scales(values[:manifest['train'][1]]))
    score['evaluation_seconds'] = time.perf_counter() - started
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction, actual=actual, origins=indices)
    score['predictions_sha256'] = file_sha256(path)
    return score


def exposure_record(run, metadata):
    records = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    if len(records) != metadata['steps'] or [r['step'] for r in records] != list(range(1, metadata['steps'] + 1)):
        raise ValueError(f'Noncontiguous training history: {run}')
    histogram = {}
    for record in records:
        for length, count in record.get('block_length_histogram', {}).items():
            histogram[length] = histogram.get(length, 0) + count
    samples = metadata['steps'] * metadata['train_config']['batch_size'] * metadata['train_config']['gradient_accumulation']
    is_dhf = metadata['train_config']['training_strategy'] == 'dhf'
    distances = [sum(count for length, count in histogram.items() if int(length) * 32 >= lead) * 7
                 for lead in range(1, TRAIN_HORIZON + 1)] if is_dhf else [samples * 7] * TRAIN_HORIZON
    return {'sampled_windows': samples, 'optimizer_steps': metadata['steps'],
            'sampled_raw_sequence_points_including_targets': samples * (CONTEXT + TRAIN_HORIZON) * 7,
            'encoded_history_point_exposure': samples * 7 * (448 if is_dhf else CONTEXT),
            'two_stream_token_slot_exposure': samples * 7 * (28 if is_dhf else 14),
            'token_count_scope': 'geometric history+decoder slots including inactive DHF prefix; excludes layers/attention work; not a FLOPs measurement',
            'supervised_points_by_forecast_distance_1_to_192': distances,
            'supervised_points': sum(record['supervised_points'] for record in records),
            'forecast_blocks': sum(record.get('forecast_blocks', 0) for record in records) if metadata['train_config']['training_strategy'] == 'dhf' else samples,
            'forecast_block_count_scope': 'DHF histogram per example; single_window one block per sampled example',
            'block_length_histogram': histogram,
            'note': 'counts include all seven targets; sampled windows may repeat; raw sequence point counts are not transformer token FLOPs'}


def selected_record(run, metadata):
    return {'model_sha256': file_sha256(run / 'model/model.safetensors'),
            'adapter_sha256': file_sha256(run / 'model/adapter.safetensors'),
            'best_step': metadata['best_step'], 'actual_steps': metadata['steps'],
            'best_validation_loss': metadata['best_validation_loss'],
            'training_seconds': metadata['training_seconds'],
            'trainable_parameters': metadata['trainable_parameters'],
            'training_exposure': exposure_record(run, metadata),
            'history_sha256': file_sha256(run / 'history.jsonl'),
            'execution_memory': json.loads((run / 'execution_memory.json').read_text()),
            'execution_memory_sha256': file_sha256(run / 'execution_memory.json')}


def run_matrix(settings, timing):
    prepared = {name: prepare_dataset(name, settings.data) for name in DATASETS}
    protocol = freeze(settings, timing, {k: v[2] for k, v in prepared.items()})
    protocol_sha = digest_json(protocol)
    selected_path = Path(settings.results) / 'selected_models.json'
    frozen = json.loads(selected_path.read_text()) if selected_path.exists() else None
    if frozen and (frozen['protocol_sha256'] != protocol_sha or set(frozen['models']) != expected_labels(protocol)):
        raise ValueError('Frozen selected-model identity mismatch')
    selected = {}
    for dataset, (frame, _, manifest) in prepared.items():
        for strategy, cfg_dict in protocol['training'].items():
            for seed in SEEDS:
                label = f'{dataset}_{strategy}_s{seed}'
                run = Path(settings.output) / label
                logging.info('TRAIN %s', label)
                artifact = run_training(frame, manifest, TrainConfig(**dict(cfg_dict, seed=seed)), run)
                selected[label] = selected_record(run, artifact.metadata)
                if frozen and selected[label] != frozen['models'][label]:
                    raise ValueError(f'Selected artifact changed: {label}')
                del artifact
                release(settings.device)
    immutable_json(selected_path, {'protocol_sha256': protocol_sha, 'models': selected})
    for dataset, (_, values, manifest) in prepared.items():
        cases = [('base', None)] + [(s, seed) for s in protocol['training'] for seed in SEEDS]
        for strategy, seed in cases:
            label = f'{dataset}_{strategy}' if seed is None else f'{dataset}_{strategy}_s{seed}'
            if strategy == 'base':
                model = load_model(MODEL_ID, revision=MODEL_REVISION, device=settings.device)
                metadata = {'training_seconds': 0., 'best_step': 0, 'actual_steps': 0,
                            'model_sha256': protocol['model_sha256']}
            else:
                metadata = selected[label]
                artifact = FineTunedModel.from_pretrained(Path(settings.output) / label / 'model', device=settings.device)
                model = artifact.model
            for horizon in HORIZONS:
                row_path = Path(settings.results) / f'{label}_h{horizon}.json'
                if row_path.exists():
                    row = json.loads(row_path.read_text())
                    expected = {'dataset': dataset, 'strategy': strategy, 'seed': seed, 'horizon': horizon,
                                'protocol_sha256': protocol_sha, **metadata}
                    if any(row.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'Existing row identity mismatch: {row_path}')
                    continue
                row = {'dataset': dataset, 'strategy': strategy, 'seed': seed, 'horizon': horizon,
                       'protocol_sha256': protocol_sha, **metadata}
                for split in ('validation', 'test'):
                    logging.info('SCORE %s H%s %s', label, horizon, split)
                    row[split] = predict_split(model, values, manifest, split, horizon,
                                              Path(settings.output) / 'predictions' / f'{label}_h{horizon}_{split}.npz', settings.device)
                immutable_json(row_path, row)
            del model
            if strategy != 'base':
                del artifact
            release(settings.device)
    summarize(settings.results)
    from audit_report_round2 import audit
    return audit(settings.results, settings.output, settings.data)


def aggregate_frames(rows):
    flattened = []
    for row in rows:
        for split in ('validation', 'test'):
            flattened.append({k: row[k] for k in ('dataset', 'strategy', 'seed', 'horizon', 'training_seconds', 'best_step', 'actual_steps')}
                             | {'split': split} | {k: v for k, v in row[split].items() if not isinstance(v, list)})
    runs = pd.DataFrame(flattened)
    summary = runs.groupby(['dataset', 'split', 'horizon', 'strategy'], sort=True).agg({k: ['mean', 'std', 'count'] for k in METRICS})
    summary.columns = ['_'.join(k) for k in summary.columns]
    return runs, summary.reset_index()


def summarize(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    paths = list(directory.glob('ETTh*.json'))
    if {p.stem for p in paths} != expected_rows(protocol):
        raise ValueError('Cannot summarize incomplete/extra result rows')
    rows = [json.loads(p.read_text()) for p in sorted(paths)]
    sha = digest_json(protocol)
    if any(row['protocol_sha256'] != sha for row in rows):
        raise ValueError('Cannot summarize rows from a different protocol')
    runs, summary = aggregate_frames(rows)
    runs.to_csv(directory / 'runs.csv', index=False)
    summary.to_csv(directory / 'summary.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'run', 'all', 'summarize', 'audit'], default='all')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--steps', type=int, choices=[50, 100, 200], default=200)
    parser.add_argument('--budget-hours', type=float, default=8)
    parser.add_argument('--supplementary', action='store_true')
    parser.add_argument('--data', default='data/validation')
    parser.add_argument('--output', default='runs/report-round2')
    parser.add_argument('--results', default='results/report_round2')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.phase == 'summarize':
        print(summarize(args.results).to_string(index=False))
        from audit_report_round2 import audit
        print(json.dumps(audit(args.results, args.output, args.data), indent=2))
    elif args.phase == 'audit':
        from audit_report_round2 import audit
        print(json.dumps(audit(args.results, args.output, args.data), indent=2))
    else:
        timing = preflight(args)
        print(json.dumps(timing, indent=2), flush=True)
        if args.phase != 'preflight':
            print(json.dumps(run_matrix(args, timing), indent=2))


if __name__ == '__main__':
    main()
