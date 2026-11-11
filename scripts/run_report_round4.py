"""Frozen three-arm synthetic curriculum experiment with shared DHF setup.

Train every adapter and freeze its checksum before any held-out forecasting.
Equal optimizer steps do not imply equal supervision or compute.
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

from timestral_ft import DataConfig, FineTunedModel, SyntheticConfig, TrainConfig, TrainingStage, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import PortableForecaster, load_model
from timestral_ft.data import DataProcessor, DHFSequenceDataset, WindowDataset, training_source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ETTh1', 'ETTh2')
STRATEGIES = ('real', 'mixed', 'staged')
CONDITIONS = ('complete',)
SEEDS = (42, 43, 44)
HORIZONS = (96, 192)
METRICS = ('masked_wql21', 'crps9_approx', 'mae', 'mse', 'mase24', 'coverage80', 'width80', 'training_seconds')
CONTEXT, TRAIN_HORIZON = 256, 192


def source_identity():
    names = ['scripts/run_report_round4.py', 'scripts/audit_report_round4.py',
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
                       training_strategy='dhf',
                       synthetic=SyntheticConfig(),
                       stages=[TrainingStage(steps // 2, 128, {'real': 0., 'mixed': .5, 'staged': 1.}[strategy]),
                               TrainingStage(steps - steps // 2, 256, 0. if strategy == 'real' else .5)],
                       context_length=CONTEXT, prediction_length=TRAIN_HORIZON, max_steps=steps,
                       batch_size=4, gradient_accumulation=4, learning_rate=1e-4,
                       lora_rank=8, lora_alpha=16, eval_every=50, patience=3,
                       dhf_min_context_patches=1, dhf_max_block_patches=6,
                       seed=seed, device=settings.device, num_threads=settings.threads)


def dataset_identity(frame, manifest, cfg):
    a, b = manifest['train'][1], manifest['validation'][1]
    processor = DataProcessor(DataConfig(time_column='date', target_columns=manifest['columns'], frequency='h')).fit(frame.iloc[:a])
    source_hash = training_source_fingerprint(processor, frame.iloc[:a])
    fingerprints = [DHFSequenceDataset(processor, frame.iloc[:a], stage.context_length, TRAIN_HORIZON,
                   min_context_steps=32).fingerprint() if stage.synthetic_probability < 1 else 'synthetic_only:' + source_hash
                   for stage in cfg.stages]
    return processor, {'train': fingerprints[0], 'stages': fingerprints, 'train_source': source_hash,
        'validation': WindowDataset(processor, frame.iloc[a:b], CONTEXT, TRAIN_HORIZON,
            history=frame.iloc[:a], stride=TRAIN_HORIZON).fingerprint()}



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
                'requested_steps': settings.steps, 'budget_hours': settings.budget_hours}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['identity'] != identity:
            raise ValueError('Preflight identity mismatch; use a fresh output/results pair')
        return existing
    frame, _, manifest = prepare_dataset('ETTh1', settings.data)
    measurements = []
    for strategy in STRATEGIES:
        started = time.perf_counter()
        artifact = run_training(frame, manifest, config(strategy, 42, 8, settings),
                                Path(settings.output) / 'preflight' / strategy, stop=6)
        synchronize(settings.device)
        seconds = max(time.perf_counter() - started, artifact.metadata['training_seconds'])
        measurements.append({'strategy': strategy, 'optimizer_steps': 6, 'wall_seconds': seconds,
                             'conservative_seconds_per_step': seconds / 6})
        del artifact
        release(settings.device)
    rates = {x['strategy']: x['conservative_seconds_per_step'] for x in measurements}
    chosen = None
    for steps in [x for x in (200, 100, 50) if x <= settings.steps]:
        projected = 3 * steps * sum(rates.values()) * 1.4
        if projected <= settings.budget_hours * 3600:
            chosen = steps
            break
    if chosen is None:
        raise RuntimeError('50-step main matrix exceeds budget; revise before scoring test')
    result = {'identity': identity, 'measurements': measurements, 'selected_steps': chosen,
              'projected_seconds_with_40pct_overhead': projected,
              'selection_rule': 'largest common 200/100/50 budget; ETTh1 training only, three arms, three seeds; never inspect test for selection'}
    write_json(path, result)
    return result


def training_configs(settings, timing):
    return {s: config(s, 42, timing['selected_steps'], settings).to_dict() for s in STRATEGIES}


def expected_labels(protocol):
    return {f'{d}_{s}_s{seed}' for d in ('ETTh1',) for s in protocol['training'] for seed in SEEDS}


def expected_rows(protocol):
    return {f'{d}_{s}_s{seed}_h{h}_{c}' for d in DATASETS for s in protocol['training']
            for seed in SEEDS for h in HORIZONS for c in CONDITIONS} | {
        f'{d}_base_h{h}_{c}' for d in DATASETS for h in HORIZONS for c in CONDITIONS}


def freeze(settings, timing, manifests):
    path = Path(settings.results) / 'protocol.json'
    old = json.loads(path.read_text()) if path.exists() else None
    checkpoint = resolve_checkpoint(MODEL_ID, revision=MODEL_REVISION)
    git_head = old['environment']['git_head_before_round4'] if old else subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    protocol = {
        'version': 4, 'paper': 'https://arxiv.org/pdf/2610.37585', 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'model_config_sha256': file_sha256(checkpoint / 'config.json'), 'sources': source_identity(),
        'datasets': manifests, 'training': training_configs(settings, timing), 'seeds': list(SEEDS),
        'context': CONTEXT, 'training_horizon': TRAIN_HORIZON, 'evaluation_horizons': list(HORIZONS), 'evaluation_conditions': list(CONDITIONS),
        'validation_selection_horizon': TRAIN_HORIZON, 'validation_selection_stride': TRAIN_HORIZON,
        'selection': 'minimum common history_scaled single-origin validation loss, every50 plus final; all adapter checksums frozen before any test forecast',
        'evaluation': 'nonoverlapping horizons, stride equal to evaluation horizon; context256; 30 origins H96,15 origins H192 each split',
        'dhf_layout': 'first half sequence320 then448, prefix32; uniform blocks1..6; full finite target supervision after prefix',
        'comparison': 'same two-stage context schedule, optimizer steps and target count; real/synthetic distribution differs; no gaps or role sampling',
        'transfer': 'only ETTh1 trains and selects; ETTh2 train segment supplies fixed scaler and MASE denominator, no target-domain weight updates or selection',
        'synthetic': 'project SyntheticConfig v1; independent mixture and generator RNG; not official unpublished recipe',
        'postprocessing': {'symmetric_averaging': True, 'clip_nonnegative': False},
        'metrics': 'global WQL21, separate CRPS9 approximation, median MAE/MSE/MASE24, coverage80/width80; train-only standardized units',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'device': settings.device,
                        'threads': settings.threads, 'git_head_before_round4': git_head},
        'preflight': timing,
        'budget_scope': '8-hour planning envelope using six-step two-context timing plus40pct allowance; no hard wall-clock cutoff is enforced by this script; actual elapsed cost is reported',
        'limitations': ['short adaptation is not official pretraining reproduction', 'ETT pretraining overlap unknown',
                        'sparse evaluation is not stride1 leaderboard', 'three seeds, no significance claim',
                        'same total supervision; different real/synthetic exposure, domain distributions and optimization paths',
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


def predict_split(model, values, manifest, split, horizon, path, device, condition):
    indices = sparse_origins(*manifest[split], horizon)
    forecaster = PortableForecaster(model.eval(), max_context=CONTEXT)
    predictions, actuals, contexts = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), 4):
            x, y = windows(values, indices[offset:offset + 4], CONTEXT, horizon)
            if condition == 'gap64':
                x[..., 128:192] = np.nan
            elif condition != 'complete':
                raise ValueError('Unknown evaluation condition')
            contexts.append(x.copy())
            q = forecaster.forecast(x, horizon, symmetric_averaging=True, clip_nonnegative=False).quantiles
            predictions.append(q.detach().cpu().numpy())
            actuals.append(y)
    synchronize(device)
    prediction, actual = np.concatenate(predictions), np.concatenate(actuals)
    score = score_arrays(prediction, actual, seasonal_scales(values[:manifest['train'][1]]))
    score['evaluation_seconds'] = time.perf_counter() - started
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction, actual=actual, origins=indices, context=np.concatenate(contexts))
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
    keys = ('real_samples', 'synthetic_samples', 'real_supervised_points', 'synthetic_supervised_points', 'supervised_points')
    counts = {key: sum(row[key] for row in records) for key in keys}
    by_stage = [{key: sum(row[key] for row in records if row['stage_index'] == i) for key in keys}
                for i in range(len(metadata['train_config']['stages']))]
    return {**counts, 'sampled_windows': counts['real_samples'] + counts['synthetic_samples'],
            'optimizer_steps': metadata['steps'], 'stages': by_stage,
            'forecast_blocks': sum(row['forecast_blocks'] for row in records), 'block_length_histogram': histogram,
            'note': 'accepted samples/exposures, not unique observations; no label sharing across source distributions'}


def selected_record(run, metadata):
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    selected_history = [row for row in history if row['step'] <= metadata['best_step']]
    selected_exposure = {key: sum(row[key] for row in selected_history) for key in
        ('real_samples', 'synthetic_samples', 'real_supervised_points', 'synthetic_supervised_points', 'supervised_points')}
    return {'best_stage_index': selected_history[-1]['stage_index'], 'selected_exposure': selected_exposure,
'model_sha256': file_sha256(run / 'model/model.safetensors'),
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
    for dataset in ('ETTh1',):
        frame, _, manifest = prepared[dataset]
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
                training_label = f'ETTh1_{strategy}_s{seed}'
                metadata = selected[training_label]
                artifact = FineTunedModel.from_pretrained(Path(settings.output) / training_label / 'model', device=settings.device)
                model = artifact.model
            for horizon in HORIZONS:
                for condition in CONDITIONS:
                    row_path = Path(settings.results) / f'{label}_h{horizon}_{condition}.json'
                    if row_path.exists():
                        row = json.loads(row_path.read_text())
                        expected = {'dataset': dataset, 'strategy': strategy, 'seed': seed, 'horizon': horizon, 'condition': condition,
                                    'protocol_sha256': protocol_sha, **metadata}
                        if any(row.get(key) != value for key, value in expected.items()):
                            raise ValueError(f'Existing row identity mismatch: {row_path}')
                        continue
                    row = {'dataset': dataset, 'strategy': strategy, 'seed': seed, 'horizon': horizon, 'condition': condition,
                           'protocol_sha256': protocol_sha, **metadata}
                    for split in ('validation', 'test'):
                        logging.info('SCORE %s H%s %s', label, horizon, split)
                        row[split] = predict_split(model, values, manifest, split, horizon,
                                                  Path(settings.output) / 'predictions' / f'{label}_h{horizon}_{condition}_{split}.npz', settings.device, condition)
                    immutable_json(row_path, row)
            del model
            if strategy != 'base':
                del artifact
            release(settings.device)
    summarize(settings.results)
    from audit_report_round4 import audit
    return audit(settings.results, settings.output, settings.data)


def aggregate_frames(rows):
    flattened = []
    for row in rows:
        for split in ('validation', 'test'):
            flattened.append({k: row[k] for k in ('dataset', 'strategy', 'seed', 'horizon', 'condition', 'training_seconds', 'best_step', 'actual_steps')}
                             | {'split': split} | {k: v for k, v in row[split].items() if not isinstance(v, list)})
    runs = pd.DataFrame(flattened)
    summary = runs.groupby(['dataset', 'split', 'horizon', 'condition', 'strategy'], sort=True).agg({k: ['mean', 'std', 'count'] for k in METRICS})
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
    parser.add_argument('--data', default='data/validation')
    parser.add_argument('--output', default='runs/report-round4')
    parser.add_argument('--results', default='results/report_round4')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.phase == 'summarize':
        print(summarize(args.results).to_string(index=False))
        from audit_report_round4 import audit
        print(json.dumps(audit(args.results, args.output, args.data), indent=2))
    elif args.phase == 'audit':
        from audit_report_round4 import audit
        print(json.dumps(audit(args.results, args.output, args.data), indent=2))
    else:
        timing = preflight(args)
        print(json.dumps(timing, indent=2), flush=True)
        if args.phase != 'preflight':
            print(json.dumps(run_matrix(args, timing), indent=2))


if __name__ == '__main__':
    main()
