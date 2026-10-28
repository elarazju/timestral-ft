"""Round-one objective comparison; bounded sparse-origin protocol, not a leaderboard.

Run --phase preflight, then --phase run (includes an immutable protocol freeze), or
--phase all. No held-out test forecast is made until every selected model is hashed.
"""

import argparse
import gc
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Share the existing pinned download checks and train-only standardization.
from benchmark_suite import MODEL_ID, MODEL_REVISION, prepare_dataset, windows
from timestral.loading import resolve_checkpoint

from timestral_ft import DataConfig, FineTunedModel, TrainConfig, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import PortableForecaster, load_model
from timestral_ft.data import DataProcessor, WindowDataset

ROOT = Path(__file__).resolve().parents[1]
LEVELS = np.array([.01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                   .55, .6, .65, .7, .75, .8, .85, .9, .95, .99])
MODES = ('history_scaled', 'normalized')
SEEDS = (42, 43, 44)
DATASETS = ('ETTh1', 'ETTh2')


def immutable_json(path, value):
    """Resume only identical provenance; never silently replace a frozen file."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen provenance mismatch: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sparse_origins(start, stop, horizon=96):
    return np.arange(start, stop - horizon + 1, horizon, dtype=np.int64)


def seasonal_scales(training, seasonality=24):
    values = np.asarray(training, dtype=np.float64)
    result = np.nanmean(np.abs(values[seasonality:] - values[:-seasonality]), axis=0)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('MASE needs positive finite training-prefix seasonal scales')
    return result


def score_arrays(prediction, actual, mase_scales):
    """Masked WQL on 21 knots; separate nine-knot CRPS approximation.

    Aggregate numerator and |label| denominator over all targets/windows/leads;
    this is not the arithmetic mean of seven independently normalized scores.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if prediction.shape != actual.shape + (21,):
        raise ValueError('Prediction must have shape actual.shape + (21,)')
    valid = np.isfinite(actual)
    if not valid.any() or not np.isfinite(prediction[valid]).all():
        raise FloatingPointError('No labels or non-finite active prediction')
    error = actual[..., None] - prediction
    pinball = np.maximum(LEVELS * error, (LEVELS - 1) * error)
    sums = np.where(valid[..., None], pinball, 0).sum(axis=(0, 1, 2))
    absolute = float(np.abs(actual[valid]).sum())
    if absolute == 0:
        raise ValueError('WQL undefined for zero absolute label sum')
    per_q = 2 * sums / absolute
    middle = prediction[..., 10]
    residual = middle - actual
    lower, upper = prediction[..., 2], prediction[..., 18]
    nine = [int(np.flatnonzero(np.isclose(LEVELS, q))[0]) for q in np.arange(.1, 1, .1)]
    return {
        'masked_wql21': float(per_q.mean()),
        'crps9_approx': float(per_q[nine].mean()),
        'mae': float(np.abs(residual[valid]).mean()),
        'mse': float(np.square(residual[valid]).mean()),
        'mase24': float((np.abs(residual) / np.asarray(mase_scales)[None, :, None])[valid].mean()),
        'coverage80': float(((actual >= lower) & (actual <= upper))[valid].mean()),
        'width80': float((upper - lower)[valid].mean()),
        'windows': len(actual), 'scored_values': int(valid.sum()),
        'per_target_mae': np.nanmean(np.where(valid, np.abs(residual), np.nan), axis=(0, 2)).tolist(),
        'per_quantile_wql': per_q.tolist(),
    }


def source_identity():
    sources = ['scripts/run_report_round1.py', 'scripts/benchmark_suite.py']
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'src/timestral_ft').glob('*.py'))]
    return {name: file_sha256(ROOT / name) for name in sources}


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def release(device):
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()


def config(mode, seed, steps, settings):
    return TrainConfig(mode='lora', loss_space=mode, validation_loss_space='history_scaled',
                       context_length=256, prediction_length=96, max_steps=steps,
                       batch_size=4, gradient_accumulation=4, learning_rate=1e-4,
                       lora_rank=8, lora_alpha=16, eval_every=50, patience=3,
                       seed=seed, device=settings.device, num_threads=settings.threads)


def run_training(frame, manifest, cfg, directory, *, stop=None):
    directory = Path(directory)
    train_end, val_end = manifest['train'][1], manifest['validation'][1]
    if (directory / 'model/finetune.json').exists() and stop is None:
        metadata = json.loads((directory / 'model/finetune.json').read_text())
        processor = DataProcessor(DataConfig(time_column='date', target_columns=manifest['columns'],
                                             frequency='h')).fit(frame.iloc[:train_end])
        training = WindowDataset(processor, frame.iloc[:train_end], 256, 96)
        validation = WindowDataset(processor, frame.iloc[train_end:val_end], 256, 96,
                                   history=frame.iloc[:train_end], stride=96)
        expected = {'train_config': cfg.to_dict(), 'base_model': MODEL_ID,
                    'base_revision': MODEL_REVISION, 'preprocessing': processor.to_dict(),
                    'fingerprints': {'train': training.fingerprint(), 'validation': validation.fingerprint()}}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'Completed-run identity mismatch: {key} in {directory}')
        return FineTunedModel.from_pretrained(directory / 'model', device=cfg.device)
    return finetune(frame.iloc[:train_end], frame.iloc[train_end:val_end], model_id=MODEL_ID,
                    revision=MODEL_REVISION,
                    data_config=DataConfig(time_column='date', target_columns=manifest['columns'], frequency='h'),
                    train_config=cfg, output_dir=directory,
                    resume=directory / 'checkpoint.pt' if (directory / 'checkpoint.pt').exists() else None,
                    stop_after_steps=stop)


def preflight(settings):
    path = Path(settings.output) / 'preflight.json'
    identity = {'sources': source_identity(), 'device': settings.device, 'threads': settings.threads,
                'requested_steps': settings.steps, 'budget_hours': settings.budget_hours}
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['identity'] != identity:
            raise ValueError('Preflight identity mismatch; choose a new output/results pair')
        return existing
    frame, _, manifest = prepare_dataset('ETTh1', settings.data)
    measurements = []
    # Ten real optimizer steps per objective, batch4 x accumulation4, no test access.
    for mode in MODES:
        started = time.perf_counter()
        artifact = run_training(frame, manifest, config(mode, 42, settings.steps, settings),
                                Path(settings.output) / 'preflight' / mode, stop=10)
        synchronize(settings.device)
        elapsed = time.perf_counter() - started
        measurements.append({'loss_space': mode, 'steps': 10,
                             'wall_seconds': elapsed, 'seconds_per_step_conservative': elapsed / 10})
        del artifact
        release(settings.device)
    # Both modes x both datasets x three seeds; 40% overhead for validation/export/eval.
    seconds_per_step = max(x['seconds_per_step_conservative'] for x in measurements)
    choices = [n for n in (200, 100, 50) if n <= settings.steps]
    if settings.steps not in choices:
        choices.insert(0, settings.steps)
    selected = next((n for n in choices if seconds_per_step * n * 12 * 1.4 <= settings.budget_hours * 3600), None)
    if selected is None:
        raise RuntimeError('Even 50-step matrix exceeds budget; revise protocol before any test scoring')
    result = {'identity': identity, 'measurements': measurements, 'selected_steps': selected,
              'projected_seconds_including_40pct_overhead': seconds_per_step * selected * 12 * 1.4,
              'selection_rule': 'largest of requested 200/100/50 fitting budget; common setting for all arms'}
    write_json(path, result)
    return result


def freeze(settings, preflight_result, manifests):
    checkpoint = resolve_checkpoint(MODEL_ID, revision=MODEL_REVISION)
    protocol = {
        'version': 1, 'paper': 'https://arxiv.org/pdf/2610.37585',
        'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'model_sha256': file_sha256(checkpoint / 'model.safetensors'),
        'model_config_sha256': file_sha256(checkpoint / 'config.json'),
        'sources': source_identity(), 'datasets': manifests,
        'training': {mode: config(mode, 42, preflight_result['selected_steps'], settings).to_dict() for mode in MODES},
        'seeds': list(SEEDS), 'context': 256, 'horizon': 96,
        'validation_stride': 96, 'test_stride': 96,
        'selection': 'minimum common history_scaled validation pinball, eval_every50 plus final; no test feedback',
        'postprocessing': {'symmetric_averaging': True, 'clip_nonnegative': False,
                           'other_options': 'unchanged PortableForecaster/upstream defaults'},
        'metrics': 'global label-weighted masked WQL21; separate nine-quantile CRPS approximation; median MAE/MSE; MASE24 training-prefix denominator; 80% coverage/width',
        'metric_units': 'train-only per-target standardized units; MASE/WQL dimensionless',
        'statistical_scope': 'three-seed mean/sample std, no significance claim; 30 nonoverlapping forecast horizons per split',
        'normalization_causality': 'training-only external scaler; each forecast uses only prefix context; normalized loss refinement uses predicted median, never future label',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'device': settings.device,
                        'threads': settings.threads,
                        'git_head_before_round1': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'preflight': preflight_result,
        'limitations': ['short adaptation only, not official training reproduction',
                        'ETT pretraining overlap is unknown', 'sparse stride96 evaluation, not stride1 leaderboard',
                        'common history-scaled validation criterion may favor that objective'],
    }
    immutable_json(Path(settings.results) / 'protocol.json', protocol)
    immutable_json(Path(settings.output) / 'protocol.json', protocol)
    return protocol


def predict_split(model, values, manifest, split, output, device):
    indices = sparse_origins(*manifest[split])
    fc = PortableForecaster(model.eval(), max_context=256)
    predictions, actuals = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), 4):
            x, y = windows(values, indices[offset:offset + 4], 256, 96)
            q = fc.forecast(x, 96, symmetric_averaging=True, clip_nonnegative=False).quantiles
            predictions.append(q.detach().cpu().numpy())
            actuals.append(y)
    synchronize(device)
    prediction, actual = np.concatenate(predictions), np.concatenate(actuals)
    score = score_arrays(prediction, actual, seasonal_scales(values[:manifest['train'][1]]))
    score['evaluation_seconds'] = time.perf_counter() - started
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, prediction=prediction, actual=actual, origins=indices)
    score['predictions_sha256'] = file_sha256(output)
    return score


def run_matrix(settings, preflight_result):
    prepared = {name: prepare_dataset(name, settings.data) for name in DATASETS}
    protocol = freeze(settings, preflight_result, {k: v[2] for k, v in prepared.items()})
    protocol_sha = digest_json(protocol)
    selected = {}
    selected_path = Path(settings.results) / 'selected_models.json'
    frozen_selected = json.loads(selected_path.read_text()) if selected_path.exists() else None
    if frozen_selected and frozen_selected['protocol_sha256'] != protocol_sha:
        raise ValueError('Selected-model protocol mismatch')
    for name, (frame, _, manifest) in prepared.items():
        for mode in MODES:
            for seed in SEEDS:
                label = f'{name}_{mode}_s{seed}'
                run = Path(settings.output) / label
                if frozen_selected:
                    frozen_hash = frozen_selected['models'][label]['model_sha256']
                    if file_sha256(run / 'model/model.safetensors') != frozen_hash:
                        raise ValueError(f'Selected model changed after freeze: {label}')
                logging.info('TRAIN %s', label)
                artifact = run_training(frame, manifest, config(mode, seed, preflight_result['selected_steps'], settings), run)
                metadata = artifact.metadata
                selected[label] = {'model_sha256': file_sha256(run / 'model/model.safetensors'),
                                   'adapter_sha256': file_sha256(run / 'model/adapter.safetensors'),
                                   'best_step': metadata['best_step'], 'actual_steps': metadata['steps'],
                                   'best_validation_loss': metadata['best_validation_loss'],
                                   'training_seconds': metadata['training_seconds'],
                                   'trainable_parameters': metadata['trainable_parameters']}
                del artifact
                release(settings.device)
    # All checksums fixed before baseline or variant held-out test forecasting.
    immutable_json(Path(settings.results) / 'selected_models.json', {'protocol_sha256': protocol_sha, 'models': selected})
    for name, (_, values, manifest) in prepared.items():
        cases = [('base', None)] + [(mode, seed) for mode in MODES for seed in SEEDS]
        for mode, seed in cases:
            label = f'{name}_{mode}' if seed is None else f'{name}_{mode}_s{seed}'
            row_path = Path(settings.results) / f'{label}.json'
            if row_path.exists():
                if json.loads(row_path.read_text())['protocol_sha256'] != protocol_sha:
                    raise ValueError('Existing row protocol mismatch')
                continue
            if mode == 'base':
                model = load_model(MODEL_ID, revision=MODEL_REVISION, device=settings.device)
                metadata = {'training_seconds': 0., 'best_step': 0, 'actual_steps': 0,
                            'model_sha256': protocol['model_sha256']}
            else:
                metadata = selected[label]
                run = Path(settings.output) / label / 'model'
                if file_sha256(run / 'model.safetensors') != metadata['model_sha256']:
                    raise ValueError('Selected model changed after freeze')
                artifact = FineTunedModel.from_pretrained(run, device=settings.device)
                model = artifact.model
            row = {'dataset': name, 'mode': mode, 'seed': seed, 'protocol_sha256': protocol_sha, **metadata}
            for split in ('validation', 'test'):
                logging.info('SCORE %s %s', label, split)
                row[split] = predict_split(model, values, manifest, split,
                                          Path(settings.output) / 'predictions' / f'{label}_{split}.npz', settings.device)
            immutable_json(row_path, row)
            del model
            if mode != 'base':
                del artifact
            release(settings.device)
    return summarize(settings.results)


def summarize(directory):
    directory = Path(directory)
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('ETTh*.json'))]
    flattened = []
    for row in rows:
        for split in ('validation', 'test'):
            flattened.append({k: row[k] for k in ('dataset', 'mode', 'seed', 'training_seconds', 'best_step', 'actual_steps')}
                             | {'split': split} | {k: v for k, v in row[split].items() if not isinstance(v, list)})
    df = pd.DataFrame(flattened)
    if df.empty:
        raise ValueError('No experiment results')
    df.to_csv(directory / 'runs.csv', index=False)
    metrics = ['masked_wql21', 'crps9_approx', 'mae', 'mse', 'mase24', 'coverage80', 'width80', 'training_seconds']
    summary = df.groupby(['dataset', 'split', 'mode'], sort=True).agg({k: ['mean', 'std', 'count'] for k in metrics})
    summary.columns = ['_'.join(k) for k in summary.columns]
    summary.reset_index().to_csv(directory / 'summary.csv', index=False)
    return summary


def diagnostics(settings):
    from timestral_ft.compat import (
        differentiable_forecast,
        differentiable_normalized_forecast,
        normalized_pinball_loss,
        pinball_loss,
    )
    model = load_model(MODEL_ID, revision=MODEL_REVISION)
    contexts = {'constant': np.ones(256), 'level_shift': np.r_[np.ones(128), np.full(128, 10.)],
                'outlier': np.r_[np.ones(255), 100.]}
    output = []
    for name, series in contexts.items():
        context = torch.tensor(series, dtype=torch.float32)[None, None]
        labels = torch.full((1, 1, 96), float(series[-1]))
        prediction = differentiable_forecast(model, context, 96)
        normalized = differentiable_normalized_forecast(model, context, 96)
        raw_loss = pinball_loss(prediction, labels, context, model.config.quantile_knots)
        norm_loss = normalized_pinball_loss(normalized, labels, context, model.config.quantile_knots)
        output.append({'fixture': name, 'history_scaled_loss': float(raw_loss.detach()),
                       'normalized_loss': float(norm_loss.detach()),
                       'note': 'different objective units; numerical magnitude is not a quality ranking'})
    write_json(Path(settings.results) / 'objective_diagnostics.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'run', 'all', 'summarize', 'diagnostics'], default='all')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--budget-hours', type=float, default=8)
    parser.add_argument('--data', default='data/validation')
    parser.add_argument('--output', default='runs/report-round1')
    parser.add_argument('--results', default='results/report_round1')
    settings = parser.parse_args()
    torch.set_num_threads(settings.threads)
    Path(settings.output).mkdir(parents=True, exist_ok=True)
    Path(settings.results).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if settings.phase == 'summarize':
        print(summarize(settings.results).to_string())
    elif settings.phase == 'diagnostics':
        diagnostics(settings)
    else:
        timing = preflight(settings)
        print(json.dumps(timing, indent=2))
        if settings.phase in ('run', 'all'):
            run_matrix(settings, timing)
            diagnostics(settings)


if __name__ == '__main__':
    main()
