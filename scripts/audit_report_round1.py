"""Independently reload all frozen artifacts/predictions and recompute metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_report_round1 import (
    DATASETS,
    MODES,
    SEEDS,
    digest_json,
    immutable_json,
    prepare_dataset,
    score_arrays,
    seasonal_scales,
    source_identity,
    sparse_origins,
)

from timestral_ft import DataConfig
from timestral_ft.artifacts import file_sha256
from timestral_ft.data import DataProcessor, WindowDataset


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def independently_score(prediction, actual, mase_scales):
    """Separate piecewise formulation; no dependency on the runner's scorer."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(actual)
    require(prediction.shape == actual.shape + (21,), 'Independent prediction shape mismatch')
    observed, forecasts = actual[valid], prediction[valid]
    require(len(observed) > 0 and np.isfinite(forecasts).all(), 'Independent metric invalid values')
    knots = np.array([.01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                      .55, .6, .65, .7, .75, .8, .85, .9, .95, .99])
    below = observed[:, None] >= forecasts
    losses = np.where(below, knots * (observed[:, None] - forecasts),
                      (1 - knots) * (forecasts - observed[:, None]))
    denom = np.abs(observed).sum()
    require(denom > 0, 'Independent WQL denominator is zero')
    weighted_losses = 2 * losses.sum(0) / denom
    delta = forecasts[:, 10] - observed
    scales = np.broadcast_to(np.asarray(mase_scales)[None, :, None], actual.shape)[valid]
    return {'masked_wql21': float(sum(weighted_losses) / 21),
            'crps9_approx': float(sum(weighted_losses[[2, 4, 6, 8, 10, 12, 14, 16, 18]]) / 9),
            'mae': float(np.abs(delta).sum() / len(observed)),
            'mse': float(np.dot(delta, delta) / len(observed)),
            'mase24': float((np.abs(delta) / scales).sum() / len(observed)),
            'coverage80': float(np.count_nonzero((observed >= forecasts[:, 2])
                                                & (observed <= forecasts[:, 18])) / len(observed)),
            'width80': float((forecasts[:, 18] - forecasts[:, 2]).sum() / len(observed))}


def validate_row_identity(row, selected, protocol):
    if row['mode'] == 'base':
        require(row['model_sha256'] == protocol['model_sha256'], 'Base row model SHA mismatch')
    else:
        for key, expected in selected.items():
            require(row.get(key) == expected, f'Row/selected ledger mismatch: {key}')


def validate_aggregate_tables(rows, results):
    flattened = []
    for row in rows:
        for split in ('validation', 'test'):
            flattened.append({k: row[k] for k in ('dataset', 'mode', 'seed', 'training_seconds',
                                                 'best_step', 'actual_steps')}
                             | {'split': split}
                             | {k: v for k, v in row[split].items() if not isinstance(v, list)})
    expected = pd.DataFrame(flattened)
    actual = pd.read_csv(Path(results) / 'runs.csv')
    order = ['dataset', 'split', 'mode', 'seed']
    expected = expected.sort_values(order).reset_index(drop=True)
    actual = actual.sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    metrics = ['masked_wql21', 'crps9_approx', 'mae', 'mse', 'mase24', 'coverage80', 'width80',
               'training_seconds']
    summary = expected.groupby(['dataset', 'split', 'mode'], sort=True).agg(
        {k: ['mean', 'std', 'count'] for k in metrics})
    summary.columns = ['_'.join(k) for k in summary.columns]
    saved_summary = pd.read_csv(Path(results) / 'summary.csv')
    pd.testing.assert_frame_equal(saved_summary, summary.reset_index(), check_dtype=False,
                                  rtol=1e-12, atol=1e-12)
    return len(actual), len(saved_summary)


def audit(results='results/report_round1', output='runs/report-round1', data='data/validation'):
    results, output = Path(results), Path(output)
    protocol = json.loads((results / 'protocol.json').read_text())
    frozen = json.loads((results / 'selected_models.json').read_text())
    require(protocol['sources'] == source_identity(), 'Runtime source differs from frozen source')
    protocol_sha = digest_json(protocol)
    require(frozen['protocol_sha256'] == protocol_sha, 'Frozen model protocol mismatch')
    expected_models = {f'{dataset}_{mode}_s{seed}' for dataset in DATASETS for mode in MODES for seed in SEEDS}
    expected_rows = expected_models | {f'{dataset}_base' for dataset in DATASETS}
    require(set(frozen['models']) == expected_models, 'Missing/extra selected models')
    actual_rows = {path.stem for path in results.glob('ETTh*.json')}
    require(actual_rows == expected_rows, 'Missing/extra result rows')
    checked_scores = 0
    rows = []
    for dataset in DATASETS:
        frame, values, manifest = prepare_dataset(dataset, data)
        train_end, val_end = manifest['train'][1], manifest['validation'][1]
        processor = DataProcessor(DataConfig(time_column='date', target_columns=manifest['columns'],
                                             frequency='h')).fit(frame.iloc[:train_end])
        fingerprints = {
            'train': WindowDataset(processor, frame.iloc[:train_end], 256, 96).fingerprint(),
            'validation': WindowDataset(processor, frame.iloc[train_end:val_end], 256, 96,
                                        history=frame.iloc[:train_end], stride=96).fingerprint(),
        }
        require(protocol['datasets'][dataset] == manifest, 'Dataset/scaler provenance mismatch')
        scales = seasonal_scales(values[:manifest['train'][1]])
        for label in sorted(x for x in expected_rows if x.startswith(dataset + '_')):
            row = json.loads((results / f'{label}.json').read_text())
            require(row['protocol_sha256'] == protocol_sha, f'Row protocol mismatch: {label}')
            validate_row_identity(row, frozen['models'].get(label), protocol)
            rows.append(row)
            if label in expected_models:
                artifact = output / label / 'model'
                saved = frozen['models'][label]
                for kind, filename in [('model', 'model.safetensors'), ('adapter', 'adapter.safetensors')]:
                    require(file_sha256(artifact / filename) == saved[kind + '_sha256'],
                            f'Artifact checksum changed: {label}/{filename}')
                metadata = json.loads((artifact / 'finetune.json').read_text())
                expected_cfg = dict(protocol['training'][row['mode']], seed=row['seed'])
                require(metadata['train_config'] == expected_cfg, f'Training config changed: {label}')
                require(metadata['base_weights_sha256'] == protocol['model_sha256'], 'Base model mismatch')
                require(metadata['base_model'] == protocol['model_id'], 'Base identity mismatch')
                require(metadata['base_revision'] == protocol['model_revision'], 'Base revision mismatch')
                require(metadata['preprocessing'] == processor.to_dict(), 'Preprocessing mismatch')
                require(metadata['fingerprints'] == fingerprints, 'Window data fingerprints mismatch')
                for row_key, meta_key in [('actual_steps', 'steps'), ('best_step', 'best_step'),
                                          ('best_validation_loss', 'best_validation_loss'),
                                          ('training_seconds', 'training_seconds'),
                                          ('trainable_parameters', 'trainable_parameters')]:
                    require(row[row_key] == metadata[meta_key], f'Artifact metadata mismatch: {row_key}')
            for split in ('validation', 'test'):
                path = output / 'predictions' / f'{label}_{split}.npz'
                require(file_sha256(path) == row[split]['predictions_sha256'], f'Prediction changed: {path}')
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved['origins'], sparse_origins(*manifest[split]))
                    actual = np.stack([values[t:t + 96].T for t in saved['origins']])
                    np.testing.assert_array_equal(saved['actual'], actual)
                    repeated = score_arrays(saved['prediction'], actual, scales)
                    independent = independently_score(saved['prediction'], actual, scales)
                for key, value in independent.items():
                    np.testing.assert_allclose(value, row[split][key], rtol=1e-12, atol=1e-12,
                                               err_msg=f'Independent metric mismatch: {label}/{split}/{key}')
                for key, value in repeated.items():
                    np.testing.assert_allclose(value, row[split][key], rtol=1e-12, atol=1e-12,
                                               err_msg=f'Metric mismatch: {label}/{split}/{key}')
                checked_scores += 1
    run_rows, summary_rows = validate_aggregate_tables(rows, results)
    record = {'passed': True, 'auditor_sha256': file_sha256(__file__), 'protocol_sha256': protocol_sha, 'selected_model_count': len(expected_models),
              'result_row_count': len(expected_rows), 'split_score_count': checked_scores, 'independent_metric_scalar_checks': checked_scores * 7, 'runs_csv_rows': run_rows, 'summary_csv_rows': summary_rows,
              'checks': ['runtime source SHA256', 'dataset SHA256 and train-only scaler',
                         'selected merged model and adapter SHA256', 'saved training configuration',
                         'prediction SHA256, origins and ground truth', 'all metrics recomputed from predictions',
                         'row identity vs frozen selected ledger', 'runs and seed mean/sample std aggregates',
                         'independent piecewise quantile/point/interval metric formulation'],
              'limitations': ['per-target/per-quantile details reuse runner; seven primary metrics independently formulated',
                              'no statistical significance claim', 'this provenance audit does not certify device numerical parity']}
    immutable_json(results / 'integrity_audit.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/report_round1')
    parser.add_argument('--output', default='runs/report-round1')
    parser.add_argument('--data', default='data/validation')
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.output, args.data), indent=2))


if __name__ == '__main__':
    main()
