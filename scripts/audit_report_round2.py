"""Fail-closed independent numerical and provenance audit of round-two results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_report_round1 import independently_score, require
from run_report_round2 import (
    CONTEXT,
    DATASETS,
    METRICS,
    TRAIN_HORIZON,
    dataset_identity,
    digest_json,
    expected_labels,
    expected_rows,
    exposure_record,
    immutable_json,
    prepare_dataset,
    score_arrays,
    seasonal_scales,
    source_identity,
    sparse_origins,
)

from timestral_ft import TrainConfig
from timestral_ft.artifacts import file_sha256


def validate_row_identity(row, selected, protocol):
    if row['strategy'] == 'base':
        require(row['model_sha256'] == protocol['model_sha256'], 'Base model identity mismatch')
        require(row['actual_steps'] == row['best_step'] == row['training_seconds'] == 0, 'Base training metadata mismatch')
    else:
        for key, value in selected.items():
            require(row.get(key) == value, f'Row/selected model mismatch: {key}')


def validate_aggregate_tables(rows, directory):
    # Independently rebuild flattening, grouping and sample-standard-deviation tables.
    entries = []
    for row in rows:
        for split in ['validation', 'test']:
            entry = {name: row[name] for name in ['dataset', 'strategy', 'seed', 'horizon',
                                                'training_seconds', 'best_step', 'actual_steps']}
            entry['split'] = split
            entry.update({key: value for key, value in row[split].items() if not isinstance(value, list)})
            entries.append(entry)
    expected = pd.DataFrame(entries)
    order = ['dataset', 'split', 'horizon', 'strategy', 'seed']
    saved = pd.read_csv(Path(directory) / 'runs.csv')
    pd.testing.assert_frame_equal(saved.sort_values(order).reset_index(drop=True),
                                  expected.sort_values(order).reset_index(drop=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    expected_summary = expected.groupby(['dataset', 'split', 'horizon', 'strategy'], sort=True).agg(
        {metric: ['mean', 'std', 'count'] for metric in METRICS})
    expected_summary.columns = ['_'.join(column) for column in expected_summary.columns]
    pd.testing.assert_frame_equal(pd.read_csv(Path(directory) / 'summary.csv'), expected_summary.reset_index(),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    return len(saved), len(expected_summary)


def audit(results='results/report_round2', output='runs/report-round2', data='data/validation'):
    results, output = Path(results), Path(output)
    protocol = json.loads((results / 'protocol.json').read_text())
    selected = json.loads((results / 'selected_models.json').read_text())
    require(protocol['sources'] == source_identity(), 'Runtime sources differ from frozen sources')
    for name, checksum in protocol['sources'].items():
        require(file_sha256(output / 'source_snapshot' / name) == checksum, 'Execution source snapshot differs')
    sha = digest_json(protocol)
    require(json.loads((output / 'protocol.json').read_text()) == protocol, 'Execution protocol differs')
    require(selected['protocol_sha256'] == sha, 'Selected model protocol mismatch')
    require(set(selected['models']) == expected_labels(protocol), 'Incomplete/extra selected models')
    require({p.stem for p in results.glob('ETTh*.json')} == expected_rows(protocol), 'Incomplete/extra result rows')
    rows, score_count = [], 0
    for dataset in DATASETS:
        frame, values, manifest = prepare_dataset(dataset, data)
        require(manifest == protocol['datasets'][dataset], 'Data/scaler provenance mismatch')
        scales = seasonal_scales(values[:manifest['train'][1]])
        for label in sorted(x for x in expected_labels(protocol) if x.startswith(dataset + '_')):
            run = output / label
            saved = selected['models'][label]
            metadata = json.loads((run / 'model/finetune.json').read_text())
            strategy = label[len(dataset) + 1:label.rfind('_s')]
            seed = int(label[label.rfind('_s') + 2:])
            expected_cfg = dict(protocol['training'][strategy], seed=seed)
            processor, fingerprints = dataset_identity(frame, manifest, TrainConfig(**expected_cfg))
            require(metadata['train_config'] == expected_cfg, 'Training configuration mismatch')
            require(metadata['base_model'] == protocol['model_id'] and metadata['base_revision'] == protocol['model_revision'], 'Base reference mismatch')
            require(metadata['base_weights_sha256'] == protocol['model_sha256'], 'Base checksum mismatch')
            require(metadata['preprocessing'] == processor.to_dict(), 'Processor mismatch')
            require(metadata['fingerprints'] == fingerprints, 'Window fingerprint mismatch')
            require(metadata['completed'], 'Training schedule incomplete')
            require(file_sha256(run / 'history.jsonl') == saved['history_sha256'], 'Training history checksum mismatch')
            require(file_sha256(run / 'execution_memory.json') == saved['execution_memory_sha256'], 'Memory record checksum mismatch')
            require(json.loads((run / 'execution_memory.json').read_text()) == saved['execution_memory'], 'Memory record mismatch')
            require(exposure_record(run, metadata) == saved['training_exposure'], 'Exposure counters mismatch')
            for kind, filename in [('model', 'model.safetensors'), ('adapter', 'adapter.safetensors')]:
                require(file_sha256(run / 'model' / filename) == saved[kind + '_sha256'], 'Artifact checksum mismatch')
            for key, meta in [('actual_steps', 'steps'), ('best_step', 'best_step'), ('training_seconds', 'training_seconds'),
                              ('best_validation_loss', 'best_validation_loss'), ('trainable_parameters', 'trainable_parameters')]:
                require(saved[key] == metadata[meta], f'Selected metadata mismatch: {key}')
            steps = saved['actual_steps']
            exposure = saved['training_exposure']
            labels_per_item = 416 if strategy.startswith('dhf') else TRAIN_HORIZON
            require(exposure['supervised_points'] == steps * 16 * 7 * labels_per_item, 'Observed-label count mismatch')
        for row_label in sorted(x for x in expected_rows(protocol) if x.startswith(dataset + '_')):
            row = json.loads((results / f'{row_label}.json').read_text())
            require(row['protocol_sha256'] == sha and row['dataset'] == dataset, 'Row provenance mismatch')
            canonical = f"{dataset}_{row['strategy']}" + ('' if row['seed'] is None else f"_s{row['seed']}") + f"_h{row['horizon']}"
            require(canonical == row_label, 'Row filename/case identity mismatch')
            model_label = row_label[:row_label.rfind('_h')]
            validate_row_identity(row, selected['models'].get(model_label), protocol)
            rows.append(row)
            for split in ['validation', 'test']:
                path = output / 'predictions' / f'{row_label}_{split}.npz'
                require(file_sha256(path) == row[split]['predictions_sha256'], 'Prediction checksum mismatch')
                horizon = row['horizon']
                with np.load(path) as arrays:
                    origins = sparse_origins(*manifest[split], horizon)
                    np.testing.assert_array_equal(arrays['origins'], origins)
                    actual = np.stack([values[t:t + horizon].T for t in origins])
                    np.testing.assert_array_equal(arrays['actual'], actual)
                    independent = independently_score(arrays['prediction'], actual, scales)
                    repeated = score_arrays(arrays['prediction'], actual, scales)
                for key, value in independent.items():
                    np.testing.assert_allclose(value, row[split][key], rtol=1e-12, atol=1e-12,
                                               err_msg=f'Independent {row_label}/{split}/{key}')
                for key, value in repeated.items():
                    np.testing.assert_allclose(value, row[split][key], rtol=1e-12, atol=1e-12,
                                               err_msg=f'Recomputed {row_label}/{split}/{key}')
                score_count += 1
    runs, summaries = validate_aggregate_tables(rows, results)
    record = {'passed': True, 'protocol_sha256': sha, 'auditor_sha256': file_sha256(__file__),
              'selected_model_count': len(expected_labels(protocol)), 'result_row_count': len(rows),
              'split_score_count': score_count, 'independent_metric_scalar_checks': score_count * 7,
              'runs_csv_rows': runs, 'summary_csv_rows': summaries,
              'checks': ['source SHA256 and execution snapshot', 'dataset checksum and train-only scaling',
                         'base, adapter and merged-weight identity', 'configuration and window fingerprints',
                         'training history and supervision counters', 'selected-checkpoint identity',
                         'forecast checksum, origins and labels', 'seven independently formulated metrics',
                         'aggregate means and sample standard deviations'],
              'limitations': ['audit does not establish causal attribution, significance, or device parity',
                              f'evaluation context{CONTEXT}; training horizon{TRAIN_HORIZON}']}
    immutable_json(results / 'integrity_audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/report_round2')
    parser.add_argument('--output', default='runs/report-round2')
    parser.add_argument('--data', default='data/validation')
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.output, args.data), indent=2))
