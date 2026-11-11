"""Fail-closed independent numerical and provenance audit of round-three results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_report_round1 import independently_score, require
from run_report_round3 import (
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
from timestral.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET

from timestral_ft import TrainConfig
from timestral_ft.artifacts import file_sha256
from timestral_ft.augmentation import prepare_augmented_batch
from timestral_ft.dhf import sample_block_layout


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
                                                'condition', 'training_seconds', 'best_step', 'actual_steps']}
            entry['split'] = split
            entry.update({key: value for key, value in row[split].items() if not isinstance(value, list)})
            entries.append(entry)
    expected = pd.DataFrame(entries)
    order = ['dataset', 'split', 'horizon', 'condition', 'strategy', 'seed']
    saved = pd.read_csv(Path(directory) / 'runs.csv')
    pd.testing.assert_frame_equal(saved.sort_values(order).reset_index(drop=True),
                                  expected.sort_values(order).reset_index(drop=True),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    expected_summary = expected.groupby(['dataset', 'split', 'horizon', 'condition', 'strategy'], sort=True).agg(
        {metric: ['mean', 'std', 'count'] for metric in METRICS})
    expected_summary.columns = ['_'.join(column) for column in expected_summary.columns]
    pd.testing.assert_frame_equal(pd.read_csv(Path(directory) / 'summary.csv'), expected_summary.reset_index(),
                                  check_dtype=False, rtol=1e-12, atol=1e-12)
    return len(saved), len(expected_summary)



def replay_training_counts(values, cfg, steps):
    """Replay independent RNG streams, derive eligibility without model/loss code.

    ETT's seven standardized numeric targets have no natural missing values.
    This also checks the claimed common window and layout draw schedule.
    """
    require(np.isfinite(values).all(), 'Counter replay expects complete ETT targets')
    sampler = torch.Generator().manual_seed(cfg.seed)
    layout_rng = torch.Generator().manual_seed(cfg.seed ^ 0x444846)
    missing_rng = torch.Generator().manual_seed(cfg.seed ^ 0x4D495353)
    role_rng = torch.Generator().manual_seed(cfg.seed ^ 0x524F4C45)
    result = {key: 0 for key in ['supervised_points', 'artificially_hidden_points', 'hidden_supervised_points',
              'original_observed_points', 'input_observed_points', 'sampled_target_rows', 'sampled_past_rows', 'sampled_known_rows']}
    for _ in range(steps * cfg.gradient_accumulation):
        starts = torch.randint(len(values) - 448 + 1, (cfg.batch_size,), generator=sampler).tolist()
        raw = torch.from_numpy(np.stack([values[start:start + 448].T for start in starts]))
        layout = sample_block_layout(cfg.batch_size, 14, 1, 6, layout_rng)
        aug = prepare_augmented_batch({'sequence': raw}, cfg, missing_rng, role_rng)
        observed = torch.isfinite(aug.inputs)
        result['original_observed_points'] += raw.numel()
        result['input_observed_points'] += int(observed.sum())
        result['artificially_hidden_points'] += int((~observed).sum())
        for key, role in [('sampled_target_rows', ROLE_TARGET), ('sampled_past_rows', ROLE_PAST_COV), ('sampled_known_rows', ROLE_KNOWN_COV)]:
            result[key] += int((aug.role_ids == role).sum())
        for i, anchors in enumerate(layout.tolist()):
            for origin in sorted(set(anchors) - {-1}):
                start, stop = origin * 32, (origin + anchors.count(origin)) * 32
                eligible = (aug.role_ids[i] == ROLE_TARGET) & observed[i, :, :start].any(-1)
                result['supervised_points'] += int(eligible.sum()) * (stop - start)
                result['hidden_supervised_points'] += int((~observed[i, :, start:stop] & eligible[:, None]).sum())
    states = {'sampler': sampler.get_state(), 'dhf_layout': layout_rng.get_state(),
              'missingness': missing_rng.get_state() if cfg.missing_probability else None,
              'roles': role_rng.get_state() if cfg.dynamic_role_sampling else None}
    return result, states

def audit(results='results/report_round3', output='runs/report-round3', data='data/validation'):
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
            require(exposure['sampled_windows'] == steps * 16, 'Microbatch/sample count mismatch')
            require(exposure['sampled_target_rows'] + exposure['sampled_past_rows'] + exposure['sampled_known_rows'] == steps * 16 * 7, 'Role row count mismatch')
            require(exposure['input_observed_points'] + exposure['artificially_hidden_points'] == exposure['original_observed_points'], 'Input/missing mask counts mismatch')
            require(0 <= exposure['hidden_supervised_points'] <= exposure['artificially_hidden_points'], 'Hidden-label count mismatch')
            require(exposure['supervised_points'] <= exposure['sampled_target_rows'] * 416, 'Supervision exceeds target capacity')
            if not expected_cfg['missing_probability']:
                require(exposure['artificially_hidden_points'] == 0, 'Disabled missingness corrupted inputs')
                require(exposure['supervised_points'] == exposure['sampled_target_rows'] * 416, 'Complete-history supervision count mismatch')
            if not expected_cfg['dynamic_role_sampling']:
                require(exposure['sampled_target_rows'] == steps * 16 * 7, 'Disabled roles reassigned targets')
            replay, replay_states = replay_training_counts(values[:manifest['train'][1]], TrainConfig(**expected_cfg), steps)
            require(all(exposure[key] == value for key, value in replay.items()), 'Replayed missing/role/supervision counters mismatch')
            checkpoint_rng = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)['rng']
            for key, state in replay_states.items():
                require(checkpoint_rng[key] is None if state is None else torch.equal(checkpoint_rng[key], state),
                        f'Replayed final checkpoint RNG differs: {key}')
            history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
            evaluations = [(record['validation_loss'], record['step']) for record in history if 'validation_loss' in record]
            require(min(evaluations) == (saved['best_validation_loss'], saved['best_step']), 'Checkpoint is not the minimum common validation loss')
        for row_label in sorted(x for x in expected_rows(protocol) if x.startswith(dataset + '_')):
            row = json.loads((results / f'{row_label}.json').read_text())
            require(row['protocol_sha256'] == sha and row['dataset'] == dataset, 'Row provenance mismatch')
            canonical = f"{dataset}_{row['strategy']}" + ('' if row['seed'] is None else f"_s{row['seed']}") + f"_h{row['horizon']}_{row['condition']}"
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
                    contexts = np.stack([values[t - CONTEXT:t].T for t in origins])
                    if row['condition'] == 'gap64':
                        contexts[..., 128:192] = np.nan
                    np.testing.assert_array_equal(arrays['context'], contexts)
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
                         'training history and independent eligibility replay from checkpointed RNG streams', 'selected-checkpoint identity',
                         'forecast checksum, origins and labels', 'seven independently formulated metrics',
                         'aggregate means and sample standard deviations'],
              'limitations': ['audit does not establish causal attribution, significance, or device parity',
                              f'evaluation context{CONTEXT}; training horizon{TRAIN_HORIZON}']}
    immutable_json(results / 'integrity_audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/report_round3')
    parser.add_argument('--output', default='runs/report-round3')
    parser.add_argument('--data', default='data/validation')
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.output, args.data), indent=2))
