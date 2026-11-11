"""Fail-closed independent numerical and provenance audit of round-four results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from audit_report_round1 import independently_score, require
from run_report_round4 import (
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
    source_identity,
    sparse_origins,
)
from timestral.loading import resolve_checkpoint

from timestral_ft import TrainConfig
from timestral_ft.artifacts import file_sha256
from timestral_ft.synthetic import generate_sequence


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
    """Independently count draws, source exposures and layouts; no model/loss code.

    Generator execution is reused only to advance its dedicated stream; source
    Bernoulli decisions, window draws, phase boundaries, supervision and layout
    tiling are reconstructed here rather than calling trainer sampling helpers.
    """
    require(np.isfinite(values).all(), 'Replay expects complete numeric ETT targets')
    sampler = torch.Generator().manual_seed(cfg.seed)
    layout = torch.Generator().manual_seed(cfg.seed ^ 0x444846)
    mixed = any(s.synthetic_probability for s in cfg.stages)
    mixture = torch.Generator().manual_seed(cfg.seed ^ 0x4D495854) if mixed else None
    synthetic = torch.Generator().manual_seed(cfg.seed ^ 0x53594E54) if mixed else None
    records, completed = [], 0
    for stage_index, stage in enumerate(cfg.stages):
        for stage_step in range(stage.steps):
            if completed >= steps:
                break
            row = dict(real_samples=0, synthetic_samples=0, real_supervised_points=0,
                       synthetic_supervised_points=0, supervised_points=0, forecast_blocks=0,
                       block_length_histogram={}, stage_index=stage_index, stage_step=stage_step + 1,
                       training_context_length=stage.context_length, synthetic_probability=stage.synthetic_probability)
            total = stage.context_length + cfg.prediction_length
            for _ in range(cfg.gradient_accumulation):
                if mixed:
                    flags = (torch.rand(cfg.batch_size, generator=mixture) < stage.synthetic_probability).tolist()
                    for flag in flags:
                        if flag:
                            x = generate_sequence(total, values.shape[1], cfg.synthetic, synthetic)
                            require(torch.isfinite(x).all(), 'Synthetic nonfinite observations')
                        else:
                            torch.randint(len(values) - total + 1, (), generator=sampler)
                else:
                    flags = [False] * cfg.batch_size
                    torch.randint(len(values) - total + 1, (cfg.batch_size,), generator=sampler)
                for flag in flags:
                    source = 'synthetic' if flag else 'real'
                    row[source + '_samples'] += 1
                    row[source + '_supervised_points'] += (total - 32) * values.shape[1]
                    position = 1
                    while position < total // 32:
                        length = min(int(torch.randint(1, 7, (), generator=layout)), total // 32 - position)
                        row['forecast_blocks'] += 1
                        key = str(length)
                        row['block_length_histogram'][key] = row['block_length_histogram'].get(key, 0) + 1
                        position += length
            row['supervised_points'] = row['real_supervised_points'] + row['synthetic_supervised_points']
            records.append(row)
            completed += 1
    states = {'sampler': sampler.get_state(), 'dhf_layout': layout.get_state(),
        'mixture': mixture.get_state() if mixed else None, 'synthetic': synthetic.get_state() if mixed else None,
        'missingness': None, 'roles': None}
    return records, states


def audit(results='results/report_round4', output='runs/report-round4', data='data/validation'):
    results, output = Path(results), Path(output)
    protocol = json.loads((results / 'protocol.json').read_text())
    selected = json.loads((results / 'selected_models.json').read_text())
    require(protocol['sources'] == source_identity(), 'Runtime sources differ from frozen sources')
    for name, checksum in protocol['sources'].items():
        require(file_sha256(output / 'source_snapshot' / name) == checksum, 'Execution source snapshot differs')
    base = resolve_checkpoint(protocol['model_id'], revision=protocol['model_revision'])
    require(file_sha256(base / 'model.safetensors') == protocol['model_sha256'], 'Base weights changed')
    require(file_sha256(base / 'config.json') == protocol['model_config_sha256'], 'Base config changed')
    sha = digest_json(protocol)
    require(json.loads((output / 'protocol.json').read_text()) == protocol, 'Execution protocol differs')
    require(selected['protocol_sha256'] == sha, 'Selected model protocol mismatch')
    require(set(selected['models']) == expected_labels(protocol), 'Incomplete/extra selected models')
    require({p.stem for p in results.glob('ETTh*.json')} == expected_rows(protocol), 'Incomplete/extra result rows')
    rows, score_count = [], 0
    for dataset in DATASETS:
        frame, values, manifest = prepare_dataset(dataset, data)
        require(manifest == protocol['datasets'][dataset], 'Data/scaler provenance mismatch')
        raw = pd.read_csv(Path(data) / f'{dataset}.csv').iloc[:14400][manifest['columns']].to_numpy(dtype=np.float64)
        train_end = manifest['train'][1]
        independent_mean, independent_std = raw[:train_end].mean(0), raw[:train_end].std(0, ddof=0)
        np.testing.assert_array_equal(independent_mean, manifest['mean'])
        np.testing.assert_array_equal(independent_std, manifest['std_ddof0'])
        np.testing.assert_array_equal(((raw - independent_mean) / independent_std).astype(np.float32), values)
        scales = np.abs(values[24:train_end].astype(np.float64) - values[:train_end - 24].astype(np.float64)).mean(0)
        require((scales > 0).all(), 'Nonpositive seasonal scale')
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
            require(exposure['real_supervised_points'] + exposure['synthetic_supervised_points'] == exposure['supervised_points'], 'Source supervision mismatch')
            replay, replay_states = replay_training_counts(values[:manifest['train'][1]], TrainConfig(**expected_cfg), steps)
            history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
            for row, independent in zip(history, replay, strict=True):
                require(all(row[key] == value for key, value in independent.items()), 'Replayed phase/source/exposure/layout differs')
            state = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)
            require(state['stage_progress'] == [len(expected_cfg['stages']), 0], 'Final stage progress mismatch')
            require(state['exposure'] == metadata['exposure'], 'Checkpoint exposure differs')
            require(all(state['exposure'][key] == exposure[key] for key in state['exposure']), 'Cumulative exposure differs')
            checkpoint_rng = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)['rng']
            for key, state in replay_states.items():
                require(checkpoint_rng[key] is None if state is None else torch.equal(checkpoint_rng[key], state),
                        f'Replayed final checkpoint RNG differs: {key}')
            history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
            until_best = [row for row in history if row['step'] <= saved['best_step']]
            require(saved['best_stage_index'] == until_best[-1]['stage_index'], 'Selected stage differs')
            independent_exposure = {key: sum(row[key] for row in until_best) for key in
                ('real_samples', 'synthetic_samples', 'real_supervised_points', 'synthetic_supervised_points', 'supervised_points')}
            require(saved['selected_exposure'] == independent_exposure, 'Selected-prefix exposure differs')
            evaluations = [(record['validation_loss'], record['step']) for record in history if 'validation_loss' in record]
            require(min(evaluations) == (saved['best_validation_loss'], saved['best_step']), 'Checkpoint is not the minimum common validation loss')
        for row_label in sorted(x for x in expected_rows(protocol) if x.startswith(dataset + '_')):
            row = json.loads((results / f'{row_label}.json').read_text())
            require(row['protocol_sha256'] == sha and row['dataset'] == dataset, 'Row provenance mismatch')
            canonical = f"{dataset}_{row['strategy']}" + ('' if row['seed'] is None else f"_s{row['seed']}") + f"_h{row['horizon']}_{row['condition']}"
            require(canonical == row_label, 'Row filename/case identity mismatch')
            model_label = row_label[:row_label.rfind('_h')].replace(dataset + '_', 'ETTh1_', 1)
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
                         'training history and independent phase/source/exposure/layout replay; generator stream advanced with versioned generator', 'selected-checkpoint identity',
                         'forecast checksum, origins and labels', 'seven independently formulated metrics',
                         'aggregate means and sample standard deviations'],
              'limitations': ['audit does not establish causal attribution, significance, or device parity',
                              f'evaluation context{CONTEXT}; training horizon{TRAIN_HORIZON}']}
    immutable_json(results / 'integrity_audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/report_round4')
    parser.add_argument('--output', default='runs/report-round4')
    parser.add_argument('--data', default='data/validation')
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.output, args.data), indent=2))
