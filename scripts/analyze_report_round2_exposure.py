"""Post-run reconstruction of realized histories; never trains or selects models."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from benchmark_suite import prepare_dataset
from run_report_round1 import digest_json, immutable_json

from timestral_ft import DataConfig
from timestral_ft.artifacts import file_sha256
from timestral_ft.data import DataProcessor, DHFSequenceDataset, WindowDataset
from timestral_ft.dhf import sample_block_layout


def analyze(results='results/report_round2', output='runs/report-round2', data='data/validation'):
    results, output = Path(results), Path(output)
    protocol = json.loads((results / 'protocol.json').read_text())
    selected = json.loads((results / 'selected_models.json').read_text())
    assert digest_json(protocol) == selected['protocol_sha256']
    root = Path(__file__).resolve().parents[1]
    for name, checksum in protocol['sources'].items():
        assert file_sha256(root / name) == checksum
    records, candidate_checks, sampler_states = {}, {}, {}
    for name, manifest in protocol['datasets'].items():
        frame, _, actual_manifest = prepare_dataset(name, data)
        assert actual_manifest == manifest
        train = frame.iloc[:manifest['train'][1]]
        assert np.isfinite(train[manifest['columns']].to_numpy()).all()
        processor = DataProcessor(DataConfig(time_column='date', target_columns=manifest['columns'],
                                            frequency='h')).fit(train)
        window = WindowDataset(processor, train, 256, 192)
        dhf = DHFSequenceDataset(processor, train, 256, 192, min_context_steps=32)
        assert window.index == dhf.index
        candidate_checks[name] = {'identical_index_lists': True, 'candidate_count': len(window)}
    for label, saved in selected['models'].items():
        run = output / label
        metadata = json.loads((run / 'model/finetune.json').read_text())
        cfg = metadata['train_config']
        state = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)
        assert state['step'] == saved['actual_steps']
        sampler_states[label] = state['rng']['sampler'].clone()
        per_step = cfg['batch_size'] * cfg['gradient_accumulation']
        histories, lengths = Counter(), Counter()
        if cfg['training_strategy'] == 'dhf':
            generator = torch.Generator().manual_seed(metadata['dhf']['layout_seed'])
            for _ in range(saved['actual_steps'] * cfg['gradient_accumulation']):
                layout = sample_block_layout(cfg['batch_size'], metadata['dhf']['padded_sequence_length'] // 32,
                                             cfg['dhf_min_context_patches'], metadata['dhf']['max_block_patches'],
                                             generator)
                for anchors in layout.tolist():
                    for start in set(anchors) - {-1}:
                        histories[start * 32] += 1
                        lengths[str(anchors.count(start))] += 1
            assert torch.equal(generator.get_state(), state['rng']['dhf_layout'])
            assert dict(lengths) == saved['training_exposure']['block_length_histogram']
            labels_per_item = 416
        else:
            assert state['rng'].get('dhf_layout') is None
            histories[256] = saved['actual_steps'] * per_step
            labels_per_item = 192
        count = sum(histories.values())
        assert count == saved['training_exposure']['forecast_blocks']
        targets = len(protocol['datasets'][label.split('_')[0]]['columns'])
        assert saved['training_exposure']['supervised_points'] == saved['actual_steps'] * per_step * targets * labels_per_item
        records[label] = {
            'completed_steps': saved['actual_steps'], 'selected_step': saved['best_step'],
            'completed_supervised_points': saved['training_exposure']['supervised_points'],
            'selected_checkpoint_supervised_points': saved['best_step'] * per_step * targets * labels_per_item,
            'history_steps_histogram_per_example_block': {str(k): v for k, v in sorted(histories.items())},
            'forecast_blocks': count, 'minimum_history': min(histories), 'maximum_history': max(histories),
            'block_weighted_mean_history': sum(k * v for k, v in histories.items()) / count,
            'checkpoint_sha256': file_sha256(run / 'checkpoint.pt'),
            'replayed_layout_final_rng_matches': True if cfg['training_strategy'] == 'dhf' else None,
        }
        del state
    pairs = {}
    for dataset in protocol['datasets']:
        for seed in protocol['seeds']:
            a, b = f'{dataset}_single_window_s{seed}', f'{dataset}_dhf_s{seed}'
            assert torch.equal(sampler_states[a], sampler_states[b])
            pairs[f'{dataset}_s{seed}'] = True
    record = {
        'passed': True, 'protocol_sha256': selected['protocol_sha256'],
        'analysis_script_sha256': file_sha256(__file__), 'models': records,
        'dataset_candidates': candidate_checks, 'paired_main_window_sampler_states_equal': pairs,
        'scope': 'post-run deterministic replay for fully observed ETT; no training, selection, or protocol changes',
        'counts': 'histograms count forecast blocks per example, not per target; labels include all seven targets',
        'limitation': 'near-equal supervision control matches completed run budget, not necessarily selected-checkpoint exposure',
    }
    immutable_json(results / 'exposure_audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/report_round2')
    parser.add_argument('--output', default='runs/report-round2')
    parser.add_argument('--data', default='data/validation')
    args = parser.parse_args()
    record = analyze(args.results, args.output, args.data)
    print(json.dumps({'passed': record['passed'], 'models': len(record['models']),
                      'paired_window_checks': len(record['paired_main_window_sampler_states_equal'])}, indent=2))
