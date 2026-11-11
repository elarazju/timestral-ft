"""Protocol invariants for paired stage schedules and frozen transfer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import run_report_round4 as report  # noqa: E402
from audit_report_round4 import replay_training_counts  # noqa: E402


def test_paired_recipe_and_transfer_matrix():
    configs = report.training_configs(SimpleNamespace(device='cpu', threads=2), {'selected_steps': 200})
    shared = []
    for strategy, cfg in configs.items():
        assert cfg['stages'][0]['synthetic_probability'] == {'real': 0, 'mixed': .5, 'staged': 1}[strategy]
        assert cfg['stages'][1]['synthetic_probability'] == (0 if strategy == 'real' else .5)
        assert [(s['steps'], s['context_length']) for s in cfg['stages']] == [(100, 128), (100, 256)]
        assert not cfg['missing_probability'] and not cfg['dynamic_role_sampling']
        shared.append({k: v for k, v in cfg.items() if k != 'stages'})
    assert shared[0] == shared[1] == shared[2]
    assert len(report.expected_labels({'training': configs})) == 9
    assert all(x.startswith('ETTh1_') for x in report.expected_labels({'training': configs}))
    assert len(report.expected_rows({'training': configs})) == 40


def test_independent_exposure_replay_has_fixed_total_but_different_sources():
    totals = []
    for strategy in report.STRATEGIES:
        cfg = report.config(strategy, 42, 4, SimpleNamespace(device='cpu', threads=2))
        rows, state = replay_training_counts(np.ones((500, 7), dtype=np.float32), cfg, 4)
        assert [r['training_context_length'] for r in rows] == [128, 128, 256, 256]
        assert sum(r['real_samples'] + r['synthetic_samples'] for r in rows) == 64
        totals.append(sum(r['supervised_points'] for r in rows))
        assert (state['synthetic'] is None) == (strategy == 'real')
        if strategy == 'staged':
            assert rows[0]['real_samples'] == rows[1]['real_samples'] == 0
    assert totals == [157696] * 3
