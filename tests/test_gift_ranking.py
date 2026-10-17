import importlib.util
from pathlib import Path

import pandas as pd


def test_official_ranking_uses_mean_task_positions_and_sorted_ties():
    script = Path(__file__).resolve().parents[1]/'scripts/rank_gift209.py'
    spec = importlib.util.spec_from_file_location('rank_gift209_test', script)
    ranking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranking)
    rows = []
    for model, values in [('A', [.1, 100.]), ('B', [.2, .3]), ('C', [.3, .4])]:
        for task, value in enumerate(values):
            rows.append(dict(model=model, dataset=str(task),
                             **{column: value for column in ranking.METRICS.values()}))
    result = ranking.aggregate_ranks(pd.DataFrame(rows).iloc[::-1])
    assert result.loc['A', 'crps_mean_task_rank'] == 2
    assert result.loc['B', 'crps_position'] == 1
    assert result.loc['A', 'crps_position'] == 2
    # Exact metric ties follow alphabetical model order, independently of input
    # filesystem ordering, as in the official leaderboard.
    tied = pd.DataFrame([dict(model=name, dataset='one',
                             **{column: 1. for column in ranking.METRICS.values()})
                         for name in ['Z', 'A']])
    result = ranking.aggregate_ranks(tied)
    assert result.loc['A', 'crps_position'] == 1
    assert result.loc['Z', 'mase_position'] == 2


def test_training_diagnostics_count_continued_prefix_once_and_fix_context(tmp_path):
    import json

    script = Path(__file__).resolve().parents[1]/'scripts/rank_gift209.py'
    spec = importlib.util.spec_from_file_location('gift_diagnostics_test', script)
    ranking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranking)
    rows = []
    for task in range(97):
        candidates = []
        for name, steps, context in [('short', 200, 256), ('continued', 800, 2048)]:
            candidates.append(dict(path=name, summary=dict(steps=steps, training_seconds=steps/10,
                              trainable_parameters=10, train_config=dict(context_length=context, learning_rate=1e-4))))
            for budget in [200, 800]:
                observed = min(budget, steps)
                ratio = (1.2 if name == 'short' else .9 if budget == 200 else .8)
                rows.append(dict(task=str(task), candidate=name, requested_budget=budget,
                                 observed_steps=observed, objective='pareto', chosen_step=observed,
                                 validation_crps_ratio=ratio, validation_mase_ratio=ratio))
        receipt = dict(candidates=candidates, selected_candidate='continued', baseline_fallback=False, best_step=400)
        (tmp_path/f'{task}.selection.json').write_text(json.dumps(receipt))
    (tmp_path/'progress.json').write_text(json.dumps(dict(started_at_utc='2026-10-17T00:00:00+00:00',
                                                         updated_at_utc='2026-10-17T00:02:00+00:00')))
    pd.DataFrame(rows).to_csv(tmp_path/'validation_ablations.csv', index=False)
    result = ranking.training_diagnostics(tmp_path)
    assert result['optimizer_steps'] == 97000  # 200 + 800, not 200 + 200 + 800
    assert result['supervised_wall_seconds_including_tests_and_waits'] == 120
    assert result['nonfallback_context_counts'] == {'2048': 97}
    diagnostics = result['validation_checkpoint_diagnostics']
    assert [r['tasks'] for r in diagnostics] == [97, 97]
    assert abs(diagnostics[0]['validation_crps_ratio_gmean']-.9) < 1e-12
    assert abs(diagnostics[1]['validation_crps_ratio_gmean']-.8) < 1e-12
