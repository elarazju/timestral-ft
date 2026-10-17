"""Simulate a completed 209M entry against a pinned public leaderboard snapshot.

This does not submit results or modify selected adapters. Mean per-task ranks
are distinct from normalized geometric-mean metric scores.
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = {'crps': 'eval_metrics/mean_weighted_sum_quantile_loss', 'mase': 'eval_metrics/MASE[0.5]'}
MODEL = 'Timestral-FT-209M-LoRA'
BASE = 'Timestral-209M'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def aggregate_ranks(raw):
    raw = raw.sort_values(['model', 'dataset']).reset_index(drop=True)
    result = pd.DataFrame(index=sorted(raw['model'].unique()))
    for name, column in METRICS.items():
        if not np.isfinite(raw[column]).all() or (raw[column] < 0).any():
            raise ValueError('Missing or negative rank metric')
        ranked = raw.assign(task_rank=raw.groupby('dataset')[column].rank(method='first'))
        means = ranked.groupby('model')['task_rank'].mean()
        result[name+'_mean_task_rank'] = means
        result[name+'_position'] = means.rank(method='min').astype(int)
    return result


def training_diagnostics(directory):
    """Costs and conditional validation diagnostics, after the test is fixed."""
    receipts = [json.loads(p.read_text()) for p in directory.glob('*.selection.json')]
    if len(receipts) != 97:
        raise ValueError('All task search receipts are required for training diagnostics')
    candidates = [c['summary'] for r in receipts for c in r['candidates']]
    chosen = [next(c['summary'] for c in r['candidates'] if c['path'] == r['selected_candidate'])
              for r in receipts if not r['baseline_fallback']]
    result = dict(tasks=97, optimizer_steps=sum(c['steps'] for c in candidates),
                  cumulative_training_loop_seconds=sum(c['training_seconds'] for c in candidates),
                  training_time_scope='Includes validation/checkpoint loops; excludes model/data startup and preflight; parallel sum is not wall time',
                  identity_fallback_tasks=sum(r['baseline_fallback'] for r in receipts),
                  selected_step_counts=dict(Counter(str(r['best_step']) for r in receipts)),
                  nonfallback_context_counts=dict(Counter(str(c['train_config']['context_length']) for c in chosen)),
                  nonfallback_learning_rate_counts=dict(Counter(str(c['train_config']['learning_rate']) for c in chosen)),
                  trainable_parameters=sorted({c['trainable_parameters'] for c in candidates}))
    progress = json.loads((directory/'progress.json').read_text())
    start = datetime.fromisoformat(progress['started_at_utc'])
    end = datetime.fromisoformat(progress['updated_at_utc'])
    result['supervised_wall_seconds_including_tests_and_waits'] = (end-start).total_seconds()
    ablations = pd.read_csv(directory/'validation_ablations.csv')
    lengths = ablations.groupby(['task', 'candidate'], as_index=False)['observed_steps'].max()
    winners = lengths[lengths['observed_steps'] == lengths.groupby('task')['observed_steps'].transform('max')]
    # In the default nested-budget search exactly one candidate is extended.
    # Custom searches with ambiguous equal-length candidates are excluded.
    winners = winners[~winners['task'].duplicated(keep=False)]
    fixed = ablations.merge(winners[['task', 'candidate']], on=['task', 'candidate'], validate='many_to_one')
    rows = []
    for (budget, objective), group in fixed.groupby(['requested_budget', 'objective']):
        rows.append(dict(budget_cap=int(budget), objective=objective, tasks=int(group['task'].nunique()),
                         validation_crps_ratio_gmean=float(np.exp(np.log(group['validation_crps_ratio']).mean())),
                         validation_mase_ratio_gmean=float(np.exp(np.log(group['validation_mase_ratio']).mean())),
                         median_selected_step=float(group['chosen_step'].median())))
    result['validation_checkpoint_diagnostics'] = rows
    result['diagnostic_scope'] = ('Only observed checkpoints in validation-selected continuation contexts. '
                                  'Not independently optimized objectives, a complete factorial experiment, or test gains.')
    (directory/'training_diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Conditional validation checkpoint diagnostics', '', result['diagnostic_scope'], '',
             '| Budget cap | Objective | Tasks | Validation CRPS / base | Validation MASE / base | Median step |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["budget_cap"]} | {r["objective"]} | {r["tasks"]} | '
                     f'{r["validation_crps_ratio_gmean"]:.6f} | {r["validation_mase_ratio_gmean"]:.6f} | {r["median_selected_step"]:g} |')
    (directory/'validation_diagnostics.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leaderboard-dir', type=Path, default=ROOT/'.cache/gift-leaderboard-current')
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results/gift_eval_209m_lora_v2')
    parser.add_argument('--wait', action='store_true', help='Wait for this run supervisor to finish its audit')
    args = parser.parse_args()
    if args.wait:
        while True:
            status_path = args.results_dir/'progress.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('phase') == 'failed':
                raise RuntimeError('Experiment failed before ranking: '+status.get('error', 'unknown error'))
            if status.get('phase') == 'complete':
                break
            time.sleep(30)
    audit = json.loads((args.results_dir/'audit.json').read_text())
    if not audit['all_checks_pass'] or len(audit['tasks']) != 97:
        raise ValueError('A complete audited 97-task result is required')
    ours_path = args.results_dir/'lora_all_results.csv'
    ours = pd.read_csv(ours_path)
    if set(ours['model']) != {MODEL} or ours['dataset'].nunique() != 97 or len(ours) != 97:
        raise ValueError('Incomplete or unexpected adapted entry')
    frames, manifest = [], {}
    for path in sorted((args.leaderboard_dir/'results').glob('*/all_results.csv')):
        df = pd.read_csv(path)
        if len(df) != 97 or set(df['dataset']) != set(ours['dataset']) or df['model'].nunique() != 1:
            raise ValueError(f'Incomplete public entry: {path}')
        if MODEL in set(df['model']):
            raise ValueError('Snapshot already contains this model; do not insert a duplicate')
        frames.append(df)
        manifest[path.parent.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    public = pd.concat(frames, ignore_index=True)
    original = aggregate_ranks(public)
    combined = aggregate_ranks(pd.concat([public, ours], ignore_index=True))
    metadata_path = args.leaderboard_dir/'space_metadata.json'
    metadata = (json.loads(metadata_path.read_text()) if metadata_path.exists() else
                {'sha': subprocess.check_output(['git', '-C', str(args.leaderboard_dir), 'rev-parse', 'HEAD'], text=True).strip()})

    def entries(frame, names):
        return {name: {key: int(value) if key.endswith('_position') else float(value)
                       for key, value in frame.loc[name].items()} for name in names}

    record = dict(computed_at_utc=datetime.now(timezone.utc).isoformat(),
                  space_revision=metadata['sha'], comparison='pinned public snapshot, not a live official placement',
                  publicly_submitted=False, public_models=len(original), with_local_models=len(combined),
                  method='Official sort(model,dataset), per-task rank(method=first), arithmetic mean over 97 tasks',
                  original_base=entries(original, [BASE]), with_local=entries(combined, [BASE, MODEL]),
                  adapted_csv_sha256=hashlib.sha256(ours_path.read_bytes()).hexdigest(),
                  public_result_sha256=manifest, ranking_source_sha256=SOURCE_SHA256)
    (args.results_dir/'ranking_simulation.json').write_text(json.dumps(record, indent=2)+'\n')
    combined.to_csv(args.results_dir/'ranking_simulation_all_models.csv')
    r = record['with_local'][MODEL]
    english = (f'\n\nLocal rank simulation against public snapshot `{metadata["sha"][:12]}`: '
               f'CRPS **{r["crps_position"]}/{len(combined)}**, MASE **{r["mase_position"]}/{len(combined)}**. '
               'This is a local insertion/recalculation, not an official leaderboard placement.\n')
    chinese = (f'\n\n相对公开快照 `{metadata["sha"][:12]}` 的本地名次模拟：'
               f'CRPS **{r["crps_position"]}/{len(combined)}**，MASE **{r["mase_position"]}/{len(combined)}**。'
               '这是插入本地结果后的重算，不是正式榜单名次。\n')
    diagnostics = training_diagnostics(args.results_dir)
    english += (f'\nFormal search used {diagnostics["optimizer_steps"]:,} optimizer steps. '
                '[Training costs and choices](training_diagnostics.json) · '
                '[Conditional validation checkpoint diagnostics](validation_diagnostics.md).\n')
    chinese += (f'\n正式搜索累计 {diagnostics["optimizer_steps"]:,} 个优化步骤。'
                '[训练耗时与选优分布](training_diagnostics.json) · '
                '[验证检查点对照](validation_diagnostics.md)。对照仅限验证集，不代表独立消融或测试收益。\n')
    for name, text in [('README.md', english), ('README.zh-CN.md', chinese)]:
        path = args.results_dir/name
        content = path.read_text()
        marker = '\n\nLocal rank simulation' if name == 'README.md' else '\n\n相对公开快照'
        path.write_text(content.split(marker)[0]+text)
    print(json.dumps({k: v for k, v in record.items() if k != 'public_result_sha256'}, indent=2))


if __name__ == '__main__':
    main()
