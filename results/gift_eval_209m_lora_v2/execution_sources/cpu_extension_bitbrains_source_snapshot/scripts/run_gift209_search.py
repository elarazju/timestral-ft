"""Validation-only 209M recipe search, then freeze all 97 adapters before testing.

The same predeclared search is applied to every task. Runs are resumable and
append receipts; tests are unavailable to the search stage. No leaderboard PR
or remote model upload is performed by this program.
"""

import argparse
import csv
import fcntl
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.cache/upstream/src'), str(ROOT / '.cache/gift-eval/src')]
BASE = 'mistralai/Timestral-209M'
REVISION = '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'
MODEL = 'Timestral-FT-209M-LoRA'
SOURCES = ['scripts/run_gift209_search.py', 'src/timestral_ft/gift.py',
           'src/timestral_ft/gift_validation.py', 'src/timestral_ft/compat.py',
           'src/timestral_ft/lora.py', 'src/timestral_ft/trainer.py', 'scripts/launch_gift209.py']


def read(path):
    return json.loads(Path(path).read_text())


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', 'search', 'test', 'report'], default='all')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/gift-eval-209m-search-v2')
    p.add_argument('--results-dir', type=Path, default=ROOT / 'results/gift_eval_209m_lora_v2')
    p.add_argument('--task', action='append')
    p.add_argument('--worker-device', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--device', default='mps')
    p.add_argument('--contexts', type=int, nargs='+', default=[256, 2048, 8192])
    p.add_argument('--screen-steps', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=800)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--validation-limit', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--row-context-budget', type=int, default=65536)
    return p.parse_args()


def sources():
    from timestral_ft.artifacts import file_sha256
    return {p: file_sha256(ROOT / p) for p in SOURCES}


def open_protocol(args):
    import numpy as np
    import pandas as pd
    import torch
    from timestral.loading import resolve_checkpoint

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import DATA_REVISION, GIFT_REVISION, atomic_json
    from timestral_ft.gift_validation import GiftValidation, GiftValidationConfig
    gift = ROOT / '.cache/gift-eval'
    head = subprocess.check_output(['git', '-C', str(gift), 'rev-parse', 'HEAD'], text=True).strip()
    if head != GIFT_REVISION:
        raise ValueError('Official evaluation source revision changed')
    if file_sha256(gift / 'notebooks/timestral.ipynb') != 'a04b708eaf44537d5dd10d78bb509caee38f3b27b7fc1245b86d7e066d6d731d':
        raise ValueError('Official notebook changed')
    inventory = read(ROOT / 'results/gift_eval_feasibility/task_inventory.json')
    assert len(inventory) == 97 and len({t['key'] for t in inventory}) == 97
    base = resolve_checkpoint(BASE, revision=REVISION)
    identity = dict(model=BASE, revision=REVISION, weights_sha256=file_sha256(base / 'model.safetensors'))
    # Device assignment uses training-prefix shapes only. No prior test score is
    # read: short univariate prefixes avoid occupying the accelerator.
    devices = {}
    for task in inventory:
        prior = read(ROOT / 'results/gift_eval_5m_lora' / (task['key'].replace('/', '__') + '.training.json'))
        devices[task['key']] = ('cpu' if prior['data']['train_length_max'] <= 256
                                and prior['data']['series'] > 1 else args.device)
    protocol = dict(
        version=2, model=MODEL, base_model=BASE, base_revision=REVISION, base_identity=identity,
        gift_revision=GIFT_REVISION, data_revision=DATA_REVISION, seed=42,
        train=dict(contexts=args.contexts, screen_steps=args.screen_steps, max_steps=args.max_steps,
                   eval_every=args.eval_every, batch_size=2, gradient_accumulation=2,
                   row_context_budget=args.row_context_budget,
                   microbatch_rule='if context * (targets + past covariates) > row_context_budget/2, use batch=1 accumulation=4; otherwise batch=2 accumulation=2',
                   learning_rates=[1e-4, 3e-4], alternate_lr_rule='screen 3e-4 at best context iff no >=1% Pareto validation gain',
                   warmup_steps=40, weight_decay=.01, gradient_clip=1., rank=8, alpha=16., dropout=0.,
                   patience=6, validation_limit=args.validation_limit),
        selection=GiftValidation('unused', GiftValidationConfig(), identity).metadata,
        seasonality_rule='GluonTS get_seasonality(task frequency), complete training history denominator',
        validation_context=32768, test_context=32768,
        selected_method='best Pareto validation checkpoint across screened contexts and continued winner; step zero eligible',
        budget_note='200-step screens are prefixes of the 800-step cosine schedule; no independent 200-step schedule claim',
        screen_tie_break='best accepted score, then minimum observed unconstrained geometric metric ratio, then longer context',
        context_cap='minimum of rounded maximum training-prefix length and floor(row_context_budget/input_rows/32)*32; deduplicate contexts',
        test_policy='freeze adapter hashes for all 97 tasks before accessing test iterators; one final paired test',
        train_devices=devices, source_sha256=sources(), python=platform.python_version(),
        torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / '.protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing_path = args.run_dir / 'protocol.json'
        if existing_path.exists():
            existing = read(existing_path)
            if existing != protocol:
                previous_sources, current_sources = existing['source_sha256'], protocol['source_sha256']
                comparable = dict(protocol, source_sha256=previous_sources)
                changed = [p for p, sha in current_sources.items() if sha != previous_sources.get(p)]
                # A reporting repair must not invalidate days of valid training.
                # Training/search sources stay immutable; test/report stages may
                # repair their runner while preserving every frozen weight hash.
                if (args.stage not in ['test', 'report'] or comparable != existing
                        or any(p.startswith('src/') for p in changed)):
                    raise ValueError('Immutable training protocol/source changed; use a new run directory')
                atomic_json(args.results_dir / f'{args.stage}_source_manifest.json',
                            dict(original_sources=previous_sources, executed_sources=current_sources,
                                 changed_runner_files=changed, frozen_weights_unchanged=True))
                protocol = existing
        for path in [args.run_dir / 'protocol.json', args.results_dir / 'protocol.json']:
            if path.exists() and read(path) != protocol:
                raise ValueError(f'Immutable protocol changed: {path}; use a new run directory')
            if not path.exists():
                atomic_json(path, protocol)
        snapshot = args.run_dir / 'source_snapshot'
        for name in SOURCES:
            dest = snapshot / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(ROOT / name, dest)
        original = ROOT / 'results/gift_eval_5m_lora/data_manifest.json'
        if not (args.results_dir / 'data_manifest.json').exists():
            for relative, info in read(original)['files'].items():
                data = ROOT / '.cache/gift-eval-data' / relative
                if data.stat().st_size != info['bytes'] or file_sha256(data) != info['sha256']:
                    raise ValueError(f'Pinned dataset checksum mismatch: {relative}')
            shutil.copyfile(original, args.results_dir / 'data_manifest.json')
    return protocol, inventory, base


def clear(device):
    import torch
    gc.collect()
    if device.startswith('mps'):
        torch.mps.empty_cache()
    elif device.startswith('cuda'):
        torch.cuda.empty_cache()


def task_directory(args, key):
    return args.run_dir / 'tasks' / key.replace('/', '__')


def history(directory):
    return [json.loads(line) for line in (directory / 'history.jsonl').read_text().splitlines()
            if line and 'validation_metrics' in line]


def candidate_order(directory):
    summary = read(directory / 'training_summary.json')
    rows = history(directory)
    unconstrained = min(math.sqrt(r['validation_metrics']['crps_ratio'] * r['validation_metrics']['mase_ratio'])
                        for r in rows)
    return (summary['validation_loss'], unconstrained, -summary['train_config']['context_length'])


def search_task(args, protocol, task, base):
    from gift_eval.data import Dataset
    from gluonts.time_feature import get_seasonality

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import GiftTrainConfig, GiftWindows, atomic_json, bounded_contexts, train_gift_task
    from timestral_ft.gift_validation import GiftValidation, GiftValidationConfig
    key = task['key']
    output = task_directory(args, key)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / 'selection.json').exists():
            selected = read(output / 'selection.json')
            if file_sha256(output / 'adapter.safetensors') != selected['adapter_sha256']:
                raise ValueError(f'Frozen selected adapter changed: {key}')
            print(f'SKIP SEARCH {key}', flush=True)
            return
        if (args.run_dir / 'frozen_selection.json').exists():
            raise ValueError('Cannot resume search after freezing test candidates')
        ds = Dataset(task['dataset'], term=task['term'], to_univariate=False)
        device = protocol['train_devices'][key]
        probe = GiftWindows(ds, 32, args.validation_limit, 42)
        input_rows = ds.target_dim + ds.past_feat_dynamic_real_dim
        contexts = bounded_contexts(args.contexts, probe.summary['train_length_max'], input_rows,
                                    args.row_context_budget)
        fingerprint = probe.fingerprint
        del probe
        validation = GiftValidation(output / 'baseline_validation.json',
                                    GiftValidationConfig(seasonality=get_seasonality(ds.freq)), protocol['base_identity'])
        candidates = []

        def fit(context, lr, stop):
            import torch

            directory = output / 'candidates' / f'ctx{context}_lr{lr:.0e}'
            batch_size = 1 if context * input_rows > args.row_context_budget / 2 else 2
            cfg = GiftTrainConfig(context_length=context, max_steps=args.max_steps, batch_size=batch_size,
                                  gradient_accumulation=4 // batch_size, learning_rate=lr, warmup_steps=40,
                                  eval_every=args.eval_every, patience=6,
                                  validation_limit=args.validation_limit, device=device, num_threads=args.threads)
            print(f'START FIT {key} context={context} lr={lr} stop={stop} device={device}', flush=True)
            checkpoint = directory / 'checkpoint.pt'
            if checkpoint.exists():
                committed = torch.load(checkpoint, map_location='cpu', weights_only=True)['step']
                stop = max(stop, committed)
            model, summary = train_gift_task(ds, task_key=key, output_dir=directory, config=cfg,
                                            model_id=BASE, revision=REVISION, stop_after_steps=stop,
                                            validation_fn=validation, validation_metadata=validation.metadata,
                                            include_initial=True, retain_validation_checkpoints=True)
            if summary['data_fingerprint'] != fingerprint:
                raise ValueError('Training/validation fingerprint changed between candidates')
            del model
            clear(device)
            print(f'DONE FIT {key} context={context} step={summary["steps"]} best={summary["best_step"]} '
                  f'score={summary["validation_loss"]:.8f}', flush=True)
            return directory

        screening_path = output / 'screen_selection.json'
        if screening_path.exists():
            screening = read(screening_path)
            candidates = [args.run_dir / p for p in screening['candidates']]
            winner = args.run_dir / screening['winner']
        else:
            for context in contexts:
                candidates.append(fit(context, 1e-4, args.screen_steps))
            winner = min(candidates, key=candidate_order)
            if read(winner / 'training_summary.json')['validation_loss'] > .99:
                context = read(winner / 'training_summary.json')['train_config']['context_length']
                candidates.append(fit(context, 3e-4, args.screen_steps))
                winner = min(candidates, key=candidate_order)
            atomic_json(screening_path, dict(winner=winner.relative_to(args.run_dir).as_posix(),
                        candidates=[p.relative_to(args.run_dir).as_posix() for p in candidates],
                        basis='fixed screening budget, validation only',
                        screening_order={p.name: candidate_order(p) for p in candidates}))
        winner_config = read(winner / 'training_summary.json')['train_config']
        fit(winner_config['context_length'], winner_config['learning_rate'], args.max_steps)
        selected_dir = min(candidates, key=candidate_order)
        selected = read(selected_dir / 'training_summary.json')
        for name in ['adapter.safetensors', 'adapter_config.json', 'training_summary.json',
                     'history.jsonl', 'initial_validation.json']:
            shutil.copyfile(selected_dir / name, output / name)
        if (base / 'LICENSE').exists():
            shutil.copyfile(base / 'LICENSE', output / 'BASE_MODEL_LICENSE')
        receipt = dict(task=key, selected_candidate=selected_dir.relative_to(args.run_dir).as_posix(),
                       best_step=selected['best_step'], validation_score=selected['validation_loss'],
                       baseline_fallback=selected['best_step'] == 0,
                       adapter_sha256=file_sha256(output / 'adapter.safetensors'),
                       data_fingerprint=fingerprint, train_device=device,
                       candidates=[dict(path=p.relative_to(args.run_dir).as_posix(),
                                        summary=read(p / 'training_summary.json'),
                                        validations=history(p)) for p in candidates],
                       selection_data='official training/validation only; test not accessed')
        atomic_json(output / 'selection.json', receipt)
        atomic_json(args.results_dir / f'{key.replace("/", "__")}.selection.json', receipt)
        print(f'SELECTED {key} candidate={selected_dir.name} step={selected["best_step"]} '
              f'score={selected["validation_loss"]:.8f} fallback={selected["best_step"] == 0}', flush=True)


def freeze(args, inventory):
    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import atomic_json
    rows = []
    for task in inventory:
        output = task_directory(args, task['key'])
        row = read(output / 'selection.json')
        sha = file_sha256(output / 'adapter.safetensors')
        if sha != row['adapter_sha256']:
            raise ValueError('Selected weight hash differs before freezing')
        rows.append(dict(task=task['key'], adapter_sha256=sha,
                         selection_sha256=file_sha256(output / 'selection.json'),
                         best_step=row['best_step'], baseline_fallback=row['baseline_fallback']))
    path = args.run_dir / 'frozen_selection.json'
    payload = dict(tasks=rows, count=len(rows), protocol_sha256=file_sha256(args.run_dir / 'protocol.json'))
    if len(rows) != 97:
        raise ValueError('All 97 task selections required before testing')
    if path.exists() and read(path) != payload:
        raise ValueError('Frozen selection changed after test access')
    atomic_json(path, payload)
    atomic_json(args.results_dir / 'frozen_selection.json', payload)


def test_namespace():
    """Use exactly the pinned official predictor and the original batch budget."""
    import gluonts.model.evaluation as ev

    from timestral_ft.gift import seasonal_error_sliced
    notebook = ROOT / '.cache/gift-eval/notebooks/timestral.ipynb'
    cells = read(notebook)['cells']
    namespace = {}
    for i in (1, 3, 4, 5):
        exec(compile(''.join(cells[i]['source']), f'{notebook}:cell{i}', 'exec'), namespace)
    os.environ['GIFT_EVAL'] = str(ROOT / '.cache/gift-eval-data')
    namespace.update(MAX_ROWS=16, MAX_TOKENS=2**17)
    ev.seasonal_error = seasonal_error_sliced
    return namespace


def test_task(args, task, namespace, reference, *, nonblocking=False):
    """Commit a paired test under one task lock; never evaluate a busy task twice."""
    import numpy as np

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.compat import PortableForecaster
    from timestral_ft.gift import atomic_json, load_gift_adapter

    key = task['key']
    directory = task_directory(args, key)
    frozen_sha = file_sha256(args.run_dir / 'frozen_selection.json')
    adapter_sha = file_sha256(directory / 'adapter.safetensors')
    frozen = read(args.run_dir / 'frozen_selection.json')
    if frozen['count'] != 97 or next(r['adapter_sha256'] for r in frozen['tasks'] if r['task'] == key) != adapter_sha:
        raise ValueError(f'Unfrozen or changed adapter: {key}')
    with (directory / '.test.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError:
            return False
        _test_task_locked(args, task, namespace, reference, frozen_sha, adapter_sha,
                          np, PortableForecaster, load_gift_adapter, atomic_json)
    return True


def _test_task_locked(args, task, namespace, reference, frozen_sha, adapter_sha,
                      np, forecaster_type, load_adapter, atomic_json):
    device = args.device
    key = task['key']
    directory = task_directory(args, key)
    for kind in ['baseline', 'lora']:
        path = args.results_dir / f'{key.replace("/", "__")}.{kind}.json'
        if path.exists():
            existing = read(path)
            if (existing['frozen_selection_sha256'] != frozen_sha
                    or existing['task'] != key or existing['kind'] != kind
                    or (kind == 'lora' and existing['adapter_sha256'] != adapter_sha)):
                raise ValueError(f'Existing test receipt identity differs: {path}')
            print(f'SKIP TEST {kind} {key}', flush=True)
            continue
        print(f'START TEST {kind} {key} device={device}', flush=True)
        started = time.perf_counter()
        if kind == 'baseline':
            fc = forecaster_type.from_pretrained(BASE, revision=REVISION, device=device, max_context=32768)
        else:
            fc = forecaster_type(load_adapter(directory, device), max_context=32768)
        metrics = namespace['evaluate_config'](fc, task['dataset'], task['term'])
        if not all(np.isfinite(v) and v >= 0 for v in metrics.values()):
            raise FloatingPointError(f'Non-finite test metric {key}')
        record = dict(task=key, kind=kind, metrics=metrics, device=device,
                      seconds=time.perf_counter()-started, frozen_selection_sha256=frozen_sha)
        if kind == 'baseline':
            record['reference_matches'] = bool(all(np.isclose(v, reference.loc[key, 'eval_metrics/'+m],
                                                                rtol=1e-3, atol=1e-5) for m, v in metrics.items()))
            record['reference_difference'] = {m: float(v-reference.loc[key, 'eval_metrics/'+m]) for m, v in metrics.items()}
        else:
            record['adapter_sha256'] = adapter_sha
        atomic_json(path, record)
        print(f'DONE TEST {kind} {key} crps={metrics["mean_weighted_sum_quantile_loss"]:.8g} '
              f'mase={metrics["MASE[0.5]"]:.8g} seconds={record["seconds"]:.1f}', flush=True)
        del fc
        clear(device)


def test(args, protocol, inventory):
    import pandas as pd

    freeze(args, inventory)
    namespace = test_namespace()
    reference = pd.read_csv(ROOT / '.cache/gift-eval/results/Timestral-209M/all_results.csv').set_index('dataset')
    for task in inventory:
        test_task(args, task, namespace, reference)


def report(args, protocol, inventory):
    import importlib.util
    import zipfile

    import pyarrow as pa
    import pyarrow.compute as pc
    import torch
    from datasets import load_from_disk
    from safetensors.torch import load_file

    from timestral_ft.artifacts import file_sha256
    from timestral_ft.gift import atomic_json
    freeze(args, inventory)
    for relative, info in read(args.results_dir / 'data_manifest.json')['files'].items():
        data = ROOT / '.cache/gift-eval-data' / relative
        if data.stat().st_size != info['bytes'] or file_sha256(data) != info['sha256']:
            raise ValueError(f'Dataset changed during experiment: {relative}')
    seasonal = {r['dataset']: r for r in csv.DictReader(
        (ROOT / '.cache/gift-eval/results/Seasonal_Naive/all_results.csv').open())}
    props = read(ROOT / '.cache/gift-eval/notebooks/dataset_properties.json')
    comparison, audit, ablations, lengths = [], [], [], {}
    submissions = {'baseline': [], 'lora': []}
    for task in inventory:
        key = task['key']
        directory = task_directory(args, key)
        selected = read(directory / 'selection.json')
        summary = read(directory / 'training_summary.json')
        candidate = args.run_dir / selected['selected_candidate']
        rows = [read(candidate / 'initial_validation.json')] + history(candidate)
        best = min(rows, key=lambda row: row['validation_loss'])
        weights = load_file(directory / 'adapter.safetensors')
        expected = load_file(candidate / 'validation_checkpoints' / f'step_{best["step"]:05d}.safetensors')
        if task['dataset'] not in lengths:
            raw = load_from_disk(str(ROOT / '.cache/gift-eval-data' / task['dataset']))
            column = raw.data.column('target')
            if pa.types.is_list(column.type.value_type) or pa.types.is_large_list(column.type.value_type):
                column = pc.list_flatten(pc.list_slice(column, 0, 1))
            lengths[task['dataset']] = pc.list_value_length(column).to_numpy()
        raw_lengths = lengths[task['dataset']]
        windows = (1 if 'm4' in task['dataset'] else
                   min(20, max(1, math.ceil(.1 * int(raw_lengths.min()) / task['horizon']))))
        training_lengths = raw_lengths - task['horizon'] * (windows + 1)
        checks = dict(selection_uses_only_validation=selected['selection_data'].endswith('test not accessed'),
                      best_validation_step=best['step'] == selected['best_step'] == summary['best_step'],
                      best_validation_value=best['validation_loss'] == selected['validation_score'],
                      selected_checkpoint_exact=set(weights) == set(expected) and all(torch.equal(v, expected[k]) for k, v in weights.items()),
                      finite_adapter=all(bool(v.isfinite().all()) for v in weights.values()),
                      only_lora=all(k.endswith(('lora_A', 'lora_B')) for k in weights),
                      parameter_count=sum(v.numel() for v in weights.values()) == summary['trainable_parameters'],
                      base_revision=summary['base_revision'] == REVISION,
                      base_weight_hash=summary['base_weights_sha256'] == protocol['base_identity']['weights_sha256'],
                      official_prefix_lengths=summary['data']['train_length_min'] == int(training_lengths.min())
                      and summary['data']['train_length_max'] == int(training_lengths.max())
                      and summary['data']['series'] == len(training_lengths)
                      and summary['data']['horizon'] == task['horizon'],
                      fallback_is_identity=selected['best_step'] != 0 or all(not bool(v.count_nonzero()) for k, v in weights.items() if k.endswith('lora_B')))
        screening = read(directory / 'screen_selection.json')
        checks['continuation_completed'] = read(args.run_dir / screening['winner'] / 'training_summary.json')['completed']
        checks['all_candidate_histories_valid'] = True
        for item in selected['candidates']:
            c = args.run_dir / item['path']
            train_rows = [json.loads(line) for line in (c / 'history.jsonl').read_text().splitlines()]
            checks['all_candidate_histories_valid'] &= (
                len(train_rows) == item['summary']['steps'] >= args.screen_steps
                and [r['step'] for r in train_rows] == list(range(1, len(train_rows)+1))
                and all(math.isfinite(r['train_loss']) and math.isfinite(r['gradient_norm']) for r in train_rows)
                and item['summary']['data_fingerprint'] == summary['data_fingerprint'])
            all_rows = [read(c / 'initial_validation.json')] + history(c)
            for budget in sorted({args.screen_steps, args.max_steps}):
                available = [r for r in all_rows if r['step'] <= budget]
                for objective, metric in [('legacy_pinball', 'scaled_pinball'), ('crps', 'crps'),
                                          ('mase', 'mase'), ('pareto', 'score')]:
                    chosen = min(available, key=lambda r: r['validation_metrics'][metric])
                    ablations.append(dict(task=key, candidate=item['path'], context=item['summary']['train_config']['context_length'],
                                          requested_budget=budget, observed_steps=max(r['step'] for r in available),
                                          objective=objective, chosen_step=chosen['step'],
                                          validation_crps_ratio=chosen['validation_metrics']['crps_ratio'],
                                          validation_mase_ratio=chosen['validation_metrics']['mase_ratio']))
        audit.append(dict(task=key, checks=checks))
        base = read(args.results_dir / f'{key.replace("/", "__")}.baseline.json')
        tuned = read(args.results_dir / f'{key.replace("/", "__")}.lora.json')
        if tuned['adapter_sha256'] != selected['adapter_sha256']:
            raise ValueError('Evaluated adapter differs from frozen selection')
        if not base['reference_matches']:
            raise ValueError(f'Original 209M does not reproduce official metrics: {key}')
        row = dict(task=key, best_step=selected['best_step'], fallback=selected['baseline_fallback'])
        for label, name in [('crps', 'mean_weighted_sum_quantile_loss'), ('mase', 'MASE[0.5]')]:
            b, t, s = base['metrics'][name], tuned['metrics'][name], float(seasonal[key]['eval_metrics/'+name])
            row.update({f'baseline_{label}': b, f'lora_{label}': t,
                        f'baseline_normalized_{label}': b/s, f'lora_normalized_{label}': t/s,
                        f'{label}_ratio': t/b})
        comparison.append(row)
        prop = props[key.split('/')[0]]
        for kind, record in [('baseline', base), ('lora', tuned)]:
            submissions[kind].append(dict(dataset=key, model=MODEL if kind == 'lora' else 'Timestral-209M-reproduced',
                                          **{'eval_metrics/'+k: v for k, v in record['metrics'].items()},
                                          domain=prop['domain'], num_variates=prop['num_variates']))
    if not all(all(r['checks'].values()) for r in audit):
        atomic_json(args.results_dir / 'audit_failures.json', audit)
        raise ValueError('Independent selection audit failed')
    aggregates = dict(tasks=len(comparison), fallback_tasks=sum(r['fallback'] for r in comparison))
    for label in ['crps', 'mase']:
        b = math.exp(sum(math.log(r[f'baseline_normalized_{label}']) for r in comparison)/len(comparison))
        t = math.exp(sum(math.log(r[f'lora_normalized_{label}']) for r in comparison)/len(comparison))
        aggregates[label] = dict(baseline=b, lora=t, change_percent=(t/b-1)*100,
                                 improved=sum(r[f'{label}_ratio'] < 1-1e-6 for r in comparison),
                                 regressed=sum(r[f'{label}_ratio'] > 1+1e-6 for r in comparison))
    for name, rows in [('task_comparison', comparison), ('validation_ablations', ablations)] + [(k+'_all_results', v) for k, v in submissions.items()]:
        with (args.results_dir / f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(args.results_dir / 'audit.json', dict(tasks=audit, all_checks_pass=True))
    atomic_json(args.results_dir / 'aggregates.json', aggregates)
    submission = args.results_dir / 'submission' / MODEL
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.results_dir / 'lora_all_results.csv', submission / 'all_results.csv')
    atomic_json(submission / 'config.json', dict(model=MODEL, model_type='fine-tuned', model_dtype='float32',
                model_link=None, code_link=None, org='elarazju', testdata_leakage='No', replication_code_available='No'))
    spec = importlib.util.spec_from_file_location('gift_validator', ROOT / '.cache/gift-eval/scripts/validate_results.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    errors = validator.validate_submission(submission)
    if errors:
        raise ValueError(f'Official submission format failed: {errors}')
    atomic_json(args.results_dir / 'submission_validation.json', dict(schema_valid=True, submitted=False,
                weights_published=False, remaining=['model_link', 'code_link', 'replication_code_available'],
                leakage_note='Adaptation uses official training/validation only; base entry declares no test leakage, without independent pretraining audit.'))
    print(json.dumps(aggregates, indent=2), flush=True)
    gains = all(aggregates[k]['change_percent'] < 0 for k in ['crps', 'mase'])
    text = '# Timestral-209M GIFT-Eval adaptation\n\n'
    text += f'Completed all 97 tasks. Both aggregate metrics improved: **{gains}**.\n\n'
    text += '| Metric | Base | Adapted | Change |\n|---|---:|---:|---:|\n'
    for k in ['crps', 'mase']:
        r = aggregates[k]
        text += f'| {k.upper()} | {r["baseline"]:.8f} | {r["lora"]:.8f} | {r["change_percent"]:+.4f}% |\n'
    text += f'\nValidation selected step-zero identity adapters for {aggregates["fallback_tasks"]} tasks. '
    text += 'All tasks trained; fallback is disclosed. These are normalized geometric means, not official ranks. '
    text += 'Weights and leaderboard submission have not been published by this run.\n'
    text += '\nTraining screens are prefixes of the longer cosine schedule. Only the chosen candidate is extended; '
    text += 'validation_ablations.csv describes observed checkpoints, not a complete factorial search. '
    text += 'All three selection objectives are compared on validation only, and no test-driven retry is performed.\n'
    if (args.results_dir / 'parallel_test_execution.json').exists():
        text += '\nFinal testing used one MPS worker and one CPU worker for short histories. '
        text += 'See parallel_test_execution.json for routing/source hashes and each test receipt for its device. '
        text += 'All frozen adapters, official test windows, context limits, and prediction settings were retained.\n'
    (args.results_dir / 'README.md').write_text(text)
    chinese = '# Timestral-209M：GIFT-Eval 完整微调实验\n\n'
    chinese += f'已完成全部 97 个任务的搜索、冻结、成对回测与审计。两项聚合指标均改善：**{gains}**。\n\n'
    chinese += '| 指标 | 原模型 | 微调后 | 变化 |\n|---|---:|---:|---:|\n'
    for k in ['crps', 'mase']:
        r = aggregates[k]
        chinese += f'| {k.upper()} | {r["baseline"]:.8f} | {r["lora"]:.8f} | {r["change_percent"]:+.4f}% |\n'
    chinese += f'\n验证集选择了 {aggregates["fallback_tasks"]} 个 step 0 回退；所有任务均实际训练。'
    chinese += '上述数值为相对 Seasonal Naive 的几何均值，越低越好，不代表官方名次。\n'
    chinese += '\n上下文和延长预算只依据验证集选择；200 步筛选沿用 800 步的学习率计划。'
    chinese += '仅优胜方案延长，因此消融记录属于已观测检查点比较，不是完全析因实验。'
    chinese += '全部 97 份权重在测试前冻结；测试结果不会触发重新训练或换配方。'
    chinese += '权重尚未公开，提交草稿仅通过格式校验，尚未正式提交榜单。\n'
    if (args.results_dir / 'parallel_test_execution.json').exists():
        chinese += '\n正式测试使用一个 MPS 进程与一个处理短历史任务的 CPU 进程并行。'
        chinese += 'parallel_test_execution.json 记录调度规则与代码校验值，各任务测试记录注明实际设备。'
        chinese += '冻结权重、全部官方测试窗口、上下文上限和预测设置保持一致。\n'
    (args.results_dir / 'README.zh-CN.md').write_text(chinese)
    package = ROOT / 'dist/timestral-ft-209m-lora-gift-eval-adapters.zip'
    package.parent.mkdir(exist_ok=True)
    temporary = package.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('README.txt', '97 validation-selected task-specific adapters.\n'
                         'Some tasks explicitly retain the step-zero base model; see selection.json.\n'
                         'Use timestral_ft.gift.load_gift_adapter(tasks/<canonical_key_with_double_underscores>).\n'
                         'Base weights are not bundled. Derivative weights retain the base model license.\n')
        for task in inventory:
            directory = task_directory(args, task['key'])
            for name in ['adapter.safetensors', 'adapter_config.json', 'training_summary.json',
                         'history.jsonl', 'selection.json', 'BASE_MODEL_LICENSE']:
                archive.write(directory / name, f'tasks/{directory.name}/{name}')
        for name in ['protocol.json', 'frozen_selection.json', 'aggregates.json', 'task_comparison.csv',
                     'validation_ablations.csv', 'audit.json', 'README.md', 'README.zh-CN.md']:
            archive.write(args.results_dir / name, f'experiment/{name}')
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupted adapter archive')
    temporary.replace(package)
    atomic_json(args.results_dir / 'package.json', dict(path=str(package), bytes=package.stat().st_size,
                sha256=file_sha256(package), weights_published=False))


def main():
    args = cli()
    if not 0 < args.screen_steps <= args.max_steps or args.screen_steps % args.eval_every:
        raise ValueError('Screen budget must be an evaluation-aligned prefix of maximum budget')
    if any(c < 32 or c > 32768 or c % 32 for c in args.contexts):
        raise ValueError('Context candidates must be patch-aligned, from 32 to 32768')
    import torch
    torch.set_num_threads(args.threads)
    os.environ['GIFT_EVAL'] = str(ROOT / '.cache/gift-eval-data')
    protocol, inventory, base = open_protocol(args)
    selected = [t for t in inventory if not args.task or t['key'] in args.task]
    if args.task and len(selected) != len(set(args.task)):
        raise ValueError('Unknown task')
    if args.stage in ['all', 'search']:
        for task in selected:
            device = protocol['train_devices'][task['key']]
            if args.worker_device and device.split(':')[0] != args.worker_device:
                continue
            if device == 'cpu':
                search_task(args, protocol, task, base)
            else:
                with (args.run_dir / '.accelerator.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    search_task(args, protocol, task, base)
    if args.stage in ['all', 'test']:
        if args.task or args.worker_device:
            raise ValueError('Final testing requires the complete 97-task frozen set')
        with (args.run_dir / '.test.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (args.run_dir / '.accelerator.lock').open('a') as accelerator:
                fcntl.flock(accelerator, fcntl.LOCK_EX)
                test(args, protocol, inventory)
    if args.stage in ['all', 'report']:
        if args.task or args.worker_device:
            raise ValueError('Complete report requires all tasks')
        report(args, protocol, inventory)


if __name__ == '__main__':
    main()
