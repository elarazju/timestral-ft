import fcntl
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from timestral_ft.artifacts import file_sha256


@pytest.fixture
def parallel(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('gift_parallel_test', scripts / 'parallel_gift209_test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def task_run(parallel, tmp_path, monkeypatch):
    import timestral_ft.compat as compat
    import timestral_ft.gift as gift

    run, results = tmp_path / 'run', tmp_path / 'results'
    directory = run / 'tasks/test__D__short'
    directory.mkdir(parents=True)
    results.mkdir()
    (directory / 'adapter.safetensors').write_bytes(b'frozen-adapter')
    sha = file_sha256(directory / 'adapter.safetensors')
    (run / 'frozen_selection.json').write_text(json.dumps(dict(count=97, tasks=[dict(task='test/D/short', adapter_sha256=sha)])))
    args = SimpleNamespace(run_dir=run, results_dir=results, device='cpu', threads=4)
    task = dict(key='test/D/short', dataset='test', term='short')
    calls = []

    class FakeForecaster:
        def __init__(self, model=None, **kwargs):
            self.kind = 'lora'

        @classmethod
        def from_pretrained(cls, *a, **kw):
            instance = cls()
            instance.kind = 'baseline'
            return instance

    def evaluate(fc, dataset, term):
        assert (dataset, term) == ('test', 'short')
        calls.append(fc.kind)
        return {'mean_weighted_sum_quantile_loss': 2., 'MASE[0.5]': 3.}

    monkeypatch.setattr(compat, 'PortableForecaster', FakeForecaster)
    monkeypatch.setattr(gift, 'load_gift_adapter', lambda *a: None)
    monkeypatch.setattr(parallel.runner, 'clear', lambda *a: None)
    reference = pd.DataFrame([{'dataset': task['key'], 'eval_metrics/mean_weighted_sum_quantile_loss': 2.,
                               'eval_metrics/MASE[0.5]': 3.}]).set_index('dataset')
    return args, task, dict(evaluate_config=evaluate), reference, calls


def test_paired_test_resume_preserves_frozen_identity(parallel, task_run):
    args, task, namespace, reference, calls = task_run
    assert parallel.runner.test_task(args, task, namespace, reference)
    assert calls == ['baseline', 'lora']
    assert parallel.runner.test_task(args, task, namespace, reference)
    assert calls == ['baseline', 'lora']
    baseline = parallel.runner.read(args.results_dir / 'test__D__short.baseline.json')
    assert baseline['reference_matches'] and baseline['device'] == 'cpu'
    frozen = args.run_dir / 'frozen_selection.json'
    frozen.write_text(frozen.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt identity'):
        parallel.runner.test_task(args, task, namespace, reference)


def test_busy_task_cannot_be_evaluated_by_second_worker(parallel, task_run):
    args, task, namespace, reference, calls = task_run
    with (parallel.runner.task_directory(args, task['key']) / '.test.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not parallel.runner.test_task(args, task, namespace, reference, nonblocking=True)
        assert calls == []
    assert parallel.runner.test_task(args, task, namespace, reference, nonblocking=True)
    assert calls == ['baseline', 'lora']


def test_interrupted_pair_only_resumes_missing_half(parallel, task_run):
    args, task, namespace, reference, calls = task_run
    original = namespace['evaluate_config']

    def fail_lora(fc, dataset, term):
        if fc.kind == 'lora':
            raise RuntimeError('power interruption')
        return original(fc, dataset, term)

    with pytest.raises(RuntimeError, match='power interruption'):
        parallel.runner.test_task(args, task, dict(evaluate_config=fail_lora), reference)
    assert calls == ['baseline']
    assert parallel.runner.test_task(args, task, namespace, reference)
    assert calls == ['baseline', 'lora']


def test_changed_adapter_rejected_before_evaluation(parallel, task_run):
    args, task, namespace, reference, calls = task_run
    (parallel.runner.task_directory(args, task['key']) / 'adapter.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed adapter'):
        parallel.runner.test_task(args, task, namespace, reference)
    assert calls == []


def test_cpu_routing_uses_shapes_and_keeps_heavy_tasks_first(parallel, tmp_path):
    args = SimpleNamespace(run_dir=tmp_path, threads=4)
    inventory = [dict(key=k) for k in ['small', 'large', 'tiny']]
    for key, size, horizon, series in [('small', 200, 12, 20), ('large', 20000, 96, 1), ('tiny', 30, 8, 1)]:
        p = parallel.runner.task_directory(args, key)
        p.mkdir(parents=True)
        (p / 'training_summary.json').write_text(json.dumps(dict(data=dict(
            train_length_max=size, horizon=horizon, series=series))))
    plan = parallel.execution_plan(args, inventory)
    assert plan['cpu'] == ['tiny', 'small']
    assert plan['mps'] == ['large', 'tiny', 'small']
    assert plan['context'] == 32768 and plan['max_tokens'] == 2**17 and plan['max_rows'] == 16


def test_explicit_cpu_queue_can_claim_multiseries_with_all_input_rows(parallel, tmp_path, monkeypatch):
    import extend_gift209_cpu as extension

    monkeypatch.setattr(extension, 'ROOT', tmp_path)
    args = SimpleNamespace(run_dir=tmp_path/'run', results_dir=tmp_path/'results')
    args.results_dir.mkdir()
    task = dict(key='large/H/medium', dataset='large/H')
    directory = extension.runner.task_directory(args, task['key'])
    directory.mkdir(parents=True)
    (directory/'training_summary.json').write_text(json.dumps(dict(data=dict(
        series=30, train_length_max=15000, horizon=480))))
    metadata = tmp_path/'.cache/gift-eval-data/large/H'
    metadata.mkdir(parents=True)
    (metadata/'dataset_info.json').write_text(json.dumps(dict(features=dict(
        target=dict(length=1), past_feat_dynamic_real=dict(length=7)))))
    # The earlier automatic policy would leave this multi-series job on MPS.
    assert extension.cpu_queue(args, [task]) == []
    selected = extension.cpu_queue(args, [task], [task['key']])
    assert len(selected) == 1 and selected[0]['total_input_rows'] == 8
    (args.results_dir/'large__H__medium.baseline.json').write_text('{}')
    (args.results_dir/'large__H__medium.lora.json').write_text('{}')
    assert extension.cpu_queue(args, [task], [task['key']]) == []


def test_explicit_cpu_queue_rejects_unknown_task_before_loading_data(parallel, tmp_path):
    import extend_gift209_cpu as extension

    with pytest.raises(ValueError, match='Unknown CPU task'):
        extension.cpu_queue(SimpleNamespace(run_dir=tmp_path, results_dir=tmp_path), [], ['unknown/H/short'])


def test_handoff_waits_for_old_helper_lock_release(parallel, tmp_path):
    path = tmp_path / 'execution.lock'
    program = (
        'import fcntl, sys; '
        'handle = open(sys.argv[1], "a"); '
        'fcntl.flock(handle, fcntl.LOCK_EX); '
        'print("locked", flush=True); '
        'sys.stdin.readline()'
    )
    process = subprocess.Popen([sys.executable, '-c', program, str(path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    timer = None
    try:
        assert process.stdout.readline().strip() == 'locked'
        with path.open('a') as handle:
            # A normal launch must not silently wait for an unrelated controller.
            with pytest.raises(BlockingIOError):
                parallel.acquire_execution_lock(handle)
            with pytest.raises(BlockingIOError):
                parallel.acquire_execution_lock(handle, after_handoff=True, timeout=.05)

            def release():
                process.stdin.write('release\n')
                process.stdin.flush()

            timer = threading.Timer(.15, release)
            timer.start()
            parallel.acquire_execution_lock(handle, after_handoff=True, timeout=5)
            assert process.wait(timeout=5) == 0
    finally:
        if timer is not None:
            timer.join(timeout=5)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
