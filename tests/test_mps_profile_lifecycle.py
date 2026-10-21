import importlib.util
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_profiler_resumes_both_production_processes_after_failure(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/profile_gift209_mps.py'
    spec = importlib.util.spec_from_file_location('mps_profile_lifecycle', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = tmp_path / 'runs/gift-eval-209m-search-v2/frozen_selection.json'
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{}')
    args = SimpleNamespace(worker_input=None, jobs=4, threads=4, output=tmp_path/'profile',
                           pause_pending_pid=101, pause_pid=102)
    monkeypatch.setattr(module, 'arguments', lambda: args)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'command', lambda pid: 'parallel_gift209_test.py' if pid == 101 else 'run_gift209_search.py')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test hardware'))
    monkeypatch.setattr(module.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(module.time, 'sleep', lambda *a: None)
    signals = []
    monkeypatch.setattr(module.os, 'kill', lambda pid, sig: signals.append((pid, sig)))

    def preparation_failure(*a):
        raise RuntimeError('preflight failed')

    monkeypatch.setattr(module, 'prepare', preparation_failure)
    with pytest.raises(RuntimeError, match='preflight failed'):
        module.main()
    assert signals == [(101, signal.SIGSTOP), (102, signal.SIGSTOP),
                       (102, signal.SIGCONT), (101, signal.SIGCONT)]
    report = json.loads((args.output/'report.json').read_text())
    assert report['status'] == 'failed'
    assert report['production_processes_resumed'] == [101, 102]


def test_profiler_never_suspends_unexpected_process(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/profile_gift209_mps.py'
    spec = importlib.util.spec_from_file_location('mps_profile_wrong_pid', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = tmp_path / 'runs/gift-eval-209m-search-v2/frozen_selection.json'
    frozen.parent.mkdir(parents=True)
    frozen.write_text('{}')
    args = SimpleNamespace(worker_input=None, jobs=4, threads=4, output=tmp_path/'profile',
                           pause_pending_pid=101, pause_pid=102)
    monkeypatch.setattr(module, 'arguments', lambda: args)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'command', lambda pid: 'unrelated-app')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='test hardware'))
    monkeypatch.setattr(module.signal, 'signal', lambda *a: None)
    signals = []
    monkeypatch.setattr(module.os, 'kill', lambda *a: signals.append(a))
    with pytest.raises(ValueError, match='Unexpected process identity'):
        module.main()
    assert signals == []
