"""Run disjoint CPU/MPS search workers, then the frozen full test and report.

This small supervisor has no model-selection logic. It writes progress/error
receipts and can outlive an interactive terminal when launched detached.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def progress(run, results):
    selections = list((run / 'tasks').glob('*/selection.json'))
    logged_steps = 0
    for path in (run / 'tasks').glob('*/candidates/*/history.jsonl'):
        lines = path.read_text().splitlines()
        for line in reversed(lines):
            try:
                logged_steps += json.loads(line)['step']
                break
            except (ValueError, KeyError):
                continue
    return dict(selected_tasks=len(selections), total_tasks=97, logged_optimizer_steps=logged_steps,
                baseline_tests=len(list(results.glob('*.baseline.json'))),
                adapted_tests=len(list(results.glob('*.lora.json'))),
                test_candidates_frozen=(run / 'frozen_selection.json').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=ROOT / '.cache/gift-probe-env/bin/python')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/gift-eval-209m-search-v1')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/gift_eval_209m_lora')
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    logs = args.run_dir / 'logs'
    logs.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    state = dict(controller_pid=os.getpid(), started_at_utc=started, phase='starting')
    processes, handles = [], []
    awake = None
    # Preserve the venv executable symlink: resolving it would select the base
    # interpreter and silently discard the isolated GIFT environment.
    command = [str(args.python.absolute()), str(ROOT / 'scripts/run_gift209_search.py'),
               '--run-dir', str(args.run_dir.resolve()), '--results-dir', str(args.results_dir.resolve())]

    def record():
        state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     progress=progress(args.run_dir, args.results_dir),
                     workers=[dict(name=name, pid=p.pid, returncode=p.poll()) for name, p in processes])
        for path in [args.run_dir / 'status.json', args.results_dir / 'progress.json']:
            atomic_json(path, state)

    def launch(name, extra):
        handle = (logs / f'{name}.log').open('a')
        handles.append(handle)
        p = subprocess.Popen(command + extra, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        processes.append((name, p))
        return p

    def wait_for(children):
        while True:
            record()
            for name, child in children:
                code = child.poll()
                if code is not None and code != 0:
                    tail = (logs / f'{name}.log').read_text()[-5000:]
                    raise RuntimeError(f'{name} exited {code}; {tail}')
            if all(child.poll() is not None for _, child in children):
                return
            time.sleep(15)

    def terminate(signum, frame):
        raise SystemExit(f'Controller received signal {signum}; checkpoints permit resumption')

    signal.signal(signal.SIGTERM, terminate)
    with (args.run_dir / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if sys.platform == 'darwin':
                awake = subprocess.Popen(['caffeinate', '-is', '-w', str(os.getpid())],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            state['phase'] = 'validation_search'
            cpu = launch('search_cpu', ['--stage', 'search', '--worker-device', 'cpu'])
            mps = launch('search_mps', ['--stage', 'search', '--worker-device', 'mps'])
            wait_for([('search_cpu', cpu), ('search_mps', mps)])
            state['phase'] = 'frozen_paired_test'
            child = launch('paired_test', ['--stage', 'test'])
            wait_for([('paired_test', child)])
            state['phase'] = 'audit_and_package'
            child = launch('report', ['--stage', 'report'])
            wait_for([('report', child)])
            state['phase'] = 'complete'
            state['aggregates'] = json.loads((args.results_dir / 'aggregates.json').read_text())
            record()
        except BaseException as error:
            state.update(phase='failed', error=str(error))
            record()
            raise
        finally:
            for _, child in processes:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
            if awake is not None:
                awake.terminate()
            for handle in handles:
                handle.close()


if __name__ == '__main__':
    main()
