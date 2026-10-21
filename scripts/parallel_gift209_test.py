"""Resume the frozen 97-task test with one CPU and one MPS worker.

Only execution order/device placement changes. Training sources, adapters,
forecast semantics, batch budgets, and all test windows remain unchanged.
"""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import launch_gift209 as controller
import run_gift209_search as runner

ROOT = runner.ROOT


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/gift-eval-209m-search-v2')
    p.add_argument('--results-dir', type=Path, default=ROOT / 'results/gift_eval_209m_lora_v2')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--worker', choices=['cpu', 'mps'])
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--takeover-pid', type=int)
    p.add_argument('--handoff-after', help='Wait for this paired test to commit before replacing the serial controller')
    return p.parse_args()


def experiment_args(options):
    protocol = runner.read(options.run_dir / 'protocol.json')
    train = protocol['train']
    return SimpleNamespace(run_dir=options.run_dir, results_dir=options.results_dir, threads=options.threads,
                           stage='test', device='mps', worker_device=None, task=None,
                           contexts=train['contexts'], screen_steps=train['screen_steps'],
                           max_steps=train['max_steps'], eval_every=train['eval_every'],
                           validation_limit=train['validation_limit'], row_context_budget=train['row_context_budget'])


def execution_plan(args, inventory):
    """Place short histories on spare CPU cores; use no test scores for routing."""
    preferred = []
    for task in inventory:
        receipt = runner.read(runner.task_directory(args, task['key']) / 'training_summary.json')
        shape = receipt['data']
        # All official configurations have at most 20 rolling test windows.
        # This conservative bound includes every target's full test history.
        bound = shape['train_length_max'] + 21 * shape['horizon']
        if bound <= 4096:
            preferred.append((bound * shape['series'], task['key']))
    cpu = [key for _, key in sorted(preferred)]
    return dict(cpu=cpu, mps=[t['key'] for t in inventory if t['key'] not in cpu] + cpu,
                policy='CPU prefers training_max_length + 21*horizon <= 4096, ordered by length bound times series; MPS steals unclaimed CPU tasks after heavy tasks',
                cpu_threads=args.threads, mps_threads=args.threads, max_rows=16, max_tokens=2**17,
                context=32768, cpu_workers=1, mps_workers=1)


def complete(args, key):
    return all((args.results_dir / f'{key.replace("/", "__")}.{kind}.json').exists()
               for kind in ['baseline', 'lora'])


def worker(options):
    import pandas as pd
    import torch

    args = experiment_args(options)
    torch.set_num_threads(options.threads)
    _, inventory, _ = runner.open_protocol(args)
    plan = runner.read(args.results_dir / 'parallel_test_execution.json')['plan']
    lookup = {t['key']: t for t in inventory}
    args.device = options.worker
    namespace = runner.test_namespace()
    reference = pd.read_csv(ROOT / '.cache/gift-eval/results/Timestral-209M/all_results.csv').set_index('dataset')
    guard = ((args.run_dir / '.accelerator.lock').open('a') if options.worker == 'mps' else nullcontext())
    with guard as accelerator:
        if accelerator is not None:
            fcntl.flock(accelerator, fcntl.LOCK_EX)
        keys = plan[options.worker]
        while True:
            pending = [key for key in keys if not complete(args, key)]
            if not pending:
                return
            for key in pending:
                runner.test_task(args, lookup[key], namespace, reference, nonblocking=True)
            time.sleep(1)


def process_command(pid):
    return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], text=True,
                          capture_output=True, check=False).stdout.strip()


def acquire_execution_lock(handle, *, after_handoff=False, timeout=30.0):
    """Allow old handoff cleanup helpers a bounded time to release inherited FDs."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if not after_handoff or time.monotonic() >= deadline:
                raise
            time.sleep(.1)


def handoff(options, state):
    """Replace only the idle supervisor and the evaluation between task commits."""
    old_pid = options.takeover_pid
    if not old_pid:
        return
    if 'scripts/launch_gift209.py' not in process_command(old_pid):
        raise ValueError('Takeover PID is not the original experiment supervisor')
    boundary = options.handoff_after
    inventory = runner.read(ROOT / 'results/gift_eval_feasibility/task_inventory.json')
    if not boundary or boundary not in {t['key'] for t in inventory}:
        raise ValueError('An official paired-test boundary is required for takeover')
    transition = options.results_dir / 'parallel_transition.json'
    controller.atomic_json(transition, dict(phase='waiting_for_task_boundary', task=boundary,
                                           old_controller_pid=old_pid, new_controller_pid=os.getpid()))
    print(f'WAIT HANDOFF after {boundary}', flush=True)
    while not complete(options, boundary):
        live = runner.read(options.results_dir / 'progress.json')
        if live.get('phase') == 'failed' or 'scripts/launch_gift209.py' not in process_command(old_pid):
            raise RuntimeError('Serial controller stopped before the requested boundary')
        time.sleep(.25)
    os.kill(old_pid, signal.SIGSTOP)
    try:
        live = runner.read(options.results_dir / 'progress.json')
        children = [w for w in live['workers'] if w['name'] == 'paired_test' and w['returncode'] is None]
        for child in children:
            pid = child['pid']
            if 'run_gift209_search.py' in process_command(pid):
                os.kill(pid, signal.SIGTERM)
                for _ in range(100):
                    if 'run_gift209_search.py' not in process_command(pid):
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Serial test worker has not stopped; refusing overlapping workers')
        # The original supervisor marks a requested shutdown as an experiment
        # failure. It is stopped outside all model computation/file commits;
        # replacing it directly avoids a false failure for the ranking watcher.
        os.kill(old_pid, signal.SIGKILL)
        for _ in range(100):
            if 'scripts/launch_gift209.py' not in process_command(old_pid):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original supervisor still owns its execution lock')
    except BaseException:
        if 'scripts/launch_gift209.py' in process_command(old_pid):
            os.kill(old_pid, signal.SIGCONT)
        raise
    controller.atomic_json(transition, dict(phase='handed_off', task=boundary, previous_state=live,
                                           new_controller_pid=os.getpid(), completed_results_preserved=True))
    state['previous_controller_pid'] = old_pid
    print(f'HANDED OFF after {boundary}; completed receipts preserved', flush=True)


def supervise(options):
    from timestral_ft.artifacts import file_sha256

    args = experiment_args(options)
    _, inventory, _ = runner.open_protocol(args)
    # Only the full 97-task set may be frozen. Workers cannot select adapters.
    runner.freeze(args, inventory)
    frozen = runner.read(args.run_dir / 'frozen_selection.json')
    if frozen['count'] != 97 or len(frozen['tasks']) != 97:
        raise ValueError('Complete frozen selection required')
    previous = runner.read(args.results_dir / 'progress.json')
    plan = execution_plan(args, inventory)
    provenance = dict(plan=plan, frozen_selection_sha256=file_sha256(args.run_dir / 'frozen_selection.json'),
                      training_protocol_unchanged=True,
                      source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in
                                     [Path(__file__), ROOT / 'scripts/run_gift209_search.py']},
                      note='Device/order change only; no hyperparameter or adapter reselection from test outcomes')
    destination = args.results_dir / 'parallel_test_execution.json'
    if destination.exists() and runner.read(destination) != provenance:
        raise ValueError('Parallel execution plan changed; preserve and audit the previous manifest first')
    controller.atomic_json(destination, provenance)
    for name, sha in provenance['source_sha256'].items():
        source = ROOT / name
        snapshot = args.run_dir / 'test_source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and file_sha256(snapshot) != sha:
            raise ValueError(f'Test source snapshot changed: {name}')
        if not snapshot.exists():
            snapshot.write_bytes(source.read_bytes())
    state = dict(controller_pid=os.getpid(), started_at_utc=previous['started_at_utc'],
                 parallel_started_at_utc=datetime.now(timezone.utc).isoformat(), phase='frozen_paired_test')
    handoff(options, state)
    children, handles = [], []
    awake = None

    def record():
        state.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),
                     progress=controller.progress(args.run_dir, args.results_dir),
                     workers=[dict(name=n, pid=p.pid, returncode=p.poll()) for n, p in children])
        for p in [args.run_dir / 'status.json', args.results_dir / 'progress.json']:
            controller.atomic_json(p, state)

    def launch(name, command):
        handle = (args.run_dir / 'logs' / f'{name}.log').open('a')
        handles.append(handle)
        p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                             stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1'))
        children.append((name, p))
        return p

    def wait_for(processes):
        while True:
            record()
            for p in processes:
                if p.poll() not in [None, 0]:
                    raise RuntimeError(f'Test/report worker {p.pid} exited {p.returncode}; see worker log')
            if all(p.poll() == 0 for p in processes):
                return
            time.sleep(10)

    def terminate(signum, frame):
        raise SystemExit(f'Parallel supervisor received signal {signum}; completed tests permit resumption')

    signal.signal(signal.SIGTERM, terminate)
    with (args.run_dir / '.controller.lock').open('a') as lock:
        acquire_execution_lock(lock, after_handoff=bool(options.takeover_pid))
        with (args.run_dir / '.test.lock').open('a') as testing:
            acquire_execution_lock(testing, after_handoff=bool(options.takeover_pid))
            try:
                if sys.platform == 'darwin':
                    awake = subprocess.Popen(['caffeinate', '-is', '-w', str(os.getpid())])
                command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
                           '--results-dir', str(args.results_dir), '--threads', str(options.threads)]
                wait_for([launch('test_cpu', command + ['--worker', 'cpu']),
                          launch('test_mps', command + ['--worker', 'mps'])])
                state['phase'] = 'audit_and_package'
                report = [sys.executable, str(ROOT / 'scripts/run_gift209_search.py'), '--stage', 'report',
                          '--run-dir', str(args.run_dir), '--results-dir', str(args.results_dir)]
                wait_for([launch('report', report)])
                state.update(phase='complete', aggregates=runner.read(args.results_dir / 'aggregates.json'))
                record()
            except BaseException as error:
                state.update(phase='failed', error=str(error))
                record()
                raise
            finally:
                for _, child in children:
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


def main():
    options = arguments()
    if options.plan_only:
        args = experiment_args(options)
        inventory = runner.read(ROOT / 'results/gift_eval_feasibility/task_inventory.json')
        print(execution_plan(args, inventory), flush=True)
    elif options.worker:
        worker(options)
    else:
        with (options.run_dir / '.parallel-controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervise(options)


if __name__ == '__main__':
    main()
