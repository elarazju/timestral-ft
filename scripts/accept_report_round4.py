"""Isolated real-5M functional and progressive context-capacity acceptance.

Long contexts use B1/V1, H32, one forward/backward and optimizer step. These are
capacity checks, not training-quality experiments or pretraining reproduction.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_report_round4 import MemorySampler, release, source_identity, synchronize
from timestral.forecaster import TimestralForecaster

from timestral_ft import FineTunedModel, SyntheticConfig, TrainConfig, TrainingStage, finetune
from timestral_ft.artifacts import file_sha256, write_json
from timestral_ft.compat import (
    PortableForecaster,
    differentiable_normalized_forecast,
    load_model,
    normalized_pinball_loss,
)
from timestral_ft.lora import apply_lora
from timestral_ft.synthetic import generate_sequence

MODEL = 'mistralai/Timestral-5M'
REVISION = 'bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f'


def worker(device, mode, context, root):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.perf_counter()
    record = {'sources': source_identity(), 'script_sha256': file_sha256(__file__), 'device': device, 'mode': mode, 'context': context, 'batch': 1, 'variables': 1, 'horizon': 32}
    with MemorySampler(device) as memory:
        if context == 64:
            values = generate_sequence(384, 1, SyntheticConfig(), torch.Generator().manual_seed(914))[0].numpy()
            frame = pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=384, freq='h'), 'target': values})
            cfg = TrainConfig(mode=mode, device=device, context_length=64, prediction_length=32, max_steps=4,
                batch_size=1, gradient_accumulation=1, eval_every=2, warmup_steps=1, num_threads=2,
                training_strategy='dhf', loss_space='normalized', validation_loss_space='history_scaled',
                lora_rank=8, lora_alpha=16, lora_dropout=.1,
                synthetic=SyntheticConfig(), stages=[TrainingStage(2, 32, 1), TrainingStage(2, 64, .5)])
            args = dict(model_id=MODEL, revision=REVISION, train_config=cfg)
            direct = finetune(frame.iloc[:256], frame.iloc[256:], output_dir=root / 'direct', **args)
            del direct
            release(device)
            partial = finetune(frame.iloc[:256], frame.iloc[256:], output_dir=root / 'resume', stop_after_steps=2, **args)
            del partial
            release(device)
            result = finetune(frame.iloc[:256], frame.iloc[256:], output_dir=root / 'resume', resume=root / 'resume/checkpoint.pt', **args)
            a, b = [torch.load(root / name / 'checkpoint.pt', weights_only=True, map_location='cpu') for name in ('direct', 'resume')]
            assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
            for key in ('sampler', 'torch', 'mixture', 'synthetic', 'dhf_layout', 'mps'):
                if a['rng'][key] is not None:
                    assert torch.equal(a['rng'][key], b['rng'][key]), key
            assert a['exposure'] == b['exposure']
            assert (root / 'direct/history.jsonl').read_bytes() == (root / 'resume/history.jsonl').read_bytes()
            if mode == 'lora':
                result.export(root / 'adapter', adapter_only=True)
                FineTunedModel.from_pretrained(root / 'adapter', device=device)
            del result
            release(device)
            loaded = FineTunedModel.from_pretrained(root / 'resume/model', device='cpu')
            x = values[128:192]
            official = TimestralForecaster.from_pretrained(root / 'resume/model').forecast(x, 32).quantiles.numpy()
            actual = PortableForecaster(loaded.model).forecast(x, 32).quantiles.numpy()
            np.testing.assert_allclose(actual, official, rtol=1e-4, atol=1e-5)
            record.update(kind='four-step staged DHF train, exact boundary resume, dropout RNG, merged/adapter reload and official export',
                exact_model_and_rng_resume=True, official_max_abs=float(np.abs(actual - official).max()),
                exposure=b['exposure'])
        else:
            model = load_model(MODEL, revision=REVISION, device=device)
            if mode == 'lora':
                apply_lora(model, 8, 16)
            sequence = generate_sequence(context + 32, 1, SyntheticConfig(), torch.Generator().manual_seed(42))[None].to(device)
            prediction = differentiable_normalized_forecast(model, sequence[..., :context], prediction_length=32)
            loss = normalized_pinball_loss(prediction, sequence[..., context:], sequence[..., :context], model.config.quantile_knots)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            parameters = [p for p in model.parameters() if p.requires_grad]
            before = [p.detach().clone() for p in parameters]
            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            torch.optim.AdamW(parameters, lr=1e-5).step()
            assert any(not torch.equal(p, old) for p, old in zip(parameters, before, strict=True))
            record.update(kind='single-origin B1/V1 forward/backward and one optimizer update', loss=float(loss.detach()), finite_gradients=True)
        synchronize(device)
    record.update(passed=True, seconds=time.perf_counter() - started, memory=memory.record())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', default='lora')
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--output', default='runs/report-round4-acceptance')
    parser.add_argument('--results', default='results/report_round4/acceptance.json')
    args = parser.parse_args()
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args.device, args.mode, args.context, root)
        write_json(root / 'result.json', result)
        print(json.dumps(result))
        return
    cases = []
    for device in ('cpu', 'mps'):
        if device == 'mps' and not torch.backends.mps.is_available():
            cases.append({'device': device, 'status': 'unavailable'})
            continue
        for context in (64, 8192, 32768):
            for mode in ('full', 'lora'):
                directory = root / f'{device}_{mode}_{context}'
                command = [sys.executable, __file__, '--worker', '--device', device, '--mode', mode,
                    '--context', str(context), '--output', str(directory)]
                if (directory / 'result.json').exists():
                    existing = json.loads((directory / 'result.json').read_text())
                    if existing.get('sources') != source_identity() or existing.get('script_sha256') != file_sha256(__file__):
                        raise ValueError(f'Acceptance source mismatch; use a fresh output: {directory}')
                    cases.append(existing)
                    continue
                directory.mkdir(exist_ok=True)
                with (directory / 'console.log').open('w') as log:
                    try:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
                        if process.returncode:
                            cases.append({'device': device, 'mode': mode, 'context': context, 'passed': False,
                                'status': 'failed', 'returncode': process.returncode, 'log': str(directory / 'console.log')})
                        else:
                            cases.append(json.loads((directory / 'result.json').read_text()))
                    except subprocess.TimeoutExpired:
                        cases.append({'device': device, 'mode': mode, 'context': context, 'passed': False, 'status': '300second timeout'})
                write_json(args.results, {'model': MODEL, 'revision': REVISION, 'cases': cases})
    record = {'model': MODEL, 'revision': REVISION, 'script_sha256': file_sha256(__file__), 'cases': cases,
        'cuda': 'unavailable; not hardware validated',
        'scope': 'short exact staged training and single-batch progressive 8192/32768 capacity only; not long-context quality or official pretraining evidence'}
    write_json(args.results, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
