import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / '.cache/upstream/src'), str(ROOT / '.cache/gift-eval/src')]
os.environ['GIFT_EVAL'] = str(ROOT / '.cache/gift-eval-data')
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import pandas as pd
import torch
import gluonts.model.evaluation as ev
from timestral_ft.compat import PortableForecaster
from timestral_ft.gift import atomic_json, load_gift_adapter, seasonal_error_sliced

torch.set_num_threads(4)
ev.seasonal_error = seasonal_error_sliced
nb = ROOT / '.cache/gift-eval/notebooks/timestral.ipynb'
ns = {}
for i in (1, 3, 4, 5):
    exec(compile(''.join(json.loads(nb.read_text())['cells'][i]['source']), f'{nb}:{i}', 'exec'), ns)
ns.update(MAX_ROWS=16, MAX_TOKENS=2**17)
os.environ['GIFT_EVAL'] = str(ROOT / '.cache/gift-eval-data')
base = 'mistralai/Timestral-209M'
rev = '9874d252ab0aae16f7f7c999f98f98d6ad7d8365'
reference = pd.read_csv(ROOT / '.cache/gift-eval/results/Timestral-209M/all_results.csv').set_index('dataset')
frozen = ROOT / 'runs/gift-eval-209m-search-v2/frozen_selection.json'
assert json.loads(frozen.read_text())['count'] == 97
records = []
for task, dataset in [('hospital/M/short', 'hospital'), ('hierarchical_sales/W/short', 'hierarchical_sales/W'), ('car_parts/M/short', 'car_parts_with_missing')]:
    for kind in ['baseline', 'lora']:
        started = time.perf_counter()
        if kind == 'baseline':
            fc = PortableForecaster.from_pretrained(base, revision=rev, device='cpu', max_context=32768)
        else:
            fc = PortableForecaster(load_gift_adapter(ROOT / 'runs/gift-eval-209m-search-v2/tasks' / task.replace('/', '__'), 'cpu'), max_context=32768)
        metrics = ns['evaluate_config'](fc, dataset, 'short')
        record = dict(task=task, kind=kind, seconds=time.perf_counter()-started, metrics=metrics, device='cpu')
        if kind == 'baseline':
            record['reference_matches'] = all(np.isclose(v, reference.loc[task, 'eval_metrics/'+m], rtol=1e-3, atol=1e-5) for m, v in metrics.items())
            assert record['reference_matches'], record
        assert all(np.isfinite(v) and v >= 0 for v in metrics.values())
        records.append(record)
        atomic_json(ROOT / '.cache/gift209/cpu_eval_profile.json', dict(records=records))
        print(json.dumps(record), flush=True)
        del fc
print('DONE', flush=True)
