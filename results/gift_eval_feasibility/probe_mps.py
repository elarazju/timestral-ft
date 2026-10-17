import sys,os,json,time,warnings
from pathlib import Path
sys.path[:0]=[str(Path('src').resolve()),str(Path('.cache/upstream/src').resolve()),str(Path('.cache/gift-eval/src').resolve())]
import torch,numpy as np
from timestral_ft.compat import PortableForecaster
warnings.filterwarnings('ignore',category=FutureWarning)
torch.set_num_threads(4)
nb=json.loads(Path('.cache/gift-eval/notebooks/timestral.ipynb').read_text());ns={}
for i in [1,3,4,5]:exec(''.join(nb['cells'][i]['source']),ns)
os.environ['GIFT_EVAL']=str(Path('.cache/gift-eval-data').resolve());ns['MAX_ROWS']=128;ns['MAX_TOKENS']=2**22
fc=PortableForecaster.from_pretrained('mistralai/Timestral-5M',revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f',device='mps')
p=Path('results/gift_eval_feasibility/probe.json');r=json.loads(p.read_text());r['mps_device_checks']=[]
for name,term,key in [('hospital','short','hospital/M/short'),('ett1/H','long','ett1/H/long')]:
 start=time.perf_counter();metrics=ns['evaluate_config'](fc,name,term);elapsed=time.perf_counter()-start
 reference=next(x['metrics'] for x in r['baseline_replications'] if x['task']==key)
 matched=all(np.isclose(metrics[k],reference[k],rtol=1e-3,atol=1e-5) for k in metrics)
 row={'task':key,'device':'mps','seconds':elapsed,'metrics':metrics,'matches_cpu_all_11_metrics':bool(matched),'tolerance':{'rtol':1e-3,'atol':1e-5}}
 r['mps_device_checks'].append(row);print(key,elapsed,matched,flush=True)
p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
