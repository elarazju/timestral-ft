import fcntl,sys,time,json
from pathlib import Path
sys.path[:0]=['src','.cache/upstream/src']
import torch
from timestral_ft.compat import PortableForecaster
torch.set_num_threads(4);torch.manual_seed(42);torch.mps.manual_seed(42)
lock=Path('runs/gift-eval-209m-search-v2/.accelerator.lock').open('a')
print('Waiting for accelerator between task jobs',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
print('Starting synthetic full-history inference preflight',flush=True)
t=time.perf_counter();fc=PortableForecaster.from_pretrained('mistralai/Timestral-209M',revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365',device='mps',max_context=32768)
x=torch.randn(1,7,32768,device='mps');cov=torch.randn(1,35,32768,device='mps')
for i in range(2):
 t=time.perf_counter();p=fc.forecast(x,900,past_covariates=cov).quantiles;torch.mps.synchronize()
 record=dict(iteration=i,context=32768,targets=7,past_covariates=35,horizon=900,seconds=time.perf_counter()-t,finite=bool(p.isfinite().all()),driver_bytes=torch.mps.driver_allocated_memory())
 print(json.dumps(record),flush=True)
 Path('.cache/gift209/full_inference_profile.json').write_text(json.dumps(record,indent=2)+'\n')
 assert record['finite']
