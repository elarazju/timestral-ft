import sys,time,json
from pathlib import Path
sys.path[:0]=['src','.cache/upstream/src']
import torch
from timestral_ft.compat import load_model,differentiable_forecast,pinball_loss
from timestral_ft.lora import apply_lora
from timestral_ft.trainer import clip_gradients
torch.set_num_threads(4);torch.manual_seed(42);torch.mps.manual_seed(42)
m=load_model('mistralai/Timestral-209M',revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365',device='cpu')
apply_lora(m);m.to('mps');m.transformer.rope.rebuild_inv_freq();m.train()
params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-4)
x=torch.randn(1,7,1536,device='mps');cov=torch.randn(1,35,1536,device='mps');y=torch.randn(1,7,900,device='mps')
records=[]
for i in range(3):
 start=time.perf_counter();opt.zero_grad(set_to_none=True)
 p=differentiable_forecast(m,x,900,past_covariates=cov)
 loss=pinball_loss(p,y,x,m.config.quantile_knots);loss.backward();norm=clip_gradients(params,1.);opt.step();torch.mps.synchronize()
 r=dict(iteration=i,seconds=time.perf_counter()-start,loss=float(loss.detach()),gradient_norm=norm,
        allocated_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
 records.append(r);print(json.dumps(r),flush=True)
 assert torch.isfinite(loss) and all(bool(p.isfinite().all()) for p in params)
Path('.cache/gift209/bounded_shape.json').write_text(json.dumps(dict(passed=True,batch_size=1,targets=7,past_covariates=35,context=1536,horizon=900,records=records),indent=2)+'\n')
