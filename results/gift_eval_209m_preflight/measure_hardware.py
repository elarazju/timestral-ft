import sys,time,json,gc
from pathlib import Path
sys.path[:0]=['src','.cache/upstream/src']
import torch
from timestral_ft.compat import load_model,differentiable_forecast,pinball_loss
from timestral_ft.lora import apply_lora
out=[]
for device in ['mps','cpu']:
 torch.set_num_threads(4)
 m=load_model('mistralai/Timestral-209M',revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365',device='cpu')
 apply_lora(m);m.to(device);m.transformer.rope.rebuild_inv_freq()
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
 for context,v,h in [(256,1,48),(1024,1,48),(4096,1,48),(1024,7,720)]:
  x=torch.randn(1,v,context,device=device); y=torch.randn(1,v,h,device=device)
  ts=[]
  for i in range(3):
   t=time.perf_counter();opt.zero_grad(set_to_none=True)
   p=differentiable_forecast(m,x,h);loss=pinball_loss(p,y,x,m.config.quantile_knots);loss.backward();opt.step()
   if device=='mps':torch.mps.synchronize()
   ts.append(time.perf_counter()-t)
  r=dict(device=device,context=context,variates=v,horizon=h,batch=1,seconds=ts,loss=float(loss.detach()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad))
  if device=='mps':r.update(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory())
  print(json.dumps(r),flush=True);out.append(r)
  Path('.cache/gift209/hardware_profile.json').write_text(json.dumps(out,indent=2)+'\n')
 del m,opt,p,loss;gc.collect()
 if device=='mps':torch.mps.empty_cache()
