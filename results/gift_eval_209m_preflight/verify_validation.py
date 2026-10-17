import sys,os,json,time
from pathlib import Path
from types import SimpleNamespace
sys.path[:0]=['src','.cache/upstream/src','.cache/gift-eval/src']
import torch,numpy as np
from gift_eval.data import Dataset
from gluonts.model import evaluate_model
from gluonts.ev.metrics import MASE,MeanWeightedSumQuantileLoss
from gluonts.time_feature import get_seasonality
from timestral_ft.compat import load_model,PortableForecaster
from timestral_ft.gift import GiftWindows,GiftTrainConfig
from timestral_ft.gift_validation import GiftValidation,GiftValidationConfig,LEVELS
from timestral_ft.lora import apply_lora
nb=Path('.cache/gift-eval/notebooks/timestral.ipynb');cells=json.loads(nb.read_text())['cells'];ns={}
for i in (1,3,4):exec(compile(''.join(cells[i]['source']),f'{nb}:{i}','exec'),ns)
os.environ['GIFT_EVAL']=str(Path('.cache/gift-eval-data').resolve());ns.update(MAX_ROWS=16,MAX_TOKENS=2**17)
torch.set_num_threads(4)
m=load_model('mistralai/Timestral-209M',revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365',device='mps');apply_lora(m);m.eval()
report=[]
for name,term in [('hospital','short'),('ett1/H','long'),('bizitobs_l2c/H','short'),('m4_yearly','short')]:
 t=time.perf_counter();ds=Dataset(name,term=term,to_univariate=False)
 w=GiftWindows(ds,256,128,42);cfg=GiftTrainConfig(context_length=256,validation_limit=128,device='mps',batch_size=2)
 val=GiftValidation('unused',GiftValidationConfig(seasonality=get_seasonality(ds.freq)))
 actual=val.measure(m,w,cfg)
 selected=set(w.validation_indices)
 inputs=[e for i,e in enumerate(ds.training_dataset) if i in selected]
 labels=[]
 for i,e in enumerate(ds.validation_dataset):
  if i in selected:labels.append(dict(e,target=e['target'][...,-ds.prediction_length:]))
 fc=PortableForecaster(m,max_context=32768)
 predictor=ns['TimestralPredictor'](fc,ds.prediction_length,bool(ds.past_feat_dynamic_real_dim))
 expected=evaluate_model(predictor,test_data=SimpleNamespace(input=inputs,label=labels),
   metrics=[MASE(),MeanWeightedSumQuantileLoss(LEVELS)],batch_size=512,axis=None,
   mask_invalid_label=True,allow_nan_forecast=False,seasonality=get_seasonality(ds.freq))
 reference=dict(mase=float(expected['MASE[0.5]'].iloc[0]),crps=float(expected['mean_weighted_sum_quantile_loss'].iloc[0]))
 matches=all(np.isclose(actual[k],reference[k],rtol=1e-5,atol=1e-6) for k in reference)
 r=dict(dataset=name,term=term,selected_series=len(inputs),actual=actual,official_validation=reference,matches=matches,seconds=time.perf_counter()-t)
 report.append(r);print(json.dumps(r),flush=True)
 Path('.cache/gift209/validation_parity.json').write_text(json.dumps(report,indent=2)+'\n')
 if not matches:raise ValueError('Official validation mismatch')
 torch.mps.empty_cache()
