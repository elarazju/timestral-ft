import sys,os,json,time,warnings,itertools
from pathlib import Path
sys.path[:0]=[str(Path('src').resolve()),str(Path('.cache/upstream/src').resolve()),str(Path('.cache/gift-eval/src').resolve())]
import numpy as np
import pandas as pd
import torch
import pyarrow.compute as pc
from gift_eval.data import Dataset
from timestral_ft.compat import PortableForecaster
from timestral_ft import DataConfig,TrainConfig,finetune
warnings.filterwarnings('ignore',category=FutureWarning)
torch.set_num_threads(4)
nb=json.loads(Path('.cache/gift-eval/notebooks/timestral.ipynb').read_text())
ns={}
for i in [1,3,4,5]:exec(''.join(nb['cells'][i]['source']),ns)
os.environ['GIFT_EVAL']=str(Path('.cache/gift-eval-data').resolve())
ns['MAX_ROWS']=128
ns['MAX_TOKENS']=2**22
fc=PortableForecaster.from_pretrained('mistralai/Timestral-5M',revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f',device='cpu')
reference=pd.read_csv('.cache/gift-eval/results/Timestral-5M/all_results.csv').set_index('dataset')
result={'gift_repo_revision':'8e6da595ac943be01e0848a02b9157af84731284','gift_data_revision':'30841734ac5cfddbd0c3bad6d09d2b6b32becbb0','base_revision':'bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f','scope':'Feasibility only: four full task baseline replications and four-step LoRA on 32 hospital train series; not a full leaderboard submission','software':{'numpy':np.__version__,'pandas':pd.__version__,'torch':str(torch.__version__)},'dataset_checks':[],'baseline_replications':[]}
for name in ['hospital','ett1/H','m4_yearly','bizitobs_l2c/H']:
 ds=Dataset(name)
 a=next(iter(ds.training_dataset));b=next(iter(ds.validation_dataset))
 raw=ds.hf_dataset.data.column('target')
 if ds.target_dim>1:raw=pc.list_flatten(pc.list_slice(raw,0,1))
 lengths=pc.list_value_length(raw).to_numpy()
 train_lengths=lengths-ds.prediction_length*(ds.windows+1)
 row={'name':name,'series':ds.hf_dataset.num_rows,'target_dim':ds.target_dim,'past_covariates':ds.past_feat_dynamic_real_dim,'horizon':ds.prediction_length,'test_windows':ds.windows,'min_train_length':int(train_lengths.min()),'max_train_length':int(train_lengths.max()),'series_eligible_with_L256':int((train_lengths>=256+ds.prediction_length).sum()),'first_train_length':a['target'].shape[-1],'first_validation_prefix_length':b['target'].shape[-1]}
 result['dataset_checks'].append(row);print('DATA',json.dumps(row),flush=True)
for name,term,key in [('hospital','short','hospital/M/short'),('ett1/H','short','ett1/H/short'),('ett1/H','long','ett1/H/long'),('bizitobs_l2c/H','short','bizitobs_l2c/H/short')]:
 start=time.perf_counter();metrics=ns['evaluate_config'](fc,name,term);seconds=time.perf_counter()-start
 diffs={m:float(metrics[m]-reference.loc[key,'eval_metrics/'+m]) for m in metrics}
 matches=all(np.isclose(metrics[m],reference.loc[key,'eval_metrics/'+m],rtol=1e-3,atol=1e-5) for m in metrics)
 row={'task':key,'device':'cpu','seconds':seconds,'metrics':metrics,'official_difference':diffs,'all_11_metrics_match':bool(matches),'comparison_tolerance':{'rtol':1e-3,'atol':1e-5}}
 result['baseline_replications'].append(row);print('BASE',key,seconds,matches,metrics,flush=True)
# Use the official train prefix and only the validation suffix as validation labels.
ds=Dataset('hospital');train_parts=[];val_parts=[]
for a,b in itertools.islice(zip(ds.training_dataset,ds.validation_dataset),32):
 assert a['item_id']==b['item_id']
 target=np.asarray(b['target']);dates=pd.period_range(start=b['start'],periods=len(target),freq=b['start'].freq).to_timestamp()
 frame=pd.DataFrame({'timestamp':dates,'series_id':str(b['item_id']),'target':target})
 train_parts.append(frame.iloc[:len(a['target'])]);val_parts.append(frame.iloc[len(a['target']):])
start=time.perf_counter()
artifact=finetune(pd.concat(train_parts,ignore_index=True),pd.concat(val_parts,ignore_index=True),model_id='mistralai/Timestral-5M',revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f',data_config=DataConfig(id_column='series_id',frequency='MS'),train_config=TrainConfig(mode='lora',context_length=24,prediction_length=12,max_steps=4,warmup_steps=1,eval_every=2,batch_size=4,gradient_accumulation=1,num_threads=4),output_dir='.cache/gift-probe-run/hospital_lora')
result['lora_smoke']={'training_series':32,'steps':4,'context':24,'horizon':12,'seconds':time.perf_counter()-start,'finite_validation_loss':artifact.metadata['best_validation_loss'],'best_step':artifact.metadata['best_step'],'trainable_parameters':artifact.metadata['trainable_parameters']}
# Native table API's context must be set appropriately for these short monthly sequences.
ft_fc=PortableForecaster(artifact.model,max_context=32768)
result['lora_smoke']['full_hospital_test_metrics']=ns['evaluate_config'](ft_fc,'hospital','short')
print('LORA',result['lora_smoke'],flush=True)
out=Path('results/gift_eval_feasibility');out.mkdir(parents=True,exist_ok=True)
(out/'probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
