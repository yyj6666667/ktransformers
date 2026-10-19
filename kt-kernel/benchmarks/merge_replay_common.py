import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(os.environ['KT_MERGE_FIXTURE_ROOT'])
MODEL=ROOT/'model/Qwen3.5-35B-A3B'


def dump(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 def encode(value):
  if isinstance(value,(set,frozenset)):return sorted(value)
  if isinstance(value,Path):return str(value)
  raise TypeError(f'Cannot serialize {type(value).__name__}')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=encode)+'\n')
 temp.replace(path)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def cases():
 return json.loads((ROOT/'data/cases.json').read_text())


def configure_torch():
 import torch
 random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 if os.environ.get('BASELINE_REQUIRE_CACHED_AUTOTUNE')=='1':
  from triton.runtime.autotuner import Autotuner
  def reject_benchmark(self,*args,**kwargs):
   raise RuntimeError(f'Frozen baseline refuses uncached autotuning: {self.base_fn.__name__}')
  Autotuner._bench=reject_benchmark


def autotune_manifest():
 from triton.runtime.autotuner import Autotuner
 seen=set();entries=[]
 for module_name,module in sorted(list(sys.modules.items())):
  if module is None or not module_name.startswith(('fla.','sglang.')):continue
  for attr,value in sorted(vars(module).items()):
   candidate=value
   for _ in range(5):
    if isinstance(candidate,Autotuner):
     if id(candidate) not in seen and candidate.cache:
      seen.add(id(candidate))
      entries.append({'module':module_name,'attribute':attr,'function':candidate.base_fn.__name__,
       'cache_results':candidate.cache_results,'choices':[
         {'key':list(key),'config':{k:v for k,v in vars(config).items() if k!='pre_hook'}}
         for key,config in candidate.cache.items()]})
     break
    if not isinstance(candidate,type) and hasattr(candidate,'fn'):
     candidate=candidate.fn
    else:break
 return entries


def clone_cpu(tensor):
 return tensor.detach().to('cpu',copy=True).contiguous()


def record_runtime(out,extra):
 import torch,transformers
 data={'time':time.time(),'python':sys.version,'prefix':sys.prefix,
       'torch':torch.__version__,'transformers':transformers.__version__,
       'cudnn':torch.backends.cudnn.version(),'model':str(MODEL),
       'model_manifest_sha256':sha(ROOT/'model/verified.json'),
       'config_sha256':sha(MODEL/'config.json'),'index_sha256':sha(MODEL/'model.safetensors.index.json'),
       'cases_sha256':sha(ROOT/'data/cases.json'),'lora':False,'optimizer_steps':0,
       'seed':42,'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),
       'scripts':{p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},
       'numerical_flags':{'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
           'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
           'cudnn_benchmark':torch.backends.cudnn.benchmark,
           'require_cached_autotune':os.environ.get('BASELINE_REQUIRE_CACHED_AUTOTUNE')=='1'},
       'extra':extra}
 dump(Path(out)/'runtime.json',data)


class Capture:
 """Copy outputs without changing the forward values; retain response positions."""
 def __init__(self):self.active=False;self.data={};self.tokens=0;self.start=0;self.count=0
 def begin(self,tokens,start,count):
  self.data={};self.tokens=tokens;self.start=start;self.count=count;self.active=True
 def end(self):self.active=False;return self.data
 def save(self,name,value):
  import torch
  if not self.active or not isinstance(value,torch.Tensor):return
  assert value.numel()%self.tokens==0,(name,value.shape,self.tokens)
  rows=value.detach().reshape(self.tokens,-1)
  self.data[name]=clone_cpu(rows[self.start:self.start+self.count])
 def hook(self,name):
  def save(module,args,value):
   self.save(name,value[0] if isinstance(value,tuple) else value)
  return save
 def install_hf(self,text_model):
  text_model.embed_tokens.register_forward_hook(self.hook('embedding'))
  text_model.norm.register_forward_hook(self.hook('final_norm'))
  for i,layer in enumerate(text_model.layers):
   layer.register_forward_hook(self.hook(f'layer.{i}.boundary'))
   for field in ['input_layernorm','post_attention_layernorm','linear_attn','self_attn','mlp']:
    if hasattr(layer,field):getattr(layer,field).register_forward_hook(self.hook(f'layer.{i}.{field}'))
   if i<4 and hasattr(layer,'linear_attn'):
    for field in ['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj']:
     getattr(layer.linear_attn,field).register_forward_hook(self.hook(f'layer.{i}.linear_attn.{field}'))
   mlp=layer.mlp
   if hasattr(mlp,'_compute_routing'):
    original=mlp._compute_routing
    def route(hidden_states,_original=original,_i=i):
     ids,weights=_original(hidden_states)
     self.save(f'layer.{_i}.router_ids',ids)
     self.save(f'layer.{_i}.router_weights',weights)
     return ids,weights
    mlp._compute_routing=route
   elif hasattr(mlp,'gate'):
    def route_hook(module,args,value,_i=i):
     if isinstance(value,tuple) and len(value)>=3:
      self.save(f'layer.{_i}.router_logits',value[0])
      self.save(f'layer.{_i}.router_weights',value[1])
      self.save(f'layer.{_i}.router_ids',value[2])
     else:self.save(f'layer.{_i}.router_logits',value)
    mlp.gate.register_forward_hook(route_hook)
