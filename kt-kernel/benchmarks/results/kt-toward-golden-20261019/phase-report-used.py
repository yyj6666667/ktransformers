from pathlib import Path
import torch,json
root=Path('/dev/shm/yyj-qwen35-35b-wbw-20261019')
exp=root/'fp32-down-merge/toward-golden'
source_map=json.loads((exp/'screen-combined/sources.json').read_text())
torch.set_num_threads(2)
report={}
routing={}
for stage in ['06_expert_id_order','09_cuda_gate_up']:
 route_stats={str(layer):dict(changed_order=0,changed_set=0,positions=0) for layer in range(40)}
 sums={str(layer):{'prefill':[0.,0],'decode':[0.,0]} for layer in range(40)}
 for f in sorted(Path(source_map[stage]).glob('*.cached.r0.pt')):
  a=torch.load(f,map_location='cpu',weights_only=False)
  b=torch.load(root/'outputs/native-step1-001/generated_cached'/f.name,map_location='cpu',weights_only=False)
  for layer in range(40):
   key=f'layer.{layer}.router_ids'
   ai=a['hidden'][key];bi=b['hidden'][key][:len(a['response'])]
   route_stats[str(layer)]['changed_order']+=int((ai!=bi).any(-1).sum())
   route_stats[str(layer)]['changed_set']+=int((ai.sort(-1).values!=bi.sort(-1).values).any(-1).sum())
   route_stats[str(layer)]['positions']+=len(ai)
   key=f'layer.{layer}.boundary'
   d=(a['hidden'][key].double()-b['hidden'][key][:len(a['response'])].double()).abs()
   for name,v in [('prefill',d[:1]),('decode',d[1:])]:
    sums[str(layer)][name][0]+=v.sum().item();sums[str(layer)][name][1]+=v.numel()
 routing[stage]=route_stats
 report[stage]={key:{name:s/n for name,(s,n) in value.items()} for key,value in sums.items()}
(exp/'routing-diagnostic.json').write_text(json.dumps(routing,indent=2)+'\n')
(exp/'phase-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({stage:{k:value[k] for k in ['0','1','2','10','39']} for stage,value in report.items()},indent=2))

print('ROUTER',json.dumps({s:{k:v for k,v in rows.items() if v['changed_set']} for s,rows in routing.items()}))
