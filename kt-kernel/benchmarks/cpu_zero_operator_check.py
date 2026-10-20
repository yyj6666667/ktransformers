"""Validate actual KT forward on retained full prompt expert fixtures."""
import argparse
import json
from pathlib import Path
import time
import torch
from cpu_zero_operator import ExactOperator, make_cpu
from kt_to_golden_ablation import load_weights
from merge_replay_common import dump, sha
import kt_kernel
p=argparse.ArgumentParser()
p.add_argument("--fixtures",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--layers",default="0")
p.add_argument("--pools",type=int,default=2)
p.add_argument("--threads",type=int,default=48)
p.add_argument("--mapping",action="store_true")
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.manual_seed(42)
cpu=make_cpu(a.pools,a.threads)
results=[]
for layer in map(int,a.layers.split(",")):
    start=time.time();w=load_weights(layer)
    gu=w["experts.gate_up_proj"];down=w["experts.down_proj"]
    mapping=torch.randperm(len(gu)) if a.mapping else None
    inverse=torch.argsort(mapping) if mapping is not None else None
    op=ExactOperator(gu[:,:512],gu[:,512:],down,cpu=cpu,mapping=mapping)
    for path in sorted(a.fixtures.glob("*.pt")):
        f=torch.load(path,map_location="cpu",weights_only=False)[layer]
        ids=f["ids"] if inverse is None else inverse[f["ids"]]
        op.prepare(f["input"],ids,f["route"])
        out=op.forward().clone();ref=f["output"]
        d=(out.float()-ref.float()).abs()
        row={"layer":layer,"case":path.name,"abs_mean":d.mean().item(),"different":int((out!=ref).sum()),"elements":out.numel()}
        results.append(row);print(json.dumps(row),flush=True)
        if row["different"]:
            torch.save({"candidate":out,"golden":ref,"fixture":f},a.output/"mismatch.pt")
            raise AssertionError(row)
    del op,w,gu,down
    print("LAYER",layer,"seconds",round(time.time()-start,2),flush=True)
dump(a.output/"results.json",{"results":results,"extension_sha256":sha(kt_kernel.kt_kernel_ext.__file__),"pools":a.pools,"threads":a.threads,"permuted_mapping":a.mapping})
