from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess

root=Path('/dev/shm/yyj-qwen35-35b-wbw-20261019')
exp=root/'fp32-down-merge'
final=exp/'compact-final'
final.mkdir(exist_ok=True)
(final/'logs').mkdir(exist_ok=True)
python=root/'actor-kt/venv/bin/python'
bench=exp/'bench_bf16_merge.py'
scripts=exp/'source-after-v3/kt-kernel/benchmarks'
history=[]

metadata=json.loads((exp/'builds.json').read_text())
for variant in ('after-v3',):
    assert (exp/f'build-{variant}/complete.txt').exists(),variant
    target=exp/'overlays'/variant/'kt_kernel'
    shutil.copytree(exp/'overlays/after/kt_kernel', target,
                    ignore=shutil.ignore_patterns('*.so','__pycache__'),dirs_exist_ok=True)
    binary,=(exp/f'build-{variant}').glob('kt_kernel_ext.*.so')
    output=target/binary.name.replace('kt_kernel_ext.','_kt_kernel_ext_avx512_bf16.')
    shutil.copy2(binary,output)
    metadata[variant]=dict(extension=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
(exp/'builds.json').write_text(json.dumps(metadata,indent=2)+'\n')
link=final/'replay-before'
if not link.exists():link.symlink_to(exp/'replay-before',target_is_directory=True)

def run(name,variant,argv):
    env=os.environ.copy()
    env.update(LD_LIBRARY_PATH='/mnt/runner_yyj/_work/_tool/Python/3.12.13/x64/lib:/usr/local/cuda-12.8/lib64',
               PYTHONPATH=str(exp/'overlays'/variant)+':'+str(root/'scripts'),KT_KERNEL_CPU_VARIANT='avx512_bf16',
               KT_MERGE_FIXTURE_ROOT=str(root),PYTHONHASHSEED='42',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               CUDA_VISIBLE_DEVICES='0,1,2,3',KT_SFT_PROFILE='0')
    row=dict(name=name,variant=variant,argv=[str(x) for x in argv],status='running',
             started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    history.append(row)
    (final/'status.json').write_text(json.dumps(history,indent=2)+'\n')
    print('START',name,row['started'],flush=True)
    with (final/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(row['argv'],env=env,stdout=log,stderr=subprocess.STDOUT)
    row.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,
               finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (final/'status.json').write_text(json.dumps(history,indent=2)+'\n')
    print('FINISH',name,row['status'],flush=True)
    if result.returncode:raise RuntimeError(name)

run('numerics-final','after-v3',[python,'-m','pytest','-q',exp/'source-after-v3/kt-kernel/test/per_commit/test_bf16_fp32_merge.py'])
order=('before','after-v3','after-v3','before')
for family in ('sft','inference'):
    for index,variant in enumerate(order):
        tag='after' if variant=='after-v3' else ('control' if variant=='bf16-parallel-control' else 'before')
        name=f'perf-{family}-{index}-{tag}'
        folder='schedule-control' if tag=='control' else 'performance'
        run(name,variant,[python,bench,'--family',family,'--tag',name,'--output',final/folder/f'{name}.json'])
run('replay-final','after-v3',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
    scripts/'replay_bf16_merge.py','--backend','kt','--max-new-tokens','32',
    '--reference',root/'outputs/native-step1-001/generated-cases.json','--output',final/'replay-after'])
run('report','after-v3',[python,exp/'report_bf16_merge.py','--experiment',final,
    '--golden',root/'outputs/native-step1-001/generated_cached',
    '--original-kt',root/'outputs/kt-step1-002/reference_cached','--output',final/'analysis'])
(final/'complete.json').write_text(json.dumps(dict(status='complete',runs=len(history)),indent=2)+'\n')
