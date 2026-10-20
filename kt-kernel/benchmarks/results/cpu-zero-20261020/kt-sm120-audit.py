import hashlib
import json
from pathlib import Path
import torch
R = Path('/dev/shm/yyj-qwen35-35b-wbw-20261019')
E = R / 'fp32-down-merge'
Z = E / 'cpu-zero-20261020'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()
old = json.loads((Z / 'scripts/validation-audit.json').read_text())
protected = {}
for relative, expected in old['protected_runtime_hashes_unchanged'].items():
    actual = sha(R / relative)
    assert actual == expected, relative
    protected[relative] = actual
golden = R / 'outputs/native-step1-001/generated_cached'
for row in old['golden_raw_record_hashes_unchanged']:
    assert sha(golden / row['file']) == row['sha256'], row
for key, relative in [('data_cases_sha256', 'data/cases.json'),
                      ('generated_prefixes_sha256', 'outputs/native-step1-001/generated-cases.json'),
                      ('model_manifest_sha256', 'model/verified.json')]:
    assert sha(R / relative) == old[key], key
torch.set_num_threads(2)
raw, elements, bit_different, nonfinite = [], 0, 0, 0
for p in sorted((Z / 'full-kt-v5-32/12_kt_sm120').glob('*.pt')):
    value = torch.load(p, map_location='cpu', weights_only=False)
    target = torch.load(golden / p.name, map_location='cpu', weights_only=False)
    pairs = [(value['logits'], target['logits'])]
    pairs += [(value['hidden'][k], target['hidden'][k]) for k in value['hidden'].keys() & target['hidden'].keys()]
    for a, b in pairs:
        assert a.dtype == b.dtype and a.shape == b.shape
        ab = a.contiguous().view(torch.uint8).reshape(-1, a.element_size())
        bb = b.contiguous().view(torch.uint8).reshape(-1, b.element_size())
        bit_different += int((ab != bb).any(-1).sum())
        elements += a.numel()
        if a.is_floating_point(): nonfinite += int((~torch.isfinite(a)).sum())
    raw.append({'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
assert len(raw) == 8 and bit_different == 0 and nonfinite == 0, (len(raw), bit_different, nonfinite)
source = E / 'source-sm120-v5/kt-kernel'
files = ['ext_bindings.cpp', 'operators/avx512/sm120_bf16_math.hpp', 'operators/avx512/sm120_bf16_moe.hpp']
result = {'protected_runtime_hashes_unchanged': protected,
          'golden_raw_record_hashes_unchanged': old['golden_raw_record_hashes_unchanged'],
          'data_cases_sha256': old['data_cases_sha256'], 'generated_prefixes_sha256': old['generated_prefixes_sha256'],
          'model_manifest_sha256': old['model_manifest_sha256'], 'candidate_raw_records': raw,
          'full_model_bit_comparison': {'elements': elements, 'different': bit_different, 'nonfinite': nonfinite},
          'compiled_sources': {p: sha(source/p) for p in files},
          'extension_sha256': sha(E / 'overlays/sm120-v5/kt_kernel/_kt_kernel_ext_avx512_bf16.cpython-312-x86_64-linux-gnu.so'),
          'no_lora': True, 'candidate_routed_expert_compute': 'CPU only',
          'scope': 'forward at fixed common prefixes, own routing and caches; no backward or RL quality claim'}
(Z / 'validation-audit-final.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['full_model_bit_comparison']), flush=True)
