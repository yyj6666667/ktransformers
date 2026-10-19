"""Audit identical inputs/build flags, immutable baselines and layout-only replay parity."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--run", default="compact-final")
    parser.add_argument("--variant", default="after-v3")
    args = parser.parse_args()
    exp, root = args.experiment, args.fixtures
    final = exp / args.run
    out = final / "analysis"
    assert (final / "complete.json").exists()
    torch.set_num_threads(2)
    raw = [json.loads(p.read_text()) for p in sorted((final / "performance").glob("*.json"))]
    assert len(raw) == 8 and all(not r["profiling"] for r in raw)
    for family in ("sft", "inference"):
        selected = [r for r in raw if r["family"] == family]
        assert len(selected) == 4
        assert len({tuple(r["weight_hashes"]) for r in selected}) == 1
        for index, tokens in enumerate((1, 4, 128, 512)):
            assert all(r["results"][index]["tokens"] == tokens for r in selected)
            assert len({tuple(r["results"][index]["input_hashes"]) for r in selected}) == 1
            groups = {tag: [r["results"][index] for r in selected if r["tag"].endswith(tag)]
                      for tag in ("before", "after")}
            assert all(len(v) == 2 for v in groups.values())
            # Storage-layout tuning must preserve the original FP32-merge result.
            direct = [json.loads(p.read_text()) for p in
                      (exp / "performance").glob(f"perf-{family}-*-after.json")]
            assert len(direct) == 2
            assert len({r["results"][index]["output_sha256"] for r in direct}
                       | {r["output_sha256"] for r in groups["after"]}) == 1

    different = elements = 0
    records = []
    for old in sorted((exp / "replay-after/reference_cached").glob("*.cached.r0.pt")):
        new = final / "replay-after/reference_cached" / old.name
        a = torch.load(old, map_location="cpu", weights_only=False)
        b = torch.load(new, map_location="cpu", weights_only=False)
        assert a["prompt"] == b["prompt"] and a["response"] == b["response"]
        assert a["hidden"].keys() == b["hidden"].keys()
        for x, y in [(a["logits"], b["logits"])] + [(a["hidden"][k], b["hidden"][k]) for k in a["hidden"]]:
            different += int((x != y).sum())
            elements += x.numel()
        records.append(dict(path=str(new), bytes=new.stat().st_size, sha256=sha(new)))
    assert len(records) == 8 and different == 0, (len(records), different, elements)
    protected = json.loads((root / "analysis/root-cause-20261019/unchanged-runtime-audit.json").read_text())
    actual = {name: sha(root / name) for name in protected}
    assert actual == protected, "a protected baseline file changed"
    sources = ["operators/amx/" + name for name in
               ("bf16-moe.hpp", "la/amx_raw_buffers.hpp", "moe_base.hpp", "sft_moe.hpp")]
    source_hashes = {name: sha(exp / f"source-{args.variant}/kt-kernel" / name) for name in sources}
    cache_fields = ("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS:", "CMAKE_CXX_FLAGS_RELEASE:",
                    "CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_FLAGS:", "CMAKE_CUDA_ARCHITECTURES:",
                    "CMAKE_BUILD_TYPE:", "CPUINFER_ENABLE_LTO:", "LLAMA_", "KTRANSFORMERS_")
    flags = {}
    for variant in ("before", args.variant):
        lines = (exp / f"build-{variant}/CMakeCache.txt").read_text().splitlines()
        flags[variant] = [line for line in lines if line.startswith(cache_fields)]
    assert flags["before"] == flags[args.variant]
    audit = dict(optimized_vs_direct_tiled=dict(different_elements=different, elements=elements),
                 operator_layout_only_output_hashes_equal=True, build_options_identical=True,
                 build_options=flags["before"], protected_baseline_hashes=actual,
                 final_source_sha256=source_hashes, final_raw_records=records,
                 compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                 filesystem=subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "--target", str(root)], text=True).strip())
    (out / "validation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in
                      ("build_options", "final_raw_records", "protected_baseline_hashes", "final_source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
