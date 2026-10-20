"""Identify a CPU arithmetic model of one measured BF16 MMA instruction."""
import argparse
from pathlib import Path
import json
import itertools

import torch
from merge_replay_common import configure_torch, dump


def round_float(value, mode):
    rounded = value.float()
    if mode == "rz":
        rounded = torch.where(rounded.double().abs() > value.abs(),
                              torch.nextafter(rounded, torch.zeros_like(rounded)), rounded)
    return rounded.double()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    products, carries, targets, unnormalized_exponents = [], [], [], []
    for component in ("gate_up", "down"):
        v = torch.load(args.reference / f"{component}-m2.pt", weights_only=False)
        x, w = v["input"].double(), v["weight"][::17].double()
        m, k = x.shape
        n = len(w)
        product = x.reshape(m, k//16, 16).permute(1, 0, 2)[:, :, None, :] * w.reshape(n, k//16, 16).permute(1, 0, 2)[:, None, :, :]
        ae = torch.frexp(x.abs())[1].reshape(m, k//16, 16).permute(1, 0, 2)[:, :, None, :]
        be = torch.frexp(w.abs())[1].reshape(n, k//16, 16).permute(1, 0, 2)[:, None, :, :]
        unnormalized_exponents.append((ae + be - 1).reshape(-1, 16))
        target = v["steps"][:, :, ::17].double()
        carry = torch.cat((torch.zeros_like(target[:1]), target[:-1]))
        products.append(product.reshape(-1, 16))
        carries.append(carry.flatten())
        targets.append(target.flatten())
    product, carry, target = torch.cat(products), torch.cat(carries), torch.cat(targets)
    unnormalized = torch.cat(unnormalized_exponents)
    results = []
    for size, pattern, mode in itertools.product((1, 2, 4, 8, 16), ("contiguous", "strided"), ("rn", "rz")):
        order = torch.arange(16).reshape(-1, size)
        if pattern == "strided":
            order = torch.arange(16).reshape(size, -1).T
        acc = carry.clone()
        for indices in order:
            acc = round_float(acc + product[:, indices].sum(-1), mode)
        results.append({"model": "exact_chunk", "size": size, "pattern": pattern, "round": mode,
                        "different": int((acc != target).sum()), "elements": len(target)})
    for size, pattern, bits, align, mode in itertools.product((4, 8, 16), ("contiguous", "strided"),
                                                              range(24, 30), ("trunc", "floor", "round"), ("rn", "rz")):
        order = torch.arange(16).reshape(-1, size)
        if pattern == "strided":
            order = torch.arange(16).reshape(size, -1).T
        acc = carry.clone()
        for indices in order:
            terms = torch.cat((acc[:, None], product[:, indices]), -1)
            exponent = torch.frexp(terms.abs().amax(-1))[1]
            quantum = torch.ldexp(torch.ones_like(acc), exponent - bits)
            aligned = getattr(torch, align)(terms / quantum[:, None])
            acc = round_float(aligned.sum(-1) * quantum, mode)
        results.append({"model": "shared_exponent", "size": size, "pattern": pattern,
                        "bits": bits, "align": align, "round": mode,
                        "different": int((acc != target).sum()), "elements": len(target)})
    terms = torch.cat((carry[:, None], product), -1)
    carry_exp = torch.where(carry != 0, torch.frexp(carry.abs())[1], -1024)
    debug = []
    for offset, bits, mode in itertools.product((-1, 0, 1, 2), range(24, 29), ("rn", "rz")):
        product_exp = torch.where(product != 0, unnormalized + offset, -1024).amax(-1)
        exponent = torch.maximum(carry_exp, product_exp)
        quantum = torch.ldexp(torch.ones_like(carry), exponent - bits)
        acc = round_float(torch.trunc(terms / quantum[:, None]).sum(-1) * quantum, mode)
        different = acc != target
        row = {"model": "unnormalized_exponent", "offset": offset, "bits": bits, "round": mode,
               "different": int(different.sum()), "elements": len(target)}
        results.append(row)
        if int(different.sum()) <= 10:
            debug.append({**row, "mismatches": [
                {"index": i, "carry": carry[i].item(), "products": product[i].tolist(),
                 "cpu": acc[i].item(), "gpu": target[i].item(), "exponent": exponent[i].item()}
                for i in different.nonzero().flatten().tolist()]})
    dump(args.output / "debug.json", debug)
    results.sort(key=lambda r: r["different"])
    dump(args.output / "results.json", results)
    print(json.dumps(results[:20], indent=2), flush=True)


if __name__ == "__main__":
    main()
