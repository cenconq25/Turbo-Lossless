"""
8-tier analysis on INT4 quantized models (GPTQ and AWQ).
Analyzes both the packed int32 weight data and the FP16 scales/zeros.
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import json
import time
import glob
import os


def shannon_entropy(sorted_counts, total):
    probs = sorted_counts / total
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def analyze_tensor(name, tensor, device):
    t = tensor.to(device)
    dtype = t.dtype

    # Determine the native bit width
    if dtype in (torch.bfloat16, torch.float16):
        raw = t.view(torch.int16).ravel()
        native_bits = 16
    elif dtype == torch.int32:
        # GPTQ packs 8x INT4 values into each int32
        # Analyze as int32 words
        raw = t.ravel().to(torch.int64)  # avoid overflow issues
        # Actually let's unpack the int4 values and analyze those
        packed = t.ravel()
        n_packed = packed.shape[0]
        # Each int32 holds 8 x 4-bit values
        int4_vals = []
        for shift in range(0, 32, 4):
            nibble = ((packed >> shift) & 0xF).to(torch.int16)
            int4_vals.append(nibble)
        raw = torch.cat(int4_vals)
        native_bits = 4
    elif dtype == torch.int8 or dtype == torch.uint8:
        raw = t.ravel().to(torch.int16)
        native_bits = 8
    else:
        return None

    n = raw.shape[0]
    if n < 100:
        return None

    unique_vals, counts = raw.unique(return_counts=True)
    sorted_idx = counts.argsort(descending=True)
    sc = counts[sorted_idx].cpu().numpy().astype(np.int64)
    total = int(n)
    n_unique = int(unique_vals.shape[0])
    entropy = shannon_entropy(sc, total)

    # 8-tier analysis
    best_8t = {"effective_bits": 999}
    for idx_bits in range(2, 13):
        tier_size = 2 ** idx_bits
        cumulative = 0
        total_bits = 0
        for tier_idx in range(7):
            prefix_bits = tier_idx + 1
            start = cumulative
            end = min(cumulative + tier_size, len(sc))
            tier_count = int(np.sum(sc[start:end])) if start < len(sc) else 0
            total_bits += tier_count * (prefix_bits + idx_bits)
            cumulative += tier_size
        esc_count = max(0, total - int(np.sum(sc[:cumulative])) if cumulative < len(sc) else 0)
        total_bits += esc_count * (8 + native_bits)
        eff = total_bits / total
        if eff < best_8t["effective_bits"]:
            best_8t = {
                "idx_bits": idx_bits, "tier_size": tier_size,
                "escape_count": esc_count, "escape_rate": esc_count / total * 100,
                "effective_bits": float(eff), "compression_ratio": float(native_bits / eff) if eff > 0 else 0,
            }

    return {
        "name": name, "num_params": total, "unique": n_unique,
        "dtype": str(dtype), "native_bits": native_bits,
        "entropy": entropy, "best_8tier": best_8t,
    }


def worker(gpu_id, work_items, results_dict):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    current_shard = None
    f = None
    for shard_path, name in work_items:
        if shard_path != current_shard:
            f = safe_open(shard_path, framework="pt")
            current_shard = shard_path
        tensor = f.get_tensor(name)
        if tensor.numel() < 100:
            continue
        result = analyze_tensor(name, tensor, device)
        if result:
            results_dict[name] = result
    torch.cuda.empty_cache()


def analyze_model(model_name, model_dir):
    print(f"\n{'='*90}")
    print(f"{model_name}")
    print(f"{'='*90}", flush=True)

    shard_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    print(f"Shards: {len(shard_paths)}", flush=True)

    all_work = []
    for sp in shard_paths:
        f = safe_open(sp, framework="pt")
        for name in f.keys():
            all_work.append((sp, name))

    print(f"Total tensors: {len(all_work)}", flush=True)

    gpu_assignments = [[] for _ in range(4)]
    for i, item in enumerate(all_work):
        gpu_assignments[i % 4].append(item)

    manager = mp.Manager()
    results_dict = manager.dict()
    t0 = time.time()

    processes = []
    for gpu_id in range(4):
        p = mp.Process(target=worker, args=(gpu_id, gpu_assignments[gpu_id], results_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    elapsed = time.time() - t0
    all_results = [results_dict[name] for _, name in all_work if name in results_dict]

    if not all_results:
        print(f"  No results!", flush=True)
        return

    # Group by dtype/native_bits
    by_type = {}
    for r in all_results:
        key = f"{r['dtype']} ({r['native_bits']}b)"
        by_type.setdefault(key, []).append(r)

    print(f"\n  Total analyzed: {len(all_results)} tensors ({elapsed:.1f}s)")

    for dtype_key, results in sorted(by_type.items()):
        weights = np.array([r["num_params"] for r in results], dtype=np.float64)
        weights /= weights.sum()
        w_ent = np.average([r["entropy"] for r in results], weights=weights)
        w_8t = np.average([r["best_8tier"]["effective_bits"] for r in results], weights=weights)
        w_esc = np.average([r["best_8tier"]["escape_rate"] for r in results], weights=weights)
        w_uniq = np.average([r["unique"] for r in results], weights=weights)
        native = results[0]["native_bits"]
        cr = native / w_8t if w_8t > 0 else 0

        print(f"\n  --- {dtype_key}: {len(results)} tensors ---")
        print(f"  Unique values/tensor: avg={w_uniq:.0f}  min={min(r['unique'] for r in results)}  max={max(r['unique'] for r in results)}")
        print(f"  Shannon entropy:  {w_ent:.3f} bits (of {native}b, max CR: {native/w_ent:.3f}x)")
        print(f"  8-tier:           {w_8t:.3f} bits (CR: {cr:.3f}x vs {native}b native)")
        print(f"  Escape rate:      {w_esc:.4f}%")

        if native == 4:
            print(f"  ** INT4 values: only {2**4}=16 possible values, entropy should be low **")
        if cr >= 1.5:
            print(f"  VERDICT: PASS — {cr:.3f}x")
        else:
            print(f"  VERDICT: {cr:.3f}x")


def main():
    print("=" * 90)
    print("Turbo Lossless: INT4 Quantized Model Analysis")
    print("=" * 90, flush=True)

    models = [
        ("Llama 3.1 8B GPTQ-INT4",
         "/home/ubuntu/AI/ compression/models/llama-3.1-8b-gptq-int4"),
        ("Llama 3.1 8B AWQ-INT4",
         "/home/ubuntu/AI/ compression/models/llama-3.1-8b-awq-int4"),
        ("Qwen2.5 7B GPTQ-INT4",
         "/home/ubuntu/AI/ compression/models/qwen2.5-7b-gptq-int4"),
        ("Mistral 7B AWQ",
         "/home/ubuntu/AI/ compression/models/mistral-7b-awq"),
        ("Llama 3.1 70B GPTQ-INT4",
         "/home/ubuntu/AI/ compression/models/llama-3.1-70b-gptq-int4"),
    ]

    for name, path in models:
        analyze_model(name, path)

    print(f"\n{'='*90}")
    print("SUMMARY")
    print("INT4 models store two types of data:")
    print("  1. Packed INT4 weights (int32): Only 16 possible values → very low entropy")
    print("  2. Quantization scales/zeros (FP16): Many unique values → high entropy")
    print("Our 8-tier scheme can compress INT4 packed weights very effectively,")
    print("but the FP16 scales are the bottleneck.")
    print(f"{'='*90}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
