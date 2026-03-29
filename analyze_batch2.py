"""
Batch 8-tier analysis: MiniMax, Yi, StarCoder2, GPTQ INT4.
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


def analyze_tensor_8tier(name, tensor_bf16, device):
    t = tensor_bf16.to(device)
    dtype_bits = 16  # default BF16/FP16

    # Handle different dtypes
    if t.dtype in (torch.bfloat16, torch.float16):
        raw = t.view(torch.int16).ravel()
        dtype_bits = 16
    elif t.dtype == torch.float32:
        # Convert to bf16 for analysis? No — analyze as-is with uint32
        # Actually skip FP32 — not our target
        return None
    elif t.dtype == torch.int32:
        # GPTQ packs INT4 into int32 — analyze the packed representation
        raw = t.view(torch.int16).ravel()  # view as int16 pairs
        dtype_bits = 16
    elif t.dtype == torch.int8 or t.dtype == torch.uint8:
        raw = t.ravel().to(torch.int16)
        dtype_bits = 8
    else:
        # Try viewing as int16
        try:
            raw = t.contiguous().view(torch.int16).ravel()
            dtype_bits = 16
        except:
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

    best_8t = {"effective_bits": 999}
    for idx_bits in range(6, 13):
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
        total_bits += esc_count * (8 + dtype_bits)
        eff = total_bits / total
        if eff < best_8t["effective_bits"]:
            best_8t = {
                "idx_bits": idx_bits, "tier_size": tier_size,
                "escape_count": esc_count, "escape_rate": esc_count / total * 100,
                "effective_bits": float(eff), "compression_ratio": float(dtype_bits / eff),
            }

    return {
        "name": name, "num_params": total, "unique_bf16": n_unique,
        "dtype": str(t.dtype), "dtype_bits": dtype_bits,
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
        if tensor.numel() < 10000:
            continue
        result = analyze_tensor_8tier(name, tensor, device)
        if result:
            results_dict[name] = result
    torch.cuda.empty_cache()


def analyze_model(model_name, model_dir, total_params_b):
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}", flush=True)

    shard_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_paths:
        shard_paths = sorted(glob.glob(os.path.join(model_dir, "**/*.safetensors"), recursive=True))
    print(f"Found {len(shard_paths)} shards", flush=True)

    all_work = []
    dtypes_seen = set()
    for sp in shard_paths:
        f = safe_open(sp, framework="pt")
        for name in f.keys():
            if "weight" in name and "norm" not in name and "embed" not in name and "layernorm" not in name.lower():
                t = f.get_tensor(name)
                dtypes_seen.add(str(t.dtype))
                all_work.append((sp, name))

    print(f"Weight tensors: {len(all_work)}, dtypes: {dtypes_seen}", flush=True)

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
        return None

    # Filter to only BF16/FP16 results for fair comparison
    bf16_results = [r for r in all_results if r["dtype_bits"] == 16]
    if not bf16_results:
        print(f"  No 16-bit tensors found! Dtypes: {set(r['dtype'] for r in all_results)}", flush=True)
        # Still show results for whatever dtype we have
        bf16_results = all_results

    weights = np.array([r["num_params"] for r in bf16_results], dtype=np.float64)
    weights /= weights.sum()

    w_ent = np.average([r["entropy"] for r in bf16_results], weights=weights)
    w_8t = np.average([r["best_8tier"]["effective_bits"] for r in bf16_results], weights=weights)
    w_esc = np.average([r["best_8tier"]["escape_rate"] for r in bf16_results], weights=weights)
    w_uniq = np.average([r["unique_bf16"] for r in bf16_results], weights=weights)
    total_params = sum(r["num_params"] for r in bf16_results)
    dtype_bits = bf16_results[0]["dtype_bits"]

    bf16_gb = total_params_b * 2 / 1e9
    comp_gb = total_params_b * (w_8t / 8) / 1e9

    print(f"\n  Tensors: {len(bf16_results)}  Params sampled: {total_params:,}  ({elapsed:.1f}s)")
    print(f"  Dtype: {bf16_results[0]['dtype']}  Unique/tensor: avg={w_uniq:.0f}")
    print(f"  Shannon entropy:  {w_ent:.3f} bits ({dtype_bits/w_ent:.3f}x)")
    print(f"  8-tier:           {w_8t:.3f} bits ({dtype_bits/w_8t:.3f}x)")
    print(f"  Escape rate:      {w_esc:.4f}%")

    cr = dtype_bits / w_8t
    verdict = "PASS" if cr >= 1.5 else "CLOSE" if cr >= 1.48 else "BELOW"
    print(f"  VERDICT: {verdict} — {cr:.3f}x", flush=True)

    return {
        "model": model_name, "tensors": len(bf16_results),
        "params_sampled": total_params, "dtype": bf16_results[0]["dtype"],
        "entropy": w_ent, "8tier_bits": w_8t,
        "cr": cr, "escape": w_esc, "unique_avg": w_uniq,
    }


def main():
    print("=" * 80)
    print("Turbo Lossless: Batch 8-Tier Analysis (Round 2)")
    print("=" * 80, flush=True)

    models = [
        ("Yi-1.5 34B (01.AI, Dense)",
         "/home/ubuntu/AI/ compression/models/yi-1.5-34b", 34e9),
        ("StarCoder2 7B (BigCode, Dense, Code)",
         "/home/ubuntu/AI/ compression/models/starcoder2-7b", 7e9),
        ("MiniMax-Text-01 (456B MoE)",
         "/home/ubuntu/AI/ compression/models/minimax-text-01", 456e9),
        ("Llama 3.1 8B GPTQ-INT4 (Quantized)",
         "/home/ubuntu/AI/ compression/models/llama-3.1-8b-gptq-int4", 8e9),
    ]

    results = []
    for name, path, params in models:
        r = analyze_model(name, path, params)
        if r:
            results.append(r)

    # Final summary
    print(f"\n{'='*80}")
    print(f"COMPLETE CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    all_models = [
        {"model": "Llama 3.1 8B", "type": "Dense", "entropy": 10.42, "8tier_bits": 10.60, "cr": 1.509, "escape": 0.03},
        {"model": "Phi-4 14B", "type": "Dense", "entropy": 10.49, "8tier_bits": 10.62, "cr": 1.507, "escape": 0.03},
        {"model": "Codestral 22B", "type": "Dense", "entropy": 10.51, "8tier_bits": 10.64, "cr": 1.504, "escape": 0.03},
        {"model": "Qwen3 30B MoE", "type": "MoE", "entropy": 10.50, "8tier_bits": 10.63, "cr": 1.505, "escape": 0.03},
        {"model": "Llama 3.1 70B", "type": "Dense", "entropy": 10.36, "8tier_bits": 10.56, "cr": 1.516, "escape": 0.05},
        {"model": "Mistral Large 123B", "type": "Dense", "entropy": 10.51, "8tier_bits": 10.64, "cr": 1.503, "escape": 0.03},
        {"model": "Qwen3-235B MoE", "type": "MoE", "entropy": 11.22, "8tier_bits": 11.38, "cr": 1.406, "escape": 0.18},
        {"model": "DeepSeek V3 671B", "type": "MoE", "entropy": 13.35, "8tier_bits": 13.50, "cr": 1.185, "escape": 2.02},
    ]

    for r in results:
        all_models.append({
            "model": r["model"], "type": "see above",
            "entropy": r["entropy"], "8tier_bits": r["8tier_bits"],
            "cr": r["cr"], "escape": r["escape"],
        })

    print(f"\n  {'Model':<40s} {'Entropy':>8s} {'8-tier':>8s} {'CR':>8s} {'Esc%':>8s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_models:
        print(f"  {r['model']:<40s} {r['entropy']:>8.2f} {r['8tier_bits']:>8.2f} {r['cr']:>8.3f}x {r['escape']:>8.4f}")

    print(f"\n{'='*80}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
