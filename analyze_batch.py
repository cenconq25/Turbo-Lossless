"""
Batch 8-tier analysis: Phi-4, DeepSeek V3, Qwen3-235B MoE.
Runs sequentially per model, parallel across GPUs within each.
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
    raw = t.view(torch.int16).ravel()
    n = raw.shape[0]
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
        total_bits += esc_count * (8 + 16)
        eff = total_bits / total
        if eff < best_8t["effective_bits"]:
            best_8t = {
                "idx_bits": idx_bits, "tier_size": tier_size,
                "escape_count": esc_count, "escape_rate": esc_count / total * 100,
                "effective_bits": float(eff), "compression_ratio": float(16 / eff),
            }

    return {
        "name": name, "num_params": total, "unique_bf16": n_unique,
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
    for sp in shard_paths:
        f = safe_open(sp, framework="pt")
        for name in f.keys():
            if "weight" in name and "norm" not in name and "embed" not in name and "layernorm" not in name.lower():
                all_work.append((sp, name))

    print(f"Weight tensors: {len(all_work)}", flush=True)

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
        print(f"  No results! Skipping.", flush=True)
        return None

    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()

    w_ent = np.average([r["entropy"] for r in all_results], weights=weights)
    w_8t = np.average([r["best_8tier"]["effective_bits"] for r in all_results], weights=weights)
    w_esc = np.average([r["best_8tier"]["escape_rate"] for r in all_results], weights=weights)
    w_uniq = np.average([r["unique_bf16"] for r in all_results], weights=weights)
    total_params = sum(r["num_params"] for r in all_results)

    bf16_gb = total_params_b * 2 / 1e9
    comp_gb = total_params_b * (w_8t / 8) / 1e9

    print(f"\n  Tensors: {len(all_results)}  Params sampled: {total_params:,}  ({elapsed:.1f}s)")
    print(f"  Unique BF16/tensor: avg={w_uniq:.0f}  min={min(r['unique_bf16'] for r in all_results)}  max={max(r['unique_bf16'] for r in all_results)}")
    print(f"  Shannon entropy:  {w_ent:.3f} bits ({16/w_ent:.3f}x)")
    print(f"  8-tier:           {w_8t:.3f} bits ({16/w_8t:.3f}x)")
    print(f"  Escape rate:      {w_esc:.4f}%")
    print(f"  BF16: {bf16_gb:.1f} GB → Compressed: {comp_gb:.1f} GB  (saved {bf16_gb-comp_gb:.1f} GB)")

    verdict = "PASS" if 16/w_8t >= 1.5 else "CLOSE" if 16/w_8t >= 1.48 else "BELOW"
    print(f"  VERDICT: {verdict} — {16/w_8t:.3f}x", flush=True)

    return {
        "model": model_name, "tensors": len(all_results),
        "params_sampled": total_params,
        "entropy": w_ent, "8tier_bits": w_8t,
        "cr": 16/w_8t, "escape": w_esc, "unique_avg": w_uniq,
        "bf16_gb": bf16_gb, "compressed_gb": comp_gb,
    }


def main():
    print("=" * 80)
    print("Turbo Lossless: Batch 8-Tier Analysis")
    print("=" * 80, flush=True)

    models = [
        ("Phi-4 (14B, Microsoft, Dense)",
         "/home/ubuntu/AI/ compression/models/phi-4", 14e9),
        ("DeepSeek V3 (671B MoE, 2 shards)",
         "/home/ubuntu/AI/ compression/models/deepseek-v3", 671e9),
        ("Qwen3-235B-A22B (235B MoE, 2 shards)",
         "/home/ubuntu/AI/ compression/models/qwen3-235b-moe", 235e9),
    ]

    results = []
    for name, path, params in models:
        r = analyze_model(name, path, params)
        if r:
            results.append(r)

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL CROSS-MODEL SUMMARY (all models)")
    print(f"{'='*80}")

    prior = [
        {"model": "Llama 3.1 8B", "entropy": 10.42, "8tier_bits": 10.60, "cr": 1.509, "escape": 0.03},
        {"model": "Codestral 22B", "entropy": 10.51, "8tier_bits": 10.64, "cr": 1.504, "escape": 0.03},
        {"model": "Qwen3 30B MoE", "entropy": 10.50, "8tier_bits": 10.63, "cr": 1.505, "escape": 0.03},
        {"model": "Llama 3.1 70B", "entropy": 10.36, "8tier_bits": 10.56, "cr": 1.516, "escape": 0.05},
        {"model": "Mistral Large 123B", "entropy": 10.51, "8tier_bits": 10.64, "cr": 1.503, "escape": 0.03},
    ]

    print(f"\n  {'Model':<35s} {'Entropy':>8s} {'8-tier':>8s} {'CR':>8s} {'Esc%':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in prior:
        print(f"  {r['model']:<35s} {r['entropy']:>8.2f} {r['8tier_bits']:>8.2f} {r['cr']:>8.3f}x {r['escape']:>8.4f}")
    for r in results:
        print(f"  {r['model']:<35s} {r['entropy']:>8.2f} {r['8tier_bits']:>8.2f} {r['cr']:>8.3f}x {r['escape']:>8.4f}")

    print(f"\n{'='*80}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
