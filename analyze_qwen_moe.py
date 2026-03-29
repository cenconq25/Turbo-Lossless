"""
8-Tier analysis on Qwen3 30B MoE (4 sampled shards).
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import math
import json
import time
import glob
import os

MODEL_DIR = "/home/ubuntu/AI/ compression/models/qwen3-30b-moe"


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
        esc_count = total - int(np.sum(sc[:cumulative])) if cumulative < len(sc) else 0
        if esc_count < 0:
            esc_count = 0
        total_bits += esc_count * (8 + 16)
        eff = total_bits / total
        if eff < best_8t["effective_bits"]:
            best_8t = {
                "idx_bits": idx_bits, "tier_size": tier_size,
                "escape_count": esc_count, "escape_rate": esc_count / total * 100,
                "effective_bits": float(eff), "compression_ratio": float(16 / eff),
            }

    cum = np.cumsum(sc)
    coverage = {}
    for k in [512, 1024, 2048, 3584, 4096, 8192]:
        coverage[k] = float(cum[min(k-1, len(sc)-1)] / total) if k <= len(sc) else 1.0

    return {
        "name": name, "num_params": total, "unique_bf16": n_unique,
        "entropy": entropy, "best_8tier": best_8t, "coverage": coverage,
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
        t0 = time.time()
        tensor = f.get_tensor(name)
        if tensor.numel() < 10000:
            continue
        result = analyze_tensor_8tier(name, tensor, device)
        elapsed = time.time() - t0
        results_dict[name] = result
        ent = result["entropy"]
        b8 = result["best_8tier"]["effective_bits"]
        cr8 = result["best_8tier"]["compression_ratio"]
        esc = result["best_8tier"]["escape_rate"]
        uniq = result["unique_bf16"]
        print(f"[GPU{gpu_id}] {name:<55s} "
              f"entropy={ent:.2f}  8tier={b8:.2f}b({cr8:.3f}x)  "
              f"esc={esc:.3f}%  uniq={uniq}  ({elapsed:.1f}s)", flush=True)
    torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("Turbo Lossless: 8-Tier Analysis on Qwen3 30B MoE (4 shards)")
    print("=" * 100, flush=True)

    shard_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))
    print(f"Found {len(shard_paths)} shards", flush=True)

    all_work = []
    for sp in shard_paths:
        f = safe_open(sp, framework="pt")
        for name in f.keys():
            if "weight" in name and "norm" not in name and "embed" not in name:
                all_work.append((sp, name))

    print(f"Total weight tensors in sampled shards: {len(all_work)}\n", flush=True)

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

    print(f"\n{'=' * 100}")
    print(f"SUMMARY — Qwen3 30B MoE ({len(all_results)} tensors from 4 shards, {elapsed:.1f}s)")
    print(f"{'=' * 100}")

    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()

    w_ent = np.average([r["entropy"] for r in all_results], weights=weights)
    w_8t = np.average([r["best_8tier"]["effective_bits"] for r in all_results], weights=weights)
    w_esc = np.average([r["best_8tier"]["escape_rate"] for r in all_results], weights=weights)
    w_uniq = np.average([r["unique_bf16"] for r in all_results], weights=weights)

    print(f"\nTotal params sampled: {sum(r['num_params'] for r in all_results):,}")
    print(f"Unique BF16 per tensor: avg={w_uniq:.0f}  "
          f"min={min(r['unique_bf16'] for r in all_results)}  "
          f"max={max(r['unique_bf16'] for r in all_results)}")

    print(f"\n--- Compression Results ---")
    print(f"  Shannon entropy:  {w_ent:.3f} bits/param  ({16/w_ent:.3f}x)")
    print(f"  8-tier:           {w_8t:.3f} bits/param  ({16/w_8t:.3f}x)")
    print(f"  8-tier escape:    {w_esc:.4f}%")

    for k in [512, 1024, 2048, 3584, 4096, 8192]:
        covs = [r["coverage"].get(k, 0) for r in all_results]
        avg = np.average(covs, weights=weights)
        print(f"  Top-{k:<5d}: {avg*100:.2f}%")

    # VRAM impact
    bf16_size = 30e9 * 2 / 1e9
    compressed_size = 30e9 * (w_8t / 8) / 1e9
    print(f"\n--- VRAM Impact (30B MoE model) ---")
    print(f"  BF16 original:  {bf16_size:.1f} GB")
    print(f"  8-tier:         {compressed_size:.1f} GB")
    print(f"  VRAM saved:     {bf16_size - compressed_size:.1f} GB")

    # Cross-model comparison
    print(f"\n--- Cross-Model Comparison ---")
    print(f"  {'Model':<25s} {'Entropy':>8s} {'8-tier':>8s} {'CR':>8s} {'Esc%':>8s}")
    print(f"  {'Llama 3.1 8B':<25s} {'10.42':>8s} {'10.60':>8s} {'1.509x':>8s} {'0.03%':>8s}")
    print(f"  {'Codestral 22B':<25s} {'10.51':>8s} {'10.64':>8s} {'1.504x':>8s} {'0.03%':>8s}")
    print(f"  {'Llama 3.1 70B':<25s} {'10.36':>8s} {'10.56':>8s} {'1.516x':>8s} {'0.05%':>8s}")
    print(f"  {'Qwen3 30B MoE':<25s} {w_ent:>8.2f} {w_8t:>8.2f} {16/w_8t:>8.3f}x {w_esc:>8.4f}%")

    print(f"\n{'=' * 100}")
    if 16 / w_8t >= 1.5:
        print(f"VERDICT: 8-TIER CONFIRMED on Qwen3 30B MoE — {16/w_8t:.3f}x compression")
    else:
        print(f"VERDICT: 8-TIER on Qwen3 30B MoE — {16/w_8t:.3f}x compression")
    print(f"{'=' * 100}")

    # Breakdown by type
    print(f"\n--- Breakdown by Tensor Type ---")
    type_groups = {}
    for r in all_results:
        name = r["name"]
        for ttype in ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj", "lm_head", "output"]:
            if ttype in name:
                type_groups.setdefault(ttype, []).append(r)
                break
        else:
            type_groups.setdefault("other", []).append(r)

    print(f"  {'Type':<12s} {'Entropy':>8s} {'8-tier':>8s} {'CR':>7s} {'Esc%':>8s} {'Unique':>7s}")
    for ttype, rs in sorted(type_groups.items()):
        tw = np.array([r["num_params"] for r in rs], dtype=np.float64)
        tw /= tw.sum()
        avg_ent = np.average([r["entropy"] for r in rs], weights=tw)
        avg_8t = np.average([r["best_8tier"]["effective_bits"] for r in rs], weights=tw)
        avg_esc = np.average([r["best_8tier"]["escape_rate"] for r in rs], weights=tw)
        avg_uniq = np.average([r["unique_bf16"] for r in rs], weights=tw)
        print(f"  {ttype:<12s} {avg_ent:>8.3f} {avg_8t:>8.3f} {16/avg_8t:>7.3f}x {avg_esc:>8.4f} {avg_uniq:>7.0f}")

    out_path = "/home/ubuntu/AI/ compression/analysis_mistral_large_results.json"
    with open(out_path, "w") as fout:
        json.dump(all_results, fout, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
