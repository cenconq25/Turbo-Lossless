"""
Deep analysis: find the maximum possible compression for BF16 weights.

Computes:
1. Shannon entropy (theoretical ceiling for any lossless scheme)
2. Global frequency codebook (no grouping — just rank BF16 values by frequency)
3. Multi-tier fixed-width coding (GPU-friendly variable-length approximation)
4. Optimal tier boundaries via exhaustive search
5. Per-tensor vs global codebook comparison
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import math
import json
import time

MODEL_PATH = "/home/ubuntu/AI/ compression/models/llama-3.1-8b/llama-3.1-8b.safetensors"


def analyze_tensor_deep(name, tensor_bf16, device):
    """Deep analysis of one tensor."""
    t = tensor_bf16.to(device)
    raw = t.view(torch.int16).ravel()
    n = raw.shape[0]

    # Full frequency table (BF16 has max 65536 unique values)
    unique_vals, counts = raw.unique(return_counts=True)
    n_unique = unique_vals.shape[0]

    # Sort by frequency (descending)
    sorted_idx = counts.argsort(descending=True)
    sorted_counts = counts[sorted_idx]
    sorted_vals = unique_vals[sorted_idx]

    probs = sorted_counts.float() / n

    # --- 1. Shannon Entropy ---
    log2_probs = torch.log2(probs)
    entropy = -float((probs * log2_probs).sum())

    # --- 2. Cumulative coverage by rank ---
    # How many top-K values cover what % of the tensor?
    cum_coverage = torch.cumsum(sorted_counts, dim=0).float() / n

    coverage_at = {}
    for k in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        if k <= n_unique:
            coverage_at[k] = float(cum_coverage[k-1])
        else:
            coverage_at[k] = 1.0

    # --- 3. Multi-tier fixed-width coding ---
    # Tier structure: tier1 (short code) | tier2 (medium) | tier3 (long) | escape (raw 16b)
    # Optimize: how many values in each tier to minimize effective bits?
    #
    # For a tier with N entries, we need ceil(log2(N)) bits for the index,
    # plus a prefix to identify which tier (like Huffman prefix-free codes).
    #
    # Simplest GPU-friendly scheme: power-of-2 tiers with prefix bits
    # Tier 0: 1-bit prefix '0' + B0-bit index = 1+B0 bits (most frequent values)
    # Tier 1: 2-bit prefix '10' + B1-bit index = 2+B1 bits
    # Tier 2: 3-bit prefix '110' + B2-bit index = 3+B2 bits
    # Tier 3: 3-bit prefix '111' + 16-bit raw = 19 bits (escape)

    # Try various tier configurations
    tier_configs = []

    # Format: [(tier_index_bits, prefix_bits), ...]
    # The last "tier" is always escape: prefix + 16 raw bits
    configs = [
        # 2-tier: short + escape
        {"name": "2tier", "tiers": [(7, 1), (8, 2)], "escape_prefix": 2, "escape_extra": 16},
        # 3-tier: short + medium + escape
        {"name": "3tier_7_7", "tiers": [(7, 1), (7, 2)], "escape_prefix": 3, "escape_extra": 16},
        {"name": "3tier_8_8", "tiers": [(8, 1), (8, 2)], "escape_prefix": 3, "escape_extra": 16},
        {"name": "3tier_8_10", "tiers": [(8, 1), (10, 2)], "escape_prefix": 3, "escape_extra": 16},
        {"name": "3tier_9_9", "tiers": [(9, 1), (9, 2)], "escape_prefix": 3, "escape_extra": 16},
        {"name": "3tier_10_10", "tiers": [(10, 1), (10, 2)], "escape_prefix": 3, "escape_extra": 16},
        # 4-tier: short + medium + long + escape
        {"name": "4tier_6_8_10", "tiers": [(6, 1), (8, 2), (10, 3)], "escape_prefix": 4, "escape_extra": 16},
        {"name": "4tier_7_9_11", "tiers": [(7, 1), (9, 2), (11, 3)], "escape_prefix": 4, "escape_extra": 16},
        {"name": "4tier_8_8_8", "tiers": [(8, 1), (8, 2), (8, 3)], "escape_prefix": 4, "escape_extra": 16},
        {"name": "4tier_7_8_10", "tiers": [(7, 1), (8, 2), (10, 3)], "escape_prefix": 4, "escape_extra": 16},
        {"name": "4tier_8_10_12", "tiers": [(8, 1), (10, 2), (12, 3)], "escape_prefix": 4, "escape_extra": 16},
    ]

    tier_results = []
    sc = sorted_counts.cpu().numpy()
    total = int(n)

    for cfg in configs:
        tiers = cfg["tiers"]
        offset = 0
        total_bits = 0
        tier_detail = []

        for idx_bits, prefix_bits in tiers:
            tier_size = 2 ** idx_bits
            tier_count = int(np.sum(sc[offset:offset + tier_size])) if offset < len(sc) else 0
            bits_per_val = prefix_bits + idx_bits
            total_bits += tier_count * bits_per_val
            tier_detail.append({
                "size": tier_size, "count": tier_count,
                "bits": bits_per_val,
                "coverage": tier_count / total
            })
            offset += tier_size

        # Escape: remaining values
        escape_count = total - int(np.sum(sc[:offset]))
        escape_bits = cfg["escape_prefix"] + cfg["escape_extra"]
        total_bits += escape_count * escape_bits

        effective_bits = total_bits / total
        tier_results.append({
            "name": cfg["name"],
            "effective_bits": float(effective_bits),
            "compression_ratio": float(16 / effective_bits) if effective_bits > 0 else 0,
            "escape_count": int(escape_count),
            "escape_rate": float(escape_count / total * 100),
            "tiers": tier_detail
        })

    # --- 4. Exhaustive search for optimal 3-tier config ---
    best_3t = {"effective_bits": 999}
    for b1 in range(5, 13):  # tier1 index bits
        for b2 in range(5, 14):  # tier2 index bits
            t1_size = 2 ** b1
            t2_size = 2 ** b2
            if t1_size + t2_size > len(sc):
                continue
            t1_count = int(np.sum(sc[:t1_size]))
            t2_count = int(np.sum(sc[t1_size:t1_size + t2_size]))
            esc_count = total - t1_count - t2_count

            # Prefix: '0' for tier1, '10' for tier2, '11' for escape
            t1_bits = 1 + b1
            t2_bits = 2 + b2
            esc_bits = 2 + 16  # prefix '11' + raw 16

            eff = (t1_count * t1_bits + t2_count * t2_bits + esc_count * esc_bits) / total
            if eff < best_3t["effective_bits"]:
                best_3t = {
                    "b1": b1, "b2": b2,
                    "t1_size": t1_size, "t2_size": t2_size,
                    "t1_coverage": t1_count / total,
                    "t2_coverage": t2_count / total,
                    "escape_rate": esc_count / total,
                    "effective_bits": float(eff),
                    "compression_ratio": float(16 / eff)
                }

    # --- 5. Exhaustive search for optimal 4-tier config ---
    best_4t = {"effective_bits": 999}
    for b1 in range(5, 11):
        for b2 in range(5, 12):
            for b3 in range(5, 14):
                t1_size = 2 ** b1
                t2_size = 2 ** b2
                t3_size = 2 ** b3
                if t1_size + t2_size + t3_size > len(sc):
                    continue
                t1_count = int(np.sum(sc[:t1_size]))
                t2_count = int(np.sum(sc[t1_size:t1_size + t2_size]))
                t3_count = int(np.sum(sc[t1_size + t2_size:t1_size + t2_size + t3_size]))
                esc_count = total - t1_count - t2_count - t3_count

                # Prefix: '0', '10', '110', '111'+raw
                t1_bits = 1 + b1
                t2_bits = 2 + b2
                t3_bits = 3 + b3
                esc_bits = 3 + 16

                eff = (t1_count * t1_bits + t2_count * t2_bits +
                       t3_count * t3_bits + esc_count * esc_bits) / total
                if eff < best_4t["effective_bits"]:
                    best_4t = {
                        "b1": b1, "b2": b2, "b3": b3,
                        "t1_size": t1_size, "t2_size": t2_size, "t3_size": t3_size,
                        "t1_coverage": t1_count / total,
                        "t2_coverage": t2_count / total,
                        "t3_coverage": t3_count / total,
                        "escape_rate": esc_count / total,
                        "effective_bits": float(eff),
                        "compression_ratio": float(16 / eff)
                    }

    # --- 6. What if we just use Huffman? (theoretical) ---
    # Huffman gets within 1 bit of entropy per symbol
    huffman_est = entropy + 0.05  # practical Huffman overhead

    return {
        "name": name,
        "num_params": int(n),
        "unique_bf16": int(n_unique),
        "entropy": entropy,
        "huffman_est_bits": float(huffman_est),
        "huffman_est_cr": float(16 / huffman_est),
        "coverage": coverage_at,
        "tier_configs": tier_results,
        "best_3tier": best_3t,
        "best_4tier": best_4t,
        # Top value stats
        "top1_pct": float(probs[0]) * 100,
        "top10_pct": float(probs[:10].sum()) * 100,
        "top100_pct": float(probs[:100].sum()) * 100,
    }


def worker(gpu_id, tensor_names, results_dict):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    f = safe_open(MODEL_PATH, framework="pt")

    for name in tensor_names:
        t0 = time.time()
        tensor = f.get_tensor(name)
        if tensor.numel() < 10000:
            continue
        result = analyze_tensor_deep(name, tensor, device)
        elapsed = time.time() - t0
        results_dict[name] = result

        ent = result["entropy"]
        b3 = result["best_3tier"]["effective_bits"]
        b4 = result["best_4tier"]["effective_bits"]
        cr3 = result["best_3tier"]["compression_ratio"]
        cr4 = result["best_4tier"]["compression_ratio"]
        print(f"[GPU{gpu_id}] {name:<40s} "
              f"entropy={ent:.2f}b  best3t={b3:.2f}b({cr3:.3f}x)  "
              f"best4t={b4:.2f}b({cr4:.3f}x)  uniq={result['unique_bf16']}  ({elapsed:.1f}s)",
              flush=True)
    torch.cuda.empty_cache()


def main():
    print("=" * 90)
    print("Turbo Lossless: Deep Optimal Compression Analysis")
    print("=" * 90, flush=True)

    f = safe_open(MODEL_PATH, framework="pt")
    all_names = list(f.keys())
    weight_names = [n for n in all_names if "weight" in n
                    and "norm" not in n
                    and "embd" not in n and "embed" not in n]

    # Sample representative tensors
    sample_layers = {0, 3, 7, 11, 15, 19, 23, 27, 31}
    sampled = []
    for n in weight_names:
        if "output.weight" == n:
            sampled.append(n)
            continue
        for layer in sample_layers:
            if f"blk.{layer}." in n:
                sampled.append(n)
                break

    print(f"Analyzing {len(sampled)} tensors across layers {sorted(sample_layers)}\n", flush=True)

    gpu_assignments = [[] for _ in range(4)]
    for i, name in enumerate(sampled):
        gpu_assignments[i % 4].append(name)

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
    all_results = [results_dict[n] for n in sampled if n in results_dict]

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"RESULTS SUMMARY ({len(all_results)} tensors, {elapsed:.1f}s)")
    print(f"{'=' * 90}")

    entropies = [r["entropy"] for r in all_results]
    uniques = [r["unique_bf16"] for r in all_results]
    b3s = [r["best_3tier"]["effective_bits"] for r in all_results]
    b4s = [r["best_4tier"]["effective_bits"] for r in all_results]
    cr3s = [r["best_3tier"]["compression_ratio"] for r in all_results]
    cr4s = [r["best_4tier"]["compression_ratio"] for r in all_results]
    huffs = [r["huffman_est_bits"] for r in all_results]

    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()

    w_ent = np.average(entropies, weights=weights)
    w_b3 = np.average(b3s, weights=weights)
    w_b4 = np.average(b4s, weights=weights)
    w_huff = np.average(huffs, weights=weights)

    print(f"\n--- Theoretical Limits ---")
    print(f"Shannon entropy:     {w_ent:.2f} bits/param  (CR: {16/w_ent:.3f}x) — absolute ceiling")
    print(f"Huffman estimate:    {w_huff:.2f} bits/param  (CR: {16/w_huff:.3f}x) — practical ceiling")

    print(f"\n--- GPU-Friendly Fixed-Width Tiered Coding ---")
    print(f"Best 3-tier:         {w_b3:.2f} bits/param  (CR: {16/w_b3:.3f}x)")
    print(f"Best 4-tier:         {w_b4:.2f} bits/param  (CR: {16/w_b4:.3f}x)")

    print(f"\n--- Unique BF16 Values Per Tensor ---")
    print(f"Mean: {np.mean(uniques):.0f}   Min: {np.min(uniques)}   Max: {np.max(uniques)}")

    print(f"\n--- Coverage by Top-K Values ---")
    for k in [64, 128, 256, 512, 1024, 2048, 4096]:
        covs = [r["coverage"].get(str(k) if isinstance(list(r["coverage"].keys())[0], str) else k, r["coverage"].get(k, 0)) for r in all_results]
        avg_cov = np.average(covs, weights=weights)
        print(f"  Top-{k:<5d} covers {avg_cov*100:.1f}% of values")

    # Show best 3-tier config details
    print(f"\n--- Optimal 3-Tier Config (most common across tensors) ---")
    from collections import Counter
    configs_3t = Counter()
    for r in all_results:
        b = r["best_3tier"]
        configs_3t[(b["b1"], b["b2"])] += 1
    most_common_3t = configs_3t.most_common(5)
    for (b1, b2), count in most_common_3t:
        t1 = 2**b1
        t2 = 2**b2
        print(f"  Tier1: {t1} entries ({1+b1}b)  Tier2: {t2} entries ({2+b2}b)  "
              f"Escape: 18b  — chosen by {count}/{len(all_results)} tensors")

    # Show best 4-tier config details
    print(f"\n--- Optimal 4-Tier Config (most common across tensors) ---")
    configs_4t = Counter()
    for r in all_results:
        b = r["best_4tier"]
        configs_4t[(b["b1"], b["b2"], b["b3"])] += 1
    most_common_4t = configs_4t.most_common(5)
    for (b1, b2, b3), count in most_common_4t:
        t1, t2, t3 = 2**b1, 2**b2, 2**b3
        print(f"  T1: {t1} ({1+b1}b)  T2: {t2} ({2+b2}b)  T3: {t3} ({3+b3}b)  "
              f"Esc: 19b  — chosen by {count}/{len(all_results)} tensors")

    # Value concentration
    print(f"\n--- Value Concentration ---")
    top1 = np.average([r["top1_pct"] for r in all_results], weights=weights)
    top10 = np.average([r["top10_pct"] for r in all_results], weights=weights)
    top100 = np.average([r["top100_pct"] for r in all_results], weights=weights)
    print(f"  Top-1 value:   {top1:.2f}% of all values")
    print(f"  Top-10 values: {top10:.2f}%")
    print(f"  Top-100 values: {top100:.2f}%")

    # Final verdict
    print(f"\n{'=' * 90}")
    print(f"COMPRESSION POTENTIAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Original BF16:              16.00 bits/param  (baseline)")
    print(f"  Shannon entropy (ceiling):  {w_ent:.2f} bits/param  ({16/w_ent:.3f}x)")
    print(f"  Huffman (practical ceiling): {w_huff:.2f} bits/param  ({16/w_huff:.3f}x)")
    print(f"  Best 4-tier GPU-friendly:   {w_b4:.2f} bits/param  ({16/w_b4:.3f}x)")
    print(f"  Best 3-tier GPU-friendly:   {w_b3:.2f} bits/param  ({16/w_b3:.3f}x)")
    print(f"  Original 15+1 scheme:       13.75 bits/param  (1.164x) [from prior analysis]")
    print(f"{'=' * 90}")

    # Save
    out_path = "/home/ubuntu/AI/ compression/optimal_results.json"

    # Convert results for JSON serialization
    serializable = []
    for r in all_results:
        sr = dict(r)
        # Convert coverage keys to strings
        sr["coverage"] = {str(k): v for k, v in r["coverage"].items()}
        serializable.append(sr)

    with open(out_path, "w") as fout:
        json.dump(serializable, fout, indent=2)
    print(f"\nDetailed results saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
