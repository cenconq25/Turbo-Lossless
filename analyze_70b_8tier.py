"""
8-Tier analysis on Llama 3.1 70B Instruct (BF16 safetensors, 30 shards).
Validates that the 8-tier scheme works on larger models.
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

MODEL_DIR = "/home/ubuntu/AI/ compression/models/llama-3.1-70b-instruct"


def shannon_entropy(sorted_counts, total):
    """Compute Shannon entropy from sorted frequency counts."""
    probs = sorted_counts / total
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def analyze_tensor_8tier(name, tensor_bf16, device):
    """Run 8-tier analysis on one tensor."""
    t = tensor_bf16.to(device)
    raw = t.view(torch.int16).ravel()
    n = raw.shape[0]

    # Full frequency table
    unique_vals, counts = raw.unique(return_counts=True)
    sorted_idx = counts.argsort(descending=True)
    sc = counts[sorted_idx].cpu().numpy().astype(np.int64)
    total = int(n)
    n_unique = int(unique_vals.shape[0])

    # Shannon entropy
    entropy = shannon_entropy(sc, total)

    # 8-tier: optimized tier sizes via exhaustive search
    # Prefix scheme: tier i has prefix of (i+1) bits, last tier = escape
    # Try different index_bits per tier
    best_8t = {"effective_bits": 999}

    for idx_bits in range(6, 13):
        tier_size = 2 ** idx_bits
        cumulative = 0
        total_bits = 0

        for tier_idx in range(7):  # 7 coded tiers
            prefix_bits = tier_idx + 1
            start = cumulative
            end = min(cumulative + tier_size, len(sc))
            tier_count = int(np.sum(sc[start:end])) if start < len(sc) else 0
            total_bits += tier_count * (prefix_bits + idx_bits)
            cumulative += tier_size

        # Escape tier
        esc_count = total - int(np.sum(sc[:cumulative])) if cumulative < len(sc) else 0
        if esc_count < 0:
            esc_count = 0
        total_bits += esc_count * (8 + 16)  # 8-bit escape prefix + 16-bit raw

        eff = total_bits / total
        if eff < best_8t["effective_bits"]:
            best_8t = {
                "idx_bits": idx_bits,
                "tier_size": tier_size,
                "total_codebook": tier_size * 7,
                "escape_count": esc_count,
                "escape_rate": esc_count / total * 100,
                "effective_bits": float(eff),
                "compression_ratio": float(16 / eff),
            }

    # Also compute 4-tier for comparison
    best_4t = {"effective_bits": 999}
    for b1 in range(6, 13):
        for b2 in range(6, 14):
            t1_size = 2 ** b1
            t2_size = 2 ** b2
            if t1_size + t2_size > len(sc):
                t1_count = int(np.sum(sc[:min(t1_size, len(sc))]))
                t2_count = int(np.sum(sc[min(t1_size, len(sc)):min(t1_size + t2_size, len(sc))]))
            else:
                t1_count = int(np.sum(sc[:t1_size]))
                t2_count = int(np.sum(sc[t1_size:t1_size + t2_size]))
            esc_count = total - t1_count - t2_count
            if esc_count < 0:
                esc_count = 0

            t1_bits = 1 + b1
            t2_bits = 2 + b2
            esc_bits = 2 + 16

            eff = (t1_count * t1_bits + t2_count * t2_bits + esc_count * esc_bits) / total
            if eff < best_4t["effective_bits"]:
                best_4t = {
                    "b1": b1, "b2": b2,
                    "effective_bits": float(eff),
                    "compression_ratio": float(16 / eff),
                }

    # Coverage stats
    cum = np.cumsum(sc)
    coverage = {}
    for k in [512, 1024, 2048, 3584, 4096, 8192]:
        if k <= len(sc):
            coverage[k] = float(cum[k-1] / total)
        else:
            coverage[k] = 1.0

    return {
        "name": name,
        "num_params": total,
        "unique_bf16": n_unique,
        "entropy": entropy,
        "best_8tier": best_8t,
        "best_4tier": best_4t,
        "coverage": coverage,
    }


def worker(gpu_id, work_items, results_dict):
    """Worker process. work_items = list of (shard_path, tensor_name)."""
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
        print(f"[GPU{gpu_id}] {name:<50s} "
              f"entropy={ent:.2f}  8tier={b8:.2f}b({cr8:.3f}x)  "
              f"esc={esc:.3f}%  uniq={uniq}  ({elapsed:.1f}s)", flush=True)

    torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("Turbo Lossless: 8-Tier Analysis on Llama 3.1 70B Instruct")
    print("=" * 100, flush=True)

    # Discover all shards and their tensors
    shard_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))
    print(f"Found {len(shard_paths)} shards", flush=True)

    # Build work list: (shard_path, tensor_name) for weight tensors
    all_work = []
    for sp in shard_paths:
        f = safe_open(sp, framework="pt")
        for name in f.keys():
            if "weight" in name and "norm" not in name and "embed" not in name:
                all_work.append((sp, name))

    print(f"Total weight tensors: {len(all_work)}")

    # Sample: every 4th tensor for speed (still ~150+ tensors across all layers)
    # For 70B with 80 layers × 7 types = 560 weight tensors
    # Sample layers: 0, 9, 19, 29, 39, 49, 59, 69, 79
    sample_layers = {0, 9, 19, 29, 39, 49, 59, 69, 79}
    sampled = []
    for sp, name in all_work:
        for layer in sample_layers:
            if f"layers.{layer}." in name:
                sampled.append((sp, name))
                break
        if "lm_head" in name or "output" in name.split(".")[-1]:
            sampled.append((sp, name))

    if not sampled:
        # Fallback: sample every 8th tensor
        sampled = all_work[::8]

    print(f"Sampling {len(sampled)} tensors across layers {sorted(sample_layers)}\n", flush=True)

    # Distribute across 4 GPUs
    gpu_assignments = [[] for _ in range(4)]
    for i, item in enumerate(sampled):
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
    all_results = [results_dict[name] for _, name in sampled if name in results_dict]

    # Summary
    print(f"\n{'=' * 100}")
    print(f"SUMMARY — Llama 3.1 70B ({len(all_results)} tensors, {elapsed:.1f}s)")
    print(f"{'=' * 100}")

    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()

    w_ent = np.average([r["entropy"] for r in all_results], weights=weights)
    w_8t = np.average([r["best_8tier"]["effective_bits"] for r in all_results], weights=weights)
    w_4t = np.average([r["best_4tier"]["effective_bits"] for r in all_results], weights=weights)
    w_esc = np.average([r["best_8tier"]["escape_rate"] for r in all_results], weights=weights)
    w_uniq = np.average([r["unique_bf16"] for r in all_results], weights=weights)

    print(f"\nTotal params sampled: {sum(r['num_params'] for r in all_results):,}")
    print(f"Unique BF16 per tensor: avg={w_uniq:.0f}  "
          f"min={min(r['unique_bf16'] for r in all_results)}  "
          f"max={max(r['unique_bf16'] for r in all_results)}")

    print(f"\n--- Compression Results ---")
    print(f"  Shannon entropy:  {w_ent:.3f} bits/param  ({16/w_ent:.3f}x)")
    print(f"  8-tier:           {w_8t:.3f} bits/param  ({16/w_8t:.3f}x)")
    print(f"  4-tier:           {w_4t:.3f} bits/param  ({16/w_4t:.3f}x)")
    print(f"  8-tier escape:    {w_esc:.4f}%")

    # Coverage
    print(f"\n--- Value Coverage ---")
    for k in [512, 1024, 2048, 3584, 4096, 8192]:
        covs = [r["coverage"].get(str(k) if isinstance(list(r["coverage"].keys())[0], str) else k,
                r["coverage"].get(k, 0)) for r in all_results]
        avg = np.average(covs, weights=weights)
        print(f"  Top-{k:<5d}: {avg*100:.2f}%")

    # Compare with 8B
    print(f"\n--- 70B vs 8B Comparison ---")
    print(f"  {'Metric':<25s} {'8B':>12s} {'70B':>12s}")
    print(f"  {'Shannon entropy':<25s} {'10.42':>12s} {w_ent:>12.3f}")
    print(f"  {'8-tier bits':<25s} {'10.60':>12s} {w_8t:>12.3f}")
    print(f"  {'8-tier CR':<25s} {'1.509x':>12s} {16/w_8t:>12.3f}x")
    print(f"  {'Unique BF16/tensor':<25s} {'~5,124':>12s} {w_uniq:>12.0f}")
    print(f"  {'Escape rate':<25s} {'~0.03%':>12s} {w_esc:>12.4f}%")

    # VRAM impact
    print(f"\n--- VRAM Impact (70B model) ---")
    bf16_size = 70e9 * 2 / 1e9
    compressed_size = 70e9 * (w_8t / 8) / 1e9
    print(f"  BF16 original:  {bf16_size:.1f} GB")
    print(f"  8-tier:         {compressed_size:.1f} GB")
    print(f"  VRAM saved:     {bf16_size - compressed_size:.1f} GB")

    # Verdict
    print(f"\n{'=' * 100}")
    if 16 / w_8t >= 1.5:
        print(f"VERDICT: 8-TIER CONFIRMED on 70B — {16/w_8t:.3f}x compression (≥1.5x target)")
    else:
        print(f"VERDICT: 8-TIER on 70B — {16/w_8t:.3f}x compression")
    print(f"{'=' * 100}")

    # Breakdown by tensor type
    print(f"\n--- Breakdown by Tensor Type ---")
    type_groups = {}
    for r in all_results:
        name = r["name"]
        if "q_proj" in name:
            ttype = "q_proj"
        elif "k_proj" in name:
            ttype = "k_proj"
        elif "v_proj" in name:
            ttype = "v_proj"
        elif "o_proj" in name:
            ttype = "o_proj"
        elif "down_proj" in name:
            ttype = "down_proj"
        elif "gate_proj" in name:
            ttype = "gate_proj"
        elif "up_proj" in name:
            ttype = "up_proj"
        elif "lm_head" in name or "output" in name:
            ttype = "lm_head"
        else:
            ttype = "other"
        type_groups.setdefault(ttype, []).append(r)

    print(f"  {'Type':<12s} {'Entropy':>8s} {'8-tier':>8s} {'CR':>7s} {'Esc%':>8s} {'Unique':>7s}")
    for ttype, rs in sorted(type_groups.items()):
        tw = np.array([r["num_params"] for r in rs], dtype=np.float64)
        tw /= tw.sum()
        avg_ent = np.average([r["entropy"] for r in rs], weights=tw)
        avg_8t = np.average([r["best_8tier"]["effective_bits"] for r in rs], weights=tw)
        avg_esc = np.average([r["best_8tier"]["escape_rate"] for r in rs], weights=tw)
        avg_uniq = np.average([r["unique_bf16"] for r in rs], weights=tw)
        print(f"  {ttype:<12s} {avg_ent:>8.3f} {avg_8t:>8.3f} {16/avg_8t:>7.3f}x {avg_esc:>8.4f} {avg_uniq:>7.0f}")

    # Save
    out_path = "/home/ubuntu/AI/ compression/analysis_70b_results.json"
    with open(out_path, "w") as fout:
        json.dump(all_results, fout, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
