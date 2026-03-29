"""
Turbo Lossless: GPU-Accelerated Weight Distribution Analysis
Uses 4 AMD GPUs in parallel to profile all 226 weight tensors.
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import json
import time
import sys

MODEL_PATH = "/home/ubuntu/AI/ compression/models/llama-3.1-8b/llama-3.1-8b.safetensors"


def analyze_tensor_gpu(name, tensor_bf16, device):
    """Analyze a single weight tensor on GPU."""
    t = tensor_bf16.to(device)
    vals = t.float().ravel()
    n = vals.shape[0]

    stats = {"name": name, "shape": list(tensor_bf16.shape), "num_params": int(n)}

    # Basic stats
    stats["mean"] = float(vals.mean())
    stats["std"] = float(vals.std())
    stats["min"] = float(vals.min())
    stats["max"] = float(vals.max())
    std = stats["std"]

    # Peak fraction at multiple thresholds
    abs_vals = vals.abs()
    for frac in [0.1, 0.25, 0.5, 1.0]:
        threshold = frac * std
        stats[f"peak_pct_{frac}std"] = float((abs_vals < threshold).float().mean() * 100)

    # Unique BF16 values (use uint16 view)
    raw_uint16 = t.view(torch.int16).ravel()  # reinterpret bits

    # Subsample for clustering if large
    if n > 2_000_000:
        idx = torch.randperm(n, device=device)[:2_000_000]
        sample = vals[idx]
        sample_uint16 = raw_uint16[idx]
    else:
        sample = vals
        sample_uint16 = raw_uint16

    sample_n = sample.shape[0]

    # K-means for 15 tail centroids on GPU
    peak_mask = sample.abs() < (0.25 * std)
    tail_vals = sample[~peak_mask]

    if tail_vals.shape[0] > 100:
        # GPU k-means: iterative Lloyd's algorithm
        centroids = _gpu_kmeans(tail_vals, 15, max_iter=20)
    else:
        centroids = torch.linspace(stats["min"], stats["max"], 15, device=device)

    # 16 means: [0.0] + 15 tail centroids
    all_means = torch.cat([torch.zeros(1, device=device), centroids.sort()[0]])

    # Assign all samples to nearest mean
    dists = (sample.unsqueeze(1) - all_means.unsqueeze(0)).abs()
    assignments = dists.argmin(dim=1)

    # Per-group analysis
    group_stats = []
    total_escapes = 0

    for g in range(16):
        mask = assignments == g
        group_count = int(mask.sum())
        if group_count == 0:
            group_stats.append({
                "group": g, "mean": float(all_means[g]), "count": 0,
                "pct": 0.0, "unique": 0, "top127_coverage": 0.0, "escape_rate": 100.0
            })
            continue

        g_uint16 = sample_uint16[mask]
        # Unique counts on GPU via bincount-like approach
        g_unique = g_uint16.unique(return_counts=True)
        unique_vals, counts = g_unique
        n_unique = int(unique_vals.shape[0])

        # Top-127 coverage
        sorted_counts = counts.sort(descending=True)[0]
        top127 = int(sorted_counts[:127].sum())
        coverage = top127 / group_count
        escapes = group_count - top127
        total_escapes += escapes

        # Zipf ratio
        if sorted_counts.shape[0] > 1:
            median_c = float(sorted_counts.float().median())
            zipf_ratio = float(sorted_counts[0]) / median_c if median_c > 0 else float('inf')
        else:
            zipf_ratio = float('inf')

        group_stats.append({
            "group": g, "mean": float(all_means[g]), "count": group_count,
            "pct": float(group_count / sample_n * 100),
            "unique": n_unique,
            "top127_coverage": float(coverage * 100),
            "escape_rate": float((1 - coverage) * 100),
            "zipf_ratio": float(zipf_ratio),
        })

    stats["groups"] = group_stats
    stats["overall_escape_rate"] = float(total_escapes / sample_n * 100)
    stats["peak_group_pct"] = float((assignments == 0).float().mean() * 100)

    peak_frac = stats["peak_group_pct"] / 100
    escape_rate = stats["overall_escape_rate"] / 100
    base_bits = peak_frac * 8 + (1 - peak_frac) * 12
    effective_bits = base_bits + escape_rate * 16
    stats["base_bits_per_param"] = float(base_bits)
    stats["effective_bits_per_param"] = float(effective_bits)
    stats["compression_ratio"] = float(16 / effective_bits) if effective_bits > 0 else 0

    return stats


def _gpu_kmeans(data, k, max_iter=20):
    """Simple GPU k-means."""
    n = data.shape[0]
    # Init: random sample
    idx = torch.randperm(n, device=data.device)[:k]
    centroids = data[idx].clone()

    for _ in range(max_iter):
        dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        labels = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = data[mask].mean()
            else:
                new_centroids[j] = centroids[j]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


def worker(gpu_id, tensor_names, results_dict):
    """Worker process for one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    f = safe_open(MODEL_PATH, framework="pt")

    for name in tensor_names:
        t0 = time.time()
        tensor = f.get_tensor(name)
        if tensor.numel() < 10000:
            continue

        result = analyze_tensor_gpu(name, tensor, device)
        elapsed = time.time() - t0

        esc = result["overall_escape_rate"]
        bits = result["effective_bits_per_param"]
        peak = result["peak_group_pct"]
        cr = result["compression_ratio"]

        print(f"[GPU{gpu_id}] {name:<45s} "
              f"peak={peak:5.1f}%  esc={esc:5.2f}%  bits={bits:5.2f}  "
              f"CR={cr:.3f}x  ({elapsed:.1f}s)", flush=True)

        results_dict[name] = result

    torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("Turbo Lossless: GPU-Accelerated Weight Distribution Analysis")
    print("Model: Llama 3.1 8B (BF16 safetensors)")
    print(f"GPUs: 4x {torch.cuda.get_device_name(0)}")
    print("=" * 70, flush=True)

    f = safe_open(MODEL_PATH, framework="pt")
    all_names = list(f.keys())
    weight_names = [n for n in all_names if "weight" in n
                    and "norm" not in n
                    and "embd" not in n and "embed" not in n]

    print(f"\nAnalyzing {len(weight_names)} weight tensors across 4 GPUs...\n", flush=True)

    # Split tensors across 4 GPUs (round-robin for load balance)
    gpu_assignments = [[] for _ in range(4)]
    for i, name in enumerate(weight_names):
        gpu_assignments[i % 4].append(name)

    # Shared dict for results
    manager = mp.Manager()
    results_dict = manager.dict()

    t0 = time.time()

    # Launch 4 worker processes
    processes = []
    for gpu_id in range(4):
        p = mp.Process(target=worker, args=(gpu_id, gpu_assignments[gpu_id], results_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed_total = time.time() - t0

    # Collect results in original order
    all_results = []
    for name in weight_names:
        if name in results_dict:
            all_results.append(results_dict[name])

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY  (completed in {elapsed_total:.1f}s)")
    print(f"{'=' * 70}")

    esc_rates = [r["overall_escape_rate"] for r in all_results]
    peak_pcts = [r["peak_group_pct"] for r in all_results]
    bits_list = [r["effective_bits_per_param"] for r in all_results]
    cr_list = [r["compression_ratio"] for r in all_results]
    total_params = sum(r["num_params"] for r in all_results)

    # Weighted averages (by param count)
    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()
    w_esc = float(np.average(esc_rates, weights=weights))
    w_peak = float(np.average(peak_pcts, weights=weights))
    w_bits = float(np.average(bits_list, weights=weights))
    w_cr = float(16.0 / w_bits)

    print(f"\nTensors analyzed: {len(all_results)}")
    print(f"Total parameters: {total_params:,}")

    print(f"\nPeak fraction (near-zero group):")
    print(f"  Weighted avg: {w_peak:.1f}%")
    print(f"  Min:          {min(peak_pcts):.1f}%")
    print(f"  Max:          {max(peak_pcts):.1f}%")

    print(f"\nEscape rate:")
    print(f"  Weighted avg: {w_esc:.2f}%")
    print(f"  Min:          {min(esc_rates):.2f}%")
    print(f"  Max:          {max(esc_rates):.2f}%")
    print(f"  Median:       {float(np.median(esc_rates)):.2f}%")

    print(f"\nEffective bits/param:")
    print(f"  Weighted avg: {w_bits:.2f}")
    print(f"  Best:         {min(bits_list):.2f}")
    print(f"  Worst:        {max(bits_list):.2f}")

    print(f"\nCompression ratio (vs 16-bit BF16):")
    print(f"  Weighted avg: {w_cr:.3f}x")
    print(f"  Best:         {max(cr_list):.3f}x")
    print(f"  Worst:        {min(cr_list):.3f}x")

    # Per tensor-type breakdown
    print(f"\nBreakdown by tensor type:")
    type_groups = {}
    for r in all_results:
        name = r["name"]
        if "attn_q" in name:
            ttype = "attn_q"
        elif "attn_k" in name:
            ttype = "attn_k"
        elif "attn_v" in name:
            ttype = "attn_v"
        elif "attn_output" in name:
            ttype = "attn_output"
        elif "ffn_down" in name:
            ttype = "ffn_down"
        elif "ffn_gate" in name:
            ttype = "ffn_gate"
        elif "ffn_up" in name:
            ttype = "ffn_up"
        elif "output" in name:
            ttype = "output"
        else:
            ttype = "other"
        type_groups.setdefault(ttype, []).append(r)

    print(f"  {'Type':<15s} {'Peak%':>7s} {'Esc%':>7s} {'Bits':>7s} {'CR':>7s} {'Unique':>8s}")
    for ttype, rs in sorted(type_groups.items()):
        avg_peak = np.mean([r["peak_group_pct"] for r in rs])
        avg_esc = np.mean([r["overall_escape_rate"] for r in rs])
        avg_bits = np.mean([r["effective_bits_per_param"] for r in rs])
        avg_cr = np.mean([r["compression_ratio"] for r in rs])
        avg_unique = np.mean([r.get("unique_bf16_total", 0) for r in rs])
        print(f"  {ttype:<15s} {avg_peak:7.1f} {avg_esc:7.2f} {avg_bits:7.2f} {avg_cr:7.3f} {avg_unique:8.0f}")

    # Verdict
    print(f"\n{'=' * 70}")
    if w_esc < 2:
        print(f"VERDICT: FEASIBLE — Weighted escape rate {w_esc:.2f}% under 2% target")
    elif w_esc < 5:
        print(f"VERDICT: MARGINAL — Weighted escape rate {w_esc:.2f}% exceeds 2% but optimizable")
    elif w_esc < 10:
        print(f"VERDICT: RISKY — Weighted escape rate {w_esc:.2f}% significantly exceeds target")
    else:
        print(f"VERDICT: NOT VIABLE as designed — Weighted escape rate {w_esc:.2f}% far exceeds 2% target")
    print(f"{'=' * 70}")

    # Save
    out_path = "/home/ubuntu/AI/ compression/analysis_results.json"
    with open(out_path, "w") as fout:
        json.dump(all_results, fout, indent=2)
    print(f"\nDetailed results saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
