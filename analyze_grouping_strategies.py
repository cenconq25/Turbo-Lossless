"""
Test different grouping strategies to minimize escape rate.
Runs on GPU x4 in parallel.

Strategies tested:
1. Original: 1 peak (at 0) + 15 tail k-means (current CLAUDE.md design)
2. Uniform k-means: 16 groups, all via k-means (no fixed zero)
3. 32 groups k-means
4. 64 groups k-means
5. 128 groups k-means
6. 256 groups k-means
7. Density-aware: more groups in dense center, fewer at tails
8. Per-percentile: equal-population groups (each group gets ~same # of values)
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import json
import time
import sys

MODEL_PATH = "/home/ubuntu/AI/ compression/models/llama-3.1-8b/llama-3.1-8b.safetensors"

# Sub-dictionary sizes to test per strategy
LUT_SIZES = [127, 255, 511, 1023]


def gpu_kmeans(data, k, max_iter=30):
    """GPU k-means."""
    n = data.shape[0]
    idx = torch.randperm(n, device=data.device)[:k]
    centroids = data[idx].clone()
    for _ in range(max_iter):
        # Chunk the distance computation to avoid OOM
        chunk_size = 500_000
        labels = torch.empty(n, dtype=torch.long, device=data.device)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = (data[start:end].unsqueeze(1) - centroids.unsqueeze(0)).abs()
            labels[start:end] = dists.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = data[mask].mean()
            else:
                new_centroids[j] = centroids[j]
        if torch.allclose(centroids, new_centroids, atol=1e-7):
            break
        centroids = new_centroids
    return centroids, labels


def compute_escape_rate(sample_uint16, labels, k, lut_size):
    """Compute escape rate for given assignments and LUT size."""
    total = sample_uint16.shape[0]
    covered = 0
    for g in range(k):
        mask = labels == g
        if not mask.any():
            continue
        g_uint16 = sample_uint16[mask]
        _, counts = g_uint16.unique(return_counts=True)
        sorted_counts = counts.sort(descending=True)[0]
        covered += int(sorted_counts[:lut_size].sum())
    escape_rate = (total - covered) / total
    return escape_rate


def equal_population_groups(sample, k):
    """Split into k groups with roughly equal population (percentile-based)."""
    sorted_vals, sorted_idx = sample.sort()
    n = sample.shape[0]
    chunk = n // k
    centroids = []
    for i in range(k):
        start = i * chunk
        end = (i + 1) * chunk if i < k - 1 else n
        centroids.append(sorted_vals[start:end].mean())
    centroids = torch.stack(centroids)
    # Assign
    dists = (sample.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    labels = dists.argmin(dim=1)
    return centroids, labels


def density_aware_groups(sample, k):
    """More groups near center (dense), fewer at tails."""
    std = sample.std().item()
    # Allocate: 60% of groups to inner 1-std region, 40% to outer
    n_inner = max(int(k * 0.6), 1)
    n_outer = k - n_inner
    inner_mask = sample.abs() < std
    outer_mask = ~inner_mask

    inner_vals = sample[inner_mask]
    outer_vals = sample[outer_mask]

    # K-means on each region
    if inner_vals.shape[0] > n_inner and n_inner > 0:
        inner_centroids, _ = gpu_kmeans(inner_vals, n_inner, max_iter=20)
    else:
        inner_centroids = torch.linspace(-std, std, n_inner, device=sample.device)

    if outer_vals.shape[0] > n_outer and n_outer > 0:
        outer_centroids, _ = gpu_kmeans(outer_vals, n_outer, max_iter=20)
    else:
        mn, mx = sample.min().item(), sample.max().item()
        outer_centroids = torch.linspace(mn, mx, n_outer, device=sample.device)

    centroids = torch.cat([inner_centroids, outer_centroids])
    # Assign all values
    dists = (sample.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    labels = dists.argmin(dim=1)
    return centroids, labels


def analyze_tensor_strategies(name, tensor_bf16, device):
    """Test all grouping strategies on one tensor."""
    t = tensor_bf16.to(device)
    vals = t.float().ravel()
    raw_uint16 = t.view(torch.int16).ravel()
    n = vals.shape[0]

    # Subsample
    if n > 2_000_000:
        idx = torch.randperm(n, device=device)[:2_000_000]
        sample = vals[idx]
        sample_uint16 = raw_uint16[idx]
    else:
        sample = vals
        sample_uint16 = raw_uint16

    results = {"name": name, "num_params": int(n)}
    strategy_results = []

    # --- Strategy 1: Original (1 peak + 15 tail) ---
    std = sample.std().item()
    peak_mask = sample.abs() < (0.25 * std)
    tail_vals = sample[~peak_mask]
    if tail_vals.shape[0] > 15:
        tail_c, _ = gpu_kmeans(tail_vals, 15, max_iter=20)
    else:
        tail_c = torch.linspace(sample.min().item(), sample.max().item(), 15, device=device)
    all_means = torch.cat([torch.zeros(1, device=device), tail_c.sort()[0]])
    dists = (sample.unsqueeze(1) - all_means.unsqueeze(0)).abs()
    labels = dists.argmin(dim=1)
    for lut in LUT_SIZES:
        esc = compute_escape_rate(sample_uint16, labels, 16, lut)
        strategy_results.append({
            "strategy": "original_1+15", "k": 16, "lut_size": lut,
            "escape_rate": float(esc * 100)
        })

    # --- Strategy 2-6: Uniform k-means with varying k ---
    for k in [16, 32, 64, 128, 256]:
        centroids, labels = gpu_kmeans(sample, k, max_iter=20)
        for lut in LUT_SIZES:
            esc = compute_escape_rate(sample_uint16, labels, k, lut)
            strategy_results.append({
                "strategy": f"kmeans_{k}", "k": k, "lut_size": lut,
                "escape_rate": float(esc * 100)
            })

    # --- Strategy 7: Density-aware ---
    for k in [16, 32, 64, 128, 256]:
        centroids, labels = density_aware_groups(sample, k)
        for lut in LUT_SIZES:
            esc = compute_escape_rate(sample_uint16, labels, k, lut)
            strategy_results.append({
                "strategy": f"density_{k}", "k": k, "lut_size": lut,
                "escape_rate": float(esc * 100)
            })

    # --- Strategy 8: Equal-population ---
    for k in [16, 32, 64, 128, 256]:
        centroids, labels = equal_population_groups(sample, k)
        for lut in LUT_SIZES:
            esc = compute_escape_rate(sample_uint16, labels, k, lut)
            strategy_results.append({
                "strategy": f"equalpop_{k}", "k": k, "lut_size": lut,
                "escape_rate": float(esc * 100)
            })

    results["strategies"] = strategy_results
    return results


def worker(gpu_id, tensor_names, results_dict):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    f = safe_open(MODEL_PATH, framework="pt")

    for name in tensor_names:
        t0 = time.time()
        tensor = f.get_tensor(name)
        if tensor.numel() < 10000:
            continue
        result = analyze_tensor_strategies(name, tensor, device)
        elapsed = time.time() - t0
        results_dict[name] = result
        # Print best result for this tensor
        best = min(result["strategies"], key=lambda x: x["escape_rate"])
        print(f"[GPU{gpu_id}] {name:<40s} best: {best['strategy']:<15s} "
              f"k={best['k']:<4d} lut={best['lut_size']:<5d} "
              f"esc={best['escape_rate']:5.2f}%  ({elapsed:.1f}s)", flush=True)
    torch.cuda.empty_cache()


def main():
    print("=" * 80)
    print("Turbo Lossless: Grouping Strategy Comparison")
    print("Testing: k=[16,32,64,128,256] x LUT=[127,255,511,1023] x 4 strategies")
    print("=" * 80, flush=True)

    f = safe_open(MODEL_PATH, framework="pt")
    all_names = list(f.keys())
    # Sample 1 tensor per type per layer-range for speed (28 tensors)
    weight_names = [n for n in all_names if "weight" in n
                    and "norm" not in n
                    and "embd" not in n and "embed" not in n]

    # Sample: layers 0, 7, 15, 23, 31 x all 7 types + output = ~36 tensors
    sample_layers = {0, 7, 15, 23, 31}
    sampled = []
    for n in weight_names:
        if "output.weight" == n:
            sampled.append(n)
            continue
        for layer in sample_layers:
            if f"blk.{layer}." in n:
                sampled.append(n)
                break

    print(f"\nSampling {len(sampled)} tensors across layers {sample_layers}\n", flush=True)

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

    # Aggregate
    all_results = [results_dict[n] for n in sampled if n in results_dict]

    # Build summary table: for each (strategy, k, lut_size), average escape rate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        for s in r["strategies"]:
            key = (s["strategy"], s["k"], s["lut_size"])
            agg[key].append(s["escape_rate"])

    print(f"\n{'=' * 80}")
    print(f"SUMMARY — Average escape rate across {len(all_results)} tensors  ({elapsed:.1f}s)")
    print(f"{'=' * 80}")

    # Compute effective bits for each config
    # For simplicity assume peak_frac ~ 15% (from prior analysis)
    # base_bits depends on encoding scheme which depends on k and lut_size
    # For now just show escape rates — the key metric

    print(f"\n{'Strategy':<20s} {'k':>4s} {'LUT':>5s} {'Esc%':>7s} {'Unique covered':>15s}")
    print("-" * 55)

    sorted_keys = sorted(agg.keys(), key=lambda x: np.mean(agg[x]))
    for key in sorted_keys:
        strat, k, lut = key
        avg_esc = np.mean(agg[key])
        print(f"{strat:<20s} {k:>4d} {lut:>5d} {avg_esc:>7.2f}%")

    # Top 10
    print(f"\n{'=' * 80}")
    print("TOP 10 CONFIGURATIONS (lowest escape rate)")
    print(f"{'=' * 80}")
    for i, key in enumerate(sorted_keys[:10]):
        strat, k, lut = key
        avg_esc = np.mean(agg[key])
        # Estimate bits: need to encode group_id + lut_index + escapes
        # group_id bits = ceil(log2(k))
        import math
        group_bits = math.ceil(math.log2(k)) if k > 1 else 0
        lut_bits = math.ceil(math.log2(lut + 1))  # +1 for escape code
        code_bits = group_bits + lut_bits
        esc_rate = avg_esc / 100
        effective_bits = (1 - esc_rate) * code_bits + esc_rate * (code_bits + 16)
        cr = 16 / effective_bits if effective_bits > 0 else 0
        print(f"  {i+1:2d}. {strat:<20s} k={k:<4d} lut={lut:<5d} "
              f"esc={avg_esc:5.2f}%  code={code_bits}b  "
              f"eff={effective_bits:.2f}b  CR={cr:.3f}x")

    # Save
    out_path = "/home/ubuntu/AI/ compression/grouping_results.json"
    with open(out_path, "w") as fout:
        json.dump(all_results, fout, indent=2)
    print(f"\nDetailed results saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
