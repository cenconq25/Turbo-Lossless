"""
Novel compression approach exploration.
Tests fundamentally different ideas beyond tiered codebooks.

Ideas tested:
A. More tiers (8, 16, 32) — does finer granularity close the gap?
B. Spatial correlation — do adjacent weights correlate? (enables delta/predictive coding)
C. Warp-local codebook — fewer unique values in small blocks = shorter codes
D. Bit-plane decomposition — compress exponent and mantissa separately
E. Row-delta coding — XOR or subtract consecutive values, encode residuals
F. Frequency-sorted rank coding — assign ranks by frequency, encode ranks with Golomb/Rice
G. Hybrid: block-adaptive tier selection — different blocks use different tier configs
"""

import torch
import torch.multiprocessing as mp
from safetensors import safe_open
import numpy as np
import math
import json
import time

MODEL_PATH = "/home/ubuntu/AI/ compression/models/llama-3.1-8b/llama-3.1-8b.safetensors"


def shannon_entropy_uint16(data):
    """Compute Shannon entropy of uint16 data on GPU."""
    unique_vals, counts = data.unique(return_counts=True)
    probs = counts.float() / data.shape[0]
    return -float((probs * torch.log2(probs)).sum())


def analyze_novel(name, tensor_bf16, device):
    """Test all novel approaches on one tensor."""
    t = tensor_bf16.to(device)
    raw = t.view(torch.int16).ravel()
    n = raw.shape[0]
    vals = t.float().ravel()

    results = {"name": name, "num_params": int(n)}

    # Baseline: global entropy
    baseline_entropy = shannon_entropy_uint16(raw)
    results["baseline_entropy"] = baseline_entropy

    # ============================================================
    # A. MORE TIERS (extend the prefix-code approach to 8, 16, 32 tiers)
    # ============================================================
    unique_vals, counts = raw.unique(return_counts=True)
    sorted_idx = counts.argsort(descending=True)
    sc = counts[sorted_idx].cpu().numpy()
    total = int(n)

    more_tiers_results = {}
    for num_tiers in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
        # Optimize tier sizes via exhaustive search over power-of-2 sizes
        # For each tier: prefix = tier_index in unary-ish code
        # Use truncated binary for prefixes to minimize overhead
        best_eff = 999
        best_cfg = None

        if num_tiers <= 6:
            # Exhaustive search for small tier counts
            # Each tier gets 2^b entries, try b from 4 to 13
            # For speed, use a greedy approach: allocate bits to maximize coverage per bit
            pass

        # Greedy optimal: for N tiers, find optimal split points
        # Prefix scheme: first tier = 1b prefix, tiers 2..N-1 use ceil(log2(N)) bits, last = escape
        # Actually, let's use a clean prefix scheme:
        # Tier i (0-indexed): prefix = i bits of '1' then '0', except last tier = all 1s
        # Tier 0: '0' + index = 1 + B0 bits
        # Tier 1: '10' + index = 2 + B1 bits
        # Tier K-1: '1...1' + 16-bit raw = K + 16 bits (escape)

        # For simplicity, try all combinations of index sizes
        # Use dynamic programming: given sorted frequencies, find optimal break points

        # DP approach: split sorted values into num_tiers-1 coded tiers + 1 escape tier
        # For each tier, cost = count_in_tier * (prefix_len + ceil(log2(tier_size)))
        # Escape cost = remaining * (num_tiers + 16 - 1)

        # Simpler: try geometric progression of tier sizes
        # Tier i gets base * ratio^i entries
        best_eff_for_ntier = 999

        # Try various base sizes
        for log_base in range(4, 12):
            cumulative = 0
            total_bits = 0
            valid = True

            for tier_idx in range(num_tiers - 1):
                # Each tier can have different size
                # Try: each tier doubles in size from previous
                if tier_idx == 0:
                    tier_size = 2 ** log_base
                else:
                    tier_size = 2 ** log_base  # uniform tier sizes

                prefix_bits = tier_idx + 1  # unary prefix
                if prefix_bits > 8:  # cap prefix overhead
                    valid = False
                    break
                index_bits = log_base
                start = cumulative
                end = min(cumulative + tier_size, len(sc))
                tier_count = int(np.sum(sc[start:end])) if start < len(sc) else 0
                total_bits += tier_count * (prefix_bits + index_bits)
                cumulative += tier_size

            if not valid:
                continue

            # Escape tier
            esc_count = total - int(np.sum(sc[:cumulative])) if cumulative < len(sc) else 0
            if esc_count < 0:
                esc_count = 0
            esc_prefix = min(num_tiers, 8)
            total_bits += esc_count * (esc_prefix + 16)

            eff = total_bits / total
            if eff < best_eff_for_ntier:
                best_eff_for_ntier = eff

        # Also try optimized variable tier sizes
        # Greedy: for each tier, find the optimal index_bits
        for combo_id in range(20):  # try 20 random combos
            cumulative = 0
            total_bits = 0
            valid = True
            rng = np.random.RandomState(combo_id + 42)

            tier_bits_list = []
            for tier_idx in range(num_tiers - 1):
                # Random tier size between 5 and 12 bits
                b = rng.randint(5, 13)
                tier_bits_list.append(b)

            for tier_idx, b in enumerate(tier_bits_list):
                tier_size = 2 ** b
                prefix_bits = tier_idx + 1
                if prefix_bits > 8:
                    valid = False
                    break
                start = cumulative
                end = min(cumulative + tier_size, len(sc))
                tier_count = int(np.sum(sc[start:end])) if start < len(sc) else 0
                total_bits += tier_count * (prefix_bits + b)
                cumulative += tier_size

            if not valid:
                continue

            esc_count = total - int(np.sum(sc[:cumulative])) if cumulative < len(sc) else 0
            if esc_count < 0:
                esc_count = 0
            esc_prefix = min(num_tiers, 8)
            total_bits += esc_count * (esc_prefix + 16)
            eff = total_bits / total
            if eff < best_eff_for_ntier:
                best_eff_for_ntier = eff

        more_tiers_results[num_tiers] = best_eff_for_ntier

    results["more_tiers"] = {str(k): v for k, v in more_tiers_results.items()}

    # ============================================================
    # B. SPATIAL CORRELATION — test if adjacent weights correlate
    # ============================================================
    # Reshape to 2D matrix and check row/col correlation
    shape = tensor_bf16.shape
    if len(shape) == 2:
        # Row-wise: entropy of deltas within each row
        row_deltas = raw.view(shape[0], shape[1])[:, 1:] ^ raw.view(shape[0], shape[1])[:, :-1]
        delta_entropy = shannon_entropy_uint16(row_deltas.ravel())

        # Col-wise
        col_deltas = raw.view(shape[0], shape[1])[1:, :] ^ raw.view(shape[0], shape[1])[:-1, :]
        col_delta_entropy = shannon_entropy_uint16(col_deltas.ravel())

        # Subtraction delta (arithmetic, not XOR)
        row_sub = raw.view(shape[0], shape[1])[:, 1:].short() - raw.view(shape[0], shape[1])[:, :-1].short()
        sub_entropy = shannon_entropy_uint16(row_sub.ravel())
    else:
        # 1D: just sequential delta
        delta_entropy = shannon_entropy_uint16((raw[1:] ^ raw[:-1]))
        col_delta_entropy = delta_entropy
        sub_entropy = shannon_entropy_uint16((raw[1:].short() - raw[:-1].short()))

    results["xor_row_delta_entropy"] = delta_entropy
    results["xor_col_delta_entropy"] = col_delta_entropy
    results["sub_row_delta_entropy"] = sub_entropy

    # ============================================================
    # C. WARP-LOCAL CODEBOOK — how many unique values in blocks?
    # ============================================================
    block_results = {}
    for block_size in [32, 64, 128, 256, 512]:
        n_blocks = n // block_size
        if n_blocks == 0:
            continue

        blocked = raw[:n_blocks * block_size].view(n_blocks, block_size)

        # Count unique values per block
        uniques_per_block = torch.zeros(n_blocks, device=device)
        for i in range(min(n_blocks, 1000)):  # sample 1000 blocks
            uniques_per_block[i] = blocked[i].unique().shape[0]

        sampled = min(n_blocks, 1000)
        avg_unique = float(uniques_per_block[:sampled].mean())
        max_unique = float(uniques_per_block[:sampled].max())

        # Compute effective bits: header (list of unique vals) + per-value local index
        # Header: avg_unique * 16 bits (raw BF16 values) or avg_unique * 13 bits (global codebook index)
        # Data: block_size * ceil(log2(avg_unique)) bits
        local_idx_bits = math.ceil(math.log2(max(avg_unique, 2)))
        header_bits_raw = avg_unique * 16  # store raw values
        header_bits_global = avg_unique * 13  # index into global 8192-entry codebook
        data_bits = block_size * local_idx_bits

        eff_raw = (header_bits_raw + data_bits) / block_size
        eff_global = (header_bits_global + data_bits) / block_size

        block_results[block_size] = {
            "avg_unique": avg_unique,
            "max_unique": max_unique,
            "local_idx_bits": local_idx_bits,
            "eff_bits_raw_header": eff_raw,
            "eff_bits_global_header": eff_global,
        }

    results["block_local"] = {str(k): v for k, v in block_results.items()}

    # ============================================================
    # D. BIT-PLANE DECOMPOSITION — compress sign, exponent, mantissa separately
    # ============================================================
    raw_uint = raw.view(torch.int16)
    # BF16: 1 sign + 8 exponent + 7 mantissa
    sign_bits = (raw_uint >> 15) & 1
    exp_bits = (raw_uint >> 7) & 0xFF
    mant_bits = raw_uint & 0x7F

    sign_entropy = shannon_entropy_uint16(sign_bits)
    exp_entropy = shannon_entropy_uint16(exp_bits)
    mant_entropy = shannon_entropy_uint16(mant_bits)
    bitplane_total = sign_entropy + exp_entropy + mant_entropy

    results["bitplane"] = {
        "sign_entropy": sign_entropy,
        "exp_entropy": exp_entropy,
        "mant_entropy": mant_entropy,
        "total": bitplane_total,
    }

    # ============================================================
    # E. SORTED-ORDER CODING — sort values within block, encode diffs
    # ============================================================
    # If we sort values within a block, the sorted deltas have very low entropy
    block_size = 256
    n_blocks = n // block_size
    if n_blocks > 0:
        blocked = raw[:n_blocks * block_size].view(n_blocks, block_size)
        # Sample blocks
        sample_blocks = min(n_blocks, 500)
        sorted_blocked = blocked[:sample_blocks].sort(dim=1)[0]
        sorted_deltas = sorted_blocked[:, 1:] - sorted_blocked[:, :-1]
        sorted_delta_entropy = shannon_entropy_uint16(sorted_deltas.ravel())
        # Effective: need permutation index (log2(block_size) * block_size bits) + sorted delta codes
        # Permutation is expensive... but if we DON'T need to preserve order (we do for weights)
        # This doesn't work for weights where position matters. Skip.
        results["sorted_delta_entropy"] = sorted_delta_entropy
    else:
        results["sorted_delta_entropy"] = 0

    # ============================================================
    # F. RICE/GOLOMB CODING on frequency ranks
    # ============================================================
    # Map each BF16 value to its frequency rank, then Rice-code the ranks
    # Ranks follow roughly Zipf distribution → Rice coding is near-optimal
    val_to_rank = torch.zeros(65536, dtype=torch.int32, device=device)
    sorted_vals = unique_vals[sorted_idx]
    # Handle negative int16 by offsetting
    val_to_rank[(sorted_vals.long() + 32768) % 65536] = torch.arange(len(sorted_vals), device=device, dtype=torch.int32)
    ranks = val_to_rank[(raw.long() + 32768) % 65536]

    # Rice parameter optimization
    best_rice = 999
    best_rice_m = 0
    for m in range(1, 16):
        # Rice coding: quotient in unary + remainder in m bits
        # Cost = floor(rank / 2^m) + 1 + m per value
        quotients = (ranks >> m).float()
        avg_cost = float(quotients.mean()) + 1 + m
        if avg_cost < best_rice:
            best_rice = avg_cost
            best_rice_m = m

    results["rice_coding"] = {
        "best_m": best_rice_m,
        "effective_bits": best_rice,
    }

    # ============================================================
    # G. HYBRID: block-adaptive tier selection
    # ============================================================
    # Each block of 256 values picks the best tier config from a small menu
    # Menu overhead: negligible (2-3 bits per block to select config)
    block_size = 256
    n_blocks = n // block_size
    if n_blocks > 0:
        blocked_uint16 = raw[:n_blocks * block_size].view(n_blocks, block_size)

        # For each block, compute the best fixed-width code
        sample_blocks = min(n_blocks, 500)
        block_bits_adaptive = []

        for bi in range(sample_blocks):
            block_data = blocked_uint16[bi]
            b_unique, b_counts = block_data.unique(return_counts=True)
            b_n = block_size
            b_nuniq = b_unique.shape[0]

            if b_nuniq <= 16:
                # 4-bit index + 16-entry local codebook
                header = b_nuniq * 16  # raw values
                data = b_n * 4
                block_bits_adaptive.append((header + data) / b_n)
            elif b_nuniq <= 32:
                header = b_nuniq * 16
                data = b_n * 5
                block_bits_adaptive.append((header + data) / b_n)
            elif b_nuniq <= 64:
                header = b_nuniq * 16
                data = b_n * 6
                block_bits_adaptive.append((header + data) / b_n)
            elif b_nuniq <= 128:
                header = b_nuniq * 16
                data = b_n * 7
                block_bits_adaptive.append((header + data) / b_n)
            else:
                block_bits_adaptive.append(16)  # just store raw

        avg_adaptive = float(np.mean(block_bits_adaptive))
        results["block_adaptive"] = {
            "avg_bits": avg_adaptive,
            "block_size": block_size,
        }
    else:
        results["block_adaptive"] = {"avg_bits": 16, "block_size": 256}

    # ============================================================
    # H. GLOBAL CODEBOOK + BYTE-ALIGNED VARIABLE LENGTH
    # ============================================================
    # What if we use: 1 byte (8 bits) for top-256, 2 bytes (16 bits) for rest?
    # Simple, GPU-friendly (byte-aligned), no bit-packing needed
    top256_count = int(np.sum(sc[:256]))
    rest_count = total - top256_count
    byte_aligned_eff = (top256_count * 8 + rest_count * 16) / total
    results["byte_aligned_256"] = byte_aligned_eff

    # Top-256 in 1 byte, next-256 in 9 bits (1-bit prefix + 8-bit idx), rest in 17 bits
    top256 = int(np.sum(sc[:256]))
    next256 = int(np.sum(sc[256:512])) if len(sc) > 256 else 0
    rest2 = total - top256 - next256
    nibble_eff = (top256 * 9 + next256 * 10 + rest2 * 18) / total
    results["three_level_byte"] = nibble_eff

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
        result = analyze_novel(name, tensor, device)
        elapsed = time.time() - t0
        results_dict[name] = result

        ent = result["baseline_entropy"]
        xor_d = result["xor_row_delta_entropy"]
        bp = result["bitplane"]["total"]
        rice = result["rice_coding"]["effective_bits"]
        print(f"[GPU{gpu_id}] {name:<40s} "
              f"entropy={ent:.2f}  xor_delta={xor_d:.2f}  bitplane={bp:.2f}  "
              f"rice={rice:.2f}  ({elapsed:.1f}s)", flush=True)
    torch.cuda.empty_cache()


def main():
    print("=" * 90)
    print("Turbo Lossless: Novel Compression Approach Exploration")
    print("=" * 90, flush=True)

    f = safe_open(MODEL_PATH, framework="pt")
    all_names = list(f.keys())
    weight_names = [n for n in all_names if "weight" in n
                    and "norm" not in n
                    and "embd" not in n and "embed" not in n]

    # Sample across layers
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

    print(f"Testing {len(sampled)} tensors\n", flush=True)

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

    weights = np.array([r["num_params"] for r in all_results], dtype=np.float64)
    weights /= weights.sum()

    print(f"\n{'=' * 90}")
    print(f"RESULTS ({len(all_results)} tensors, {elapsed:.1f}s)")
    print(f"{'=' * 90}")

    # Baseline
    w_ent = np.average([r["baseline_entropy"] for r in all_results], weights=weights)
    print(f"\nBaseline Shannon entropy: {w_ent:.3f} bits/param ({16/w_ent:.3f}x)")

    # A. More tiers
    print(f"\n--- A. More Tiers (prefix-coded) ---")
    for nt in ["2", "3", "4", "6", "8", "12", "16", "24", "32"]:
        vals = [r["more_tiers"].get(nt, 99) for r in all_results]
        avg = np.average(vals, weights=weights)
        print(f"  {nt:>2s} tiers: {avg:.3f} bits/param ({16/avg:.3f}x)")

    # B. Spatial correlation
    print(f"\n--- B. Spatial Correlation (delta coding entropy) ---")
    w_xor_row = np.average([r["xor_row_delta_entropy"] for r in all_results], weights=weights)
    w_xor_col = np.average([r["xor_col_delta_entropy"] for r in all_results], weights=weights)
    w_sub_row = np.average([r["sub_row_delta_entropy"] for r in all_results], weights=weights)
    print(f"  XOR row-delta entropy:  {w_xor_row:.3f} bits ({16/w_xor_row:.3f}x)")
    print(f"  XOR col-delta entropy:  {w_xor_col:.3f} bits ({16/w_xor_col:.3f}x)")
    print(f"  SUB row-delta entropy:  {w_sub_row:.3f} bits ({16/w_sub_row:.3f}x)")
    print(f"  (Baseline entropy:      {w_ent:.3f} bits)")
    if w_xor_row < w_ent:
        print(f"  ** XOR delta IMPROVES entropy by {w_ent - w_xor_row:.3f} bits! **")
    else:
        print(f"  No spatial correlation advantage (delta entropy >= baseline)")

    # C. Block-local codebook
    print(f"\n--- C. Warp/Block-Local Codebook ---")
    for bs in ["32", "64", "128", "256", "512"]:
        vals_raw = [r["block_local"].get(bs, {}).get("eff_bits_raw_header", 99) for r in all_results]
        vals_glob = [r["block_local"].get(bs, {}).get("eff_bits_global_header", 99) for r in all_results]
        uniq = [r["block_local"].get(bs, {}).get("avg_unique", 0) for r in all_results]
        avg_raw = np.average(vals_raw, weights=weights)
        avg_glob = np.average(vals_glob, weights=weights)
        avg_uniq = np.average(uniq, weights=weights)
        print(f"  Block-{bs:>3s}: avg_unique={avg_uniq:.0f}  "
              f"raw_hdr={avg_raw:.2f}b  global_hdr={avg_glob:.2f}b")

    # D. Bit-plane
    print(f"\n--- D. Bit-Plane Decomposition ---")
    w_sign = np.average([r["bitplane"]["sign_entropy"] for r in all_results], weights=weights)
    w_exp = np.average([r["bitplane"]["exp_entropy"] for r in all_results], weights=weights)
    w_mant = np.average([r["bitplane"]["mant_entropy"] for r in all_results], weights=weights)
    w_bp = np.average([r["bitplane"]["total"] for r in all_results], weights=weights)
    print(f"  Sign entropy:     {w_sign:.3f} bits (of 1 bit)")
    print(f"  Exponent entropy: {w_exp:.3f} bits (of 8 bits)")
    print(f"  Mantissa entropy: {w_mant:.3f} bits (of 7 bits)")
    print(f"  Total:            {w_bp:.3f} bits ({16/w_bp:.3f}x)")
    print(f"  (vs joint entropy: {w_ent:.3f} bits)")
    if w_bp > w_ent:
        print(f"  ** Decomposition LOSES {w_bp - w_ent:.3f} bits due to cross-field correlation **")

    # E. Rice coding on frequency ranks
    print(f"\n--- F. Rice/Golomb Coding on Frequency Ranks ---")
    w_rice = np.average([r["rice_coding"]["effective_bits"] for r in all_results], weights=weights)
    print(f"  Best Rice coding: {w_rice:.3f} bits/param ({16/w_rice:.3f}x)")

    # G. Block-adaptive
    print(f"\n--- G. Block-Adaptive Local Codebook ---")
    w_adapt = np.average([r["block_adaptive"]["avg_bits"] for r in all_results], weights=weights)
    print(f"  Adaptive (256-block): {w_adapt:.3f} bits/param ({16/w_adapt:.3f}x)")

    # H. Byte-aligned
    print(f"\n--- H. Byte-Aligned Schemes ---")
    w_byte = np.average([r["byte_aligned_256"] for r in all_results], weights=weights)
    print(f"  1-byte/2-byte (256 split): {w_byte:.3f} bits/param ({16/w_byte:.3f}x)")

    # ============================================================
    # RANKING
    # ============================================================
    print(f"\n{'=' * 90}")
    print(f"FINAL RANKING — All approaches")
    print(f"{'=' * 90}")

    approaches = [
        ("Shannon entropy (limit)", w_ent),
        ("XOR row-delta + entropy coding", w_xor_row),
        ("XOR col-delta + entropy coding", w_xor_col),
        ("SUB row-delta + entropy coding", w_sub_row),
        ("Bit-plane decomposition", w_bp),
        ("Rice/Golomb rank coding", w_rice),
        ("Block-adaptive local codebook", w_adapt),
        ("Byte-aligned 256", w_byte),
    ]

    # Add tier results
    for nt in ["2", "3", "4", "6", "8", "12", "16", "24", "32"]:
        vals = [r["more_tiers"].get(nt, 99) for r in all_results]
        avg = np.average(vals, weights=weights)
        approaches.append((f"{nt}-tier prefix code", avg))

    # Add block-local results
    for bs in ["32", "64", "128", "256", "512"]:
        vals = [r["block_local"].get(bs, {}).get("eff_bits_global_header", 99) for r in all_results]
        avg = np.average(vals, weights=weights)
        approaches.append((f"Block-{bs} local codebook", avg))

    # Sort by bits
    approaches.sort(key=lambda x: x[1])

    print(f"\n  {'Rank':>4s}  {'Approach':<45s} {'bits/param':>10s} {'CR':>7s} {'GPU-friendly':>13s}")
    print(f"  {'-'*4}  {'-'*45} {'-'*10} {'-'*7} {'-'*13}")

    gpu_friendly = {
        "Shannon entropy (limit)": "N/A",
        "4-tier prefix code": "YES",
        "3-tier prefix code": "YES",
        "6-tier prefix code": "YES",
        "8-tier prefix code": "YES",
        "Rice/Golomb rank coding": "Moderate",
        "Byte-aligned 256": "VERY",
        "XOR row-delta + entropy coding": "Moderate",
        "Bit-plane decomposition": "YES",
    }

    for rank, (name, bits) in enumerate(approaches, 1):
        cr = 16 / bits if bits > 0 else 0
        gpu = gpu_friendly.get(name, "Possible")
        print(f"  {rank:>4d}  {name:<45s} {bits:>10.3f} {cr:>7.3f}x {gpu:>13s}")

    # Save
    out_path = "/home/ubuntu/AI/ compression/novel_results.json"
    with open(out_path, "w") as fout:
        json.dump(all_results, fout, indent=2, default=str)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
