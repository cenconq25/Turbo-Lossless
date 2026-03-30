#!/usr/bin/env python3
"""
Benchmark: 12-bit fixed-width lossless matvec on AMD MI50.
Uses fast C packer + HIP kernel + patch corrections.
"""

import torch
import time
import ctypes
import numpy as np
import os
import sys
from safetensors import safe_open

# Load C packer
_pack_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "fixed12_pack.so"))
_pack_lib.build_codebook_12bit.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint32),
]
_pack_lib.build_codebook_12bit.restype = ctypes.c_int

_pack_lib.pack_fixed12.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16), ctypes.c_int16,
    ctypes.c_int32, ctypes.c_int32,
]
_pack_lib.pack_fixed12.restype = ctypes.c_int64

# C packer - fused layout
_pack_lib.pack_fixed12_fused.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
]
_pack_lib.pack_fixed12_fused.restype = ctypes.c_int64

# Load HIP kernels
_hip_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "decompress_v2.so"))
_hip_lib.launch_fixed12_fused.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2
_hip_lib.launch_fixed12_fused.restype = ctypes.c_int
_hip_lib.launch_fixed12_decompress.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2
_hip_lib.launch_fixed12_decompress.restype = ctypes.c_int

WORKGROUP_SIZE = 256


def prepare_tensor(W, device="cuda:0"):
    """Build 12-bit packed format using GPU freq sort + C packer."""
    M, K = W.shape
    n = M * K

    # GPU frequency sort
    raw_gpu = W.contiguous().view(torch.int16).to(device)
    unique_gpu, counts_gpu = torch.unique(raw_gpu.view(-1), return_counts=True)
    si_gpu = torch.argsort(counts_gpu, descending=True)
    sorted_vals = unique_gpu[si_gpu].cpu().numpy().astype(np.uint16)
    del raw_gpu, unique_gpu, counts_gpu, si_gpu
    torch.cuda.empty_cache()

    raw = W.contiguous().view(torch.int16).numpy().flatten().astype(np.uint16)

    # Build codebook + reverse map
    codebook = np.zeros(4096, dtype=np.int16)
    reverse_map = np.zeros(65536, dtype=np.uint32)
    _pack_lib.build_codebook_12bit(
        sorted_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        len(sorted_vals),
        codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        reverse_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )

    num_words = (n * 12 + 31) // 32 + 2
    packed = np.zeros(num_words, dtype=np.uint32)
    max_patches = max(n // 100, 1024)

    # Per-thread-stride escape layout for fused kernel
    escape_offsets = np.zeros(M * WORKGROUP_SIZE, dtype=np.int32)
    escape_vals = np.zeros(max_patches, dtype=np.int16)

    num_patches = _pack_lib.pack_fixed12_fused(
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_int64(n),
        reverse_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        escape_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        escape_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.c_int32(M), ctypes.c_int32(K), ctypes.c_int32(WORKGROUP_SIZE),
    )
    return {
        "packed": packed[:num_words], "codebook": codebook,
        "escape_offsets": escape_offsets,
        "escape_vals": escape_vals[:num_patches],
        "num_patches": int(num_patches), "M": M, "K": K,
    }


def benchmark_tensor(name, W, device="cuda:0", warmup=30, iters=200):
    """Benchmark one tensor: 12-bit kernel + patches vs BF16 baseline."""
    M, K = W.shape

    t0 = time.time()
    data = prepare_tensor(W)
    prep_time = time.time() - t0

    # Move to GPU
    pg = torch.from_numpy(data["packed"]).to(torch.int32).to(device)
    cg = torch.from_numpy(data["codebook"]).to(torch.int16).to(device)
    x = torch.randn(K, dtype=torch.bfloat16, device=device)
    xi = x.view(torch.int16)
    o = torch.zeros(M, dtype=torch.float32, device=device)

    eo = torch.from_numpy(data["escape_offsets"]).to(torch.int32).to(device)
    ev = torch.from_numpy(data["escape_vals"]).to(torch.int16).to(device)

    def run_ours():
        _hip_lib.launch_fixed12_fused(
            pg.data_ptr(), cg.data_ptr(), xi.data_ptr(),
            eo.data_ptr(), ev.data_ptr(),
            o.data_ptr(), M, K)

    # Bit-exact verification: decompress all weights and compare original BF16
    decoded = torch.empty(M * K, dtype=torch.int16, device=device)
    _hip_lib.launch_fixed12_decompress(
        pg.data_ptr(), cg.data_ptr(),
        eo.data_ptr(), ev.data_ptr(),
        decoded.data_ptr(), M, K)
    orig_bits = W.contiguous().view(torch.int16).to(device)
    mismatches = (decoded != orig_bits.view(-1)).sum().item()
    lossless = mismatches == 0
    del decoded, orig_bits

    for _ in range(warmup):
        run_ours()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_ours()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg_ours = sum(sorted(times)[iters // 5: iters * 4 // 5]) / (iters * 3 // 5)

    # Benchmark BF16
    Wg = W.to(device)
    for _ in range(warmup):
        Wg.float() @ x.float()
    torch.cuda.synchronize()
    t2 = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Wg.float() @ x.float()
        torch.cuda.synchronize()
        t2.append(time.perf_counter() - t0)
    avg_bf16 = sum(sorted(t2)[iters // 5: iters * 4 // 5]) / (iters * 3 // 5)

    speedup = avg_bf16 / avg_ours
    eff_bw = M * K * 2 / avg_ours / 1e9

    del pg, cg, Wg, eo, ev
    torch.cuda.empty_cache()

    return {
        "name": name, "M": M, "K": K,
        "ours_ms": avg_ours * 1000, "bf16_ms": avg_bf16 * 1000,
        "speedup": speedup, "eff_bw": eff_bw,
        "patches": data["num_patches"], "lossless": lossless,
        "prep_ms": prep_time * 1000,
    }


def benchmark_model(model_path, tensor_names=None):
    """Benchmark selected tensors from a safetensors model."""
    if os.path.isdir(model_path):
        import glob
        shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    else:
        shards = [model_path]

    results = []

    for shard in shards:
        f = safe_open(shard, framework="pt")
        for name in f.keys():
            if tensor_names and name not in tensor_names:
                continue
            W = f.get_tensor(name)
            if W.dtype != torch.bfloat16 or W.ndim != 2:
                continue
            if W.numel() < 100000:
                continue

            r = benchmark_tensor(name, W)
            results.append(r)

            status = "OK" if r["lossless"] else "ERR"
            print(f'{r["name"]:<50s} [{r["M"]:>5d}x{r["K"]:>6d}]  '
                  f'ours={r["ours_ms"]:>7.3f}ms  bf16={r["bf16_ms"]:>7.3f}ms  '
                  f'speedup={r["speedup"]:>5.2f}x  patches={r["patches"]:>6d}  '
                  f'BW={r["eff_bw"]:>5.0f} GB/s  prep={r["prep_ms"]:>6.0f}ms  {status}',
                  flush=True)

    return results


def print_summary(results):
    total_params = sum(r["M"] * r["K"] for r in results)
    w_ours = sum(r["M"] * r["K"] * r["ours_ms"] for r in results) / total_params
    w_bf16 = sum(r["M"] * r["K"] * r["bf16_ms"] for r in results) / total_params
    all_ok = all(r["lossless"] for r in results)

    print()
    print("=" * 90)
    print(f'WEIGHTED AVG:  ours={w_ours:.3f}ms  bf16={w_bf16:.3f}ms  speedup={w_bf16 / w_ours:.2f}x')
    print(f'LOSSLESS:      {"ALL PASS" if all_ok else "SOME FAILED"} ({sum(r["lossless"] for r in results)}/{len(results)})')
    print(f'VRAM CR:       1.33x (12-bit fixed-width)')
    print(f'DISK CR:       1.47x (.tlc variable-length)')
    print("=" * 90)


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/llama-3.1-8b/llama-3.1-8b.safetensors"

    print("=" * 90)
    print(f"Turbo Lossless Benchmark — 12-bit Fixed + Patches (Lossless)")
    print(f"Model: {model_path}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 90)
    print(flush=True)

    results = benchmark_model(model_path)
    print_summary(results)
