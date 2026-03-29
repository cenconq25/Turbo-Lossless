#!/usr/bin/env python3
"""Test harness for the TLC HIP GPU kernel."""

import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tlc_runtime import TLCModel
from tlc_decode import decode_tlc


DEVICE = torch.device("cuda:0")
TEST_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model.tlc")
LLAMA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "llama-3.1-8b", "llama-3.1-8b.tlc")


def test_decompress_only():
    """Test bit-perfect decompression of all BF16 tensors in test_model.tlc."""
    print("\n" + "-" * 60)
    print("TEST: test_decompress_only")
    print("-" * 60)

    assert os.path.exists(TEST_MODEL_PATH), f"Test model not found: {TEST_MODEL_PATH}"

    model = TLCModel(TEST_MODEL_PATH, device=DEVICE)
    cpu_tensors = decode_tlc(TEST_MODEL_PATH)

    all_passed = True
    for name, cpu_tensor in cpu_tensors.items():
        if cpu_tensor.dtype != torch.bfloat16:
            print(f"  [{name}] Skipping (dtype={cpu_tensor.dtype}, not BF16)")
            continue

        gpu_tensor = model.decompress(name)
        gpu_cpu = gpu_tensor.cpu()

        gpu_bits = gpu_cpu.view(torch.int16)
        cpu_bits = cpu_tensor.view(torch.int16)

        match = torch.equal(gpu_bits, cpu_bits)
        status = "OK" if match else "FAIL"
        if not match:
            all_passed = False
            mismatches = (gpu_bits != cpu_bits).sum().item()
            total = cpu_bits.numel()
            print(f"  [{name}] {status} -- {mismatches}/{total} values differ (shape={list(cpu_tensor.shape)})")
        else:
            print(f"  [{name}] {status} -- bit-perfect match (shape={list(cpu_tensor.shape)})")

    assert all_passed, "Some tensors did not match bit-perfectly"
    print("  test_decompress_only PASSED")


def test_matvec_small():
    """Test fused matvec against CPU reference for the small test model."""
    print("\n" + "-" * 60)
    print("TEST: test_matvec_small")
    print("-" * 60)

    assert os.path.exists(TEST_MODEL_PATH), f"Test model not found: {TEST_MODEL_PATH}"

    model = TLCModel(TEST_MODEL_PATH, device=DEVICE)
    cpu_tensors = decode_tlc(TEST_MODEL_PATH)

    weight_name = "layer.0.weight"
    assert weight_name in cpu_tensors, f"Tensor '{weight_name}' not found in test model. Available: {list(cpu_tensors.keys())}"

    W = cpu_tensors[weight_name]  # [256, 512]
    assert W.dtype == torch.bfloat16, f"Expected BF16, got {W.dtype}"
    M, K = W.shape
    print(f"  Weight shape: [{M}, {K}]")

    torch.manual_seed(42)
    x = torch.randn(K, dtype=torch.bfloat16)

    y_ref = W.float() @ x.float()

    x_gpu = x.to(DEVICE)
    y_gpu = model.matvec(weight_name, x_gpu)
    y_gpu_cpu = y_gpu.cpu().float()

    max_abs_err = (y_gpu_cpu - y_ref).abs().max().item()
    ref_norm = y_ref.abs().max().item()
    max_rel_err = max_abs_err / (ref_norm + 1e-12)

    print(f"  Max absolute error: {max_abs_err:.6e}")
    print(f"  Max relative error: {max_rel_err:.6e}")
    print(f"  Reference max magnitude: {ref_norm:.6e}")

    passed = torch.allclose(y_gpu_cpu, y_ref, atol=1e-2, rtol=1e-2)
    status = "OK" if passed else "FAIL"
    print(f"  torch.allclose(atol=1e-2, rtol=1e-2): {status}")

    assert passed, "Matvec result does not match CPU reference within tolerance"
    print("  test_matvec_small PASSED")


def test_decompress_llama8b():
    """Test bit-perfect decompression of a Llama 3.1 8B weight tensor."""
    print("\n" + "-" * 60)
    print("TEST: test_decompress_llama8b")
    print("-" * 60)

    model = TLCModel(LLAMA_MODEL_PATH, device=DEVICE)

    # Find the first BF16 weight tensor
    tensor_names = model.tensor_names()
    target_name = None
    for name in tensor_names:
        if "blk.0.attn_k.weight" in name:
            target_name = name
            break
    if target_name is None:
        # Fall back to first BF16 tensor
        for name in tensor_names:
            info = model.tensor_info(name)
            if info.get("dtype") == torch.bfloat16 or str(info.get("dtype")) == "torch.bfloat16":
                target_name = name
                break
    if target_name is None:
        target_name = tensor_names[0]

    print(f"  Target tensor: {target_name}")

    # GPU decompress with timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_tensor = model.decompress(target_name)
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - t0
    print(f"  GPU decompress time: {gpu_time * 1000:.2f} ms")

    # CPU decode with timing
    t0 = time.perf_counter()
    cpu_tensors = decode_tlc(LLAMA_MODEL_PATH)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU decode time: {cpu_time * 1000:.2f} ms")

    cpu_tensor = cpu_tensors[target_name]
    gpu_cpu = gpu_tensor.cpu()

    if cpu_tensor.dtype == torch.bfloat16:
        gpu_bits = gpu_cpu.view(torch.int16)
        cpu_bits = cpu_tensor.view(torch.int16)
        match = torch.equal(gpu_bits, cpu_bits)
        if not match:
            mismatches = (gpu_bits != cpu_bits).sum().item()
            total = cpu_bits.numel()
            print(f"  FAIL -- {mismatches}/{total} values differ")
        else:
            print(f"  OK -- bit-perfect match (shape={list(cpu_tensor.shape)})")
        assert match, "Llama 8B tensor did not match bit-perfectly"
    else:
        match = torch.equal(gpu_cpu, cpu_tensor)
        print(f"  {'OK' if match else 'FAIL'} -- exact match (dtype={cpu_tensor.dtype})")
        assert match, "Llama 8B tensor did not match"

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print(f"  Speedup: {speedup:.1f}x (GPU vs CPU)")
    print("  test_decompress_llama8b PASSED")


def test_matvec_llama8b():
    """Test fused matvec on a Llama 3.1 8B weight tensor against CPU reference."""
    print("\n" + "-" * 60)
    print("TEST: test_matvec_llama8b")
    print("-" * 60)

    model = TLCModel(LLAMA_MODEL_PATH, device=DEVICE)

    # Find the first BF16 weight tensor
    tensor_names = model.tensor_names()
    target_name = None
    for name in tensor_names:
        if "blk.0.attn_k.weight" in name:
            target_name = name
            break
    if target_name is None:
        for name in tensor_names:
            info = model.tensor_info(name)
            if info.get("dtype") == torch.bfloat16 or str(info.get("dtype")) == "torch.bfloat16":
                target_name = name
                break
    if target_name is None:
        target_name = tensor_names[0]

    print(f"  Target tensor: {target_name}")

    # Get reference weight via CPU decode
    cpu_tensors = decode_tlc(LLAMA_MODEL_PATH)
    W = cpu_tensors[target_name]
    print(f"  Weight shape: {list(W.shape)}, dtype: {W.dtype}")

    if W.dtype != torch.bfloat16:
        print("  Skipping matvec test (tensor is not BF16)")
        return

    M, K = W.shape

    torch.manual_seed(42)
    x = torch.randn(K, dtype=torch.bfloat16)

    # CPU reference
    y_ref = W.float() @ x.float()

    # GPU fused matvec
    x_gpu = x.to(DEVICE)
    torch.cuda.synchronize()
    y_gpu = model.matvec(target_name, x_gpu)
    torch.cuda.synchronize()
    y_gpu_cpu = y_gpu.cpu().float()

    max_abs_err = (y_gpu_cpu - y_ref).abs().max().item()
    ref_norm = y_ref.abs().max().item()
    max_rel_err = max_abs_err / (ref_norm + 1e-12)

    print(f"  Max absolute error: {max_abs_err:.6e}")
    print(f"  Max relative error: {max_rel_err:.6e}")
    print(f"  Reference max magnitude: {ref_norm:.6e}")

    passed = torch.allclose(y_gpu_cpu, y_ref, atol=1e-2, rtol=1e-2)
    status = "OK" if passed else "FAIL"
    print(f"  torch.allclose(atol=1e-2, rtol=1e-2): {status}")

    assert passed, "Llama 8B matvec result does not match CPU reference within tolerance"
    print("  test_matvec_llama8b PASSED")


def benchmark():
    """Benchmark decompression and matvec throughput."""
    print("\n" + "-" * 60)
    print("BENCHMARK")
    print("-" * 60)

    # Prefer Llama 8B, fall back to test model
    if os.path.exists(LLAMA_MODEL_PATH):
        model_path = LLAMA_MODEL_PATH
        print(f"  Using Llama 8B model: {model_path}")
    else:
        model_path = TEST_MODEL_PATH
        print(f"  Using test model: {model_path}")

    model = TLCModel(model_path, device=DEVICE)
    tensor_names = model.tensor_names()

    # Pick the largest BF16 tensor
    target_name = None
    target_size = 0
    for name in tensor_names:
        info = model.tensor_info(name)
        dtype = info.get("dtype")
        if dtype == torch.bfloat16 or str(dtype) == "torch.bfloat16":
            shape = info.get("shape", [])
            size = 1
            for d in shape:
                size *= d
            if size > target_size:
                target_size = size
                target_name = name

    if target_name is None:
        target_name = tensor_names[0]
        info = model.tensor_info(target_name)
        shape = info.get("shape", [])
        target_size = 1
        for d in shape:
            target_size *= d

    info = model.tensor_info(target_name)
    shape = info.get("shape", [])
    compressed_size = info.get("compressed_size", target_size * 2)

    print(f"  Tensor: {target_name}")
    print(f"  Shape: {shape}")
    print(f"  Elements: {target_size:,}")
    print(f"  Decompressed size: {target_size * 2 / 1e6:.2f} MB (BF16)")
    print(f"  Compressed size: {compressed_size / 1e6:.2f} MB")

    num_warmup = 10
    num_runs = 100

    # Benchmark decompress
    print(f"\n  Decompress benchmark ({num_warmup} warmup + {num_runs} timed runs):")
    for _ in range(num_warmup):
        _ = model.decompress(target_name)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model.decompress(target_name)
        torch.cuda.synchronize()
    decompress_total = time.perf_counter() - t0
    decompress_avg = decompress_total / num_runs

    compressed_bw = (compressed_size / 1e9) / decompress_avg
    decompressed_bw = (target_size * 2 / 1e9) / decompress_avg

    print(f"    Average time: {decompress_avg * 1e6:.1f} us")
    print(f"    Compressed bandwidth: {compressed_bw:.2f} GB/s")
    print(f"    Effective decompressed bandwidth: {decompressed_bw:.2f} GB/s")

    # Benchmark matvec (only for 2D BF16 tensors)
    dtype = info.get("dtype")
    is_bf16 = (dtype == torch.bfloat16 or str(dtype) == "torch.bfloat16")
    if is_bf16 and len(shape) == 2:
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(K, dtype=torch.bfloat16, device=DEVICE)

        print(f"\n  Matvec benchmark ({num_warmup} warmup + {num_runs} timed runs):")
        for _ in range(num_warmup):
            _ = model.matvec(target_name, x)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = model.matvec(target_name, x)
            torch.cuda.synchronize()
        matvec_total = time.perf_counter() - t0
        matvec_avg = matvec_total / num_runs

        flops = 2 * M * K
        gflops = (flops / 1e9) / matvec_avg

        print(f"    Average time: {matvec_avg * 1e6:.1f} us")
        print(f"    GFLOPS: {gflops:.2f}")
        print(f"    Compressed bandwidth: {(compressed_size / 1e9) / matvec_avg:.2f} GB/s")
    else:
        print("\n  Skipping matvec benchmark (tensor is not 2D BF16)")

    print("  Benchmark complete.")


if __name__ == "__main__":
    print("=" * 60)
    print("TLC Kernel Tests")
    print("=" * 60)

    test_decompress_only()
    test_matvec_small()

    if os.path.exists(LLAMA_MODEL_PATH):
        test_decompress_llama8b()
        test_matvec_llama8b()
        benchmark()
    else:
        print("\nSkipping Llama 8B tests (no .tlc file)")

    print("\nAll tests passed!")
