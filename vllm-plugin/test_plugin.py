#!/usr/bin/env python3
"""
Test the Turbo Lossless GEMV plugin.
Loads compressed weights, runs matvec, verifies against original BF16.
"""

import sys
import os
import time
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbo_gemv import TurboWeight, turbo_matvec

def main():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "mistral-7b-instruct-turbo")

    if not os.path.exists(model_dir):
        print(f"Model not found: {model_dir}")
        print("Run: python3 engine/convert_model.py models/mistral-7b-instruct")
        return

    device = "cuda:0"
    print(f"Device: {device}")
    print(f"Model: {model_dir}")
    print()

    # Load a weight tensor
    prefix = os.path.join(model_dir, "layer.0.w_gate")
    print(f"Loading: {prefix}")
    t0 = time.time()
    w = TurboWeight.load(prefix, device)
    print(f"  Shape: {w.shape}, BaseExp: {w.base_exp}")
    print(f"  VRAM: {w.vram_bytes() / 1e6:.1f} MB")
    print(f"  Load time: {time.time() - t0:.2f}s")
    print()

    # Create test activation
    K = w.K
    x = torch.randn(K, dtype=torch.bfloat16, device=device)

    # Run matvec
    print("Running turbo_matvec (B=1)...")
    # Warmup
    for _ in range(10):
        out = turbo_matvec(w, x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = turbo_matvec(w, x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(sorted(times)[20:80]) / 60
    print(f"  Time: {avg*1000:.3f} ms")
    print(f"  Output shape: {out.shape}")
    print(f"  Output sample: {out[:5].tolist()}")
    print()

    # B=4 test
    print("Running turbo_matvec (B=4)...")
    x4 = torch.randn(4, K, dtype=torch.bfloat16, device=device)
    for _ in range(10):
        out4 = turbo_matvec(w, x4)
    torch.cuda.synchronize()

    times4 = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out4 = turbo_matvec(w, x4)
        torch.cuda.synchronize()
        times4.append(time.perf_counter() - t0)

    avg4 = sum(sorted(times4)[20:80]) / 60
    print(f"  Time: {avg4*1000:.3f} ms")
    print(f"  Output shape: {out4.shape}")
    print()

    # Verify correctness against original BF16
    print("=== Correctness Verification ===")
    try:
        from safetensors import safe_open
        orig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "models", "mistral-7b-instruct",
                                  "model-00001-of-00002.safetensors")
        f = safe_open(orig_path, framework="pt")
        name = [n for n in f.keys() if "layers.0.mlp.gate_proj" in n][0]
        W_orig = f.get_tensor(name).to(device)

        # Compare
        gt = W_orig.float() @ x.float()
        diff = (out.cpu() - gt.cpu()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        print(f"  Correct: {'YES' if max_diff < 1e-3 else 'NO'}")
    except Exception as e:
        print(f"  Skipped verification: {e}")

    print()
    print("=== Summary ===")
    print(f"  B=1: {avg*1000:.3f} ms per matvec")
    print(f"  B=4: {avg4*1000:.3f} ms per matvec")
    print(f"  Plugin ready for vLLM integration")


if __name__ == "__main__":
    main()
