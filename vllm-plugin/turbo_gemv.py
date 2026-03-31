"""
Turbo Lossless GEMV — Python wrapper for structured12 decode-matvec kernel.
Drop-in replacement for torch.matmul on compressed weights.

Usage:
    from turbo_gemv import TurboWeight, turbo_matvec

    # Load compressed weight
    w = TurboWeight.load("path/to/layer.0.w_gate")

    # Use like torch.matmul
    output = turbo_matvec(w, activation_bf16)  # same result as W_bf16 @ activation
"""

import torch
import ctypes
import numpy as np
import os
import struct

# Load the compiled HIP kernel
_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decompress_v2.so")
if not os.path.exists(_lib_path):
    raise RuntimeError(f"decompress_v2.so not found at {_lib_path}. Run: hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip")

_hip = ctypes.CDLL(_lib_path)

# Structured12 batch4 async launcher
_hip.launch_structured12_batch4_async.argtypes = [
    ctypes.c_void_p, ctypes.c_int,  # packed, base_exp
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # act0-3
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # esc_row_base, esc_counts, esc_vals
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # out0-3
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p  # M, K, stream
]
_hip.launch_structured12_batch4_async.restype = ctypes.c_int

# V2 (B=1) async launcher
_hip.launch_structured12_v2_async.argtypes = [
    ctypes.c_void_p, ctypes.c_int,  # packed, base_exp
    ctypes.c_void_p,  # activations
    ctypes.c_void_p,  # output
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p  # M, K, stream
]
_hip.launch_structured12_v2_async.restype = ctypes.c_int

# Patches v2 async
_hip.launch_patches_v2_async.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # row_off, cols, correct, wrong
    ctypes.c_void_p, ctypes.c_void_p,  # activations, output
    ctypes.c_int, ctypes.c_void_p  # M, stream
]
_hip.launch_patches_v2_async.restype = ctypes.c_int


class TurboWeight:
    """Compressed weight tensor on GPU, ready for matvec."""

    def __init__(self, M, K, base_exp, packed, row_offsets, patch_cols,
                 patch_correct, patch_wrong, escape_row_base, escape_counts, escape_vals):
        self.M = M
        self.K = K
        self.base_exp = base_exp
        self.packed = packed            # int32 tensor on GPU
        self.row_offsets = row_offsets   # int32 tensor on GPU
        self.patch_cols = patch_cols     # int32 tensor on GPU
        self.patch_correct = patch_correct  # int16 tensor on GPU
        self.patch_wrong = patch_wrong  # int16 tensor on GPU
        self.escape_row_base = escape_row_base  # int32 tensor on GPU
        self.escape_counts = escape_counts      # uint8 tensor on GPU (or None)
        self.escape_vals = escape_vals          # int16 tensor on GPU

    @classmethod
    def load(cls, prefix, device="cuda:0"):
        """Load a compressed weight from turbo format files.

        Args:
            prefix: path prefix, e.g., "models/mistral-7b-instruct-turbo/layer.0.w_gate"
            device: GPU device
        """
        # Read dims
        with open(f"{prefix}.dims") as f:
            parts = f.read().strip().split()
            M, K, num_patches = int(parts[0]), int(parts[1]), int(parts[2])
            base_exp = int(parts[3]) if len(parts) > 3 else 0

        # Load tensors
        packed = torch.from_numpy(np.fromfile(f"{prefix}.packed.bin", dtype=np.int32)).to(device)

        # CSR escape data
        ro_data = np.fromfile(f"{prefix}.row_off.bin", dtype=np.int32)
        row_offsets = torch.from_numpy(ro_data).to(device) if len(ro_data) > 0 else None

        pc_data = np.fromfile(f"{prefix}.patch_cols.bin", dtype=np.int32)
        patch_cols = torch.from_numpy(pc_data).to(device) if len(pc_data) > 0 else None

        pcv_data = np.fromfile(f"{prefix}.patch_correct.bin", dtype=np.int16)
        patch_correct = torch.from_numpy(pcv_data).to(torch.int16).to(device) if len(pcv_data) > 0 else None

        pw_data = np.fromfile(f"{prefix}.patch_wrong.bin", dtype=np.int16)
        patch_wrong = torch.from_numpy(pw_data).to(torch.int16).to(device) if len(pw_data) > 0 else None

        # Build escape tables (same logic as model.cpp)
        WG = 256
        counts = np.zeros(M * WG, dtype=np.int32)
        abs_off = np.zeros(M * WG, dtype=np.int32)

        if num_patches > 0 and row_offsets is not None:
            h_ro = ro_data
            h_cols = pc_data
            h_correct = pcv_data

            for r in range(M):
                for p in range(h_ro[r], h_ro[r + 1]):
                    tid = h_cols[p] % WG
                    counts[r * WG + tid] += 1

            row_base = np.zeros(M, dtype=np.int32)
            total = 0
            for r in range(M):
                row_base[r] = total
                for t in range(WG):
                    abs_off[r * WG + t] = total
                    total += counts[r * WG + t]

            esc_vals = np.zeros(max(total, 1), dtype=np.int16)
            fill = abs_off.copy()
            for r in range(M):
                for p in range(h_ro[r], h_ro[r + 1]):
                    tid = h_cols[p] % WG
                    idx = r * WG + tid
                    esc_vals[fill[idx]] = h_correct[p]
                    fill[idx] += 1

            d_row_base = torch.from_numpy(row_base).to(torch.int32).to(device)
            d_esc_counts = torch.from_numpy(counts.astype(np.uint8)).to(torch.uint8).to(device)
            d_esc_vals = torch.from_numpy(esc_vals).to(torch.int16).to(device)
        else:
            d_row_base = torch.zeros(M, dtype=torch.int32, device=device)
            d_esc_counts = torch.zeros(M * WG, dtype=torch.uint8, device=device)
            d_esc_vals = torch.zeros(1, dtype=torch.int16, device=device)

        return cls(M, K, base_exp, packed, row_offsets, patch_cols,
                   patch_correct, patch_wrong, d_row_base, d_esc_counts, d_esc_vals)

    @property
    def shape(self):
        return (self.M, self.K)

    def vram_bytes(self):
        """Total VRAM used by this weight."""
        total = self.packed.numel() * 4  # int32
        if self.escape_row_base is not None:
            total += self.escape_row_base.numel() * 4
        if self.escape_counts is not None:
            total += self.escape_counts.numel()
        if self.escape_vals is not None:
            total += self.escape_vals.numel() * 2
        return total


def turbo_matvec(weight: TurboWeight, activation: torch.Tensor) -> torch.Tensor:
    """Matrix-vector multiply using structured12 compressed weights.

    Args:
        weight: TurboWeight (compressed)
        activation: BF16 tensor [K] or [B, K]

    Returns:
        FP32 tensor [M] or [B, M]
    """
    assert activation.dtype == torch.bfloat16, f"Expected BF16, got {activation.dtype}"

    if activation.dim() == 1:
        # B=1: two-pass (v2 + patches)
        act_i16 = activation.view(torch.int16)
        output = torch.zeros(weight.M, dtype=torch.float32, device=activation.device)

        _hip.launch_structured12_v2_async(
            weight.packed.data_ptr(), weight.base_exp,
            act_i16.data_ptr(), output.data_ptr(),
            weight.M, weight.K, 0)

        # Apply patches for lossless
        if weight.row_offsets is not None and weight.patch_cols is not None:
            _hip.launch_patches_v2_async(
                weight.row_offsets.data_ptr(), weight.patch_cols.data_ptr(),
                weight.patch_correct.data_ptr(), weight.patch_wrong.data_ptr(),
                act_i16.data_ptr(), output.data_ptr(),
                weight.M, 0)

        return output

    elif activation.shape[0] == 4:
        # B=4
        act = activation.view(torch.int16)
        K = weight.K
        outputs = [torch.zeros(weight.M, dtype=torch.float32, device=activation.device) for _ in range(4)]

        _hip.launch_structured12_batch4_async(
            weight.packed.data_ptr(), weight.base_exp,
            act[0].data_ptr(), act[1].data_ptr(), act[2].data_ptr(), act[3].data_ptr(),
            weight.escape_row_base.data_ptr(),
            weight.escape_counts.data_ptr() if weight.escape_counts is not None else 0,
            weight.escape_vals.data_ptr(),
            outputs[0].data_ptr(), outputs[1].data_ptr(),
            outputs[2].data_ptr(), outputs[3].data_ptr(),
            weight.M, weight.K, 0)

        return torch.stack(outputs)

    else:
        # General B: process as multiple B=1
        results = []
        for i in range(activation.shape[0]):
            results.append(turbo_matvec(weight, activation[i]))
        return torch.stack(results)


def verify_correctness(weight: TurboWeight, original_bf16: torch.Tensor,
                        activation: torch.Tensor, rtol=1e-4):
    """Verify turbo_matvec matches torch.matmul on original BF16 weights."""
    # Ground truth
    gt = original_bf16.float() @ activation.float()

    # Our result
    ours = turbo_matvec(weight, activation)

    torch.cuda.synchronize()

    diff = (ours.cpu() - gt.cpu()).abs()
    max_diff = diff.max().item()

    return max_diff < rtol, max_diff
