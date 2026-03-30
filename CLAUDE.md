# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).

## Current Results

- **2.87x** weighted avg inference speedup (fused kernel, all 226 Llama 8B tensors, 100% lossless)
- **349 GB/s** effective bandwidth on large tensors (34% of MI50 32GB's 1 TB/s HBM2 peak)
- **1.47x** disk compression (.tlc 8-tier variable-length)
- **1.33x** VRAM compression (12-bit fixed-width + escape table)
- Validated compression on 11 BF16 models, 7 architectures, 8B–671B params

## Architecture

### Fused Kernel (decompress_v2.hip)

Single GPU kernel launch per matvec. Decodes 12-bit packed indices, looks up L1-cached codebook, handles escapes inline via per-thread offset table, accumulates FMA.

**Key design:**
- Branchless 64-bit read: 12-bit index straddles word boundary 37% of the time. Always load two uint32 as one uint64 and shift. Zero branches, zero divergence. 1.88x → 2.29x.
- LDS codebook: MI50 has 1 TB/s HBM — kernel is compute-bound. 8 KB codebook in LDS (1-cycle) vs L1 (~10-cycle). 2.29x → 2.87x.
- Per-thread escape table: `escape_offsets[row * 256 + tid]` → O(1) lookup, zero scanning.
- 2x loop unroll: ILP for overlapping packed reads with LDS codebook lookups and FMA.
- Wavefront reduction via `__shfl_down` + shared `warp_sums[4]`.

**Optimization history:**
- atomicAdd patches: 0.02x on token_embd — MI50 lacks hardware float atomics → CAS stall.
- CSR wavefront-parallel patches (two-pass): 1.73x.
- L1-cached codebook + 2x unroll (two-pass): 1.88x.
- Fused merge-scan: 0.15x on token_embd — O(N) scan per escape.
- Fused binary search: 1.32x on token_embd — L2 latency per comparison.
- Per-thread escape offset table (fused): 1.88x — O(1) escape, single kernel.
- Branchless 64-bit read: 2.29x — eliminated 37% warp divergence.
- **LDS codebook (fused): 2.87x — 1-cycle vs 10-cycle codebook lookup.**

### Disk Format (.tlc)

8-tier variable-length prefix code per tensor. Tiers 0-6 use 512-entry codebooks (9-bit index + unary prefix). Tier 7 is raw BF16 escape. ~10.6 bits/param average.

### VRAM Format (12-bit fixed)

4096-entry codebook per tensor. Each weight stored as 12-bit index. Index 4095 = escape sentinel. Escape values stored in per-thread-stride order with offset table for O(1) kernel access.

### Packing (fixed12_pack.c)

Two packing functions:
- `pack_fixed12()`: CSR row-grouped patches for two-pass path
- `pack_fixed12_fused()`: Per-thread-stride escape layout for fused kernel

Both use GPU `torch.unique()` for frequency sort, then C loop for 12-bit bit-packing.

## Compression Validation

| Model | Params | Type | CR | Escape % |
|-------|--------|------|----|----------|
| Llama 3.1 8B | 8B | Dense | 1.509x | 0.03% |
| Phi-4 | 14B | Dense | 1.507x | 0.03% |
| Codestral 22B | 22B | Dense | 1.504x | 0.03% |
| Llama 3.1 70B | 70B | Dense | 1.516x | 0.05% |
| Mistral Large 123B | 123B | Dense | 1.503x | 0.03% |
| MiniMax-Text-01 | 456B | MoE | 1.507x | 0.03% |
| DeepSeek V3 | 671B | MoE | 1.185x | 2.02% |

Dense models: 1.50x+ consistently. Large MoE reduced due to training recipe diversity.

## Implementation Status

| Component | Status |
|-----------|--------|
| Compression analysis (12 models) | Done |
| .tlc encoder/decoder | Done |
| Round-trip verification | Done (292 tensors, 0 mismatches) |
| Fused GPU kernel (12-bit) | Done |
| Benchmark harness | Done |
| End-to-end runtime | WIP (tlc_runtime.py needs fused kernel wiring) |
| NVIDIA/CUDA port | TODO (warp size 64→32) |

## Scope

- **Target**: Dense BF16 safetensors models (Llama, Mistral, Phi, Qwen, Yi, etc.)
- **Works well**: All dense + small MoE → 1.50x+ compression, 1.88x inference
- **Partial**: Large MoE (DeepSeek V3) → 1.2x compression
- **Out of scope**: FP16, FP32, GGUF, GPTQ/AWQ INT4
