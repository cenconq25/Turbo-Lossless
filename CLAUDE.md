# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).

## Current Results

- **1.88x** weighted avg inference speedup (fused kernel, all 226 Llama 8B tensors, 100% lossless)
- **1.47x** disk compression (.tlc 8-tier variable-length)
- **1.33x** VRAM compression (12-bit fixed-width + escape table)
- Validated compression on 11 BF16 models, 7 architectures, 8B–671B params

## Architecture

### Fused Kernel (decompress_v2.hip)

Single GPU kernel launch per matvec. Decodes 12-bit packed indices, looks up L1-cached codebook, handles escapes inline via per-thread offset table, accumulates FMA.

**Key design:**
- L1-cached codebook: 8 KB fits in MI50's 16 KB L1. Zero LDS overhead, max occupancy.
- Per-thread escape table: `escape_offsets[row * 256 + tid]` → O(1) lookup, zero scanning.
- 2x loop unroll: ILP for overlapping packed reads with codebook lookups and FMA.
- Wavefront reduction via `__shfl_down` + shared `warp_sums[4]`.

**Why this design:**
- LDS codebook was slower (1.80x) due to per-block load barrier.
- Fused merge-scan was 0.15x on token_embd — strided thread access = O(N) scan per escape.
- Binary search fused was 1.32x on token_embd — L2 latency per comparison.
- Two-pass (separate patch kernel) was 1.88x but 2 kernel launches.
- atomicAdd patches: MI50/gfx906 lacks hardware float atomics → 580ms CAS stall.

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
