# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine targeting AMD MI50 (ROCm) and NVIDIA H100 (CUDA).

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).

---

## Architecture

### Dual-Format Compression

**Disk format (.tlc)**: 8-tier variable-length prefix code. ~10.6 bits/param → **1.47x** compression. Per-tensor codebook with 7 coded tiers (512 entries each) + escape tier for raw BF16 fallback.

**VRAM format**: Fixed-width 12-bit packed indices. 4096-entry codebook per tensor → **1.33x** compression. Converted from .tlc at model load time.

### GPU Inference Pipeline

1. **Load**: Read .tlc from disk, convert variable-length → 12-bit fixed-width in VRAM
2. **Matvec**: Each thread reads 12 bits at `(row * K + col) * 12`, looks up codebook (L1-cached), multiplies by activation
3. **Patch**: Escape values (~0.08%) corrected via CSR row-grouped wavefront-parallel kernel (no atomics)

### Key Numbers (Llama 3.1 8B)

- 226 weight tensors (2D BF16), 66 non-weight tensors (norms, biases → F32 passthrough)
- ~5,124 unique BF16 values per tensor (range 4,223–6,437)
- Top-4095 coverage: 99.92% (12-bit codebook)
- Escape rate: 0.08% typical, up to 0.78% for token_embd (9,066 unique values)
- Shannon entropy: 10.42 bits/param (theoretical ceiling: 1.535x)

### Kernel Design (decompress_v2.hip)

**Matvec kernel** (`fixed12_matvec_v2`):
- One block (256 threads / 4 wavefronts) per row
- Codebook (8 KB) stays in L1 cache — NOT loaded into LDS (higher occupancy)
- Contiguous tile access pattern: adjacent threads read adjacent bit positions
- Cross-warp reduction via `warp_sums[4]` shared memory array

**Patch kernel** (`apply_patches`):
- One wavefront (64 threads) per row
- CSR format: `row_offsets[M+1]` + `patch_cols[num_patches]` + values
- Threads cooperatively process patches with `__shfl_down` reduction
- Zero atomics — each wavefront writes to unique `output[row]`

### Why L1 Cache, Not LDS

MI50 has 16 KB L1 per CU. The 8 KB codebook fits entirely. Benefits:
- Zero LDS allocation → maximum wavefront occupancy
- No per-block codebook load barrier (`__syncthreads`)
- L1 latency comparable to LDS (~4 cycles) after first access
- Measured: +0.07x speedup vs LDS codebook approach

### Why CSR Patches, Not AtomicAdd

MI50/gfx906 lacks hardware `atomicAdd(float)` — it compiles to a CAS retry loop. With 408K patches (token_embd), this caused 580ms stall due to contention. CSR row-grouped format + wavefront reduction: 580ms → 0.2ms.

## Current Results

**Inference**: 1.88x weighted avg speedup (all 226 tensors, 100% lossless, AMD MI50)
**Disk compression**: 1.47x (.tlc 8-tier variable-length)
**VRAM compression**: 1.33x (12-bit fixed-width)
**Validation**: 11 BF16 models, 7 architectures, 8B–671B parameters

## Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| Empirical analysis (12 models) | **Done** | Results documented here |
| .tlc encoder/decoder | **Done** | `tlc_encode.py`, `tlc_decode.py`, `tlc_format.py` |
| Round-trip verification | **Done** | `tlc_verify.py` (292 tensors, 0 mismatches) |
| GPU matvec kernel (12-bit) | **Done** | `decompress_v2.hip`, `decompress_matmul.hip` |
| Benchmark harness | **Done** | `bench_fixed12.py`, `fixed12_pack.c` |
| End-to-end runtime | **WIP** | `tlc_runtime.py` (needs update for fixed-width path) |
| NVIDIA/CUDA port | **TODO** | Requires warp size 64→32 adjustment |

## Empirical Analysis Summary

Profiled across 12 models, 7 architectures, using 4x AMD MI50 GPUs.

### Cross-Model Compression

| Model | Params | Type | CR | Escape % |
|-------|--------|------|----|----------|
| Llama 3.1 8B | 8B | Dense | 1.509x | 0.03% |
| Phi-4 | 14B | Dense | 1.507x | 0.03% |
| Codestral 22B | 22B | Dense | 1.504x | 0.03% |
| Qwen3 30B-A3B | 30B | MoE | 1.505x | 0.03% |
| Llama 3.1 70B | 70B | Dense | 1.516x | 0.05% |
| Mistral Large 123B | 123B | Dense | 1.503x | 0.03% |
| MiniMax-Text-01 | 456B | MoE | 1.507x | 0.03% |
| DeepSeek V3 | 671B | MoE | 1.185x | 2.02% |

Dense models and small MoE: **1.50x+** consistently. Large MoE (DeepSeek V3): reduced due to FP8 mixed-precision training producing higher weight diversity.

### Approach Selection

| Rank | Approach | bits/param | CR | Why selected/rejected |
|------|----------|-----------|------|----------------------|
| 1 | Bit-plane decomposition | 10.50 | 1.524x | Best theoretical, hard to fuse into matmul |
| 2 | Rice/Golomb rank coding | 10.55 | 1.517x | Variable-length, hard to parallelize |
| **3** | **8-tier prefix code** | **10.60** | **1.509x** | **Selected: GPU-friendly, fusable, simple prefix decode** |
| — | Delta/predictive coding | 11.10 | 1.442x | BF16 weights are spatially uncorrelated |
| — | Block-local codebook | 17–20 | <1.0x | Per-block header overhead exceeds savings |
| — | Original 15+1 scheme | 13.75 | 1.164x | Wrong assumptions about weight distributions |

## Scope

- **Target**: Dense transformer models in BF16 safetensors (Llama, Mistral, Phi, Qwen, Yi, etc.)
- **Works well**: All dense models + small MoE → 1.50x+
- **Partial**: Large MoE (DeepSeek V3, Qwen3-235B) → 1.2–1.4x
- **Not worth it**: GPTQ/AWQ INT4 models → ~1.11x total
- **Out of scope**: FP16, FP32, GGUF
