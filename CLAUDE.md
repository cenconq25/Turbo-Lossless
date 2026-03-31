# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).
- Do not commit unless explicitly told to.

## Current Results (Measured on MI50 32GB)

### Engine End-to-End (Mistral 7B Instruct)

| Mode | tok/s total | VRAM | vs llama.cpp BF16 (32.2 tok/s) |
|------|------------:|-----:|:-------------------------------|
| B=1 | 26.1 | ~10 GB | 0.81x |
| B=4 | 61.5 | 10.3 GB | 1.91x faster |
| **B=8** | **71.7** | **10.3 GB** | **2.23x faster, 1.32x less VRAM** |

### Compression

- VRAM: **1.33x** (structured 12-bit, 4-bit exponent group + 1-bit sign + 7-bit mantissa)
- Escape rate: **0.03%** (15 consecutive exponents cover 99.97%)
- Decode: **BaseExp + group** (1 integer ADD, no LDS, no lookup table)

## Encoding: Structured 12-Bit

Format per element: `[4-bit exp_group][1-bit sign][7-bit mantissa]` = 12 bits

- BaseExp = 106 (for Mistral 7B, found per-tensor by `find_base_exp()`)
- Groups 1-15 → exponent = BaseExp + group (consecutive: 107-121)
- Group 0 → escape sentinel (read exact BF16 from escape table)
- Sign and mantissa pass through unchanged from original BF16

Decode (GPU kernel, pure ALU):
```
group    = raw12 >> 8
exponent = BaseExp + group    // 1 ADD
sign     = (raw12 >> 7) & 1
mantissa = raw12 & 0x7F
bf16     = (sign << 15) | (exponent << 7) | mantissa
```

No LDS codebook. No lookup table. No memory access for decode. Zero LDS usage for codebook = 2x occupancy vs old approach.

## Kernel Architecture (decompress_v2.hip)

### Production Kernels (Structured12)

| Kernel | Purpose | LDS |
|--------|---------|-----|
| `structured12_matvec_v2` | B=1 two-pass (engine) | 0 KB codebook |
| `structured12_matvec_batch4` | B=4 fused (engine, FAST=1) | 0 KB codebook |
| `structured12_matvec_batch4_scan` | B=4 on-the-fly scan (FAST=0) | 0 KB codebook |
| `structured12_matvec_batch8` | B=8 fused (engine, FAST=1) | 0 KB codebook |

### Legacy Kernels (Codebook LDS — used by bench_fixed12.py)

| Kernel | Purpose | LDS |
|--------|---------|-----|
| `fixed12_matvec_fused` | B=1 fused with escape_offsets table | 8 KB |
| `fixed12_matvec_batch4` | B=4 with escape_counts + warp shuffle | 8 KB |
| `fixed12_matvec_batch8_v2` | B=8 with escape_counts | 8 KB |

### Engine Kernels (kernels.hip)

| Kernel | Purpose |
|--------|---------|
| `rms_norm_bf16_batch_kernel` | Fused RMSNorm → BF16 output |
| `add_rms_norm_bf16_batch_kernel` | Fused residual add + RMSNorm → BF16 |
| `silu_mul_bf16_batch_kernel` | Fused SiLU*mul → BF16 output |
| `rope_kernel` / `rope_batch_kernel` | RoPE with precomputed frequencies |
| `flash_attention_kernel` | Flash Attention v2 tiled (seq >= 1024) |
| `attention_all_heads_kernel` | Naive attention (seq < 1024) |
| `argmax_kernel` | Greedy sampling |

## Engine Data Flow (inference.cpp)

```
embed_lookup → hidden (FP32)

for each layer:
  add_rms_norm_bf16(cur, res) → bf16_a         // fused residual + norm + BF16
  structured12_matvec(wq, bf16_a) → q_buf      // exponent = BaseExp + group
  structured12_matvec(wk, bf16_a) → k_buf
  structured12_matvec(wv, bf16_a) → v_buf
  rope_batch(q, k)
  store_kv_batch(k, v → kv_cache)
  attention(q, kv_cache) → attn_out             // naive < 1024, flash >= 1024
  fp32_to_bf16(attn_out) → bf16_a
  structured12_matvec(wo, bf16_a) → res
  add_rms_norm_bf16(res, cur) → bf16_a
  structured12_matvec(w_gate, bf16_a) → gate
  structured12_matvec(w_up, bf16_a) → up
  silu_mul_bf16(gate, up) → bf16_b
  structured12_matvec(w_down, bf16_b) → cur
  add_batch(cur, res)

rms_norm_bf16(cur) → bf16_a
structured12_matvec(output_proj, bf16_a) → logits
argmax(logits) → next_token
```

## Optimization History

| Step | Result | What changed |
|------|--------|-------------|
| Starting point | 0.05x | atomicAdd CAS stall on MI50 |
| CSR wavefront patches | 1.73x | Row-grouped, no atomics |
| Fused + 2x unroll | 1.88x | Single kernel, O(1) escape |
| Branchless 64-bit read | 2.29x | Eliminated 37% warp divergence |
| LDS codebook | 2.87x | 1-cycle vs 10-cycle L1 |
| BF16 activations | +24% B=4 | Halved L2 activation bandwidth |
| uint8 escape counts | -360 MB | Replaced uint16 thread_off table |
| Flash Attention | 20K+ ctx | Constant LDS, tiled KV |
| B=8 kernel | 61.4 tok/s | Decode once, 8 FMAs |
| Structured12 decode | 67.1 tok/s | BaseExp + group: no LDS, pure ALU |
| 32-bit addressing | 24.0→ B=1 | Pre-compute row base, 32-bit bp in loop |
| Fused add+RMSNorm cross-layer | +3.7% | Saves 31 kernel launches per forward |
| **Pointer-based addressing** | **71.7 tok/s B=8** | **Base ptr + constant byte offsets → immediate offset loads** |

## Tested and Rejected

| Approach | Impact | Reason |
|----------|--------|--------|
| Float32 LDS codebook | -40% | Halves occupancy |
| L1-only codebook | -3% | 4-cycle vs 1-cycle LDS |
| Non-temporal packed loads | -6% | Bypassed L2 too aggressively |
| Software-pipelined activation | -8% | Register pressure |
| Expsplit register LUT `table[code]` | -69% | v_readlane slower than LDS on MI50 |
| Dual-row per block | -11% | Register spill + branch overhead |
| 11-bit codebook | N/A | 28% escape on attention tensors |
| Concurrent streams (gate/up) | broken | Race condition on shared buffer |
| Fused B=1 single-pass escape | -14% | Per-block prefix sum overhead > saved launches |
| Branchless escape in batch4 | -6% | Branch is 99.97% predicted, waits on memory not branch |

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `decompress_v2.hip` | Production | Structured12 kernels + legacy codebook kernels |
| `structured12_pack.c` | Production | `find_base_exp()` + `pack_structured12_csr()` |
| `fixed12_pack.c` | Legacy | Codebook packer (for bench_fixed12.py) |
| `expsplit_pack.c` | Research | Predecessor to structured12 |
| `bench_fixed12.py` | Legacy | Per-tensor kernel benchmark |
| `engine/main.cpp` | Production | CLI entry point |
| `engine/model.h` | Production | CompressedWeight with base_exp field |
| `engine/model.cpp` | Production | Model loader, reads base_exp from .dims |
| `engine/inference.cpp` | Production | forward_b1/b4/b8 with structured12 decode |
| `engine/kernels.hip` | Production | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/convert_model.py` | Production | BF16 safetensors → structured12 format |

## Scope

- **Best for**: Autoregressive decoding of dense BF16 models
- **Batch**: B=8 for max throughput, B=4 balanced, B=1 lowest latency
- **Hardware**: AMD MI50 32GB tested; portable to NVIDIA (warp size 64→32)
- **Tokenizer**: Sentencepiece only (no tiktoken/Llama 3.1 yet)
- **Not for**: Prefill, GPTQ/AWQ INT4, MoE with tiny experts
