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
| B=1 | 18.7 | ~10 GB | 0.58x |
| B=4 | 46.0 | 10.3 GB | 1.43x faster |
| **B=8** | **61.4** | **10.3 GB** | **1.91x faster, 1.32x less VRAM** |

### Kernel-Level (bench_fixed12.py, Llama 8B)

- **2.90x** weighted avg B=1 speedup (226 tensors, bit-perfect)
- **353 GB/s** peak effective bandwidth
- **19% HBM utilization** (hardware limit for random LDS lookup pattern)

### Compression

- VRAM: **1.33x** (12-bit codebook, 4096 entries)
- Disk (.tlc): **1.47x** (8-tier variable-length)
- Dense models: 1.50x+ disk, 1.33x VRAM
- Tested: 11 models, 7 architectures, 1265+ tensors, all bit-perfect

## Kernel Architecture (decompress_v2.hip)

### Production Kernels

| Kernel | Purpose | LDS |
|--------|---------|-----|
| `fixed12_matvec_fused` | B=1 fused (bench_fixed12.py) | 8KB cb + escape_offsets |
| `fixed12_matvec_v2` | B=1 two-pass (engine, no escape table) | 8KB cb |
| `fixed12_matvec_batch4` | B=4 fused (engine) | 8KB cb + escape counts prefix sum |
| `fixed12_matvec_batch8_v2` | B=8 fused (engine) | 8KB cb + escape counts prefix sum |
| `fixed12_decompress_fused` | Decompress to BF16 (verification) | 8KB cb |

### Engine Kernels (kernels.hip)

| Kernel | Purpose |
|--------|---------|
| `rms_norm_kernel` / `rms_norm_batch_kernel` | RMSNorm (FP32 output) |
| `rms_norm_bf16_batch_kernel` | Fused RMSNorm -> BF16 output |
| `add_rms_norm_bf16_batch_kernel` | Fused residual add + RMSNorm -> BF16 |
| `silu_mul_bf16_batch_kernel` | Fused SiLU*mul -> BF16 output |
| `rope_kernel` / `rope_batch_kernel` | RoPE with precomputed frequencies |
| `store_kv_kernel` / `store_kv_batch_kernel` | KV cache store (FP32 -> BF16) |
| `attention_all_heads_kernel` | Naive multi-head attention (seq < 1024) |
| `flash_attention_kernel` | Flash Attention v2 tiled (seq >= 1024) |
| `argmax_kernel` | Greedy sampling |
| `fp32_to_bf16_kernel` | FP32 -> BF16 conversion |

### Core Decode Loop (all matvec kernels share)

1. Branchless 64-bit read -> 12-bit index (`read12`)
2. LDS codebook lookup (`cb[idx]`, 1-cycle, 8KB int16)
3. O(1) escape check (`idx != 4095`, 99.92% predicted)
4. 2x unrolled FMA with incremental bit positions
5. Wavefront shuffle + cross-warp LDS reduction

### Escape Handling (Engine)

- `escape_row_base[M]` — int32, absolute start per row in escape_vals
- `escape_counts[M*256]` — uint8, per-thread escape count (TURBO_FAST=1 only)
- `escape_vals[num_patches]` — int16, correct BF16 values in thread-stride order
- Warp shuffle prefix sum computes per-thread offset from counts (6 shuffle ops)
- TURBO_FAST=0: on-the-fly scan of packed data (no escape_counts table, saves 361 MB)

## Packing (fixed12_pack.c)

- `build_codebook_12bit()`: Frequency-sort -> top 4095 values -> codebook + reverse map
- `pack_fixed12_csr()`: 12-bit packing + CSR escape data (row_offsets, patch_cols, correct, wrong)
- `pack_fixed12_fused()`: 12-bit packing + per-thread-stride escape table (for bench_fixed12.py)

## Engine Data Flow (inference.cpp)

### Per Token (B=4/B=8)

```
embed_lookup -> hidden (FP32)

for each layer:
  fused_add_rms_norm_bf16(cur, res) -> bf16_a     # residual + norm + BF16 in one kernel
  batch_matvec(wq, bf16_a) -> q_buf                # fused decode-matvec
  batch_matvec(wk, bf16_a) -> k_buf
  batch_matvec(wv, bf16_a) -> v_buf
  rope_batch(q, k)
  store_kv_batch(k, v -> kv_cache)
  attention(q, kv_cache) -> attn_out               # naive < 1024, flash >= 1024
  fp32_to_bf16(attn_out) -> bf16_a
  batch_matvec(wo, bf16_a) -> res
  fused_add_rms_norm_bf16(res, cur) -> bf16_a
  batch_matvec(w_gate, bf16_a) -> ffn_gate
  batch_matvec(w_up, bf16_a) -> ffn_up
  fused_silu_mul_bf16(gate, up) -> bf16_b
  batch_matvec(w_down, bf16_b) -> cur
  add_batch(cur, res)

rms_norm_bf16(cur) -> bf16_a
batch_matvec(output_proj, bf16_a) -> logits
argmax(logits) -> next_token
```

## Optimization History

| Step | Speedup | What changed |
|------|---------|-------------|
| Starting point | 0.05x | atomicAdd CAS stall on MI50 |
| CSR wavefront patches | 1.73x | Row-grouped, no atomics |
| Fused + 2x unroll | 1.88x | Single kernel, O(1) escape |
| Branchless 64-bit read | 2.29x | Eliminated 37% warp divergence |
| LDS codebook | 2.87x | 1-cycle vs 10-cycle L1 |
| Incremental bit positions | 2.90x | iadd64 vs imul64 |
| BF16 activations | +24% B=4 | Halved L2 activation bandwidth |
| Fused RMSNorm->BF16 | -launches | Eliminated separate conversion kernels |
| uint8 escape counts | -360 MB | Replaced uint16 thread_off table |
| Flash Attention | 20K+ ctx | Constant LDS, tiled KV |
| B=8 kernel | 61.4 tok/s | Decode once, 8 FMAs |

## Tested and Rejected

| Approach | Impact | Reason |
|----------|--------|--------|
| Float32 LDS codebook (16 KB) | -40% | Halves occupancy |
| L1-only codebook (no LDS) | -3% | 4-cycle L1 vs 1-cycle LDS |
| Non-temporal packed loads | -6% | Bypassed L2 too aggressively |
| Software-pipelined activation | -8% | Register pressure killed occupancy |
| 4x loop unroll | -1% | Register pressure |
| Expsplit (register LUT decode) | -69% | v_readlane slower than LDS on MI50 |
| 11-bit codebook | N/A | 28% escape rate on wq tensors |
| LDS activation cache | -40% | Occupancy |
| Concurrent streams (gate/up) | broken | Race condition on shared BF16 buffer |

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `decompress_v2.hip` | Production | Fused kernels: B=1, B=2, B=4, B=8, decompress, FP32 variants, expsplit |
| `fixed12_pack.c` | Production | Codebook + 12-bit packing + CSR + fused escape table |
| `bench_fixed12.py` | Production | Benchmark driver + bit-exact verification |
| `expsplit_pack.c` | Research | Exponent-split encoder (faster on NVIDIA, slower on MI50) |
| `engine/main.cpp` | Production | CLI: parse args, load model, generate |
| `engine/model.h` | Production | Model/weight/config data structures |
| `engine/model.cpp` | Production | Model loader + escape table builder |
| `engine/inference.h` | Production | InferenceState struct |
| `engine/inference.cpp` | Production | forward_b1/b4/b8 + generate |
| `engine/kernels.hip` | Production | RMSNorm, RoPE, attention (naive+flash), SiLU, argmax |
| `engine/tokenizer.cpp` | Production | Sentencepiece wrapper |
| `engine/sampler.cpp` | Production | GPU argmax with pre-allocated buffer |
| `engine/convert_model.py` | Production | BF16 safetensors -> turbo format converter |

## Scope

- **Best for**: Autoregressive decoding of dense BF16 models
- **Production batch**: B=8 for max throughput, B=4 for balanced, B=1 for lowest latency
- **Hardware**: AMD MI50 32GB tested; portable to H100/A100 (warp size 64->32)
- **Tokenizer**: Sentencepiece only (no tiktoken/Llama 3.1 yet)
- **Not beneficial**: Prefill, GPTQ/AWQ INT4, MoE with tiny experts
