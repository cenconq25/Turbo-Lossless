# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats. We will only support BF16 in future.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).
- Do not commit unless explicitly told to.
- GPU 1 may be degraded from previous crashes — use GPU 0 (HIP_VISIBLE_DEVICES=0).

## Build & Run

```bash
# Build packers
gcc -O3 -shared -fPIC -o structured12_pack.so structured12_pack.c
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c

# Build engine — NVIDIA (RTX 5070 Ti, sm_120)
cd engine && ln -sf kernels.hip kernels.cu && ln -sf ../decompress_v2.hip decompress_v2.cu
nvcc -O3 -arch=sm_120 -I.. -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.cu decompress_v2.cu ../nvidia_kernels.cu ../nvidia_kernels_v3.cu \
  -lcublas -lsentencepiece -lcuda -std=c++17

# Build engine — AMD (MI50, gfx906)
cd engine && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17

# Run
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 ./turbo-engine <model_dir> "<prompt>" <max_tokens> [batch_size]
```

| Variable | Effect |
|----------|--------|
| `CUDA_VISIBLE_DEVICES=N` / `HIP_VISIBLE_DEVICES=N` | Select GPU |
| `TURBO_FAST=1` | Pre-computed escape counts (+10% speed, +361 MB VRAM) |
| `TURBO_CTX=N` | Max context length (default 2048) |
| `TURBO_PROFILE=1` | Print per-token timing: matvec/attn/norm/silu breakdown |
| `TURBO_KERNEL=1\|2\|3` | NVIDIA kernel: 1=V1, **2=V2 cp.async** (recommended), 3=V3 TMA (Mistral only) |
| `TURBO_CUBLAS=1` | Force cuBLAS path for all tensors (debug) |

## Current Results

### RTX 5070 Ti 16GB (NVIDIA Blackwell, 896 GB/s) — Measured 2026-04-02

#### Mistral 7B Instruct (V3 TMA for B>=64, V2 for B<64)

| Mode | tok/s total | VRAM | vs vLLM BF16 |
|------|------------:|-----:|:-------------|
| B=1 | 60.0 | ~10 GB | 1.10x |
| B=8 | 162.6 | ~10 GB | — |
| B=16 | 673.1 | ~10 GB | — |
| B=32 | 1136.3 | ~10 GB | 1.64x |
| B=64 | 1514.2 | ~10 GB | 1.77x |
| B=128 | 2196.6 | ~10 GB | 2.33x |
| **B=256** | **2553.5** | **~10 GB** | **2.93x** |

#### Llama 3.1 8B Instruct (V2 kernel, TURBO_KERNEL=2)

| Mode | tok/s total | VRAM | Notes |
|------|------------:|-----:|:------|
| B=1 | 57.0 | ~10.5 GB | vLLM OOM |
| B=4 | 113.7 | ~10.5 GB | vLLM OOM |
| B=8 | 154.3 | ~10.5 GB | vLLM OOM |
| B=16 | 627.1 | ~10.5 GB | vLLM OOM |
| B=32 | 1069.5 | ~10.5 GB | vLLM OOM |
| B=64 | 1359.1 | ~10.5 GB | vLLM OOM |
| B=128 | 1594.7 | ~10.5 GB | vLLM OOM |
| **B=256** | **1673.9** | **~10.5 GB** | **vLLM OOM** |

**Note:** V3 TMA produces incorrect output for Llama 3.1 8B — use `TURBO_KERNEL=2`.

### MI50 32GB (AMD GCN, 1.0 TB/s)

#### Mistral 7B Instruct

| Mode | tok/s total | VRAM | vs llama.cpp BF16 (33.0 tok/s) |
|------|------------:|-----:|:-------------------------------|
| B=1 | 32.6 | ~10 GB | 0.99x (**matched!**) |
| B=4 | 67.0 | 10.3 GB | 2.03x faster |
| **B=8** | **80.7** | **10.3 GB** | **2.45x faster, 1.32x less VRAM** |

### Tested Models

| Model | tok/s B=1 (RTX 5070 Ti) | tok/s B=1 (MI50) | Tokenizer | Status |
|-------|------------------------:|-----------------:|-----------|--------|
| Mistral 7B Instruct | 60.0 | 32.6 | sentencepiece | Production (V2+V3) |
| Llama 3.1 8B Instruct | 57.0 | 31.8 | HF BPE | Production (V2 only, V3 TMA broken) |

### Compression

- VRAM: **1.33x** (12 bits per weight, same as structured 12-bit)
- Escape rate: **0.03%** (15 consecutive exponents cover 99.97%)
- Decode: **BaseExp + group** (1 integer ADD, no LDS, no lookup table)

## Encoding: Structured 12-Bit

Format per element: `[4-bit exp_group][1-bit sign][7-bit mantissa]` = 12 bits

- BaseExp found per-tensor by `find_base_exp()` (typically 105-109)
- Groups 1-15 -> exponent = BaseExp + group
- Group 0 -> escape sentinel (read exact BF16 from CSR patch table)
- Sign and mantissa pass through unchanged from original BF16

### Storage: Split12 Format (Primary)

Two byte-aligned arrays for zero HBM read amplification:
```
Array 1 (sign_mantissa): [sign 1][mantissa 7] = 1 byte per element
Array 2 (groups):        [group 4]            = 0.5 bytes per element (2 nibbles/byte)
```
Files: `*.sm.bin` (sign+mantissa), `*.gr.bin` (groups)

### Storage: Packed12 Format (Fallback)

12 bits packed contiguously in uint32 words. Used when split12 files not present.
File: `*.packed.bin`

The engine auto-detects: if `.sm.bin`/`.gr.bin` exist, uses split12; else uses packed12.

### Escape Handling

CSR format per tensor:
- `*.row_off.bin`: [M+1] row pointers
- `*.patch_cols.bin`: [num_patches] column indices
- `*.patch_correct.bin`: [num_patches] correct BF16 values
- `*.patch_wrong.bin`: [num_patches] wrong BF16 (from group=0 decode)
- `*.dims`: "M K num_patches base_exp"

At load time, `model.cpp` builds:
- `escape_row_base[M]`: absolute start offset per row
- `escape_counts[M*256]`: per-thread escape count (TURBO_FAST=1 only)
- `escape_vals[num_patches]`: correct BF16 in thread-stride order

## Kernel Architecture (decompress_v2.hip — 1312 lines)

### Split12 Kernels (Primary — byte-aligned, zero amplification)

| Kernel | Purpose |
|--------|---------|
| `split12_matvec_v2` | B=1 single-row |
| `split12_matvec_v2_multirow` | B=1 two rows per block (shared activation) |
| `split12_matvec_v2_dual` | B=1 gate+up fused (shared activation) |
| `split12_matvec_batch4` | B=4 with escape handling |
| `split12_matvec_batch8` | B=8 with escape handling |

### Structured12 Kernels (Fallback — packed 12-bit)

| Kernel | Purpose |
|--------|---------|
| `structured12_matvec_v2` | B=1 two-pass (no escape in main pass) |
| `structured12_matvec_v2_dual` | B=1 gate+up fused |
| `structured12_matvec_batch4` | B=4 fused (FAST=1 required) |
| `structured12_matvec_batch8` | B=8 fused (FAST=1 required) |

### Support Kernels

| Kernel | Purpose |
|--------|---------|
| `apply_patches_v2` | Patch correction for escape values |

### Engine Kernels (kernels.hip — 1171 lines)

| Kernel | Purpose |
|--------|---------|
| `rms_norm_bf16_batch_kernel` | Fused RMSNorm -> BF16 (warp-shuffle reduction) |
| `add_rms_norm_bf16_batch_kernel` | Fused add + RMSNorm -> BF16 |
| `silu_mul_bf16_batch_kernel` | Fused SiLU*mul -> BF16 |
| `rope_kernel` / `rope_batch_kernel` | RoPE with precomputed frequencies |
| `rope_store_kv_kernel` | Fused RoPE + KV cache store (B=1) |
| `flash_attention_kernel/batch` | Flash Attention v2 tiled (seq >= 1024), outputs BF16 |
| `attention_all_heads_kernel/batch` | Naive attention (seq < 1024), outputs BF16 |
| `argmax_kernel` | Greedy sampling (supports batched: 1 block per sequence) |
| `embed_lookup_kernel` | Token ID -> embedding vector (with bounds check) |

## Engine Data Flow (inference.cpp — 603 lines)

### B=1 Forward (forward_b1)

```
embed_lookup -> hidden (FP32)

for each layer:
  [layer 0: rms_norm_bf16(cur) -> bf16_a]
  [layer 1+: bf16_a already set by previous layer's fused add+norm]

  MATVEC_B1(wq, bf16_a) -> q_buf        // split12 or structured12 auto-select
  MATVEC_B1(wk, bf16_a) -> k_buf
  MATVEC_B1(wv, bf16_a) -> v_buf
  rope_store_kv(q, k, v -> kv_cache)    // fused RoPE + KV store
  attention(q, kv_cache) -> bf16_a       // outputs BF16 directly (no fp32_to_bf16)
  MATVEC_B1(wo, bf16_a) -> res
  add_rms_norm_bf16(res, cur) -> bf16_a  // fused add + norm for FFN
  split12_dual(gate+up, bf16_a) -> ffn_gate, ffn_up  // fused gate+up
  + patches for gate and up
  silu_mul_bf16(gate, up) -> bf16_b
  MATVEC_B1(w_down, bf16_b) -> cur
  add_rms_norm_bf16(cur, res, NEXT_attn_norm) -> bf16_a  // cross-layer fusion!

MATVEC_B1(output_proj, bf16_a) -> logits
argmax(logits) -> next_token
```

### MATVEC_B1 Macro

Auto-selects kernel: split12 (if .sm.bin loaded) or structured12 (fallback).
Patches applied separately via `apply_patches_v2`.

### B=4 / B=8 Forward

Same structure but uses `BATCH4_MATVEC` / `BATCH8_MATVEC` macros.
These also auto-select split12 vs structured12.
Batch kernels decode weight once, multiply by B activation vectors.

## Tokenizer (tokenizer.cpp — 379 lines)

Auto-detects tokenizer type:
1. **Sentencepiece** (`tokenizer.model`): Mistral, Gemma, older models
2. **HuggingFace BPE** (`vocab.bin` + `merges.bin` + `byte_encoder.bin`): Llama 3.x, GPT-style

For HF BPE models, pre-extract binary files from tokenizer.json using Python:
```python
# See conversion scripts for details
# vocab.bin: [n_vocab][bos_id][eos_id] + [len][bytes] per token
# merges.bin: [n_merges] + [len_a][len_b][bytes_a][bytes_b] per merge
# byte_encoder.bin: 256 entries of [utf8_len][utf8_bytes]
```

## Model Conversion

### From HuggingFace safetensors (Mistral)
```bash
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/
```

### From GGUF (Llama 3.1)
GGUF tensor shapes use GGML convention: `shape=[ne[0], ne[1]]` where ne[0] is the **contiguous dimension** (columns). Reshape as `[ne[1], ne[0]]` in C order — NO transpose needed.

### Per-tensor .dims Format
```
M K num_patches base_exp
```
Example: `4096 4096 5358 107`

## Key Optimizations Applied

| Optimization | Impact | How |
|---|---|---|
| Split12 format | +5% B=1 | Byte-aligned arrays, zero read amplification |
| Multi-row kernel | +4% B=1 | 2 rows/block share activation loads |
| Pointer-based addressing | +9% all | Immediate load offsets, no 64-bit pointer math |
| Fused add+RMSNorm cross-layer | +4% | Saves 31 kernel launches per forward |
| Attention outputs BF16 | +2% | Eliminates fp32_to_bf16 (32 launches saved) |
| Fused rope+store_kv | +0.5% | 1 launch instead of 2 |
| Fused gate+up dual kernel | +2% B=1 | Shared activation reads |
| Warp-shuffle RMSNorm | +1% | Fewer syncs, less shared memory |
| Single accumulator B=8 | +3% B=8 | Saves 8 VGPRs, reduces register pressure |
| Batched argmax | +0.5% B=8 | 1 launch for all sequences |

## Known Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| V3 TMA produces garbage for Llama 3.1 8B | Open | Use `TURBO_KERNEL=2` (V2 cp.async) |
| forward_b1/b4/b8 used stream=0 (race with sampling) | **Fixed** | Changed to `state->stream` (2026-04-02) |

## Tested and Rejected

| Approach | Impact | Reason |
|----------|--------|--------|
| Fused B=1 single-pass escape | -14% | Per-block prefix sum overhead > saved launches |
| Inline patches in split12 multirow | -5% | Extra kernel args reduce occupancy |
| FP32 activation matvec | crash | Null pointer in patches correction |
| Dual-stream gate/up | crash | Race condition on non-blocking stream |
| Buffer loads for batch4/batch8 | -7-12% | Inline asm overhead > address savings |
| 8x unroll | +0% | Compiler serializes into 2x4 |
| Blocked element access | -84% | Zero ILP, worse L2 cache |
| Staged rocBLAS GEMV | -83% | Materializing full BF16 matrix is 6x slower |
| Fused silu+matvec | -16% | 4x more activation bandwidth |
| 256-thread RMSNorm | -59% | Fewer threads = worse memory latency hiding |
| Speculative decoding (same model) | N/A | Draft cost = verify cost, no net benefit |

## File Map

| File | Lines | Purpose |
|------|------:|---------|
| `decompress_v2.hip` | 1312 | All GPU matvec kernels (split12 + structured12) |
| `structured12_pack.c` | 118 | Packer for structured12 format (used by convert_model.py) |
| `split12_pack.c` | 128 | Packer for split12 byte-aligned format |
| `engine/main.cpp` | 80 | CLI: model_path, prompt, max_tokens, batch_size |
| `engine/model.h` | 78 | CompressedWeight struct (packed + split_sm/split_gr + escape) |
| `engine/model.cpp` | 268 | Loads config.bin, weights, builds escape tables, proper GPU cleanup |
| `engine/inference.h` | 47 | InferenceState struct (buffers, streams, events) |
| `engine/inference.cpp` | 603 | forward_b1/b4/b8 + generate loop + profiling macros |
| `engine/kernels.hip` | 1171 | Non-matvec GPU kernels (norm, rope, attention, silu, argmax) |
| `engine/tokenizer.h` | 16 | Tokenizer struct (type 0=sentencepiece, 1=HF BPE) |
| `engine/tokenizer.cpp` | 379 | Auto-detect + BPE encode/decode with byte-level mapping |
| `engine/sampler.h` | 11 | Sampler interface |
| `engine/sampler.cpp` | 32 | Greedy + batched argmax |
| `engine/convert_model.py` | 207 | HF safetensors -> turbo format |

**Total: ~4450 lines.**

## Scope

- **Best for**: Autoregressive decoding of dense BF16 transformer models
- **Batch**: B=8 for max throughput, B=4 balanced, B=1 lowest latency
- **Hardware**: AMD MI50 32GB tested (gfx906); portable to NVIDIA (warp size 64->32)
- **Tokenizer**: Sentencepiece + HuggingFace BPE (covers Mistral, Llama 3.x, GPT-style)
- **Not for**: Prefill optimization, GPTQ/AWQ INT4, MoE routing, FP16/FP32 models
