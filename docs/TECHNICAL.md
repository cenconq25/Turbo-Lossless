# Technical Details

Deep dive into Turbo Lossless internals: encoding format, kernel architecture, benchmarks, engine data flow, and optimization history.

---

## Scope

- **Best for**: Autoregressive decoding of dense BF16 transformer models
- **Batch**: B=8 for max throughput, B=4 balanced, B=1 lowest latency
- **Hardware**: NVIDIA RTX 5070 Ti, AMD MI50 32GB tested; portable via `gpu_compat.h`
- **Tokenizer**: Sentencepiece + HuggingFace BPE (covers Mistral, Llama 3.x, GPT-style)
- **Not for**: Prefill optimization, GPTQ/AWQ INT4, MoE routing, FP16/FP32 models

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

---

## Encoding: Structured 12-Bit

Format per element: `[4-bit exp_group][1-bit sign][7-bit mantissa]` = 12 bits

```
BF16:         [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit: [group 4][sign 1][mantissa 7]     = 12 bits

Decode: exponent = BaseExp + group   <-- one ADD
```

- BaseExp found per-tensor by `split12_find_base_exp()` (typically 105-109)
- Groups 1-15 -> exponent = BaseExp + group
- Group 0 -> escape sentinel (read exact BF16 from CSR patch table)
- Sign and mantissa pass through unchanged from original BF16

### Compression Stats

- VRAM: **1.33x** (12 bits per weight, same as structured 12-bit)
- Escape rate: **0.03%** (15 consecutive exponents cover 99.97%)
- Decode: **BaseExp + group** (1 integer ADD, no LDS, no lookup table)

### Storage: Split12 Format

Two byte-aligned arrays for zero HBM read amplification:
```
Array 1 (sign_mantissa): [sign 1][mantissa 7] = 1 byte per element
Array 2 (groups):        [group 4]            = 0.5 bytes per element (2 nibbles/byte)
```
Files: `*.sm.bin` (sign+mantissa), `*.gr.bin` (groups)

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

---

## Compared to Other Lossless BF16 Methods

| | **Turbo** | **ZipServ** | **DFloat11** | **ZipNN** | **NeuZip** | **Huff-LLM** |
|---|---|---|---|---|---|---|
| **Venue** | -- | ASPLOS'26 | NeurIPS'25 | IEEE'25 | arXiv'24 | arXiv'25 |
| **Bits/weight** | **12.0 (fixed)** | ~11.3 | ~11.0 | ~11 | ~10.6 | ~11.6 |
| **Decode cost** | **1 ADD** | Bitmap+popcount | Huffman LUT | CPU zstd | ANS | CAM |
| **Escape rate** | **0.03%** | ~3% | 0% | 0% | 0% | 0% |
| **Fused decode?** | **Yes** (matvec) | **Yes** (tensor core) | No (separate) | No | No | ASIC only |
| **GPU decode** | Yes | Yes | Yes | No (CPU) | Yes | No (ASIC) |
| **Hardware** | **NVIDIA + AMD** | NVIDIA | NVIDIA | CPU | NVIDIA | Custom |

**Our trade-off:** We use 0.7 more bits/weight than ZipServ, but decode with 1 instruction instead of 5-8, have 100x fewer escapes, and run on NVIDIA + AMD.

---

## Benchmarks

### RTX 5070 Ti 16GB (NVIDIA Blackwell, 896 GB/s) -- Measured 2026-04-02

Single GPU, 200-token generation, output verified.

#### Single-User (B=1)

| Model | Params | llama.cpp | vLLM | Turbo | Speedup |
|-------|-------:|----------:|-----:|------:|--------:|
| Llama 2 7B | 6.74B | 59.6 | 43.9 | **64.7** | **1.47x** vs vLLM |
| Mistral 7B | 7.25B | 55.7 | 54.7 | **60.0** | **1.10x** vs vLLM |
| Llama 3.1 8B | 8.03B | 52.9 | OOM | **57.0** | **1.08x** vs llama.cpp |

#### Multi-User (total tok/s)

| Model | Params | B=32 | B=64 | B=128 | B=256 | vLLM B=256 | Speedup |
|-------|-------:|-----:|-----:|------:|------:|-----------:|--------:|
| Llama 2 7B | 6.74B | 1,289 | 1,605 | 2,576 | **2,931** | 1,086 | **2.70x** |
| Mistral 7B | 7.25B | 1,136 | 1,514 | 2,197 | **2,554** | 872 | **2.93x** |
| Llama 3.1 8B | 8.03B | 1,069 | 1,439 | 2,111 | **2,471** | OOM | -- |

#### Detailed Per-Batch -- Mistral 7B Instruct (V3 TMA for B>=64, V2 for B<64)

| Mode | tok/s total | VRAM | vs vLLM BF16 |
|------|------------:|-----:|:-------------|
| B=1 | 60.0 | ~10 GB | 1.10x |
| B=8 | 162.6 | ~10 GB | -- |
| B=16 | 673.1 | ~10 GB | -- |
| B=32 | 1136.3 | ~10 GB | 1.64x |
| B=64 | 1514.2 | ~10 GB | 1.77x |
| B=128 | 2196.6 | ~10 GB | 2.33x |
| **B=256** | **2553.5** | **~10 GB** | **2.93x** |

#### Detailed Per-Batch -- Llama 3.1 8B Instruct (V3 TMA for B>=64, V2 for B<64)

| Mode | tok/s total | VRAM | Notes |
|------|------------:|-----:|:------|
| B=1 | 57.0 | ~10.5 GB | vLLM OOM |
| B=4 | 113.5 | ~10.5 GB | vLLM OOM |
| B=8 | 154.3 | ~10.5 GB | vLLM OOM |
| B=16 | 627.5 | ~10.5 GB | vLLM OOM |
| B=32 | 1068.6 | ~10.5 GB | vLLM OOM |
| B=64 | 1438.6 | ~10.5 GB | vLLM OOM |
| B=128 | 2110.8 | ~10.5 GB | vLLM OOM |
| **B=256** | **2470.7** | **~10.5 GB** | **vLLM OOM** |

#### VRAM Usage

| Model | Model VRAM | Overhead (B=1) | Total (B=1) | Total (B=256) | vLLM Total |
|-------|----------:|-----------:|------------:|--------------:|-----------:|
| Llama 2 7B | 10.1 GB | 1.2 GB | **11.3 GB** | OOM (MHA) | ~14.7 GB |
| Mistral 7B | 10.2 GB | 0.9 GB | **11.1 GB** | 12.7 GB | 15.3 GB |
| Llama 3.1 8B | 11.5 GB | 0.9 GB | **12.4 GB** | 14.1 GB | OOM |

Overhead = KV cache + escape tables + TURBO_FAST + activation buffers + CUDA context. Llama 2 7B uses MHA (32/32 heads) -- 4x larger KV cache than GQA models, OOMs at B=256 on 16 GB.

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
| Llama 3.1 8B Instruct | 57.0 | 31.8 | HF BPE | Production (V2+V3) |

### Compression Works on Everything

Tested across 11 models -- LLMs up to 405B, MoE, image, and video:

| Model | Params | Type | Escape Rate | Compression |
|-------|-------:|:----:|------------:|:-----------:|
| Llama 3.1 405B | 405B | Dense LLM | 0.034% | 1.33x |
| Llama 3.1 70B | 70B | Dense LLM | 0.018% | 1.33x |
| Mixtral 8x7B | 46.7B | MoE LLM | 0.050% | 1.33x |
| SDXL UNet | 2.6B | Image (FP16) | 0.233% | 1.33x |
| CogVideoX 2B | 1.7B | Video (FP16) | 0.128% | 1.33x |
| Gemma 4 E4B | 8.0B | Multimodal | 1.512% | 1.31x |

Dense LLMs: <0.1% escapes. MoE: same. Image/video: works. Multimodal: higher escapes but still compresses.

---

## Kernel Architecture

### Per-Row Kernels (`decompress_v2.hip`) -- B=1 to B=8

Bandwidth-bound matvec. Each thread streams weights, decodes inline (1 ADD), accumulates. Portable NVIDIA + AMD via `gpu_compat.h`.

| Kernel | Batch | What It Does Differently |
|--------|:-----:|--------------------------|
| `split12_matvec_v2` | B=1 | Baseline: 4x unrolled, pointer-stride addressing |
| `split12_matvec_v2_multirow` | B=1 | 2 rows/block share activation loads -- halves activation memory traffic (+4%) |
| `split12_matvec_v2_dual` | B=1 | Gate+up fused -- two weight matrices share one activation load |
| `split12_matvec_batch4` | B=4 | Decode weight once x 4 activations. Inline escape via warp-shuffle prefix sum |
| `split12_matvec_batch8` | B=8 | Same as batch4 x 8. Single accumulator per batch (saves 8 VGPRs) |
| `apply_patches_v2` | All | Escape patch correction (separate kernel for B=1, inline for B=4/8) |

B=1 escape patches: batched `apply_patches_v2` kernel eliminates ~192 tiny launches per token.
B=4/8 escape patches: handled inline via `if (group == 0) val = escape_table[ptr++]`.

### Fused Decode+GEMM (`nvidia_kernels.cu`) -- B>8, NVIDIA Only

Decode Split12 weights directly into tensor core registers, run PTX `mma.sync.aligned.m16n8k16`. All versions use K-slice interleaving from ZipServ: `decode(slice N+1)` overlaps `mma.sync(slice N)`, hiding decode behind tensor core math. Register double-buffering (`a[2][4]`, `b[2][2]`) enables ping-pong.

#### V1 and V2 -- Software `cp.async` Pipeline

Same loading mechanism: all threads cooperatively issue `cp.async.cg.shared.global` (16-byte async DRAM->shared copies, bypassing registers). Double-buffered K-tiles overlap loading with compute. `cp_async_commit()` / `cp_async_wait<N>()` for pipeline control, `__syncthreads()` for barriers.

Only difference is tile size and occupancy:

| | V1 | V2 |
|--|----|----|
| Threads | 256 (8 warps) | 128 (4 warps) |
| Tile MxK | 128x64 | 64x64 |
| Shared memory | ~40 KB | ~20 KB |
| Blocks/SM | 1 | 2-3 (`__launch_bounds__(128, 3)`) |
| Patch correction | Fused in output write | Separate kernel |

V1: 1 block/SM -> SM idles when stalled on data. V2: 2-3 blocks -> block B runs while block A stalls. Same approach ZipServ uses for 2.21x over cuBLAS.

#### V3 -- Blackwell Hardware TMA

Fundamentally different architecture. Replaces software `cp.async` with the Tensor Memory Accelerator (TMA) -- dedicated hardware on SM90+/Blackwell that copies entire 2D tiles autonomously.

**How TMA loading works:**

1. **TMA descriptors** created on host via `cuTensorMapEncodeTiled()` encoding shape, strides, tile size, swizzle. Three descriptors: sign-mantissa (64B swizzle), groups (32B swizzle), activations (128B swizzle). Copied to device memory once.

2. **One elected thread** per warp (via `elect.sync` PTX) issues `cp.async.bulk.tensor.2d.shared::cluster.global.tile`. The other 127 threads do nothing for loading -- 100% free for decode and compute.

3. **Mbarrier synchronisation** replaces `__syncthreads()`. Elected thread calls `mbar_expect_tx(bytes)` to declare bytes in flight. TMA hardware auto-signals mbarrier (`mbarrier::complete_tx`) on completion. Waiting threads spin on `mbar_wait(phase)` using hardware parity.

4. **Swizzle patterns** applied automatically by TMA during copy to eliminate bank conflicts. Shared memory reads must compensate with XOR address translation:
   ```c
   // SM: 64B swizzle -> per-row XOR
   int swiz = ((row >> 1) & 3) << 4;
   uint32_t val = *(sm + row * TILE_K + ((col & ~3) ^ swiz));
   // B (activations): 128B swizzle
   uint32_t val = *(B + row * TILE_K*2 + (byte_col ^ ((row & 7) << 4)));
   ```

5. **Two-stage pipeline** alternates mbar0/mbar1 with phase toggling, overlapping TMA loads of next tile with compute on current tile.

**Why V3 wins at B>=64:** In V1/V2, threads spend cycles on `cp.async` load loops. In V3, TMA hardware handles all data movement -- threads spend 100% on decode + `mma.sync`. TMA setup overhead (descriptors, elect_sync, mbarrier) amortises at large B.

#### V1 vs V2 vs V3 Summary

| | **V1** | **V2** | **V3 (Blackwell TMA)** |
|---|---|---|---|
| **HBM -> shared** | Software `cp.async` (16B) | Software `cp.async` (16B) | Hardware TMA (`cp.async.bulk.tensor.2d`) |
| **Who loads** | All 256 threads | All 128 threads | 1 elected thread -> hardware |
| **Threads** | 256 (8 warps) | 128 (4 warps) | 128 (4 warps) |
| **Tile MxK** | 128x64 | 64x64 | 64x64 |
| **Shared mem** | ~40 KB | ~20 KB | ~22 KB |
| **Blocks/SM** | 1 | 2-3 | 2 |
| **Sync** | `__syncthreads()` | `__syncthreads()` | Hardware mbarrier |
| **Bank conflicts** | Manual padding | Manual padding | TMA hardware swizzle |
| **Shared reads** | Direct offset | Direct offset | XOR address translation |
| **Patches** | Fused in output | Separate kernel | Fused in output |
| **GPU** | Any CUDA | Any CUDA | SM90+ (Hopper/Blackwell) |
| **Best for** | Fallback | B=16..63 | **B>=64 (default)** |

#### Auto-Selection Logic

| Batch | Kernel | Why |
|:-----:|--------|-----|
| B=1 | Per-row multirow + batched patches | Bandwidth-bound; tensor cores can't help |
| B=4 | Per-row batch4 (inline escapes) | Still bandwidth-bound; per-row simpler |
| B=8 | Per-row batch8 (inline escapes) | Per-row wins by avoiding GEMM overhead |
| B=16..63 | V2 fused decode+GEMM | Tensor cores useful; high occupancy wins |
| B>=64 | **V3 TMA** fused decode+GEMM | TMA frees all threads for compute |

Override: `TURBO_KERNEL=1|2|3`

### Engine Kernels (`engine/kernels.hip`)

Non-matvec kernels, portable NVIDIA + AMD:

| Kernel | Purpose |
|--------|---------|
| `rms_norm_bf16_batch_kernel` | Fused RMSNorm -> BF16 (warp-shuffle reduction) |
| `add_rms_norm_bf16_batch_kernel` | Fused add + RMSNorm -> BF16. Cross-layer: layer N residual + layer N+1 attn norm in one kernel (saves 31 launches) |
| `silu_mul_bf16_batch_kernel` | Fused SiLU x mul -> BF16 |
| `rope_kernel` / `rope_batch_kernel` | RoPE with precomputed frequencies |
| `rope_store_kv_kernel` | Fused RoPE + KV cache store (B=1) |
| `flash_attention_kernel/batch` | Flash Attention v2 tiled, constant 34 KB shared mem (seq >= 1024) |
| `attention_all_heads_kernel/batch` | Naive attention (seq < 1024), outputs BF16 |
| `argmax_kernel` | Greedy sampling (batched: 1 block per sequence) |
| `embed_lookup_kernel` | Token ID -> embedding vector (with bounds check) |

All compute FP32 internally, convert BF16 at output boundaries only.

---

## Engine Data Flow (inference.cpp)

### B=1 Forward (forward_b1)

```
embed_lookup -> hidden (FP32)

for each layer:
  [layer 0: rms_norm_bf16(cur) -> bf16_a]
  [layer 1+: bf16_a already set by previous layer's fused add+norm]

  MATVEC_B1_NOPATCH(wq, bf16_a) -> q_buf
  MATVEC_B1_NOPATCH(wk, bf16_a) -> k_buf
  MATVEC_B1_NOPATCH(wv, bf16_a) -> v_buf
  rope_store_kv(q, k, v -> kv_cache)    // fused RoPE + KV store
  attention(q, kv_cache) -> bf16_a       // outputs BF16 directly (no fp32_to_bf16)
  MATVEC_B1_NOPATCH(wo, bf16_a) -> res
  add_rms_norm_bf16(res, cur) -> bf16_a  // fused add + norm for FFN
  split12_dual(gate+up, bf16_a) -> ffn_gate, ffn_up  // fused gate+up
  + patches for gate and up
  silu_mul_bf16(gate, up) -> bf16_b
  MATVEC_B1_NOPATCH(w_down, bf16_b) -> cur
  add_rms_norm_bf16(cur, res, NEXT_attn_norm) -> bf16_a  // cross-layer fusion!

MATVEC_B1_NOPATCH(output_proj, bf16_a) -> logits
argmax(logits) -> next_token
```

### MATVEC_B1_NOPATCH Macro

Uses split12 kernel (.sm.bin required).
Patches applied separately via batched `apply_patches_v2`.

### B=4 / B=8 Forward

Same structure but uses `BATCH4_MATVEC` / `BATCH8_MATVEC` macros.
Batch kernels decode weight once, multiply by B activation vectors.

---

## Tokenizer (tokenizer.cpp)

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

---

## Model Conversion

### From HuggingFace safetensors (Mistral)
```bash
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/
```

### From GGUF (Llama 3.1)
GGUF tensor shapes use GGML convention: `shape=[ne[0], ne[1]]` where ne[0] is the **contiguous dimension** (columns). Reshape as `[ne[1], ne[0]]` in C order -- NO transpose needed.

### Per-tensor .dims Format
```
M K num_patches base_exp
```
Example: `4096 4096 5358 107`

---

## Build & Run

```bash
# Build packer
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c

# Build engine -- NVIDIA (RTX 5070 Ti, sm_120)
cd engine && ln -sf kernels.hip kernels.cu && ln -sf ../decompress_v2.hip decompress_v2.cu
nvcc -O3 -arch=sm_120 -I.. -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.cu decompress_v2.cu ../nvidia_kernels.cu ../nvidia_kernels_v3.cu \
  -lcublas -lsentencepiece -lcuda -std=c++17

# Build engine -- AMD (MI50, gfx906)
cd engine && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17

# Run
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 ./turbo-engine <model_dir> "<prompt>" <max_tokens> [batch_size]
```

### Environment Variables

| Variable | Default | Effect |
|----------|:-------:|--------|
| `CUDA_VISIBLE_DEVICES=N` / `HIP_VISIBLE_DEVICES=N` | all | Select GPU |
| `TURBO_FAST=1` | off | Pre-compute escape tables. **Recommended.** +10% speed, +361 MB VRAM |
| `TURBO_CTX=N` | 8192 | Max context length |
| `TURBO_PROFILE=1` | off | Per-token timing breakdown |
| `TURBO_KERNEL=1\|2\|3` | auto | Force NVIDIA kernel version (V1/V2/V3 TMA) |
| `TURBO_CUBLAS=1` | off | Force cuBLAS path for all tensors (debug) |

---

## Benchmarking Guide

### Single-User (B=1)
```bash
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 ./engine/turbo-engine \
  models/mistral-7b-instruct-turbo "[INST] Write an essay about AI. [/INST]" 200

# With profiling
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 TURBO_PROFILE=1 ./engine/turbo-engine \
  models/mistral-7b-instruct-turbo "[INST] Write an essay about AI. [/INST]" 200
```

### Multi-User (B=32, 256)
```bash
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 ./engine/turbo-engine \
  models/mistral-7b-instruct-turbo "[INST] Write an essay about AI. [/INST]" 200 256
```

### What to Check
1. **Verify output is coherent** -- not garbage. If nonsensical, the benchmark is invalid.
2. **Use a non-display GPU** -- the monitor GPU has lower available bandwidth.
3. **Run 3 times** -- take the median. First run may be slower (cold caches).
4. **Match token count** -- compare same number of generated tokens across engines.

### Reading Profile Output
```
[PROFILE] 100 tok: mv 154.2 (92%) | attn 7.7 norm 4.2 silu 0.8 misc 0.0 | total 166.9 (59.9 t/s)
```
| Field | Meaning |
|-------|---------|
| `mv 154.2 (92%)` | Matvec time in ms (% of total). Constant regardless of context |
| `attn 7.7` | Attention time. Grows linearly with context (~0.065ms per token) |
| `norm 4.2` | RMSNorm + residual add |
| `silu 0.8` | SiLU activation |
| `total 166.9 (59.9 t/s)` | Total ms per 10 tokens (speed) |

---

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

---

## Known Issues (All Fixed)

| Issue | Status | Fix |
|-------|--------|-----|
| V3 TMA garbage for large-vocab models (Llama 128K vocab) | **Fixed** | cuBLAS weight_buf was over-allocated (~2GB), causing V3 OOM. Now dynamically scans all weights |
| forward_b1/b4/b8 used stream=0 (race with sampling) | **Fixed** | Changed to `state->stream` (2026-04-02) |
| Silent garbage on GPU OOM (no malloc error checking) | **Fixed** | All hipMalloc/cudaMalloc calls wrapped with GPU_CHECK macro (2026-04-03) |
| Killed process leaves GPU memory dirty for 5-15s | **Fixed** | SIGTERM/SIGINT handler calls free_inference_state + free_model (2026-04-03) |

---

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

---

## File Map

~5,500 lines of C++/CUDA/Python.

| File | Lines | Purpose |
|------|------:|---------|
| `decompress_v2.hip` | 1312 | All GPU matvec kernels (split12) |
| `nvidia_kernels.cu` | 586 | NVIDIA fused decode+GEMM (V1/V2/V3 TMA) |
| `engine/kernels.hip` | 1171 | Non-matvec GPU kernels (norm, rope, attention, silu, argmax) |
| `engine/inference.cpp` | 732 | Forward pass + generation loop |
| `engine/inference.h` | 47 | InferenceState struct (buffers, streams, events) |
| `engine/model.cpp` | 303 | Model loader + escape table builder |
| `engine/model.h` | 78 | CompressedWeight struct (packed + split_sm/split_gr + escape) |
| `engine/tokenizer.cpp` | 379 | Auto-detect + BPE encode/decode with byte-level mapping |
| `engine/tokenizer.h` | 16 | Tokenizer struct (type 0=sentencepiece, 1=HF BPE) |
| `engine/sampler.cpp` | 32 | Greedy + batched argmax |
| `engine/sampler.h` | 11 | Sampler interface |
| `engine/main.cpp` | 104 | CLI entry point + signal handler |
| `engine/convert_model.py` | 207 | HF safetensors -> turbo format |
| `split12_pack.c` | 128 | C packer library (find_base_exp + pack) |
| `gpu_compat.h` | 100 | AMD/NVIDIA kernel compatibility layer |

---

## Acknowledgements

The V3 fused decode+GEMM kernel uses tensor core patterns inspired by [ZipServ / ZipGEMM](https://github.com/HPMLL/ZipServ_ASPLOS26) (Fan et al., ASPLOS 2026). The core compression (Split12 encoding, 1-ADD decode) is independently developed.
