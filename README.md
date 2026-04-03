# Turbo Lossless — 1.33x Smaller, 2.93x Faster, Decode with 1 ADD

### **1.33x** compression for most models. Up to **2.93x** faster than vLLM for multi-user (B=256). **3-7x fewer ops** than other lossless methods — just an ADD.

> **BF16 → 12-bit lossless. One integer ADD to decode. Zero precision loss.**

```
BF16:         [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit: [group 4][sign 1][mantissa 7]     = 12 bits

Decode: exponent = BaseExp + group   ← that's it. One ADD.
```

**1.33x smaller. Up to 2.93x faster than vLLM (at B=256). Runs models where competitors OOM.**

---

## Why It Works

Neural network weights cluster tightly — **15 consecutive BF16 exponents cover 99.97%** of all values. We replace the 8-bit exponent with a 4-bit group code. The 0.03% outliers get their exact value stored in a tiny escape table.

Stored as two byte-aligned arrays (**Split12**) — zero GPU read amplification:
```
.sm.bin:  [S|MMMMMMM] ...   1 byte per weight (sign + mantissa)
.gr.bin:  [GGGG|GGGG] ...   2 groups per byte (nibble-packed)
```

---

## Compared to Other Lossless BF16 Methods

| | **Turbo** | **ZipServ** | **DFloat11** | **ZipNN** | **NeuZip** | **Huff-LLM** |
|---|---|---|---|---|---|---|
| **Venue** | — | ASPLOS'26 | NeurIPS'25 | IEEE'25 | arXiv'24 | arXiv'25 |
| **Bits/weight** | **12.0 (fixed)** | ~11.3 | ~11.0 | ~11 | ~10.6 | ~11.6 |
| **Decode cost** | **1 ADD** | Bitmap+popcount | Huffman LUT | CPU zstd | ANS | CAM |
| **Escape rate** | **0.03%** | ~3% | 0% | 0% | 0% | 0% |
| **Fused decode?** | **Yes** (matvec) | **Yes** (tensor core) | No (separate) | No | No | ASIC only |
| **GPU decode** | Yes | Yes | Yes | No (CPU) | Yes | No (ASIC) |
| **Hardware** | **NVIDIA + AMD** | NVIDIA | NVIDIA | CPU | NVIDIA | Custom |

**Our trade-off:** We use 0.7 more bits/weight than ZipServ, but decode with 1 instruction instead of 5-8, have 100x fewer escapes, and run on NVIDIA, AMD, Intel — you name it.

---

## Benchmarks

Single GPU — NVIDIA RTX 5070 Ti 16 GB. All tok/s, 200-token generation, output verified.

### Single-User (B=1)

| Model | Params | llama.cpp | vLLM | Turbo | Speedup |
|-------|-------:|----------:|-----:|------:|--------:|
| Llama 2 7B | 6.74B | 59.6 | 43.9 | **64.7** | **1.47x** vs vLLM |
| Mistral 7B | 7.25B | 55.7 | 54.7 | **60.0** | **1.10x** vs vLLM |
| Llama 3.1 8B | 8.03B | 52.9 | OOM | **57.0** | **1.08x** vs llama.cpp |

### Multi-User (total tok/s)

| Model | Params | B=32 | B=64 | B=128 | B=256 | vLLM B=256 | Speedup |
|-------|-------:|-----:|-----:|------:|------:|-----------:|--------:|
| Llama 2 7B | 6.74B | 1,289 | 1,605 | 2,576 | **2,931** | 1,086 | **2.70x** |
| Mistral 7B | 7.25B | 1,136 | 1,514 | 2,197 | **2,554** | 872 | **2.93x** |
| Llama 3.1 8B | 8.03B | 1,069 | 1,439 | 2,111 | **2,471** | OOM | — |

### VRAM Usage + Overhead

| Model | Model VRAM | Overhead (B=1) | Total (B=1) | Total (B=256) | vLLM Total |
|-------|----------:|-----------:|------------:|--------------:|-----------:|
| Llama 2 7B | 10.1 GB | 1.2 GB | **11.3 GB** | OOM (MHA) | ~14.7 GB |
| Mistral 7B | 10.2 GB | 0.9 GB | **11.1 GB** | 12.7 GB | 15.3 GB |
| Llama 3.1 8B | 11.5 GB | 0.9 GB | **12.4 GB** | 14.1 GB | OOM |

Overhead = KV cache + escape tables + TURBO_FAST + activation buffers + CUDA context. Llama 2 7B uses MHA (32/32 heads) — 4x larger KV cache than GQA models, OOMs at B=256 on 16 GB.

---

## Compression Works on Everything

Tested across 11 models — LLMs up to 405B, MoE, image, and video:

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

## Quick Start

**One command** — auto-detects GPU, builds if needed, converts if needed:

```bash
# Single prompt
./turbo models/mistral-7b-instruct-turbo "What is the meaning of life?" 200

# Interactive — model loads once, stays in VRAM, answer prompts instantly
./turbo models/mistral-7b-instruct-turbo -i
```

Interactive mode loads the model **once** (~4s), then keeps it in VRAM. Every subsequent prompt goes straight to generation at full speed — no reloading:

```
  ✓ Model loaded in 4s — staying in VRAM

  ▶ What is gravity?

  turbo
  Gravity is a fundamental force of nature...
  ─────────────────────────────────────
  153 tokens  •  60.3 tok/s  •  2.73s

  ▶ What is DNA?          ← no reload, instant

  turbo
  DNA stands for deoxyribonucleic acid...
  ─────────────────────────────────────
  200 tokens  •  60.2 tok/s  •  3.52s
```

First run will auto-build the engine. To convert a HuggingFace model:
```bash
./turbo models/mistral-7b-instruct "Hello" 200    # auto-converts to turbo format
```

<details>
<summary>Manual build (if you prefer)</summary>

```bash
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c
cd engine
ln -sf kernels.hip kernels.cu && ln -sf ../decompress_v2.hip decompress_v2.cu
nvcc -O3 -arch=sm_120 -I.. -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.cu decompress_v2.cu ../nvidia_kernels.cu ../nvidia_kernels_v3.cu \
  -lcublas -lsentencepiece -lcuda -std=c++17
```
</details>

| Variable | Default | Effect |
|----------|:-------:|--------|
| `CUDA_VISIBLE_DEVICES=N` | all | Select GPU |
| `TURBO_FAST=1` | off | Pre-compute escape tables. **Recommended.** +10% speed |
| `TURBO_CTX=N` | 2048 | Max context length |
| `TURBO_PROFILE=1` | off | Per-token timing breakdown |
| `TURBO_KERNEL=1\|2\|3` | auto | Force NVIDIA kernel version (V1/V2/V3 TMA) |

BF16 and FP16 safetensors supported. No GGUF, no FP32, no quantized formats.

---

## Kernel Architecture

Two kernel families for different batch sizes. All decode Split12 weights on-the-fly with 1 ADD — no codebook, no shared memory lookup table.

### Per-Row Kernels (`decompress_v2.hip`) — B=1 to B=8

For small batches, matvec is bandwidth-bound: each output element is a single dot product. One block processes one (or two) rows. Each thread streams through the weight array, decodes inline, and accumulates. Portable across NVIDIA and AMD via `gpu_compat.h`.

| Kernel | Batch | What It Does Differently |
|--------|:-----:|--------------------------|
| `split12_matvec_v2` | B=1 | Baseline: 4x unrolled loop, pointer-stride addressing |
| `split12_matvec_v2_multirow` | B=1 | **2 rows per block** — both rows read the same activation vector, cutting activation memory traffic in half (+4%) |
| `split12_matvec_v2_dual` | B=1 | **Gate+up fused** — two different weight matrices (FFN gate and up) share one activation load. One kernel instead of two, half the activation reads |
| `split12_matvec_batch4` | B=4 | **4 activations × 1 weight decode** — decode each weight once, multiply by 4 activation vectors. Inline escape handling via warp-shuffle prefix sum (no separate patch kernel) |
| `split12_matvec_batch8` | B=8 | Same as batch4 but 8 activations. Uses **single accumulator** per batch element (not dual) to save 8 registers — critical for fitting in the register file |

Escape patches: B=1 uses a separate batched `apply_patches_v2` kernel (eliminates ~192 tiny launches per token). B=4/8 handle escapes inline during the main loop via `if (group == 0) val = escape_table[ptr++]`.

### Fused Decode+GEMM (`nvidia_kernels.cu`) — B>8, NVIDIA Only

For larger batches, we switch to tensor core GEMM. The challenge: weights are compressed, so we can't just call cuBLAS. Instead, we decode Split12 weights directly into tensor core registers and run PTX `mma.sync.aligned.m16n8k16` on the decoded BF16 values.

All three versions share the same compute pattern — **K-slice interleaving** from [ZipServ](https://github.com/HPMLL/ZipServ_ASPLOS26): within each K-tile, `decode(slice N+1)` runs in parallel with `mma.sync(slice N)`, hiding the 3-ALU-op decode behind tensor core math. Register double-buffering (`a[2][4]`, `b[2][2]`) enables this ping-pong without stalls.

The versions differ in **how data moves from HBM to shared memory** — which is the bottleneck.

#### V1 and V2 — Software `cp.async` Pipeline

V1 and V2 use the **same loading mechanism**: all threads cooperatively issue `cp.async.cg.shared.global` instructions (16-byte async copies from DRAM to shared memory, bypassing registers). Each thread loops over its portion of the tile: `for i = tid to tile_size/16 step blockDim`. Double-buffered K-tiles overlap loading with compute.

The only difference is **tile size and occupancy**:

| | V1 | V2 |
|--|----|----|
| **Threads** | 256 (8 warps) | 128 (4 warps) |
| **Tile M×K** | 128×64 | 64×64 |
| **Shared memory** | ~40 KB | ~20 KB |
| **Blocks per SM** | 1 | 2-3 (`__launch_bounds__(128, 3)`) |
| **Patch correction** | Fused in output write | Separate kernel |

V1's large tile means only 1 block fits per SM — when it stalls waiting for data, the SM idles. V2 halves the tile so 2-3 blocks fit: when block A stalls, block B runs. This occupancy trick is the same approach ZipServ uses to reach 2.21x over cuBLAS.

Both use `cp_async_commit()` / `cp_async_wait<N>()` for pipeline control and `__syncthreads()` for barrier synchronisation.

#### V3 — Blackwell Hardware TMA

V3 is a **fundamentally different architecture**, not just a tuning of V1/V2. It replaces the software `cp.async` pipeline with the Tensor Memory Accelerator (TMA) — dedicated hardware on SM90+/Blackwell GPUs that copies entire 2D tiles autonomously.

**How TMA loading works:**

1. **TMA descriptors** are created on the host via `cuTensorMapEncodeTiled()`, encoding the tensor's shape, strides, tile size, and swizzle pattern. Three descriptors: one each for sign-mantissa (64B swizzle), groups (32B swizzle), and activations (128B swizzle). Descriptors are copied to device memory once.

2. **One elected thread** per warp (via `elect.sync` PTX) issues `cp.async.bulk.tensor.2d.shared::cluster.global.tile` — a single instruction that tells TMA hardware to copy an entire tile. The other 127 threads do nothing for loading; they're free for decode and compute.

3. **Mbarrier synchronisation** replaces `__syncthreads()`. The elected thread calls `mbar_expect_tx(bytes)` to declare how many bytes are in flight. TMA hardware automatically signals the mbarrier (`mbarrier::complete_tx`) when the transfer finishes. Waiting threads spin on `mbar_wait(phase)` using hardware parity — lighter than software barriers.

4. **Swizzle patterns** are applied automatically by TMA during the copy to eliminate shared memory bank conflicts. The trade-off: all subsequent shared memory reads must apply XOR-based address translation to compensate:
   ```
   // SM: 64B swizzle → per-row XOR offset
   int swiz = ((row >> 1) & 3) << 4;
   uint32_t val = *(sm + row * TILE_K + ((col & ~3) ^ swiz));

   // B (activations): 128B swizzle
   uint32_t val = *(B + row * TILE_K*2 + (byte_col ^ ((row & 7) << 4)));
   ```

5. **Two-stage pipeline** alternates between mbar0/mbar1 with phase toggling, overlapping TMA loads of the next tile with compute on the current tile.

**Why V3 is fastest at B≥64:** TMA offloads all data movement to hardware. In V1/V2, threads spend cycles issuing `cp.async` loads and computing loop indices. In V3, threads spend 100% of their time on decode + `mma.sync`. At large B, the compute-to-load ratio is high enough for TMA's setup overhead (descriptors, elect_sync, mbarrier protocol) to pay off.

#### Summary: V1 vs V2 vs V3

| | **V1** | **V2** | **V3 (Blackwell TMA)** |
|---|---|---|---|
| **HBM → shared memory** | Software `cp.async` (16B chunks) | Software `cp.async` (16B chunks) | Hardware TMA (`cp.async.bulk.tensor.2d`) |
| **Who loads data** | All 256 threads cooperate | All 128 threads cooperate | 1 elected thread issues TMA; hardware does the rest |
| **Threads** | 256 (8 warps) | 128 (4 warps) | 128 (4 warps) |
| **Tile M×K** | 128×64 | 64×64 | 64×64 |
| **Shared memory** | ~40 KB | ~20 KB | ~22 KB |
| **Blocks per SM** | 1 | 2-3 | 2 |
| **Synchronisation** | `__syncthreads()` | `__syncthreads()` | Hardware mbarrier (parity-based, `mbar_wait`) |
| **Bank conflict avoidance** | Manual padding (`OUT_PAD+1`) | Manual padding | TMA hardware swizzle (64B/32B/128B per tensor) |
| **Shared mem reads** | Direct offset | Direct offset | XOR address translation (undo swizzle) |
| **Patch correction** | Fused in output write | Separate kernel | Fused in output write |
| **GPU requirement** | Any CUDA GPU | Any CUDA GPU | SM90+ (Hopper/Blackwell) |
| **Best for** | Fallback / debug | B=16..63 | **B≥64 (default)** |

### Auto-Selection

| Batch Size | Kernel | Why |
|:----------:|:-------|:----|
| B=1 | Per-row multirow + batched patches | Bandwidth-bound; tensor cores can't help at B=1 |
| B=4 | Per-row batch4 (inline escapes) | Still bandwidth-bound; per-row is simpler and fast |
| B=8 | Per-row batch8 (inline escapes) | Transition zone; per-row wins by avoiding GEMM overhead |
| B=16..63 | **V2** fused decode+GEMM | Enough columns to fill tensor cores; high occupancy wins |
| B≥64 | **V3 TMA** fused decode+GEMM | TMA hardware loading frees all threads for compute |

Override with `TURBO_KERNEL=1|2|3`.

### Engine Kernels (`engine/kernels.hip`)

Non-matvec kernels, portable across NVIDIA and AMD:

| Kernel | What It Does |
|--------|--------------|
| `rms_norm_bf16_batch` | RMSNorm with warp-shuffle reduction, outputs BF16 directly (no separate conversion) |
| `add_rms_norm_bf16_batch` | Residual add + RMSNorm → BF16 in one pass. Fused **across layers** — layer N's down-projection residual + layer N+1's attention norm in a single kernel (saves 31 launches per forward) |
| `silu_mul_bf16_batch` | SiLU(gate) × up → BF16. Three ops fused into one kernel |
| `rope_store_kv` | Applies RoPE rotation to Q and K, then stores K and V into the KV cache as BF16. One kernel instead of three |
| `flash_attention` | Flash Attention v2 with online softmax. Constant 34 KB shared memory regardless of sequence length (tiles K/V in blocks of 128) |
| `attention_all_heads` | Simpler attention for short sequences (< 1024 tokens). Loads full score vector into shared memory |
| `argmax` | Batched greedy sampling — 1 block per sequence, shared memory tree reduction |

All kernels compute in FP32 internally, convert to BF16 only at output boundaries.

---

## Acknowledgements

The V3 fused decode+GEMM kernel uses tensor core patterns inspired by [ZipServ / ZipGEMM](https://github.com/HPMLL/ZipServ_ASPLOS26) (Fan et al., ASPLOS 2026). The core compression (Split12 encoding, 1-ADD decode) is independently developed.

---

## File Map

~5,500 lines of C++/CUDA/Python.

| File | Lines | Purpose |
|------|------:|---------|
| `decompress_v2.hip` | 901 | Split12 per-row matvec kernels (B=1/4/8) |
| `engine/kernels.hip` | 937 | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/inference.cpp` | 732 | Forward pass + generation loop |
| `nvidia_kernels.cu` | 586 | NVIDIA fused decode+GEMM (V1/V2/V3 TMA) |
| `engine/tokenizer.cpp` | 363 | Sentencepiece + HF BPE auto-detect |
| `engine/model.cpp` | 303 | Model loader + escape table builder |
| `engine/convert_model.py` | 206 | BF16/FP16 safetensors → Turbo format |
| `split12_pack.c` | 128 | C packer library (find_base_exp + pack) |
| `gpu_compat.h` | 100 | AMD/NVIDIA kernel compatibility layer |
| `engine/main.cpp` | 104 | CLI entry point + signal handler |
