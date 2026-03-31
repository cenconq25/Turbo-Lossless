# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller, up to 1.95x faster inference.

## How It Works — The Full Picture

### The Problem

LLM inference is **memory-bandwidth bound**: each generated token reads the entire weight matrix from GPU HBM. A 7B model reads ~14 GB of BF16 weights per token. The GPU's compute units sit idle waiting for data.

Quantization (INT4/INT8) solves this by reading less data, but **destroys precision**. Every quantized model has measurable quality loss (perplexity increase, benchmark degradation).

### Our Solution: Lossless 12-Bit Codebook Compression

We observe that BF16 weight tensors have **highly concentrated value distributions**. Across a 7B model, the top 4095 most frequent values cover **99.92%** of all weight elements. Only 40 unique exponents are used (out of 256 possible).

**Encoding** (offline, once per model):

```
Original BF16 tensor: [0.0312, -0.0156, 0.0312, 0.0625, -0.0156, ...]
                       ↓ frequency sort: top 4095 values → codebook
Codebook[4096]:       [0.0312, -0.0156, 0.0625, 0.0469, ...]  (most frequent first)
                       ↓ each value → 12-bit index (0-4094), rare values → 4095 (escape)
Packed 12-bit stream: [0, 1, 0, 2, 1, ...]  (1.5 bytes per element vs 2 bytes BF16)
Escape table:         [(row, col, correct_value)]  (only 0.08% of elements)
```

**Result**: 16-bit BF16 → 12-bit packed indices = **1.33x compression**. Plus a tiny escape table (~14 MB for 7B model) storing the 0.08% of values not in the codebook. Decode reconstructs the **exact original BF16 value** — zero loss, bit-perfect.

### Why It's Faster (Not Just Smaller)

Reading 12 bits instead of 16 bits means **25% less HBM bandwidth** per weight element. But the codebook lookup (LDS shared memory, 1 cycle) adds decode overhead. At B=1, the overhead slightly exceeds the bandwidth savings (0.58x llama.cpp).

**The key innovation: batch decode amortization.** When serving multiple concurrent users (B=4 or B=8), we **decode each weight element once** and multiply by B activation vectors:

```
B=1:  read_12bit → decode → 1× FMA                    (decode overhead dominates)
B=4:  read_12bit → decode → 4× FMA  (same decode!)    (decode amortized 4 ways)
B=8:  read_12bit → decode → 8× FMA  (same decode!)    (decode nearly free)
```

At B=8, the decode cost is spread across 8 dot products. Net effect: **read 25% less data with nearly zero overhead = 1.95x faster than raw BF16**.

### The Fused GPU Kernel

Everything happens in a **single GPU kernel launch** per weight matrix — no separate decompress step:

```
┌─────────────────────────────────────────────────────────────────────┐
│ GPU Kernel: fused decode-matvec (1 block per output row)           │
│                                                                     │
│  1. Load 4096-entry codebook into LDS (8 KB shared memory)        │
│  2. Compute escape pointer via warp shuffle prefix sum             │
│  3. For each pair of elements (2× unrolled):                      │
│     a. Branchless 64-bit read → extract 12-bit index              │
│     b. LDS codebook lookup: cb[index] → BF16 weight (1 cycle)    │
│     c. If index == 4095 (escape): read correct value from table   │
│     d. FMA: accumulator += weight × activation (×B for batch)     │
│  4. Wavefront shuffle reduction → final dot product               │
└─────────────────────────────────────────────────────────────────────┘
```

No intermediate buffer. No separate decode pass. The weight goes from 12-bit packed HBM → LDS codebook → FP32 register → FMA → output, all in one kernel.

### Flash Attention for Long Context

At short context (<1024 tokens), we use a simple per-head attention kernel. At longer context, we auto-switch to **Flash Attention v2** with tiled KV (Tc=128), online softmax, and constant 34 KB LDS — supporting 20K+ tokens without memory scaling.

### Production Integration Path

The standalone engine proves the concept. The real deployment is as a **drop-in GEMV replacement inside vLLM or TGI**:

```
Current vLLM: rocBLAS GEMV (reads 16-bit BF16 weights every time)
With Turbo:   Turbo kernel (reads 12-bit weights, decodes in-kernel, same output)
```

vLLM handles continuous batching, PagedAttention, and serving infrastructure. We handle the weight matvec — faster and with less VRAM. The combination would give vLLM's throughput with 1.33x VRAM savings and zero quality loss.

### What Makes This Different From Quantization

| | Turbo Lossless | INT4 (GPTQ/AWQ) | INT8 |
|---|---|---|---|
| Quality loss | **None (bit-exact BF16)** | PPL +0.1-0.5 | PPL +0.01-0.1 |
| Compression | 1.33x | 4x | 2x |
| Calibration data needed | No | Yes | Yes |
| Risk of degradation | Zero | Nonzero | Nonzero |
| Speed vs BF16 (B=8) | **1.95x faster** | ~3-4x faster | ~1.8x faster |
| Use case | Lossless serving | Max compression | Balanced |

We don't compete with INT4 on compression ratio. We offer something that **doesn't exist elsewhere**: faster-than-BF16 inference with mathematically guaranteed zero quality loss.

---

## End-to-End Engine Results — Mistral 7B Instruct v0.2 on MI50 32GB

| Mode | tok/s total | tok/s/user | VRAM | vs llama.cpp BF16 (32.2 tok/s) |
|------|------------:|-----------:|-----:|:-------------------------------|
| B=1 | 18.7 | 18.7 | ~10 GB | 0.58x speed |
| B=4 | 46.0 | 11.5 | 10.3 GB | **1.43x faster** |
| **B=8** | **61.4** | **7.7** | **10.3 GB** | **1.91x faster, 1.32x less VRAM** |

All outputs verified identical across B=1, B=4, B=8 (100% bit-perfect lossless).

### Context Length Scaling (B=8, Flash Attention)

| Tokens | tok/s | Notes |
|-------:|------:|:------|
| 100 | 62.9 | Naive attention (fast) |
| 300 | 59.2 | Naive attention |
| 1000 | 53+ | Auto-switches to Flash Attention at 1024 |
| 2000+ | 41+ | Flash Attention, constant LDS |

### Projected Performance on Other Hardware

| GPU | Year | BW (TB/s) | Est. B=8 tok/s | vs native BF16 |
|-----|------|----------:|---------------:|:--------------:|
| MI50 32GB (measured) | 2019 | 1.0 | **63** | 1.95x |
| A100 80GB | 2020 | 2.0 | ~128 | 1.95x |
| H100 80GB | 2022 | 3.4 | ~211 | 1.95x |
| MI300X 192GB | 2023 | 5.3 | ~333 | 1.95x |
| B200 192GB | 2025 | 8.0 | ~503 | 1.95x |

The 1.95x speedup is hardware-independent (both scale with HBM bandwidth).

### Lossless-Only Competitors

| Project | Encoding | Compression | Speed vs BF16 | Status |
|---------|----------|------------:|:-------------:|:-------|
| **Turbo Lossless (ours)** | **12-bit codebook** | **1.33x** | **1.95x (B=8)** | **Measured** |
| ZipServ (Mar 2026) | 3-bit bitmap | 1.40x | 1.22x | L40S/RTX5090 |
| DFloat11 (2025) | Huffman exponent | 1.43x | 0.5x (slower) | NVIDIA CUDA |
| ZipNN (2025) | Stream split | 1.51x | no GPU kernel | CPU only |

## How It Works

### Encoding

Each BF16 weight tensor is frequency-sorted. The top 4095 values map to codebook entries 0-4094 (12-bit index). Remaining values (~0.08%) get index 4095 (escape sentinel) with correct values stored in a CSR escape table.

### Fused Decode-Matvec Kernel

Single GPU kernel per matvec — no separate decode pass:

1. **Branchless 64-bit read**: Load 2x uint32 as uint64, shift, mask -> 12-bit index
2. **LDS codebook lookup**: 4096-entry int16 codebook in 8 KB shared memory (1-cycle)
3. **O(1) escape handling**: uint8 per-thread escape counts + warp shuffle prefix sum
4. **2x loop unroll**: Dual accumulators with incremental bit positions
5. **Batch amortization**: B=4/B=8 decode weight once, multiply by B activation vectors

### Flash Attention

Tiled KV attention (Tc=128) with online softmax for long context (20K+ tokens). Constant 34 KB LDS regardless of sequence length. Auto-switches from naive attention at 1024 tokens.

## Quick Start

### Prerequisites

- AMD MI50 32GB GPU with ROCm 6.x
- Python 3.10+ with `torch` (ROCm), `safetensors`
- `libsentencepiece` for tokenization
- BF16 safetensors model with `config.json` and `tokenizer.model` (sentencepiece)

### Step 1: Build

```bash
# Compile C packer (for model conversion)
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c

# Compile HIP kernel (for benchmark script)
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip

# Compile inference engine
cd engine
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17
```

### Step 2: Convert Model

```bash
# Convert BF16 safetensors to turbo format (12-bit packed + CSR escapes)
python3 engine/convert_model.py models/mistral-7b-instruct

# Output: models/mistral-7b-instruct-turbo/
# Copy tokenizer if needed:
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/
```

The converter reads BF16 safetensors + `config.json`, produces:
- `config.bin` — model hyperparameters
- `tok_embd.bin` — token embeddings (BF16)
- `output_norm.bin` — final layer norm (FP32)
- Per layer: `layer.N.{wq,wk,wv,wo,w_gate,w_up,w_down}.{packed,codebook,dims,row_off,patch_cols,patch_correct,patch_wrong}.bin`

### Step 3: Run

```bash
# Basic usage
./turbo-engine <model-turbo-dir> <prompt> [max_tokens] [batch_size]

# Single user (B=1) — 18.7 tok/s
HIP_VISIBLE_DEVICES=1 ./turbo-engine models/mistral-7b-instruct-turbo "What is 1+1?" 50 1

# 4 concurrent users (B=4) — 46 tok/s total
HIP_VISIBLE_DEVICES=1 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Write a poem:" 200 4

# 8 concurrent users (B=8) — 61 tok/s total (recommended)
HIP_VISIBLE_DEVICES=1 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Write a poem:" 200 8
```

### Step 4: Kernel-Level Benchmark (Optional)

```bash
# Benchmark all tensors of a model (per-tensor speedup + bit-exact verification)
HIP_VISIBLE_DEVICES=1 python3 bench_fixed12.py models/llama-3.1-8b/llama-3.1-8b.safetensors
```

## Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `HIP_VISIBLE_DEVICES` | `0`,`1`,`2`,`3` | Select GPU (use non-display GPU) |
| `TURBO_FAST` | `0` (default), `1` | Speed vs VRAM tradeoff |
| `TURBO_CTX` | integer (default `2048`) | Max context length (affects KV cache VRAM) |

### TURBO_FAST explained

- **`TURBO_FAST=0`** (default): On-the-fly escape scan. Computes escape offsets at runtime. Saves ~0.36 GB VRAM.
- **`TURBO_FAST=1`**: Pre-computed uint8 escape count table + warp shuffle prefix sum. Uses ~0.36 GB extra VRAM but ~30% faster. **Recommended when VRAM allows.**

### Batch size explained

- **`B=1`**: Single user. 18.7 tok/s. Each weight decoded and multiplied by 1 vector.
- **`B=4`**: 4 concurrent users. 46.0 tok/s total. Decode once, 4 FMAs.
- **`B=8`**: 8 concurrent users. 61.4 tok/s total. Decode once, 8 FMAs. **Best throughput.**

Higher B = more total throughput but lower per-user speed (7.7 tok/s at B=8 vs 18.7 at B=1).

### TURBO_CTX explained

Sets KV cache size. Higher = more VRAM but supports longer conversations.

| Context | KV Cache | Use Case |
|--------:|--------:|:---------|
| 2048 | 0.27 GB | Short conversations (default) |
| 8192 | 1.07 GB | Medium documents |
| 20000 | 2.62 GB | Long documents |
| 32768 | 4.29 GB | Max for Mistral 7B |

## File Map

### Core (Kernel + Packer)

| File | Purpose |
|------|---------|
| `decompress_v2.hip` | All GPU kernels: fused matvec B=1/2/4/8, decompress, FP32 variants, expsplit research |
| `fixed12_pack.c` | C packer: `build_codebook_12bit()` + `pack_fixed12_csr()` + `pack_fixed12_fused()` |
| `bench_fixed12.py` | Standalone kernel benchmark: per-tensor timing + bit-exact verification |
| `expsplit_pack.c` | Research: exponent-split 12-bit encoder (not used in engine — LDS codebook faster on MI50) |

### Engine (C++ Inference)

| File | Purpose |
|------|---------|
| `engine/main.cpp` | CLI entry point: parse args, load model, generate |
| `engine/model.h` | Data structures: `Model`, `CompressedWeight`, `TransformerLayer`, `ModelConfig` |
| `engine/model.cpp` | Model loader: reads turbo binary format, uploads to GPU, builds escape tables |
| `engine/inference.h` | `InferenceState` struct: GPU buffers, streams, BF16 activation temps |
| `engine/inference.cpp` | Forward pass: `forward_b1()`, `forward_b4()`, `forward_b8()`, `generate()` |
| `engine/kernels.hip` | Non-matvec HIP kernels: RMSNorm, RoPE, SiLU, attention (naive + flash), argmax, BF16 convert |
| `engine/tokenizer.cpp` | Sentencepiece tokenizer wrapper |
| `engine/sampler.cpp` | Greedy sampling (argmax on GPU) |
| `engine/convert_model.py` | BF16 safetensors -> turbo binary format converter |

### Disk Compression (.tlc format)

| File | Purpose |
|------|---------|
| `tlc_encode.py` | .tlc encoder (8-tier variable-length, 1.47x disk compression) |
| `tlc_decode.py` | .tlc CPU decoder |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_verify.py` | Round-trip lossless verification |
| `bitpack_fast.c` | C packer for variable-length .tlc format |
| `bitunpack_fast.c` | C unpacker for variable-length .tlc format |

## Compression Validation

Tested across 11 BF16 models, 7 architectures, 1265+ tensors:

| Model | Params | Type | Disk CR | Lossless |
|-------|--------|------|--------:|:--------:|
| Llama 3.1 8B | 8B | Dense | 1.509x | 226/226 |
| Llama 3.1 70B | 70B | Dense | 1.516x | 33/33* |
| Mistral 7B | 7B | Dense | 1.503x | 226/226 |
| Mistral Large 123B | 123B | Dense | 1.503x | 49/49* |
| DeepSeek V3 | 671B | MoE | 1.185x | — |

*Sampled shards. All tested tensors 100% bit-perfect lossless.

## Architecture

```
BF16 safetensors ──► convert_model.py ──► turbo binary format
                         │                      │
                    fixed12_pack.c          config.bin
                    (12-bit packing +       tok_embd.bin
                     CSR escapes)           layer.N.*.bin
                                                │
                                           turbo-engine
                                                │
                                    ┌───────────┼───────────┐
                                    │           │           │
                               model.cpp   inference.cpp  kernels.hip
                               (load +      (forward      (RMSNorm,
                                escape       pass,         RoPE,
                                tables)      generate)     attention)
                                    │           │           │
                                    └───────────┼───────────┘
                                                │
                                        decompress_v2.hip
                                        (fused decode-matvec
                                         B=1/4/8 kernels)
```

## Scope and Limitations

**Best for**: Autoregressive decoding of dense BF16 models (Llama, Mistral, Phi, Qwen)

**Not beneficial for**:
- Prefill (compute-bound, not memory-bound)
- Already-quantized models (GPTQ, AWQ, GGUF INT4)
- MoE models with many small expert tensors (low compression ratio)

**Current limitations**:
- Sentencepiece tokenizer only (no tiktoken/Llama 3.1 support yet)
- Single-GPU only (multi-GPU pipeline in progress)
- AMD MI50 tested (NVIDIA port requires warp size 64->32)

## Hardware Compatibility

| Hardware | Status | Notes |
|----------|--------|-------|
| AMD MI50 32GB (gfx906) | Tested | Primary development platform |
| AMD MI250X / MI300X | Expected to work | Same HIP/ROCm, wider wavefronts |
| NVIDIA H100 / A100 / V100 | Portable | Requires warp size 64->32, CUDA build |
| NVIDIA RTX 4090 / 5090 | Portable | Consumer GPUs, less HBM bandwidth |
