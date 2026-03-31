# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller, up to 1.95x faster inference.

## How It Works

### The Problem

LLM inference is **memory-bandwidth bound**: each generated token reads the entire weight matrix from GPU HBM. A 7B model reads ~14 GB of BF16 weights per token. The GPU's compute units sit idle waiting for data.

Quantization (INT4/INT8) solves this by reading less data, but **destroys precision**. Every quantized model has measurable quality loss.

### The Codebook — Simple Example

Imagine a weight tensor with these values:

```
Original BF16: [0.031, -0.016, 0.031, 0.031, -0.016, 0.062, 0.031, -0.016, 0.099, 0.031]
```

**Step 1 — Count frequencies:**
```
0.031  appears 5 times (50%)  ← most common
-0.016 appears 3 times (30%)
0.062  appears 1 time  (10%)
0.099  appears 1 time  (10%)  ← rare
```

**Step 2 — Build codebook** (top 4095 values get indices, index 4095 = escape):
```
Codebook[0] = 0.031     ← most frequent
Codebook[1] = -0.016
Codebook[2] = 0.062
(0.099 doesn't fit → gets escape sentinel 4095)
```

**Step 3 — Replace values with 12-bit indices:**
```
Original:  [0.031, -0.016, 0.031, 0.031, -0.016, 0.062, 0.031, -0.016, 0.099,  0.031]
Indices:   [  0,      1,     0,     0,      1,     2,     0,      1,    4095,     0   ]
                                                                         ↑ escape!
```

**Step 4 — Pack as 12-bit stream:**
```
Original: 10 values × 16 bits = 160 bits
Packed:   10 values × 12 bits = 120 bits  ← 1.33x smaller
```

### The Escape Table — Handling Rare Values

The 0.08% of values that don't fit in the codebook get index 4095 (escape sentinel). Their correct values are stored separately:

```
In the kernel, when a thread reads index 4095:

  index = read_12bit(packed_data)

  if index != 4095:                          ← 99.92% of the time
      weight = codebook[index]               ← 1-cycle LDS lookup
  else:
      weight = escape_table[pointer++]       ← read exact BF16 value
```

The escape table uses CSR format (row offsets + column indices + correct values). The branch predictor gets the 99.92% case right — near zero overhead.

**Every single value is reconstructed exactly.** The codebook values ARE the original BF16 values — not approximations. The escapes store the exact BF16 bits. Bit-for-bit identical.

### Why BF16 Compresses So Well

BF16 has only 16 bits: 1 sign + 8 exponent + 7 mantissa = **65,536 possible values total**. Neural network weights use far fewer — we found only ~7,000 unique values per tensor and only 40 unique exponents across a 7B model. The top 7 exponents cover 96.7% of all weights.

This extreme concentration is why 4,095 codebook entries cover 99.92%.

### Why It's Faster (Batch Decode Amortization)

Reading 12 bits instead of 16 bits saves 25% HBM bandwidth, but the codebook lookup adds decode overhead. At B=1 (single user), the overhead exceeds the savings.

**The key innovation**: when serving B concurrent users, we **decode each weight once** and multiply by B activation vectors:

```
B=1:  read 12-bit → decode → 1× multiply           (overhead dominates → 0.58x)
B=4:  read 12-bit → decode → 4× multiply            (1.43x faster than BF16)
B=8:  read 12-bit → decode → 8× multiply            (1.95x faster than BF16)
```

Same weight decode cost, B× the useful work. At B=8, decode is nearly free.

### The Fused GPU Kernel

Everything in a **single kernel launch** — no separate decompress step:

```
┌─────────────────────────────────────────────────────────────────┐
│  Fused Decode-Matvec Kernel (1 block per output row)           │
│                                                                 │
│  1. Load 4096-entry codebook into LDS (8 KB shared memory)    │
│  2. Compute escape pointer via warp shuffle prefix sum         │
│  3. For each pair of elements (2× unrolled):                  │
│     a. Branchless 64-bit read → extract 12-bit index          │
│     b. LDS codebook lookup → BF16 weight (1 cycle)            │
│     c. If escape (0.08%): read correct value from table       │
│     d. FMA: accumulator += weight × activation (×B for batch) │
│  4. Wavefront shuffle reduction → final dot product           │
└─────────────────────────────────────────────────────────────────┘
```

Weight goes from 12-bit packed HBM → LDS codebook → FP32 register → FMA → output. All in one kernel.

### What About Activations and KV Cache?

Only **weights** are compressed. Everything else stays native precision:

```
Weights:      12-bit compressed (read from HBM, decoded in kernel)
Activations:  BF16 for matvec input, FP32 internally
KV Cache:     BF16 (not compressed — only 2.6% of VRAM for 7B at ctx=2048)
Norms/RoPE:   FP32
```

The KV cache grows with context length but is too small and too dynamic (new entry every token) to benefit from codebook compression.

### What Makes This Different From Quantization

| | Turbo Lossless | INT4 (GPTQ/AWQ) | INT8 |
|---|---|---|---|
| Quality loss | **None (bit-exact)** | PPL +0.1-0.5 | PPL +0.01-0.1 |
| Compression | 1.33x | 4x | 2x |
| Calibration needed | No | Yes | Yes |
| Speed vs BF16 (B=8) | **1.95x faster** | ~3-4x faster | ~1.8x faster |
| Use case | Lossless serving | Max compression | Balanced |

We don't compete with INT4 on compression ratio. We offer something that doesn't exist elsewhere: **faster-than-BF16 inference with zero quality loss.**

---

## Benchmark Results

### End-to-End — Mistral 7B Instruct v0.2 on MI50 32GB

| Mode | tok/s total | tok/s/user | VRAM | vs llama.cpp BF16 (32.2 tok/s) |
|------|------------:|-----------:|-----:|:-------------------------------|
| B=1 | 18.7 | 18.7 | ~10 GB | 0.58x |
| B=4 | 46.0 | 11.5 | 10.3 GB | **1.43x faster** |
| **B=8** | **61.4** | **7.7** | **10.3 GB** | **1.91x faster, 1.32x less VRAM** |

Verified: 1+1=2, 3*3=9, 10*100=1000, ANU=Canberra, correct Python code generation. B=1 == B=4 == B=8 identical output.

### Context Length Scaling (B=8)

| Tokens | tok/s | Attention Mode |
|-------:|------:|:---------------|
| 100 | 62.9 | Naive (fast) |
| 300 | 59.2 | Naive |
| 1000+ | 53+ | Flash Attention (auto at 1024) |
| 2000+ | 41+ | Flash Attention |
| 20000 | supported | Flash Attention, 34 KB constant LDS |

### Projected Performance on Other Hardware

| GPU | Year | BW (TB/s) | Est. B=8 tok/s | vs native BF16 |
|-----|------|----------:|---------------:|:--------------:|
| MI50 32GB (measured) | 2019 | 1.0 | **63** | 1.95x |
| A100 80GB | 2020 | 2.0 | ~128 | 1.95x |
| H100 80GB | 2022 | 3.4 | ~211 | 1.95x |
| MI300X 192GB | 2023 | 5.3 | ~333 | 1.95x |
| B200 192GB | 2025 | 8.0 | ~503 | 1.95x |

The 1.95x speedup is hardware-independent — both compressed and BF16 scale with HBM bandwidth.

### Lossless-Only Competitors

| Project | Encoding | Compression | Speed vs BF16 | Status |
|---------|----------|------------:|:-------------:|:-------|
| **Turbo Lossless** | **12-bit codebook** | **1.33x** | **1.95x (B=8)** | **Measured on MI50** |
| ZipServ (2026) | 3-bit bitmap | 1.40x | 1.22x | L40S/RTX5090 |
| DFloat11 (2025) | Huffman exponent | 1.43x | 0.5x (slower) | NVIDIA CUDA |
| ZipNN (2025) | Stream split | 1.51x | no GPU kernel | CPU only |

### Compression Validation

| Model | Params | Disk CR | Lossless |
|-------|--------|--------:|:--------:|
| Llama 3.1 8B | 8B | 1.509x | 226/226 |
| Llama 3.1 70B | 70B | 1.516x | 33/33* |
| Mistral 7B | 7B | 1.503x | 226/226 |
| Mistral Large 123B | 123B | 1.503x | 49/49* |

*Sampled shards. All tested tensors 100% bit-perfect.

---

## Quick Start

### Prerequisites

- AMD MI50 32GB GPU with ROCm 6.x
- Python 3.10+ with `torch` (ROCm), `safetensors`
- `libsentencepiece` for tokenization
- BF16 safetensors model with `config.json` and `tokenizer.model` (sentencepiece)

### Build

```bash
# Compile C packer
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c

# Compile inference engine
cd engine
/opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17
```

### Convert Model

```bash
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/
```

### Run

```bash
# B=1 single user (18.7 tok/s)
HIP_VISIBLE_DEVICES=1 ./turbo-engine models/mistral-7b-instruct-turbo "What is 1+1?" 50 1

# B=4 four users (46 tok/s total)
HIP_VISIBLE_DEVICES=1 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Write a poem:" 200 4

# B=8 eight users (61 tok/s total — recommended)
HIP_VISIBLE_DEVICES=1 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Write a poem:" 200 8
```

### Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `HIP_VISIBLE_DEVICES` | `0`,`1`,`2`,`3` | Select GPU (avoid display GPU) |
| `TURBO_FAST` | `0` (default), `1` | `1` = pre-computed escape table, ~30% faster, +0.36 GB VRAM |
| `TURBO_CTX` | integer (default `2048`) | Max context length (KV cache size) |

---

## File Map

| File | Purpose |
|------|---------|
| **`decompress_v2.hip`** | Fused decode-matvec GPU kernels (B=1/2/4/8) |
| **`fixed12_pack.c`** | 12-bit codebook builder + packer |
| **`bench_fixed12.py`** | Per-tensor kernel benchmark + bit-exact verification |
| `engine/main.cpp` | CLI entry point |
| `engine/model.cpp` | Model loader + escape table builder |
| `engine/inference.cpp` | Forward pass (B=1/4/8) + generate loop |
| `engine/kernels.hip` | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/convert_model.py` | BF16 safetensors -> turbo format converter |
| `engine/tokenizer.cpp` | Sentencepiece tokenizer |
| `engine/sampler.cpp` | GPU argmax sampling |
| `expsplit_pack.c` | Research: exponent-split encoder (future NVIDIA optimization) |

## Architecture

```
BF16 safetensors ──► convert_model.py ──► turbo format (12-bit packed + CSR escapes)
                                                │
                                           turbo-engine
                                                │
                            ┌───────────────────┼───────────────────┐
                            │                   │                   │
                       model.cpp          inference.cpp        kernels.hip
                       (load weights,     (forward pass,       (RMSNorm, RoPE,
                        build escape       B=1/4/8 dispatch,    Flash Attention,
                        tables)            generate loop)       SiLU, argmax)
                            │                   │                   │
                            └───────────────────┼───────────────────┘
                                                │
                                        decompress_v2.hip
                                        (fused 12-bit decode + matvec,
                                         LDS codebook, batch amortization)
```

## Limitations

- **Sentencepiece only** — no tiktoken (Llama 3.1 not supported yet)
- **Single GPU** — multi-GPU pipeline in progress
- **AMD MI50 tested** — NVIDIA port needs warp size 64→32 (~20 lines)
- **Not for prefill** — compute-bound, our method helps memory-bound decode only
- **Not for INT4/GGUF** — input must be BF16 safetensors

## Production Path

The standalone engine proves the concept. The real deployment target is as a **vLLM/TGI backend**:

```
Current:  vLLM → rocBLAS GEMV (reads 16-bit weights)
Future:   vLLM → Turbo kernel (reads 12-bit weights, same output, 1.33x less VRAM)
```
