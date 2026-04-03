# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller VRAM. Tested on a single NVIDIA RTX 5070 Ti 16 GB.

**Highlights:**
- Runs **Llama 3.1 8B on 16 GB** where vLLM OOMs
- **2.93x faster** than vLLM at B=256 (Mistral 7B)
- **~4,500 lines** of C++/CUDA, no Python runtime

**BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

---

## How It Works

LLM inference is memory-bandwidth bound. A 7B model reads ~14 GB of BF16 weights per token. Quantization (INT4/INT8) reduces this but destroys precision.

**Our insight:** BF16 has 8-bit exponent but only uses ~40 of 256 values. 15 consecutive exponents cover **99.97%** of all weights. We replace the 8-bit exponent with a 4-bit group code:

```
Original BF16:  [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit:   [group 4][sign 1][mantissa 7]     = 12 bits

Decode: exponent = BaseExp + group  (one integer ADD)
```

**1.33x compression, zero information loss.** The 0.03% of values outside the 15-exponent window are stored exactly in a tiny CSR escape table.

Stored as two byte-aligned arrays (Split12 format) for zero HBM read amplification:
- Array 1: `[sign][mantissa]` = 1 byte/element
- Array 2: `[group]` = 0.5 byte/element (nibble-packed)

---

## Benchmarks (Single GPU: RTX 5070 Ti 16 GB)

All benchmarks measured with 200-token generation, output verified coherent.

### Mistral 7B Instruct

| Batch | Kernel | vLLM BF16 | Turbo 12-bit | vs vLLM | VRAM |
|------:|:------:|----------:|-------------:|:-------:|-----:|
| B=1 | split12 | 54.7 tok/s | **60.0 tok/s** | **1.10x** | 11.1 GB |
| B=8 | split12 | 414.6 | **162.6** | — | 11.1 GB |
| B=32 | split12 | 694.2 | **1,136** | **1.64x** | 11.2 GB |
| B=64 | V3 TMA | 853 | **1,514** | **1.77x** | 12.7 GB |
| B=128 | V3 TMA | 942 | **2,197** | **2.33x** | 12.7 GB |
| B=256 | V3 TMA | 872 | **2,554** | **2.93x** | 12.7 GB |

vLLM fills the entire 16 GB card (15.3 GB) and can serve only ~1 user. Turbo serves **256 users at 12.7 GB**.

### Llama 3.1 8B Instruct

| Batch | Kernel | vLLM BF16 | Turbo 12-bit | VRAM |
|------:|:------:|----------:|-------------:|-----:|
| B=1 | split12 | OOM | **57.0 tok/s** | 12.4 GB |
| B=8 | split12 | OOM | **154.3** | 12.4 GB |
| B=32 | split12 | OOM | **1,069** | 12.5 GB |
| B=64 | V3 TMA | OOM | **1,439** | 14.0 GB |
| B=128 | V3 TMA | OOM | **2,111** | 14.0 GB |
| B=256 | V3 TMA | OOM | **2,471** | 14.1 GB |

vLLM **cannot load** Llama 8B BF16 on a 16 GB card (needs ~17 GB). Turbo runs it at 14.1 GB serving 256 users.

Kernels: **split12** = per-row bandwidth-optimized matvec (B<=32), **V3 TMA** = fused decode+GEMM with Blackwell tensor memory loads (B>=64). Auto-selected based on batch size.

### VRAM: Turbo vs vLLM

|  | vLLM (Mistral) | Turbo (Mistral) | vLLM (Llama) | Turbo (Llama) |
|--|---------------:|----------------:|-------------:|--------------:|
| Model weights | 13.2 GB | **10.2 GB** | ~14.7 GB | **11.5 GB** |
| Runtime overhead | ~2.1 GB | **~0.9 GB** | OOM | **~0.9 GB** |
| **Total (B=1)** | **15.3 GB** | **11.1 GB** | **OOM** | **12.4 GB** |
| Max batch users | ~1 | **>256** | 0 | **>256** |

Turbo uses 57% less runtime overhead than vLLM (lean C++ vs Python/PyTorch).

### Tested Models

| Model | B=1 tok/s | Escape Rate | Compression | Status |
|-------|----------:|------------:|:-----------:|--------|
| Mistral 7B Instruct | 60.0 | 0.031% | 1.33x | Production |
| Llama 3.1 8B Instruct | 57.0 | 0.021% | 1.33x | Production |

---

## Quick Start

```bash
# Build packers
gcc -O3 -shared -fPIC -o structured12_pack.so structured12_pack.c
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c

# Build engine (NVIDIA)
cd engine && ln -sf kernels.hip kernels.cu && ln -sf ../decompress_v2.hip decompress_v2.cu
nvcc -O3 -arch=sm_120 -I.. -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.cu decompress_v2.cu ../nvidia_kernels.cu ../nvidia_kernels_v3.cu \
  -lcublas -lsentencepiece -lcuda -std=c++17

# Convert model
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/

# Run
CUDA_VISIBLE_DEVICES=0 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Hello" 200 8
```

### Environment Variables

| Variable | Effect |
|----------|--------|
| `CUDA_VISIBLE_DEVICES=N` | Select GPU |
| `TURBO_FAST=1` | Pre-computed escape counts (+10% speed, +361 MB VRAM) |
| `TURBO_CTX=N` | Max context length (default 2048) |
| `TURBO_PROFILE=1` | Per-token timing breakdown |
| `TURBO_KERNEL=1\|2\|3` | Fused GEMM kernel for B>=64: V1 baseline, V2 cp.async, **V3 TMA** (default). No effect at B<=32 |

B<=32 always uses split12 per-row matvec. B>=64 uses fused decode+GEMM (V3 TMA by default, override with `TURBO_KERNEL`).

---

## File Map

~4,450 lines of production code.

| File | Purpose |
|------|---------|
| `gpu_compat.h` | AMD/NVIDIA compatibility layer |
| `decompress_v2.hip` | GPU matvec kernels (split12 + structured12, B=1/4/8) |
| `nvidia_kernels.cu` | NVIDIA fused decode+GEMM (V1/V2/V3 TMA) |
| `engine/inference.cpp` | Forward pass + generate loop |
| `engine/kernels.hip` | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/model.cpp` | Model loader + escape table builder |
| `engine/tokenizer.cpp` | Sentencepiece + HF BPE auto-detect |
| `engine/convert_model.py` | BF16 safetensors to turbo format converter |
