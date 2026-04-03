# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller VRAM. Proof-of-concept tested on a single NVIDIA RTX 5070 Ti 16 GB.

- Runs **Llama 3.1 8B on 16 GB** where vLLM OOMs
- **2.93x faster** than vLLM at B=256 (Mistral 7B)
- **~4,750 lines** of C++/CUDA, no Python runtime

**BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

---

## How It Works

BF16 weights have sparse exponents — only ~40 of 256 possible values are used, and 15 consecutive exponents cover 99.97%. We replace the 8-bit exponent with a 4-bit group code:

```
BF16:        [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit: [group 4][sign 1][mantissa 7]    = 12 bits

Decode: exponent = BaseExp + group  (one integer ADD)
```

**1.33x compression, zero information loss.** The 0.03% of outlier values are stored exactly in a small CSR escape table (~3 MB for 7B model). Stored as two byte-aligned arrays for zero HBM read amplification.

---

## Benchmarks

**Hardware:** Single GPU — NVIDIA RTX 5070 Ti 16 GB (Blackwell, 896 GB/s). All results with 200-token generation, output verified coherent.

### Mistral 7B Instruct

| Batch | Kernel | llama.cpp | vLLM | Turbo | vs llama.cpp | vs vLLM | VRAM |
|------:|:------:|----------:|-----:|------:|:------------:|:-------:|-----:|
| B=1 | split12 | 55.7 | 54.7 | **60.0** | **1.08x** | **1.10x** | 11.1 GB |
| B=8 | split12 | — | 414.6 | **162.6** | — | — | 11.1 GB |
| B=32 | split12 | — | 694.2 | **1,136** | — | **1.64x** | 11.2 GB |
| B=64 | V3 TMA | — | 853 | **1,514** | — | **1.77x** | 12.7 GB |
| B=128 | V3 TMA | — | 942 | **2,197** | — | **2.33x** | 12.7 GB |
| B=256 | V3 TMA | — | 872 | **2,554** | — | **2.93x** | 12.7 GB |

All values in tok/s. llama.cpp uses 13.5 GB. vLLM uses 15.3 GB (max ~1 user). Turbo serves 256 users at 12.7 GB.

### Llama 3.1 8B Instruct

| Batch | Kernel | llama.cpp | vLLM* | Turbo | vs llama.cpp | vs vLLM* | VRAM |
|------:|:------:|----------:|------:|------:|:------------:|:--------:|-----:|
| B=1 | split12 | 52.9 | ~50 | **57.0** | **1.08x** | **~1.14x** | 12.4 GB |
| B=8 | split12 | — | ~380 | **154.3** | — | — | 12.4 GB |
| B=32 | split12 | — | ~640 | **1,069** | — | **~1.67x** | 12.5 GB |
| B=64 | V3 TMA | — | ~780 | **1,439** | — | **~1.84x** | 14.0 GB |
| B=128 | V3 TMA | — | ~860 | **2,111** | — | **~2.45x** | 14.0 GB |
| B=256 | V3 TMA | — | ~800 | **2,471** | — | **~3.09x** | 14.1 GB |

All values in tok/s. *vLLM cannot fit Llama 8B on 16 GB — values estimated from Mistral scaling ratio. llama.cpp uses 15.0 GB. Turbo: 14.1 GB serving 256 users.

**Kernels:** `split12` = per-row bandwidth-optimized matvec (B<=32). `V3 TMA` = fused decode+GEMM with Blackwell tensor memory loads (B>8). Auto-selected.

### VRAM Comparison

| | Mistral (vLLM) | Mistral (Turbo) | Llama (vLLM) | Llama (Turbo) |
|--|---------------:|----------------:|-------------:|--------------:|
| Weights | 13.2 GB | **10.2 GB** | ~14.7 GB | **11.5 GB** |
| Overhead | ~2.1 GB | ~0.9 GB | OOM | ~0.9 GB |
| **Total** | 15.3 GB | **11.1 GB** | OOM | **12.4 GB** |
| Max users | ~1 | **>256** | 0 | **>256** |

### Tested Models

| Model | B=1 tok/s | Escape Rate | Compression |
|-------|----------:|------------:|:-----------:|
| Mistral 7B Instruct | 60.0 | 0.031% | 1.33x |
| Llama 3.1 8B Instruct | 57.0 | 0.021% | 1.33x |

---

## Quick Start

```bash
# Build packers
gcc -O3 -shared -fPIC -o structured12_pack.so structured12_pack.c
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c

# Build engine (NVIDIA — HIP kernels are cross-compatible via gpu_compat.h)
cd engine
ln -sf kernels.hip kernels.cu
ln -sf ../decompress_v2.hip decompress_v2.cu
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

| Variable | Default | Effect |
|----------|:-------:|--------|
| `CUDA_VISIBLE_DEVICES=N` | all | Select GPU |
| `TURBO_FAST=1` | off | Pre-compute escape tables. **Recommended.** +10% speed, +361 MB VRAM |
| `TURBO_CTX=N` | 2048 | Max context length (tokens) |
| `TURBO_PROFILE=1` | off | Print per-token timing breakdown |

**Debug only** (no need to set these):

| Variable | Effect |
|----------|--------|
| `TURBO_KERNEL=N` | Override fused GEMM kernel at B>8: 1=V1, 2=V2 cp.async, 3=V3 TMA. Default auto-selects V3 |
| `TURBO_CUBLAS=1` | Force cuBLAS path for all matmuls (slow, for correctness testing) |

---

## File Map

| File | Purpose |
|------|---------|
| `gpu_compat.h` | AMD/NVIDIA kernel compatibility layer |
| `decompress_v2.hip` | Split12 per-row matvec kernels (B=1/4/8) |
| `nvidia_kernels.cu` | NVIDIA fused decode+GEMM (V1/V2/V3 TMA) |
| `engine/inference.cpp` | Forward pass + generation loop |
| `engine/kernels.hip` | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/model.cpp` | Model loader + escape table builder |
| `engine/tokenizer.cpp` | Sentencepiece + HF BPE auto-detect |
| `engine/convert_model.py` | BF16 safetensors to Turbo format converter |

~4,750 lines of C++/CUDA.
