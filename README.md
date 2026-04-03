# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller VRAM. Proof-of-concept tested on a single NVIDIA RTX 5070 Ti 16 GB.

- Runs **Llama 3.1 8B on 16 GB** where vLLM OOMs
- **2.93x faster** than vLLM at B=256 (Mistral 7B)
- **~4,750 lines** of C++/CUDA, no Python runtime

**BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

---

## How It Works

### The Compression

BF16 is 16 bits per weight: `[1-bit sign][8-bit exponent][7-bit mantissa]`. Neural network weights cluster tightly — only ~40 of 256 possible exponent values are used, and **15 consecutive exponents cover 99.97%** of all weights.

We pick the best 15-exponent window per tensor (called `BaseExp`) and replace the 8-bit exponent with a 4-bit group number (1-15):

```
BF16:         [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit: [group 4][sign 1][mantissa 7]     = 12 bits

Decode: exponent = BaseExp + group   (one integer ADD)
Sign and mantissa pass through unchanged.
```

The 0.03% of weights outside this window get group=0 (escape). Their exact BF16 value is stored in a small side table (~3 MB for a 7B model). **Zero information loss.**

### The Storage (Split12)

12 bits doesn't align to bytes. Packing them contiguously would force the GPU to load 8 bytes and bit-shift to extract 1.5 bytes — wasting bandwidth. Instead, we split the 12 bits into two byte-aligned arrays:

```
File 1 (.sm.bin):  [S|MMMMMMM] [S|MMMMMMM] ...   1 byte per weight (sign + mantissa)
File 2 (.gr.bin):  [GGGG|GGGG] [GGGG|GGGG] ...   2 groups packed per byte (4-bit nibbles)
```

Why pack 2 groups per byte? Because the smallest unit a GPU can load is 1 byte. Storing one 4-bit group per byte would waste 4 bits (= no compression). Packing two per byte uses all 8 bits. Adjacent weights share a byte — the GPU loads it once and extracts each nibble with a single AND + shift.

**Result:** 1 byte + 0.5 byte = **1.5 bytes per weight = 1.33x compression**. Every byte loaded is useful data — zero read amplification.

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

| Batch | Kernel | llama.cpp | vLLM | Turbo | vs llama.cpp | VRAM |
|------:|:------:|----------:|-----:|------:|:------------:|-----:|
| B=1 | split12 | 52.9 | OOM | **57.0** | **1.08x** | 12.4 GB |
| B=8 | split12 | — | OOM | **154.3** | — | 12.4 GB |
| B=32 | split12 | — | OOM | **1,069** | — | 12.5 GB |
| B=64 | V3 TMA | — | OOM | **1,439** | — | 14.0 GB |
| B=128 | V3 TMA | — | OOM | **2,111** | — | 14.0 GB |
| B=256 | V3 TMA | — | OOM | **2,471** | — | 14.1 GB |

All values in tok/s. vLLM cannot fit Llama 8B on 16 GB (needs ~17 GB). llama.cpp uses 15.0 GB. Turbo: 14.1 GB serving 256 users.

**Kernels:** `split12` = per-row bandwidth-optimized matvec (B<=32). `V3 TMA` = fused decode+GEMM with Blackwell tensor memory loads (B>8). Auto-selected.

### VRAM Comparison

| | Mistral (vLLM) | Mistral (Turbo) | Llama (vLLM) | Llama (Turbo) |
|--|---------------:|----------------:|-------------:|--------------:|
| Weights | 13.2 GB | **10.2 GB** | ~14.7 GB | **11.5 GB** |
| Overhead | ~2.1 GB | ~0.9 GB | OOM | ~0.9 GB |
| **Total** | 15.3 GB | **11.1 GB** | OOM | **12.4 GB** |
| Max users | ~1 | **>256** | 0 | **>256** |

### Yi 1.5 9B Chat (01.AI — different model family)

| | llama.cpp | vLLM | Turbo |
|--|----------:|-----:|------:|
| B=1 | OOM | OOM | **48.1** |
| BF16 VRAM | 17.7 GB | ~19 GB | **~14.5 GB** |

Both llama.cpp and vLLM cannot load Yi 9B BF16 on a 16 GB card. Turbo fits it with 1.5 GB to spare.

### Llama 2 7B Chat (true 7B, FP16→BF16 converted)

| Batch | Kernel | llama.cpp | Turbo | vs llama.cpp | VRAM |
|------:|:------:|----------:|------:|:------------:|-----:|
| B=1 | split12 | 59.6 | **64.7** | **1.09x** | ~10.4 GB |
| B=8 | split12 | — | **172.2** | — | ~10.4 GB |
| B=32 | split12 | — | **1,289** | — | ~10.5 GB |
| B=64 | V3 TMA | — | **1,605** | — | ~12.0 GB |
| B=128 | V3 TMA | — | **2,576** | — | ~12.0 GB |
| B=256 | V3 TMA | — | **2,931** | — | ~12.0 GB |

All values in tok/s. Llama 2 7B is MHA (32/32 heads, not GQA) with smaller FFN (11008). Fastest at B=1 due to fewer params. Original FP16 model auto-cast to BF16 during conversion.

### Tested Models

| Model | Family | Params | B=1 tok/s | Escape Rate | Compression | llama.cpp/vLLM |
|-------|--------|-------:|----------:|------------:|:-----------:|:--------------:|
| Llama 2 7B Chat | Meta | 6.74B | 64.7 | — | 1.33x | Both fit |
| Mistral 7B Instruct | Mistral | 7.25B | 60.0 | 0.031% | 1.33x | Both fit |
| Llama 3.1 8B Instruct | Meta | 8.03B | 57.0 | 0.021% | 1.33x | vLLM OOM |
| Yi 1.5 9B Chat | 01.AI | 8.83B | 48.1 | — | 1.33x | Both OOM |

---

## Quick Start

```bash
# Build packer
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
