# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller VRAM. Proof-of-concept tested on a single NVIDIA RTX 5070 Ti 16 GB.

- **1.33x compression**, zero information loss
- **8-9% faster** than llama.cpp and vLLM at B=1
- **Up to 2.93x faster** than vLLM at B=256
- Runs models **where competitors OOM** (Llama 8B, Yi 9B on 16 GB)

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

Single GPU — NVIDIA RTX 5070 Ti 16 GB (Blackwell, 896 GB/s). All tok/s, 200-token generation, output verified coherent.

### Single-User Latency (B=1)

| Model | Params | llama.cpp | vLLM | Turbo | Speedup | Turbo VRAM |
|-------|-------:|----------:|-----:|------:|--------:|-----------:|
| Llama 2 7B | 6.74B | 59.6 | — | **64.7** | **1.09x** | 10.4 GB |
| Mistral 7B | 7.25B | 55.7 | 54.7 | **60.0** | **1.08x** | 11.1 GB |
| Llama 3.1 8B | 8.03B | 52.9 | OOM | **57.0** | **1.08x** | 12.4 GB |
| Yi 1.5 9B | 8.83B | OOM | OOM | **48.1** | — | ~14.5 GB |

### Batch Throughput (Mistral 7B)

| Batch | Kernel | Turbo | vLLM | vs vLLM |
|------:|:------:|------:|-----:|--------:|
| B=1 | split12 | **60** | 54.7 | **1.10x** |
| B=32 | split12 | **1,136** | 694 | **1.64x** |
| B=64 | V3 TMA | **1,514** | 853 | **1.77x** |
| B=128 | V3 TMA | **2,197** | 942 | **2.33x** |
| B=256 | V3 TMA | **2,554** | 872 | **2.93x** |

`split12` = per-row bandwidth-optimized matvec. `V3 TMA` = fused decode+GEMM with Blackwell tensor memory loads. Auto-selected.

### VRAM Comparison (16 GB card)

| Model | BF16 (llama.cpp) | BF16 (vLLM) | Turbo 12-bit | Fits? |
|-------|------------------:|-----------:|-------------:|:-----:|
| Llama 2 7B | 12.6 GB | ~14.7 GB | **10.4 GB** | All fit |
| Mistral 7B | 13.5 GB | 15.3 GB | **11.1 GB** | All fit |
| Llama 3.1 8B | 15.0 GB | OOM | **12.4 GB** | vLLM OOM |
| Yi 1.5 9B | OOM | OOM | **~14.5 GB** | Only Turbo |

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

# Convert model (supports BF16 and FP16 safetensors)
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

**Debug only:** `TURBO_KERNEL=N` (override fused GEMM at B>8: 1/2/3), `TURBO_CUBLAS=1` (force cuBLAS path).

---

## File Map

~4,750 lines of C++/CUDA.

| File | Purpose |
|------|---------|
| `gpu_compat.h` | AMD/NVIDIA kernel compatibility layer |
| `decompress_v2.hip` | Split12 per-row matvec kernels (B=1/4/8) |
| `nvidia_kernels.cu` | NVIDIA fused decode+GEMM (V1/V2/V3 TMA) |
| `engine/inference.cpp` | Forward pass + generation loop |
| `engine/kernels.hip` | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/model.cpp` | Model loader + escape table builder |
| `engine/tokenizer.cpp` | Sentencepiece + HF BPE auto-detect |
| `engine/convert_model.py` | BF16/FP16 safetensors to Turbo format converter |
