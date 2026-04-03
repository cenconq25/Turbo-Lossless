# Turbo Lossless — 1.33x Smaller, 2.93x Faster, Decode with 1 ADD

### **1.33x** compression for most models. Up to **2.93x** faster than vLLM for multi-user. **3-7x fewer ops** than other lossless methods — just an ADD.

> **BF16 → 12-bit lossless. One integer ADD to decode. Zero precision loss.**

```
BF16:         [sign 1][exponent 8][mantissa 7]  = 16 bits
Turbo 12-bit: [group 4][sign 1][mantissa 7]     = 12 bits

Decode: exponent = BaseExp + group   ← that's it. One ADD.
```

**1.33x smaller. 2.93x faster than vLLM at B=256. Runs models where competitors OOM.**

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

BF16 and FP16 safetensors supported. No GGUF, no FP32, no quantized formats.

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
