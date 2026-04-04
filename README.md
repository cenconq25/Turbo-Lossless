# Turbo Lossless

**1.33x smaller. Up to 2.93x faster than vLLM. Zero precision loss.**

100% bit-perfect lossless compression for BF16 safetensors models. Replaces the 8-bit exponent with a 4-bit group code -- decode is a single integer ADD.

> **Note:** Research proof of concept. KV cache and attention are not fully optimised -- expect slowdown over long conversations.

![Turbo Lossless CLI](docs/turbo-screenshot.png)

---

## Highlights

- **1.33x compression** -- 12 bits per weight, fixed rate, no entropy coding
- **1 ADD to decode** -- 3-7x fewer ops than other lossless methods
- **2.93x faster** than vLLM at B=256 (Mistral 7B, RTX 5070 Ti)
- **Runs where others OOM** -- Llama 3.1 8B fits in 12.4 GB (vLLM OOMs on 16 GB)
- **NVIDIA + AMD** -- RTX 5070 Ti and MI50 tested

## Quick Start

```bash
# Single prompt
./turbo models/mistral-7b-instruct-turbo "What is the meaning of life?" 200

# Interactive chat
./turbo models/mistral-7b-instruct-turbo -i
```

First run auto-builds the engine. To convert a HuggingFace model:
```bash
./turbo models/mistral-7b-instruct "Hello" 200    # auto-converts to turbo format
```

Set `TURBO_FAST=1` for +10% speed (recommended). See [Technical Details](docs/TECHNICAL.md) for build instructions, environment variables, and benchmarking.

## Results

### Single-User (B=1) -- RTX 5070 Ti

| Model | Turbo tok/s | vs vLLM | vs llama.cpp |
|-------|------------:|--------:|-------------:|
| Llama 2 7B | **64.7** | **1.47x** | 1.09x |
| Mistral 7B | **60.0** | **1.10x** | 1.08x |
| Llama 3.1 8B | **57.0** | vLLM OOM | 1.08x |

### Multi-User B=256 (total tok/s)

| Model | Turbo | vLLM | Speedup |
|-------|------:|-----:|--------:|
| Llama 2 7B | **2,931** | 1,086 | **2.70x** |
| Mistral 7B | **2,554** | 872 | **2.93x** |
| Llama 3.1 8B | **2,471** | OOM | -- |

BF16 safetensors only. No GGUF, no FP32, no quantized formats.

---

## Learn More

- **[Technical Details](docs/TECHNICAL.md)** -- encoding format, kernel architecture, full benchmarks, build instructions, file map
- **[CLAUDE.md](CLAUDE.md)** -- complete kernel internals and engine data flow

## Acknowledgements

The V3 fused decode+GEMM kernel uses tensor core patterns inspired by [ZipServ / ZipGEMM](https://github.com/HPMLL/ZipServ_ASPLOS26) (Fan et al., ASPLOS 2026).
