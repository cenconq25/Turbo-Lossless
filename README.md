# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller, 29% less VRAM. Matches llama.cpp at B=1, beats it by 30%+ at B>=2.

**BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## How It Works

### The Problem

LLM inference is **memory-bandwidth bound**: each generated token reads the entire weight matrix from GPU HBM. A 7B model reads ~14 GB of BF16 weights per token. The GPU's compute units sit idle waiting for data.

Quantization (INT4/INT8) solves this by reading less data, but **destroys precision**.

### The Insight: BF16 Has Only 40 Active Exponents

BF16 is 16 bits: `[1-bit sign][8-bit exponent][7-bit mantissa]`. Neural network weights cluster tightly around zero. Only **40 out of 256 possible exponents** are used, and 15 consecutive exponents cover **99.97%** of all weights.

The sign and mantissa are near-random (can't compress). The exponent has only 2.6 bits of entropy in 8 bits — **5.4 bits wasted**.

### Structured 12-Bit Encoding

We compress by replacing the 8-bit exponent with a 4-bit group code:

```
Original BF16:  [sign 1][exponent 8][mantissa 7]  = 16 bits
Our encoding:   [exp_group 4][sign 1][mantissa 7]  = 12 bits

Sign and mantissa pass through UNCHANGED.
Exponent: 15 consecutive values -> groups 1-15. Group 0 = escape.
Decode:   exponent = BaseExp + group  (ONE integer add, no lookup table)
```

**Result**: 16 bits -> 12 bits = **1.33x compression**, zero information loss, decode is 1 ADD.

### Split12 Storage Format

The 12-bit values are stored in two **byte-aligned** arrays for zero read amplification:

```
Array 1: [sign 1][mantissa 7] = 1 byte per element  (perfectly aligned loads)
Array 2: [group 4]            = 0.5 byte per element (nibble-packed, 2 per byte)
Total: 1.5 bytes/element = same 1.33x compression, but zero HBM waste
```

### Escape Handling (0.03% of values)

Values with rare exponents get group=0 (escape). Their exact BF16 is stored in a tiny CSR table (~3 MB for 7B model). Branch predicted 99.97% correct.

### Batch Decode Amortization

Serving B concurrent users: decode each weight ONCE, multiply by B vectors:

```
B=1:  decode -> 1x FMA    (near-parity with BF16)
B=4:  decode -> 4x FMA    (2.0x faster than BF16)
B=8:  decode -> 8x FMA    (2.4x faster than BF16)
```

---

## Benchmarks — Mistral 7B Instruct, MI50 32GB

| Mode | tok/s total | tok/s/user | VRAM | vs llama.cpp BF16 (33.0) |
|------|------------:|-----------:|-----:|:-------------------------|
| B=1 | 32.6 | 32.6 | ~10 GB | 0.99x (**matched!**) |
| B=4 | 67.0 | 16.8 | 10.3 GB | **2.03x faster** |
| **B=8** | **80.7** | **10.1** | **10.3 GB** | **2.45x faster, 1.32x less VRAM** |

### vs llama.cpp BF16 (same GPU, same model)

| Batch | llama.cpp BF16 | Turbo Lossless | Winner | VRAM |
|------:|---------------:|---------------:|:------:|-----:|
| B=1 | 33.0 | **32.6** | **Matched** (99%) | 14.5 vs **10.3 GB** |
| B=4 | 50.9 | **67.0** | **Turbo +32%** | 14.5 vs **10.3 GB** |
| **B=8** | 58.7 | **80.7** | **Turbo +37%** | 14.5 vs **10.3 GB** |

**Matched llama.cpp at B=1**, faster at B>=2, **29% less VRAM at all batch sizes**.

### Supported Models

| Model | Params | Tokenizer | Status |
|-------|-------:|-----------|--------|
| Mistral 7B / 7B Instruct | 7B | sentencepiece | **Tested** |
| Llama 3.1 8B | 8B | HF BPE (tiktoken) | **Tested** |
| Any BF16 safetensors transformer | varies | sentencepiece or HF BPE | Should work |

### Hardware Projection (B=8)

| GPU | BW (TB/s) | Est. tok/s | vs native BF16 |
|-----|----------:|-----------:|:--------------:|
| MI50 (measured) | 1.0 | **81** | 2.45x |
| A100 80GB | 2.0 | ~160 | 2.45x |
| H100 80GB | 3.4 | ~275 | 2.45x |
| MI300X | 5.3 | ~430 | 2.45x |
| B200 | 8.0 | ~650 | 2.45x |

---

## Quick Start

```bash
# 1. Build packers
gcc -O3 -shared -fPIC -o structured12_pack.so structured12_pack.c
gcc -O3 -shared -fPIC -o split12_pack.so split12_pack.c

# 2. Build engine
cd engine && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17

# 3. Convert model (sentencepiece models like Mistral)
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/

# 4. Generate split12 files (optional, +5% B=1 speed)
python3 -c "... see convert script ..."

# 5. Run
HIP_VISIBLE_DEVICES=0 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Hello" 200 8
```

### Usage

```
./turbo-engine <model_dir> "<prompt>" <max_tokens> [batch_size]
```

| Variable | Effect |
|----------|--------|
| `HIP_VISIBLE_DEVICES=N` | Select GPU (use non-display GPU) |
| `TURBO_FAST=1` | Pre-computed escape counts (+10% speed, +361 MB VRAM) |
| `TURBO_CTX=N` | Max context length (default 2048) |
| `TURBO_PROFILE=1` | Print per-token timing breakdown |

---

## File Map

| File | Lines | Purpose |
|------|------:|---------|
| `decompress_v2.hip` | 1312 | All GPU matvec kernels (structured12 + split12, B=1/4/8) |
| `structured12_pack.c` | 118 | Packer: `find_base_exp()` + `pack_structured12_csr()` |
| `split12_pack.c` | 128 | Split12 packer: byte-aligned sign+mantissa + nibble groups |
| `engine/inference.cpp` | 603 | Forward pass (B=1/4/8) + generate loop |
| `engine/kernels.hip` | 1171 | RMSNorm, RoPE, Flash Attention, SiLU, argmax, embed |
| `engine/model.cpp` | 268 | Model loader + escape table builder |
| `engine/tokenizer.cpp` | 379 | Auto-detect sentencepiece / HF BPE tokenizer |
| `engine/convert_model.py` | 207 | BF16 safetensors -> turbo format converter |
| `engine/main.cpp` | 80 | CLI entry point |

**Total: ~4450 lines of production code.**
