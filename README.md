# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for BF16 LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller, 29% less VRAM. Matches llama.cpp at B=1, beats it by 30% at B≥2.

**BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## How It Works

### The Problem

LLM inference is **memory-bandwidth bound**: each generated token reads the entire weight matrix from GPU HBM. A 7B model reads ~14 GB of BF16 weights per token. The GPU's compute units sit idle waiting for data.

Quantization (INT4/INT8) solves this by reading less data, but **destroys precision**.

### The Insight: BF16 Has Only 40 Active Exponents

BF16 is 16 bits: `[1-bit sign][8-bit exponent][7-bit mantissa]`. Neural network weights cluster tightly around zero. Only **40 out of 256 possible exponents** are used, and 15 consecutive exponents (109-123) cover **99.97%** of all weights.

The sign and mantissa are near-random (can't compress). The exponent has only 2.6 bits of entropy in 8 bits — **5.4 bits wasted**.

### Structured 12-Bit Encoding

We compress by replacing the 8-bit exponent with a 4-bit group code:

```
Original BF16:  [sign 1][exponent 8][mantissa 7]  = 16 bits
Our encoding:   [exp_group 4][sign 1][mantissa 7]  = 12 bits

Sign and mantissa pass through UNCHANGED.
Exponent: 15 consecutive values → groups 1-15. Group 0 = escape.
Decode:   exponent = BaseExp + group  (ONE integer add, no lookup table)
```

**Example:**
```
BF16 value -0.00195: sign=1, exponent=118, mantissa=0000000
  Encode: group = 118 - 106 = 12 → pack [12][1][0000000] = 12 bits
  Decode: exponent = 106 + 12 = 118 → reconstruct exact BF16 → -0.00195
```

**Result**: 16 bits → 12 bits = **1.33x compression**, zero information loss, decode is 1 ADD.

### Escape Handling (0.03% of values)

Values with rare exponents get group=0 (escape). Their exact BF16 stored in a tiny CSR table (~3 MB for 7B model). Branch predicted 99.97% correct.

### Batch Decode Amortization

Serving B concurrent users: decode each weight ONCE, multiply by B vectors:

```
B=1:  decode → 1× FMA    (overhead dominates → 0.62x BF16)
B=4:  decode → 4× FMA    (1.57x faster than BF16)
B=8:  decode → 8× FMA    (2.08x faster than BF16)
```

### Fused GPU Kernel

Single kernel launch, no separate decompress step, NO LDS codebook:

```
1. Compute escape pointer (warp shuffle prefix sum)
2. Read 12-bit value (branchless 64-bit extract)
3. Arithmetic decode: BaseExp + group (1 ADD, zero memory access)
4. Reconstruct BF16 (shifts + OR)
5. FMA: weight × activation (×B for batch)
6. Wavefront shuffle reduction → output
```

### vs Quantization

| | Turbo Lossless | INT4 (GPTQ/AWQ) | INT8 |
|---|---|---|---|
| Quality loss | **None (bit-exact)** | PPL +0.1-0.5 | PPL +0.01-0.1 |
| Compression | 1.33x | 4x | 2x |
| Speed vs BF16 (B=8) | **2.08x** | ~3-4x | ~1.8x |

### vs Lossless Competitors

| Project | Compression | Speed vs BF16 | Escape rate | Decode |
|---------|------------:|:-------------:|------------:|:-------|
| **Turbo Lossless** | **1.33x** | **2.08x (B=8)** | **0.03%** | **1 ADD** |
| ZipServ (2026) | 1.40x | 1.22x (vs vLLM) | 3.0% | 3 bitmaps (~10 ALU) |
| DFloat11 (2025) | 1.43x | 0.5x (slower) | 0% | Huffman |

---

## Benchmarks — Mistral 7B Instruct, MI50 32GB

| Mode | tok/s total | tok/s/user | VRAM | vs llama.cpp BF16 (33.0) |
|------|------------:|-----------:|-----:|:-------------------------|
| B=1 | 32.6 | 32.6 | ~10 GB | 0.99x (**matched!**) |
| B=4 | 64.3 | 16.1 | 10.3 GB | **1.95x faster** |
| **B=8** | **77.4** | **9.7** | **10.3 GB** | **2.35x faster, 1.32x less VRAM** |
| B=16 | ~77 | ~4.8 | ~10.5 GB | 2.33x (plateau) |
| B=32 | ~77 | ~2.4 | ~11 GB | 2.33x (plateau) |

**Beats llama.cpp BF16 at B≥2.** At B=1, our 12-bit decode overhead makes us 15% slower. At B≥2, decode amortization more than compensates — we read 25% less data from HBM.

### vs llama.cpp BF16 (same GPU, same model)

| Batch | llama.cpp BF16 | Turbo Lossless | Winner | VRAM |
|------:|---------------:|---------------:|:------:|-----:|
| B=1 | 33.0 | **32.6** | **Matched** (99%) | 14.5 vs **10.3 GB** |
| B=2 | 44.4 | **65.2** | **Turbo +47%** | 14.5 vs **10.3 GB** |
| B=4 | 50.9 | **64.3** | **Turbo +26%** | 14.5 vs **10.3 GB** |
| **B=8** | 58.7 | **77.4** | **Turbo +32%** | 14.5 vs **10.3 GB** |

**Matched llama.cpp at B=1**, faster at B≥2, **29% less VRAM at all batch sizes**.

### Hardware Projection (B=8)

| GPU | BW (TB/s) | Est. tok/s | vs native BF16 |
|-----|----------:|-----------:|:--------------:|
| MI50 (measured) | 1.0 | **77** | 2.35x |
| A100 80GB | 2.0 | ~155 | 2.35x |
| H100 80GB | 3.4 | ~265 | 2.35x |
| MI300X | 5.3 | ~410 | 2.35x |
| B200 | 8.0 | ~620 | 2.35x |

---

## Quick Start

```bash
# Build
gcc -O3 -shared -fPIC -o structured12_pack.so structured12_pack.c
cd engine && /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -o turbo-engine \
  main.cpp model.cpp inference.cpp tokenizer.cpp sampler.cpp \
  kernels.hip ../decompress_v2.hip -lhipblas -lsentencepiece -std=c++17

# Convert model
python3 engine/convert_model.py models/mistral-7b-instruct
cp models/mistral-7b-instruct/tokenizer.model models/mistral-7b-instruct-turbo/

# Run (B=8 recommended)
HIP_VISIBLE_DEVICES=1 TURBO_FAST=1 ./turbo-engine models/mistral-7b-instruct-turbo "Write a poem:" 200 8
```

| Variable | Effect |
|----------|--------|
| `HIP_VISIBLE_DEVICES=N` | Select GPU |
| `TURBO_FAST=1` | Pre-computed escape counts (+10% speed, +361 MB VRAM) |
| `TURBO_CTX=N` | Max context length (default 2048) |

## File Map

| File | Purpose |
|------|---------|
| `decompress_v2.hip` | Fused structured12 decode-matvec kernels (B=1/4/8) |
| `structured12_pack.c` | Packer: `find_base_exp()` + `pack_structured12_csr()` |
| `engine/convert_model.py` | BF16 safetensors → structured12 format |
| `engine/inference.cpp` | Forward pass B=1/4/8 + generate loop |
| `engine/kernels.hip` | RMSNorm, RoPE, Flash Attention, SiLU, argmax |
| `engine/model.cpp` | Model loader + escape table builder |
