# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller, 2.23x faster inference.

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

| Mode | tok/s total | tok/s/user | VRAM | vs llama.cpp BF16 (32.2) |
|------|------------:|-----------:|-----:|:-------------------------|
| B=1 | 26.1 | 26.1 | ~10 GB | 0.81x |
| B=4 | 61.5 | 15.4 | 10.3 GB | **1.91x faster** |
| **B=8** | **71.7** | **9.0** | **10.3 GB** | **2.23x faster, 1.32x less VRAM** |
| B=16 | 70.7 | 4.4 | ~10.5 GB | 2.20x (plateau) |
| B=32 | 71.0 | 2.2 | ~11 GB | 2.20x (plateau) |

B=8 is the throughput ceiling — beyond B=8, weight decode is fully HBM bandwidth saturated. B=16/32 run as multiple B=8 passes with no additional amortization.

### Hardware Projection (B=8)

| GPU | BW (TB/s) | Est. tok/s | vs native BF16 |
|-----|----------:|-----------:|:--------------:|
| MI50 (measured) | 1.0 | **72** | 2.23x |
| A100 80GB | 2.0 | ~150 | 2.23x |
| H100 80GB | 3.4 | ~250 | 2.23x |
| MI300X | 5.3 | ~380 | 2.23x |
| B200 | 8.0 | ~580 | 2.23x |

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
