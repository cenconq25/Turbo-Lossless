# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller, 1.88x faster inference.

## Results

### Inference Benchmark: AMD MI50 (gfx906)

Measured on Llama 3.1 8B — all 226 weight tensors, 100% lossless:

| Tensor | Shape | Ours | BF16 | Speedup |
|--------|-------|------|------|---------|
| attn_k | [4096x1024] | 0.090ms | 0.110ms | 1.23x |
| attn_q | [4096x4096] | 0.201ms | 0.357ms | **1.78x** |
| attn_output | [4096x4096] | 0.209ms | 0.356ms | **1.70x** |
| ffn_down | [14336x4096] | 0.623ms | 1.078ms | **1.73x** |
| ffn_gate | [4096x14336] | 0.574ms | 1.097ms | **1.91x** |
| ffn_up | [4096x14336] | 0.573ms | 1.094ms | **1.91x** |
| output | [4096x128256] | 4.817ms | 9.394ms | **1.95x** |
| token_embd | [4096x128256] | 5.016ms | 9.381ms | **1.87x** |
| **Weighted avg** | **(all 226 tensors)** | | | **1.88x** |

Larger tensors benefit most — output head achieves 218 GB/s effective bandwidth.

### Compression Ratios

Validated on 11 BF16 models across 7 architectures:

| Model | Params | Type | CR | VRAM Saved |
|-------|--------|------|----|-----------|
| Llama 3.1 8B | 8B | Dense | **1.509x** | 5.0 GB |
| Phi-4 | 14B | Dense | **1.507x** | 9.4 GB |
| Codestral 22B | 22B | Dense | **1.504x** | 14.0 GB |
| Llama 3.1 70B | 70B | Dense | **1.516x** | 47.6 GB |
| Mistral Large 123B | 123B | Dense | **1.503x** | 82.4 GB |
| MiniMax-Text-01 | 456B | MoE | **1.507x** | 307 GB |

9/11 models achieve **1.50x+**. Large MoE models (DeepSeek V3, Qwen3-235B) achieve 1.2-1.4x due to higher weight diversity.

## How It Works

### Two-Format Architecture

**On disk**: 8-tier variable-length prefix code (.tlc format, **1.47x** compression). Per-tensor codebook maps the most frequent BF16 values to short codes.

**In VRAM**: Fixed-width 12-bit packed indices + 4096-entry codebook per tensor (**1.33x** compression). Converted from .tlc at model load time.

**At inference**: Each thread reads 12 packed bits, looks up the codebook (L1-cached), and multiplies by the activation vector. Escape values (~0.08%) are corrected via a wavefront-parallel patch kernel with zero atomics.

### Key Design Decisions

1. **12-bit fixed-width in VRAM** (not variable-length): Every value's bit position is deterministic — `col * 12`. Zero serial dependencies. Maximum GPU parallelism.

2. **L1-cached codebook** (not LDS): The 8 KB codebook fits in MI50's 16 KB L1 cache. Skipping LDS load eliminates the per-block overhead and maximizes occupancy.

3. **2x loop unroll**: Two columns decoded and accumulated per iteration for instruction-level parallelism. Overlaps packed data reads with codebook lookups and FMA. 4x unroll tested but register pressure reduced occupancy.

3. **CSR row-grouped patches** (not per-element atomicAdd): MI50/gfx906 lacks hardware float atomics — `atomicAdd(float)` compiles to a CAS retry loop. 408K patches caused 580ms stall. Row-grouped CSR format + wavefront reduction eliminated all atomics. Measured: 580ms → 0.2ms.

## Running the Benchmark

### Build

```bash
# C packer
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c

# HIP kernels (AMD MI50)
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_matmul.so decompress_matmul.hip
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
```

### Run

```bash
python3 bench_fixed12.py models/llama-3.1-8b/llama-3.1-8b.safetensors
```

### File Overview

| File | Purpose |
|------|---------|
| `bench_fixed12.py` | Benchmark: GPU freq sort, C packing, kernel timing vs BF16 baseline |
| `fixed12_pack.c` | C packer: 12-bit indices + CSR row-grouped escape patches |
| `decompress_v2.hip` | V2 kernel: L1-cached codebook matvec + wavefront-parallel patches |
| `decompress_matmul.hip` | V1 kernels: variable-length decode, fixed-width matvec, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length format for disk) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for loading .tlc models |
| `tlc_verify.py` | Round-trip lossless verification |
| `bitpack_fast.c` | C packer for .tlc variable-length format |
| `bitunpack_fast.c` | C unpacker for .tlc variable-length format |

## Target Hardware

- AMD MI50 / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA) — requires warp size adjustment (64→32)
