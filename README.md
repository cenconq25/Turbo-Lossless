# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller, 1.88x faster inference.

## Benchmark: Fused 12-bit Kernel on AMD MI50

Single fused kernel — decode + escape correction + matvec in one GPU dispatch. Zero atomics, zero separate patch passes.

Measured on Llama 3.1 8B, all 226 weight tensors, 100% lossless bit-perfect:

| Tensor Type | Shape | Ours | BF16 | Speedup | BW |
|-------------|-------|------|------|---------|-----|
| attn_k | [4096x1024] | 0.078ms | 0.110ms | 1.41x | 108 GB/s |
| attn_v | [4096x1024] | 0.074ms | 0.111ms | 1.50x | 114 GB/s |
| attn_q | [4096x4096] | 0.205ms | 0.358ms | 1.75x | 164 GB/s |
| attn_output | [4096x4096] | 0.180ms | 0.355ms | **1.97x** | 186 GB/s |
| ffn_down | [14336x4096] | 0.607ms | 1.080ms | 1.78x | 194 GB/s |
| ffn_gate | [4096x14336] | 0.564ms | 1.090ms | **1.93x** | 208 GB/s |
| ffn_up | [4096x14336] | 0.567ms | 1.096ms | **1.93x** | 207 GB/s |
| output | [4096x128256] | 4.931ms | 9.355ms | **1.90x** | 213 GB/s |
| token_embd | [4096x128256] | 4.955ms | 9.387ms | **1.89x** | 212 GB/s |
| **Weighted avg** | **(226 tensors)** | **1.084ms** | **2.042ms** | **1.88x** | |

All tensors lossless (226/226). Larger tensors approach 213 GB/s effective bandwidth.

### Compression Ratios

| Format | Compression | Usage |
|--------|------------|-------|
| .tlc on disk | **1.47x** | 8-tier variable-length prefix code |
| VRAM (12-bit fixed) | **1.33x** | 4096-entry codebook + escape table |

### Cross-Model Validation

Compression validated on 11 BF16 models across 7 architectures:

| Model | Params | Type | CR |
|-------|--------|------|----|
| Llama 3.1 8B | 8B | Dense | **1.509x** |
| Phi-4 | 14B | Dense | **1.507x** |
| Codestral 22B | 22B | Dense | **1.504x** |
| Llama 3.1 70B | 70B | Dense | **1.516x** |
| Mistral Large 123B | 123B | Dense | **1.503x** |
| MiniMax-Text-01 | 456B | MoE | **1.507x** |

9/11 models achieve 1.50x+. Large MoE models (DeepSeek V3, Qwen3-235B) achieve 1.2-1.4x due to higher weight diversity.

## How It Works

### Fused Kernel Architecture

Every BF16 weight is encoded as a 12-bit codebook index. The top 4095 most frequent values map to codebook entries 0-4094. Rare values (escape, idx=4095) are stored in a per-thread escape table for O(1) inline lookup.

**Single kernel per matvec** — no separate decode or patch pass:

```
Thread tid in row block:
  1. Read 12-bit index from packed VRAM          (L2/HBM)
  2. If idx < 4095: lookup codebook[idx]          (L1 cache, 8 KB)
     If idx == 4095: read escape_vals[esc_ptr++]  (O(1), per-thread offset table)
  3. Multiply weight × activation, accumulate     (registers)
```

### Key Design Decisions

**L1-cached codebook (not LDS)**: The 8 KB codebook fits in MI50's 16 KB L1 cache. Skipping LDS load eliminates the per-block `__syncthreads` barrier and maximizes wavefront occupancy.

**Per-thread escape offset table**: Each thread gets a pre-computed pointer into the escape value array via `escape_offsets[row * 256 + tid]`. On escape (`idx == 4095`), just read and increment — zero scanning, zero divergence. This is what makes the fused kernel fast on high-escape tensors like token_embd (408K patches, 1.89x).

**2x loop unroll**: Two columns decoded per iteration for instruction-level parallelism. Overlaps packed data reads with L1 codebook lookups and FMA. 4x tested but register pressure reduced occupancy.

### What Was Tested and Rejected

| Approach | Result | Why |
|----------|--------|-----|
| LDS codebook (8 KB in shared memory) | 1.80x | Per-block load barrier reduces occupancy |
| Fused kernel with CSR merge-scan | 0.15x on token_embd | Strided thread access causes O(N) scan per escape |
| Fused kernel with binary search | 1.32x on token_embd | L2 latency on each comparison, register pressure |
| Separate patch kernel with atomicAdd | 0.02x on token_embd | MI50 lacks hardware float atomics (CAS retry loop) |
| CSR wavefront-parallel patches (two-pass) | 1.88x | Fast, but two kernel launches add overhead on small tensors |
| 4x loop unroll | 1.86x | Register pressure reduces occupancy |
| 11-bit codebook (no patches) | 2.08x | Not lossless (~1.4% wrong values) |

## Build & Run

### Prerequisites

- BF16 safetensors model in `models/` directory
- ROCm (AMD) or CUDA (NVIDIA) toolkit
- Python: `torch`, `safetensors`, `numpy`

### Build

```bash
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
```

### Run

```bash
python3 bench_fixed12.py models/llama-3.1-8b/llama-3.1-8b.safetensors
```

### File Overview

| File | Purpose |
|------|---------|
| `bench_fixed12.py` | Benchmark: GPU freq sort, C packing, fused kernel timing |
| `fixed12_pack.c` | C packer: 12-bit indices + per-thread escape offset table |
| `decompress_v2.hip` | Fused kernel: L1-cached codebook + O(1) escape + 2x unroll |
| `decompress_matmul.hip` | V1 kernels: variable-length decode, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length for disk) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for .tlc models |
| `tlc_verify.py` | Round-trip lossless verification |

## Target Hardware

- AMD MI50 / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA) — requires warp size 64→32 adjustment
