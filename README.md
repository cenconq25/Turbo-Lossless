# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller, 2.29x faster inference.

## Benchmark: Fused 12-bit Kernel on AMD MI50

Single fused kernel — decode + escape correction + matvec in one GPU dispatch. Zero atomics, zero separate passes.

Measured on Llama 3.1 8B, all 226 weight tensors, 100% lossless bit-perfect:

| Tensor Type | Shape | Ours | BF16 | Speedup | BW |
|-------------|-------|------|------|---------|-----|
| attn_k | [4096x1024] | 0.064ms | 0.110ms | 1.71x | 130 GB/s |
| attn_v | [4096x1024] | 0.061ms | 0.111ms | 1.83x | 138 GB/s |
| attn_q | [4096x4096] | 0.149ms | 0.358ms | **2.40x** | 225 GB/s |
| attn_output | [4096x4096] | 0.141ms | 0.355ms | **2.52x** | 238 GB/s |
| ffn_down | [14336x4096] | 0.475ms | 1.079ms | **2.27x** | 247 GB/s |
| ffn_gate | [4096x14336] | 0.467ms | 1.091ms | **2.34x** | 252 GB/s |
| ffn_up | [4096x14336] | 0.468ms | 1.091ms | **2.33x** | 251 GB/s |
| output | [4096x128256] | 4.089ms | 9.341ms | **2.28x** | 257 GB/s |
| token_embd | [4096x128256] | 4.103ms | 9.377ms | **2.29x** | 256 GB/s |
| **Weighted avg** | **(226 tensors)** | **0.889ms** | **2.040ms** | **2.29x** | |

All tensors lossless (226/226). Large tensors achieve 257 GB/s effective bandwidth (~51% of MI50 theoretical peak).

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
  1. Branchless 64-bit read of 12-bit index       (HBM → L1/L2)
  2. If idx < 4095: lookup codebook[idx]           (L1 cache, 8 KB)
     If idx == 4095: read escape_vals[esc_ptr++]   (O(1), per-thread offset table)
  3. Multiply weight × activation, accumulate      (registers)
```

### Key Optimizations

**Branchless 64-bit read** (+0.41x): The 12-bit index can straddle a 32-bit word boundary (~37% of reads). The original conditional branch caused severe warp divergence on MI50's 64-wide wavefronts. Fix: always load two 32-bit words as one 64-bit value and shift — zero branches, zero divergence. This single change improved speedup from 1.88x to 2.29x.

**L1-cached codebook**: The 8 KB codebook fits in MI50's 16 KB L1 cache. No LDS load, no `__syncthreads` barrier. Maximum wavefront occupancy.

**Per-thread escape offset table**: `escape_offsets[row * 256 + tid]` gives each thread its pre-computed pointer. On escape, just read and increment — O(1), zero scanning.

**2x loop unroll**: Two columns per iteration for instruction-level parallelism. Overlaps packed data reads with L1 codebook lookups and FMA.

### What Was Tested and Rejected

| Approach | Result | Why |
|----------|--------|-----|
| Branching read12 (conditional word boundary) | 1.88x | 37% branch rate → warp divergence |
| LDS codebook (8 KB in shared memory) | 1.80x | Per-block load barrier reduces occupancy |
| Fused with CSR merge-scan | 0.15x on token_embd | Strided threads scan O(N) patches each |
| Fused with binary search | 1.32x on token_embd | L2 latency per comparison |
| atomicAdd patch kernel | 0.02x on token_embd | MI50 lacks hardware float atomics |
| Two-pass (separate patch kernel) | 1.88x | Two launches, overhead on small tensors |
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
| `decompress_v2.hip` | Fused kernel: branchless read + L1 codebook + O(1) escape + 2x unroll |
| `decompress_matmul.hip` | V1 kernels: variable-length decode, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length for disk) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for .tlc models |
| `tlc_verify.py` | Round-trip lossless verification |

## Target Hardware

- AMD MI50 / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA) — requires warp size 64→32 adjustment
