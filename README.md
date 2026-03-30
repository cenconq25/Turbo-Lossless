# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller, 2.87x faster inference.

## Benchmark: Fused 12-bit Kernel on AMD MI50 (32GB)

Single fused kernel — decode + escape correction + matvec in one GPU dispatch. Zero atomics, zero separate passes.

Measured on Llama 3.1 8B, all 226 weight tensors, 100% lossless bit-perfect:

| Tensor Type | Shape | Ours | BF16 | Speedup | BW |
|-------------|-------|------|------|---------|-----|
| attn_k | [4096x1024] | 0.088ms | 0.111ms | 1.25x | 95 GB/s |
| attn_v | [4096x1024] | 0.087ms | 0.111ms | 1.28x | 96 GB/s |
| attn_q | [4096x4096] | 0.166ms | 0.358ms | **2.15x** | 202 GB/s |
| attn_output | [4096x4096] | 0.143ms | 0.358ms | **2.50x** | 235 GB/s |
| ffn_down | [14336x4096] | 0.476ms | 1.084ms | **2.28x** | 247 GB/s |
| ffn_gate | [4096x14336] | 0.379ms | 1.091ms | **2.88x** | 310 GB/s |
| ffn_up | [4096x14336] | 0.377ms | 1.091ms | **2.90x** | 312 GB/s |
| output | [4096x128256] | 3.013ms | 9.360ms | **3.11x** | 349 GB/s |
| token_embd | [4096x128256] | 3.099ms | 9.409ms | **3.04x** | 339 GB/s |
| **Weighted avg** | **(226 tensors)** | **0.712ms** | **2.042ms** | **2.87x** | |

All tensors lossless (226/226). Large tensors achieve 349 GB/s effective bandwidth (34% of MI50's 1 TB/s HBM2 peak).

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
  1. Branchless 64-bit read of 12-bit index       (HBM → L2)
  2. If idx < 4095: lookup codebook[idx]           (LDS, 1-cycle)
     If idx == 4095: read escape_vals[esc_ptr++]   (O(1), per-thread offset table)
  3. Multiply weight × activation, accumulate      (registers)
```

### Key Optimizations

**Branchless 64-bit read** (+22%): 12-bit values straddle 32-bit word boundaries 37% of the time. Always load two words as one uint64 and shift — zero branches, zero warp divergence. This single change: 1.88x → 2.29x.

**LDS codebook** (+25%): MI50 has 1 TB/s HBM bandwidth — the kernel is compute-bound, not bandwidth-bound. Moving the 8 KB codebook from L1 cache (~10 cycle latency) to LDS (1 cycle) eliminates the main compute bottleneck. This change: 2.29x → 2.87x.

**Per-thread escape offset table**: `escape_offsets[row * 256 + tid]` gives each thread its pre-computed pointer. On escape, just read and increment — O(1), zero scanning.

**2x loop unroll**: Two columns per iteration for instruction-level parallelism. Overlaps packed data reads with LDS codebook lookups and FMA.

### Optimization History

| Step | Speedup | Change |
|------|---------|--------|
| Starting point (broken atomicAdd) | 0.05x | MI50 CAS retry loop stall |
| CSR wavefront-parallel patches | 1.73x | Row-grouped, no atomics |
| L1-cached codebook (no LDS) | 1.80x | Max occupancy |
| 2x loop unroll | 1.88x | ILP |
| Fused kernel (O(1) escape table) | 1.88x | Single launch, per-thread offsets |
| Branchless 64-bit read | 2.29x | Eliminated 37% warp divergence |
| **LDS codebook** | **2.87x** | **1-cycle vs 10-cycle lookup** |

### What Was Tested and Rejected

| Approach | Result | Why |
|----------|--------|-----|
| Fused with CSR merge-scan | 0.15x on token_embd | Strided threads scan O(N) patches each |
| Fused with binary search | 1.32x on token_embd | L2 latency per comparison |
| atomicAdd patch kernel | 0.02x on token_embd | MI50 lacks hardware float atomics |
| 4x loop unroll | -3% vs 2x | Register pressure reduces occupancy |
| Software pipelining | -1% vs 2x | Compiler already schedules well at -O3 |
| 11-bit codebook (2048 entries) | N/A | 9.6% escape rate on attn_k — too high |

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
| `decompress_v2.hip` | Fused kernel: branchless read + LDS codebook + O(1) escape + 2x unroll |
| `decompress_matmul.hip` | V1 kernels: variable-length decode, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length for disk) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for .tlc models |
| `tlc_verify.py` | Round-trip lossless verification |

## Target Hardware

- AMD MI50 32GB / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA) — requires warp size 64→32 adjustment
