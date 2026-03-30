# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller, 2.90x faster inference.

## Benchmark: Fused 12-bit Kernel on AMD MI50 32GB

Single fused kernel — decode + escape correction + matvec in one GPU dispatch. Zero atomics, zero separate passes.

Measured on Llama 3.1 8B, all 226 weight tensors, 100% bit-perfect lossless:

| Tensor Type | Shape | Ours | BF16 | Speedup | BW |
|-------------|-------|------|------|---------|-----|
| attn_k | [4096x1024] | 0.089ms | 0.111ms | 1.25x | 94 GB/s |
| attn_v | [4096x1024] | 0.087ms | 0.111ms | 1.28x | 96 GB/s |
| attn_q | [4096x4096] | 0.160ms | 0.358ms | **2.24x** | 210 GB/s |
| attn_output | [4096x4096] | 0.145ms | 0.355ms | **2.47x** | 232 GB/s |
| ffn_down | [14336x4096] | 0.471ms | 1.084ms | **2.30x** | 249 GB/s |
| ffn_gate | [4096x14336] | 0.375ms | 1.091ms | **2.90x** | 313 GB/s |
| ffn_up | [4096x14336] | 0.375ms | 1.084ms | **2.89x** | 313 GB/s |
| output | [4096x128256] | 2.975ms | 9.339ms | **3.14x** | 353 GB/s |
| token_embd | [4096x128256] | 3.048ms | 9.380ms | **3.08x** | 345 GB/s |
| **Weighted avg** | **(226 tensors)** | **0.704ms** | **2.039ms** | **2.90x** | |

Large tensors achieve 353 GB/s effective bandwidth (34% of MI50's 1 TB/s HBM2 peak). Kernel is at the compute-memory balance point — remaining gap is data-dependent LDS lookup latency.

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
  1. bp += stride_bits (incremental, avoids 64-bit multiply)
  2. Branchless 64-bit read → extract 12-bit index
  3. If idx < 4095: lookup cb[idx] in LDS (1 cycle)
     If idx == 4095: read escape_vals[esc_ptr++] (O(1))
  4. bf16_to_float conversion (2 ALU)
  5. FMA: sum += weight × activation
```

### Key Optimizations

| Optimization | Impact | Detail |
|-------------|--------|--------|
| **Branchless 64-bit read** | 1.88→2.29x | 12-bit values cross word boundary 37% of time. Always load uint64, shift — zero divergence |
| **LDS codebook (8 KB int16)** | 2.29→2.87x | 1-cycle LDS vs 10-cycle L1. Compute-bound at 1 TB/s HBM |
| **Per-thread escape table** | Enables fused | `escape_offsets[row*256+tid]` → O(1), no scanning |
| **2x loop unroll** | +5% | ILP for overlapping reads with compute |
| **Incremental bit positions** | +1% | `bp += const` (iadd64) vs `col*12` (imul64) |
| **Bitwise warp indexing** | <1% | `tid >> 6` and `tid & 63` vs integer div/mod |

### Tested and Rejected

| Approach | Result | Why rejected |
|----------|--------|-------------|
| Float32 LDS codebook (16 KB) | -40% | Halves occupancy (4 vs 8 blocks/CU) |
| Split codebook (2K LDS + L1) | -12% | Extra branch divergence on random idx |
| L1-only codebook (no LDS) | -22% | 10-cycle latency, compute-bound kernel |
| Float32 pre-converted activations | -2% | Doubles activation bandwidth (4 vs 2 bytes) |
| LDS activation cache | -40% | Same occupancy problem as float32 codebook |
| 4x loop unroll | -1% | Register pressure reduces occupancy |
| Software pipelining | -1% | Compiler already schedules well at -O3 |
| Fused CSR merge-scan | -19x on token_embd | O(N) scan for strided thread access |
| Fused binary search | -2.2x on token_embd | L2 latency per comparison |
| atomicAdd patch kernel | -58x on token_embd | MI50 lacks hardware float atomics |
| `__launch_bounds__(256, 7)` | 0% | Compiler already makes good register decisions |
| 11-bit codebook (2048 entries) | N/A | 9.6% escape rate on attn_k — too high |

### Performance Limits

At 353 GB/s effective bandwidth (34% of 1 TB/s peak), the kernel is at the **compute-memory balance point**:
- 8 ALU + 3 memory ops per element
- Data-dependent LDS lookup (codebook[random_idx]) creates serial dependency
- 8 KB LDS = 8 blocks/CU = 32 wavefronts = 80% occupancy (sweet spot)
- More occupancy requires less LDS, but reducing codebook size increases escape rate

## Correctness Verification

Each tensor is verified **bit-exact** using a separate decompress kernel:

1. **Decompress kernel** (`fixed12_decompress_fused`): Decodes every 12-bit index through the same codebook + escape logic as the matvec kernel, writes raw int16 BF16 values to a buffer
2. **Bit comparison**: `(decoded != original.view(int16)).sum() == 0` — every BF16 bit pattern must match exactly
3. **Result**: 226/226 tensors pass with zero mismatches on Llama 3.1 8B

This is stronger than matvec error checking (`|y - y_ref| < threshold`) which could miss individual weight errors that cancel out in the dot product.

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

### What the benchmark does

1. For each 2D BF16 tensor (>100K params):
   - **GPU frequency sort**: `torch.unique()` on GPU for codebook construction
   - **C packing**: `pack_fixed12_fused()` produces 12-bit packed data + per-thread escape offset table
   - **Bit-exact verification**: Decompress kernel decodes all weights, compare against original BF16
   - **Timing**: Fused matvec kernel vs `W.float() @ x.float()` BF16 baseline (30 warmup + 200 timed runs, trimmed mean)
2. Reports per-tensor speedup, effective bandwidth, lossless status, weighted average

### File Overview

| File | Purpose |
|------|---------|
| `bench_fixed12.py` | Benchmark: GPU freq sort, C packing, fused kernel timing + verification |
| `fixed12_pack.c` | C packer: 12-bit indices + per-thread escape offset table |
| `decompress_v2.hip` | Fused kernel: matvec + decompress-only (fully commented) |
| `decompress_matmul.hip` | V1 kernels: variable-length decode, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length for disk) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for .tlc models |
| `tlc_verify.py` | Round-trip lossless verification |
| `bitpack_fast.c` | C packer for .tlc variable-length format |
| `bitunpack_fast.c` | C unpacker for .tlc variable-length format |

## Batch Matvec (Concurrent Request Serving)

Decode each weight once, multiply by B activation vectors. Amortizes decode cost across batch:

| Batch | output [4096x128256] per-vec | vs BF16 batch | Amortization |
|-------|------------------------------|--------------|-------------|
| B=1 | 2.979ms | 3.14x | — |
| B=2 | 1.645ms | 9.78x | 1.81x |
| B=4 | 1.258ms | 5.34x | 2.38x |

B=4 per-vector cost is **58% cheaper** than single matvec. Production-relevant for vLLM batched serving (2-8 concurrent requests).

## Use Case

This engine accelerates **autoregressive decoding** (token generation) where inference is memory-bandwidth bound. Each decoding step does matrix × vector (weight × single token activation), reading the entire weight matrix. Our 12-bit format reads 25% less data with negligible decode overhead.

Batch matvec further amortizes decode cost when serving multiple concurrent requests — common in production LLM serving with vLLM, TGI, etc.

Not beneficial for **prefill** (prompt processing) which is compute-bound — weights are reused across the token batch.

## Target Hardware

- AMD MI50 32GB / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA) — requires warp size 64→32 adjustment
