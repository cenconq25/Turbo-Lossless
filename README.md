# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.33x smaller in VRAM, up to 3.14x faster inference.

## Results

### Single Matvec (B=1) — AMD MI50 32GB

Llama 3.1 8B, all 226 weight tensors, bit-perfect verified:

| Tensor Type | Shape | Ours | BF16 | Speedup | Effective BW |
|-------------|-------|------|------|---------|-------------|
| attn_k | [4096×1024] | 0.089ms | 0.111ms | 1.25x | 94 GB/s |
| attn_v | [4096×1024] | 0.087ms | 0.111ms | 1.28x | 96 GB/s |
| attn_q | [4096×4096] | 0.160ms | 0.358ms | **2.24x** | 210 GB/s |
| attn_output | [4096×4096] | 0.145ms | 0.355ms | **2.47x** | 232 GB/s |
| ffn_down | [14336×4096] | 0.471ms | 1.084ms | **2.30x** | 249 GB/s |
| ffn_gate | [4096×14336] | 0.375ms | 1.091ms | **2.90x** | 313 GB/s |
| ffn_up | [4096×14336] | 0.375ms | 1.084ms | **2.89x** | 313 GB/s |
| output | [4096×128256] | 2.975ms | 9.339ms | **3.14x** | 353 GB/s |
| token_embd | [4096×128256] | 3.048ms | 9.380ms | **3.08x** | 345 GB/s |
| **Weighted avg** | **(226 tensors)** | **0.704ms** | **2.039ms** | **2.90x** | |

### Batch Matvec — Concurrent Request Serving

Decode each weight once, multiply by B activation vectors. All bit-exact with single matvec:

| Batch | Per-vector (output head) | vs BF16 | Amortization |
|-------|-------------------------|---------|-------------|
| B=1 | 2.97ms | 3.14x | — |
| B=2 | 1.64ms | 9.78x | 1.81x |
| B=4 | 1.08ms | 8.69x | 2.76x |
| B=8 | 0.88ms | 3.84x | 3.39x |

B=4 is the production sweet spot. B=8 hits diminishing returns as ALU saturates.

### Compression

| Format | Ratio | Usage |
|--------|-------|-------|
| .tlc on disk | **1.47x** | 8-tier variable-length prefix code |
| VRAM (12-bit) | **1.33x** | 4096-entry codebook + per-thread escape table |

Validated on 11 BF16 models across 7 architectures (Llama, Mistral, Phi, Qwen, Yi, BigCode, MiniMax). 9/11 achieve 1.50x+ disk compression.

### Cross-Model Inference Benchmark

| Model | Params | Tensors | B=1 Speedup | B=4 Speedup | B=4 Amortize | Lossless |
|-------|--------|---------|------------|------------|-------------|---------|
| **Llama 3.1 8B** | 8B | 226 | **2.90x** | **8.12x** | 2.80x | 226/226 |
| **Llama 3.1 70B** | 70B | 33* | **2.69x** | **7.35x** | 2.73x | 33/33 |
| **Phi-4** | 14B | 77 | **2.48x** | **7.10x** | 2.86x | 77/77 |
| **Mistral 7B** | 7B | 226 | **2.43x** | **7.04x** | 2.89x | 226/226 |
| **Qwen2.5 7B** | 7B | 198 | **2.26x** | **6.60x** | 2.93x | 198/198 |
| **Phi-3 Mini** | 3.8B | 130 | **2.17x** | **6.31x** | 2.91x | 130/130 |
| **TinyLlama** | 1.1B | 156 | **1.88x** | **5.56x** | 2.95x | 156/156 |
| **StableLM 1.6B** | 1.6B | 170 | **1.76x** | **5.16x** | 2.93x | 170/170 |

8 models (1.1B–70B), 6 architectures, 1216 tensors — all bit-perfect lossless. Larger models benefit more (bigger tensors = better block amortization). Batch amortization ~2.85x universal.

*70B: sampled 33 tensors (2 of 30 shards), per-tensor benchmark on single card.

## How It Works

### Encoding

Each BF16 weight tensor's unique values are frequency-sorted. The top 4095 map to codebook entries 0–4094 (12-bit index). Remaining values (~0.08%) get index 4095 (escape sentinel) and their correct BF16 values are stored in a per-thread escape table.

### Fused Kernel (decompress_v2.hip)

Single GPU kernel per matvec — no separate decode or patch pass:

1. **Branchless 64-bit read**: Load 2×uint32 as uint64, shift, mask → 12-bit index. Eliminates 37% warp divergence from word-boundary crossing.
2. **LDS codebook lookup**: 4096-entry int16 codebook in 8 KB shared memory. 1-cycle access (vs 10-cycle L1). 8 blocks/CU = 80% occupancy.
3. **O(1) escape handling**: `escape_offsets[row × 256 + tid]` gives each thread its pre-computed pointer. On `idx == 4095`, just read and increment. Zero scanning.
4. **2× loop unroll**: Two elements per iteration with incremental bit positions (iadd64 vs imul64). Dual accumulators for ILP.
5. **FMA accumulate**: `sum += weight × activation`, wavefront shuffle reduction.

Batch kernels (B=2, B=4, B=8) share the decode step across multiple activation vectors — same codebook lookup, multiple FMAs.

### Correctness

Each tensor verified **bit-exact** via a decompress kernel that decodes every 12-bit index and compares the raw int16 (BF16 bits) against the original tensor. Zero mismatches on 226/226 tensors. Batch kernels verified bit-identical to individual single matvec calls.

## Reproducing Results

### Prerequisites

- AMD MI50 32GB GPU with ROCm installed
- Python 3.10+ with `torch` (ROCm), `safetensors`, `numpy`
- Llama 3.1 8B in BF16 safetensors format in `models/llama-3.1-8b/`

### Build

```bash
# Compile C packer
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c

# Compile HIP kernel
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
```

### Run Full Benchmark (B=1, all 226 tensors)

```bash
python3 bench_fixed12.py models/llama-3.1-8b/llama-3.1-8b.safetensors
```

Output: per-tensor timing, speedup vs BF16, bit-exact lossless status, weighted average.

### Run Batch Benchmark

```python
# In Python — example for B=4
import ctypes
_hip = ctypes.CDLL("decompress_v2.so")
_hip.launch_fixed12_batch4.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int] * 2
_hip.launch_fixed12_batch4(
    packed_ptr, codebook_ptr,
    act0_ptr, act1_ptr, act2_ptr, act3_ptr,
    escape_offsets_ptr, escape_vals_ptr,
    out0_ptr, out1_ptr, out2_ptr, out3_ptr,
    M, K)
```

### File Overview

| File | Purpose |
|------|---------|
| **`decompress_v2.hip`** | Production kernels: B=1, B=2, B=4, B=8 matvec + decompress for verification |
| **`fixed12_pack.c`** | C packer: codebook construction + 12-bit packing + per-thread escape table |
| **`bench_fixed12.py`** | Benchmark driver: GPU freq sort, C packing, kernel timing, bit-exact verification |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length for disk storage) |
| `tlc_decode.py` | .tlc CPU decoder |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_verify.py` | .tlc round-trip lossless verification |
| `tlc_runtime.py` | GPU runtime for .tlc models (WIP — needs fused kernel integration) |
| `bitpack_fast.c` | C packer for .tlc variable-length format |
| `bitunpack_fast.c` | C unpacker for .tlc variable-length format |

## Use Case

**Accelerates autoregressive decoding** (token generation) where each step does matrix × vector. Memory-bandwidth bound — our 12-bit format reads 25% less data with negligible decode overhead. Batch matvec further amortizes decode cost for concurrent request serving (vLLM, TGI).

**Not beneficial for prefill** (prompt processing) which is compute-bound.

## Target Hardware

- AMD MI50 32GB / MI300X (ROCm / HIP) — tested
- NVIDIA H100 / H200 / A100 (CUDA) — portable, requires warp size 64→32 adjustment
