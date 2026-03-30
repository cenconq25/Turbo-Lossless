# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).

## Current Results

- **2.90x** single matvec speedup (fused kernel, all 226 Llama 8B tensors, bit-perfect)
- **B=4: 2.76x amortization** (per-vector 58% cheaper, 8.69x vs BF16 batch)
- **B=8: 3.39x amortization** (per-vector 0.88ms, output head)
- **353 GB/s** effective BW at B=1 (34% of MI50 32GB's 1 TB/s HBM2 peak)
- **1.47x** disk compression, **1.33x** VRAM compression
- All batch kernels bit-exact with single matvec
- Use case: autoregressive decoding, not prefill

## Kernel Architecture (decompress_v2.hip)

Five production kernels in one file:

| Kernel | Purpose |
|--------|---------|
| `fixed12_matvec_fused` | B=1 matvec (production single-request) |
| `fixed12_matvec_batch2` | B=2 matvec (2 concurrent requests) |
| `fixed12_matvec_batch4` | B=4 matvec (production sweet spot) |
| `fixed12_matvec_batch8` | B=8 matvec (high concurrency) |
| `fixed12_decompress_fused` | Decompress to BF16 buffer (verification) |

All share the same core loop:
1. Branchless 64-bit read → 12-bit index
2. LDS codebook lookup (8 KB int16, 1-cycle)
3. O(1) escape via per-thread offset table
4. 2× unrolled FMA with incremental bit positions
5. Wavefront shuffle + cross-warp LDS reduction

Batch kernels decode weight once, multiply by B activation vectors. Same decode cost, B× FMAs.

## Packing (fixed12_pack.c)

Two functions:
- `build_codebook_12bit()`: Frequency-sort → top 4095 values → codebook + reverse map
- `pack_fixed12_fused()`: 12-bit bit-packing + per-thread-stride escape table

Escape layout: `escape_offsets[row * 256 + tid]` = exclusive prefix sum. Values ordered by `(row, tid=col%256, encounter_order)`. Three passes: count, prefix sum, fill.

## Optimization History

| Step | Speedup | What changed |
|------|---------|-------------|
| Starting point | 0.05x | atomicAdd CAS stall on MI50 (no HW float atomics) |
| CSR wavefront patches | 1.73x | Row-grouped, no atomics, two-pass |
| Fused + 2× unroll | 1.88x | Single kernel, O(1) per-thread escape table |
| Branchless 64-bit read | 2.29x | Eliminated 37% warp divergence |
| LDS codebook | 2.87x | 1-cycle vs 10-cycle L1 (compute-bound at 1 TB/s) |
| Incremental bit positions | 2.90x | iadd64 vs imul64 in hot loop |
| Batch B=2/4/8 | up to 3.39x amort | Decode once, multiply B vectors |

## Tested and Rejected

| Approach | Impact | Reason |
|----------|--------|--------|
| Float32 LDS codebook (16 KB) | -40% | Halves occupancy (4 vs 8 blocks/CU) |
| Split 2K LDS + L1 codebook | -12% | Branch divergence on random idx |
| L1-only codebook | -22% | 10-cycle latency, compute-bound |
| Float32 activations | -2% | Doubles activation bandwidth |
| LDS activation cache | -40% | Occupancy (same as float32 codebook) |
| 4× loop unroll | -1% | Register pressure |
| Software pipelining | -1% | Compiler handles at -O3 |
| Fused CSR merge-scan | -19× on token_embd | O(N) scan for strided access |
| Fused binary search | -2.2× on token_embd | L2 latency per comparison |
| __launch_bounds__ hints | 0% | Compiler already smart |
| 11-bit codebook (2048) | N/A | 9.6% escape rate on attn_k |
| 10-bit codebook (1024) | N/A | 38% escape rate on attn_k |

## Compression Validation

| Model | Params | Type | Disk CR |
|-------|--------|------|---------|
| Llama 3.1 8B | 8B | Dense | 1.509x |
| Llama 3.1 70B | 70B | Dense | 1.516x |
| Mistral Large 123B | 123B | Dense | 1.503x |
| MiniMax-Text-01 | 456B | MoE | 1.507x |
| DeepSeek V3 | 671B | MoE | 1.185x |

11 models, 7 architectures. Dense: 1.50x+. Large MoE: 1.2-1.4x.

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `decompress_v2.hip` | Production | Fused kernels: B=1, B=2, B=4, B=8, decompress |
| `fixed12_pack.c` | Production | Codebook + 12-bit packing + escape table |
| `bench_fixed12.py` | Production | Benchmark driver + bit-exact verification |
| `tlc_encode.py` | Done | .tlc disk format encoder |
| `tlc_decode.py` | Done | .tlc disk format decoder (CPU) |
| `tlc_format.py` | Done | .tlc format definitions |
| `tlc_verify.py` | Done | .tlc round-trip verification |
| `tlc_runtime.py` | WIP | GPU runtime (needs fused kernel wiring) |
| `bitpack_fast.c` | Done | C packer for .tlc variable-length |
| `bitunpack_fast.c` | Done | C unpacker for .tlc variable-length |

## Scope

- **Best for**: Autoregressive decoding of dense BF16 models
- **Production target**: vLLM/TGI integration with batch B=4 kernel
- **Hardware**: AMD MI50 32GB tested; portable to H100/A100 (warp size 64→32)
- **Not beneficial**: Prefill, GPTQ/AWQ INT4 models
