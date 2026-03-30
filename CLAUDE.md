# Turbo Lossless: BF16 Compression Engine

100% bit-perfect lossless inference engine for BF16 safetensors models.

**Input: BF16 safetensors only.** No GGUF, no FP16, no FP32, no quantized formats.

## Instructions

- Always use team mode (multi-agent parallelism) to accelerate development speed.
- Do not push to remote repo unless explicitly told to.
- When benchmarking, use a non-display GPU (not the one connected to the monitor).

## Current Results

- **2.90x** weighted avg inference speedup (fused kernel, all 226 Llama 8B tensors, bit-perfect)
- **353 GB/s** effective bandwidth on large tensors (34% of MI50 32GB's 1 TB/s HBM2 peak)
- **1.47x** disk compression (.tlc 8-tier variable-length)
- **1.33x** VRAM compression (12-bit fixed-width + escape table)
- Validated compression on 11 BF16 models, 7 architectures, 8B–671B params
- Use case: autoregressive decoding (memory-bound matvec), not prefill (compute-bound matmul)

## Fused Kernel Architecture (decompress_v2.hip)

Single GPU kernel launch per matvec. Decodes 12-bit packed indices, looks up LDS codebook, handles escapes inline via per-thread offset table, accumulates FMA.

**Hot loop per element (~8 ALU + 3 memory ops):**
1. Incremental bit position (iadd64, avoids imul64)
2. Branchless 64-bit read → 12-bit index (2 global loads + shift + mask)
3. LDS codebook lookup (1 cycle, 8 KB int16)
4. bf16_to_float (2 ALU)
5. Activation read + bf16_to_float (1 global load + 2 ALU)
6. FMA accumulate
7. Escape check (branch predicted 99.92%)

**Design constraints on MI50 32GB:**
- 8 KB LDS (int16 codebook) = 8 blocks/CU = 80% occupancy (sweet spot)
- 16 KB LDS (float32 codebook) = 4 blocks/CU = 40% slower (occupancy kills it)
- L1 cache codebook = 10-cycle latency = 22% slower than LDS
- Kernel is compute-bound at 34% of 1 TB/s HBM bandwidth

**Optimization history (this session):**
- 0.05x: atomicAdd patches — MI50 CAS retry loop stall
- 1.73x: CSR wavefront-parallel patches, no atomics
- 1.88x: Fused kernel + 2x unroll + per-thread escape O(1) table
- 2.29x: Branchless 64-bit read — eliminated 37% warp divergence
- 2.87x: LDS codebook — 1-cycle vs 10-cycle L1 access
- **2.90x: Incremental bit positions + dead code removal + bit-exact verification**

**Tested and rejected:**
- Float32 LDS codebook: -40% (occupancy)
- Split 2K LDS + L1: -12% (branch divergence)
- LDS activation cache: -40% (occupancy)
- Float32 activations: -2% (bandwidth)
- 4x unroll: -1% (register pressure)
- Software pipelining: -1% (compiler already does it)
- Fused merge-scan: -19x on token_embd
- Fused binary search: -2.2x on token_embd
- __launch_bounds__ hints: 0% (compiler is smart)

## Correctness Verification

Bit-exact verification using decompress kernel (`fixed12_decompress_fused`):
1. Decompress all weights via same codebook + escape logic
2. Compare every int16 (BF16 bit pattern) against original tensor
3. Zero mismatches required: `(decoded != original).sum() == 0`
4. Stronger than matvec error threshold which could miss canceling errors

## Packing (fixed12_pack.c)

`pack_fixed12_fused()`: Per-thread-stride escape layout
- Escape values ordered by `(row, tid=col%256, encounter_order)`
- `escape_offsets[row * 256 + tid]` = exclusive prefix sum over (row, tid) counts
- Three passes: count escapes per (row,tid), prefix sum, fill values

## Compression Validation

| Model | Params | Type | CR | Escape % |
|-------|--------|------|----|----------|
| Llama 3.1 8B | 8B | Dense | 1.509x | 0.03% |
| Llama 3.1 70B | 70B | Dense | 1.516x | 0.05% |
| Mistral Large 123B | 123B | Dense | 1.503x | 0.03% |
| MiniMax-Text-01 | 456B | MoE | 1.507x | 0.03% |
| DeepSeek V3 | 671B | MoE | 1.185x | 2.02% |

## Implementation Status

| Component | Status |
|-----------|--------|
| Fused GPU kernel (12-bit matvec) | Done — decompress_v2.hip |
| Bit-exact verification kernel | Done — decompress_v2.hip |
| Benchmark harness | Done — bench_fixed12.py |
| C packer (fused escape layout) | Done — fixed12_pack.c |
| .tlc encoder/decoder | Done — tlc_encode.py, tlc_decode.py |
| Round-trip verification | Done — tlc_verify.py |
| End-to-end runtime | WIP — tlc_runtime.py needs fused kernel wiring |
| NVIDIA/CUDA port | TODO — warp size 64→32 |
| Batch matvec (small batch decode) | TODO — reuse weight reads across batch |

## Scope

- **Target**: Dense BF16 safetensors models (Llama, Mistral, Phi, Qwen, Yi, etc.)
- **Best for**: Autoregressive decoding (memory-bound matvec)
- **Works well**: All dense + small MoE → 1.50x+ compression, 2.90x inference
- **Partial**: Large MoE (DeepSeek V3) → 1.2x compression
- **Not beneficial**: Prefill (compute-bound), GPTQ/AWQ INT4 (~1.11x)
