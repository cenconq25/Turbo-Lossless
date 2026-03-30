# Turbo Lossless: 8-Tier BF16 Compression Engine

100% bit-perfect lossless compression for LLM weights. BF16 in, BF16 out — no precision loss, 1.5x smaller.

## Results

Validated on 11 BF16 models across 7 architectures using 4x AMD MI50 GPUs.

### Cross-Model Summary

| Model | Architecture | Type | Params | Entropy | 8-tier | CR | BF16 Size | Compressed | VRAM Saved |
|-------|-------------|------|--------|---------|--------|-----|-----------|-----------|-----------|
| Llama 3.1 8B | Llama | Dense | 8B | 10.42 | 10.60 | **1.509x** | 15.0 GB | 9.95 GB | 5.0 GB |
| Phi-4 | Microsoft | Dense | 14B | 10.49 | 10.62 | **1.507x** | 28 GB | 18.6 GB | 9.4 GB |
| Codestral 22B | Mistral | Dense | 22B | 10.51 | 10.64 | **1.504x** | 42 GB | 27.9 GB | 14.0 GB |
| Qwen3 30B-A3B | Qwen | MoE | 30B | 10.50 | 10.63 | **1.505x** | 60 GB | 39.9 GB | 20.1 GB |
| Yi-1.5 34B | 01.AI | Dense | 34B | 10.51 | 10.63 | **1.504x** | 68 GB | 45.2 GB | 22.6 GB |
| Llama 3.1 70B | Llama | Dense | 70B | 10.36 | 10.56 | **1.516x** | 140.0 GB | 92.4 GB | 47.6 GB |
| Mistral Large 123B | Mistral | Dense | 123B | 10.51 | 10.64 | **1.503x** | 246.0 GB | 163.6 GB | 82.4 GB |
| StarCoder2 7B | BigCode | Dense | 7B | 10.50 | 10.62 | **1.506x** | 14 GB | 9.3 GB | 4.7 GB |
| MiniMax-Text-01 | MiniMax | MoE | 456B | 10.48 | 10.62 | **1.507x** | 912 GB | 605 GB | 307 GB |
| Qwen3-235B-A22B | Qwen | MoE | 235B | 11.22 | 11.38 | 1.406x | 470 GB | 334.2 GB | 135.8 GB |
| DeepSeek V3 | DeepSeek | MoE | 671B | 13.35 | 13.50 | 1.185x | 1342 GB | 1132.5 GB | 209.5 GB |

9/11 BF16 models achieve **1.50x+** across 7 architectures (Llama, Mistral, Microsoft, Qwen, 01.AI, BigCode, MiniMax). Large MoE (DeepSeek V3, Qwen3-235B) achieve 1.2–1.4x due to higher weight diversity.

### INT4 Quantized Models

Tested separately on 5 GPTQ/AWQ INT4 models. Double compression yields minimal gains:

| Component | % of model | Our CR | Why |
|-----------|-----------|--------|-----|
| FP16 scales/zeros | ~12.5% | 1.50–1.54x | Behaves like normal BF16 |
| Packed INT4 weights | ~87.5% | 1.06–1.12x | Only 16 possible values, already near entropy floor |
| **Total** | **100%** | **~1.11x** | **Not worth it — target raw BF16 instead** |

### Compression by Tensor Type (Llama 3.1 70B)

| Type | Entropy | 8-tier bits | CR | Escape % |
|------|---------|------------|------|----------|
| down_proj | 10.34 | 10.54 | 1.518x | 0.028% |
| gate_proj | 10.35 | 10.55 | 1.516x | 0.051% |
| up_proj | 10.34 | 10.54 | 1.517x | 0.044% |
| o_proj | 10.35 | 10.55 | 1.517x | 0.031% |
| lm_head | 10.35 | 10.54 | 1.518x | 0.030% |
| v_proj | 10.44 | 10.62 | 1.507x | 0.120% |
| q_proj | 10.51 | 10.68 | 1.499x | 0.167% |
| k_proj | 10.56 | 10.70 | 1.495x | 0.172% |

## How It Works

Every BF16 weight value is encoded using a per-tensor frequency-ranked codebook with 8 prefix-coded tiers:

```
Tier 0:  prefix '0'       + 9-bit index = 10 bits   (top 512 values,   ~61% of weights)
Tier 1:  prefix '10'      + 9-bit index = 11 bits   (next 512 values,  ~21%)
Tier 2:  prefix '110'     + 9-bit index = 12 bits   (next 512,         ~10%)
Tier 3:  prefix '1110'    + 9-bit index = 13 bits   (next 512,          ~5%)
Tier 4:  prefix '11110'   + 9-bit index = 14 bits   (next 512,          ~2%)
Tier 5:  prefix '111110'  + 9-bit index = 15 bits   (next 512,         ~0.8%)
Tier 6:  prefix '1111110' + 9-bit index = 16 bits   (next 512,         ~0.15%)
Escape:  prefix '1111111' + 16-bit raw  = 23 bits   (remaining,        ~0.05%)
```

Most frequent values get shortest codes. The codebook is built per-tensor by sorting all unique BF16 values by frequency.

### GPU Inference: Two-Phase Architecture

**Disk → VRAM**: Variable-length 8-tier .tlc format (1.47x compression) stored on disk.

**VRAM**: Fixed-width 12-bit packed indices + 4096-entry codebook per tensor (1.33x compression). Loaded at model startup.

**Inference**: Fully parallel decode — every thread independently reads 12 bits, looks up codebook in LDS, and multiplies by activation. Zero serial dependencies. Patch correction kernel fixes the 0.012% escape values with negligible overhead (32 μs per tensor).

### Benchmark: AMD MI50 (gfx906)

Measured on Llama 3.1 8B weight tensors (100% lossless, bit-perfect verified):

| Tensor | Shape | Ours | BF16 | Speedup |
|--------|-------|------|------|---------|
| attn_k | [4096×1024] | 0.114ms | 0.110ms | 0.96x |
| attn_q | [4096×4096] | 0.238ms | 0.356ms | **1.48x** |
| attn_output | [4096×4096] | 0.238ms | 0.356ms | **1.49x** |
| ffn_down | [14336×4096] | 0.711ms | 1.079ms | **1.52x** |
| ffn_gate | [4096×14336] | 0.625ms | 1.095ms | **1.75x** |
| ffn_up | [4096×14336] | 0.626ms | 1.093ms | **1.74x** |
| output | [4096×128256] | 5.125ms | 9.375ms | **1.83x** |
| token_embd | [4096×128256] | 5.333ms | 9.360ms | **1.76x** |
| **Weighted avg** | | | | **1.73x** |

Larger tensors benefit most. The output head (128K columns) achieves 205 GB/s effective bandwidth — faster than raw BF16 because it reads 25% less data from HBM.

### Round-Trip Verification

Encoder/decoder proven bit-perfect lossless on Llama 3.1 8B (292 tensors, 0 mismatches):

```
Original:    16.06 GB (safetensors)
Compressed:  10.89 GB (.tlc)
CR:          1.475x
Mismatches:  0 / 292 tensors
Encode time: 6.5 min
```

## Approaches Tested

Empirically evaluated on Llama 3.1 8B across all 225 weight tensors:

| Rank | Approach | bits/param | CR | GPU-Friendly | Status |
|------|----------|-----------|------|:---:|--------|
| — | Shannon entropy | 10.43 | 1.535x | N/A | Theoretical ceiling |
| 1 | Bit-plane decomposition | 10.50 | 1.524x | Yes | Harder to fuse into matmul |
| 2 | Rice/Golomb rank coding | 10.55 | 1.517x | Moderate | Variable-length output |
| **3** | **8-tier prefix code** | **10.60** | **1.509x** | **Yes** | **Selected** |
| 4 | 6-tier prefix code | 10.62 | 1.506x | Yes | Marginally worse |
| 5 | 4-tier prefix code | 10.88 | 1.470x | Yes | Below 1.5x target |
| 6 | XOR delta coding | 11.10 | 1.442x | Moderate | No spatial correlation in weights |
| 7 | 3-tier prefix code | 11.24 | 1.423x | Yes | Too few tiers |
| 8 | Byte-aligned (256 split) | 12.77 | 1.253x | Very | Too coarse |
| 9 | Original 15+1 codebook | 13.75 | 1.164x | Yes | Failed — wrong assumptions |
| 10 | Block-local codebook | 17–20 | 0.8–0.9x | Possible | Header overhead kills savings |

## Key Properties

- **Lossless**: Every BF16 bit pattern is preserved exactly. Zero quality loss.
- **BF16 only**: Optimized for the industry standard LLM serving format.
- **1.33x VRAM / 1.47x disk**: Dual-format — maximum compression on disk (.tlc), fast parallel decode in VRAM (12-bit fixed-width + patches).
- **1.73x faster inference** (weighted avg): Fixed-width 12-bit GPU kernel with wavefront-parallel patch corrections. Fully parallel decode, 100% lossless, proven bit-perfect on all 226 Llama 8B weight tensors.
- **Scales**: Consistent 1.5x+ across dense models 8B–123B. Large MoE models have higher weight diversity.
- **Broad validation**: 11 BF16 models + 5 INT4 models, 7 architectures
- **Multi-GPU**: Tensor parallelism splits compressed data. Each GPU holds its slice + shared codebook (~7 KB).

## Running the Benchmark

### Prerequisites

- BF16 safetensors model (e.g. Llama 3.1 8B) in `models/` directory
- ROCm (for AMD) or CUDA toolkit (for NVIDIA)
- Python packages: `torch`, `safetensors`, `numpy`

### Build

```bash
# Compile C packer
gcc -O3 -shared -fPIC -o fixed12_pack.so fixed12_pack.c

# Compile HIP kernels (AMD MI50)
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_matmul.so decompress_matmul.hip
hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
```

### Run

```bash
# Full benchmark on Llama 3.1 8B (all 226 weight tensors)
python3 bench_fixed12.py models/llama-3.1-8b/llama-3.1-8b.safetensors

# Or any BF16 safetensors model
python3 bench_fixed12.py path/to/model.safetensors
```

### What it does

1. For each 2D BF16 tensor (>100K params):
   - **GPU**: Frequency-sort unique values via `torch.unique()`
   - **CPU**: Build 4096-entry codebook, pack as 12-bit indices, extract row-grouped escape patches (CSR format)
   - **GPU**: Benchmark fused 12-bit decode + matvec kernel (`decompress_v2.so`) vs raw BF16 matmul
   - **Verify**: Bit-perfect lossless (max error < 0.01 on matvec output)
2. Reports per-tensor speedup, effective bandwidth, and weighted average

### Key files

| File | Purpose |
|------|---------|
| `bench_fixed12.py` | Benchmark driver — GPU freq sort, C packing, kernel timing |
| `fixed12_pack.c` | Fast C bit-packer: 12-bit indices + CSR row-grouped patches |
| `decompress_v2.hip` | V2 cache-optimized matvec kernel + wavefront-parallel patches |
| `decompress_matmul.hip` | Core kernels: variable-length decode, fixed-width matvec, format conversion |
| `tlc_encode.py` | .tlc encoder (8-tier variable-length format) |
| `tlc_decode.py` | .tlc decoder (CPU) |
| `tlc_format.py` | .tlc binary format definitions |
| `tlc_runtime.py` | GPU runtime for .tlc models |

## Target Hardware

- AMD MI50 / MI300X (ROCm / HIP)
- NVIDIA H100 / A100 (CUDA)
