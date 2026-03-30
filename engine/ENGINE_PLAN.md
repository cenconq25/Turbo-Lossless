# Turbo Lossless Inference Engine

Lightweight C++ inference engine with batched 12-bit compressed decode.

## Target
- Mistral 7B / Llama 8B on single MI50 32GB
- Multi-user batched decode (B=1 to B=8)
- Beat llama.cpp BF16 (32.5 tok/s) for multi-user serving

## Architecture

```
[Tokenizer] → [Model Loader] → [Forward Pass] → [Sampler] → [Detokenizer]
                                      |
                    [RMSNorm] [RoPE] [Attention] [MLP]
                                                   |
                                         [Our 12-bit Fused Kernel]
```

## Components

### 1. Model Loader (`model.h/cpp`)
- Load BF16 safetensors from disk
- Compress weights to 12-bit on GPU (reuse our C packer)
- Upload codebook + packed data + escape table to GPU
- Load config.json for model dimensions

### 2. HIP Kernels (`kernels.hip`)
- `rms_norm`: RMSNorm (simple: normalize + scale)
- `rope`: Rotary Position Embeddings
- `attention`: Q×K^T softmax, ×V (use hipBLAS for matmul parts)
- `silu_mul`: SiLU activation × gate (for SwiGLU MLP)
- `embed_lookup`: token → embedding vector
- `argmax`: greedy sampling
- Reuse: `decompress_v2.hip` for weight matvec

### 3. Forward Pass (`inference.h/cpp`)
- One transformer layer: norm → attn → norm → mlp
- KV cache: pre-allocated [layers × max_seq × head_dim]
- Support B=1 and B=4 batch decode

### 4. Tokenizer (`tokenizer.h/cpp`)
- Use sentencepiece library (Mistral/Llama tokenizer)
- Encode prompt → token IDs
- Decode token IDs → text

### 5. Server (`main.cpp`)
- CLI mode: single prompt → generate
- Batch mode: multiple prompts, batched decode

## Build
```bash
cd engine
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON
cmake --build build
./build/turbo-engine models/mistral-7b "The capital of France is"
```

## Dependencies
- ROCm / HIP (GPU compute)
- hipBLAS (attention matmul)
- sentencepiece (tokenizer)
- nlohmann/json (config parsing)
- safetensors C reader (weight loading)

## File Structure
```
engine/
  CMakeLists.txt
  main.cpp              — CLI entry point
  model.h / model.cpp   — model loading + weight compression
  inference.h / .cpp    — forward pass orchestration
  kernels.hip           — all HIP kernels (norm, rope, attn, silu, embed)
  tokenizer.h / .cpp    — sentencepiece wrapper
  sampler.h / .cpp      — greedy / top-p sampling
  ../decompress_v2.hip  — our fused 12-bit matvec kernel (reuse)
  ../fixed12_pack.c     — C packer (reuse)
```
