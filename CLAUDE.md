# llama-mixed-precision-engine

Fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) for experimenting with mixed-precision inference strategies.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Common CMake options:
- `-DGGML_CUDA=ON` — enable CUDA backend
- `-DGGML_METAL=ON` — enable Metal backend (macOS)
- `-DGGML_BLAS=ON` — enable BLAS backend
- `-DBUILD_SHARED_LIBS=OFF` — static build

Binaries are output to `build/bin/`.

## Project structure

- `ggml/` — tensor library (backends: CPU, CUDA, Metal, Vulkan, etc.)
  - `ggml/src/ggml-cpu/` — CPU backend and quantization kernels
  - `ggml/src/ggml-cuda/` — CUDA backend
  - `ggml/include/` — public headers
- `src/` — llama.cpp core library (model loading, inference, sampling)
- `include/` — public API headers (`llama.h`)
- `common/` — shared utilities used by examples/tools
- `examples/` — example programs (batched inference, embedding, etc.)
- `tools/` — standalone tools (server, quantize, etc.)
- `convert_hf_to_gguf.py` — HuggingFace to GGUF model converter
- `gguf-py/` — Python GGUF library

## Key files for mixed-precision work

- `ggml/src/ggml-quants.c` — quantization/dequantization implementations
- `ggml/src/ggml-cpu/` — CPU compute kernels (matrix multiply, etc.)
- `ggml/src/ggml-cuda/` — CUDA kernels
- `ggml/include/ggml.h` — tensor types and quantization format enums
- `src/llama.cpp` — model loading, layer graph construction
- `src/llama-model.cpp` — model architecture definitions

## Testing

```bash
cd build && ctest --output-on-failure
```

## Code style

- C/C++ codebase, C11/C++17
- 4-space indentation
- `snake_case` for functions/variables
- `UPPER_CASE` for macros and enum values
- Prefix public API: `ggml_`, `llama_`, `common_`
