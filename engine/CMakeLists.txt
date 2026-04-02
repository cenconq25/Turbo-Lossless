cmake_minimum_required(VERSION 3.16)
project(turbo-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU platform: AMD (ROCm/HIP) vs NVIDIA (CUDA)
if(NOT DEFINED GPU_PLATFORM)
    if(EXISTS /opt/rocm/bin/hipcc)
        set(GPU_PLATFORM "AMD")
    else()
        set(GPU_PLATFORM "NVIDIA")
    endif()
endif()

message(STATUS "GPU Platform: ${GPU_PLATFORM}")

set(ENGINE_SOURCES
    main.cpp
    model.cpp
    inference.cpp
    tokenizer.cpp
    sampler.cpp
    kernels.hip
    multi_gpu.cpp
)

if(GPU_PLATFORM STREQUAL "AMD")
    # AMD ROCm/HIP
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
    add_executable(turbo-engine ${ENGINE_SOURCES})
    target_compile_options(turbo-engine PRIVATE -O3 --offload-arch=gfx906)
    target_link_libraries(turbo-engine PRIVATE hipblas sentencepiece)

elseif(GPU_PLATFORM STREQUAL "NVIDIA")
    # NVIDIA CUDA
    enable_language(CUDA)
    # Rename .hip files for nvcc (it expects .cu)
    set_source_files_properties(kernels.hip PROPERTIES LANGUAGE CUDA)
    add_executable(turbo-engine ${ENGINE_SOURCES})
    # Default to sm_80 (A100), override with -DCUDA_ARCH=sm_XX
    if(NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH "sm_80")
    endif()
    target_compile_options(turbo-engine PRIVATE -O3)
    set_target_properties(turbo-engine PROPERTIES CUDA_ARCHITECTURES "80")

    # NCCL for multi-GPU tensor parallelism
    set(NCCL_INCLUDE_DIR "/home/roy/miniconda3/lib/python3.13/site-packages/nvidia/nccl/include")
    set(NCCL_LIB_DIR "/home/roy/miniconda3/lib/python3.13/site-packages/nvidia/nccl/lib")
    target_include_directories(turbo-engine PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_directories(turbo-engine PRIVATE ${NCCL_LIB_DIR})
    target_link_libraries(turbo-engine PRIVATE cublas sentencepiece nccl)
endif()
