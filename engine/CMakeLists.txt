cmake_minimum_required(VERSION 3.16)
project(turbo-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Use hipcc for everything — it handles both C++ and HIP
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

add_executable(turbo-engine
    main.cpp
    model.cpp
    inference.cpp
    tokenizer.cpp
    sampler.cpp
    kernels.hip
)

target_compile_options(turbo-engine PRIVATE -O3 --offload-arch=gfx906)
target_link_libraries(turbo-engine PRIVATE hipblas)
