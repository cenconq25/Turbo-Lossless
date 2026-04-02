/**
 * GPU Compatibility Layer — AMD HIP / NVIDIA CUDA
 *
 * Auto-detects platform at compile time:
 *   - AMD (hipcc):  native HIP, 64-wide wavefronts
 *   - NVIDIA (nvcc): maps HIP names → CUDA equivalents, 32-wide warps
 *
 * Usage: #include "gpu_compat.h" instead of <hip/hip_runtime.h> or <cuda_runtime.h>
 */
#pragma once

// ============================================================
// Platform detection
// ============================================================
#if defined(__HIP__) || defined(__HIPCC__)
#define TURBO_AMD 1
#define TURBO_WARP_SIZE 64
#include <hip/hip_runtime.h>

#else  // NVIDIA CUDA
#define TURBO_NVIDIA 1
#define TURBO_WARP_SIZE 32
#include <cuda_runtime.h>

// --- HIP → CUDA API mapping ---

// Memory management
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemGetInfo            cudaMemGetInfo
#define hipMemcpyHostToDevice    cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost    cudaMemcpyDeviceToHost

// Device management
#define hipSetDevice             cudaSetDevice
#define hipGetDevice             cudaGetDevice
#define hipDeviceSynchronize     cudaDeviceSynchronize
#define hipGetLastError          cudaGetLastError
#define hipGetErrorString        cudaGetErrorString
#define hipSuccess               cudaSuccess

// Stream types and management
#define hipStream_t              cudaStream_t
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking     cudaStreamNonBlocking
#define hipStreamDestroy         cudaStreamDestroy
#define hipStreamSynchronize     cudaStreamSynchronize

// Event types and management
#define hipEvent_t               cudaEvent_t
#define hipEventCreateWithFlags  cudaEventCreateWithFlags
#define hipEventCreate           cudaEventCreate
#define hipEventDisableTiming    cudaEventDisableTiming
#define hipEventDestroy          cudaEventDestroy
#define hipEventRecord           cudaEventRecord
#define hipStreamWaitEvent       cudaStreamWaitEvent
#define hipEventSynchronize      cudaEventSynchronize
#define hipEventElapsedTime      cudaEventElapsedTime

// Kernel launch (hipLaunchKernelGGL → triple-angle-bracket)
#define hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...) \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)

// Warp shuffle: CUDA requires _sync variants with explicit mask
#define __shfl(val, srcLane, width)      __shfl_sync(0xFFFFFFFF, val, srcLane, width)
#define __shfl_up(val, delta, width)     __shfl_up_sync(0xFFFFFFFF, val, delta, width)
#define __shfl_down(val, delta, width)   __shfl_down_sync(0xFFFFFFFF, val, delta, width)
#define __shfl_xor(val, laneMask, width) __shfl_xor_sync(0xFFFFFFFF, val, laneMask, width)

#endif  // platform detection

// ============================================================
// Platform-independent constants
// ============================================================
#define TURBO_WORKGROUP_SIZE 256
#define TURBO_NUM_WARPS      (TURBO_WORKGROUP_SIZE / TURBO_WARP_SIZE)

// Suppress nodiscard warnings portably
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-result"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

// ============================================================
// GPU error checking macro
// ============================================================
#include <cstdio>
#include <cstdlib>

#define GPU_CHECK(call) do { \
    auto _gpu_err = (call); \
    if (_gpu_err != hipSuccess) { \
        fprintf(stderr, "GPU error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(_gpu_err)); \
        exit(1); \
    } \
} while(0)
