/**
 * Turbo Lossless — V2 Fused 12-bit Matvec Kernel
 * L1-cached codebook + inline escape handling via CSR patches
 *
 * Single kernel launch: decode 12-bit indices, lookup codebook,
 * handle escapes inline (binary search CSR patches), FMA accumulate.
 * No separate patch kernel. No atomics.
 */

#include <hip/hip_runtime.h>
#include <cstdint>

#define WARP_SIZE      64
#define WORKGROUP_SIZE 256

__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

__device__ __forceinline__
uint32_t read12(const uint32_t* __restrict__ p, int64_t bp) {
    int w = bp >> 5;
    int s = bp & 31;
    uint32_t v = p[w] >> s;
    if (s + 12 > 32)
        v |= p[w + 1] << (32 - s);
    return v & 0xFFF;
}

// Binary search for col in sorted patch_cols[start..end)
__device__ __forceinline__
int binary_search(const int32_t* __restrict__ patch_cols, int start, int end, int col) {
    while (start < end) {
        int mid = (start + end) >> 1;
        if (patch_cols[mid] < col)
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

// Fused 12-bit matvec with inline escape correction
// - L1-cached codebook (8 KB, zero LDS)
// - When idx == 4095 (escape), binary search CSR patches for correct value
// - Single kernel launch, no atomics, no separate patch pass
__global__ void fixed12_matvec_fused(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ activations,
    const int32_t*  __restrict__ row_offsets,   // CSR: M+1 entries
    const int32_t*  __restrict__ patch_cols,    // sorted column indices per row
    const int16_t*  __restrict__ patch_vals,    // correct BF16 values
    float*          __restrict__ output,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    // Load this row's patch range once (L1-cached after first wavefront reads it)
    int patch_start = row_offsets[row];
    int patch_end   = row_offsets[row + 1];

    __shared__ float warp_sums[4];
    if (tid < 4) warp_sums[tid] = 0.0f;
    __syncthreads();

    float sum = 0.0f;
    int64_t row_base = (int64_t)row * K * 12;

    for (int col = tid; col < K; col += WORKGROUP_SIZE) {
        int64_t bp = row_base + (int64_t)col * 12;
        uint32_t idx = read12(packed, bp);

        float w_val;
        if (__builtin_expect(idx != 4095, 1)) {
            w_val = bf16_to_float(codebook[idx]);
        } else {
            // Escape: binary search this row's patches for col
            int pos = binary_search(patch_cols, patch_start, patch_end, col);
            w_val = bf16_to_float(patch_vals[pos]);
        }

        sum += w_val * bf16_to_float(activations[col]);
    }

    // Wavefront reduction
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);

    if ((tid % WARP_SIZE) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();

    if (tid == 0)
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// Two-pass matvec kernel: L1-cached codebook, 2x unrolled inner loop
__global__ void fixed12_matvec_v2(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    __shared__ float warp_sums[4];
    if (tid < 4) warp_sums[tid] = 0.0f;
    __syncthreads();

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    int64_t row_base = (int64_t)row * K * 12;
    int stride = WORKGROUP_SIZE;

    // 2x unrolled: process two columns per iteration for ILP
    int col = tid;
    for (; col + stride < K; col += stride * 2) {
        int64_t bp0 = row_base + (int64_t)col * 12;
        int64_t bp1 = row_base + (int64_t)(col + stride) * 12;
        uint32_t idx0 = read12(packed, bp0);
        uint32_t idx1 = read12(packed, bp1);
        sum0 += bf16_to_float(codebook[idx0]) * bf16_to_float(activations[col]);
        sum1 += bf16_to_float(codebook[idx1]) * bf16_to_float(activations[col + stride]);
    }
    // Handle remainder
    if (col < K) {
        int64_t bp = row_base + (int64_t)col * 12;
        uint32_t idx = read12(packed, bp);
        sum0 += bf16_to_float(codebook[idx]) * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;

    // Wavefront reduction
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);

    if ((tid % WARP_SIZE) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();

    if (tid == 0)
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

__global__ void apply_patches(
    const int32_t*  __restrict__ row_offsets,
    const int32_t*  __restrict__ patch_cols,
    const int16_t*  __restrict__ correct_vals,
    const int16_t*  __restrict__ wrong_vals,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row];
    int end   = row_offsets[row + 1];

    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE) {
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(activations[patch_cols[p]]);
    }

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);

    if (tid == 0 && correction != 0.0f)
        output[row] += correction;
}

extern "C" {

// Fused kernel — single launch, handles escapes inline
int launch_fixed12_fused(
    const void* packed, const void* codebook, const void* activations,
    const void* row_offsets, const void* patch_cols, const void* patch_vals,
    void* output, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_fused,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)patch_vals,
        (float*)output, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

// Legacy two-pass interface
int launch_fixed12_v2(
    const void* packed, const void* codebook,
    const void* activations, void* output, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_v2,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations, (float*)output, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

int launch_patches_v2(
    const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M)
{
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches,
        dim3(M), dim3(WARP_SIZE), 0, 0,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const int16_t*)activations, (float*)output, M);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

}
