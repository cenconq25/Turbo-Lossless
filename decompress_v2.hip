/**
 * Turbo Lossless — V2 12-bit Matvec Kernels (AMD MI50 / gfx906)
 *
 * Three kernel variants:
 *   1. Fused: single launch, inline escape via merge-scan of CSR patches
 *   2. Two-pass: matvec + separate wavefront-parallel patch correction
 *
 * Common optimizations:
 *   - L1-cached codebook (8 KB fits in 16 KB L1, zero LDS overhead)
 *   - 2x loop unroll for instruction-level parallelism
 *   - Contiguous tile access (adjacent threads read adjacent bit positions)
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

// Branchless 12-bit read: always load 2 words, combine via 64-bit shift.
// Eliminates warp divergence from word-boundary crossing (37% of reads).
__device__ __forceinline__
uint32_t read12(const uint32_t* __restrict__ p, int64_t bp) {
    int w = bp >> 5;
    int s = bp & 31;
    uint64_t combined = (uint64_t)p[w] | ((uint64_t)p[w + 1] << 32);
    return (uint32_t)((combined >> s) & 0xFFF);
}

// ---------------------------------------------------------------------------
// Fused kernel: single launch, O(1) escape via per-thread offset table
// ---------------------------------------------------------------------------
// escape_offsets[row * 256 + tid] = index into escape_vals where this
// thread's escapes begin. Thread just increments a pointer on each escape.
// Zero scanning, zero divergent loops, zero atomics.
__global__ void fixed12_matvec_fused(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ activations,
    const int32_t*  __restrict__ escape_offsets,  // M * 256
    const int16_t*  __restrict__ escape_vals,
    float*          __restrict__ output,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    // O(1) lookup: where this thread's escapes start
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    // LDS codebook (8 KB) — 1-cycle access, compute-bound kernel
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[4];

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    if (tid < 4) warp_sums[tid] = 0.0f;
    __syncthreads();
    const int16_t* act = activations;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    // Precompute bit positions — use incremental adds instead of per-iteration multiply
    // col * 12 as 64-bit multiply is ~4 cycles on GCN (no native imul64)
    // Incremental: bp += stride_bits is just a 64-bit add (~2 cycles)
    int64_t bp0 = (int64_t)row * K * 12 + (int64_t)tid * 12;
    const int64_t stride_bits = (int64_t)WORKGROUP_SIZE * 12;      // 3072
    const int64_t stride_bits_2x = stride_bits * 2;                // 6144

    // 2x unrolled with LDS codebook + incremental bit positions
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp0);
        uint32_t idx1 = read12(packed, bp0 + stride_bits);

        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0])
            : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1])
            : bf16_to_float(escape_vals[esc_ptr++]);

        sum0 += w0 * bf16_to_float(activations[col]);
        sum1 += w1 * bf16_to_float(activations[col + WORKGROUP_SIZE]);
        bp0 += stride_bits_2x;
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp0);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx])
            : bf16_to_float(escape_vals[esc_ptr++]);
        sum0 += w * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);

    if ((tid % WARP_SIZE) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();

    if (tid == 0)
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Two-pass matvec: multi-row blocks for small K, single-row for large K
// ---------------------------------------------------------------------------
// For K <= 16384: 4 rows per block (one wavefront per row, 64 threads each)
//   - Each thread does K/64 columns with 2x unroll
//   - L1 codebook shared across 4 rows
// For K > 16384: 1 row per block (4 wavefronts, 256 threads)
//   - Each thread does K/256 columns with 2x unroll

// Multi-row: 4 rows per block, one wavefront per row
__global__ void fixed12_matvec_multirow(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    int base_row = blockIdx.x * 4;
    int warp_id  = threadIdx.x / WARP_SIZE;   // 0..3
    int lane     = threadIdx.x % WARP_SIZE;   // 0..63
    int row      = base_row + warp_id;
    if (row >= M) return;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    int64_t row_base = (int64_t)row * K * 12;

    int col = lane;
    for (; col + WARP_SIZE < K; col += WARP_SIZE * 2) {
        int64_t bp0 = row_base + (int64_t)col * 12;
        int64_t bp1 = row_base + (int64_t)(col + WARP_SIZE) * 12;
        uint32_t idx0 = read12(packed, bp0);
        uint32_t idx1 = read12(packed, bp1);
        sum0 += bf16_to_float(codebook[idx0]) * bf16_to_float(activations[col]);
        sum1 += bf16_to_float(codebook[idx1]) * bf16_to_float(activations[col + WARP_SIZE]);
    }
    if (col < K) {
        int64_t bp = row_base + (int64_t)col * 12;
        uint32_t idx = read12(packed, bp);
        sum0 += bf16_to_float(codebook[idx]) * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;

    // Wavefront reduction (within each warp independently)
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);

    if (lane == 0)
        output[row] = sum;
}

// Single-row: 1 row per block, 256 threads (4 wavefronts cooperate)
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

    int col = tid;
    for (; col + stride < K; col += stride * 2) {
        int64_t bp0 = row_base + (int64_t)col * 12;
        int64_t bp1 = row_base + (int64_t)(col + stride) * 12;
        uint32_t idx0 = read12(packed, bp0);
        uint32_t idx1 = read12(packed, bp1);
        sum0 += bf16_to_float(codebook[idx0]) * bf16_to_float(activations[col]);
        sum1 += bf16_to_float(codebook[idx1]) * bf16_to_float(activations[col + stride]);
    }
    if (col < K) {
        int64_t bp = row_base + (int64_t)col * 12;
        uint32_t idx = read12(packed, bp);
        sum0 += bf16_to_float(codebook[idx]) * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;

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

// ---------------------------------------------------------------------------
// C launch wrappers
// ---------------------------------------------------------------------------
extern "C" {

int launch_fixed12_fused(
    const void* packed, const void* codebook, const void* activations,
    const void* escape_offsets, const void* escape_vals,
    void* output, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_fused,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (float*)output, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

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
