/**
 * Turbo Lossless — Fused 12-bit Matvec Kernel
 * Target: AMD MI50 32GB (gfx906), 1 TB/s HBM2, 60 CUs, 64-wide wavefronts
 *
 * Performance: 2.90x lossless speedup, 353 GB/s effective BW, bit-perfect on 226/226 tensors
 *
 * Architecture:
 *   - 12-bit packed indices (4096-entry codebook, idx 4095 = escape sentinel)
 *   - Single fused kernel: decode + escape correction + dot product in one launch
 *   - Per-thread escape offset table for O(1) inline escape handling
 *
 * Hot loop per element (~8 ALU + 3 memory ops):
 *   1. bp += stride_bits                        (1 iadd64, incremental — avoids imul64)
 *   2. read12: load 2×uint32, combine, shift    (2 global loads, 3 ALU — branchless)
 *   3. cb[idx]: LDS codebook lookup              (1 LDS read, 1 cycle — 8 KB, 4096 int16)
 *   4. bf16_to_float(cb[idx])                   (2 ALU — zero-extend + shift)
 *   5. bf16_to_float(activations[col])          (1 global load + 2 ALU)
 *   6. sum += w * a                             (1 FMA)
 *   7. escape check: idx != 4095               (~0 — branch predicted 99.92%)
 *
 * Memory hierarchy choices (MI50 32GB):
 *   - Codebook in LDS (8 KB int16): 1-cycle access, 8 blocks/CU = 80% occupancy
 *     REJECTED: float32 LDS (16 KB) — halves occupancy, 40% slower despite no bank conflicts
 *     REJECTED: L1 cache only — 10-cycle latency, 22% slower than LDS
 *     REJECTED: split 2K LDS + L1 — extra branch divergence, 12% slower
 *   - Activations via L2 (4 MB shared): int16, bf16_to_float per access
 *     REJECTED: float32 pre-converted — doubles activation bandwidth, net loss
 *   - Packed data via HBM: branchless uint64 read, ~100 cycle latency hidden by 32 wavefronts
 *     REJECTED: branching read12 — 37% branch rate caused warp divergence, 22% slower
 *
 * Escape handling:
 *   - escape_offsets[row * 256 + tid] = pre-computed pointer per thread
 *   - On idx==4095: read escape_vals[esc_ptr++], O(1), zero scanning
 *     REJECTED: CSR merge-scan — O(patches/row) scan per escape, 19x slower on token_embd
 *     REJECTED: binary search — L2 latency per comparison, 2.2x slower on token_embd
 *     REJECTED: atomicAdd patches — MI50 lacks HW float atomics, CAS loop 580ms stall
 *
 * Loop optimization:
 *   - 2x unroll: best ILP without register pressure
 *     REJECTED: 4x unroll — register pressure reduces occupancy, 1% slower
 *     REJECTED: software pipelining — compiler already schedules at -O3
 *   - Incremental bit position: bp += const instead of col * 12 (iadd64 vs imul64)
 *
 * Build: hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
 */

#include <hip/hip_runtime.h>
#include <cstdint>

#define WARP_SIZE      64   // MI50 wavefront width
#define WORKGROUP_SIZE 256  // 4 wavefronts per block

// BF16 → float32: BF16 is upper 16 bits of IEEE float32
__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

// Branchless 12-bit extract: always load 2 words as uint64, shift, mask.
// 12-bit values cross 32-bit word boundaries 37% of the time.
// Branching version caused severe warp divergence → 22% slower.
__device__ __forceinline__
uint32_t read12(const uint32_t* __restrict__ p, int64_t bp) {
    int w = bp >> 5;
    int s = bp & 31;
    uint64_t combined = (uint64_t)p[w] | ((uint64_t)p[w + 1] << 32);
    return (uint32_t)((combined >> s) & 0xFFF);
}

// ---------------------------------------------------------------------------
// Fused matvec: y[row] = Σ_k decode(packed[row,k]) × activations[k]
// ---------------------------------------------------------------------------
// Grid: dim3(M), Block: dim3(256)
// One block per row. 4 wavefronts cooperate via shared warp_sums reduction.
// LDS: 8 KB codebook (int16[4096]) + 16 bytes warp_sums = 8208 bytes
// Max concurrent blocks per CU: floor(64KB / 8208) = 7
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_fused(
    const uint32_t* __restrict__ packed,        // 12-bit packed weight indices [M×K×12 bits]
    const int16_t*  __restrict__ codebook,      // BF16 codebook [4096] (same for all rows)
    const int16_t*  __restrict__ activations,   // BF16 activation vector [K]
    const int32_t*  __restrict__ escape_offsets, // per-thread escape pointers [M × 256]
    const int16_t*  __restrict__ escape_vals,    // correct BF16 values for escapes
    float*          __restrict__ output,         // float32 result vector [M]
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // O(1) escape: each thread gets its pre-computed start pointer
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    // Load codebook into LDS — 1-cycle access vs ~10-cycle L1
    // 8 KB as int16 = 8 blocks/CU at 80% occupancy (sweet spot for MI50)
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[4];

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    __syncthreads();

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    // Incremental bit position: bp += const avoids imul64 (~4 cyc → iadd64 ~2 cyc)
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;

    // Main loop: 2x unrolled, process 2 elements per iteration per thread
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        // Branchless 12-bit decode (2 elements)
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));

        // Codebook lookup or escape (branch predicted 99.92% non-escape)
        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0])
            : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1])
            : bf16_to_float(escape_vals[esc_ptr++]);

        // FMA accumulate
        sum0 += w0 * bf16_to_float(activations[col]);
        sum1 += w1 * bf16_to_float(activations[col + WORKGROUP_SIZE]);
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    // Remainder (at most 1 element per thread)
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx])
            : bf16_to_float(escape_vals[esc_ptr++]);
        sum0 += w * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;

    // Wavefront reduction via shuffle
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);

    // Cross-warp reduction via LDS (4 wavefronts → 1 result)
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();

    if (tid == 0)
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Batch matvec: Y[M×B] = decode(packed[M×K]) × X[K×B]
// ---------------------------------------------------------------------------
// Decode each weight ONCE, multiply by B activation vectors.
// Amortizes decode cost (read12 + LDS lookup) across batch dimension.
// B=2: ~1.5x decode amortization (2 FMAs per decode vs 1)
// B=4: ~2x decode amortization (4 FMAs per decode vs 1)
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch2(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ act0,          // activation vector 0 [K]
    const int16_t*  __restrict__ act1,          // activation vector 1 [K]
    const int32_t*  __restrict__ escape_offsets,
    const int16_t*  __restrict__ escape_vals,
    float*          __restrict__ out0,           // output vector 0 [M]
    float*          __restrict__ out1,           // output vector 1 [M]
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[8];  // 4 per output vector

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    __syncthreads();

    float s0a = 0.0f, s0b = 0.0f;  // sums for output 0 (2x unroll)
    float s1a = 0.0f, s1b = 0.0f;  // sums for output 1 (2x unroll)
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));

        // Decode once, use twice
        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]);

        float a0_0 = bf16_to_float(act0[col]);
        float a0_1 = bf16_to_float(act0[col + WORKGROUP_SIZE]);
        float a1_0 = bf16_to_float(act1[col]);
        float a1_1 = bf16_to_float(act1[col + WORKGROUP_SIZE]);

        s0a += w0 * a0_0;  s0b += w1 * a0_1;
        s1a += w0 * a1_0;  s1b += w1 * a1_1;
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += w * bf16_to_float(act0[col]);
        s1a += w * bf16_to_float(act1[col]);
    }
    float sum0 = s0a + s0b;
    float sum1 = s1a + s1b;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[tid >> 6] = sum0;
        warp_sums[(tid >> 6) + 4] = sum1;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        out1[row] = warp_sums[4] + warp_sums[5] + warp_sums[6] + warp_sums[7];
    }
}

// Batch4: decode once, multiply by 4 activation vectors
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch4(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t*  __restrict__ escape_offsets,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[16];  // 4 per output vector

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    __syncthreads();

    // Use same 2x unroll pattern as single/batch2 for bit-exact accumulation
    float s0a = 0.0f, s0b = 0.0f;
    float s1a = 0.0f, s1b = 0.0f;
    float s2a = 0.0f, s2b = 0.0f;
    float s3a = 0.0f, s3b = 0.0f;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));

        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]);

        s0a += w0 * bf16_to_float(act0[col]);  s0b += w1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1a += w0 * bf16_to_float(act1[col]);  s1b += w1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2a += w0 * bf16_to_float(act2[col]);  s2b += w1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3a += w0 * bf16_to_float(act3[col]);  s3b += w1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += w * bf16_to_float(act0[col]);
        s1a += w * bf16_to_float(act1[col]);
        s2a += w * bf16_to_float(act2[col]);
        s3a += w * bf16_to_float(act3[col]);
    }
    float s0 = s0a + s0b, s1 = s1a + s1b, s2 = s2a + s2b, s3 = s3a + s3b;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
    }
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[wid]      = s0;
        warp_sums[wid + 4]  = s1;
        warp_sums[wid + 8]  = s2;
        warp_sums[wid + 12] = s3;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        out1[row] = warp_sums[4] + warp_sums[5] + warp_sums[6] + warp_sums[7];
        out2[row] = warp_sums[8] + warp_sums[9] + warp_sums[10] + warp_sums[11];
        out3[row] = warp_sums[12] + warp_sums[13] + warp_sums[14] + warp_sums[15];
    }
}

// ---------------------------------------------------------------------------
// Decompress-only: decode all weights to BF16 buffer for bit-exact verification
// ---------------------------------------------------------------------------
// Same decode logic as matvec but writes raw BF16 values instead of dot product.
// Used to verify every decoded weight matches the original BF16 bit pattern.
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_decompress_fused(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int32_t*  __restrict__ escape_offsets,
    const int16_t*  __restrict__ escape_vals,
    int16_t*        __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    __shared__ int16_t cb[4096];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    __syncthreads();

    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int base = row * K;

    for (int col = tid; col < K; col += WORKGROUP_SIZE) {
        uint32_t idx = read12(packed, bp);
        output[base + col] = (__builtin_expect(idx != 4095, 1))
            ? cb[idx]
            : escape_vals[esc_ptr++];
        bp += (int64_t)(WORKGROUP_SIZE * 12);
    }
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

int launch_fixed12_batch2(
    const void* packed, const void* codebook,
    const void* act0, const void* act1,
    const void* escape_offsets, const void* escape_vals,
    void* out0, void* out1, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_batch2,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)act0, (const int16_t*)act1,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (float*)out0, (float*)out1, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

int launch_fixed12_batch4(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* esc_off, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_batch4,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int32_t*)esc_off, (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

int launch_fixed12_decompress(
    const void* packed, const void* codebook,
    const void* escape_offsets, const void* escape_vals,
    void* output, int M, int K)
{
    hipLaunchKernelGGL(fixed12_decompress_fused,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (int16_t*)output, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

}
