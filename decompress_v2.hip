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
// Macro for the batch4 FMA loop body (shared between fast and scan kernels)
#define BATCH4_FMA_BODY() \
    int col = tid; \
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) { \
        uint32_t idx0 = read12(packed, bp); \
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12)); \
        float w0 = (__builtin_expect(idx0 != 4095, 1)) \
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]); \
        float w1 = (__builtin_expect(idx1 != 4095, 1)) \
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]); \
        s0a += w0 * bf16_to_float(act0[col]);  s0b += w1 * bf16_to_float(act0[col + WORKGROUP_SIZE]); \
        s1a += w0 * bf16_to_float(act1[col]);  s1b += w1 * bf16_to_float(act1[col + WORKGROUP_SIZE]); \
        s2a += w0 * bf16_to_float(act2[col]);  s2b += w1 * bf16_to_float(act2[col + WORKGROUP_SIZE]); \
        s3a += w0 * bf16_to_float(act3[col]);  s3b += w1 * bf16_to_float(act3[col + WORKGROUP_SIZE]); \
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12); \
    } \
    if (col < K) { \
        uint32_t idx = read12(packed, bp); \
        float w = (__builtin_expect(idx != 4095, 1)) \
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]); \
        s0a += w * bf16_to_float(act0[col]); \
        s1a += w * bf16_to_float(act1[col]); \
        s2a += w * bf16_to_float(act2[col]); \
        s3a += w * bf16_to_float(act3[col]); \
    }

#define BATCH4_REDUCE_BODY() \
    float s0 = s0a + s0b, s1 = s1a + s1b, s2 = s2a + s2b, s3 = s3a + s3b; \
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) { \
        s0 += __shfl_down(s0, off, WARP_SIZE); \
        s1 += __shfl_down(s1, off, WARP_SIZE); \
        s2 += __shfl_down(s2, off, WARP_SIZE); \
        s3 += __shfl_down(s3, off, WARP_SIZE); \
    } \
    int wid = tid >> 6; \
    if ((tid & (WARP_SIZE - 1)) == 0) { \
        warp_sums[wid]      = s0; \
        warp_sums[wid + 4]  = s1; \
        warp_sums[wid + 8]  = s2; \
        warp_sums[wid + 12] = s3; \
    } \
    __syncthreads(); \
    if (tid == 0) { \
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3]; \
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7]; \
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11]; \
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15]; \
    }

// Fast batch4: LDS codebook (8KB, 1-cycle access)
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch4(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[16];
    __shared__ int warp_esc_totals[4];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];

    // Escape count prefix sum (same as before)
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;
    BATCH4_FMA_BODY()
    BATCH4_REDUCE_BODY()
}

// Forward declaration for structured12 decode (defined later in file)
__device__ __forceinline__ float structured12_decode(uint32_t raw12, int base_exp);

// Scan batch4: on-the-fly escape scan using structured12 format (group==0 = escape)
// No LDS codebook, no escape_counts table. Saves max VRAM.
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_batch4_scan(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t*  __restrict__ escape_row_base,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[16];
    __shared__ int esc_scan[WORKGROUP_SIZE];

    // Phase 1: count escapes (group==0 in structured12 format)
    // 32-bit bp to avoid int64 overhead in hot loop
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp_start = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int stride = WORKGROUP_SIZE * 12;
    int my_esc = 0;
    { int bp_s = bp_start; int c = tid;
      for (; c + WORKGROUP_SIZE < K; c += WORKGROUP_SIZE * 2) {
          int w0 = bp_s >> 5; int s0 = bp_s & 31;
          uint32_t r0 = (uint32_t)(((uint64_t)row_packed[w0] | ((uint64_t)row_packed[w0+1] << 32)) >> s0) & 0xFFF;
          int w1 = (bp_s+stride) >> 5; int s1 = (bp_s+stride) & 31;
          uint32_t r1 = (uint32_t)(((uint64_t)row_packed[w1] | ((uint64_t)row_packed[w1+1] << 32)) >> s1) & 0xFFF;
          if ((r0 >> 8) == 0) my_esc++;
          if ((r1 >> 8) == 0) my_esc++;
          bp_s += stride * 2;
      }
      if (c < K) { int w0 = bp_s >> 5; int s0 = bp_s & 31;
          uint32_t r0 = (uint32_t)(((uint64_t)row_packed[w0] | ((uint64_t)row_packed[w0+1] << 32)) >> s0) & 0xFFF;
          if ((r0 >> 8) == 0) my_esc++; }
    }
    esc_scan[tid] = my_esc;
    __syncthreads();
    if (tid == 0) { int s=0; for (int i=0;i<WORKGROUP_SIZE;i++) { int c=esc_scan[i]; esc_scan[i]=s; s+=c; } }
    __syncthreads();
    int esc_ptr = escape_row_base[row] + esc_scan[tid];

    // Phase 2: FMA with structured12 decode — 32-bit bp
    int bp = bp_start;
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        int w0i = bp >> 5; int s0i = bp & 31;
        int w1i = (bp+stride) >> 5; int s1i = (bp+stride) & 31;
        uint32_t raw0 = (uint32_t)(((uint64_t)row_packed[w0i] | ((uint64_t)row_packed[w0i+1] << 32)) >> s0i) & 0xFFF;
        uint32_t raw1 = (uint32_t)(((uint64_t)row_packed[w1i] | ((uint64_t)row_packed[w1i+1] << 32)) >> s1i) & 0xFFF;
        uint32_t g0 = raw0 >> 8, g1 = raw1 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1))
            ? structured12_decode(raw0, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1))
            ? structured12_decode(raw1, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += wt0 * bf16_to_float(act0[col]);  s0b += wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1a += wt0 * bf16_to_float(act1[col]);  s1b += wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2a += wt0 * bf16_to_float(act2[col]);  s2b += wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3a += wt0 * bf16_to_float(act3[col]);  s3b += wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        bp += stride * 2;
    }
    if (col < K) {
        int wi = bp >> 5; int si = bp & 31;
        uint32_t raw = (uint32_t)(((uint64_t)row_packed[wi] | ((uint64_t)row_packed[wi+1] << 32)) >> si) & 0xFFF;
        uint32_t g = raw >> 8;
        float wt = (__builtin_expect(g != 0, 1))
            ? structured12_decode(raw, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += wt * bf16_to_float(act0[col]);
        s1a += wt * bf16_to_float(act1[col]);
        s2a += wt * bf16_to_float(act2[col]);
        s3a += wt * bf16_to_float(act3[col]);
    }

    BATCH4_REDUCE_BODY()
}

// Batch8: decode once, multiply by 8 activation vectors
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch8(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t* const* __restrict__ activations,  // array of 8 activation pointers
    const int32_t*  __restrict__ escape_offsets,
    const int16_t*  __restrict__ escape_vals,
    float* const*   __restrict__ outputs,             // array of 8 output pointers
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];

    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[32];  // 4 per output vector × 8

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE)
        cb[i] = codebook[i];
    __syncthreads();

    // Load activation and output pointers from global arrays
    const int16_t* act[8];
    float* out[8];
    for (int b = 0; b < 8; b++) {
        act[b] = activations[b];
        out[b] = outputs[b];
    }

    float sa[8] = {0}, sb[8] = {0};  // 2x unroll accumulators per batch
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));

        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]);

        for (int b = 0; b < 8; b++) {
            sa[b] += w0 * bf16_to_float(act[b][col]);
            sb[b] += w1 * bf16_to_float(act[b][col + WORKGROUP_SIZE]);
        }
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]);
        for (int b = 0; b < 8; b++)
            sa[b] += w * bf16_to_float(act[b][col]);
    }

    // Reduce all 8 sums
    for (int b = 0; b < 8; b++) {
        float s = sa[b] + sb[b];
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            s += __shfl_down(s, off, WARP_SIZE);
        if ((tid & (WARP_SIZE - 1)) == 0)
            warp_sums[(tid >> 6) + b * 4] = s;
    }
    __syncthreads();
    if (tid < 8)
        out[tid][row] = warp_sums[tid * 4] + warp_sums[tid * 4 + 1] + warp_sums[tid * 4 + 2] + warp_sums[tid * 4 + 3];
}

// Batch8 FMA body macro: decode once, 8 FMAs per weight (explicit pointers)
#define BATCH8_FMA_BODY() \
    int col = tid; \
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) { \
        uint32_t idx0 = read12(packed, bp); \
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12)); \
        float w0 = (__builtin_expect(idx0 != 4095, 1)) \
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]); \
        float w1 = (__builtin_expect(idx1 != 4095, 1)) \
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]); \
        s0a += w0 * bf16_to_float(act0[col]);  s0b += w1 * bf16_to_float(act0[col + WORKGROUP_SIZE]); \
        s1a += w0 * bf16_to_float(act1[col]);  s1b += w1 * bf16_to_float(act1[col + WORKGROUP_SIZE]); \
        s2a += w0 * bf16_to_float(act2[col]);  s2b += w1 * bf16_to_float(act2[col + WORKGROUP_SIZE]); \
        s3a += w0 * bf16_to_float(act3[col]);  s3b += w1 * bf16_to_float(act3[col + WORKGROUP_SIZE]); \
        s4a += w0 * bf16_to_float(act4[col]);  s4b += w1 * bf16_to_float(act4[col + WORKGROUP_SIZE]); \
        s5a += w0 * bf16_to_float(act5[col]);  s5b += w1 * bf16_to_float(act5[col + WORKGROUP_SIZE]); \
        s6a += w0 * bf16_to_float(act6[col]);  s6b += w1 * bf16_to_float(act6[col + WORKGROUP_SIZE]); \
        s7a += w0 * bf16_to_float(act7[col]);  s7b += w1 * bf16_to_float(act7[col + WORKGROUP_SIZE]); \
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12); \
    } \
    if (col < K) { \
        uint32_t idx = read12(packed, bp); \
        float w = (__builtin_expect(idx != 4095, 1)) \
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]); \
        s0a += w * bf16_to_float(act0[col]); \
        s1a += w * bf16_to_float(act1[col]); \
        s2a += w * bf16_to_float(act2[col]); \
        s3a += w * bf16_to_float(act3[col]); \
        s4a += w * bf16_to_float(act4[col]); \
        s5a += w * bf16_to_float(act5[col]); \
        s6a += w * bf16_to_float(act6[col]); \
        s7a += w * bf16_to_float(act7[col]); \
    }

#define BATCH8_REDUCE_BODY() \
    float s0 = s0a + s0b, s1 = s1a + s1b, s2 = s2a + s2b, s3 = s3a + s3b; \
    float s4 = s4a + s4b, s5 = s5a + s5b, s6 = s6a + s6b, s7 = s7a + s7b; \
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) { \
        s0 += __shfl_down(s0, off, WARP_SIZE); \
        s1 += __shfl_down(s1, off, WARP_SIZE); \
        s2 += __shfl_down(s2, off, WARP_SIZE); \
        s3 += __shfl_down(s3, off, WARP_SIZE); \
        s4 += __shfl_down(s4, off, WARP_SIZE); \
        s5 += __shfl_down(s5, off, WARP_SIZE); \
        s6 += __shfl_down(s6, off, WARP_SIZE); \
        s7 += __shfl_down(s7, off, WARP_SIZE); \
    } \
    int wid = tid >> 6; \
    if ((tid & (WARP_SIZE - 1)) == 0) { \
        warp_sums[wid]       = s0; \
        warp_sums[wid + 4]   = s1; \
        warp_sums[wid + 8]   = s2; \
        warp_sums[wid + 12]  = s3; \
        warp_sums[wid + 16]  = s4; \
        warp_sums[wid + 20]  = s5; \
        warp_sums[wid + 24]  = s6; \
        warp_sums[wid + 28]  = s7; \
    } \
    __syncthreads(); \
    if (tid == 0) { \
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3]; \
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7]; \
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11]; \
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15]; \
        out4[row] = warp_sums[16]+warp_sums[17]+warp_sums[18]+warp_sums[19]; \
        out5[row] = warp_sums[20]+warp_sums[21]+warp_sums[22]+warp_sums[23]; \
        out6[row] = warp_sums[24]+warp_sums[25]+warp_sums[26]+warp_sums[27]; \
        out7[row] = warp_sums[28]+warp_sums[29]+warp_sums[30]+warp_sums[31]; \
    }

// Fast batch8: LDS codebook (8KB, 1-cycle access), explicit pointers, escape_counts prefix sum
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch8_v2(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int16_t*  __restrict__ act4, const int16_t* __restrict__ act5,
    const int16_t*  __restrict__ act6, const int16_t* __restrict__ act7,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    float* __restrict__ out4, float* __restrict__ out5,
    float* __restrict__ out6, float* __restrict__ out7,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[32];  // 4 warps × 8 outputs
    __shared__ int warp_esc_totals[4];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];

    // Escape count prefix sum (same as batch4)
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;
    float s4a=0,s4b=0,s5a=0,s5b=0,s6a=0,s6b=0,s7a=0,s7b=0;
    BATCH8_FMA_BODY()
    BATCH8_REDUCE_BODY()
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
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_batch4,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

int launch_fixed12_batch8(
    const void* packed, const void* codebook,
    const void* act_ptrs, const void* esc_off, const void* esc_vals,
    const void* out_ptrs, int M, int K)
{
    hipLaunchKernelGGL(fixed12_matvec_batch8,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t* const*)act_ptrs,
        (const int32_t*)esc_off, (const int16_t*)esc_vals,
        (float* const*)out_ptrs, M, K);
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

// Async versions — no hipDeviceSynchronize, use stream ordering
extern "C" {
int launch_fixed12_fused_async(
    const void* packed, const void* codebook, const void* activations,
    const void* escape_offsets, const void* escape_vals,
    void* output, int M, int K, void* stream)
{
    hipLaunchKernelGGL(fixed12_matvec_fused,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (float*)output, M, K);
    return 0;
}

int launch_fixed12_batch4_async(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream)
{
    if (esc_counts) {
        // Fast path: uint8 count table + shared-memory prefix sum
        hipLaunchKernelGGL(fixed12_matvec_batch4,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint32_t*)packed, (const int16_t*)codebook,
            (const int16_t*)a0, (const int16_t*)a1,
            (const int16_t*)a2, (const int16_t*)a3,
            (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
            (const int16_t*)esc_vals,
            (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    } else {
        // VRAM-saving path: use structured12 scan kernel (legacy launcher)
        hipLaunchKernelGGL(structured12_matvec_batch4_scan,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint32_t*)packed, 0,
            (const int16_t*)a0, (const int16_t*)a1,
            (const int16_t*)a2, (const int16_t*)a3,
            (const int32_t*)esc_row_base,
            (const int16_t*)esc_vals,
            (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    }
    return 0;
}

int launch_fixed12_batch8_v2_async(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* a4, const void* a5, const void* a6, const void* a7,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3,
    void* o4, void* o5, void* o6, void* o7,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(fixed12_matvec_batch8_v2,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int16_t*)a4, (const int16_t*)a5,
        (const int16_t*)a6, (const int16_t*)a7,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3,
        (float*)o4, (float*)o5, (float*)o6, (float*)o7,
        M, K);
    return 0;
}
}

// ---------------------------------------------------------------------------
// Two-pass kernels: matvec (no escape) + wavefront-parallel patch correction
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_v2(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[4];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();
    float sum0 = 0.0f, sum1 = 0.0f;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        sum0 += bf16_to_float(cb[idx0]) * bf16_to_float(activations[col]);
        sum1 += bf16_to_float(cb[idx1]) * bf16_to_float(activations[col + WORKGROUP_SIZE]);
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        sum0 += bf16_to_float(cb[idx]) * bf16_to_float(activations[col]);
    }
    float sum = sum0 + sum1;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

__global__ void apply_patches_v2(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const int16_t* __restrict__ activations,
    float*         __restrict__ output,
    int M)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row];
    int end   = row_offsets[row + 1];
    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(activations[patch_cols[p]]);
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) output[row] += correction;
}

extern "C" {
int launch_fixed12_v2(const void* packed, const void* codebook,
    const void* activations, void* output, int M, int K) {
    hipLaunchKernelGGL(fixed12_matvec_v2, dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations, (float*)output, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}
int launch_patches_v2(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M) {
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches_v2, dim3(M), dim3(WARP_SIZE), 0, 0,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const int16_t*)activations, (float*)output, M);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}
}

// Batch4 matvec WITHOUT escape handling (two-pass: patch correction after)
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch4_noesc(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[16];

    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();

    float s0a=0, s0b=0, s1a=0, s1b=0, s2a=0, s2b=0, s3a=0, s3b=0;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        float w0 = bf16_to_float(cb[idx0]);
        float w1 = bf16_to_float(cb[idx1]);
        s0a += w0 * bf16_to_float(act0[col]); s0b += w1 * bf16_to_float(act0[col+WORKGROUP_SIZE]);
        s1a += w0 * bf16_to_float(act1[col]); s1b += w1 * bf16_to_float(act1[col+WORKGROUP_SIZE]);
        s2a += w0 * bf16_to_float(act2[col]); s2b += w1 * bf16_to_float(act2[col+WORKGROUP_SIZE]);
        s3a += w0 * bf16_to_float(act3[col]); s3b += w1 * bf16_to_float(act3[col+WORKGROUP_SIZE]);
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = bf16_to_float(cb[idx]);
        s0a += w * bf16_to_float(act0[col]);
        s1a += w * bf16_to_float(act1[col]);
        s2a += w * bf16_to_float(act2[col]);
        s3a += w * bf16_to_float(act3[col]);
    }
    float s0=s0a+s0b, s1=s1a+s1b, s2=s2a+s2b, s3=s3a+s3b;

    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
    }
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]    = s0; warp_sums[wid+4]  = s1;
        warp_sums[wid+8]  = s2; warp_sums[wid+12] = s3;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
    }
}

extern "C" {
int launch_fixed12_batch4_noesc(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    void* o0, void* o1, void* o2, void* o3, int M, int K) {
    hipLaunchKernelGGL(fixed12_matvec_batch4_noesc,
        dim3(M), dim3(WORKGROUP_SIZE), 0, 0,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)a0, (const int16_t*)a1, (const int16_t*)a2, (const int16_t*)a3,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}
}

extern "C" {
int launch_fixed12_batch4_noesc_async(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_batch4_noesc,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)a0, (const int16_t*)a1, (const int16_t*)a2, (const int16_t*)a3,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return 0;
}
}

// Async two-pass v2 (no sync)
extern "C" {
int launch_fixed12_v2_async(const void* packed, const void* codebook,
    const void* activations, void* output, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_v2, dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const int16_t*)activations, (float*)output, M, K);
    return 0;
}
int launch_patches_v2_async(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M, void* stream) {
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches_v2, dim3(M), dim3(WARP_SIZE), 0, (hipStream_t)stream,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const int16_t*)activations, (float*)output, M);
    return 0;
}

// FP32-input patch correction (avoids separate FP32→BF16 conversion kernel)
__global__ void apply_patches_v2_f32(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const float*   __restrict__ activations,  // FP32 input
    float*         __restrict__ output,
    int M)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row];
    int end   = row_offsets[row + 1];
    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * activations[patch_cols[p]];
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) output[row] += correction;
}

int launch_patches_v2_f32_async(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M, void* stream) {
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches_v2_f32, dim3(M), dim3(WARP_SIZE), 0, (hipStream_t)stream,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const float*)activations, (float*)output, M);
    return 0;
}
}

// Batch4 FP32 patch correction: 4 sequences in one launch
// Grid: (M, 4), each block handles one row for one sequence
__global__ void apply_patches_v2_f32_batch4(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const float* __restrict__ act0, const float* __restrict__ act1,
    const float* __restrict__ act2, const float* __restrict__ act3,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M)
{
    int row = blockIdx.x;
    int seq = blockIdx.y;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row];
    int end   = row_offsets[row + 1];

    const float* act = (seq == 0) ? act0 : (seq == 1) ? act1 : (seq == 2) ? act2 : act3;
    float* out = (seq == 0) ? out0 : (seq == 1) ? out1 : (seq == 2) ? out2 : out3;

    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * act[patch_cols[p]];
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) out[row] += correction;
}

extern "C" {
int launch_patches_v2_f32_batch4_async(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* a0, const void* a1, const void* a2, const void* a3,
    void* o0, void* o1, void* o2, void* o3,
    int M, void* stream) {
    if (M == 0) return 0;
    dim3 grid(M, 4);
    hipLaunchKernelGGL(apply_patches_v2_f32_batch4, grid, dim3(WARP_SIZE), 0, (hipStream_t)stream,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const float*)a0, (const float*)a1, (const float*)a2, (const float*)a3,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M);
    return 0;
}
}

// Batch4 that accepts FP32 activations directly (fused BF16 conversion)
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch4_f32(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const float* __restrict__ act0, const float* __restrict__ act1,
    const float* __restrict__ act2, const float* __restrict__ act3,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[16];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();

    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        float w0 = bf16_to_float(cb[idx0]);
        float w1 = bf16_to_float(cb[idx1]);
        // Read FP32 activations directly — no BF16 conversion needed
        s0a += w0*act0[col]; s0b += w1*act0[col+WORKGROUP_SIZE];
        s1a += w0*act1[col]; s1b += w1*act1[col+WORKGROUP_SIZE];
        s2a += w0*act2[col]; s2b += w1*act2[col+WORKGROUP_SIZE];
        s3a += w0*act3[col]; s3b += w1*act3[col+WORKGROUP_SIZE];
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = bf16_to_float(cb[idx]);
        s0a += w*act0[col]; s1a += w*act1[col]; s2a += w*act2[col]; s3a += w*act3[col];
    }
    float s0=s0a+s0b,s1=s1a+s1b,s2=s2a+s2b,s3=s3a+s3b;
    for (int off=WARP_SIZE/2;off>0;off>>=1) {
        s0+=__shfl_down(s0,off,WARP_SIZE); s1+=__shfl_down(s1,off,WARP_SIZE);
        s2+=__shfl_down(s2,off,WARP_SIZE); s3+=__shfl_down(s3,off,WARP_SIZE);
    }
    int wid=tid>>6;
    if((tid&(WARP_SIZE-1))==0) { warp_sums[wid]=s0;warp_sums[wid+4]=s1;warp_sums[wid+8]=s2;warp_sums[wid+12]=s3; }
    __syncthreads();
    if(tid==0) {
        out0[row]=warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row]=warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row]=warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row]=warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
    }
}

// B=1 that accepts FP32 directly
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_v2_f32(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const float*    __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[4];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();
    float sum0=0,sum1=0;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        sum0 += bf16_to_float(cb[idx0]) * activations[col];
        sum1 += bf16_to_float(cb[idx1]) * activations[col + WORKGROUP_SIZE];
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) { uint32_t idx = read12(packed, bp); sum0 += bf16_to_float(cb[idx]) * activations[col]; }
    float sum = sum0 + sum1;
    for (int off=WARP_SIZE/2;off>0;off>>=1) sum += __shfl_down(sum,off,WARP_SIZE);
    if((tid&(WARP_SIZE-1))==0) warp_sums[tid>>6] = sum;
    __syncthreads();
    if(tid==0) output[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
}

extern "C" {
int launch_fixed12_batch4_f32(const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_batch4_f32, dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const float*)a0, (const float*)a1, (const float*)a2, (const float*)a3,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return 0;
}
int launch_fixed12_v2_f32(const void* packed, const void* codebook,
    const void* activations, void* output, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_v2_f32, dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const float*)activations, (float*)output, M, K);
    return 0;
}
}

// ===========================================================================
// FP32-input fused kernel: single-pass with O(1) escape, FP32 activations
// ===========================================================================
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_fused_f32(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const float*    __restrict__ activations,
    const int32_t*  __restrict__ escape_offsets,
    const int16_t*  __restrict__ escape_vals,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    int esc_ptr = escape_offsets[row * WORKGROUP_SIZE + tid];
    __shared__ int16_t cb[4096];
    __shared__ float warp_sums[4];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();
    float sum0 = 0.0f, sum1 = 0.0f;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]);
        sum0 += w0 * activations[col];
        sum1 += w1 * activations[col + WORKGROUP_SIZE];
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]);
        sum0 += w * activations[col];
    }
    float sum = sum0 + sum1;
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
}

// FP32-input fused batch4
__launch_bounds__(WORKGROUP_SIZE)
__global__ void fixed12_matvec_batch4_fused_f32(
    const uint32_t* __restrict__ packed,
    const int16_t*  __restrict__ codebook,
    const float* __restrict__ act0, const float* __restrict__ act1,
    const float* __restrict__ act2, const float* __restrict__ act3,
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
    __shared__ float warp_sums[16];
    for (int i = tid; i < 4096; i += WORKGROUP_SIZE) cb[i] = codebook[i];
    __syncthreads();
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;
    int64_t bp = (int64_t)row * K * 12 + (int64_t)tid * 12;
    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t idx0 = read12(packed, bp);
        uint32_t idx1 = read12(packed, bp + (int64_t)(WORKGROUP_SIZE * 12));
        float w0 = (__builtin_expect(idx0 != 4095, 1))
            ? bf16_to_float(cb[idx0]) : bf16_to_float(escape_vals[esc_ptr++]);
        float w1 = (__builtin_expect(idx1 != 4095, 1))
            ? bf16_to_float(cb[idx1]) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += w0*act0[col]; s0b += w1*act0[col+WORKGROUP_SIZE];
        s1a += w0*act1[col]; s1b += w1*act1[col+WORKGROUP_SIZE];
        s2a += w0*act2[col]; s2b += w1*act2[col+WORKGROUP_SIZE];
        s3a += w0*act3[col]; s3b += w1*act3[col+WORKGROUP_SIZE];
        bp += (int64_t)(WORKGROUP_SIZE * 2 * 12);
    }
    if (col < K) {
        uint32_t idx = read12(packed, bp);
        float w = (__builtin_expect(idx != 4095, 1))
            ? bf16_to_float(cb[idx]) : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += w*act0[col]; s1a += w*act1[col]; s2a += w*act2[col]; s3a += w*act3[col];
    }
    float s0=s0a+s0b, s1=s1a+s1b, s2=s2a+s2b, s3=s3a+s3b;
    for (int off=WARP_SIZE/2;off>0;off>>=1) {
        s0+=__shfl_down(s0,off,WARP_SIZE); s1+=__shfl_down(s1,off,WARP_SIZE);
        s2+=__shfl_down(s2,off,WARP_SIZE); s3+=__shfl_down(s3,off,WARP_SIZE);
    }
    int wid=tid>>6;
    if((tid&(WARP_SIZE-1))==0) { warp_sums[wid]=s0;warp_sums[wid+4]=s1;warp_sums[wid+8]=s2;warp_sums[wid+12]=s3; }
    __syncthreads();
    if(tid==0) {
        out0[row]=warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row]=warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row]=warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row]=warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
    }
}

extern "C" {
int launch_fixed12_fused_f32_async(
    const void* packed, const void* codebook, const void* activations,
    const void* escape_offsets, const void* escape_vals,
    void* output, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_fused_f32,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const float*)activations,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (float*)output, M, K);
    return 0;
}
int launch_fixed12_batch4_fused_f32_async(
    const void* packed, const void* codebook,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* escape_offsets, const void* escape_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream) {
    hipLaunchKernelGGL(fixed12_matvec_batch4_fused_f32,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, (const int16_t*)codebook,
        (const float*)a0, (const float*)a1, (const float*)a2, (const float*)a3,
        (const int32_t*)escape_offsets, (const int16_t*)escape_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return 0;
}
}

// ===========================================================================
// Structured 12-bit Matvec: NO LDS codebook, decode by arithmetic
// Format: [4-bit exp_group][1-bit sign][7-bit mantissa] = 12 bits
// Decode: exponent = BaseExp + group (1 ADD), then reconstruct BF16
// 0 KB LDS for codebook → 2x occupancy → 2x loads in flight
// ===========================================================================

__device__ __forceinline__
float structured12_decode(uint32_t raw12, int base_exp) {
    uint32_t group    = raw12 >> 8;           // top 4 bits
    uint32_t sign     = (raw12 >> 7) & 1;     // bit 7
    uint32_t mantissa = raw12 & 0x7F;          // bottom 7 bits
    uint32_t exponent = base_exp + group;      // ONE add
    uint32_t bf16_bits = (sign << 15) | (exponent << 7) | mantissa;
    union { uint32_t u; float f; } c;
    c.u = bf16_bits << 16;
    return c.f;
}

__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_batch4(
    const uint32_t* __restrict__ packed,
    int base_exp,                              // BaseExp (e.g., 106)
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // NO LDS codebook — only warp_sums (64 bytes) + esc_totals (16 bytes)
    __shared__ float warp_sums[16];
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles (same as before)
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Main FMA loop — pointer-based addressing with constant offsets
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;  // constant — stride is multiple of 32
    const char* base_ptr = (const char*)(row_packed + (bp >> 5));
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t raw0 = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t raw1 = (uint32_t)(*(const uint64_t*)(base_ptr + 384) >> shift) & 0xFFF;

        uint32_t g0 = raw0 >> 8, g1 = raw1 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1))
            ? structured12_decode(raw0, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1))
            ? structured12_decode(raw1, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);

        s0a += wt0 * bf16_to_float(act0[col]);  s0b += wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1a += wt0 * bf16_to_float(act1[col]);  s1b += wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2a += wt0 * bf16_to_float(act2[col]);  s2b += wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3a += wt0 * bf16_to_float(act3[col]);  s3b += wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        base_ptr += 768;
    }
    if (col < K) {
        uint32_t raw = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t g = raw >> 8;
        float wt = (__builtin_expect(g != 0, 1))
            ? structured12_decode(raw, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += wt * bf16_to_float(act0[col]);
        s1a += wt * bf16_to_float(act1[col]);
        s2a += wt * bf16_to_float(act2[col]);
        s3a += wt * bf16_to_float(act3[col]);
    }

    // Warp reduction + output (same as codebook version)
    float s0=s0a+s0b, s1=s1a+s1b, s2=s2a+s2b, s3=s3a+s3b;
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
    }
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]=s0; warp_sums[wid+4]=s1;
        warp_sums[wid+8]=s2; warp_sums[wid+12]=s3;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row]=warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row]=warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row]=warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row]=warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 two-pass matvec with FP32 activations (no bf16→fp32 needed)
// Used for wo matvec where attention output is already FP32
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2_fp32act(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const float*    __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[4];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;
    const char* pack_ptr = (const char*)(row_packed + (bp >> 5));
    const char* act_ptr = (const char*)(activations + tid);
    // Activations: 1024 bytes between elements (256 float), 4096 per iteration
    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        uint32_t d0 = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        uint32_t d1 = (uint32_t)(*(const uint64_t*)(pack_ptr + 384) >> shift) & 0xFFF;
        uint32_t d2 = (uint32_t)(*(const uint64_t*)(pack_ptr + 768) >> shift) & 0xFFF;
        uint32_t d3 = (uint32_t)(*(const uint64_t*)(pack_ptr + 1152) >> shift) & 0xFFF;
        sum0 += structured12_decode(d0, base_exp) * *(const float*)(act_ptr);
        sum1 += structured12_decode(d1, base_exp) * *(const float*)(act_ptr + 1024);
        sum2 += structured12_decode(d2, base_exp) * *(const float*)(act_ptr + 2048);
        sum3 += structured12_decode(d3, base_exp) * *(const float*)(act_ptr + 3072);
        pack_ptr += 1536;
        act_ptr += 4096;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        uint32_t d = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        sum0 += structured12_decode(d, base_exp) * *(const float*)(act_ptr);
        pack_ptr += 384;
        act_ptr += 1024;
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 blocked-access matvec — each thread reads consecutive elements
// Eliminates read amplification: loads are contiguous across threads (0% waste)
// Current strided approach: ~1.33x read amplification (128 byte cache line / 96 useful)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2_blocked(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[4];

    // Each thread processes K/256 consecutive elements
    const int epe = K / WORKGROUP_SIZE;  // elements per thread (16 for K=4096)
    const int start_elem = tid * epe;

    // Base bit position for this thread's block of elements
    int64_t row_bits = (int64_t)row * K * 12;
    int64_t my_bits = row_bits + (int64_t)start_elem * 12;
    const uint32_t* my_packed = packed + (int)(my_bits / 32);
    int bit_off = (int)(my_bits % 32);

    // Pre-load all packed words for this thread's block
    // epe elements = epe*12 bits. With bit_off, need (bit_off + epe*12 + 31)/32 words
    // For epe=16: 192 bits + up to 31 = 223 bits → 7 words max
    // For epe=56: 672 bits + 31 = 703 bits → 22 words max
    // Use loop for generality

    float sum = 0.0f;
    int bp = bit_off;
    for (int e = 0; e < epe; e++) {
        int w = bp >> 5;
        int s = bp & 31;
        uint64_t combined = (uint64_t)my_packed[w] | ((uint64_t)my_packed[w+1] << 32);
        uint32_t raw = (uint32_t)(combined >> s) & 0xFFF;
        sum += structured12_decode(raw, base_exp) * bf16_to_float(activations[start_elem + e]);
        bp += 12;
    }

    // Warp reduction
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 two-pass matvec (no escape handling — patches applied separately)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[4];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;
    const char* pack_ptr = (const char*)(row_packed + (bp >> 5));
    const char* act_ptr = (const char*)(activations + tid);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        uint32_t d0 = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        uint32_t d1 = (uint32_t)(*(const uint64_t*)(pack_ptr + 384) >> shift) & 0xFFF;
        uint32_t d2 = (uint32_t)(*(const uint64_t*)(pack_ptr + 768) >> shift) & 0xFFF;
        uint32_t d3 = (uint32_t)(*(const uint64_t*)(pack_ptr + 1152) >> shift) & 0xFFF;
        sum0 += structured12_decode(d0, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += structured12_decode(d1, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += structured12_decode(d2, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += structured12_decode(d3, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1536));
        pack_ptr += 1536;
        act_ptr += 2048;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        uint32_t d = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        sum0 += structured12_decode(d, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        pack_ptr += 384;
        act_ptr += 512;
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 DUAL matvec: processes TWO weight tensors (gate+up)
// sharing the same activation reads. Saves 1 kernel launch per layer and
// shares L2 cache for activations.
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2_dual(
    const uint32_t* __restrict__ packed_a, int base_exp_a,  // first tensor (gate)
    const uint32_t* __restrict__ packed_b, int base_exp_b,  // second tensor (up)
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output_a,
    float*          __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[4];
    __shared__ float warp_sums_b[4];

    float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;

    // Both tensors have same dimensions, so same row layout
    int64_t row_bit_start = (int64_t)row * K * 12;
    int bp = (int)(row_bit_start % 32) + tid * 12;
    const int shift = bp & 31;

    // Pointer-based addressing for tensor A
    const char* pa_ptr = (const char*)((const uint32_t*)packed_a + (row_bit_start / 32) + (bp >> 5));
    // Pointer-based addressing for tensor B
    const char* pb_ptr = (const char*)((const uint32_t*)packed_b + (row_bit_start / 32) + (bp >> 5));
    // Activation pointer
    const char* act_ptr = (const char*)(activations + tid);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Read activations ONCE (shared between both tensors)
        float af0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float af1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float af2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float af3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        // Decode weights from tensor A (gate)
        uint32_t da0 = (uint32_t)(*(const uint64_t*)(pa_ptr) >> shift) & 0xFFF;
        uint32_t da1 = (uint32_t)(*(const uint64_t*)(pa_ptr + 384) >> shift) & 0xFFF;
        uint32_t da2 = (uint32_t)(*(const uint64_t*)(pa_ptr + 768) >> shift) & 0xFFF;
        uint32_t da3 = (uint32_t)(*(const uint64_t*)(pa_ptr + 1152) >> shift) & 0xFFF;
        sa0 += structured12_decode(da0, base_exp_a) * af0;
        sa1 += structured12_decode(da1, base_exp_a) * af1;
        sa2 += structured12_decode(da2, base_exp_a) * af2;
        sa3 += structured12_decode(da3, base_exp_a) * af3;

        // Decode weights from tensor B (up)
        uint32_t db0 = (uint32_t)(*(const uint64_t*)(pb_ptr) >> shift) & 0xFFF;
        uint32_t db1 = (uint32_t)(*(const uint64_t*)(pb_ptr + 384) >> shift) & 0xFFF;
        uint32_t db2 = (uint32_t)(*(const uint64_t*)(pb_ptr + 768) >> shift) & 0xFFF;
        uint32_t db3 = (uint32_t)(*(const uint64_t*)(pb_ptr + 1152) >> shift) & 0xFFF;
        sb0 += structured12_decode(db0, base_exp_b) * af0;
        sb1 += structured12_decode(db1, base_exp_b) * af1;
        sb2 += structured12_decode(db2, base_exp_b) * af2;
        sb3 += structured12_decode(db3, base_exp_b) * af3;

        pa_ptr += 1536;
        pb_ptr += 1536;
        act_ptr += 2048;
    }
    // Remainder
    for (; col < K; col += WORKGROUP_SIZE) {
        float af = bf16_to_float(*(const int16_t*)(act_ptr));
        uint32_t da = (uint32_t)(*(const uint64_t*)(pa_ptr) >> shift) & 0xFFF;
        uint32_t db = (uint32_t)(*(const uint64_t*)(pb_ptr) >> shift) & 0xFFF;
        sa0 += structured12_decode(da, base_exp_a) * af;
        sb0 += structured12_decode(db, base_exp_b) * af;
        pa_ptr += 384;
        pb_ptr += 384;
        act_ptr += 512;
    }

    // Reduction for tensor A
    float sum_a = sa0 + sa1 + sa2 + sa3;
    // Reduction for tensor B
    float sum_b = sb0 + sb1 + sb2 + sb3;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums_a[tid >> 6] = sum_a;
        warp_sums_b[tid >> 6] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        output_a[row] = warp_sums_a[0] + warp_sums_a[1] + warp_sums_a[2] + warp_sums_a[3];
        output_b[row] = warp_sums_b[0] + warp_sums_b[1] + warp_sums_b[2] + warp_sums_b[3];
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 matvec with buffer_load_dwordx2 (GFX9 SRD)
// Uses buffer resource descriptors for packed data + activation loads.
// Issue 4 packed loads + 4 activation loads, then single s_waitcnt.
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2_bufload(
    uint32_t psrd0, uint32_t psrd1, uint32_t psrd2, uint32_t psrd3,  // packed SRD
    uint32_t asrd0, uint32_t asrd1, uint32_t asrd2, uint32_t asrd3,  // activation SRD
    int base_exp,
    float*          __restrict__ output,
    int M, int K,
    const int32_t*  __restrict__ patch_row_offsets,
    const int32_t*  __restrict__ patch_cols,
    const int16_t*  __restrict__ patch_correct_vals,
    const int16_t*  __restrict__ patch_wrong_vals,
    const int16_t*  __restrict__ patch_activations)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[4];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // Compute initial byte offset into packed data for this thread
    int64_t row_bit_start = (int64_t)row * K * 12;
    int bp = (int)(row_bit_start % 32) + tid * 12;
    const int shift = bp & 31;
    // pack_off = byte offset of the dword containing our first bit
    uint32_t pack_off = (uint32_t)((row_bit_start / 32) * 4 + (bp >> 5) * 4);
    // act_off = byte offset for activation (int16_t) for this thread's column
    uint32_t act_off = (uint32_t)(tid * 2);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Issue all 4 packed loads + 4 activation loads, then wait once
        uint32_t plo0, phi0, plo1, phi1, plo2, phi2, plo3, phi3;
        uint32_t a0, a1, a2, a3;

        asm volatile(
            // Set up packed SRD in s[20:23]
            "s_mov_b32 s20, %[ps0]\n"
            "s_mov_b32 s21, %[ps1]\n"
            "s_mov_b32 s22, %[ps2]\n"
            "s_mov_b32 s23, %[ps3]\n"
            // Set up activation SRD in s[24:27]
            "s_mov_b32 s24, %[as0]\n"
            "s_mov_b32 s25, %[as1]\n"
            "s_mov_b32 s26, %[as2]\n"
            "s_mov_b32 s27, %[as3]\n"
            // Issue 4 packed loads (dwordx2 each)
            "buffer_load_dwordx2 v[0:1], %[po0], s[20:23], 0 offen\n"
            "buffer_load_dwordx2 v[2:3], %[po1], s[20:23], 0 offen\n"
            "buffer_load_dwordx2 v[4:5], %[po2], s[20:23], 0 offen\n"
            "buffer_load_dwordx2 v[6:7], %[po3], s[20:23], 0 offen\n"
            // Issue 4 activation loads (ushort each, zero-extended to 32-bit)
            "buffer_load_ushort v[8], %[ao0], s[24:27], 0 offen\n"
            "buffer_load_ushort v[9], %[ao1], s[24:27], 0 offen\n"
            "buffer_load_ushort v[10], %[ao2], s[24:27], 0 offen\n"
            "buffer_load_ushort v[11], %[ao3], s[24:27], 0 offen\n"
            // Wait for ALL loads
            "s_waitcnt vmcnt(0)\n"
            // Move results out
            "v_mov_b32 %[r0l], v0\n"
            "v_mov_b32 %[r0h], v1\n"
            "v_mov_b32 %[r1l], v2\n"
            "v_mov_b32 %[r1h], v3\n"
            "v_mov_b32 %[r2l], v4\n"
            "v_mov_b32 %[r2h], v5\n"
            "v_mov_b32 %[r3l], v6\n"
            "v_mov_b32 %[r3h], v7\n"
            "v_mov_b32 %[ra0], v8\n"
            "v_mov_b32 %[ra1], v9\n"
            "v_mov_b32 %[ra2], v10\n"
            "v_mov_b32 %[ra3], v11\n"
            : [r0l]"=v"(plo0), [r0h]"=v"(phi0),
              [r1l]"=v"(plo1), [r1h]"=v"(phi1),
              [r2l]"=v"(plo2), [r2h]"=v"(phi2),
              [r3l]"=v"(plo3), [r3h]"=v"(phi3),
              [ra0]"=v"(a0), [ra1]"=v"(a1), [ra2]"=v"(a2), [ra3]"=v"(a3)
            : [ps0]"s"(psrd0), [ps1]"s"(psrd1), [ps2]"s"(psrd2), [ps3]"s"(psrd3),
              [as0]"s"(asrd0), [as1]"s"(asrd1), [as2]"s"(asrd2), [as3]"s"(asrd3),
              [po0]"v"(pack_off), [po1]"v"(pack_off + 384u),
              [po2]"v"(pack_off + 768u), [po3]"v"(pack_off + 1152u),
              [ao0]"v"(act_off), [ao1]"v"(act_off + 512u),
              [ao2]"v"(act_off + 1024u), [ao3]"v"(act_off + 1536u)
            : "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11",
              "s20","s21","s22","s23","s24","s25","s26","s27","memory"
        );

        // Decode packed 12-bit values
        uint32_t d0 = (uint32_t)(((uint64_t)plo0 | ((uint64_t)phi0 << 32)) >> shift) & 0xFFF;
        uint32_t d1 = (uint32_t)(((uint64_t)plo1 | ((uint64_t)phi1 << 32)) >> shift) & 0xFFF;
        uint32_t d2 = (uint32_t)(((uint64_t)plo2 | ((uint64_t)phi2 << 32)) >> shift) & 0xFFF;
        uint32_t d3 = (uint32_t)(((uint64_t)plo3 | ((uint64_t)phi3 << 32)) >> shift) & 0xFFF;

        // Decode activations from raw uint16 to float
        float af0 = bf16_to_float((int16_t)a0);
        float af1 = bf16_to_float((int16_t)a1);
        float af2 = bf16_to_float((int16_t)a2);
        float af3 = bf16_to_float((int16_t)a3);

        sum0 += structured12_decode(d0, base_exp) * af0;
        sum1 += structured12_decode(d1, base_exp) * af1;
        sum2 += structured12_decode(d2, base_exp) * af2;
        sum3 += structured12_decode(d3, base_exp) * af3;

        pack_off += 1536u;
        act_off += 2048u;
    }
    // Remainder loop
    for (; col < K; col += WORKGROUP_SIZE) {
        uint32_t plo, phi;
        uint32_t aval;
        asm volatile(
            "s_mov_b32 s20, %[ps0]\n"
            "s_mov_b32 s21, %[ps1]\n"
            "s_mov_b32 s22, %[ps2]\n"
            "s_mov_b32 s23, %[ps3]\n"
            "s_mov_b32 s24, %[as0]\n"
            "s_mov_b32 s25, %[as1]\n"
            "s_mov_b32 s26, %[as2]\n"
            "s_mov_b32 s27, %[as3]\n"
            "buffer_load_dwordx2 v[0:1], %[po], s[20:23], 0 offen\n"
            "buffer_load_ushort v[2], %[ao], s[24:27], 0 offen\n"
            "s_waitcnt vmcnt(0)\n"
            "v_mov_b32 %[rl], v0\n"
            "v_mov_b32 %[rh], v1\n"
            "v_mov_b32 %[ra], v2\n"
            : [rl]"=v"(plo), [rh]"=v"(phi), [ra]"=v"(aval)
            : [ps0]"s"(psrd0), [ps1]"s"(psrd1), [ps2]"s"(psrd2), [ps3]"s"(psrd3),
              [as0]"s"(asrd0), [as1]"s"(asrd1), [as2]"s"(asrd2), [as3]"s"(asrd3),
              [po]"v"(pack_off), [ao]"v"(act_off)
            : "v0","v1","v2","s20","s21","s22","s23","s24","s25","s26","s27","memory"
        );
        uint32_t d = (uint32_t)(((uint64_t)plo | ((uint64_t)phi << 32)) >> shift) & 0xFFF;
        sum0 += structured12_decode(d, base_exp) * bf16_to_float((int16_t)aval);
        pack_off += 384u;
        act_off += 512u;
    }

    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) {
        float result = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        // Inline patch correction for escape values (group=0)
        if (patch_row_offsets != nullptr) {
            int start = patch_row_offsets[row];
            int end   = patch_row_offsets[row + 1];
            for (int p = start; p < end; p++) {
                int col = patch_cols[p];
                result += (bf16_to_float(patch_correct_vals[p]) - bf16_to_float(patch_wrong_vals[p]))
                          * bf16_to_float(patch_activations[col]);
            }
        }
        output[row] = result;
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 fused matvec (single-pass with escape counts)
// Replaces two-pass (v2 + patches) — saves 224 kernel launches per forward
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_fused_b1(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ activations,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float warp_sums[4];
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Main FMA loop — pointer-based addressing with constant offsets
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;
    const char* base_ptr = (const char*)(row_packed + (bp >> 5));
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        uint32_t d0 = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t d1 = (uint32_t)(*(const uint64_t*)(base_ptr + 384) >> shift) & 0xFFF;
        uint32_t d2 = (uint32_t)(*(const uint64_t*)(base_ptr + 768) >> shift) & 0xFFF;
        uint32_t d3 = (uint32_t)(*(const uint64_t*)(base_ptr + 1152) >> shift) & 0xFFF;

        uint32_t g0 = d0 >> 8, g1 = d1 >> 8, g2 = d2 >> 8, g3 = d3 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1)) ? structured12_decode(d0, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1)) ? structured12_decode(d1, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        float wt2 = (__builtin_expect(g2 != 0, 1)) ? structured12_decode(d2, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        float wt3 = (__builtin_expect(g3 != 0, 1)) ? structured12_decode(d3, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);

        sum0 += wt0 * bf16_to_float(activations[col]);
        sum1 += wt1 * bf16_to_float(activations[col + WORKGROUP_SIZE]);
        sum2 += wt2 * bf16_to_float(activations[col + WORKGROUP_SIZE * 2]);
        sum3 += wt3 * bf16_to_float(activations[col + WORKGROUP_SIZE * 3]);
        base_ptr += 1536;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        int wi = bp >> 5; int si = bp & 31;
        uint32_t d = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t g = d >> 8;
        float wt = (__builtin_expect(g != 0, 1)) ? structured12_decode(d, base_exp) : bf16_to_float(escape_vals[esc_ptr++]);
        sum0 += wt * bf16_to_float(activations[col]);
        base_ptr += 384;
    }

    // Warp reduction + output
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=8 fused matvec (single-pass with escape counts)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_batch8(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int16_t*  __restrict__ act4, const int16_t* __restrict__ act5,
    const int16_t*  __restrict__ act6, const int16_t* __restrict__ act7,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    float* __restrict__ out4, float* __restrict__ out5,
    float* __restrict__ out6, float* __restrict__ out7,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // NO LDS codebook — only warp_sums (128 bytes) + esc_totals (16 bytes)
    __shared__ float warp_sums[32];  // 4 warps x 8 outputs
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Main FMA loop — pointer-based addressing with constant offsets
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;  // constant — stride is multiple of 32
    const char* base_ptr = (const char*)(row_packed + (bp >> 5));
    // Single accumulators (saves 8 VGPRs vs split a/b — critical for B=8 register pressure)
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t raw0 = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t raw1 = (uint32_t)(*(const uint64_t*)(base_ptr + 384) >> shift) & 0xFFF;

        uint32_t g0 = raw0 >> 8, g1 = raw1 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1))
            ? structured12_decode(raw0, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1))
            ? structured12_decode(raw1, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);

        s0 += wt0 * bf16_to_float(act0[col]) + wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1 += wt0 * bf16_to_float(act1[col]) + wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2 += wt0 * bf16_to_float(act2[col]) + wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3 += wt0 * bf16_to_float(act3[col]) + wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        s4 += wt0 * bf16_to_float(act4[col]) + wt1 * bf16_to_float(act4[col + WORKGROUP_SIZE]);
        s5 += wt0 * bf16_to_float(act5[col]) + wt1 * bf16_to_float(act5[col + WORKGROUP_SIZE]);
        s6 += wt0 * bf16_to_float(act6[col]) + wt1 * bf16_to_float(act6[col + WORKGROUP_SIZE]);
        s7 += wt0 * bf16_to_float(act7[col]) + wt1 * bf16_to_float(act7[col + WORKGROUP_SIZE]);
        base_ptr += 768;
    }
    if (col < K) {
        uint32_t raw = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t g = raw >> 8;
        float wt = (__builtin_expect(g != 0, 1))
            ? structured12_decode(raw, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        s0 += wt * bf16_to_float(act0[col]);
        s1 += wt * bf16_to_float(act1[col]);
        s2 += wt * bf16_to_float(act2[col]);
        s3 += wt * bf16_to_float(act3[col]);
        s4 += wt * bf16_to_float(act4[col]);
        s5 += wt * bf16_to_float(act5[col]);
        s6 += wt * bf16_to_float(act6[col]);
        s7 += wt * bf16_to_float(act7[col]);
    }

    // Warp reduction + output
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
        s4 += __shfl_down(s4, off, WARP_SIZE);
        s5 += __shfl_down(s5, off, WARP_SIZE);
        s6 += __shfl_down(s6, off, WARP_SIZE);
        s7 += __shfl_down(s7, off, WARP_SIZE);
    }
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[wid]       = s0;
        warp_sums[wid + 4]   = s1;
        warp_sums[wid + 8]   = s2;
        warp_sums[wid + 12]  = s3;
        warp_sums[wid + 16]  = s4;
        warp_sums[wid + 20]  = s5;
        warp_sums[wid + 24]  = s6;
        warp_sums[wid + 28]  = s7;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
        out4[row] = warp_sums[16]+warp_sums[17]+warp_sums[18]+warp_sums[19];
        out5[row] = warp_sums[20]+warp_sums[21]+warp_sums[22]+warp_sums[23];
        out6[row] = warp_sums[24]+warp_sums[25]+warp_sums[26]+warp_sums[27];
        out7[row] = warp_sums[28]+warp_sums[29]+warp_sums[30]+warp_sums[31];
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit launch wrappers
// ---------------------------------------------------------------------------
extern "C" {

int launch_structured12_v2_async(
    const void* packed, int base_exp,
    const void* activations, void* output, int M, int K, void* stream,
    const void* patch_row_offsets, const void* patch_cols,
    const void* patch_correct_vals, const void* patch_wrong_vals)
{
    // Compute SRDs for packed data buffer
    uint64_t pack_addr = (uint64_t)packed;
    uint32_t psrd0 = (uint32_t)(pack_addr);
    uint32_t psrd1 = (uint32_t)(pack_addr >> 32) & 0xFFFF;
    // Size: M rows * K cols * 12 bits / 8 + some slack for the dwordx2 reads
    uint32_t psrd2 = (uint32_t)((int64_t)M * K * 12 / 8 + 8);
    uint32_t psrd3 = 0x00020000;  // DATA_FORMAT=4, raw buffer (NUM_FORMAT=0)

    // Compute SRDs for activation buffer
    uint64_t act_addr = (uint64_t)activations;
    uint32_t asrd0 = (uint32_t)(act_addr);
    uint32_t asrd1 = (uint32_t)(act_addr >> 32) & 0xFFFF;
    uint32_t asrd2 = (uint32_t)(K * 2);  // K int16 elements = K*2 bytes
    uint32_t asrd3 = 0x00020000;

    // Try original v2 kernel (23 VGPRs, 100% occupancy) vs bufload (33 VGPRs, 70%)
    hipLaunchKernelGGL(structured12_matvec_v2,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)activations, (float*)output, M, K);
    // Inline patches on thread 0 via separate lightweight launch
    if (patch_row_offsets) {
        hipLaunchKernelGGL(apply_patches_v2,
            dim3(M), dim3(WARP_SIZE), 0, (hipStream_t)stream,
            (const int32_t*)patch_row_offsets, (const int32_t*)patch_cols,
            (const int16_t*)patch_correct_vals, (const int16_t*)patch_wrong_vals,
            (const int16_t*)activations, (float*)output, M);
    }
    return 0;
}

int launch_structured12_v2_fp32act_async(
    const void* packed, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_v2_fp32act,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const float*)activations, (float*)output, M, K);
    return 0;
}

int launch_structured12_v2_dual_async(
    const void* packed_a, int base_exp_a,
    const void* packed_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_v2_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed_a, base_exp_a,
        (const uint32_t*)packed_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

int launch_structured12_fused_b1_async(
    const void* packed, int base_exp,
    const void* activations,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* output, int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_fused_b1,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)activations,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)output, M, K);
    return 0;
}

int launch_structured12_batch4_async(
    const void* packed, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream)
{
    if (esc_counts) {
        hipLaunchKernelGGL(structured12_matvec_batch4,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint32_t*)packed, base_exp,
            (const int16_t*)a0, (const int16_t*)a1,
            (const int16_t*)a2, (const int16_t*)a3,
            (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
            (const int16_t*)esc_vals,
            (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    } else {
        hipLaunchKernelGGL(structured12_matvec_batch4_scan,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint32_t*)packed, base_exp,
            (const int16_t*)a0, (const int16_t*)a1,
            (const int16_t*)a2, (const int16_t*)a3,
            (const int32_t*)esc_row_base,
            (const int16_t*)esc_vals,
            (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    }
    return 0;
}

int launch_structured12_batch8_async(
    const void* packed, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* a4, const void* a5, const void* a6, const void* a7,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3,
    void* o4, void* o5, void* o6, void* o7,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_batch8,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int16_t*)a4, (const int16_t*)a5,
        (const int16_t*)a6, (const int16_t*)a7,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3,
        (float*)o4, (float*)o5, (float*)o6, (float*)o7,
        M, K);
    return 0;
}

}

// ---------------------------------------------------------------------------
// Split12 matvec: byte-aligned dual-array format
// sign_mantissa[M*K]: uint8_t per element = [sign 1][mantissa 7]
// groups[M*K/2]: nibble-packed = 2 groups per byte (low=even, high=odd)
// ALL loads are byte-aligned → ZERO read amplification vs 1.33x with 12-bit packed
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v2(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[4];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // Pointer setup: row-major layout
    const uint8_t* sm_ptr  = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr  = groups + (int64_t)row * K / 2 + tid / 2;
    const char*    act_ptr = (const char*)(activations + tid);

    // Each thread always accesses same nibble (tid parity is constant since 256 is even)
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Load sign+mantissa (4 byte-aligned loads)
        uint8_t sm0 = sm_ptr[0];
        uint8_t sm1 = sm_ptr[256];
        uint8_t sm2 = sm_ptr[512];
        uint8_t sm3 = sm_ptr[768];

        // Load group nibbles (4 byte-aligned loads, each shared between 2 elements)
        uint8_t gb0 = gr_ptr[0];
        uint8_t gb1 = gr_ptr[128];
        uint8_t gb2 = gr_ptr[256];
        uint8_t gb3 = gr_ptr[384];

        // Extract group from correct nibble
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
        uint32_t g2 = is_odd ? (gb2 >> 4) : (gb2 & 0xF);
        uint32_t g3 = is_odd ? (gb3 >> 4) : (gb3 & 0xF);

        // Decode to BF16 → float (pure ALU, no memory)
        union { uint32_t u; float f; } c0, c1, c2, c3;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
        c2.u = (((uint32_t)(sm2 >> 7)) << 15 | ((uint32_t)(base_exp + g2)) << 7 | (sm2 & 0x7F)) << 16;
        c3.u = (((uint32_t)(sm3 >> 7)) << 15 | ((uint32_t)(base_exp + g3)) << 7 | (sm3 & 0x7F)) << 16;

        // FMA with activations
        sum0 += c0.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += c1.f * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += c2.f * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += c3.f * bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        sm_ptr  += 1024;   // 256 * 4 bytes
        gr_ptr  += 512;    // 256 * 4 / 2 bytes
        act_ptr += 2048;   // 256 * 4 * 2 bytes (int16)
    }
    // Tail loop
    for (; col < K; col += WORKGROUP_SIZE) {
        uint8_t sm = sm_ptr[0];
        uint8_t gb = gr_ptr[0];
        uint32_t g = is_odd ? (gb >> 4) : (gb & 0xF);
        union { uint32_t u; float f; } c;
        c.u = (((uint32_t)(sm >> 7)) << 15 | ((uint32_t)(base_exp + g)) << 7 | (sm & 0x7F)) << 16;
        sum0 += c.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sm_ptr  += 256;
        gr_ptr  += 128;
        act_ptr += 512;
    }

    // Warp reduction + output (same as v2)
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
}

// ---------------------------------------------------------------------------
// Split12 multirow: 2 rows per block, shared activation reads
// Grid: (M/2) blocks. Each block computes 2 adjacent output rows.
// 25% fewer loads per element: 12 loads / 8 elements vs 8 loads / 4 elements
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v2_multirow(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,     // [K] BF16
    float* __restrict__ output,                  // [M]
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[8]; // 4 per row

    // Row 0 accumulators
    float sum_a0 = 0.0f, sum_a1 = 0.0f, sum_a2 = 0.0f, sum_a3 = 0.0f;
    // Row 1 accumulators
    float sum_b0 = 0.0f, sum_b1 = 0.0f, sum_b2 = 0.0f, sum_b3 = 0.0f;

    const int has_row1 = (row1 < M);

    // Pointer setup for row 0
    const uint8_t* sm_ptr0  = sign_mantissa + (int64_t)row0 * K + tid;
    const uint8_t* gr_ptr0  = groups + (int64_t)row0 * K / 2 + tid / 2;
    // Pointer setup for row 1
    const uint8_t* sm_ptr1  = sign_mantissa + (int64_t)row1 * K + tid;
    const uint8_t* gr_ptr1  = groups + (int64_t)row1 * K / 2 + tid / 2;
    // Shared activation pointer
    const char*    act_ptr  = (const char*)(activations + tid);

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Load activations ONCE (shared between both rows)
        float a0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float a1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float a2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float a3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        // Row 0: load sign+mantissa and groups
        uint8_t sm0_0 = sm_ptr0[0];
        uint8_t sm0_1 = sm_ptr0[256];
        uint8_t sm0_2 = sm_ptr0[512];
        uint8_t sm0_3 = sm_ptr0[768];
        uint8_t gb0_0 = gr_ptr0[0];
        uint8_t gb0_1 = gr_ptr0[128];
        uint8_t gb0_2 = gr_ptr0[256];
        uint8_t gb0_3 = gr_ptr0[384];

        uint32_t g0_0 = is_odd ? (gb0_0 >> 4) : (gb0_0 & 0xF);
        uint32_t g0_1 = is_odd ? (gb0_1 >> 4) : (gb0_1 & 0xF);
        uint32_t g0_2 = is_odd ? (gb0_2 >> 4) : (gb0_2 & 0xF);
        uint32_t g0_3 = is_odd ? (gb0_3 >> 4) : (gb0_3 & 0xF);

        union { uint32_t u; float f; } c0_0, c0_1, c0_2, c0_3;
        c0_0.u = (((uint32_t)(sm0_0 >> 7)) << 15 | ((uint32_t)(base_exp + g0_0)) << 7 | (sm0_0 & 0x7F)) << 16;
        c0_1.u = (((uint32_t)(sm0_1 >> 7)) << 15 | ((uint32_t)(base_exp + g0_1)) << 7 | (sm0_1 & 0x7F)) << 16;
        c0_2.u = (((uint32_t)(sm0_2 >> 7)) << 15 | ((uint32_t)(base_exp + g0_2)) << 7 | (sm0_2 & 0x7F)) << 16;
        c0_3.u = (((uint32_t)(sm0_3 >> 7)) << 15 | ((uint32_t)(base_exp + g0_3)) << 7 | (sm0_3 & 0x7F)) << 16;

        sum_a0 += c0_0.f * a0;
        sum_a1 += c0_1.f * a1;
        sum_a2 += c0_2.f * a2;
        sum_a3 += c0_3.f * a3;

        // Row 1: load sign+mantissa and groups
        if (has_row1) {
            uint8_t sm1_0 = sm_ptr1[0];
            uint8_t sm1_1 = sm_ptr1[256];
            uint8_t sm1_2 = sm_ptr1[512];
            uint8_t sm1_3 = sm_ptr1[768];
            uint8_t gb1_0 = gr_ptr1[0];
            uint8_t gb1_1 = gr_ptr1[128];
            uint8_t gb1_2 = gr_ptr1[256];
            uint8_t gb1_3 = gr_ptr1[384];

            uint32_t g1_0 = is_odd ? (gb1_0 >> 4) : (gb1_0 & 0xF);
            uint32_t g1_1 = is_odd ? (gb1_1 >> 4) : (gb1_1 & 0xF);
            uint32_t g1_2 = is_odd ? (gb1_2 >> 4) : (gb1_2 & 0xF);
            uint32_t g1_3 = is_odd ? (gb1_3 >> 4) : (gb1_3 & 0xF);

            union { uint32_t u; float f; } c1_0, c1_1, c1_2, c1_3;
            c1_0.u = (((uint32_t)(sm1_0 >> 7)) << 15 | ((uint32_t)(base_exp + g1_0)) << 7 | (sm1_0 & 0x7F)) << 16;
            c1_1.u = (((uint32_t)(sm1_1 >> 7)) << 15 | ((uint32_t)(base_exp + g1_1)) << 7 | (sm1_1 & 0x7F)) << 16;
            c1_2.u = (((uint32_t)(sm1_2 >> 7)) << 15 | ((uint32_t)(base_exp + g1_2)) << 7 | (sm1_2 & 0x7F)) << 16;
            c1_3.u = (((uint32_t)(sm1_3 >> 7)) << 15 | ((uint32_t)(base_exp + g1_3)) << 7 | (sm1_3 & 0x7F)) << 16;

            sum_b0 += c1_0.f * a0;
            sum_b1 += c1_1.f * a1;
            sum_b2 += c1_2.f * a2;
            sum_b3 += c1_3.f * a3;
        }

        sm_ptr0 += 1024;
        gr_ptr0 += 512;
        sm_ptr1 += 1024;
        gr_ptr1 += 512;
        act_ptr += 2048;
    }
    // Tail loop
    for (; col < K; col += WORKGROUP_SIZE) {
        float a = bf16_to_float(*(const int16_t*)(act_ptr));
        uint8_t sm0 = sm_ptr0[0];
        uint8_t gb0 = gr_ptr0[0];
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        union { uint32_t u; float f; } c0;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        sum_a0 += c0.f * a;

        if (has_row1) {
            uint8_t sm1 = sm_ptr1[0];
            uint8_t gb1 = gr_ptr1[0];
            uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
            union { uint32_t u; float f; } c1;
            c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
            sum_b0 += c1.f * a;
        }

        sm_ptr0 += 256;
        gr_ptr0 += 128;
        sm_ptr1 += 256;
        gr_ptr1 += 128;
        act_ptr += 512;
    }

    // Warp reduction for row 0
    float sum0 = sum_a0 + sum_a1 + sum_a2 + sum_a3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid >> 6] = sum0;

    // Warp reduction for row 1
    float sum1 = sum_b0 + sum_b1 + sum_b2 + sum_b3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[4 + (tid >> 6)] = sum1;

    __syncthreads();
    if (tid == 0) {
        output[row0] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        if (has_row1)
            output[row1] = warp_sums[4] + warp_sums[5] + warp_sums[6] + warp_sums[7];
    }
}

extern "C" {

int launch_split12_v2_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    // Use multirow (2 rows/block) when M >= 2
    if (M >= 2) {
        int grid = (M + 1) / 2;  // ceil(M/2)
        hipLaunchKernelGGL(split12_matvec_v2_multirow,
            dim3(grid), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    } else {
        hipLaunchKernelGGL(split12_matvec_v2,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    }
    return 0;
}

}
