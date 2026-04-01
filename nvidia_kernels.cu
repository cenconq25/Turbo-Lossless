/**
 * NVIDIA-Optimized Kernels for Turbo Lossless Engine
 * Purpose-built for modern NVIDIA GPUs (sm_80+) with tensor core support.
 *
 * Architecture:
 *   B=1:  Per-row split12 matvec (bandwidth-bound, same as AMD but 32-wide warps)
 *   B>=2: Tiled GEMM with on-the-fly decode → shared memory → tensor core WMMA
 *         Reads 1.33x less data than BF16 cuBLAS, same tensor core throughput
 *
 * This file replaces decompress_v2.hip for NVIDIA builds.
 * AMD keeps the original decompress_v2.hip untouched.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

// ============================================================
// Constants
// ============================================================
#define WARP_SIZE 32
#define BLOCK_DIM 256
#define NUM_WARPS (BLOCK_DIM / WARP_SIZE)

// WMMA tile dimensions for BF16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tiled GEMM parameters
#define TILE_M  64   // output rows per block
#define TILE_N  16   // batch items per block (= WMMA_N, expands via grid)
#define TILE_K  16   // K columns per WMMA operation

// ============================================================
// Helpers
// ============================================================
__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

// Convert split12 element to __nv_bfloat16
__device__ __forceinline__
__nv_bfloat16 decode_split12_bf16(uint8_t sm, uint8_t group, int base_exp) {
    uint16_t bf16_bits = ((uint16_t)(sm >> 7) << 15) |
                         ((uint16_t)(base_exp + group) << 7) |
                         (sm & 0x7F);
    return *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
}

// ============================================================
// B=1 Split12 Matvec (per-row, bandwidth-optimized)
// Same algorithm as AMD version but tuned for 32-wide warps
// ============================================================
__launch_bounds__(BLOCK_DIM)
__global__ void nv_split12_matvec(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const uint8_t* sm_ptr = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr = groups + (int64_t)row * K / 2 + tid / 2;
    const char* act_ptr = (const char*)(activations + tid);
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + BLOCK_DIM * 3 < K; col += BLOCK_DIM * 4) {
        uint8_t sm0 = sm_ptr[0], sm1 = sm_ptr[256], sm2 = sm_ptr[512], sm3 = sm_ptr[768];
        uint8_t gb0 = gr_ptr[0], gb1 = gr_ptr[128], gb2 = gr_ptr[256], gb3 = gr_ptr[384];
        uint32_t g0 = is_odd ? (gb0>>4) : (gb0&0xF);
        uint32_t g1 = is_odd ? (gb1>>4) : (gb1&0xF);
        uint32_t g2 = is_odd ? (gb2>>4) : (gb2&0xF);
        uint32_t g3 = is_odd ? (gb3>>4) : (gb3&0xF);

        union { uint32_t u; float f; } c0, c1, c2, c3;
        c0.u = (((uint32_t)(sm0>>7))<<15 | ((uint32_t)(base_exp+g0))<<7 | (sm0&0x7F)) << 16;
        c1.u = (((uint32_t)(sm1>>7))<<15 | ((uint32_t)(base_exp+g1))<<7 | (sm1&0x7F)) << 16;
        c2.u = (((uint32_t)(sm2>>7))<<15 | ((uint32_t)(base_exp+g2))<<7 | (sm2&0x7F)) << 16;
        c3.u = (((uint32_t)(sm3>>7))<<15 | ((uint32_t)(base_exp+g3))<<7 | (sm3&0x7F)) << 16;

        sum0 += c0.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += c1.f * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += c2.f * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += c3.f * bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        sm_ptr += 1024; gr_ptr += 512; act_ptr += 2048;
    }
    for (; col < K; col += BLOCK_DIM) {
        uint8_t sm = sm_ptr[0], gb = gr_ptr[0];
        uint32_t g = is_odd ? (gb>>4) : (gb&0xF);
        union { uint32_t u; float f; } c;
        c.u = (((uint32_t)(sm>>7))<<15 | ((uint32_t)(base_exp+g))<<7 | (sm&0x7F)) << 16;
        sum0 += c.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sm_ptr += 256; gr_ptr += 128; act_ptr += 512;
    }

    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int w = 0; w < NUM_WARPS; w++) s += warp_sums[w];
        output[row] = s;
    }
}

// B=1 multirow variant (2 rows per block, shared activation reads)
__launch_bounds__(BLOCK_DIM)
__global__ void nv_split12_matvec_multirow(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];
    float sa0=0,sa1=0,sa2=0,sa3=0, sb0=0,sb1=0,sb2=0,sb3=0;
    const int has_row1 = (row1 < M);

    const uint8_t* sm0p = sign_mantissa + (int64_t)row0 * K + tid;
    const uint8_t* gr0p = groups + (int64_t)row0 * K / 2 + tid / 2;
    const uint8_t* sm1p = sign_mantissa + (int64_t)row1 * K + tid;
    const uint8_t* gr1p = groups + (int64_t)row1 * K / 2 + tid / 2;
    const char* act_ptr = (const char*)(activations + tid);
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + BLOCK_DIM * 3 < K; col += BLOCK_DIM * 4) {
        float a0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float a1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float a2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float a3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        #define DECODE4(ptr, grp, s0, s1, s2, s3) { \
            uint8_t _s0=ptr[0],_s1=ptr[256],_s2=ptr[512],_s3=ptr[768]; \
            uint8_t _g0=grp[0],_g1=grp[128],_g2=grp[256],_g3=grp[384]; \
            uint32_t _e0=is_odd?(_g0>>4):(_g0&0xF), _e1=is_odd?(_g1>>4):(_g1&0xF); \
            uint32_t _e2=is_odd?(_g2>>4):(_g2&0xF), _e3=is_odd?(_g3>>4):(_g3&0xF); \
            union{uint32_t u;float f;} _c0,_c1,_c2,_c3; \
            _c0.u=((uint32_t)(_s0>>7)<<15|(uint32_t)(base_exp+_e0)<<7|(_s0&0x7F))<<16; \
            _c1.u=((uint32_t)(_s1>>7)<<15|(uint32_t)(base_exp+_e1)<<7|(_s1&0x7F))<<16; \
            _c2.u=((uint32_t)(_s2>>7)<<15|(uint32_t)(base_exp+_e2)<<7|(_s2&0x7F))<<16; \
            _c3.u=((uint32_t)(_s3>>7)<<15|(uint32_t)(base_exp+_e3)<<7|(_s3&0x7F))<<16; \
            s0+=_c0.f*a0; s1+=_c1.f*a1; s2+=_c2.f*a2; s3+=_c3.f*a3; }

        DECODE4(sm0p, gr0p, sa0, sa1, sa2, sa3);
        if (has_row1) { DECODE4(sm1p, gr1p, sb0, sb1, sb2, sb3); }
        #undef DECODE4

        sm0p+=1024; gr0p+=512; sm1p+=1024; gr1p+=512; act_ptr+=2048;
    }
    for (; col < K; col += BLOCK_DIM) {
        float a = bf16_to_float(*(const int16_t*)(act_ptr));
        uint8_t s0=sm0p[0], g0=gr0p[0]; uint32_t e0=is_odd?(g0>>4):(g0&0xF);
        union{uint32_t u;float f;} c0;
        c0.u=((uint32_t)(s0>>7)<<15|(uint32_t)(base_exp+e0)<<7|(s0&0x7F))<<16;
        sa0 += c0.f * a;
        if (has_row1) {
            uint8_t s1=sm1p[0], g1=gr1p[0]; uint32_t e1=is_odd?(g1>>4):(g1&0xF);
            union{uint32_t u;float f;} c1;
            c1.u=((uint32_t)(s1>>7)<<15|(uint32_t)(base_exp+e1)<<7|(s1&0x7F))<<16;
            sb0 += c1.f * a;
        }
        sm0p+=256; gr0p+=128; sm1p+=256; gr1p+=128; act_ptr+=512;
    }

    float sum0 = sa0+sa1+sa2+sa3, sum1 = sb0+sb1+sb2+sb3;
    for (int off=WARP_SIZE/2; off>0; off>>=1) {
        sum0 += __shfl_down_sync(0xFFFFFFFF, sum0, off, WARP_SIZE);
        sum1 += __shfl_down_sync(0xFFFFFFFF, sum1, off, WARP_SIZE);
    }
    if ((tid&(WARP_SIZE-1))==0) {
        warp_sums[tid/WARP_SIZE] = sum0;
        warp_sums[NUM_WARPS + tid/WARP_SIZE] = sum1;
    }
    __syncthreads();
    if (tid == 0) {
        float s0=0, s1=0;
        for (int w=0; w<NUM_WARPS; w++) { s0+=warp_sums[w]; s1+=warp_sums[NUM_WARPS+w]; }
        output[row0] = s0;
        if (has_row1) output[row1] = s1;
    }
}

// ============================================================
// B=1 Patch correction (escape values)
// ============================================================
__global__ void nv_apply_patches(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const int16_t* __restrict__ activations,
    float* __restrict__ output, int M)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row], end = row_offsets[row + 1];
    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(activations[patch_cols[p]]);
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        correction += __shfl_down_sync(0xFFFFFFFF, correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) output[row] += correction;
}

// ============================================================
// Batched GEMM with Tensor Cores + On-the-fly Decode
//
// Grid:  (ceil(M/TILE_M), ceil(B/TILE_N))
// Block: 256 threads (8 warps)
//
// Each block computes a TILE_M × TILE_N output tile.
// For each K-tile of 16 columns:
//   1. Decode TILE_M × 16 weights to BF16 in shared memory
//   2. Load 16 × TILE_N activations (BF16) into shared memory
//   3. WMMA: C[TILE_M × TILE_N] += A[TILE_M × 16] × B[16 × TILE_N]
//
// Tensor core throughput: 176 TFLOPS BF16
// Weight bandwidth: 1.33x less than BF16 cuBLAS
// ============================================================

// Shared memory layout for GEMM tile
// Weight tile:     TILE_M × TILE_K __nv_bfloat16 = 64 × 16 × 2 = 2 KB
// Activation tile: TILE_K × TILE_N __nv_bfloat16 = 16 × 16 × 2 = 512 B
// Total per tile iteration: ~2.5 KB (fits easily in 48 KB shared memory)

__launch_bounds__(BLOCK_DIM)
__global__ void nv_split12_gemm_tc(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,     // [B * K] BF16 (strided)
    int act_stride,                              // stride between sequences in activations
    const int32_t* __restrict__ escape_row_base, // [M]
    const uint8_t* __restrict__ escape_counts,   // [M * 256] (TURBO_FAST only)
    const int16_t* __restrict__ escape_vals,     // [total_escapes]
    float* __restrict__ output,                  // [B * M] (strided)
    int out_stride,                              // stride between sequences in output
    int M, int K, int B_total)
{
    // Block position in the output grid
    const int tile_m_start = blockIdx.x * TILE_M;  // starting output row
    const int tile_n_start = blockIdx.y * TILE_N;  // starting batch item
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    // const int lane = tid & (WARP_SIZE - 1);  // used in future optimizations

    // Bounds
    const int rows_this_tile = min(TILE_M, M - tile_m_start);
    const int cols_this_tile = min(TILE_N, B_total - tile_n_start);
    if (rows_this_tile <= 0 || cols_this_tile <= 0) return;

    // Shared memory for weight and activation tiles
    // Weight: [TILE_M][TILE_K] in row-major BF16
    // Activation: [TILE_K][TILE_N] in col-major BF16 (for WMMA B fragment)
    extern __shared__ char smem[];
    __nv_bfloat16* w_tile = (__nv_bfloat16*)smem;                           // [TILE_M * TILE_K]
    __nv_bfloat16* a_tile = (__nv_bfloat16*)(smem + TILE_M * TILE_K * 2);  // [TILE_K * TILE_N]

    // WMMA fragments: each warp handles a 16×16 sub-tile of the output
    // With TILE_M=64 and TILE_N=16: need 4 WMMA tiles vertically (4 × 16 = 64 rows)
    // 8 warps: each warp handles one 16×16 sub-tile, warps 0-3 do the 4 tiles,
    // warps 4-7 assist with data loading but don't compute (or we use 4 warps only)

    // Simpler: 4 warps compute (one 16×16 tile each), 4 warps idle for compute
    // Better: each of 8 warps computes part of the output, then combine

    // Actually: with TILE_M=64 and TILE_N=16:
    //   4 WMMA operations needed: rows [0:16], [16:32], [32:48], [48:64] × [0:16]
    //   Assign to warps 0,1,2,3. Warps 4-7 help with data loading.

    // Accumulator fragments (one per warp that computes)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Which 16-row sub-tile this warp handles
    int warp_tile_row = warp_id;  // warp 0→rows 0-15, warp 1→16-31, etc.
    bool warp_computes = (warp_id < (TILE_M / WMMA_M));  // warps 0-3 compute

    // K-tile loop
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        int k_count = min(TILE_K, K - k_start);

        // Phase 1: Cooperative decode of weight tile → shared memory BF16
        // 256 threads decode TILE_M × TILE_K = 64 × 16 = 1024 elements
        // Each thread decodes ~4 elements
        for (int i = tid; i < TILE_M * k_count; i += BLOCK_DIM) {
            int m_local = i / k_count;
            int k_local = i % k_count;
            int row = tile_m_start + m_local;
            int col = k_start + k_local;

            if (row < M) {
                uint8_t sm = sign_mantissa[(int64_t)row * K + col];
                uint8_t gb = groups[(int64_t)row * K / 2 + col / 2];
                uint32_t group = (col & 1) ? (gb >> 4) : (gb & 0xF);

                __nv_bfloat16 val;
                if (__builtin_expect(group != 0, 1)) {
                    val = decode_split12_bf16(sm, group, base_exp);
                } else {
                    // Escape: use correct BF16 value
                    // For simplicity, decode as base_exp+0 (will be corrected by patch kernel)
                    val = decode_split12_bf16(sm, 0, base_exp);
                }
                w_tile[m_local * TILE_K + k_local] = val;
            } else {
                w_tile[m_local * TILE_K + k_local] = __float2bfloat16(0.0f);
            }
        }

        // Phase 2: Cooperative load of activation tile → shared memory BF16
        // TILE_K × TILE_N = 16 × 16 = 256 elements (1 per thread)
        for (int i = tid; i < k_count * cols_this_tile; i += BLOCK_DIM) {
            int k_local = i / cols_this_tile;
            int n_local = i % cols_this_tile;
            int b_idx = tile_n_start + n_local;
            // Column-major for WMMA B fragment
            a_tile[n_local * TILE_K + k_local] =
                *reinterpret_cast<const __nv_bfloat16*>(&activations[b_idx * act_stride + k_start + k_local]);
        }
        // Zero-pad if needed
        if (k_count < TILE_K) {
            for (int i = tid; i < TILE_M * (TILE_K - k_count); i += BLOCK_DIM) {
                int m_local = i / (TILE_K - k_count);
                int k_local = k_count + i % (TILE_K - k_count);
                w_tile[m_local * TILE_K + k_local] = __float2bfloat16(0.0f);
            }
            for (int i = tid; i < (TILE_K - k_count) * cols_this_tile; i += BLOCK_DIM) {
                int k_local = k_count + i / cols_this_tile;
                int n_local = i % cols_this_tile;
                a_tile[n_local * TILE_K + k_local] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Phase 3: WMMA — tensor core matrix multiply
        if (warp_computes && warp_tile_row * WMMA_M < rows_this_tile) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

            // Load weight sub-tile [16×16] from shared memory
            wmma::load_matrix_sync(a_frag, w_tile + warp_tile_row * WMMA_M * TILE_K, TILE_K);
            // Load activation tile [16×16] from shared memory
            wmma::load_matrix_sync(b_frag, a_tile, TILE_K);
            // Tensor core FMA
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    // Phase 4: Store accumulator to output
    if (warp_computes && warp_tile_row * WMMA_M < rows_this_tile) {
        // Store to shared memory first, then write to global with stride
        float* out_smem = (float*)smem;  // reuse shared memory
        wmma::store_matrix_sync(out_smem + warp_tile_row * WMMA_M * TILE_N, acc_frag, TILE_N, wmma::mem_row_major);
        __syncthreads();

        // Write to global output with proper stride
        for (int i = tid; i < WMMA_M * cols_this_tile; i += BLOCK_DIM) {
            int m_local = i / cols_this_tile;
            int n_local = i % cols_this_tile;
            int row = tile_m_start + warp_tile_row * WMMA_M + m_local;
            int b_idx = tile_n_start + n_local;
            if (row < M && b_idx < B_total) {
                output[b_idx * out_stride + row] = out_smem[warp_tile_row * WMMA_M * TILE_N + m_local * TILE_N + n_local];
            }
        }
    }
}

// ============================================================
// Launch wrappers (extern "C" for inference.cpp)
// ============================================================
extern "C" {

// B=1 split12 matvec
int nv_launch_split12_v2_async(
    const void* sm, const void* gr, int base_exp,
    const void* act, void* out, int M, int K, void* stream)
{
    if (M >= 2) {
        int grid = (M + 1) / 2;
        nv_split12_matvec_multirow<<<grid, BLOCK_DIM, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const int16_t*)act, (float*)out, M, K);
    } else {
        nv_split12_matvec<<<M, BLOCK_DIM, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const int16_t*)act, (float*)out, M, K);
    }
    return 0;
}

// B=1 patch correction
int nv_launch_patches_async(
    const void* row_off, const void* cols, const void* correct, const void* wrong,
    const void* act, void* out, int M, void* stream)
{
    if (M == 0) return 0;
    nv_apply_patches<<<M, WARP_SIZE, 0, (cudaStream_t)stream>>>(
        (const int32_t*)row_off, (const int32_t*)cols,
        (const int16_t*)correct, (const int16_t*)wrong,
        (const int16_t*)act, (float*)out, M);
    return 0;
}

// Batched tensor core GEMM
int nv_launch_split12_gemm_tc_async(
    const void* sm, const void* gr, int base_exp,
    const void* act, int act_stride,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* out, int out_stride,
    int M, int K, int B, void* stream)
{
    dim3 grid((M + TILE_M - 1) / TILE_M, (B + TILE_N - 1) / TILE_N);
    int smem = TILE_M * TILE_K * sizeof(__nv_bfloat16) +  // weight tile
               TILE_K * TILE_N * sizeof(__nv_bfloat16);    // activation tile

    nv_split12_gemm_tc<<<grid, BLOCK_DIM, smem, (cudaStream_t)stream>>>(
        (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
        (const int16_t*)act, act_stride,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)out, out_stride,
        M, K, B);

    // Escape correction is handled by the caller via separate nv_launch_patches_async calls
    // (one per batch item, using the CSR patch data from CompressedWeight)

    return 0;
}

}
