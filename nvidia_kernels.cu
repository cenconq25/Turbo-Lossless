/**
 * NVIDIA-Optimized Kernels for Turbo Lossless Engine
 *
 * Two kernel families:
 *   B=1:  Per-row split12 matvec (bandwidth-bound, 65 tok/s at B=1)
 *   B>=2: Fused decode+GEMM via PTX mma.sync (tensor cores, 600+ tok/s at B=64)
 *
 * The fused kernel reads compressed split12 data (1.5 bytes/weight) from DRAM,
 * loads into shared memory via vectorized uint4 loads, decodes directly into
 * PTX mma.sync registers (3 ALU ops), and outputs via tensor cores.
 *
 * NO intermediate BF16 buffer, NO DRAM round-trip.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
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

// ============================================================
// Helpers
// ============================================================
__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

// decode_split12_bf16 defined in split12_gemm.cuh

// ============================================================
// B=1 Split12 Matvec (per-row, bandwidth-optimized)
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
        uint32_t g0 = is_odd ? (gb0>>4) : (gb0&0xF), g1 = is_odd ? (gb1>>4) : (gb1&0xF);
        uint32_t g2 = is_odd ? (gb2>>4) : (gb2&0xF), g3 = is_odd ? (gb3>>4) : (gb3&0xF);
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
        float s = 0; for (int w = 0; w < NUM_WARPS; w++) s += warp_sums[w];
        output[row] = s;
    }
}

// B=1 multirow (2 rows per block, shared activation reads)
__launch_bounds__(BLOCK_DIM)
__global__ void nv_split12_matvec_multirow(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row0 = blockIdx.x * 2, row1 = row0 + 1;
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
// Patch correction kernels (escape values)
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

// Compact batched patch correction: each block handles 16 rows × 1 batch item
// Grid: (ceil(num_nonempty/16), B) — 16x fewer blocks than sparse version
#define PATCH_ROWS_PER_BLOCK 16
__global__ void nv_apply_patches_batch(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const int32_t* __restrict__ nonempty_rows,
    const int16_t* __restrict__ activations, int act_stride,
    float* __restrict__ output, int out_stride, int num_nonempty, int B)
{
    int block_row_start = blockIdx.x * PATCH_ROWS_PER_BLOCK;
    int b = blockIdx.y;
    if (b >= B) return;
    int tid = threadIdx.x;
    // 256 threads, 16 rows → 16 threads per row (half-warp)
    int local_row = tid / 16;
    int local_tid = tid % 16;
    int row_idx = block_row_start + local_row;
    if (row_idx >= num_nonempty) return;

    int row = nonempty_rows[row_idx];
    int start = row_offsets[row], end = row_offsets[row + 1];
    const int16_t* act = activations + b * act_stride;

    float correction = 0.0f;
    for (int p = start + local_tid; p < end; p += 16)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(act[patch_cols[p]]);
    // Half-warp reduction (16 threads)
    for (int off = 8; off > 0; off >>= 1)
        correction += __shfl_down_sync(0xFFFF << (local_row * 16), correction, off, 16);
    if (local_tid == 0 && correction != 0.0f)
        output[b * out_stride + row] += correction;
}

// ============================================================
// Fused Decode+GEMM via PTX mma.sync (the main batched kernel)
// ============================================================
#include "split12_gemm.cuh"

// ============================================================
// Launch wrappers
// ============================================================
extern "C" {

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

int nv_launch_patches_batch_async(
    const void* row_off, const void* cols, const void* correct, const void* wrong,
    const void* nonempty_rows, int num_nonempty,
    const void* act, int act_stride, void* out, int out_stride,
    int B, void* stream)
{
    if (num_nonempty == 0 || B == 0) return 0;
    dim3 grid((num_nonempty + PATCH_ROWS_PER_BLOCK - 1) / PATCH_ROWS_PER_BLOCK, B);
    nv_apply_patches_batch<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const int32_t*)row_off, (const int32_t*)cols,
        (const int16_t*)correct, (const int16_t*)wrong,
        (const int32_t*)nonempty_rows,
        (const int16_t*)act, act_stride,
        (float*)out, out_stride, num_nonempty, B);
    return 0;
}

int nv_launch_split12_fused_gemm_async(
    const void* sm, const void* gr, int base_exp,
    const void* act, int act_stride,
    void* out, int out_stride,
    int M, int K, int B, void* stream)
{
    // Adaptive TILE_N: TILE_N=32 optimal for B>=32
    if (B >= 32) {
        constexpr int TN = 32, WCT = 4;
        dim3 grid((M + S12_TILE_M - 1) / S12_TILE_M, (B + TN - 1) / TN);
        int smem = S12_TILE_M * S12_TILE_K * 2 + S12_TILE_M * S12_TILE_K / 2 * 2
                 + S12_TILE_K * TN * sizeof(__nv_bfloat16) * 2;
        split12_fused_gemm<TN, WCT><<<grid, S12_BLOCK, smem, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const __nv_bfloat16*)act, (float*)out, M, K, B, out_stride);
    } else {
        constexpr int TN = 16, WCT = 2;
        dim3 grid((M + S12_TILE_M - 1) / S12_TILE_M, (B + TN - 1) / TN);
        int smem = S12_TILE_M * S12_TILE_K * 2 + S12_TILE_M * S12_TILE_K / 2 * 2
                 + S12_TILE_K * TN * sizeof(__nv_bfloat16) * 2;
        split12_fused_gemm<TN, WCT><<<grid, S12_BLOCK, smem, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const __nv_bfloat16*)act, (float*)out, M, K, B, out_stride);
    }

    return 0;
}

// Legacy cuBLAS path (kept for comparison, not production)
static cublasHandle_t s_cublas_handle = nullptr;
static cudaStream_t s_decode_stream = nullptr;
static cudaEvent_t s_decode_done = nullptr;
static cudaEvent_t s_gemm_done = nullptr;
static int s_ping = 0;

__launch_bounds__(256)
__global__ void nv_decode_split12_to_bf16(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    __nv_bfloat16* __restrict__ bf16_out, int M, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    uint8_t sm = sign_mantissa[idx];
    uint8_t gb = groups[idx / 2];
    uint32_t group = (idx & 1) ? (gb >> 4) : (gb & 0xF);
    uint16_t bf16_bits = ((uint16_t)(sm >> 7) << 15) |
                         ((uint16_t)(base_exp + group) << 7) | (sm & 0x7F);
    bf16_out[idx] = *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
}

int nv_launch_split12_cublas_batch_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, int act_stride,
    void* output, int out_stride,
    void* bf16_weight_buf, int buf_half_elems,
    int M, int K, int B, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    if (!s_decode_stream) {
        cudaStreamCreateWithFlags(&s_decode_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&s_decode_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s_gemm_done, cudaEventDisableTiming);
    }
    if (!s_cublas_handle) cublasCreate(&s_cublas_handle);

    __nv_bfloat16* buf0 = (__nv_bfloat16*)bf16_weight_buf;
    __nv_bfloat16* cur_buf = s_ping ? (buf0 + buf_half_elems) : buf0;

    cudaStreamWaitEvent(s_decode_stream, s_gemm_done, 0);
    int total = M * K, threads = 256, blocks = (total + threads - 1) / threads;
    nv_decode_split12_to_bf16<<<blocks, threads, 0, s_decode_stream>>>(
        (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
        cur_buf, M, K);
    cudaEventRecord(s_decode_done, s_decode_stream);
    cudaStreamWaitEvent(s, s_decode_done, 0);

    cublasSetStream(s_cublas_handle, s);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(s_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, B, K, &alpha,
        cur_buf, CUDA_R_16BF, K, activations, CUDA_R_16BF, act_stride, &beta,
        output, CUDA_R_32F, out_stride, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(s_gemm_done, s);
    s_ping ^= 1;
    return 0;
}

}
