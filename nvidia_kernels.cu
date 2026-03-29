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
        uint8_t sm0 = __ldg(&sm_ptr[0]), sm1 = __ldg(&sm_ptr[256]), sm2 = __ldg(&sm_ptr[512]), sm3 = __ldg(&sm_ptr[768]);
        uint8_t gb0 = __ldg(&gr_ptr[0]), gb1 = __ldg(&gr_ptr[128]), gb2 = __ldg(&gr_ptr[256]), gb3 = __ldg(&gr_ptr[384]);
        uint32_t g0 = is_odd ? (gb0>>4) : (gb0&0xF), g1 = is_odd ? (gb1>>4) : (gb1&0xF);
        uint32_t g2 = is_odd ? (gb2>>4) : (gb2&0xF), g3 = is_odd ? (gb3>>4) : (gb3&0xF);
        union { uint32_t u; float f; } c0, c1, c2, c3;
        c0.u = (((uint32_t)(sm0>>7))<<15 | ((uint32_t)(base_exp+g0))<<7 | (sm0&0x7F)) << 16;
        c1.u = (((uint32_t)(sm1>>7))<<15 | ((uint32_t)(base_exp+g1))<<7 | (sm1&0x7F)) << 16;
        c2.u = (((uint32_t)(sm2>>7))<<15 | ((uint32_t)(base_exp+g2))<<7 | (sm2&0x7F)) << 16;
        c3.u = (((uint32_t)(sm3>>7))<<15 | ((uint32_t)(base_exp+g3))<<7 | (sm3&0x7F)) << 16;
        sum0 += c0.f * bf16_to_float(__ldg((const int16_t*)(act_ptr)));
        sum1 += c1.f * bf16_to_float(__ldg((const int16_t*)(act_ptr + 512)));
        sum2 += c2.f * bf16_to_float(__ldg((const int16_t*)(act_ptr + 1024)));
        sum3 += c3.f * bf16_to_float(__ldg((const int16_t*)(act_ptr + 1536)));
        sm_ptr += 1024; gr_ptr += 512; act_ptr += 2048;
    }
    for (; col < K; col += BLOCK_DIM) {
        uint8_t sm = __ldg(&sm_ptr[0]), gb = __ldg(&gr_ptr[0]);
        uint32_t g = is_odd ? (gb>>4) : (gb&0xF);
        union { uint32_t u; float f; } c;
        c.u = (((uint32_t)(sm>>7))<<15 | ((uint32_t)(base_exp+g))<<7 | (sm&0x7F)) << 16;
        sum0 += c.f * bf16_to_float(__ldg((const int16_t*)(act_ptr)));
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
        float a0 = bf16_to_float(__ldg((const int16_t*)(act_ptr)));
        float a1 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 512)));
        float a2 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 1024)));
        float a3 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 1536)));
        #define DECODE4(ptr, grp, s0, s1, s2, s3) { \
            uint8_t _s0=__ldg(&ptr[0]),_s1=__ldg(&ptr[256]),_s2=__ldg(&ptr[512]),_s3=__ldg(&ptr[768]); \
            uint8_t _g0=__ldg(&grp[0]),_g1=__ldg(&grp[128]),_g2=__ldg(&grp[256]),_g3=__ldg(&grp[384]); \
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
        float a = bf16_to_float(__ldg((const int16_t*)(act_ptr)));
        uint8_t s0=__ldg(&sm0p[0]), g0=__ldg(&gr0p[0]); uint32_t e0=is_odd?(g0>>4):(g0&0xF);
        union{uint32_t u;float f;} c0;
        c0.u=((uint32_t)(s0>>7)<<15|(uint32_t)(base_exp+e0)<<7|(s0&0x7F))<<16;
        sa0 += c0.f * a;
        if (has_row1) {
            uint8_t s1=__ldg(&sm1p[0]), g1=__ldg(&gr1p[0]); uint32_t e1=is_odd?(g1>>4):(g1&0xF);
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
// B=1 Dual Matvec (gate+up fused, shared activation reads)
// ============================================================
__launch_bounds__(BLOCK_DIM)
__global__ void nv_split12_matvec_dual(
    const uint8_t* __restrict__ sm_a, const uint8_t* __restrict__ gr_a, int base_exp_a,
    const uint8_t* __restrict__ sm_b, const uint8_t* __restrict__ gr_b, int base_exp_b,
    const int16_t* __restrict__ activations,
    float* __restrict__ output_a, float* __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];  // [0..NW-1] for A, [NW..2*NW-1] for B
    float sa0=0,sa1=0,sa2=0,sa3=0, sb0=0,sb1=0,sb2=0,sb3=0;

    const uint8_t* sma_p = sm_a + (int64_t)row * K + tid;
    const uint8_t* gra_p = gr_a + (int64_t)row * K / 2 + tid / 2;
    const uint8_t* smb_p = sm_b + (int64_t)row * K + tid;
    const uint8_t* grb_p = gr_b + (int64_t)row * K / 2 + tid / 2;
    const char* act_ptr = (const char*)(activations + tid);
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + BLOCK_DIM * 3 < K; col += BLOCK_DIM * 4) {
        float a0 = bf16_to_float(__ldg((const int16_t*)(act_ptr)));
        float a1 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 512)));
        float a2 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 1024)));
        float a3 = bf16_to_float(__ldg((const int16_t*)(act_ptr + 1536)));
        // Decode weight A
        {
            uint8_t _s0=__ldg(&sma_p[0]),_s1=__ldg(&sma_p[256]),_s2=__ldg(&sma_p[512]),_s3=__ldg(&sma_p[768]);
            uint8_t _g0=__ldg(&gra_p[0]),_g1=__ldg(&gra_p[128]),_g2=__ldg(&gra_p[256]),_g3=__ldg(&gra_p[384]);
            uint32_t _e0=is_odd?(_g0>>4):(_g0&0xF), _e1=is_odd?(_g1>>4):(_g1&0xF);
            uint32_t _e2=is_odd?(_g2>>4):(_g2&0xF), _e3=is_odd?(_g3>>4):(_g3&0xF);
            union{uint32_t u;float f;} _c0,_c1,_c2,_c3;
            _c0.u=((uint32_t)(_s0>>7)<<15|(uint32_t)(base_exp_a+_e0)<<7|(_s0&0x7F))<<16;
            _c1.u=((uint32_t)(_s1>>7)<<15|(uint32_t)(base_exp_a+_e1)<<7|(_s1&0x7F))<<16;
            _c2.u=((uint32_t)(_s2>>7)<<15|(uint32_t)(base_exp_a+_e2)<<7|(_s2&0x7F))<<16;
            _c3.u=((uint32_t)(_s3>>7)<<15|(uint32_t)(base_exp_a+_e3)<<7|(_s3&0x7F))<<16;
            sa0+=_c0.f*a0; sa1+=_c1.f*a1; sa2+=_c2.f*a2; sa3+=_c3.f*a3;
        }
        // Decode weight B
        {
            uint8_t _s0=__ldg(&smb_p[0]),_s1=__ldg(&smb_p[256]),_s2=__ldg(&smb_p[512]),_s3=__ldg(&smb_p[768]);
            uint8_t _g0=__ldg(&grb_p[0]),_g1=__ldg(&grb_p[128]),_g2=__ldg(&grb_p[256]),_g3=__ldg(&grb_p[384]);
            uint32_t _e0=is_odd?(_g0>>4):(_g0&0xF), _e1=is_odd?(_g1>>4):(_g1&0xF);
            uint32_t _e2=is_odd?(_g2>>4):(_g2&0xF), _e3=is_odd?(_g3>>4):(_g3&0xF);
            union{uint32_t u;float f;} _c0,_c1,_c2,_c3;
            _c0.u=((uint32_t)(_s0>>7)<<15|(uint32_t)(base_exp_b+_e0)<<7|(_s0&0x7F))<<16;
            _c1.u=((uint32_t)(_s1>>7)<<15|(uint32_t)(base_exp_b+_e1)<<7|(_s1&0x7F))<<16;
            _c2.u=((uint32_t)(_s2>>7)<<15|(uint32_t)(base_exp_b+_e2)<<7|(_s2&0x7F))<<16;
            _c3.u=((uint32_t)(_s3>>7)<<15|(uint32_t)(base_exp_b+_e3)<<7|(_s3&0x7F))<<16;
            sb0+=_c0.f*a0; sb1+=_c1.f*a1; sb2+=_c2.f*a2; sb3+=_c3.f*a3;
        }
        sma_p+=1024; gra_p+=512; smb_p+=1024; grb_p+=512; act_ptr+=2048;
    }
    for (; col < K; col += BLOCK_DIM) {
        float a = bf16_to_float(__ldg((const int16_t*)(act_ptr)));
        {
            uint8_t s=__ldg(&sma_p[0]), g=__ldg(&gra_p[0]); uint32_t e=is_odd?(g>>4):(g&0xF);
            union{uint32_t u;float f;} c;
            c.u=((uint32_t)(s>>7)<<15|(uint32_t)(base_exp_a+e)<<7|(s&0x7F))<<16;
            sa0 += c.f * a;
        }
        {
            uint8_t s=__ldg(&smb_p[0]), g=__ldg(&grb_p[0]); uint32_t e=is_odd?(g>>4):(g&0xF);
            union{uint32_t u;float f;} c;
            c.u=((uint32_t)(s>>7)<<15|(uint32_t)(base_exp_b+e)<<7|(s&0x7F))<<16;
            sb0 += c.f * a;
        }
        sma_p+=256; gra_p+=128; smb_p+=256; grb_p+=128; act_ptr+=512;
    }
    float sumA = sa0+sa1+sa2+sa3, sumB = sb0+sb1+sb2+sb3;
    for (int off=WARP_SIZE/2; off>0; off>>=1) {
        sumA += __shfl_down_sync(0xFFFFFFFF, sumA, off, WARP_SIZE);
        sumB += __shfl_down_sync(0xFFFFFFFF, sumB, off, WARP_SIZE);
    }
    if ((tid&(WARP_SIZE-1))==0) {
        warp_sums[tid/WARP_SIZE] = sumA;
        warp_sums[NUM_WARPS + tid/WARP_SIZE] = sumB;
    }
    __syncthreads();
    if (tid == 0) {
        float sA=0, sB=0;
        for (int w=0; w<NUM_WARPS; w++) { sA+=warp_sums[w]; sB+=warp_sums[NUM_WARPS+w]; }
        output_a[row] = sA;
        output_b[row] = sB;
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

// Flat batched patch correction: 1 thread per (row, batch) pair
// No warp reduction needed — each thread serializes ~2 patches (avg)
// Grid: (ceil(num_nonempty * B / 256)), block: 256 threads
__global__ void nv_apply_patches_batch(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const int32_t* __restrict__ nonempty_rows,
    const int16_t* __restrict__ activations, int act_stride,
    float* __restrict__ output, int out_stride, int num_nonempty, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = idx / B;
    int b = idx % B;
    if (row_idx >= num_nonempty) return;

    int row = nonempty_rows[row_idx];
    int start = row_offsets[row], end = row_offsets[row + 1];
    const int16_t* act = activations + b * act_stride;

    float correction = 0.0f;
    for (int p = start; p < end; p++)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(act[patch_cols[p]]);
    if (correction != 0.0f)
        output[b * out_stride + row] += correction;
}

// ============================================================
// Fused Decode+GEMM via PTX mma.sync (the main batched kernel)
// ============================================================
#include "split12_gemm.cuh"
#include "split12_gemm_v2.cuh"
// V3 TMA kernel — MUST be in same CU (cross-CU __grid_constant__ crashes on SM120)
#include <cuda.h>
#include "split12_gemm_v3.cuh"

static int launch_v3_same_cu(
    const void* sm, const void* gr, int base_exp,
    const void* act, int act_stride, void* out, int out_stride,
    int M, int K, int B, void* stream,
    const void* p_ro, const void* p_co, const void* p_cv, const void* p_wv)
{
    auto launch = [&](auto TN_v, auto WCT_v) {
        constexpr int TN = decltype(TN_v)::value, WCT = decltype(WCT_v)::value;
        dim3 grid((B+TN-1)/TN, (M+V3_TILE_M-1)/V3_TILE_M);
        int smem = V3_TILE_M*V3_TILE_K*2 + V3_TILE_M*V3_TILE_K/2*2 + V3_TILE_K*TN*2*2 + 128;
        static bool c=false; if(!c){cudaFuncSetAttribute(split12_fused_gemm_v3<TN,WCT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,smem);c=true;}
        // Device-resident TMA descriptors
        static CUtensorMap *d_sd=0,*d_gd=0,*d_bd=0;
        if(!d_sd){cudaMalloc(&d_sd,sizeof(CUtensorMap));cudaMalloc(&d_gd,sizeof(CUtensorMap));cudaMalloc(&d_bd,sizeof(CUtensorMap));}
        alignas(64) CUtensorMap sd,gd,bd; cuuint32_t se[2]={1,1};
        cuuint64_t s1[2]={(cuuint64_t)K,(cuuint64_t)M},s2[1]={(cuuint64_t)K};
        cuuint32_t s3[2]={V3_TILE_K,V3_TILE_M};
        cuTensorMapEncodeTiled(&sd,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)sm,s1,s2,s3,se,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuuint64_t g1[2]={(cuuint64_t)(K/2),(cuuint64_t)M},g2[1]={(cuuint64_t)(K/2)};
        cuuint32_t g3[2]={V3_TILE_K/2,V3_TILE_M};
        cuTensorMapEncodeTiled(&gd,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,(void*)gr,g1,g2,g3,se,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_32B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuuint64_t b1[2]={(cuuint64_t)K,(cuuint64_t)B},b2[1]={(cuuint64_t)(K*2)};
        cuuint32_t b3[2]={V3_TILE_K,(cuuint32_t)TN};
        cuTensorMapEncodeTiled(&bd,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)act,b1,b2,b3,se,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cudaMemcpyAsync(d_sd,&sd,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        cudaMemcpyAsync(d_gd,&gd,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        cudaMemcpyAsync(d_bd,&bd,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        split12_fused_gemm_v3<TN,WCT><<<grid,V3_BLOCK,smem,(cudaStream_t)stream>>>(
            d_sd,d_gd,d_bd,base_exp,(float*)out,M,K,B,out_stride,
            (const int32_t*)p_ro,(const int32_t*)p_co,(const int16_t*)p_cv,(const int16_t*)p_wv);
    };
    if(B>=128)launch(std::integral_constant<int,64>{},std::integral_constant<int,8>{});
    else if(B>=32)launch(std::integral_constant<int,32>{},std::integral_constant<int,4>{});
    else launch(std::integral_constant<int,16>{},std::integral_constant<int,2>{});
    return 0;
}

// Keep extern for nvidia_kernels_v3.cu backward compat (disabled)
extern "C" int nv_launch_split12_fused_gemm_v3_async(
    const void* sm, const void* gr, int base_exp,
    const void* act, int act_stride,
    void* out, int out_stride,
    int M, int K, int B, void* stream,
    const void* patch_row_off, const void* patch_cols,
    const void* patch_correct, const void* patch_wrong);

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

int nv_launch_split12_v2_dual_async(
    const void* sm_a, const void* gr_a, int base_exp_a,
    const void* sm_b, const void* gr_b, int base_exp_b,
    const void* act, void* out_a, void* out_b, int M, int K, void* stream)
{
    nv_split12_matvec_dual<<<M, BLOCK_DIM, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)sm_a, (const uint8_t*)gr_a, base_exp_a,
        (const uint8_t*)sm_b, (const uint8_t*)gr_b, base_exp_b,
        (const int16_t*)act, (float*)out_a, (float*)out_b, M, K);
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
    int total_pairs = num_nonempty * B;
    int blocks = (total_pairs + 255) / 256;
    nv_apply_patches_batch<<<blocks, 256, 0, (cudaStream_t)stream>>>(
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
    // DEBUG: this function should be called for every GEMM tensor
    int M, int K, int B, void* stream,
    const void* patch_row_off, const void* patch_cols,
    const void* patch_correct, const void* patch_wrong)
{
    // V2 high-occupancy kernel (4 warps, TILE_M=64, 2-3 blocks/SM)
    auto launch_v2 = [&](auto TN_v, auto WCT_v) {
        constexpr int TN = decltype(TN_v)::value, WCT = decltype(WCT_v)::value;
        dim3 grid((B + TN - 1) / TN, (M + V2_TILE_M - 1) / V2_TILE_M);
        int smem = V2_TILE_M * V2_TILE_K * 2 + V2_TILE_M * V2_TILE_K / 2 * 2
                 + V2_TILE_K * TN * (int)sizeof(__nv_bfloat16) * 2;
        static bool cfg2 = false;
        if (!cfg2 && smem > 49152) {
            cudaFuncSetAttribute(split12_fused_gemm_v2<TN, WCT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cfg2 = true;
        }
        split12_fused_gemm_v2<TN, WCT><<<grid, V2_BLOCK, smem, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const __nv_bfloat16*)act, (float*)out, M, K, B, out_stride,
            (const int32_t*)patch_row_off, (const int32_t*)patch_cols,
            (const int16_t*)patch_correct, (const int16_t*)patch_wrong);
    };

    // Kernel version selection
    static int s_kernel_ver = -1;
    if (s_kernel_ver < 0) {
        const char* e = getenv("TURBO_KERNEL");
        if (e && e[0] == '2') { s_kernel_ver = 2; printf("  GEMM: V2 kernel (cp.async)\n"); }
        else if (e && e[0] == '1') { s_kernel_ver = 1; printf("  GEMM: V1 kernel (8 warps)\n"); }
        else { s_kernel_ver = 3; printf("  GEMM: V3 TMA kernel (auto)\n"); }
    }

    if (s_kernel_ver == 3 && B >= 64) {
        // V3 TMA: faster at B>=64, overhead too high for smaller batches
        return launch_v3_same_cu(
            sm, gr, base_exp, act, act_stride, out, out_stride,
            M, K, B, stream, patch_row_off, patch_cols, patch_correct, patch_wrong);
    }
    if (s_kernel_ver >= 2) {
        if (B >= 128) launch_v2(std::integral_constant<int,64>{}, std::integral_constant<int,8>{});
        else if (B >= 32) launch_v2(std::integral_constant<int,32>{}, std::integral_constant<int,4>{});
        else launch_v2(std::integral_constant<int,16>{}, std::integral_constant<int,2>{});
        return 0;
    }

    // V1 fallback: Adaptive TILE_N and TILE_K
    auto launch = [&](auto TN_v, auto WCT_v, auto TK_v) {
        constexpr int TN = decltype(TN_v)::value, WCT = decltype(WCT_v)::value;
        constexpr int TK = decltype(TK_v)::value;
        // x=N-tiles (fast), y=M-tiles: consecutive blocks share weight data → L2 reuse
        dim3 grid((B + TN - 1) / TN, (M + S12_TILE_M - 1) / S12_TILE_M);
        int smem = S12_TILE_M * TK * 2              // sm_buf (double-buffered)
                 + S12_TILE_M * TK / 2 * 2          // gr_buf (double-buffered)
                 + TK * TN * (int)sizeof(__nv_bfloat16) * 2; // B_buf (double-buffered)
        // Request extended shared memory if needed (>48 KB)
        static bool configured = false;
        if (!configured && smem > 49152) {
            cudaFuncSetAttribute(split12_fused_gemm<TN, WCT, TK>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            configured = true;
        }
        split12_fused_gemm<TN, WCT, TK><<<grid, S12_BLOCK, smem, (cudaStream_t)stream>>>(
            (const uint8_t*)sm, (const uint8_t*)gr, base_exp,
            (const __nv_bfloat16*)act, (float*)out, M, K, B, out_stride,
            (const int32_t*)patch_row_off, (const int32_t*)patch_cols,
            (const int16_t*)patch_correct, (const int16_t*)patch_wrong);
    };
    if (B >= 128)
        launch(std::integral_constant<int,64>{}, std::integral_constant<int,8>{}, std::integral_constant<int,64>{});
    else if (B >= 32)
        launch(std::integral_constant<int,32>{}, std::integral_constant<int,4>{}, std::integral_constant<int,64>{});
    else
        launch(std::integral_constant<int,16>{}, std::integral_constant<int,2>{}, std::integral_constant<int,64>{});

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
