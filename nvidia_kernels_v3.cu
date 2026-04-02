/**
 * V3 TMA kernel — separate compilation unit
 * Compiled independently to avoid interfering with V1/V2 binary.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "split12_gemm_v3.cuh"

extern "C" {

int nv_launch_split12_fused_gemm_v3_async(
    const void* sm, const void* gr, int base_exp,
    const void* act, int act_stride,
    void* out, int out_stride,
    int M, int K, int B, void* stream,
    const void* patch_row_off, const void* patch_cols,
    const void* patch_correct, const void* patch_wrong)
{
    auto launch = [&](auto TN_v, auto WCT_v) {
        constexpr int TN = decltype(TN_v)::value, WCT = decltype(WCT_v)::value;
        dim3 grid((B + TN - 1) / TN, (M + V3_TILE_M - 1) / V3_TILE_M);
        int smem = V3_TILE_M * V3_TILE_K * 2 + V3_TILE_M * V3_TILE_K / 2 * 2
                 + V3_TILE_K * TN * (int)sizeof(__nv_bfloat16) * 2 + 128;
        static bool cfg = false;
        if (!cfg) {
            cudaFuncSetAttribute(split12_fused_gemm_v3<TN, WCT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cfg = true;
        }

        alignas(64) CUtensorMap sm_desc, gr_desc, b_desc;
        {
            cuuint64_t sd[2]={(cuuint64_t)K,(cuuint64_t)M}, ss[1]={(cuuint64_t)K};
            cuuint32_t sb[2]={V3_TILE_K, V3_TILE_M}, se[2]={1,1};
            // 64B swizzle eliminates 4-way bank conflicts on sm reads
            CUresult r = cuTensorMapEncodeTiled(&sm_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                (void*)sm, sd, ss, sb, se,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r) { fprintf(stderr, "V3 SM desc err: %d\n", (int)r); return -1; }

            cuuint64_t gd[2]={(cuuint64_t)(K/2),(cuuint64_t)M}, gs[1]={(cuuint64_t)(K/2)};
            cuuint32_t gb[2]={V3_TILE_K/2, V3_TILE_M};
            // 32B swizzle for gr (32 bytes/row)
            r = cuTensorMapEncodeTiled(&gr_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                (void*)gr, gd, gs, gb, se,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r) { fprintf(stderr, "V3 GR desc err: %d\n", (int)r); return -1; }

            cuuint64_t bd[2]={(cuuint64_t)K,(cuuint64_t)B}, bs[1]={(cuuint64_t)(K*2)};
            cuuint32_t bb[2]={V3_TILE_K,(cuuint32_t)TN};
            // 128B swizzle for B (128 bytes/row = 64 bf16 elements)
            r = cuTensorMapEncodeTiled(&b_desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                (void*)act, bd, bs, bb, se,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (r) { fprintf(stderr, "V3 B desc err: %d\n", (int)r); return -1; }
        }

        // Copy descriptors to device (cross-CU __grid_constant__ crashes on SM120)
        static CUtensorMap *d_sd=0,*d_gd=0,*d_bd=0;
        if(!d_sd){cudaMalloc(&d_sd,sizeof(CUtensorMap));cudaMalloc(&d_gd,sizeof(CUtensorMap));cudaMalloc(&d_bd,sizeof(CUtensorMap));}
        cudaMemcpyAsync(d_sd,&sm_desc,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        cudaMemcpyAsync(d_gd,&gr_desc,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        cudaMemcpyAsync(d_bd,&b_desc,sizeof(CUtensorMap),cudaMemcpyHostToDevice,(cudaStream_t)stream);
        split12_fused_gemm_v3<TN, WCT><<<grid, V3_BLOCK, smem, (cudaStream_t)stream>>>(
            d_sd, d_gd, d_bd, base_exp,
            (float*)out, M, K, B, out_stride,
            (const int32_t*)patch_row_off, (const int32_t*)patch_cols,
            (const int16_t*)patch_correct, (const int16_t*)patch_wrong);
    };

    if (B >= 128) launch(std::integral_constant<int,64>{}, std::integral_constant<int,8>{});
    else if (B >= 32) launch(std::integral_constant<int,32>{}, std::integral_constant<int,4>{});
    else launch(std::integral_constant<int,16>{}, std::integral_constant<int,2>{});
    return 0;
}

}
