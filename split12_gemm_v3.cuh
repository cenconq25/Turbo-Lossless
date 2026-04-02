/**
 * Split12 Fused Decode+GEMM v3 — TMA-accelerated kernel
 * Uses cp.async.bulk.tensor (TMA) for hardware-accelerated data loading.
 * Descriptors passed as device pointers (not __grid_constant__).
 * Same decode+MMA pipeline as V2 (4 warps, TILE_M=64, K-slice interleaving)
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdint>

#define V3_TILE_M 64
#define V3_TILE_K 64
#define V3_N_WARPS 4
#define V3_BLOCK (V3_N_WARPS * 32)

__device__ __forceinline__ void v3_mma(uint32_t* C, const uint32_t* A, const uint32_t* B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=r"(C[0]),"=r"(C[1]),"=r"(C[2]),"=r"(C[3])
        :"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
         "r"(B[0]),"r"(B[1]),"r"(C[0]),"r"(C[1]),"r"(C[2]),"r"(C[3]));
}

__device__ __forceinline__ uint32_t v3_dec(uint8_t s0, uint8_t s1, uint8_t g0, uint8_t g1, int be) {
    uint16_t b0 = ((uint16_t)(s0>>7)<<15)|((uint16_t)(be+g0)<<7)|(s0&0x7F);
    uint16_t b1 = ((uint16_t)(s1>>7)<<15)|((uint16_t)(be+g1)<<7)|(s1&0x7F);
    return ((uint32_t)b1<<16)|b0;
}

__device__ __forceinline__ uint32_t s32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void v3_decode_a(uint32_t a[4],
    const uint8_t* sm, const uint8_t* gr, int r0, int r1, int twg, int kk, int be)
{
    int k0 = kk + twg*2, k1 = kk + 8 + twg*2;
    const uint32_t* s0=(const uint32_t*)(sm+r0*V3_TILE_K);
    const uint32_t* s1=(const uint32_t*)(sm+r1*V3_TILE_K);
    uint32_t w0r0=s0[k0/4],w0r1=s1[k0/4],w1r0=s0[k1/4],w1r1=s1[k1/4];
    int h0=(k0%4)*8,h1=(k1%4)*8;
    const uint32_t* g0=(const uint32_t*)(gr+r0*(V3_TILE_K/2));
    const uint32_t* g1=(const uint32_t*)(gr+r1*(V3_TILE_K/2));
    uint32_t gw0r0=g0[k0/8],gw0r1=g1[k0/8],gw1r0=g0[k1/8],gw1r1=g1[k1/8];
    int gh0=((k0/2)%4)*8,gh1=((k1/2)%4)*8;
    a[0]=v3_dec((w0r0>>h0)&0xFF,(w0r0>>(h0+8))&0xFF,(gw0r0>>gh0)&0xF,(gw0r0>>(gh0+4))&0xF,be);
    a[1]=v3_dec((w0r1>>h0)&0xFF,(w0r1>>(h0+8))&0xFF,(gw0r1>>gh0)&0xF,(gw0r1>>(gh0+4))&0xF,be);
    a[2]=v3_dec((w1r0>>h1)&0xFF,(w1r0>>(h1+8))&0xFF,(gw1r0>>gh1)&0xF,(gw1r0>>(gh1+4))&0xF,be);
    a[3]=v3_dec((w1r1>>h1)&0xFF,(w1r1>>(h1+8))&0xFF,(gw1r1>>gh1)&0xF,(gw1r1>>(gh1+4))&0xF,be);
}

template<int V3_TILE_N, int WCT>
__launch_bounds__(V3_BLOCK, 3)
__global__ void split12_fused_gemm_v3(
    __grid_constant__ const CUtensorMap sm_tma,
    __grid_constant__ const CUtensorMap gr_tma,
    __grid_constant__ const CUtensorMap b_tma,
    int base_exp,
    float* __restrict__ C_matrix,
    int M, int K, int N, int out_stride,
    const int32_t* __restrict__ p_ro, const int32_t* __restrict__ p_co,
    const int16_t* __restrict__ p_cv, const int16_t* __restrict__ p_wv)
{
    const int tid = threadIdx.x, warp_id = tid/32, lane_id = tid&31;
    const int block_m = blockIdx.y * V3_TILE_M, block_n = blockIdx.x * V3_TILE_N;

    extern __shared__ __align__(128) char smem[];
    uint8_t* sm_buf = (uint8_t*)smem;
    uint8_t* gr_buf = sm_buf + V3_TILE_M * V3_TILE_K * 2;
    __nv_bfloat16* B_buf = (__nv_bfloat16*)(gr_buf + V3_TILE_M * V3_TILE_K / 2 * 2);
    constexpr int DSIZE = V3_TILE_M*V3_TILE_K*2 + V3_TILE_M*V3_TILE_K/2*2 + V3_TILE_K*V3_TILE_N*2*2;
    uint64_t* mbar = (uint64_t*)(smem + ((DSIZE+7)&~7));

    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(s32(&mbar[0])), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(s32(&mbar[1])), "r"(1) : "memory");
    }
    __syncthreads();

    uint32_t a[2][4], b_reg[WCT][2][2], c[WCT][4];
    #pragma unroll
    for (int j=0; j<WCT; j++) c[j][0]=c[j][1]=c[j][2]=c[j][3]=0;
    int r0=warp_id*16+lane_id/4, r1=r0+8, twg=lane_id%4, bg=lane_id/4, bt=lane_id%4;

    // TMA load: thread 0 issues, mbarrier tracks completion
    #define V3_LOAD(stg, ks) do { if (tid==0) { \
        uint32_t tx = V3_TILE_M*V3_TILE_K + V3_TILE_M*V3_TILE_K/2 + V3_TILE_K*V3_TILE_N*2; \
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" \
            :: "r"(s32(&mbar[stg])), "r"(tx) : "memory"); \
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" \
            :: "r"(s32(sm_buf+(stg)*V3_TILE_M*V3_TILE_K)), "l"(&sm_tma), "r"((int)(ks)), "r"(block_m), "r"(s32(&mbar[stg])) : "memory"); \
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" \
            :: "r"(s32(gr_buf+(stg)*V3_TILE_M*V3_TILE_K/2)), "l"(&gr_tma), "r"((int)((ks)/2)), "r"(block_m), "r"(s32(&mbar[stg])) : "memory"); \
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" \
            :: "r"(s32((uint8_t*)B_buf+(stg)*V3_TILE_K*V3_TILE_N*2)), "l"(&b_tma), "r"((int)(ks)), "r"(block_n), "r"(s32(&mbar[stg])) : "memory"); \
    } } while(0)

    #define V3_WAIT(stg, ph) do { \
        uint32_t _d=0; while(!_d) { asm volatile("{\n\t .reg .pred P;\n\t" \
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t" \
            "selp.b32 %0, 1, 0, P;\n\t}" \
            : "=r"(_d) : "r"(s32(&mbar[stg])), "r"(ph) : "memory"); } \
    } while(0)

    int ph[2] = {0, 0};
    V3_LOAD(0, 0);
    V3_WAIT(0, ph[0]); ph[0]^=1;
    __syncthreads();

    uint8_t *sm_r=sm_buf, *gr_r=gr_buf; __nv_bfloat16* B_r=B_buf;
    v3_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
    #pragma unroll
    for(int n=0;n<WCT;n++){
        const __nv_bfloat16* bp=B_r+(n*8+bg)*V3_TILE_K;
        b_reg[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
        b_reg[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
    }

    #pragma unroll(1)
    for (int tk=0; tk < K/V3_TILE_K; tk++) {
        int br=tk%2, bw=1-br;
        if (tk+1 < K/V3_TILE_K) V3_LOAD(bw, (tk+1)*V3_TILE_K);

        sm_r=sm_buf+br*V3_TILE_M*V3_TILE_K; gr_r=gr_buf+br*V3_TILE_M*V3_TILE_K/2;
        B_r=B_buf+br*V3_TILE_K*V3_TILE_N;

        #pragma unroll
        for (int ks=1; ks<V3_TILE_K/16; ks++) {
            int cur=ks%2, prev=1-cur;
            v3_decode_a(a[cur], sm_r, gr_r, r0, r1, twg, ks*16, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*8+bg)*V3_TILE_K+ks*16;
                b_reg[n][cur][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b_reg[n][cur][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
            #pragma unroll
            for(int n=0;n<WCT;n++) v3_mma(c[n], a[prev], b_reg[n][prev]);
        }
        { int last=(V3_TILE_K/16-1)%2;
          #pragma unroll
          for(int n=0;n<WCT;n++) v3_mma(c[n], a[last], b_reg[n][last]); }

        if (tk+1 < K/V3_TILE_K) {
            V3_WAIT(bw, ph[bw]); ph[bw]^=1;
            __syncthreads();
            sm_r=sm_buf+bw*V3_TILE_M*V3_TILE_K; gr_r=gr_buf+bw*V3_TILE_M*V3_TILE_K/2;
            B_r=B_buf+bw*V3_TILE_K*V3_TILE_N;
            v3_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*8+bg)*V3_TILE_K;
                b_reg[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b_reg[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
        }
    }
    #undef V3_LOAD
    #undef V3_WAIT

    // Coalesced output
    { __syncthreads();
      const int OP=V3_TILE_N+1; float* os=(float*)smem;
      int cg=lane_id/4, ct=lane_id%4;
      #pragma unroll
      for(int n=0;n<WCT;n++){ float* cf=reinterpret_cast<float*>(c[n]); int nb=n*8;
        int m0=warp_id*16+cg, m1=m0+8;
        os[m0*OP+nb+ct*2]=cf[0]; os[m0*OP+nb+ct*2+1]=cf[1];
        os[m1*OP+nb+ct*2]=cf[2]; os[m1*OP+nb+ct*2+1]=cf[3]; }
      __syncthreads();
      for(int idx=tid;idx<V3_TILE_M*V3_TILE_N;idx+=V3_BLOCK){
        int m=idx%V3_TILE_M, b=idx/V3_TILE_M;
        int gm=block_m+m, gb=block_n+b;
        if(gm<M&&gb<N) C_matrix[gb*out_stride+gm]=os[m*OP+b]; }
    }
}
