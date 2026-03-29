/**
 * Split12 Fused Decode+GEMM v3 — TMA-accelerated kernel for SM120
 * Based on gau-nernst/learn-cuda/02c_matmul_sm120 (96% SOL on RTX 5090)
 *
 * Key patterns from working SM120 TMA code:
 *   - elect.sync for TMA thread election
 *   - fence.mbarrier_init.release.cluster after init
 *   - mbarrier.try_wait.parity.acquire.cta with timeout
 *   - cp.async.bulk.tensor.2d.shared::cta (not ::cluster)
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

__device__ __forceinline__ uint32_t s32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

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

// elect.sync: returns 1 for exactly one thread in the warp (0 for others)
__device__ __forceinline__ int elect_sync() {
    int pred = 0;
    asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xFFFFFFFF;\n @P mov.s32 %0, 1;\n}" : "+r"(pred));
    return pred;
}

// mbarrier helpers matching gau-nernst's working SM120 patterns
__device__ __forceinline__ void mbar_init(int addr, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(int addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(size) : "memory");
}
__device__ __forceinline__ void mbar_wait(int addr, int phase) {
    int ticks = 0x989680;  // ~10ms timeout, same as gau-nernst
    asm volatile("{\n .reg .pred P1;\n LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@!P1 bra.uni LAB_WAIT;\n}" :: "r"(addr), "r"(phase), "r"(ticks));
}

// BUG FIX: must use shared::cluster (not ::cta) for mbarrier completion
__device__ __forceinline__ void tma_g2s(int dst, const void* tmap, int x, int y, int mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

// Swizzle helpers (for future use when swizzle is enabled)
__device__ __forceinline__ int swiz64(int a) { return a ^ ((a >> 3) & 0x60); }
__device__ __forceinline__ int swiz32(int a) { return a ^ ((a >> 3) & 0x20); }
__device__ __forceinline__ int swiz128(int a) { return a ^ ((a >> 3) & 0xE0); }

// Swizzle-aware decode: TMA stores data with per-row XOR offset
// For Swizzle<B,M,S>: XOR offset = ((row_addr >> (M+S)) & ((1<<B)-1)) << M
// All uint32 reads are 4-byte aligned → never cross 8-byte boundary → safe with swizzle
//
// SM 64B swizzle: Swizzle<2,4,3> → offset = ((r >> 1) & 3) << 4  (rows grouped by 2)
// GR 32B swizzle: Swizzle<1,4,3> → offset = ((r >> 2) & 1) << 4  (rows grouped by 4)

__device__ __forceinline__ void v3_decode_a(uint32_t a[4],
    const uint8_t* sm, const uint8_t* gr, int r0, int r1, int twg, int kk, int be)
{
    int k0=kk+twg*2, k1=kk+8+twg*2;

    // Per-row swizzle offsets for SM (64B: Swizzle<2,4,3>)
    int sx0 = ((r0 >> 1) & 3) << 4;  // XOR offset for row r0
    int sx1 = ((r1 >> 1) & 3) << 4;  // XOR offset for row r1

    // Read uint32 from swizzled sm_buf: physical_col = (k & ~3) ^ swiz_offset
    uint32_t w0r0=*(const uint32_t*)(sm + r0*V3_TILE_K + ((k0&~3) ^ sx0));
    uint32_t w0r1=*(const uint32_t*)(sm + r1*V3_TILE_K + ((k0&~3) ^ sx1));
    uint32_t w1r0=*(const uint32_t*)(sm + r0*V3_TILE_K + ((k1&~3) ^ sx0));
    uint32_t w1r1=*(const uint32_t*)(sm + r1*V3_TILE_K + ((k1&~3) ^ sx1));
    int h0=(k0%4)*8,h1=(k1%4)*8;

    // Per-row swizzle for GR (32B: Swizzle<1,4,3>)
    int gx0 = ((r0 >> 2) & 1) << 4;
    int gx1 = ((r1 >> 2) & 1) << 4;

    uint32_t gw0r0=*(const uint32_t*)(gr + r0*(V3_TILE_K/2) + (((k0/2)&~3) ^ gx0));
    uint32_t gw0r1=*(const uint32_t*)(gr + r1*(V3_TILE_K/2) + (((k0/2)&~3) ^ gx1));
    uint32_t gw1r0=*(const uint32_t*)(gr + r0*(V3_TILE_K/2) + (((k1/2)&~3) ^ gx0));
    uint32_t gw1r1=*(const uint32_t*)(gr + r1*(V3_TILE_K/2) + (((k1/2)&~3) ^ gx1));
    int gh0=((k0/2)%4)*8,gh1=((k1/2)%4)*8;

    a[0]=v3_dec((w0r0>>h0)&0xFF,(w0r0>>(h0+8))&0xFF,(gw0r0>>gh0)&0xF,(gw0r0>>(gh0+4))&0xF,be);
    a[1]=v3_dec((w0r1>>h0)&0xFF,(w0r1>>(h0+8))&0xFF,(gw0r1>>gh0)&0xF,(gw0r1>>(gh0+4))&0xF,be);
    a[2]=v3_dec((w1r0>>h1)&0xFF,(w1r0>>(h1+8))&0xFF,(gw1r0>>gh1)&0xF,(gw1r0>>(gh1+4))&0xF,be);
    a[3]=v3_dec((w1r1>>h1)&0xFF,(w1r1>>(h1+8))&0xFF,(gw1r1>>gh1)&0xF,(gw1r1>>(gh1+4))&0xF,be);
}

template<int V3_TILE_N, int WCT>
__launch_bounds__(V3_BLOCK, 2)
__global__ void split12_fused_gemm_v3(
    const CUtensorMap* __restrict__ sm_tma,
    const CUtensorMap* __restrict__ gr_tma,
    const CUtensorMap* __restrict__ b_tma,
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
    constexpr int MBAR_OFF = (DSIZE + 7) & ~7;
    // mbar addresses (as uint32 shared memory offsets for PTX)
    int mbar0 = s32(smem + MBAR_OFF);
    int mbar1 = s32(smem + MBAR_OFF + 8);

    // Init mbarriers — use elect.sync + fence (gau-nernst pattern)
    if (warp_id == 0 && elect_sync()) {
        mbar_init(mbar0, 1);
        mbar_init(mbar1, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    uint32_t a[2][4], b_reg[WCT][2][2], c[WCT][4];
    #pragma unroll
    for (int j=0; j<WCT; j++) c[j][0]=c[j][1]=c[j][2]=c[j][3]=0;
    int r0=warp_id*16+lane_id/4, r1=r0+8, twg=lane_id%4, bg=lane_id/4, bt=lane_id%4;

    // BUG FIX: TMA loads FIRST, then expect_tx LAST (gau-nernst pattern)
    // Also add __syncthreads() before TMA to prevent overwriting in-use smem (Bug 3)
    #define V3_LOAD(mbar_addr, stg, ks) do { \
        if (warp_id == 0 && elect_sync()) { \
            tma_g2s(s32(sm_buf+(stg)*V3_TILE_M*V3_TILE_K), sm_tma, (int)(ks), block_m, mbar_addr); \
            tma_g2s(s32(gr_buf+(stg)*V3_TILE_M*V3_TILE_K/2), gr_tma, (int)((ks)/2), block_m, mbar_addr); \
            tma_g2s(s32((uint8_t*)B_buf+(stg)*V3_TILE_K*V3_TILE_N*2), b_tma, (int)(ks), block_n, mbar_addr); \
            uint32_t tx = V3_TILE_M*V3_TILE_K + V3_TILE_M*V3_TILE_K/2 + V3_TILE_K*V3_TILE_N*2; \
            mbar_expect_tx(mbar_addr, tx); \
        } \
    } while(0)

    int phase = 0;
    V3_LOAD(mbar0, 0, 0);

    // Wait for first tile (warp 0 waits, then sync all)
    if (warp_id == 0) mbar_wait(mbar0, phase);
    __syncthreads();

    // B read helper: 128B swizzle (Swizzle<3,4,3>)
    // B layout: [TILE_N rows][TILE_K bf16 cols] = [N][64] bf16, 128 bytes/row
    // swiz offset = ((row & 7) << 4) — XOR into byte address within row
    #define B_READ(base, row, byte_col) \
        (*(const uint32_t*)((const uint8_t*)(base) + (row)*V3_TILE_K*2 + ((byte_col) ^ (((row)&7)<<4))))

    uint8_t *sm_r=sm_buf, *gr_r=gr_buf; __nv_bfloat16* B_r=B_buf;
    v3_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
    #pragma unroll
    for(int n=0;n<WCT;n++){
        int brow = n*8+bg;
        b_reg[n][0][0]=B_READ(B_r, brow, bt*4);
        b_reg[n][0][1]=B_READ(B_r, brow, 16+bt*4);
    }

    int num_k = K / V3_TILE_K;
    int stage = 0;
    #pragma unroll(1)
    for (int tk=0; tk < num_k; tk++) {
        int next_stage = 1 - stage;
        int next_mbar = (stage == 0) ? mbar1 : mbar0;

        // BUG FIX: sync before TMA to ensure previous compute finished reading smem
        if (tk > 0) __syncthreads();
        if (tk+1 < num_k)
            V3_LOAD(next_mbar, next_stage, (tk+1)*V3_TILE_K);

        sm_r=sm_buf+stage*V3_TILE_M*V3_TILE_K;
        gr_r=gr_buf+stage*V3_TILE_M*V3_TILE_K/2;
        B_r=B_buf+stage*V3_TILE_K*V3_TILE_N;

        #pragma unroll
        for (int ks=1; ks<V3_TILE_K/16; ks++) {
            int cur=ks%2, prev=1-cur;
            v3_decode_a(a[cur], sm_r, gr_r, r0, r1, twg, ks*16, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                int brow = n*8+bg;
                b_reg[n][cur][0]=B_READ(B_r, brow, ks*32+bt*4);
                b_reg[n][cur][1]=B_READ(B_r, brow, ks*32+16+bt*4);
            }
            #pragma unroll
            for(int n=0;n<WCT;n++) v3_mma(c[n], a[prev], b_reg[n][prev]);
        }
        { int last=(V3_TILE_K/16-1)%2;
          #pragma unroll
          for(int n=0;n<WCT;n++) v3_mma(c[n], a[last], b_reg[n][last]); }

        if (tk+1 < num_k) {
            int next_phase = (next_stage == 0 && stage == 1) ? (phase ^ 1) : phase;
            if (warp_id == 0) mbar_wait(next_mbar, next_phase);
            __syncthreads();

            sm_r=sm_buf+next_stage*V3_TILE_M*V3_TILE_K;
            gr_r=gr_buf+next_stage*V3_TILE_M*V3_TILE_K/2;
            B_r=B_buf+next_stage*V3_TILE_K*V3_TILE_N;
            v3_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                int brow = n*8+bg;
                b_reg[n][0][0]=B_READ(B_r, brow, bt*4);
                b_reg[n][0][1]=B_READ(B_r, brow, 16+bt*4);
            }
        }

        stage = next_stage;
        if (stage == 0) phase ^= 1;
    }
    #undef V3_LOAD

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
