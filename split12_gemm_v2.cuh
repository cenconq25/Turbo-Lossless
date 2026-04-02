/**
 * Split12 Fused Decode+GEMM v2 — High-occupancy kernel
 *
 * Key changes from v1:
 *   - 4 warps (128 threads) instead of 8 (256) — 2-3x more blocks per SM
 *   - TILE_M=64 instead of 128 — half the shared memory per block
 *   - ~15-20 KB smem → fits 2-3 blocks per SM → better bandwidth utilization
 *   - Same K-slice interleaving, cp.async pipeline, mma.sync
 *
 * ZipServ achieves 2.21x over cuBLAS with this exact config (4 warps, TM=64, TK=64).
 * Our simpler decode (3 ALU ops vs bitmap+popcount) should match or beat them.
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define V2_TILE_M 64
#define V2_TILE_K 64
#define V2_N_WARPS 4
#define V2_BLOCK (V2_N_WARPS * 32)  // 128 threads

// cp.async helpers (shared with v1)
__device__ __forceinline__
void v2_cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(global_ptr));
}
__device__ __forceinline__ void v2_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void v2_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__
void v2_mma_m16n8k16(uint32_t* C, const uint32_t* A, const uint32_t* B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=r"(C[0]),"=r"(C[1]),"=r"(C[2]),"=r"(C[3])
        :"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
         "r"(B[0]),"r"(B[1]),"r"(C[0]),"r"(C[1]),"r"(C[2]),"r"(C[3]));
}

__device__ __forceinline__
uint32_t v2_decode_pair(uint8_t sm0, uint8_t sm1, uint8_t g0, uint8_t g1, int be) {
    uint16_t b0 = ((uint16_t)(sm0>>7)<<15)|((uint16_t)(be+g0)<<7)|(sm0&0x7F);
    uint16_t b1 = ((uint16_t)(sm1>>7)<<15)|((uint16_t)(be+g1)<<7)|(sm1&0x7F);
    return ((uint32_t)b1<<16)|b0;
}

// Decode one K-slice of A (weight) into MMA register fragment
__device__ __forceinline__
void v2_decode_a(uint32_t a[4],
    const uint8_t* sm, const uint8_t* gr,
    int r0, int r1, int twg, int kk, int be)
{
    int k0 = kk + twg * 2, k1 = kk + 8 + twg * 2;
    const uint32_t* s0 = (const uint32_t*)(sm + r0 * V2_TILE_K);
    const uint32_t* s1 = (const uint32_t*)(sm + r1 * V2_TILE_K);
    uint32_t w0r0=s0[k0/4], w0r1=s1[k0/4], w1r0=s0[k1/4], w1r1=s1[k1/4];
    int h0=(k0%4)*8, h1=(k1%4)*8;
    const uint32_t* g0=(const uint32_t*)(gr+r0*(V2_TILE_K/2));
    const uint32_t* g1=(const uint32_t*)(gr+r1*(V2_TILE_K/2));
    uint32_t gw0r0=g0[k0/8],gw0r1=g1[k0/8],gw1r0=g0[k1/8],gw1r1=g1[k1/8];
    int gh0=((k0/2)%4)*8, gh1=((k1/2)%4)*8;

    a[0]=v2_decode_pair((w0r0>>h0)&0xFF,(w0r0>>(h0+8))&0xFF,
        (gw0r0>>gh0)&0xF,(gw0r0>>(gh0+4))&0xF,be);
    a[1]=v2_decode_pair((w0r1>>h0)&0xFF,(w0r1>>(h0+8))&0xFF,
        (gw0r1>>gh0)&0xF,(gw0r1>>(gh0+4))&0xF,be);
    a[2]=v2_decode_pair((w1r0>>h1)&0xFF,(w1r0>>(h1+8))&0xFF,
        (gw1r0>>gh1)&0xF,(gw1r0>>(gh1+4))&0xF,be);
    a[3]=v2_decode_pair((w1r1>>h1)&0xFF,(w1r1>>(h1+8))&0xFF,
        (gw1r1>>gh1)&0xF,(gw1r1>>(gh1+4))&0xF,be);
}

// ============================================================
// V2 kernel: 4 warps, TILE_M=64, high-occupancy
// ============================================================
template<int V2_TILE_N, int WCT>
__launch_bounds__(V2_BLOCK, 3)  // hint: 3 blocks per SM target
__global__ void split12_fused_gemm_v2(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const __nv_bfloat16* __restrict__ B_matrix,
    float* __restrict__ C_matrix,
    int M, int K, int N, int out_stride,
    const int32_t* __restrict__ patch_row_off,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ patch_correct,
    const int16_t* __restrict__ patch_wrong)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    const int block_m = blockIdx.y * V2_TILE_M;
    const int block_n = blockIdx.x * V2_TILE_N;

    extern __shared__ __align__(128) char smem[];
    // Layout: sm(2×64×64) + gr(2×64×32) + B(2×64×N×2)
    // Total for TN=32: 8+4+8 = 20 KB → fits 2-3 blocks per SM
    uint8_t* sm_buf = (uint8_t*)smem;
    uint8_t* gr_buf = sm_buf + V2_TILE_M * V2_TILE_K * 2;
    __nv_bfloat16* B_buf = (__nv_bfloat16*)(gr_buf + V2_TILE_M * V2_TILE_K / 2 * 2);

    uint32_t a[2][4];
    uint32_t b[WCT][2][2];
    uint32_t c[WCT][4];
    #pragma unroll
    for (int j = 0; j < WCT; j++) c[j][0]=c[j][1]=c[j][2]=c[j][3]=0;

    int r0 = warp_id*MMA_M + lane_id/4;
    int r1 = warp_id*MMA_M + 8 + lane_id/4;
    int twg = lane_id % 4;
    int bg = lane_id/4, bt = lane_id%4;

    // cp.async tile load macros (128 threads — each thread does more loads)
    #define VL2_SM(dst, ks) do { \
        for(int _i=tid;_i<V2_TILE_M*V2_TILE_K/16;_i+=V2_BLOCK){ \
        int _b=_i*16,_m=_b/V2_TILE_K,_k=_b%V2_TILE_K; \
        v2_cp_async_16((uint8_t*)(dst)+_i*16, sign_mantissa+(int64_t)(block_m+_m)*K+(ks)+_k);} }while(0)
    #define VL2_GR(dst, ks) do { \
        for(int _i=tid;_i<V2_TILE_M*V2_TILE_K/2/16;_i+=V2_BLOCK){ \
        int _b=_i*16,_m=_b/(V2_TILE_K/2),_k=_b%(V2_TILE_K/2); \
        v2_cp_async_16((uint8_t*)(dst)+_i*16, groups+(int64_t)(block_m+_m)*K/2+(ks)/2+_k);} }while(0)
    #define VL2_B(dst, ks) do { \
        for(int _i=tid;_i<V2_TILE_K*V2_TILE_N*2/16;_i+=V2_BLOCK){ \
        int _b=_i*16,_e=_b/2,_n=_e/V2_TILE_K,_k=_e%V2_TILE_K; \
        v2_cp_async_16((uint8_t*)(dst)+_i*16, B_matrix+(int64_t)(block_n+_n)*K+(ks)+_k);} }while(0)

    // Load first tile
    VL2_SM(sm_buf, 0); VL2_GR(gr_buf, 0); VL2_B(B_buf, 0);
    v2_cp_async_commit();
    v2_cp_async_wait<0>();
    __syncthreads();

    uint8_t* sm_r = sm_buf;
    uint8_t* gr_r = gr_buf;
    __nv_bfloat16* B_r = B_buf;
    v2_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
    #pragma unroll
    for(int n=0;n<WCT;n++){
        const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*V2_TILE_K;
        b[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
        b[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
    }

    // K-tile loop
    #pragma unroll(1)
    for (int tk = 0; tk < K / V2_TILE_K; tk++) {
        int br = tk % 2, bw = 1 - br;

        if (tk + 1 < K / V2_TILE_K) {
            int kn = (tk+1)*V2_TILE_K;
            VL2_SM(sm_buf+bw*V2_TILE_M*V2_TILE_K, kn);
            VL2_GR(gr_buf+bw*V2_TILE_M*V2_TILE_K/2, kn);
            VL2_B(B_buf+bw*V2_TILE_K*V2_TILE_N, kn);
            v2_cp_async_commit();
        }

        sm_r = sm_buf + br*V2_TILE_M*V2_TILE_K;
        gr_r = gr_buf + br*V2_TILE_M*V2_TILE_K/2;
        B_r = B_buf + br*V2_TILE_K*V2_TILE_N;

        #define N_SLICES (V2_TILE_K / MMA_K)
        #pragma unroll
        for (int ks = 1; ks < N_SLICES; ks++) {
            int cur = ks % 2, prev = 1 - cur;
            v2_decode_a(a[cur], sm_r, gr_r, r0, r1, twg, ks*MMA_K, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*V2_TILE_K+ks*MMA_K;
                b[n][cur][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b[n][cur][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
            #pragma unroll
            for(int n=0;n<WCT;n++) v2_mma_m16n8k16(c[n], a[prev], b[n][prev]);
        }

        {
            int last = (N_SLICES - 1) % 2;
            #pragma unroll
            for(int n=0;n<WCT;n++) v2_mma_m16n8k16(c[n], a[last], b[n][last]);
        }
        #undef N_SLICES

        v2_cp_async_wait<0>();
        __syncthreads();

        if (tk + 1 < K / V2_TILE_K) {
            sm_r = sm_buf + bw*V2_TILE_M*V2_TILE_K;
            gr_r = gr_buf + bw*V2_TILE_M*V2_TILE_K/2;
            B_r = B_buf + bw*V2_TILE_K*V2_TILE_N;
            v2_decode_a(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*V2_TILE_K;
                b[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
        }
    }
    #undef VL2_SM
    #undef VL2_GR
    #undef VL2_B

    // Output with coalesced writes via shared memory transpose
    {
        const int OUT_PAD = V2_TILE_N + 1;
        float* out_smem = (float*)smem;
        int cg = lane_id / 4, ct = lane_id % 4;

        #pragma unroll
        for (int n = 0; n < WCT; n++) {
            float* cf = reinterpret_cast<float*>(c[n]);
            int nb = n * MMA_N;
            int m0 = warp_id * MMA_M + cg, m1 = m0 + 8;
            out_smem[m0 * OUT_PAD + nb + ct*2]     = cf[0];
            out_smem[m0 * OUT_PAD + nb + ct*2 + 1] = cf[1];
            out_smem[m1 * OUT_PAD + nb + ct*2]     = cf[2];
            out_smem[m1 * OUT_PAD + nb + ct*2 + 1] = cf[3];
        }
        __syncthreads();

        for (int idx = tid; idx < V2_TILE_M * V2_TILE_N; idx += V2_BLOCK) {
            int m = idx % V2_TILE_M;
            int b = idx / V2_TILE_M;
            int gm = block_m + m, gb = block_n + b;
            if (gm < M && gb < N)
                C_matrix[gb * out_stride + gm] = out_smem[m * OUT_PAD + b];
        }
    }
}
