/**
 * Split12 Fused Decode+GEMM — ZipServ-derived architecture (Apache 2.0)
 * Adapted from ZipServ (arXiv 2603.17435) for split12 format.
 *
 * Key optimizations from ZipServ:
 *   1. K-slice interleaving: decode(i+1) overlaps with mma(i)
 *   2. Register double-buffering for A and B fragments
 *   3. Single __syncthreads per K-tile (not per K-slice)
 *
 * Our split12 decode: 3 ALU ops vs ZipServ's bitmap+popcount
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define S12_TILE_M 128      // 8 warp rows x 16 — 2x compute parallelism
#define S12_N_WARPS 8       // 256 threads — 2x loading + compute throughput
#define S12_BLOCK (S12_N_WARPS * 32)

// ============================================================
// PTX helpers + cp.async for DRAM->shared memory pipeline
// ============================================================

// cp.async: copy 16 bytes from global to shared without touching registers
__device__ __forceinline__
void cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(global_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__
void mma_m16n8k16(uint32_t* C, const uint32_t* A, const uint32_t* B) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        :"=r"(C[0]),"=r"(C[1]),"=r"(C[2]),"=r"(C[3])
        :"r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),
         "r"(B[0]),"r"(B[1]),"r"(C[0]),"r"(C[1]),"r"(C[2]),"r"(C[3]));
}

__device__ __forceinline__
uint32_t decode_split12_pair(uint8_t sm0, uint8_t sm1, uint8_t g0, uint8_t g1, int be) {
    uint16_t b0 = ((uint16_t)(sm0>>7)<<15)|((uint16_t)(be+g0)<<7)|(sm0&0x7F);
    uint16_t b1 = ((uint16_t)(sm1>>7)<<15)|((uint16_t)(be+g1)<<7)|(sm1&0x7F);
    return ((uint32_t)b1<<16)|b0;
}

__device__ __forceinline__
__nv_bfloat16 decode_split12_bf16(uint8_t sm, uint8_t group, int base_exp) {
    uint16_t bf16_bits = ((uint16_t)(sm >> 7) << 15) |
                         ((uint16_t)(base_exp + group) << 7) | (sm & 0x7F);
    return *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
}

// ============================================================
// Decode one K-slice of A into register buffer
// ============================================================
__device__ __forceinline__
void decode_a_slice(uint32_t a[4],
    const uint8_t* sm, const uint8_t* gr,
    int r0, int r1, int twg, int kk, int be, int tile_k)
{
    int k0 = kk + twg * 2, k1 = kk + 8 + twg * 2;
    // Vectorized uint32 reads from shared memory
    const uint32_t* s0 = (const uint32_t*)(sm + r0 * tile_k);
    const uint32_t* s1 = (const uint32_t*)(sm + r1 * tile_k);
    uint32_t w0r0=s0[k0/4], w0r1=s1[k0/4], w1r0=s0[k1/4], w1r1=s1[k1/4];
    int h0=(k0%4)*8, h1=(k1%4)*8;
    const uint32_t* g0=(const uint32_t*)(gr+r0*(tile_k/2));
    const uint32_t* g1=(const uint32_t*)(gr+r1*(tile_k/2));
    uint32_t gw0r0=g0[k0/8],gw0r1=g1[k0/8],gw1r0=g0[k1/8],gw1r1=g1[k1/8];
    int gh0=((k0/2)%4)*8, gh1=((k1/2)%4)*8;

    a[0]=decode_split12_pair((w0r0>>h0)&0xFF,(w0r0>>(h0+8))&0xFF,
        (gw0r0>>gh0)&0xF,(gw0r0>>(gh0+4))&0xF,be);
    a[1]=decode_split12_pair((w0r1>>h0)&0xFF,(w0r1>>(h0+8))&0xFF,
        (gw0r1>>gh0)&0xF,(gw0r1>>(gh0+4))&0xF,be);
    a[2]=decode_split12_pair((w1r0>>h1)&0xFF,(w1r0>>(h1+8))&0xFF,
        (gw1r0>>gh1)&0xF,(gw1r0>>(gh1+4))&0xF,be);
    a[3]=decode_split12_pair((w1r1>>h1)&0xFF,(w1r1>>(h1+8))&0xFF,
        (gw1r1>>gh1)&0xF,(gw1r1>>(gh1+4))&0xF,be);
}

// ============================================================
// Main kernel: ZipServ-style interleaved decode + mma
// Parameterized TILE_K for flexibility (64 or 128)
// ============================================================
template<int S12_TILE_N, int WCT, int S12_TILE_K = 64>
__launch_bounds__(S12_BLOCK)
__global__ void split12_fused_gemm(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const __nv_bfloat16* __restrict__ B_matrix,
    float* __restrict__ C_matrix,
    int M, int K, int N, int out_stride,
    // Optional inline patch correction (NULL to skip)
    const int32_t* __restrict__ patch_row_off,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ patch_correct,
    const int16_t* __restrict__ patch_wrong)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;
    // Grid: x=N-tiles (fast), y=M-tiles (slow)
    // All N-tiles for the same M-tile dispatch together → weight data reused in L2
    const int block_m = blockIdx.y * S12_TILE_M;
    const int block_n = blockIdx.x * S12_TILE_N;

    extern __shared__ __align__(128) char smem[];
    // Layout: sm_load(2xMxK) + gr_load(2xMxK/2) + B(2xKxNx2)
    uint8_t* sm_buf = (uint8_t*)smem;                                          // 2xTILE_MxTILE_K
    uint8_t* gr_buf = sm_buf + S12_TILE_M * S12_TILE_K * 2;                   // 2xTILE_MxTILE_K/2
    __nv_bfloat16* B_buf = (__nv_bfloat16*)(gr_buf + S12_TILE_M * S12_TILE_K / 2 * 2); // 2xTILE_KxTILE_Nx2

    // Register double-buffers (ZipServ pattern)
    uint32_t a[2][4];          // 2 ping-pong buffers for A fragment
    uint32_t b[WCT][2][2];    // [n_tile][2 buffers][2 regs]

    // Accumulators
    uint32_t c[WCT][4];
    #pragma unroll
    for (int j = 0; j < WCT; j++) c[j][0]=c[j][1]=c[j][2]=c[j][3]=0;

    int r0 = warp_id*MMA_M + lane_id/4;
    int r1 = warp_id*MMA_M + 8 + lane_id/4;
    int twg = lane_id % 4;
    int bg = lane_id/4, bt = lane_id%4;

    // cp.async tile load macros
    #define VL_SM(dst, ks) do { \
        for(int _i=tid;_i<S12_TILE_M*S12_TILE_K/16;_i+=S12_BLOCK){ \
        int _b=_i*16,_m=_b/S12_TILE_K,_k=_b%S12_TILE_K; \
        cp_async_16((uint8_t*)(dst)+_i*16, sign_mantissa+(int64_t)(block_m+_m)*K+(ks)+_k);} }while(0)
    #define VL_GR(dst, ks) do { \
        for(int _i=tid;_i<S12_TILE_M*S12_TILE_K/2/16;_i+=S12_BLOCK){ \
        int _b=_i*16,_m=_b/(S12_TILE_K/2),_k=_b%(S12_TILE_K/2); \
        cp_async_16((uint8_t*)(dst)+_i*16, groups+(int64_t)(block_m+_m)*K/2+(ks)/2+_k);} }while(0)
    #define VL_B(dst, ks) do { \
        for(int _i=tid;_i<S12_TILE_K*S12_TILE_N*2/16;_i+=S12_BLOCK){ \
        int _b=_i*16,_e=_b/2,_n=_e/S12_TILE_K,_k=_e%S12_TILE_K; \
        cp_async_16((uint8_t*)(dst)+_i*16, B_matrix+(int64_t)(block_n+_n)*K+(ks)+_k);} }while(0)

    // Load first tile via cp.async
    VL_SM(sm_buf, 0); VL_GR(gr_buf, 0); VL_B(B_buf, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // Pre-decode slice 0 into buffer 0
    uint8_t* sm_r = sm_buf;
    uint8_t* gr_r = gr_buf;
    __nv_bfloat16* B_r = B_buf;
    decode_a_slice(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp, S12_TILE_K);
    #pragma unroll
    for(int n=0;n<WCT;n++){
        const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*S12_TILE_K;
        b[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
        b[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
    }

    // K-tile loop with ZipServ interleaving
    #pragma unroll(1)
    for (int tk = 0; tk < K / S12_TILE_K; tk++) {
        int br = tk % 2, bw = 1 - br;

        // cp.async load next tile (runs in background during compute below!)
        if (tk + 1 < K / S12_TILE_K) {
            int kn = (tk+1)*S12_TILE_K;
            VL_SM(sm_buf+bw*S12_TILE_M*S12_TILE_K, kn);
            VL_GR(gr_buf+bw*S12_TILE_M*S12_TILE_K/2, kn);
            VL_B(B_buf+bw*S12_TILE_K*S12_TILE_N, kn);
            cp_async_commit();
        }

        sm_r = sm_buf + br*S12_TILE_M*S12_TILE_K;
        gr_r = gr_buf + br*S12_TILE_M*S12_TILE_K/2;
        B_r = B_buf + br*S12_TILE_K*S12_TILE_N;

        // === GENERALIZED INTERLEAVED K-slices (ZipServ pattern) ===
        #define N_SLICES (S12_TILE_K / MMA_K)

        #pragma unroll
        for (int ks = 1; ks < N_SLICES; ks++) {
            int cur = ks % 2, prev = 1 - cur;
            // Decode slice ks into buf[cur]
            decode_a_slice(a[cur], sm_r, gr_r, r0, r1, twg, ks*MMA_K, base_exp, S12_TILE_K);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*S12_TILE_K+ks*MMA_K;
                b[n][cur][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b[n][cur][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
            // MMA slice ks-1 from buf[prev]
            #pragma unroll
            for(int n=0;n<WCT;n++) mma_m16n8k16(c[n], a[prev], b[n][prev]);
        }

        // MMA last slice (overlaps with cp.async in background)
        {
            int last = (N_SLICES - 1) % 2;
            #pragma unroll
            for(int n=0;n<WCT;n++) mma_m16n8k16(c[n], a[last], b[n][last]);
        }
        #undef N_SLICES

        // Wait for cp.async, sync
        cp_async_wait<0>();
        __syncthreads();

        // Pre-decode slice 0 of next tile
        if (tk + 1 < K / S12_TILE_K) {
            sm_r = sm_buf + bw*S12_TILE_M*S12_TILE_K;
            gr_r = gr_buf + bw*S12_TILE_M*S12_TILE_K/2;
            B_r = B_buf + bw*S12_TILE_K*S12_TILE_N;
            decode_a_slice(a[0], sm_r, gr_r, r0, r1, twg, 0, base_exp, S12_TILE_K);
            #pragma unroll
            for(int n=0;n<WCT;n++){
                const __nv_bfloat16* bp=B_r+(n*MMA_N+bg)*S12_TILE_K;
                b[n][0][0]=*reinterpret_cast<const uint32_t*>(&bp[bt*2]);
                b[n][0][1]=*reinterpret_cast<const uint32_t*>(&bp[8+bt*2]);
            }
        }
    }
    #undef VL_SM
    #undef VL_GR
    #undef VL_B

    // Output: transpose via shared memory for coalesced global writes
    // Without this, each thread writes to b*out_stride+m — scattered (stride>>128B)
    // With transpose: consecutive threads write consecutive m values — perfect coalescing
    {
        // Repurpose shared memory as output buffer (K-tile data no longer needed)
        // Pad stride by +1 to avoid 32-way bank conflicts on read
        const int OUT_PAD = S12_TILE_N + 1;
        float* out_smem = (float*)smem;  // 128 × (TILE_N+1) × 4 bytes

        int cg = lane_id / 4, ct = lane_id % 4;

        // Step 1: write MMA results to shared memory
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

        // Step 2: coalesced write to global — consecutive threads write consecutive m
        for (int idx = tid; idx < S12_TILE_M * S12_TILE_N; idx += S12_BLOCK) {
            int m = idx % S12_TILE_M;
            int b = idx / S12_TILE_M;
            int gm = block_m + m, gb = block_n + b;
            if (gm < M && gb < N) {
                float val = out_smem[m * OUT_PAD + b];
                // Inline patch correction (fused — no separate kernel launch)
                if (patch_row_off) {
                    int ps = patch_row_off[gm], pe = patch_row_off[gm + 1];
                    for (int p = ps; p < pe; p++) {
                        union { uint32_t u; float f; } cv, wv, av;
                        cv.u = ((uint32_t)(uint16_t)patch_correct[p]) << 16;
                        wv.u = ((uint32_t)(uint16_t)patch_wrong[p]) << 16;
                        av.u = ((uint32_t)(uint16_t)((const int16_t*)B_matrix)[
                            (int64_t)gb * K + patch_cols[p]]) << 16;
                        val += (cv.f - wv.f) * av.f;
                    }
                }
                C_matrix[gb * out_stride + gm] = val;
            }
        }
    }
}
