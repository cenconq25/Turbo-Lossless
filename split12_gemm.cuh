/**
 * Split12 Fused Decode+GEMM Kernel — ZipGEMM-inspired
 *
 * Decodes split12 compressed weights directly into tensor core registers.
 * NO DRAM round-trip: reads 1.5 bytes/weight from DRAM → decodes in registers → mma.sync
 *
 * Adapted from ZipServ (arXiv 2603.17435) for our split12 format:
 *   split12: [sign 1][mantissa 7] (1 byte) + [group 4] (nibble) = 12 bits
 *   Decode: exponent = base_exp + group (1 ADD, no lookup)
 *
 * Key: our format is SIMPLER than TCA-TBE (no bitmaps, no popcount, no conditional paths)
 *
 * Hardware: sm_80+ (Ampere/Hopper/Ada/Blackwell) with BF16 tensor cores
 */

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

// MMA dimensions
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Tile dimensions
#define S12_TILE_M 64      // 4 warp rows × 16
#define S12_TILE_K 64      // 4 K-slices × 16 (MMA_K) — optimal for 48 KB smem
#define S12_TILE_N_MAX 64  // max batch tile (adaptive)
#define S12_N_WARPS 4      // warps per block
#define S12_BLOCK (S12_N_WARPS * 32)  // 128 threads

// ============================================================
// PTX Inline Assembly Helpers
// ============================================================

// mma.sync.aligned.m16n8k16: BF16 → FP32 accumulation
__device__ __forceinline__
void mma_m16n8k16(uint32_t* C, const uint32_t* A, const uint32_t* B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{ %0, %1, %2, %3 }, "
        "{ %4, %5, %6, %7 }, "
        "{ %8, %9 }, "
        "{ %10, %11, %12, %13 };\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
}

// cp.async: async copy 16 bytes from global to shared
__device__ __forceinline__
void cp_async_16(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ldmatrix: load 8×8 BF16 matrix from shared memory → 4 registers
__device__ __forceinline__
void ldmatrix_x4(uint32_t* regs, const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

// ============================================================
// Split12 decode: sign_mantissa byte + group nibble → BF16 pair in register
// ============================================================
__device__ __forceinline__
uint32_t decode_split12_pair(uint8_t sm0, uint8_t sm1, uint8_t group0, uint8_t group1, int base_exp) {
    // Decode two BF16 values and pack into one uint32_t (bf16x2)
    uint16_t bf16_0 = ((uint16_t)(sm0 >> 7) << 15) |
                      ((uint16_t)(base_exp + group0) << 7) |
                      (sm0 & 0x7F);
    uint16_t bf16_1 = ((uint16_t)(sm1 >> 7) << 15) |
                      ((uint16_t)(base_exp + group1) << 7) |
                      (sm1 & 0x7F);
    return ((uint32_t)bf16_1 << 16) | bf16_0;
}

// ============================================================
// Main Fused Decode+GEMM Kernel
//
// C[M,N] = W[M,K] × A[K,N] where W is split12 compressed
// Grid: (ceil(M/TILE_M), ceil(N/S12_TILE_N))
// Block: 128 threads (4 warps)
// ============================================================
template<int S12_TILE_N, int WARP_COL_TENSORS>
__launch_bounds__(S12_BLOCK)
__global__ void split12_fused_gemm(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const __nv_bfloat16* __restrict__ B_matrix,  // [N * K] activations (batch-major)
    float* __restrict__ C_matrix,                // [N * out_stride] output (batch-major)
    int M, int K, int N, int out_stride)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;

    // Block position
    const int block_m = blockIdx.x * S12_TILE_M;
    const int block_n = blockIdx.y * S12_TILE_N;
    const int tile_m_start = block_m;
    const int rows_this_tile = min(S12_TILE_M, M - block_m);
    const int cols_this_tile = min((int)S12_TILE_N, N - block_n);

    // Shared memory: double-buffered weight data + B matrix
    extern __shared__ __align__(128) char smem_raw[];
    // Layout: [sm_buf0][sm_buf1][gr_buf0][gr_buf1][B_buf0][B_buf1]
    uint8_t* sm_buf = (uint8_t*)smem_raw;
    // sm: S12_TILE_M × S12_TILE_K × 2 buffers = 64×64×2 = 8192 bytes
    uint8_t* gr_buf = sm_buf + S12_TILE_M * S12_TILE_K * 2;
    // gr: S12_TILE_M × S12_TILE_K / 2 × 2 buffers = 64×32×2 = 4096 bytes
    __nv_bfloat16* B_buf = (__nv_bfloat16*)(gr_buf + S12_TILE_M * S12_TILE_K / 2 * 2);
    // B: S12_TILE_K × S12_TILE_N × 2 buffers
    // NO w_decoded buffer needed — direct decode to PTX registers!

    // PTX m16n8k16 accumulators: 4 float32 (=4 uint32) per N-tile
    // For WARP_COL_TENSORS=2: 2 accumulators covering N=16 (two m16n8k16 ops)
    uint32_t c_regs[WARP_COL_TENSORS][4];
    #pragma unroll
    for (int j = 0; j < WARP_COL_TENSORS; j++)
        c_regs[j][0] = c_regs[j][1] = c_regs[j][2] = c_regs[j][3] = 0;

    const int num_k_tiles = (K + S12_TILE_K - 1) / S12_TILE_K;

    // Load first K-tile into buffer 0
    // Cooperatively load sign_mantissa and groups
    {
        const int k_start = 0;
        for (int i = tid; i < S12_TILE_M * S12_TILE_K; i += S12_BLOCK) {
            int m_local = i / S12_TILE_K;
            int k_local = i % S12_TILE_K;
            int row = block_m + m_local;
            int col = k_start + k_local;
            if (row < M && col < K)
                sm_buf[i] = sign_mantissa[(int64_t)row * K + col];
            else
                sm_buf[i] = 0;
        }
        for (int i = tid; i < S12_TILE_M * S12_TILE_K / 2; i += S12_BLOCK) {
            int m_local = i / (S12_TILE_K / 2);
            int k_local = i % (S12_TILE_K / 2);
            int row = block_m + m_local;
            int col = k_start + k_local * 2;
            if (row < M && col < K)
                gr_buf[i] = groups[(int64_t)row * K / 2 + col / 2];
            else
                gr_buf[i] = 0;
        }
        // Load B matrix tile
        for (int i = tid; i < S12_TILE_K * S12_TILE_N; i += S12_BLOCK) {
            int k_local = i / S12_TILE_N;
            int n_local = i % S12_TILE_N;
            int col = k_start + k_local;
            int n_idx = block_n + n_local;
            if (col < K && n_idx < N)
                B_buf[n_local * S12_TILE_K + k_local] = B_matrix[n_idx * K + col];
            else
                B_buf[n_local * S12_TILE_K + k_local] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // K-tile loop
    for (int tile_k = 0; tile_k < num_k_tiles; tile_k++) {
        int buf_read = tile_k % 2;
        int buf_write = 1 - buf_read;

        // Async load next K-tile into write buffer (overlaps with compute)
        if (tile_k + 1 < num_k_tiles) {
            int k_next = (tile_k + 1) * S12_TILE_K;
            uint8_t* sm_write = sm_buf + buf_write * S12_TILE_M * S12_TILE_K;
            uint8_t* gr_write = gr_buf + buf_write * S12_TILE_M * S12_TILE_K / 2;
            __nv_bfloat16* B_write = B_buf + buf_write * S12_TILE_K * S12_TILE_N;

            for (int i = tid; i < S12_TILE_M * S12_TILE_K; i += S12_BLOCK) {
                int m_local = i / S12_TILE_K;
                int k_local = i % S12_TILE_K;
                int row = block_m + m_local;
                int col = k_next + k_local;
                sm_write[i] = (row < M && col < K) ? sign_mantissa[(int64_t)row * K + col] : 0;
            }
            for (int i = tid; i < S12_TILE_M * S12_TILE_K / 2; i += S12_BLOCK) {
                int m_local = i / (S12_TILE_K / 2);
                int k_local = i % (S12_TILE_K / 2);
                int row = block_m + m_local;
                int col = k_next + k_local * 2;
                gr_write[i] = (row < M && col < K) ? groups[(int64_t)row * K / 2 + col / 2] : 0;
            }
            for (int i = tid; i < S12_TILE_K * S12_TILE_N; i += S12_BLOCK) {
                int k_local = i / S12_TILE_N;
                int n_local = i % S12_TILE_N;
                int col = k_next + k_local;
                int n_idx = block_n + n_local;
                __nv_bfloat16* dst = B_write + n_local * S12_TILE_K + k_local;
                *dst = (col < K && n_idx < N) ? B_matrix[n_idx * K + col] : __float2bfloat16(0.0f);
            }
        }

        // DIRECT DECODE-TO-REGISTER: no intermediate BF16 buffer, no WMMA load
        // Reads compressed bytes from shared memory → decodes to PTX mma registers → tensor core
        uint8_t* sm_read = sm_buf + buf_read * S12_TILE_M * S12_TILE_K;
        uint8_t* gr_read = gr_buf + buf_read * S12_TILE_M * S12_TILE_K / 2;
        __nv_bfloat16* B_read = B_buf + buf_read * S12_TILE_K * S12_TILE_N;

        int warp_m_start = warp_id * MMA_M;  // rows this warp handles (0,16,32,48)
        int row0 = warp_m_start + lane_id / 4;       // rows 0-7 within warp tile
        int row1 = warp_m_start + 8 + lane_id / 4;   // rows 8-15 within warp tile
        int twg = lane_id % 4;

        // Macro to extract group nibble from shared memory
        #define GR_SM(gr_ptr, row, k) \
            (((k) & 1) ? ((gr_ptr)[(row) * (S12_TILE_K/2) + (k)/2] >> 4) \
                       : ((gr_ptr)[(row) * (S12_TILE_K/2) + (k)/2] & 0xF))

        #pragma unroll
        for (int k_slice = 0; k_slice < S12_TILE_K / MMA_K; k_slice++) {
            int kk = k_slice * MMA_K;

            // Decode A directly into PTX mma.sync.m16n8k16 register layout:
            //   a[0] = bf16x2(A[row0][kk+twg*2],   A[row0][kk+twg*2+1])   K cols 0-7
            //   a[1] = bf16x2(A[row1][kk+twg*2],   A[row1][kk+twg*2+1])   rows 8-15, K 0-7
            //   a[2] = bf16x2(A[row0][kk+8+twg*2], A[row0][kk+8+twg*2+1]) K cols 8-15
            //   a[3] = bf16x2(A[row1][kk+8+twg*2], A[row1][kk+8+twg*2+1]) rows 8-15, K 8-15
            uint32_t a_regs[4];
            int k0 = kk + twg * 2, k1 = kk + 8 + twg * 2;

            a_regs[0] = decode_split12_pair(
                sm_read[row0 * S12_TILE_K + k0],     sm_read[row0 * S12_TILE_K + k0 + 1],
                GR_SM(gr_read, row0, k0),             GR_SM(gr_read, row0, k0 + 1), base_exp);
            a_regs[1] = decode_split12_pair(
                sm_read[row1 * S12_TILE_K + k0],     sm_read[row1 * S12_TILE_K + k0 + 1],
                GR_SM(gr_read, row1, k0),             GR_SM(gr_read, row1, k0 + 1), base_exp);
            a_regs[2] = decode_split12_pair(
                sm_read[row0 * S12_TILE_K + k1],     sm_read[row0 * S12_TILE_K + k1 + 1],
                GR_SM(gr_read, row0, k1),             GR_SM(gr_read, row0, k1 + 1), base_exp);
            a_regs[3] = decode_split12_pair(
                sm_read[row1 * S12_TILE_K + k1],     sm_read[row1 * S12_TILE_K + k1 + 1],
                GR_SM(gr_read, row1, k1),             GR_SM(gr_read, row1, k1 + 1), base_exp);

            // Full PTX: load B from shared memory + mma.sync for each N-tile
            // B fragment layout for m16n8k16: B[K=16][N=8], col-major
            //   b[0] = bf16x2(B[K=twg*2][N=gid], B[K=twg*2+1][N=gid])
            //   b[1] = bf16x2(B[K=8+twg*2][N=gid], B[K=8+twg*2+1][N=gid])
            // B_read stored as [N][K], so B(N=n, K=k) = B_read[n * S12_TILE_K + k]
            int b_gid = lane_id / 4;   // N position within 8-wide N-tile
            int b_twg = lane_id % 4;   // K pair index

            #pragma unroll
            for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
                __nv_bfloat16* b_ptr = B_read + (n_tile * MMA_N + b_gid) * S12_TILE_K + kk;
                int bk0 = b_twg * 2;

                uint32_t b_regs[2];
                __nv_bfloat162 bp0 = make_bfloat162(b_ptr[bk0], b_ptr[bk0 + 1]);
                __nv_bfloat162 bp1 = make_bfloat162(b_ptr[8 + bk0], b_ptr[8 + bk0 + 1]);
                b_regs[0] = *reinterpret_cast<uint32_t*>(&bp0);
                b_regs[1] = *reinterpret_cast<uint32_t*>(&bp1);

                mma_m16n8k16(c_regs[n_tile], a_regs, b_regs);
            }
        }
        #undef GR_SM

        __syncthreads();  // Wait before overwriting read buffer
    }

    // Store PTX m16n8k16 accumulators directly to global memory
    // C output mapping: c[0]=C[gid][twg*2], c[1]=C[gid][twg*2+1],
    //                   c[2]=C[8+gid][twg*2], c[3]=C[8+gid][twg*2+1]
    // where gid=lane/4, twg=lane%4, relative to warp's 16-row tile
    int c_gid = lane_id / 4;  // row offset 0-7
    int c_twg = lane_id % 4;  // col pair 0-3

    for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
        float* cf = reinterpret_cast<float*>(c_regs[n_tile]);
        int n_base = block_n + n_tile * MMA_N;  // MMA_N=8

        int out_m0 = tile_m_start + warp_id * MMA_M + c_gid;
        int out_m1 = tile_m_start + warp_id * MMA_M + 8 + c_gid;
        int out_b0 = n_base + c_twg * 2;
        int out_b1 = n_base + c_twg * 2 + 1;

        if (out_m0 < M && out_b0 < N) C_matrix[out_b0 * out_stride + out_m0] = cf[0];
        if (out_m0 < M && out_b1 < N) C_matrix[out_b1 * out_stride + out_m0] = cf[1];
        if (out_m1 < M && out_b0 < N) C_matrix[out_b0 * out_stride + out_m1] = cf[2];
        if (out_m1 < M && out_b1 < N) C_matrix[out_b1 * out_stride + out_m1] = cf[3];
    }
}
