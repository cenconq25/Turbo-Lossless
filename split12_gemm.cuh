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
#include <cstdint>

// MMA dimensions
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Tile dimensions
#define S12_TILE_M 64      // 4 warp rows × 16
#define S12_TILE_K 64      // 4 K-slices × 16 (MMA_K)
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
    const __nv_bfloat16* __restrict__ B_matrix,  // [K * N] activations (col-major for ldmatrix)
    float* __restrict__ C_matrix,                // [M * N] output
    int M, int K, int N)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid & 31;

    // Block position
    const int block_m = blockIdx.x * S12_TILE_M;
    const int block_n = blockIdx.y * S12_TILE_N;

    // Shared memory: double-buffered weight data + B matrix
    extern __shared__ __align__(128) char smem_raw[];
    // Layout: [sm_buf0][sm_buf1][gr_buf0][gr_buf1][B_buf0][B_buf1]
    uint8_t* sm_buf = (uint8_t*)smem_raw;
    // sm: S12_TILE_M × S12_TILE_K × 2 buffers = 64×64×2 = 8192 bytes
    uint8_t* gr_buf = sm_buf + S12_TILE_M * S12_TILE_K * 2;
    // gr: S12_TILE_M × S12_TILE_K / 2 × 2 buffers = 64×32×2 = 4096 bytes
    __nv_bfloat16* B_buf = (__nv_bfloat16*)(gr_buf + S12_TILE_M * S12_TILE_K / 2 * 2);
    // B: S12_TILE_K × S12_TILE_N × 2 buffers

    // Accumulator registers: each warp handles 16×WARP_COL_TENSORS×8 output
    // For S12_TILE_N=16, WARP_COL_TENSORS=2: each warp has 2 MMA accumulators
    uint32_t c_regs[WARP_COL_TENSORS][4];  // 4 float32 values per MMA output
    #pragma unroll
    for (int j = 0; j < WARP_COL_TENSORS; j++)
        for (int i = 0; i < 4; i++) c_regs[j][i] = 0;

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

        // Compute: 4 K-slices of MMA_K=16 each
        uint8_t* sm_read = sm_buf + buf_read * S12_TILE_M * S12_TILE_K;
        uint8_t* gr_read = gr_buf + buf_read * S12_TILE_M * S12_TILE_K / 2;
        __nv_bfloat16* B_read = B_buf + buf_read * S12_TILE_K * S12_TILE_N;

        // Each warp handles rows [warp_id*16 .. warp_id*16+15]
        int warp_m_start = warp_id * MMA_M;

        #pragma unroll
        for (int k_slice = 0; k_slice < S12_TILE_K / MMA_K; k_slice++) {
            int k_off = k_slice * MMA_K;

            // Decode A matrix (weights) into mma registers
            // mma.m16n8k16 A operand: each thread holds 4 uint32_t (8 BF16 values)
            // Thread layout: lane_id maps to specific (row, col) in 16×16 tile
            // Row = lane_id / 4 for first 8 rows, lane_id / 4 + 8 for second 8 rows
            // Col pairs at: (lane_id % 4) * 2 and (lane_id % 4) * 2 + 1
            uint32_t a_regs[4];

            int a_row_base = lane_id / 4;  // 0-7
            int a_col_base = (lane_id % 4) * 2;  // 0,2,4,6

            // First 8 rows (a_regs[0], a_regs[1] = k cols 0-7, 8-15)
            {
                int row = warp_m_start + a_row_base;
                // k cols 0-7: 4 pairs
                int k0 = k_off + a_col_base;
                int k1 = k_off + a_col_base + 1;
                uint8_t sm0 = sm_read[row * S12_TILE_K + k0];
                uint8_t sm1 = sm_read[row * S12_TILE_K + k1];
                uint8_t gb0 = gr_read[row * (S12_TILE_K/2) + k0/2];
                uint8_t gb1 = gr_read[row * (S12_TILE_K/2) + k1/2];
                uint8_t g0 = (k0 & 1) ? (gb0 >> 4) : (gb0 & 0xF);
                uint8_t g1 = (k1 & 1) ? (gb1 >> 4) : (gb1 & 0xF);
                a_regs[0] = decode_split12_pair(sm0, sm1, g0, g1, base_exp);

                int k2 = k_off + 8 + a_col_base;
                int k3 = k_off + 8 + a_col_base + 1;
                uint8_t sm2 = sm_read[row * S12_TILE_K + k2];
                uint8_t sm3 = sm_read[row * S12_TILE_K + k3];
                uint8_t gb2 = gr_read[row * (S12_TILE_K/2) + k2/2];
                uint8_t gb3 = gr_read[row * (S12_TILE_K/2) + k3/2];
                uint8_t g2 = (k2 & 1) ? (gb2 >> 4) : (gb2 & 0xF);
                uint8_t g3 = (k3 & 1) ? (gb3 >> 4) : (gb3 & 0xF);
                a_regs[1] = decode_split12_pair(sm2, sm3, g2, g3, base_exp);
            }
            // Second 8 rows (a_regs[2], a_regs[3])
            {
                int row = warp_m_start + a_row_base + 8;
                int k0 = k_off + a_col_base;
                int k1 = k_off + a_col_base + 1;
                uint8_t sm0 = sm_read[row * S12_TILE_K + k0];
                uint8_t sm1 = sm_read[row * S12_TILE_K + k1];
                uint8_t gb0 = gr_read[row * (S12_TILE_K/2) + k0/2];
                uint8_t gb1 = gr_read[row * (S12_TILE_K/2) + k1/2];
                uint8_t g0 = (k0 & 1) ? (gb0 >> 4) : (gb0 & 0xF);
                uint8_t g1 = (k1 & 1) ? (gb1 >> 4) : (gb1 & 0xF);
                a_regs[2] = decode_split12_pair(sm0, sm1, g0, g1, base_exp);

                int k2 = k_off + 8 + a_col_base;
                int k3 = k_off + 8 + a_col_base + 1;
                uint8_t sm2 = sm_read[row * S12_TILE_K + k2];
                uint8_t sm3 = sm_read[row * S12_TILE_K + k3];
                uint8_t gb2 = gr_read[row * (S12_TILE_K/2) + k2/2];
                uint8_t gb3 = gr_read[row * (S12_TILE_K/2) + k3/2];
                uint8_t g2 = (k2 & 1) ? (gb2 >> 4) : (gb2 & 0xF);
                uint8_t g3 = (k3 & 1) ? (gb3 >> 4) : (gb3 & 0xF);
                a_regs[3] = decode_split12_pair(sm2, sm3, g2, g3, base_exp);
            }

            // Load B matrix from shared memory and compute MMA for each N-tile
            #pragma unroll
            for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
                // B operand for mma.m16n8k16: B[K=16][N=8] col-major
                // Thread mapping: groupId=lane/4 → N position, twg=lane%4 → K position pair
                // b[0] = bf16x2(B(K=twg*2, N=groupId), B(K=twg*2+1, N=groupId))
                // b[1] = bf16x2(B(K=8+twg*2, N=groupId), B(K=8+twg*2+1, N=groupId))
                // B_read stored as [N][K], so B(N=n, K=k) = B_read[n * S12_TILE_K + k]
                uint32_t b_regs[2];
                int b_n = lane_id / 4;             // groupId → N dimension (0-7)
                int b_k = (lane_id % 4) * 2;      // twg*2 → K dimension pair start (0,2,4,6)
                __nv_bfloat16* b_ptr = B_read + (n_tile * MMA_N + b_n) * S12_TILE_K + k_off;

                __nv_bfloat16 v0 = b_ptr[b_k];
                __nv_bfloat16 v1 = b_ptr[b_k + 1];
                __nv_bfloat162 pair0 = {v0, v1};
                b_regs[0] = *reinterpret_cast<uint32_t*>(&pair0);

                __nv_bfloat16 v2 = b_ptr[8 + b_k];
                __nv_bfloat16 v3 = b_ptr[8 + b_k + 1];
                __nv_bfloat162 pair1 = {v2, v3};
                b_regs[1] = *reinterpret_cast<uint32_t*>(&pair1);

                mma_m16n8k16(c_regs[n_tile], a_regs, b_regs);
            }
        }

        __syncthreads();  // Wait before overwriting read buffer
    }

    // Store accumulator to global memory
    // Output layout: C[b * M + m] (batch-major, matches inference.cpp expectation)
    // MMA m16n8k16 output mapping per lane:
    //   c_regs[0] → row=(lane/4), col=(lane%4)*2    — first 8 rows
    //   c_regs[1] → row=(lane/4), col=(lane%4)*2+1
    //   c_regs[2] → row=(lane/4)+8, col=(lane%4)*2  — second 8 rows
    //   c_regs[3] → row=(lane/4)+8, col=(lane%4)*2+1
    for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
        float* c_ptr = reinterpret_cast<float*>(c_regs[n_tile]);
        int out_n_base = block_n + n_tile * MMA_N;

        int lane_row0 = lane_id / 4;
        int lane_row1 = lane_row0 + 8;
        int lane_col0 = (lane_id % 4) * 2;
        int lane_col1 = lane_col0 + 1;

        int out_m0 = block_m + warp_id * MMA_M + lane_row0;
        int out_m1 = block_m + warp_id * MMA_M + lane_row1;
        int out_b0 = out_n_base + lane_col0;
        int out_b1 = out_n_base + lane_col1;

        // Batch-major: C[b][m] stored as C[b * M + m]
        if (out_m0 < M && out_b0 < N)
            C_matrix[out_b0 * M + out_m0] = c_ptr[0];
        if (out_m0 < M && out_b1 < N)
            C_matrix[out_b1 * M + out_m0] = c_ptr[1];
        if (out_m1 < M && out_b0 < N)
            C_matrix[out_b0 * M + out_m1] = c_ptr[2];
        if (out_m1 < M && out_b1 < N)
            C_matrix[out_b1 * M + out_m1] = c_ptr[3];
    }
}
