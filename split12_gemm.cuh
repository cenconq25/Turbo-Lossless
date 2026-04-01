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
    // Decoded BF16 weight tile: S12_TILE_M × S12_TILE_K × 2 bytes = 8192 bytes
    __nv_bfloat16* w_decoded = (__nv_bfloat16*)((char*)B_buf + S12_TILE_K * S12_TILE_N * sizeof(__nv_bfloat16) * 2);

    // WMMA accumulator fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WARP_COL_TENSORS];
    #pragma unroll
    for (int j = 0; j < WARP_COL_TENSORS; j++)
        wmma::fill_fragment(c_frag[j], 0.0f);

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

        // Compute: decode to BF16 in shared memory, then WMMA load + mma
        uint8_t* sm_read = sm_buf + buf_read * S12_TILE_M * S12_TILE_K;
        uint8_t* gr_read = gr_buf + buf_read * S12_TILE_M * S12_TILE_K / 2;
        __nv_bfloat16* B_read = B_buf + buf_read * S12_TILE_K * S12_TILE_N;

        // Decode weight tile: split12 → BF16 in dedicated shared memory buffer
        for (int i = tid; i < S12_TILE_M * S12_TILE_K; i += S12_BLOCK) {
            int m_local = i / S12_TILE_K;
            int k_local = i % S12_TILE_K;
            uint8_t sm_val = sm_read[i];
            uint8_t gb = gr_read[m_local * (S12_TILE_K/2) + k_local/2];
            uint8_t group = (k_local & 1) ? (gb >> 4) : (gb & 0xF);
            w_decoded[m_local * S12_TILE_K + k_local] = decode_split12_bf16(sm_val, group, base_exp);
        }
        __syncthreads();

        // Each warp handles rows [warp_id*16 .. warp_id*16+15]
        int warp_m_start = warp_id * MMA_M;

        #pragma unroll
        for (int k_slice = 0; k_slice < S12_TILE_K / MMA_K; k_slice++) {
            int k_off = k_slice * MMA_K;

            // Use WMMA to load A from decoded BF16 in shared memory
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, w_decoded + warp_m_start * S12_TILE_K + k_off, S12_TILE_K);

            // Load B and compute MMA for each N-tile
            #pragma unroll
            for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, B_read + (n_tile * MMA_N) * S12_TILE_K + k_off, S12_TILE_K);
                wmma::mma_sync(c_frag[n_tile], a_frag, b_frag, c_frag[n_tile]);
            }
        }

        __syncthreads();  // Wait before overwriting read buffer
    }

    // Store accumulators via WMMA → shared memory → global (batch-major)
    float* out_smem = (float*)smem_raw;

    for (int n_tile = 0; n_tile < WARP_COL_TENSORS; n_tile++) {
        // WMMA store to shared memory as row-major [16 × 16]
        if (warp_id < 4 && warp_id * MMA_M < rows_this_tile) {
            wmma::store_matrix_sync(out_smem + warp_id * MMA_M * 16,
                                    c_frag[n_tile], 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Scatter to global: C[b * out_stride + m]
        int n_base = block_n + n_tile * 16;
        int n_count = min(16, N - n_base);
        if (n_count > 0) {
            for (int i = tid; i < rows_this_tile * n_count; i += S12_BLOCK) {
                int m_local = i / n_count;
                int n_local = i % n_count;
                int row = tile_m_start + m_local;
                int b_idx = n_base + n_local;
                if (row < M && b_idx < N)
                    C_matrix[b_idx * out_stride + row] = out_smem[m_local * 16 + n_local];
            }
        }
        __syncthreads();
    }
}
