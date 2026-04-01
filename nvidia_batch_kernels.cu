/**
 * NVIDIA-Optimized Batch Matvec Kernels — Per-Row with B-Chunking
 *
 * Key insight: keep the fast per-row design (1 block per output row, 256 threads),
 * but handle arbitrary batch sizes in ONE kernel launch by processing B in chunks of 8.
 * Weight data is read from DRAM once, then served from L1/L2 cache for subsequent chunks.
 *
 * This avoids the overhead of tiled kernels (shared memory syncs, cooperative loads)
 * while eliminating the B/8 sequential launches that re-read ALL weights from DRAM.
 */

#include "gpu_compat.h"
#include <cstdint>

#define WS      TURBO_WARP_SIZE
#define WG      TURBO_WORKGROUP_SIZE
#define NW      TURBO_NUM_WARPS
#define B_CHUNK 8  // process 8 batch items per register set

__device__ __forceinline__
float bf16f(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

// ---------------------------------------------------------------------------
// Split12 per-row batch matvec: handles any B in a single kernel launch.
// For each row, iterates B in chunks of 8. Within each chunk, processes K
// columns with 8 accumulators. Weight bytes hit L1 cache for subsequent chunks.
// ---------------------------------------------------------------------------
__launch_bounds__(256)
__global__ void split12_batch_anyB(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,     // [B * act_stride] BF16
    int act_stride,                              // stride between sequences
    const int32_t* __restrict__ escape_row_base, // [M]
    const uint8_t* __restrict__ escape_counts,   // [M * 256]
    const int16_t* __restrict__ escape_vals,     // [total_escapes]
    float* __restrict__ output,                  // [B * out_stride]
    int out_stride,                              // stride between output sequences
    int M, int K, int B)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int lane = tid & (WS - 1);
    const int warp_id = tid / WS;
    const int is_odd = tid & 1;

    // Shared memory for warp reduction
    __shared__ float warp_sums[NW];
    __shared__ int warp_esc_totals[NW];

    // Escape prefix sum (same for all B chunks — escapes are in the weights)
    int my_esc_count = (int)escape_counts[row * WG + tid];
    int prefix = my_esc_count;
    for (int off = 1; off < WS; off <<= 1) {
        int v = __shfl_up(prefix, off, WS);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WS - 1, WS);
    prefix -= my_esc_count;
    if (lane == WS - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_base = escape_row_base[row] + cross + prefix;

    // Weight pointers (constant across B chunks)
    const uint8_t* sm_ptr = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr = groups + (int64_t)row * K / 2 + tid / 2;

    // Process B in chunks of B_CHUNK
    for (int b_start = 0; b_start < B; b_start += B_CHUNK) {
        int b_count = min(B_CHUNK, B - b_start);

        // Reset accumulators
        float s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;

        // Reset escape pointer for this B chunk
        int esc_ptr = esc_base;

        // Activation base pointers for this chunk
        const int16_t* a[B_CHUNK];
        for (int bc = 0; bc < b_count; bc++)
            a[bc] = activations + (b_start + bc) * act_stride;

        // Weight pointer reset for this B chunk
        const uint8_t* sm_p = sm_ptr;
        const uint8_t* gr_p = gr_ptr;

        // Main FMA loop: process 2 columns per iteration
        int col = tid;
        for (; col + WG < K; col += WG * 2) {
            // Decode 2 weights
            uint8_t sm0 = sm_p[0], sm1 = sm_p[WG];
            uint8_t gb0 = gr_p[0], gb1 = gr_p[WG/2];
            uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
            uint32_t g1 = (col + WG) & 1 ? (gb1 >> 4) : (gb1 & 0xF);  // next col parity

            float wt0, wt1;
            if (__builtin_expect(g0 != 0, 1)) {
                union { uint32_t u; float f; } c;
                c.u = ((uint32_t)(sm0 >> 7) << 15 | (uint32_t)(base_exp + g0) << 7 | (sm0 & 0x7F)) << 16;
                wt0 = c.f;
            } else {
                wt0 = bf16f(escape_vals[esc_ptr++]);
            }
            if (__builtin_expect(g1 != 0, 1)) {
                union { uint32_t u; float f; } c;
                c.u = ((uint32_t)(sm1 >> 7) << 15 | (uint32_t)(base_exp + g1) << 7 | (sm1 & 0x7F)) << 16;
                wt1 = c.f;
            } else {
                wt1 = bf16f(escape_vals[esc_ptr++]);
            }

            // FMA with up to 8 activation vectors
            if (b_count > 0) s0 += wt0 * bf16f(a[0][col]) + wt1 * bf16f(a[0][col + WG]);
            if (b_count > 1) s1 += wt0 * bf16f(a[1][col]) + wt1 * bf16f(a[1][col + WG]);
            if (b_count > 2) s2 += wt0 * bf16f(a[2][col]) + wt1 * bf16f(a[2][col + WG]);
            if (b_count > 3) s3 += wt0 * bf16f(a[3][col]) + wt1 * bf16f(a[3][col + WG]);
            if (b_count > 4) s4 += wt0 * bf16f(a[4][col]) + wt1 * bf16f(a[4][col + WG]);
            if (b_count > 5) s5 += wt0 * bf16f(a[5][col]) + wt1 * bf16f(a[5][col + WG]);
            if (b_count > 6) s6 += wt0 * bf16f(a[6][col]) + wt1 * bf16f(a[6][col + WG]);
            if (b_count > 7) s7 += wt0 * bf16f(a[7][col]) + wt1 * bf16f(a[7][col + WG]);

            sm_p += WG * 2;
            gr_p += WG;  // WG * 2 / 2
        }
        // Tail
        if (col < K) {
            uint8_t sm0 = sm_p[0];
            uint8_t gb0 = gr_p[0];
            uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
            float wt;
            if (__builtin_expect(g0 != 0, 1)) {
                union { uint32_t u; float f; } c;
                c.u = ((uint32_t)(sm0 >> 7) << 15 | (uint32_t)(base_exp + g0) << 7 | (sm0 & 0x7F)) << 16;
                wt = c.f;
            } else {
                wt = bf16f(escape_vals[esc_ptr++]);
            }
            if (b_count > 0) s0 += wt * bf16f(a[0][col]);
            if (b_count > 1) s1 += wt * bf16f(a[1][col]);
            if (b_count > 2) s2 += wt * bf16f(a[2][col]);
            if (b_count > 3) s3 += wt * bf16f(a[3][col]);
            if (b_count > 4) s4 += wt * bf16f(a[4][col]);
            if (b_count > 5) s5 += wt * bf16f(a[5][col]);
            if (b_count > 6) s6 += wt * bf16f(a[6][col]);
            if (b_count > 7) s7 += wt * bf16f(a[7][col]);
        }

        // Warp reduction + output for each active batch item
        float sums[B_CHUNK] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int bc = 0; bc < b_count; bc++) {
            float val = sums[bc];
            for (int off = WS / 2; off > 0; off >>= 1)
                val += __shfl_down(val, off, WS);
            if (lane == 0) warp_sums[warp_id] = val;
            __syncthreads();
            if (tid == 0) {
                float total = 0;
                for (int w = 0; w < NW; w++) total += warp_sums[w];
                output[(b_start + bc) * out_stride + row] = total;
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
extern "C" {

int launch_split12_tiled_batch_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, int act_stride,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* output, int out_stride,
    int M, int K, int B, void* stream)
{
    split12_batch_anyB<<<M, WG, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
        (const int16_t*)activations, act_stride,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)output, out_stride,
        M, K, B);
    return 0;
}

int launch_packed12_tiled_batch_async(
    const void* packed, int base_exp,
    const void* activations, int act_stride,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* output, int out_stride,
    int M, int K, int B, void* stream)
{
    // Fallback: for packed12, use the per-row kernel in chunks
    // (packed12 tiled kernel deferred — split12 is the primary path)
    return -1;  // caller should fall back to legacy B=8 chunking
}

}
