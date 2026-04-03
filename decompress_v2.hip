/**
 * Turbo Lossless — Structured 12-bit Matvec Kernels
 * Target: AMD MI50 32GB (gfx906), 1 TB/s HBM2, 60 CUs, 64-wide wavefronts
 *
 * Architecture:
 *   - Structured 12-bit encoding: [4-bit exp_group][1-bit sign][7-bit mantissa]
 *   - Decode: exponent = BaseExp + group (1 integer ADD, pure ALU, no LDS codebook)
 *   - Group 0 = escape sentinel → read exact BF16 from escape table
 *   - 0 KB LDS for codebook → 2x occupancy vs codebook approach
 *
 * Formats:
 *   - Packed 12-bit: bit-packed stream, 1.33x read amplification
 *   - Split12: byte-aligned dual-array (sign_mantissa[uint8] + groups[nibble-packed])
 *     Zero read amplification, preferred when available
 *
 * Build: hipcc -O3 --offload-arch=gfx906 -shared -fPIC -o decompress_v2.so decompress_v2.hip
 */

#include "gpu_compat.h"
#include <cstdint>

#define WARP_SIZE      TURBO_WARP_SIZE
#define WORKGROUP_SIZE TURBO_WORKGROUP_SIZE
#define NUM_WARPS      TURBO_NUM_WARPS

// BF16 → float32: BF16 is upper 16 bits of IEEE float32
__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

// ---------------------------------------------------------------------------
// Extract 12-bit value from packed bitstream
// ptr is always 4-byte aligned; shift is constant per thread (< 32)
// On NVIDIA: use two 4-byte aligned reads (uint64_t load from 4-byte-aligned ptr is UB)
// On AMD: single 8-byte load works fine with HIP's relaxed alignment rules
// ---------------------------------------------------------------------------
__device__ __forceinline__
uint32_t extract_packed12(const char* ptr, int shift) {
#ifdef TURBO_NVIDIA
    uint32_t lo = *(const uint32_t*)ptr;
    uint32_t hi = *(const uint32_t*)(ptr + 4);
    return (uint32_t)(((uint64_t)hi << 32 | lo) >> shift) & 0xFFF;
#else
    return (uint32_t)(*(const uint64_t*)ptr >> shift) & 0xFFF;
#endif
}

// ---------------------------------------------------------------------------
// Structured 12-bit decode: pure ALU, no LDS, no memory access
// Format: [4-bit group][1-bit sign][7-bit mantissa] = 12 bits
// ---------------------------------------------------------------------------
__device__ __forceinline__
float structured12_decode(uint32_t raw12, int base_exp) {
    uint32_t group    = raw12 >> 8;           // top 4 bits
    uint32_t sign     = (raw12 >> 7) & 1;     // bit 7
    uint32_t mantissa = raw12 & 0x7F;          // bottom 7 bits
    uint32_t exponent = base_exp + group;      // ONE add
    uint32_t bf16_bits = (sign << 15) | (exponent << 7) | mantissa;
    union { uint32_t u; float f; } c;
    c.u = bf16_bits << 16;
    return c.f;
}

// ---------------------------------------------------------------------------
// Batched patch descriptor for B=1 multi-tensor patch correction
// Eliminates ~192 tiny kernel launches per token (7 per layer × 32 layers)
// ---------------------------------------------------------------------------
struct PatchBatchDesc {
    const int32_t* row_offsets;
    const int32_t* patch_cols;
    const int16_t* patch_correct;
    const int16_t* patch_wrong;
    const int16_t* activations;
    float*         output;
    int            M;
};

// ---------------------------------------------------------------------------
// Batched patch correction: one launch corrects up to N weight tensors
// Grid: (max_M, num_descs), Block: WARP_SIZE
// blockIdx.y selects tensor, blockIdx.x selects row within that tensor
// ---------------------------------------------------------------------------
__global__ void apply_patches_b1_batched(
    const PatchBatchDesc* __restrict__ descs,
    int num_descs)
{
    int desc_idx = blockIdx.y;
    if (desc_idx >= num_descs) return;

    PatchBatchDesc d = descs[desc_idx];
    int row = blockIdx.x;
    if (row >= d.M) return;

    int tid = threadIdx.x;
    int start = d.row_offsets[row];
    int end   = d.row_offsets[row + 1];
    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(d.patch_correct[p]) - bf16_to_float(d.patch_wrong[p]))
                      * bf16_to_float(d.activations[d.patch_cols[p]]);
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) d.output[row] += correction;
}

// ---------------------------------------------------------------------------
// Patch correction kernel: wavefront-parallel two-pass escape correction
// Applied after structured12_matvec_v2 to fix escape values (group=0)
// ---------------------------------------------------------------------------
__global__ void apply_patches_v2(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ patch_cols,
    const int16_t* __restrict__ correct_vals,
    const int16_t* __restrict__ wrong_vals,
    const int16_t* __restrict__ activations,
    float*         __restrict__ output,
    int M)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int start = row_offsets[row];
    int end   = row_offsets[row + 1];
    float correction = 0.0f;
    for (int p = start + tid; p < end; p += WARP_SIZE)
        correction += (bf16_to_float(correct_vals[p]) - bf16_to_float(wrong_vals[p]))
                      * bf16_to_float(activations[patch_cols[p]]);
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        correction += __shfl_down(correction, off, WARP_SIZE);
    if (tid == 0 && correction != 0.0f) output[row] += correction;
}

extern "C" {
int launch_patches_v2(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M) {
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches_v2, dim3(M), dim3(WARP_SIZE), 0, 0,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const int16_t*)activations, (float*)output, M);
    return (hipDeviceSynchronize() == hipSuccess) ? 0 : -1;
}

int launch_patches_v2_async(const void* row_offsets, const void* patch_cols,
    const void* correct_vals, const void* wrong_vals,
    const void* activations, void* output, int M, void* stream) {
    if (M == 0) return 0;
    hipLaunchKernelGGL(apply_patches_v2, dim3(M), dim3(WARP_SIZE), 0, (hipStream_t)stream,
        (const int32_t*)row_offsets, (const int32_t*)patch_cols,
        (const int16_t*)correct_vals, (const int16_t*)wrong_vals,
        (const int16_t*)activations, (float*)output, M);
    return 0;
}

// Batched B=1 patch correction: corrects multiple weight tensors in one launch
// descs_gpu: pre-allocated GPU buffer for PatchBatchDesc array
// descs_cpu: host-side descriptor array (up to 7 entries per layer)
// num_descs: number of valid descriptors (0 = skip)
// max_M: largest M across all descriptors (grid X dimension)
int launch_patches_b1_batched_async(void* descs_gpu, const void* descs_cpu,
    int num_descs, int max_M, void* stream) {
    if (num_descs == 0 || max_M == 0) return 0;
    // Async copy descriptors to GPU (small: num_descs * 56 bytes)
    hipMemcpyAsync(descs_gpu, descs_cpu, num_descs * sizeof(PatchBatchDesc),
                   hipMemcpyHostToDevice, (hipStream_t)stream);
    dim3 grid(max_M, num_descs);
    hipLaunchKernelGGL(apply_patches_b1_batched, grid, dim3(WARP_SIZE), 0, (hipStream_t)stream,
        (const PatchBatchDesc*)descs_gpu, num_descs);
    return 0;
}
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 two-pass matvec (no escape handling — patches applied separately)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;
    const char* pack_ptr = (const char*)(row_packed + (bp >> 5));
    const char* act_ptr = (const char*)(activations + tid);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        uint32_t d0 = extract_packed12(pack_ptr, shift);
        uint32_t d1 = extract_packed12(pack_ptr + 384, shift);
        uint32_t d2 = extract_packed12(pack_ptr + 768, shift);
        uint32_t d3 = extract_packed12(pack_ptr + 1152, shift);
        sum0 += structured12_decode(d0, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += structured12_decode(d1, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += structured12_decode(d2, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += structured12_decode(d3, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1536));
        pack_ptr += 1536;
        act_ptr += 2048;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        uint32_t d = extract_packed12(pack_ptr, shift);
        sum0 += structured12_decode(d, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        pack_ptr += 384;
        act_ptr += 512;
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();
    if (tid == 0) { float s = 0; for (int w = 0; w < NUM_WARPS; w++) s += warp_sums[w]; output[row] = s; }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=1 DUAL matvec: processes TWO weight tensors (gate+up)
// sharing the same activation reads. Saves 1 kernel launch per layer and
// shares L2 cache for activations.
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_v2_dual(
    const uint32_t* __restrict__ packed_a, int base_exp_a,  // first tensor (gate)
    const uint32_t* __restrict__ packed_b, int base_exp_b,  // second tensor (up)
    const int16_t*  __restrict__ activations,
    float*          __restrict__ output_a,
    float*          __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[NUM_WARPS];
    __shared__ float warp_sums_b[NUM_WARPS];

    float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;

    // Both tensors have same dimensions, so same row layout
    int64_t row_bit_start = (int64_t)row * K * 12;
    int bp = (int)(row_bit_start % 32) + tid * 12;
    const int shift = bp & 31;

    // Pointer-based addressing for tensor A
    const char* pa_ptr = (const char*)((const uint32_t*)packed_a + (row_bit_start / 32) + (bp >> 5));
    // Pointer-based addressing for tensor B
    const char* pb_ptr = (const char*)((const uint32_t*)packed_b + (row_bit_start / 32) + (bp >> 5));
    // Activation pointer
    const char* act_ptr = (const char*)(activations + tid);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Read activations ONCE (shared between both tensors)
        float af0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float af1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float af2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float af3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        // Decode weights from tensor A (gate)
        uint32_t da0 = extract_packed12(pa_ptr, shift);
        uint32_t da1 = extract_packed12(pa_ptr + 384, shift);
        uint32_t da2 = extract_packed12(pa_ptr + 768, shift);
        uint32_t da3 = extract_packed12(pa_ptr + 1152, shift);
        sa0 += structured12_decode(da0, base_exp_a) * af0;
        sa1 += structured12_decode(da1, base_exp_a) * af1;
        sa2 += structured12_decode(da2, base_exp_a) * af2;
        sa3 += structured12_decode(da3, base_exp_a) * af3;

        // Decode weights from tensor B (up)
        uint32_t db0 = extract_packed12(pb_ptr, shift);
        uint32_t db1 = extract_packed12(pb_ptr + 384, shift);
        uint32_t db2 = extract_packed12(pb_ptr + 768, shift);
        uint32_t db3 = extract_packed12(pb_ptr + 1152, shift);
        sb0 += structured12_decode(db0, base_exp_b) * af0;
        sb1 += structured12_decode(db1, base_exp_b) * af1;
        sb2 += structured12_decode(db2, base_exp_b) * af2;
        sb3 += structured12_decode(db3, base_exp_b) * af3;

        pa_ptr += 1536;
        pb_ptr += 1536;
        act_ptr += 2048;
    }
    // Remainder
    for (; col < K; col += WORKGROUP_SIZE) {
        float af = bf16_to_float(*(const int16_t*)(act_ptr));
        uint32_t da = extract_packed12(pa_ptr, shift);
        uint32_t db = extract_packed12(pb_ptr, shift);
        sa0 += structured12_decode(da, base_exp_a) * af;
        sb0 += structured12_decode(db, base_exp_b) * af;
        pa_ptr += 384;
        pb_ptr += 384;
        act_ptr += 512;
    }

    // Reduction for tensor A
    float sum_a = sa0 + sa1 + sa2 + sa3;
    // Reduction for tensor B
    float sum_b = sb0 + sb1 + sb2 + sb3;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums_a[tid / WARP_SIZE] = sum_a;
        warp_sums_b[tid / WARP_SIZE] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0, sb = 0;
        for (int w = 0; w < NUM_WARPS; w++) { sa += warp_sums_a[w]; sb += warp_sums_b[w]; }
        output_a[row] = sa; output_b[row] = sb;
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=4 fused matvec (single-pass with escape counts)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_batch4(
    const uint32_t* __restrict__ packed,
    int base_exp,                              // BaseExp (e.g., 106)
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // NO LDS codebook — only warp_sums + esc_totals
    __shared__ float warp_sums[NUM_WARPS * 4];
    __shared__ int warp_esc_totals[NUM_WARPS];

    // Escape count prefix sum via warp shuffles (same as before)
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Main FMA loop — pointer-based addressing with constant offsets
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;  // constant — stride is multiple of 32
    const char* base_ptr = (const char*)(row_packed + (bp >> 5));
    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t raw0 = extract_packed12(base_ptr, shift);
        uint32_t raw1 = extract_packed12(base_ptr + 384, shift);

        uint32_t g0 = raw0 >> 8, g1 = raw1 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1))
            ? structured12_decode(raw0, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1))
            ? structured12_decode(raw1, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);

        s0a += wt0 * bf16_to_float(act0[col]);  s0b += wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1a += wt0 * bf16_to_float(act1[col]);  s1b += wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2a += wt0 * bf16_to_float(act2[col]);  s2b += wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3a += wt0 * bf16_to_float(act3[col]);  s3b += wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        base_ptr += 768;
    }
    if (col < K) {
        uint32_t raw = extract_packed12(base_ptr, shift);
        uint32_t g = raw >> 8;
        float wt = (__builtin_expect(g != 0, 1))
            ? structured12_decode(raw, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        s0a += wt * bf16_to_float(act0[col]);
        s1a += wt * bf16_to_float(act1[col]);
        s2a += wt * bf16_to_float(act2[col]);
        s3a += wt * bf16_to_float(act3[col]);
    }

    // Warp reduction + output (same as codebook version)
    float s0=s0a+s0b, s1=s1a+s1b, s2=s2a+s2b, s3=s3a+s3b;
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
    }
    int wid = tid / WARP_SIZE;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]=s0; warp_sums[wid+NUM_WARPS]=s1;
        warp_sums[wid+NUM_WARPS*2]=s2; warp_sums[wid+NUM_WARPS*3]=s3;
    }
    __syncthreads();
    if (tid == 0) {
        float r0=0,r1=0,r2=0,r3=0;
        for (int w = 0; w < NUM_WARPS; w++) {
            r0 += warp_sums[w]; r1 += warp_sums[w+NUM_WARPS];
            r2 += warp_sums[w+NUM_WARPS*2]; r3 += warp_sums[w+NUM_WARPS*3];
        }
        out0[row]=r0; out1[row]=r1; out2[row]=r2; out3[row]=r3;
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit B=8 fused matvec (single-pass with escape counts)
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void structured12_matvec_batch8(
    const uint32_t* __restrict__ packed,
    int base_exp,
    const int16_t*  __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t*  __restrict__ act2, const int16_t* __restrict__ act3,
    const int16_t*  __restrict__ act4, const int16_t* __restrict__ act5,
    const int16_t*  __restrict__ act6, const int16_t* __restrict__ act7,
    const int32_t*  __restrict__ escape_row_base,
    const uint8_t*  __restrict__ escape_counts,
    const int16_t*  __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    float* __restrict__ out4, float* __restrict__ out5,
    float* __restrict__ out6, float* __restrict__ out7,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // NO LDS codebook — only warp_sums + esc_totals
    __shared__ float warp_sums[NUM_WARPS * 8];
    __shared__ int warp_esc_totals[NUM_WARPS];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Main FMA loop — pointer-based addressing with constant offsets
    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;  // constant — stride is multiple of 32
    const char* base_ptr = (const char*)(row_packed + (bp >> 5));
    // Single accumulators (saves 8 VGPRs vs split a/b — critical for B=8 register pressure)
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        uint32_t raw0 = extract_packed12(base_ptr, shift);
        uint32_t raw1 = extract_packed12(base_ptr + 384, shift);

        uint32_t g0 = raw0 >> 8, g1 = raw1 >> 8;
        float wt0 = (__builtin_expect(g0 != 0, 1))
            ? structured12_decode(raw0, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        float wt1 = (__builtin_expect(g1 != 0, 1))
            ? structured12_decode(raw1, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);

        s0 += wt0 * bf16_to_float(act0[col]) + wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1 += wt0 * bf16_to_float(act1[col]) + wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2 += wt0 * bf16_to_float(act2[col]) + wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3 += wt0 * bf16_to_float(act3[col]) + wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        s4 += wt0 * bf16_to_float(act4[col]) + wt1 * bf16_to_float(act4[col + WORKGROUP_SIZE]);
        s5 += wt0 * bf16_to_float(act5[col]) + wt1 * bf16_to_float(act5[col + WORKGROUP_SIZE]);
        s6 += wt0 * bf16_to_float(act6[col]) + wt1 * bf16_to_float(act6[col + WORKGROUP_SIZE]);
        s7 += wt0 * bf16_to_float(act7[col]) + wt1 * bf16_to_float(act7[col + WORKGROUP_SIZE]);
        base_ptr += 768;
    }
    if (col < K) {
        uint32_t raw = extract_packed12(base_ptr, shift);
        uint32_t g = raw >> 8;
        float wt = (__builtin_expect(g != 0, 1))
            ? structured12_decode(raw, base_exp)
            : bf16_to_float(escape_vals[esc_ptr++]);
        s0 += wt * bf16_to_float(act0[col]);
        s1 += wt * bf16_to_float(act1[col]);
        s2 += wt * bf16_to_float(act2[col]);
        s3 += wt * bf16_to_float(act3[col]);
        s4 += wt * bf16_to_float(act4[col]);
        s5 += wt * bf16_to_float(act5[col]);
        s6 += wt * bf16_to_float(act6[col]);
        s7 += wt * bf16_to_float(act7[col]);
    }

    // Warp reduction + output
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
        s4 += __shfl_down(s4, off, WARP_SIZE);
        s5 += __shfl_down(s5, off, WARP_SIZE);
        s6 += __shfl_down(s6, off, WARP_SIZE);
        s7 += __shfl_down(s7, off, WARP_SIZE);
    }
    int wid = tid / WARP_SIZE;
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[wid]              = s0;
        warp_sums[wid + NUM_WARPS]   = s1;
        warp_sums[wid + NUM_WARPS*2] = s2;
        warp_sums[wid + NUM_WARPS*3] = s3;
        warp_sums[wid + NUM_WARPS*4] = s4;
        warp_sums[wid + NUM_WARPS*5] = s5;
        warp_sums[wid + NUM_WARPS*6] = s6;
        warp_sums[wid + NUM_WARPS*7] = s7;
    }
    __syncthreads();
    if (tid == 0) {
        float r[8] = {0};
        for (int w = 0; w < NUM_WARPS; w++)
            for (int o = 0; o < 8; o++) r[o] += warp_sums[w + NUM_WARPS * o];
        out0[row]=r[0]; out1[row]=r[1]; out2[row]=r[2]; out3[row]=r[3];
        out4[row]=r[4]; out5[row]=r[5]; out6[row]=r[6]; out7[row]=r[7];
    }
}

// ---------------------------------------------------------------------------
// Structured 12-bit launch wrappers
// ---------------------------------------------------------------------------
extern "C" {

int launch_structured12_v2_async(
    const void* packed, int base_exp,
    const void* activations, void* output, int M, int K, void* stream,
    const void* patch_row_offsets, const void* patch_cols,
    const void* patch_correct_vals, const void* patch_wrong_vals)
{
    hipLaunchKernelGGL(structured12_matvec_v2,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)activations, (float*)output, M, K);
    // Inline patches on thread 0 via separate lightweight launch
    if (patch_row_offsets) {
        hipLaunchKernelGGL(apply_patches_v2,
            dim3(M), dim3(WARP_SIZE), 0, (hipStream_t)stream,
            (const int32_t*)patch_row_offsets, (const int32_t*)patch_cols,
            (const int16_t*)patch_correct_vals, (const int16_t*)patch_wrong_vals,
            (const int16_t*)activations, (float*)output, M);
    }
    return 0;
}

int launch_structured12_v2_dual_async(
    const void* packed_a, int base_exp_a,
    const void* packed_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_v2_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed_a, base_exp_a,
        (const uint32_t*)packed_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

int launch_structured12_batch4_async(
    const void* packed, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_batch4,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return 0;
}

int launch_structured12_batch8_async(
    const void* packed, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* a4, const void* a5, const void* a6, const void* a7,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3,
    void* o4, void* o5, void* o6, void* o7,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(structured12_matvec_batch8,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint32_t*)packed, base_exp,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int16_t*)a4, (const int16_t*)a5,
        (const int16_t*)a6, (const int16_t*)a7,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3,
        (float*)o4, (float*)o5, (float*)o6, (float*)o7,
        M, K);
    return 0;
}

}

// ---------------------------------------------------------------------------
// Split12 matvec: byte-aligned dual-array format
// sign_mantissa[M*K]: uint8_t per element = [sign 1][mantissa 7]
// groups[M*K/2]: nibble-packed = 2 groups per byte (low=even, high=odd)
// ALL loads are byte-aligned → ZERO read amplification vs 1.33x with 12-bit packed
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v2(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // Pointer setup: row-major layout
    const uint8_t* sm_ptr  = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr  = groups + (int64_t)row * K / 2 + tid / 2;
    const char*    act_ptr = (const char*)(activations + tid);

    // Each thread always accesses same nibble (tid parity is constant since 256 is even)
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Load sign+mantissa (4 byte-aligned loads)
        uint8_t sm0 = sm_ptr[0];
        uint8_t sm1 = sm_ptr[256];
        uint8_t sm2 = sm_ptr[512];
        uint8_t sm3 = sm_ptr[768];

        // Load group nibbles (4 byte-aligned loads, each shared between 2 elements)
        uint8_t gb0 = gr_ptr[0];
        uint8_t gb1 = gr_ptr[128];
        uint8_t gb2 = gr_ptr[256];
        uint8_t gb3 = gr_ptr[384];

        // Extract group from correct nibble
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
        uint32_t g2 = is_odd ? (gb2 >> 4) : (gb2 & 0xF);
        uint32_t g3 = is_odd ? (gb3 >> 4) : (gb3 & 0xF);

        // Decode to BF16 → float (pure ALU, no memory)
        union { uint32_t u; float f; } c0, c1, c2, c3;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
        c2.u = (((uint32_t)(sm2 >> 7)) << 15 | ((uint32_t)(base_exp + g2)) << 7 | (sm2 & 0x7F)) << 16;
        c3.u = (((uint32_t)(sm3 >> 7)) << 15 | ((uint32_t)(base_exp + g3)) << 7 | (sm3 & 0x7F)) << 16;

        // FMA with activations
        sum0 += c0.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += c1.f * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += c2.f * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += c3.f * bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        sm_ptr  += 1024;   // 256 * 4 bytes
        gr_ptr  += 512;    // 256 * 4 / 2 bytes
        act_ptr += 2048;   // 256 * 4 * 2 bytes (int16)
    }
    // Tail loop
    for (; col < K; col += WORKGROUP_SIZE) {
        uint8_t sm = sm_ptr[0];
        uint8_t gb = gr_ptr[0];
        uint32_t g = is_odd ? (gb >> 4) : (gb & 0xF);
        union { uint32_t u; float f; } c;
        c.u = (((uint32_t)(sm >> 7)) << 15 | ((uint32_t)(base_exp + g)) << 7 | (sm & 0x7F)) << 16;
        sum0 += c.f * bf16_to_float(*(const int16_t*)(act_ptr));
        sm_ptr  += 256;
        gr_ptr  += 128;
        act_ptr += 512;
    }

    // Warp reduction + output (same as v2)
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid / WARP_SIZE] = sum;
    __syncthreads();
    if (tid == 0) { float s = 0; for (int w = 0; w < NUM_WARPS; w++) s += warp_sums[w]; output[row] = s; }
}

// ---------------------------------------------------------------------------
// Split12 multirow: 2 rows per block, shared activation reads
// Grid: (M/2) blocks. Each block computes 2 adjacent output rows.
// 25% fewer loads per element: 12 loads / 8 elements vs 8 loads / 4 elements
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v2_multirow(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ activations,     // [K] BF16
    float* __restrict__ output,                  // [M]
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];

    // Row 0 accumulators
    float sum_a0 = 0.0f, sum_a1 = 0.0f, sum_a2 = 0.0f, sum_a3 = 0.0f;
    // Row 1 accumulators
    float sum_b0 = 0.0f, sum_b1 = 0.0f, sum_b2 = 0.0f, sum_b3 = 0.0f;

    const int has_row1 = (row1 < M);

    // Pointer setup for row 0
    const uint8_t* sm_ptr0  = sign_mantissa + (int64_t)row0 * K + tid;
    const uint8_t* gr_ptr0  = groups + (int64_t)row0 * K / 2 + tid / 2;
    // Pointer setup for row 1
    const uint8_t* sm_ptr1  = sign_mantissa + (int64_t)row1 * K + tid;
    const uint8_t* gr_ptr1  = groups + (int64_t)row1 * K / 2 + tid / 2;
    // Shared activation pointer
    const char*    act_ptr  = (const char*)(activations + tid);

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Load activations ONCE (shared between both rows)
        float a0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float a1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float a2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float a3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        // Row 0: load sign+mantissa and groups
        uint8_t sm0_0 = sm_ptr0[0];
        uint8_t sm0_1 = sm_ptr0[256];
        uint8_t sm0_2 = sm_ptr0[512];
        uint8_t sm0_3 = sm_ptr0[768];
        uint8_t gb0_0 = gr_ptr0[0];
        uint8_t gb0_1 = gr_ptr0[128];
        uint8_t gb0_2 = gr_ptr0[256];
        uint8_t gb0_3 = gr_ptr0[384];

        uint32_t g0_0 = is_odd ? (gb0_0 >> 4) : (gb0_0 & 0xF);
        uint32_t g0_1 = is_odd ? (gb0_1 >> 4) : (gb0_1 & 0xF);
        uint32_t g0_2 = is_odd ? (gb0_2 >> 4) : (gb0_2 & 0xF);
        uint32_t g0_3 = is_odd ? (gb0_3 >> 4) : (gb0_3 & 0xF);

        union { uint32_t u; float f; } c0_0, c0_1, c0_2, c0_3;
        c0_0.u = (((uint32_t)(sm0_0 >> 7)) << 15 | ((uint32_t)(base_exp + g0_0)) << 7 | (sm0_0 & 0x7F)) << 16;
        c0_1.u = (((uint32_t)(sm0_1 >> 7)) << 15 | ((uint32_t)(base_exp + g0_1)) << 7 | (sm0_1 & 0x7F)) << 16;
        c0_2.u = (((uint32_t)(sm0_2 >> 7)) << 15 | ((uint32_t)(base_exp + g0_2)) << 7 | (sm0_2 & 0x7F)) << 16;
        c0_3.u = (((uint32_t)(sm0_3 >> 7)) << 15 | ((uint32_t)(base_exp + g0_3)) << 7 | (sm0_3 & 0x7F)) << 16;

        sum_a0 += c0_0.f * a0;
        sum_a1 += c0_1.f * a1;
        sum_a2 += c0_2.f * a2;
        sum_a3 += c0_3.f * a3;

        // Row 1: load sign+mantissa and groups
        if (has_row1) {
            uint8_t sm1_0 = sm_ptr1[0];
            uint8_t sm1_1 = sm_ptr1[256];
            uint8_t sm1_2 = sm_ptr1[512];
            uint8_t sm1_3 = sm_ptr1[768];
            uint8_t gb1_0 = gr_ptr1[0];
            uint8_t gb1_1 = gr_ptr1[128];
            uint8_t gb1_2 = gr_ptr1[256];
            uint8_t gb1_3 = gr_ptr1[384];

            uint32_t g1_0 = is_odd ? (gb1_0 >> 4) : (gb1_0 & 0xF);
            uint32_t g1_1 = is_odd ? (gb1_1 >> 4) : (gb1_1 & 0xF);
            uint32_t g1_2 = is_odd ? (gb1_2 >> 4) : (gb1_2 & 0xF);
            uint32_t g1_3 = is_odd ? (gb1_3 >> 4) : (gb1_3 & 0xF);

            union { uint32_t u; float f; } c1_0, c1_1, c1_2, c1_3;
            c1_0.u = (((uint32_t)(sm1_0 >> 7)) << 15 | ((uint32_t)(base_exp + g1_0)) << 7 | (sm1_0 & 0x7F)) << 16;
            c1_1.u = (((uint32_t)(sm1_1 >> 7)) << 15 | ((uint32_t)(base_exp + g1_1)) << 7 | (sm1_1 & 0x7F)) << 16;
            c1_2.u = (((uint32_t)(sm1_2 >> 7)) << 15 | ((uint32_t)(base_exp + g1_2)) << 7 | (sm1_2 & 0x7F)) << 16;
            c1_3.u = (((uint32_t)(sm1_3 >> 7)) << 15 | ((uint32_t)(base_exp + g1_3)) << 7 | (sm1_3 & 0x7F)) << 16;

            sum_b0 += c1_0.f * a0;
            sum_b1 += c1_1.f * a1;
            sum_b2 += c1_2.f * a2;
            sum_b3 += c1_3.f * a3;
        }

        sm_ptr0 += 1024;
        gr_ptr0 += 512;
        sm_ptr1 += 1024;
        gr_ptr1 += 512;
        act_ptr += 2048;
    }
    // Tail loop
    for (; col < K; col += WORKGROUP_SIZE) {
        float a = bf16_to_float(*(const int16_t*)(act_ptr));
        uint8_t sm0 = sm_ptr0[0];
        uint8_t gb0 = gr_ptr0[0];
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        union { uint32_t u; float f; } c0;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        sum_a0 += c0.f * a;

        if (has_row1) {
            uint8_t sm1 = sm_ptr1[0];
            uint8_t gb1 = gr_ptr1[0];
            uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
            union { uint32_t u; float f; } c1;
            c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
            sum_b0 += c1.f * a;
        }

        sm_ptr0 += 256;
        gr_ptr0 += 128;
        sm_ptr1 += 256;
        gr_ptr1 += 128;
        act_ptr += 512;
    }

    // Warp reduction for row 0
    float sum0 = sum_a0 + sum_a1 + sum_a2 + sum_a3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid / WARP_SIZE] = sum0;

    // Warp reduction for row 1
    float sum1 = sum_b0 + sum_b1 + sum_b2 + sum_b3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[NUM_WARPS + (tid / WARP_SIZE)] = sum1;

    __syncthreads();
    if (tid == 0) {
        float s0 = 0, s1 = 0;
        for (int w = 0; w < NUM_WARPS; w++) { s0 += warp_sums[w]; s1 += warp_sums[NUM_WARPS + w]; }
        output[row0] = s0;
        if (has_row1)
            output[row1] = s1;
    }
}

// ---------------------------------------------------------------------------
// Split12 dual matvec: process 2 tensors (gate + up) sharing activation reads
// Each block handles 1 row for BOTH tensors. Activation loaded once, 2x FMA.
// Saves ~25% HBM bandwidth on the two largest FFN tensors.
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v2_dual(
    const uint8_t* __restrict__ sm_a, const uint8_t* __restrict__ gr_a, int base_exp_a,  // tensor A (gate)
    const uint8_t* __restrict__ sm_b, const uint8_t* __restrict__ gr_b, int base_exp_b,  // tensor B (up)
    const int16_t* __restrict__ activations,    // shared BF16 input
    float*         __restrict__ output_a,
    float*         __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[NUM_WARPS];
    __shared__ float warp_sums_b[NUM_WARPS];

    float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;

    // Pointer setup: row-major layout for both tensors
    const uint8_t* sma_ptr = sm_a + (int64_t)row * K + tid;
    const uint8_t* gra_ptr = gr_a + (int64_t)row * K / 2 + tid / 2;
    const uint8_t* smb_ptr = sm_b + (int64_t)row * K + tid;
    const uint8_t* grb_ptr = gr_b + (int64_t)row * K / 2 + tid / 2;
    const char*    act_ptr = (const char*)(activations + tid);

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        // Load activations ONCE (shared between both tensors)
        float af0 = bf16_to_float(*(const int16_t*)(act_ptr));
        float af1 = bf16_to_float(*(const int16_t*)(act_ptr + 512));
        float af2 = bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        float af3 = bf16_to_float(*(const int16_t*)(act_ptr + 1536));

        // Tensor A: load sign+mantissa and groups
        uint8_t sma0 = sma_ptr[0];
        uint8_t sma1 = sma_ptr[256];
        uint8_t sma2 = sma_ptr[512];
        uint8_t sma3 = sma_ptr[768];
        uint8_t gba0 = gra_ptr[0];
        uint8_t gba1 = gra_ptr[128];
        uint8_t gba2 = gra_ptr[256];
        uint8_t gba3 = gra_ptr[384];

        uint32_t ga0 = is_odd ? (gba0 >> 4) : (gba0 & 0xF);
        uint32_t ga1 = is_odd ? (gba1 >> 4) : (gba1 & 0xF);
        uint32_t ga2 = is_odd ? (gba2 >> 4) : (gba2 & 0xF);
        uint32_t ga3 = is_odd ? (gba3 >> 4) : (gba3 & 0xF);

        union { uint32_t u; float f; } ca0, ca1, ca2, ca3;
        ca0.u = (((uint32_t)(sma0 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga0)) << 7 | (sma0 & 0x7F)) << 16;
        ca1.u = (((uint32_t)(sma1 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga1)) << 7 | (sma1 & 0x7F)) << 16;
        ca2.u = (((uint32_t)(sma2 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga2)) << 7 | (sma2 & 0x7F)) << 16;
        ca3.u = (((uint32_t)(sma3 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga3)) << 7 | (sma3 & 0x7F)) << 16;

        sa0 += ca0.f * af0;
        sa1 += ca1.f * af1;
        sa2 += ca2.f * af2;
        sa3 += ca3.f * af3;

        // Tensor B: load sign+mantissa and groups
        uint8_t smb0 = smb_ptr[0];
        uint8_t smb1 = smb_ptr[256];
        uint8_t smb2 = smb_ptr[512];
        uint8_t smb3 = smb_ptr[768];
        uint8_t gbb0 = grb_ptr[0];
        uint8_t gbb1 = grb_ptr[128];
        uint8_t gbb2 = grb_ptr[256];
        uint8_t gbb3 = grb_ptr[384];

        uint32_t gb0 = is_odd ? (gbb0 >> 4) : (gbb0 & 0xF);
        uint32_t gb1 = is_odd ? (gbb1 >> 4) : (gbb1 & 0xF);
        uint32_t gb2 = is_odd ? (gbb2 >> 4) : (gbb2 & 0xF);
        uint32_t gb3 = is_odd ? (gbb3 >> 4) : (gbb3 & 0xF);

        union { uint32_t u; float f; } cb0, cb1, cb2, cb3;
        cb0.u = (((uint32_t)(smb0 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb0)) << 7 | (smb0 & 0x7F)) << 16;
        cb1.u = (((uint32_t)(smb1 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb1)) << 7 | (smb1 & 0x7F)) << 16;
        cb2.u = (((uint32_t)(smb2 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb2)) << 7 | (smb2 & 0x7F)) << 16;
        cb3.u = (((uint32_t)(smb3 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb3)) << 7 | (smb3 & 0x7F)) << 16;

        sb0 += cb0.f * af0;
        sb1 += cb1.f * af1;
        sb2 += cb2.f * af2;
        sb3 += cb3.f * af3;

        sma_ptr += 1024;
        gra_ptr += 512;
        smb_ptr += 1024;
        grb_ptr += 512;
        act_ptr += 2048;
    }
    // Tail loop
    for (; col < K; col += WORKGROUP_SIZE) {
        float af = bf16_to_float(*(const int16_t*)(act_ptr));

        uint8_t sma = sma_ptr[0];
        uint8_t gba = gra_ptr[0];
        uint32_t ga = is_odd ? (gba >> 4) : (gba & 0xF);
        union { uint32_t u; float f; } ca;
        ca.u = (((uint32_t)(sma >> 7)) << 15 | ((uint32_t)(base_exp_a + ga)) << 7 | (sma & 0x7F)) << 16;
        sa0 += ca.f * af;

        uint8_t smb = smb_ptr[0];
        uint8_t gbb = grb_ptr[0];
        uint32_t gb = is_odd ? (gbb >> 4) : (gbb & 0xF);
        union { uint32_t u; float f; } cb;
        cb.u = (((uint32_t)(smb >> 7)) << 15 | ((uint32_t)(base_exp_b + gb)) << 7 | (smb & 0x7F)) << 16;
        sb0 += cb.f * af;

        sma_ptr += 256;
        gra_ptr += 128;
        smb_ptr += 256;
        grb_ptr += 128;
        act_ptr += 512;
    }

    // Reduction for tensor A
    float sum_a = sa0 + sa1 + sa2 + sa3;
    // Reduction for tensor B
    float sum_b = sb0 + sb1 + sb2 + sb3;

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums_a[tid / WARP_SIZE] = sum_a;
        warp_sums_b[tid / WARP_SIZE] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0, sb = 0;
        for (int w = 0; w < NUM_WARPS; w++) { sa += warp_sums_a[w]; sb += warp_sums_b[w]; }
        output_a[row] = sa; output_b[row] = sb;
    }
}

// ===========================================================================
// VARIANT A: __ldg() read-only cache path (NVIDIA texture cache bypass L1)
// On AMD: __ldg is a no-op, falls back to normal load
// ===========================================================================
#ifdef TURBO_NVIDIA
#define TURBO_LDG(ptr) __ldg(ptr)
#else
#define TURBO_LDG(ptr) (*(ptr))
#endif

// ---------------------------------------------------------------------------
// Split12 multirow v3_ldg: __ldg() on all weight and activation reads
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_ldg_multirow(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];

    float sum_a0 = 0.0f, sum_a1 = 0.0f, sum_a2 = 0.0f, sum_a3 = 0.0f;
    float sum_b0 = 0.0f, sum_b1 = 0.0f, sum_b2 = 0.0f, sum_b3 = 0.0f;

    const int has_row1 = (row1 < M);

    const uint8_t* sm_ptr0 = sign_mantissa + (int64_t)row0 * K + tid;
    const uint8_t* gr_ptr0 = groups + (int64_t)row0 * K / 2 + tid / 2;
    const uint8_t* sm_ptr1 = sign_mantissa + (int64_t)row1 * K + tid;
    const uint8_t* gr_ptr1 = groups + (int64_t)row1 * K / 2 + tid / 2;
    const int16_t* act_base = activations + tid;

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        float a0 = bf16_to_float(TURBO_LDG(act_base));
        float a1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float a2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float a3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));

        uint8_t sm0_0 = TURBO_LDG(sm_ptr0);
        uint8_t sm0_1 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE);
        uint8_t sm0_2 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 2);
        uint8_t sm0_3 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 3);
        uint8_t gb0_0 = TURBO_LDG(gr_ptr0);
        uint8_t gb0_1 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE / 2);
        uint8_t gb0_2 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE);
        uint8_t gb0_3 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 3 / 2);

        uint32_t g0_0 = is_odd ? (gb0_0 >> 4) : (gb0_0 & 0xF);
        uint32_t g0_1 = is_odd ? (gb0_1 >> 4) : (gb0_1 & 0xF);
        uint32_t g0_2 = is_odd ? (gb0_2 >> 4) : (gb0_2 & 0xF);
        uint32_t g0_3 = is_odd ? (gb0_3 >> 4) : (gb0_3 & 0xF);

        union { uint32_t u; float f; } c0_0, c0_1, c0_2, c0_3;
        c0_0.u = (((uint32_t)(sm0_0 >> 7)) << 15 | ((uint32_t)(base_exp + g0_0)) << 7 | (sm0_0 & 0x7F)) << 16;
        c0_1.u = (((uint32_t)(sm0_1 >> 7)) << 15 | ((uint32_t)(base_exp + g0_1)) << 7 | (sm0_1 & 0x7F)) << 16;
        c0_2.u = (((uint32_t)(sm0_2 >> 7)) << 15 | ((uint32_t)(base_exp + g0_2)) << 7 | (sm0_2 & 0x7F)) << 16;
        c0_3.u = (((uint32_t)(sm0_3 >> 7)) << 15 | ((uint32_t)(base_exp + g0_3)) << 7 | (sm0_3 & 0x7F)) << 16;

        sum_a0 += c0_0.f * a0;
        sum_a1 += c0_1.f * a1;
        sum_a2 += c0_2.f * a2;
        sum_a3 += c0_3.f * a3;

        if (has_row1) {
            uint8_t sm1_0 = TURBO_LDG(sm_ptr1);
            uint8_t sm1_1 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE);
            uint8_t sm1_2 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 2);
            uint8_t sm1_3 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 3);
            uint8_t gb1_0 = TURBO_LDG(gr_ptr1);
            uint8_t gb1_1 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE / 2);
            uint8_t gb1_2 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE);
            uint8_t gb1_3 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 3 / 2);

            uint32_t g1_0 = is_odd ? (gb1_0 >> 4) : (gb1_0 & 0xF);
            uint32_t g1_1 = is_odd ? (gb1_1 >> 4) : (gb1_1 & 0xF);
            uint32_t g1_2 = is_odd ? (gb1_2 >> 4) : (gb1_2 & 0xF);
            uint32_t g1_3 = is_odd ? (gb1_3 >> 4) : (gb1_3 & 0xF);

            union { uint32_t u; float f; } c1_0, c1_1, c1_2, c1_3;
            c1_0.u = (((uint32_t)(sm1_0 >> 7)) << 15 | ((uint32_t)(base_exp + g1_0)) << 7 | (sm1_0 & 0x7F)) << 16;
            c1_1.u = (((uint32_t)(sm1_1 >> 7)) << 15 | ((uint32_t)(base_exp + g1_1)) << 7 | (sm1_1 & 0x7F)) << 16;
            c1_2.u = (((uint32_t)(sm1_2 >> 7)) << 15 | ((uint32_t)(base_exp + g1_2)) << 7 | (sm1_2 & 0x7F)) << 16;
            c1_3.u = (((uint32_t)(sm1_3 >> 7)) << 15 | ((uint32_t)(base_exp + g1_3)) << 7 | (sm1_3 & 0x7F)) << 16;

            sum_b0 += c1_0.f * a0;
            sum_b1 += c1_1.f * a1;
            sum_b2 += c1_2.f * a2;
            sum_b3 += c1_3.f * a3;
        }

        sm_ptr0 += WORKGROUP_SIZE * 4;
        gr_ptr0 += WORKGROUP_SIZE * 4 / 2;
        sm_ptr1 += WORKGROUP_SIZE * 4;
        gr_ptr1 += WORKGROUP_SIZE * 4 / 2;
        act_base += WORKGROUP_SIZE * 4;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        float a = bf16_to_float(TURBO_LDG(act_base));
        uint8_t sm0 = TURBO_LDG(sm_ptr0);
        uint8_t gb0 = TURBO_LDG(gr_ptr0);
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        union { uint32_t u; float f; } c0;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        sum_a0 += c0.f * a;
        if (has_row1) {
            uint8_t sm1 = TURBO_LDG(sm_ptr1);
            uint8_t gb1 = TURBO_LDG(gr_ptr1);
            uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
            union { uint32_t u; float f; } c1;
            c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
            sum_b0 += c1.f * a;
        }
        sm_ptr0 += WORKGROUP_SIZE;
        gr_ptr0 += WORKGROUP_SIZE / 2;
        sm_ptr1 += WORKGROUP_SIZE;
        gr_ptr1 += WORKGROUP_SIZE / 2;
        act_base += WORKGROUP_SIZE;
    }

    float sum0 = sum_a0 + sum_a1 + sum_a2 + sum_a3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid / WARP_SIZE] = sum0;

    float sum1 = sum_b0 + sum_b1 + sum_b2 + sum_b3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[NUM_WARPS + (tid / WARP_SIZE)] = sum1;

    __syncthreads();
    if (tid == 0) {
        float s0 = 0, s1 = 0;
        for (int w = 0; w < NUM_WARPS; w++) { s0 += warp_sums[w]; s1 += warp_sums[NUM_WARPS + w]; }
        output[row0] = s0;
        if (has_row1) output[row1] = s1;
    }
}

// ===========================================================================
// VARIANT B: 8-element unroll per iteration (halves loop count)
// For K=4096, WG=256: loop runs 2 iterations instead of 4
// ===========================================================================
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_unroll8_multirow(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];

    float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
    float sa4 = 0.0f, sa5 = 0.0f, sa6 = 0.0f, sa7 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;
    float sb4 = 0.0f, sb5 = 0.0f, sb6 = 0.0f, sb7 = 0.0f;

    const int has_row1 = (row1 < M);

    const uint8_t* sm_ptr0 = sign_mantissa + (int64_t)row0 * K + tid;
    const uint8_t* gr_ptr0 = groups + (int64_t)row0 * K / 2 + tid / 2;
    const uint8_t* sm_ptr1 = sign_mantissa + (int64_t)row1 * K + tid;
    const uint8_t* gr_ptr1 = groups + (int64_t)row1 * K / 2 + tid / 2;
    const int16_t* act_base = activations + tid;

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 7 < K; col += WORKGROUP_SIZE * 8) {
        float a0 = bf16_to_float(TURBO_LDG(act_base));
        float a1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float a2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float a3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));
        float a4 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 4));
        float a5 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 5));
        float a6 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 6));
        float a7 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 7));

        uint8_t sm0_0 = TURBO_LDG(sm_ptr0);
        uint8_t sm0_1 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE);
        uint8_t sm0_2 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 2);
        uint8_t sm0_3 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 3);
        uint8_t sm0_4 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 4);
        uint8_t sm0_5 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 5);
        uint8_t sm0_6 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 6);
        uint8_t sm0_7 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 7);
        uint8_t gb0_0 = TURBO_LDG(gr_ptr0);
        uint8_t gb0_1 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE / 2);
        uint8_t gb0_2 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE);
        uint8_t gb0_3 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 3 / 2);
        uint8_t gb0_4 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 2);
        uint8_t gb0_5 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 5 / 2);
        uint8_t gb0_6 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 3);
        uint8_t gb0_7 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 7 / 2);

        uint32_t g0_0 = is_odd ? (gb0_0 >> 4) : (gb0_0 & 0xF);
        uint32_t g0_1 = is_odd ? (gb0_1 >> 4) : (gb0_1 & 0xF);
        uint32_t g0_2 = is_odd ? (gb0_2 >> 4) : (gb0_2 & 0xF);
        uint32_t g0_3 = is_odd ? (gb0_3 >> 4) : (gb0_3 & 0xF);
        uint32_t g0_4 = is_odd ? (gb0_4 >> 4) : (gb0_4 & 0xF);
        uint32_t g0_5 = is_odd ? (gb0_5 >> 4) : (gb0_5 & 0xF);
        uint32_t g0_6 = is_odd ? (gb0_6 >> 4) : (gb0_6 & 0xF);
        uint32_t g0_7 = is_odd ? (gb0_7 >> 4) : (gb0_7 & 0xF);

        union { uint32_t u; float f; } c0, c1, c2, c3, c4, c5, c6, c7;
        c0.u = (((uint32_t)(sm0_0 >> 7)) << 15 | ((uint32_t)(base_exp + g0_0)) << 7 | (sm0_0 & 0x7F)) << 16;
        c1.u = (((uint32_t)(sm0_1 >> 7)) << 15 | ((uint32_t)(base_exp + g0_1)) << 7 | (sm0_1 & 0x7F)) << 16;
        c2.u = (((uint32_t)(sm0_2 >> 7)) << 15 | ((uint32_t)(base_exp + g0_2)) << 7 | (sm0_2 & 0x7F)) << 16;
        c3.u = (((uint32_t)(sm0_3 >> 7)) << 15 | ((uint32_t)(base_exp + g0_3)) << 7 | (sm0_3 & 0x7F)) << 16;
        c4.u = (((uint32_t)(sm0_4 >> 7)) << 15 | ((uint32_t)(base_exp + g0_4)) << 7 | (sm0_4 & 0x7F)) << 16;
        c5.u = (((uint32_t)(sm0_5 >> 7)) << 15 | ((uint32_t)(base_exp + g0_5)) << 7 | (sm0_5 & 0x7F)) << 16;
        c6.u = (((uint32_t)(sm0_6 >> 7)) << 15 | ((uint32_t)(base_exp + g0_6)) << 7 | (sm0_6 & 0x7F)) << 16;
        c7.u = (((uint32_t)(sm0_7 >> 7)) << 15 | ((uint32_t)(base_exp + g0_7)) << 7 | (sm0_7 & 0x7F)) << 16;

        sa0 += c0.f * a0; sa1 += c1.f * a1; sa2 += c2.f * a2; sa3 += c3.f * a3;
        sa4 += c4.f * a4; sa5 += c5.f * a5; sa6 += c6.f * a6; sa7 += c7.f * a7;

        if (has_row1) {
            uint8_t sm1_0 = TURBO_LDG(sm_ptr1);
            uint8_t sm1_1 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE);
            uint8_t sm1_2 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 2);
            uint8_t sm1_3 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 3);
            uint8_t sm1_4 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 4);
            uint8_t sm1_5 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 5);
            uint8_t sm1_6 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 6);
            uint8_t sm1_7 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 7);
            uint8_t gb1_0 = TURBO_LDG(gr_ptr1);
            uint8_t gb1_1 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE / 2);
            uint8_t gb1_2 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE);
            uint8_t gb1_3 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 3 / 2);
            uint8_t gb1_4 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 2);
            uint8_t gb1_5 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 5 / 2);
            uint8_t gb1_6 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 3);
            uint8_t gb1_7 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 7 / 2);

            uint32_t g1_0 = is_odd ? (gb1_0 >> 4) : (gb1_0 & 0xF);
            uint32_t g1_1 = is_odd ? (gb1_1 >> 4) : (gb1_1 & 0xF);
            uint32_t g1_2 = is_odd ? (gb1_2 >> 4) : (gb1_2 & 0xF);
            uint32_t g1_3 = is_odd ? (gb1_3 >> 4) : (gb1_3 & 0xF);
            uint32_t g1_4 = is_odd ? (gb1_4 >> 4) : (gb1_4 & 0xF);
            uint32_t g1_5 = is_odd ? (gb1_5 >> 4) : (gb1_5 & 0xF);
            uint32_t g1_6 = is_odd ? (gb1_6 >> 4) : (gb1_6 & 0xF);
            uint32_t g1_7 = is_odd ? (gb1_7 >> 4) : (gb1_7 & 0xF);

            union { uint32_t u; float f; } d0, d1, d2, d3, d4, d5, d6, d7;
            d0.u = (((uint32_t)(sm1_0 >> 7)) << 15 | ((uint32_t)(base_exp + g1_0)) << 7 | (sm1_0 & 0x7F)) << 16;
            d1.u = (((uint32_t)(sm1_1 >> 7)) << 15 | ((uint32_t)(base_exp + g1_1)) << 7 | (sm1_1 & 0x7F)) << 16;
            d2.u = (((uint32_t)(sm1_2 >> 7)) << 15 | ((uint32_t)(base_exp + g1_2)) << 7 | (sm1_2 & 0x7F)) << 16;
            d3.u = (((uint32_t)(sm1_3 >> 7)) << 15 | ((uint32_t)(base_exp + g1_3)) << 7 | (sm1_3 & 0x7F)) << 16;
            d4.u = (((uint32_t)(sm1_4 >> 7)) << 15 | ((uint32_t)(base_exp + g1_4)) << 7 | (sm1_4 & 0x7F)) << 16;
            d5.u = (((uint32_t)(sm1_5 >> 7)) << 15 | ((uint32_t)(base_exp + g1_5)) << 7 | (sm1_5 & 0x7F)) << 16;
            d6.u = (((uint32_t)(sm1_6 >> 7)) << 15 | ((uint32_t)(base_exp + g1_6)) << 7 | (sm1_6 & 0x7F)) << 16;
            d7.u = (((uint32_t)(sm1_7 >> 7)) << 15 | ((uint32_t)(base_exp + g1_7)) << 7 | (sm1_7 & 0x7F)) << 16;

            sb0 += d0.f * a0; sb1 += d1.f * a1; sb2 += d2.f * a2; sb3 += d3.f * a3;
            sb4 += d4.f * a4; sb5 += d5.f * a5; sb6 += d6.f * a6; sb7 += d7.f * a7;
        }

        sm_ptr0 += WORKGROUP_SIZE * 8;
        gr_ptr0 += WORKGROUP_SIZE * 8 / 2;
        sm_ptr1 += WORKGROUP_SIZE * 8;
        gr_ptr1 += WORKGROUP_SIZE * 8 / 2;
        act_base += WORKGROUP_SIZE * 8;
    }
    // 4-element tail
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        float a0 = bf16_to_float(TURBO_LDG(act_base));
        float a1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float a2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float a3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));

        uint8_t sm0_0 = TURBO_LDG(sm_ptr0); uint8_t sm0_1 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE);
        uint8_t sm0_2 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 2); uint8_t sm0_3 = TURBO_LDG(sm_ptr0 + WORKGROUP_SIZE * 3);
        uint8_t gb0_0 = TURBO_LDG(gr_ptr0); uint8_t gb0_1 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE / 2);
        uint8_t gb0_2 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE); uint8_t gb0_3 = TURBO_LDG(gr_ptr0 + WORKGROUP_SIZE * 3 / 2);

        uint32_t g0_0 = is_odd ? (gb0_0 >> 4) : (gb0_0 & 0xF);
        uint32_t g0_1 = is_odd ? (gb0_1 >> 4) : (gb0_1 & 0xF);
        uint32_t g0_2 = is_odd ? (gb0_2 >> 4) : (gb0_2 & 0xF);
        uint32_t g0_3 = is_odd ? (gb0_3 >> 4) : (gb0_3 & 0xF);

        union { uint32_t u; float f; } c0, c1, c2, c3;
        c0.u = (((uint32_t)(sm0_0 >> 7)) << 15 | ((uint32_t)(base_exp + g0_0)) << 7 | (sm0_0 & 0x7F)) << 16;
        c1.u = (((uint32_t)(sm0_1 >> 7)) << 15 | ((uint32_t)(base_exp + g0_1)) << 7 | (sm0_1 & 0x7F)) << 16;
        c2.u = (((uint32_t)(sm0_2 >> 7)) << 15 | ((uint32_t)(base_exp + g0_2)) << 7 | (sm0_2 & 0x7F)) << 16;
        c3.u = (((uint32_t)(sm0_3 >> 7)) << 15 | ((uint32_t)(base_exp + g0_3)) << 7 | (sm0_3 & 0x7F)) << 16;

        sa0 += c0.f * a0; sa1 += c1.f * a1; sa2 += c2.f * a2; sa3 += c3.f * a3;

        if (has_row1) {
            uint8_t sm1_0 = TURBO_LDG(sm_ptr1); uint8_t sm1_1 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE);
            uint8_t sm1_2 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 2); uint8_t sm1_3 = TURBO_LDG(sm_ptr1 + WORKGROUP_SIZE * 3);
            uint8_t gb1_0 = TURBO_LDG(gr_ptr1); uint8_t gb1_1 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE / 2);
            uint8_t gb1_2 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE); uint8_t gb1_3 = TURBO_LDG(gr_ptr1 + WORKGROUP_SIZE * 3 / 2);

            uint32_t g1_0 = is_odd ? (gb1_0 >> 4) : (gb1_0 & 0xF);
            uint32_t g1_1 = is_odd ? (gb1_1 >> 4) : (gb1_1 & 0xF);
            uint32_t g1_2 = is_odd ? (gb1_2 >> 4) : (gb1_2 & 0xF);
            uint32_t g1_3 = is_odd ? (gb1_3 >> 4) : (gb1_3 & 0xF);

            union { uint32_t u; float f; } d0, d1, d2, d3;
            d0.u = (((uint32_t)(sm1_0 >> 7)) << 15 | ((uint32_t)(base_exp + g1_0)) << 7 | (sm1_0 & 0x7F)) << 16;
            d1.u = (((uint32_t)(sm1_1 >> 7)) << 15 | ((uint32_t)(base_exp + g1_1)) << 7 | (sm1_1 & 0x7F)) << 16;
            d2.u = (((uint32_t)(sm1_2 >> 7)) << 15 | ((uint32_t)(base_exp + g1_2)) << 7 | (sm1_2 & 0x7F)) << 16;
            d3.u = (((uint32_t)(sm1_3 >> 7)) << 15 | ((uint32_t)(base_exp + g1_3)) << 7 | (sm1_3 & 0x7F)) << 16;

            sb0 += d0.f * a0; sb1 += d1.f * a1; sb2 += d2.f * a2; sb3 += d3.f * a3;
        }

        sm_ptr0 += WORKGROUP_SIZE * 4; gr_ptr0 += WORKGROUP_SIZE * 4 / 2;
        sm_ptr1 += WORKGROUP_SIZE * 4; gr_ptr1 += WORKGROUP_SIZE * 4 / 2;
        act_base += WORKGROUP_SIZE * 4;
    }
    // Single-element tail
    for (; col < K; col += WORKGROUP_SIZE) {
        float a = bf16_to_float(TURBO_LDG(act_base));
        uint8_t sm0 = TURBO_LDG(sm_ptr0); uint8_t gb0 = TURBO_LDG(gr_ptr0);
        uint32_t g0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);
        union { uint32_t u; float f; } c0;
        c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
        sa0 += c0.f * a;
        if (has_row1) {
            uint8_t sm1 = TURBO_LDG(sm_ptr1); uint8_t gb1 = TURBO_LDG(gr_ptr1);
            uint32_t g1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);
            union { uint32_t u; float f; } c1;
            c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
            sb0 += c1.f * a;
        }
        sm_ptr0 += WORKGROUP_SIZE; gr_ptr0 += WORKGROUP_SIZE / 2;
        sm_ptr1 += WORKGROUP_SIZE; gr_ptr1 += WORKGROUP_SIZE / 2;
        act_base += WORKGROUP_SIZE;
    }

    float sum0 = sa0 + sa1 + sa2 + sa3 + sa4 + sa5 + sa6 + sa7;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid / WARP_SIZE] = sum0;

    float sum1 = sb0 + sb1 + sb2 + sb3 + sb4 + sb5 + sb6 + sb7;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[NUM_WARPS + (tid / WARP_SIZE)] = sum1;

    __syncthreads();
    if (tid == 0) {
        float s0 = 0, s1 = 0;
        for (int w = 0; w < NUM_WARPS; w++) { s0 += warp_sums[w]; s1 += warp_sums[NUM_WARPS + w]; }
        output[row0] = s0;
        if (has_row1) output[row1] = s1;
    }
}

// ===========================================================================
// VARIANT C: Vectorized uint32_t loads for sign_mantissa (4 bytes at once)
// and uint16_t loads for groups (2 bytes = 4 nibbles at once)
// Thread tid loads 4 consecutive elements via wide loads instead of strided bytes
// ===========================================================================

// Helper: decode 4 elements from a uint32_t of sm bytes + uint16_t of gr bytes
// Element layout: b0(even)->low nibble, b1(odd)->high nibble, b2(even)->low, b3(odd)->high
__device__ __forceinline__
void decode_vec4(uint32_t sm_vec, uint16_t gr_vec, int base_exp,
                 float& w0, float& w1, float& w2, float& w3) {
    uint8_t sm0 = (uint8_t)(sm_vec);
    uint8_t sm1 = (uint8_t)(sm_vec >> 8);
    uint8_t sm2 = (uint8_t)(sm_vec >> 16);
    uint8_t sm3 = (uint8_t)(sm_vec >> 24);
    uint8_t gr01 = (uint8_t)(gr_vec);
    uint8_t gr23 = (uint8_t)(gr_vec >> 8);

    uint32_t g0 = gr01 & 0xF;
    uint32_t g1 = gr01 >> 4;
    uint32_t g2 = gr23 & 0xF;
    uint32_t g3 = gr23 >> 4;

    union { uint32_t u; float f; } c0, c1, c2, c3;
    c0.u = (((uint32_t)(sm0 >> 7)) << 15 | ((uint32_t)(base_exp + g0)) << 7 | (sm0 & 0x7F)) << 16;
    c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
    c2.u = (((uint32_t)(sm2 >> 7)) << 15 | ((uint32_t)(base_exp + g2)) << 7 | (sm2 & 0x7F)) << 16;
    c3.u = (((uint32_t)(sm3 >> 7)) << 15 | ((uint32_t)(base_exp + g3)) << 7 | (sm3 & 0x7F)) << 16;
    w0 = c0.f; w1 = c1.f; w2 = c2.f; w3 = c3.f;
}

__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_vec4_multirow(
    const uint8_t* __restrict__ sign_mantissa,
    const uint8_t* __restrict__ groups,
    int base_exp,
    const int16_t* __restrict__ activations,
    float* __restrict__ output,
    int M, int K)
{
    const int row0 = blockIdx.x * 2;
    const int row1 = row0 + 1;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[NUM_WARPS * 2];

    float sum_a0 = 0.0f, sum_a1 = 0.0f;
    float sum_b0 = 0.0f, sum_b1 = 0.0f;

    const int has_row1 = (row1 < M);

    // Vec4 addressing: thread tid loads 4 consecutive bytes at offset tid*4
    // With 256 threads, one pass covers 256*4 = 1024 elements
    const uint8_t* sm_base0 = sign_mantissa + (int64_t)row0 * K;
    const uint8_t* gr_base0 = groups + (int64_t)row0 * K / 2;
    const uint8_t* sm_base1 = sign_mantissa + (int64_t)row1 * K;
    const uint8_t* gr_base1 = groups + (int64_t)row1 * K / 2;

    int elem = tid * 4;
    for (; elem + WORKGROUP_SIZE * 4 - 1 < K; elem += WORKGROUP_SIZE * 4) {
        uint32_t sm_v0 = *(const uint32_t*)(sm_base0 + elem);
        uint16_t gr_v0 = *(const uint16_t*)(gr_base0 + elem / 2);
        float w0_0, w0_1, w0_2, w0_3;
        decode_vec4(sm_v0, gr_v0, base_exp, w0_0, w0_1, w0_2, w0_3);

        float a0 = bf16_to_float(TURBO_LDG(activations + elem));
        float a1 = bf16_to_float(TURBO_LDG(activations + elem + 1));
        float a2 = bf16_to_float(TURBO_LDG(activations + elem + 2));
        float a3 = bf16_to_float(TURBO_LDG(activations + elem + 3));

        sum_a0 += w0_0 * a0 + w0_1 * a1 + w0_2 * a2 + w0_3 * a3;

        if (has_row1) {
            uint32_t sm_v1 = *(const uint32_t*)(sm_base1 + elem);
            uint16_t gr_v1 = *(const uint16_t*)(gr_base1 + elem / 2);
            float w1_0, w1_1, w1_2, w1_3;
            decode_vec4(sm_v1, gr_v1, base_exp, w1_0, w1_1, w1_2, w1_3);
            sum_b0 += w1_0 * a0 + w1_1 * a1 + w1_2 * a2 + w1_3 * a3;
        }
    }
    // Scalar tail
    for (; elem < K; elem++) {
        int is_odd_e = elem & 1;
        uint8_t sm = sm_base0[elem];
        uint8_t gb = gr_base0[elem / 2];
        uint32_t g = is_odd_e ? (gb >> 4) : (gb & 0xF);
        union { uint32_t u; float f; } c;
        c.u = (((uint32_t)(sm >> 7)) << 15 | ((uint32_t)(base_exp + g)) << 7 | (sm & 0x7F)) << 16;
        float a = bf16_to_float(activations[elem]);
        sum_a0 += c.f * a;
        if (has_row1) {
            uint8_t sm1 = sm_base1[elem];
            uint8_t gb1 = gr_base1[elem / 2];
            uint32_t g1 = is_odd_e ? (gb1 >> 4) : (gb1 & 0xF);
            union { uint32_t u; float f; } c1;
            c1.u = (((uint32_t)(sm1 >> 7)) << 15 | ((uint32_t)(base_exp + g1)) << 7 | (sm1 & 0x7F)) << 16;
            sum_b0 += c1.f * a;
        }
    }

    float sum0 = sum_a0 + sum_a1;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum0 += __shfl_down(sum0, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid / WARP_SIZE] = sum0;

    float sum1 = sum_b0 + sum_b1;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[NUM_WARPS + (tid / WARP_SIZE)] = sum1;

    __syncthreads();
    if (tid == 0) {
        float s0 = 0, s1 = 0;
        for (int w = 0; w < NUM_WARPS; w++) { s0 += warp_sums[w]; s1 += warp_sums[NUM_WARPS + w]; }
        output[row0] = s0;
        if (has_row1) output[row1] = s1;
    }
}

// ===========================================================================
// DUAL kernel variants (gate+up fused, same 3 optimization strategies)
// ===========================================================================

// ---------------------------------------------------------------------------
// Dual v3_ldg: __ldg() on all reads
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_ldg_dual(
    const uint8_t* __restrict__ sm_a, const uint8_t* __restrict__ gr_a, int base_exp_a,
    const uint8_t* __restrict__ sm_b, const uint8_t* __restrict__ gr_b, int base_exp_b,
    const int16_t* __restrict__ activations,
    float*         __restrict__ output_a,
    float*         __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[NUM_WARPS];
    __shared__ float warp_sums_b[NUM_WARPS];

    float sa0 = 0.0f, sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f, sb2 = 0.0f, sb3 = 0.0f;

    const uint8_t* sma_ptr = sm_a + (int64_t)row * K + tid;
    const uint8_t* gra_ptr = gr_a + (int64_t)row * K / 2 + tid / 2;
    const uint8_t* smb_ptr = sm_b + (int64_t)row * K + tid;
    const uint8_t* grb_ptr = gr_b + (int64_t)row * K / 2 + tid / 2;
    const int16_t* act_base = activations + tid;

    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        float af0 = bf16_to_float(TURBO_LDG(act_base));
        float af1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float af2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float af3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));

        uint8_t sma0 = TURBO_LDG(sma_ptr);
        uint8_t sma1 = TURBO_LDG(sma_ptr + WORKGROUP_SIZE);
        uint8_t sma2 = TURBO_LDG(sma_ptr + WORKGROUP_SIZE * 2);
        uint8_t sma3 = TURBO_LDG(sma_ptr + WORKGROUP_SIZE * 3);
        uint8_t gba0 = TURBO_LDG(gra_ptr);
        uint8_t gba1 = TURBO_LDG(gra_ptr + WORKGROUP_SIZE / 2);
        uint8_t gba2 = TURBO_LDG(gra_ptr + WORKGROUP_SIZE);
        uint8_t gba3 = TURBO_LDG(gra_ptr + WORKGROUP_SIZE * 3 / 2);

        uint32_t ga0 = is_odd ? (gba0 >> 4) : (gba0 & 0xF);
        uint32_t ga1 = is_odd ? (gba1 >> 4) : (gba1 & 0xF);
        uint32_t ga2 = is_odd ? (gba2 >> 4) : (gba2 & 0xF);
        uint32_t ga3 = is_odd ? (gba3 >> 4) : (gba3 & 0xF);

        union { uint32_t u; float f; } ca0, ca1, ca2, ca3;
        ca0.u = (((uint32_t)(sma0 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga0)) << 7 | (sma0 & 0x7F)) << 16;
        ca1.u = (((uint32_t)(sma1 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga1)) << 7 | (sma1 & 0x7F)) << 16;
        ca2.u = (((uint32_t)(sma2 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga2)) << 7 | (sma2 & 0x7F)) << 16;
        ca3.u = (((uint32_t)(sma3 >> 7)) << 15 | ((uint32_t)(base_exp_a + ga3)) << 7 | (sma3 & 0x7F)) << 16;

        sa0 += ca0.f * af0; sa1 += ca1.f * af1; sa2 += ca2.f * af2; sa3 += ca3.f * af3;

        uint8_t smb0 = TURBO_LDG(smb_ptr);
        uint8_t smb1 = TURBO_LDG(smb_ptr + WORKGROUP_SIZE);
        uint8_t smb2 = TURBO_LDG(smb_ptr + WORKGROUP_SIZE * 2);
        uint8_t smb3 = TURBO_LDG(smb_ptr + WORKGROUP_SIZE * 3);
        uint8_t gbb0 = TURBO_LDG(grb_ptr);
        uint8_t gbb1 = TURBO_LDG(grb_ptr + WORKGROUP_SIZE / 2);
        uint8_t gbb2 = TURBO_LDG(grb_ptr + WORKGROUP_SIZE);
        uint8_t gbb3 = TURBO_LDG(grb_ptr + WORKGROUP_SIZE * 3 / 2);

        uint32_t gb0 = is_odd ? (gbb0 >> 4) : (gbb0 & 0xF);
        uint32_t gb1 = is_odd ? (gbb1 >> 4) : (gbb1 & 0xF);
        uint32_t gb2 = is_odd ? (gbb2 >> 4) : (gbb2 & 0xF);
        uint32_t gb3 = is_odd ? (gbb3 >> 4) : (gbb3 & 0xF);

        union { uint32_t u; float f; } cb0, cb1, cb2, cb3;
        cb0.u = (((uint32_t)(smb0 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb0)) << 7 | (smb0 & 0x7F)) << 16;
        cb1.u = (((uint32_t)(smb1 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb1)) << 7 | (smb1 & 0x7F)) << 16;
        cb2.u = (((uint32_t)(smb2 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb2)) << 7 | (smb2 & 0x7F)) << 16;
        cb3.u = (((uint32_t)(smb3 >> 7)) << 15 | ((uint32_t)(base_exp_b + gb3)) << 7 | (smb3 & 0x7F)) << 16;

        sb0 += cb0.f * af0; sb1 += cb1.f * af1; sb2 += cb2.f * af2; sb3 += cb3.f * af3;

        sma_ptr += WORKGROUP_SIZE * 4; gra_ptr += WORKGROUP_SIZE * 4 / 2;
        smb_ptr += WORKGROUP_SIZE * 4; grb_ptr += WORKGROUP_SIZE * 4 / 2;
        act_base += WORKGROUP_SIZE * 4;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        float af = bf16_to_float(TURBO_LDG(act_base));
        uint8_t sma = TURBO_LDG(sma_ptr); uint8_t gba = TURBO_LDG(gra_ptr);
        uint32_t ga = is_odd ? (gba >> 4) : (gba & 0xF);
        union { uint32_t u; float f; } ca;
        ca.u = (((uint32_t)(sma >> 7)) << 15 | ((uint32_t)(base_exp_a + ga)) << 7 | (sma & 0x7F)) << 16;
        sa0 += ca.f * af;
        uint8_t smb = TURBO_LDG(smb_ptr); uint8_t gbb = TURBO_LDG(grb_ptr);
        uint32_t gb = is_odd ? (gbb >> 4) : (gbb & 0xF);
        union { uint32_t u; float f; } cb;
        cb.u = (((uint32_t)(smb >> 7)) << 15 | ((uint32_t)(base_exp_b + gb)) << 7 | (smb & 0x7F)) << 16;
        sb0 += cb.f * af;
        sma_ptr += WORKGROUP_SIZE; gra_ptr += WORKGROUP_SIZE / 2;
        smb_ptr += WORKGROUP_SIZE; grb_ptr += WORKGROUP_SIZE / 2;
        act_base += WORKGROUP_SIZE;
    }

    float sum_a = sa0 + sa1 + sa2 + sa3;
    float sum_b = sb0 + sb1 + sb2 + sb3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums_a[tid / WARP_SIZE] = sum_a;
        warp_sums_b[tid / WARP_SIZE] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0, sb = 0;
        for (int w = 0; w < NUM_WARPS; w++) { sa += warp_sums_a[w]; sb += warp_sums_b[w]; }
        output_a[row] = sa; output_b[row] = sb;
    }
}

// ---------------------------------------------------------------------------
// Dual v3_unroll8: 8-element unroll per iteration
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_unroll8_dual(
    const uint8_t* __restrict__ sm_a, const uint8_t* __restrict__ gr_a, int base_exp_a,
    const uint8_t* __restrict__ sm_b, const uint8_t* __restrict__ gr_b, int base_exp_b,
    const int16_t* __restrict__ activations,
    float*         __restrict__ output_a,
    float*         __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[NUM_WARPS];
    __shared__ float warp_sums_b[NUM_WARPS];

    float sa0=0, sa1=0, sa2=0, sa3=0, sa4=0, sa5=0, sa6=0, sa7=0;
    float sb0=0, sb1=0, sb2=0, sb3=0, sb4=0, sb5=0, sb6=0, sb7=0;

    const uint8_t* sma_ptr = sm_a + (int64_t)row * K + tid;
    const uint8_t* gra_ptr = gr_a + (int64_t)row * K / 2 + tid / 2;
    const uint8_t* smb_ptr = sm_b + (int64_t)row * K + tid;
    const uint8_t* grb_ptr = gr_b + (int64_t)row * K / 2 + tid / 2;
    const int16_t* act_base = activations + tid;
    const int is_odd = tid & 1;

    int col = tid;
    for (; col + WORKGROUP_SIZE * 7 < K; col += WORKGROUP_SIZE * 8) {
        float af0 = bf16_to_float(TURBO_LDG(act_base));
        float af1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float af2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float af3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));
        float af4 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 4));
        float af5 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 5));
        float af6 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 6));
        float af7 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 7));

        // Tensor A
        uint8_t a0s = TURBO_LDG(sma_ptr); uint8_t a1s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE);
        uint8_t a2s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*2); uint8_t a3s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*3);
        uint8_t a4s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*4); uint8_t a5s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*5);
        uint8_t a6s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*6); uint8_t a7s = TURBO_LDG(sma_ptr + WORKGROUP_SIZE*7);
        uint8_t a0g = TURBO_LDG(gra_ptr); uint8_t a1g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE/2);
        uint8_t a2g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE); uint8_t a3g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE*3/2);
        uint8_t a4g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE*2); uint8_t a5g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE*5/2);
        uint8_t a6g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE*3); uint8_t a7g = TURBO_LDG(gra_ptr + WORKGROUP_SIZE*7/2);

        #define DECODE_S12(sm_val, gb_val, bexp) ({ \
            uint32_t _g = is_odd ? ((gb_val) >> 4) : ((gb_val) & 0xF); \
            union { uint32_t u; float f; } _c; \
            _c.u = (((uint32_t)((sm_val) >> 7)) << 15 | ((uint32_t)((bexp) + _g)) << 7 | ((sm_val) & 0x7F)) << 16; \
            _c.f; })

        sa0 += DECODE_S12(a0s, a0g, base_exp_a) * af0;
        sa1 += DECODE_S12(a1s, a1g, base_exp_a) * af1;
        sa2 += DECODE_S12(a2s, a2g, base_exp_a) * af2;
        sa3 += DECODE_S12(a3s, a3g, base_exp_a) * af3;
        sa4 += DECODE_S12(a4s, a4g, base_exp_a) * af4;
        sa5 += DECODE_S12(a5s, a5g, base_exp_a) * af5;
        sa6 += DECODE_S12(a6s, a6g, base_exp_a) * af6;
        sa7 += DECODE_S12(a7s, a7g, base_exp_a) * af7;

        // Tensor B
        uint8_t b0s = TURBO_LDG(smb_ptr); uint8_t b1s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE);
        uint8_t b2s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*2); uint8_t b3s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*3);
        uint8_t b4s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*4); uint8_t b5s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*5);
        uint8_t b6s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*6); uint8_t b7s = TURBO_LDG(smb_ptr + WORKGROUP_SIZE*7);
        uint8_t b0g = TURBO_LDG(grb_ptr); uint8_t b1g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE/2);
        uint8_t b2g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE); uint8_t b3g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE*3/2);
        uint8_t b4g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE*2); uint8_t b5g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE*5/2);
        uint8_t b6g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE*3); uint8_t b7g = TURBO_LDG(grb_ptr + WORKGROUP_SIZE*7/2);

        sb0 += DECODE_S12(b0s, b0g, base_exp_b) * af0;
        sb1 += DECODE_S12(b1s, b1g, base_exp_b) * af1;
        sb2 += DECODE_S12(b2s, b2g, base_exp_b) * af2;
        sb3 += DECODE_S12(b3s, b3g, base_exp_b) * af3;
        sb4 += DECODE_S12(b4s, b4g, base_exp_b) * af4;
        sb5 += DECODE_S12(b5s, b5g, base_exp_b) * af5;
        sb6 += DECODE_S12(b6s, b6g, base_exp_b) * af6;
        sb7 += DECODE_S12(b7s, b7g, base_exp_b) * af7;

        sma_ptr += WORKGROUP_SIZE * 8; gra_ptr += WORKGROUP_SIZE * 8 / 2;
        smb_ptr += WORKGROUP_SIZE * 8; grb_ptr += WORKGROUP_SIZE * 8 / 2;
        act_base += WORKGROUP_SIZE * 8;
    }
    // 4-element tail
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        float af0 = bf16_to_float(TURBO_LDG(act_base));
        float af1 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE));
        float af2 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 2));
        float af3 = bf16_to_float(TURBO_LDG(act_base + WORKGROUP_SIZE * 3));

        sa0 += DECODE_S12(TURBO_LDG(sma_ptr), TURBO_LDG(gra_ptr), base_exp_a) * af0;
        sa1 += DECODE_S12(TURBO_LDG(sma_ptr+WORKGROUP_SIZE), TURBO_LDG(gra_ptr+WORKGROUP_SIZE/2), base_exp_a) * af1;
        sa2 += DECODE_S12(TURBO_LDG(sma_ptr+WORKGROUP_SIZE*2), TURBO_LDG(gra_ptr+WORKGROUP_SIZE), base_exp_a) * af2;
        sa3 += DECODE_S12(TURBO_LDG(sma_ptr+WORKGROUP_SIZE*3), TURBO_LDG(gra_ptr+WORKGROUP_SIZE*3/2), base_exp_a) * af3;

        sb0 += DECODE_S12(TURBO_LDG(smb_ptr), TURBO_LDG(grb_ptr), base_exp_b) * af0;
        sb1 += DECODE_S12(TURBO_LDG(smb_ptr+WORKGROUP_SIZE), TURBO_LDG(grb_ptr+WORKGROUP_SIZE/2), base_exp_b) * af1;
        sb2 += DECODE_S12(TURBO_LDG(smb_ptr+WORKGROUP_SIZE*2), TURBO_LDG(grb_ptr+WORKGROUP_SIZE), base_exp_b) * af2;
        sb3 += DECODE_S12(TURBO_LDG(smb_ptr+WORKGROUP_SIZE*3), TURBO_LDG(grb_ptr+WORKGROUP_SIZE*3/2), base_exp_b) * af3;

        sma_ptr += WORKGROUP_SIZE * 4; gra_ptr += WORKGROUP_SIZE * 4 / 2;
        smb_ptr += WORKGROUP_SIZE * 4; grb_ptr += WORKGROUP_SIZE * 4 / 2;
        act_base += WORKGROUP_SIZE * 4;
    }
    // Single-element tail
    for (; col < K; col += WORKGROUP_SIZE) {
        float af = bf16_to_float(TURBO_LDG(act_base));
        sa0 += DECODE_S12(TURBO_LDG(sma_ptr), TURBO_LDG(gra_ptr), base_exp_a) * af;
        sb0 += DECODE_S12(TURBO_LDG(smb_ptr), TURBO_LDG(grb_ptr), base_exp_b) * af;
        sma_ptr += WORKGROUP_SIZE; gra_ptr += WORKGROUP_SIZE / 2;
        smb_ptr += WORKGROUP_SIZE; grb_ptr += WORKGROUP_SIZE / 2;
        act_base += WORKGROUP_SIZE;
    }
    #undef DECODE_S12

    float sum_a = sa0 + sa1 + sa2 + sa3 + sa4 + sa5 + sa6 + sa7;
    float sum_b = sb0 + sb1 + sb2 + sb3 + sb4 + sb5 + sb6 + sb7;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums_a[tid / WARP_SIZE] = sum_a;
        warp_sums_b[tid / WARP_SIZE] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0, sb = 0;
        for (int w = 0; w < NUM_WARPS; w++) { sa += warp_sums_a[w]; sb += warp_sums_b[w]; }
        output_a[row] = sa; output_b[row] = sb;
    }
}

// ---------------------------------------------------------------------------
// Dual v3_vec4: vectorized uint32_t loads
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_v3_vec4_dual(
    const uint8_t* __restrict__ sm_a, const uint8_t* __restrict__ gr_a, int base_exp_a,
    const uint8_t* __restrict__ sm_b, const uint8_t* __restrict__ gr_b, int base_exp_b,
    const int16_t* __restrict__ activations,
    float*         __restrict__ output_a,
    float*         __restrict__ output_b,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums_a[NUM_WARPS];
    __shared__ float warp_sums_b[NUM_WARPS];

    float sa0 = 0.0f, sa1 = 0.0f;
    float sb0 = 0.0f, sb1 = 0.0f;

    const uint8_t* sma_base = sm_a + (int64_t)row * K;
    const uint8_t* gra_base = gr_a + (int64_t)row * K / 2;
    const uint8_t* smb_base = sm_b + (int64_t)row * K;
    const uint8_t* grb_base = gr_b + (int64_t)row * K / 2;

    int elem = tid * 4;
    for (; elem + WORKGROUP_SIZE * 4 - 1 < K; elem += WORKGROUP_SIZE * 4) {
        uint32_t sma_v = *(const uint32_t*)(sma_base + elem);
        uint16_t gra_v = *(const uint16_t*)(gra_base + elem / 2);
        float wa0, wa1, wa2, wa3;
        decode_vec4(sma_v, gra_v, base_exp_a, wa0, wa1, wa2, wa3);

        float a0 = bf16_to_float(TURBO_LDG(activations + elem));
        float a1 = bf16_to_float(TURBO_LDG(activations + elem + 1));
        float a2 = bf16_to_float(TURBO_LDG(activations + elem + 2));
        float a3 = bf16_to_float(TURBO_LDG(activations + elem + 3));

        sa0 += wa0 * a0 + wa1 * a1 + wa2 * a2 + wa3 * a3;

        uint32_t smb_v = *(const uint32_t*)(smb_base + elem);
        uint16_t grb_v = *(const uint16_t*)(grb_base + elem / 2);
        float wb0, wb1, wb2, wb3;
        decode_vec4(smb_v, grb_v, base_exp_b, wb0, wb1, wb2, wb3);

        sb0 += wb0 * a0 + wb1 * a1 + wb2 * a2 + wb3 * a3;
    }
    // Scalar tail
    for (; elem < K; elem++) {
        int is_odd_e = elem & 1;
        float a = bf16_to_float(activations[elem]);
        uint8_t sma = sma_base[elem]; uint8_t gba = gra_base[elem / 2];
        uint32_t ga = is_odd_e ? (gba >> 4) : (gba & 0xF);
        union { uint32_t u; float f; } ca;
        ca.u = (((uint32_t)(sma >> 7)) << 15 | ((uint32_t)(base_exp_a + ga)) << 7 | (sma & 0x7F)) << 16;
        sa0 += ca.f * a;
        uint8_t smb = smb_base[elem]; uint8_t gbb = grb_base[elem / 2];
        uint32_t gb = is_odd_e ? (gbb >> 4) : (gbb & 0xF);
        union { uint32_t u; float f; } cb;
        cb.u = (((uint32_t)(smb >> 7)) << 15 | ((uint32_t)(base_exp_b + gb)) << 7 | (smb & 0x7F)) << 16;
        sb0 += cb.f * a;
    }

    float sum_a = sa0 + sa1;
    float sum_b = sb0 + sb1;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_a += __shfl_down(sum_a, off, WARP_SIZE);
        sum_b += __shfl_down(sum_b, off, WARP_SIZE);
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums_a[tid / WARP_SIZE] = sum_a;
        warp_sums_b[tid / WARP_SIZE] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0, sb = 0;
        for (int w = 0; w < NUM_WARPS; w++) { sa += warp_sums_a[w]; sb += warp_sums_b[w]; }
        output_a[row] = sa; output_b[row] = sb;
    }
}

// ===========================================================================
// Launch wrappers for all v3 variants + original v2
// ===========================================================================
extern "C" {

// --- v3 Multirow launch wrappers ---

int launch_split12_v3_ldg_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    if (M >= 2) {
        int grid = (M + 1) / 2;
        hipLaunchKernelGGL(split12_matvec_v3_ldg_multirow,
            dim3(grid), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    } else {
        hipLaunchKernelGGL(split12_matvec_v2,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    }
    return 0;
}

int launch_split12_v3_unroll8_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    if (M >= 2) {
        int grid = (M + 1) / 2;
        hipLaunchKernelGGL(split12_matvec_v3_unroll8_multirow,
            dim3(grid), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    } else {
        hipLaunchKernelGGL(split12_matvec_v2,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    }
    return 0;
}

int launch_split12_v3_vec4_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    if (M >= 2) {
        int grid = (M + 1) / 2;
        hipLaunchKernelGGL(split12_matvec_v3_vec4_multirow,
            dim3(grid), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    } else {
        hipLaunchKernelGGL(split12_matvec_v2,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    }
    return 0;
}

// --- v3 Dual launch wrappers ---

int launch_split12_v3_ldg_dual_async(
    const void* sm_a, const void* gr_a, int base_exp_a,
    const void* sm_b, const void* gr_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_v3_ldg_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sm_a, (const uint8_t*)gr_a, base_exp_a,
        (const uint8_t*)sm_b, (const uint8_t*)gr_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

int launch_split12_v3_unroll8_dual_async(
    const void* sm_a, const void* gr_a, int base_exp_a,
    const void* sm_b, const void* gr_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_v3_unroll8_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sm_a, (const uint8_t*)gr_a, base_exp_a,
        (const uint8_t*)sm_b, (const uint8_t*)gr_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

int launch_split12_v3_vec4_dual_async(
    const void* sm_a, const void* gr_a, int base_exp_a,
    const void* sm_b, const void* gr_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_v3_vec4_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sm_a, (const uint8_t*)gr_a, base_exp_a,
        (const uint8_t*)sm_b, (const uint8_t*)gr_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

// --- Original v2 wrappers (unchanged) ---

int launch_split12_v2_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* activations, void* output, int M, int K, void* stream)
{
    if (M >= 2) {
        int grid = (M + 1) / 2;
        hipLaunchKernelGGL(split12_matvec_v2_multirow,
            dim3(grid), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    } else {
        hipLaunchKernelGGL(split12_matvec_v2,
            dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
            (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
            (const int16_t*)activations, (float*)output, M, K);
    }
    return 0;
}

int launch_split12_v2_dual_async(
    const void* sm_a, const void* gr_a, int base_exp_a,
    const void* sm_b, const void* gr_b, int base_exp_b,
    const void* activations,
    void* output_a, void* output_b,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_v2_dual,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sm_a, (const uint8_t*)gr_a, base_exp_a,
        (const uint8_t*)sm_b, (const uint8_t*)gr_b, base_exp_b,
        (const int16_t*)activations,
        (float*)output_a, (float*)output_b,
        M, K);
    return 0;
}

}

// ---------------------------------------------------------------------------
// Split12 B=4 batched matvec with escape handling
// Reads from byte-aligned sign_mantissa[] + nibble-packed groups[]
// Same escape prefix sum as structured12_matvec_batch4
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_batch4(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t* __restrict__ act2, const int16_t* __restrict__ act3,
    const int32_t* __restrict__ escape_row_base,
    const uint8_t* __restrict__ escape_counts,
    const int16_t* __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float warp_sums[NUM_WARPS * 4];
    __shared__ int warp_esc_totals[NUM_WARPS];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Pointer setup
    const uint8_t* sm_ptr  = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr  = groups + (int64_t)row * K / 2 + tid / 2;
    const int is_odd = tid & 1;

    float s0a=0,s0b=0,s1a=0,s1b=0,s2a=0,s2b=0,s3a=0,s3b=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        // First element
        uint8_t sm0 = sm_ptr[0];
        uint8_t gb0 = gr_ptr[0];
        uint32_t group0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);

        float wt0;
        if (__builtin_expect(group0 != 0, 1)) {
            uint32_t sign = sm0 >> 7;
            uint32_t mantissa = sm0 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group0) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt0 = c.f;
        } else {
            wt0 = bf16_to_float(escape_vals[esc_ptr++]);
        }

        // Second element (col + WORKGROUP_SIZE)
        uint8_t sm1 = sm_ptr[256];
        uint8_t gb1 = gr_ptr[128];
        uint32_t group1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);

        float wt1;
        if (__builtin_expect(group1 != 0, 1)) {
            uint32_t sign = sm1 >> 7;
            uint32_t mantissa = sm1 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group1) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt1 = c.f;
        } else {
            wt1 = bf16_to_float(escape_vals[esc_ptr++]);
        }

        s0a += wt0 * bf16_to_float(act0[col]);  s0b += wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1a += wt0 * bf16_to_float(act1[col]);  s1b += wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2a += wt0 * bf16_to_float(act2[col]);  s2b += wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3a += wt0 * bf16_to_float(act3[col]);  s3b += wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);

        sm_ptr += 512;   // 256 * 2
        gr_ptr += 256;   // 256 * 2 / 2
    }
    // Tail
    if (col < K) {
        uint8_t sm0 = sm_ptr[0];
        uint8_t gb0 = gr_ptr[0];
        uint32_t group0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);

        float wt;
        if (__builtin_expect(group0 != 0, 1)) {
            uint32_t sign = sm0 >> 7;
            uint32_t mantissa = sm0 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group0) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt = c.f;
        } else {
            wt = bf16_to_float(escape_vals[esc_ptr++]);
        }

        s0a += wt * bf16_to_float(act0[col]);
        s1a += wt * bf16_to_float(act1[col]);
        s2a += wt * bf16_to_float(act2[col]);
        s3a += wt * bf16_to_float(act3[col]);
    }

    // Warp reduction + output
    float s0=s0a+s0b, s1=s1a+s1b, s2=s2a+s2b, s3=s3a+s3b;
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
    }
    int wid = tid / WARP_SIZE;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]=s0; warp_sums[wid+NUM_WARPS]=s1;
        warp_sums[wid+NUM_WARPS*2]=s2; warp_sums[wid+NUM_WARPS*3]=s3;
    }
    __syncthreads();
    if (tid == 0) {
        float r0=0,r1=0,r2=0,r3=0;
        for (int w = 0; w < NUM_WARPS; w++) {
            r0 += warp_sums[w]; r1 += warp_sums[w+NUM_WARPS];
            r2 += warp_sums[w+NUM_WARPS*2]; r3 += warp_sums[w+NUM_WARPS*3];
        }
        out0[row]=r0; out1[row]=r1; out2[row]=r2; out3[row]=r3;
    }
}

// ---------------------------------------------------------------------------
// Split12 B=8 batched matvec with escape handling
// Same as batch4 but 8 activation vectors, single accumulators
// ---------------------------------------------------------------------------
__launch_bounds__(WORKGROUP_SIZE)
__global__ void split12_matvec_batch8(
    const uint8_t* __restrict__ sign_mantissa,  // [M * K]
    const uint8_t* __restrict__ groups,          // [M * K / 2]
    int base_exp,
    const int16_t* __restrict__ act0, const int16_t* __restrict__ act1,
    const int16_t* __restrict__ act2, const int16_t* __restrict__ act3,
    const int16_t* __restrict__ act4, const int16_t* __restrict__ act5,
    const int16_t* __restrict__ act6, const int16_t* __restrict__ act7,
    const int32_t* __restrict__ escape_row_base,
    const uint8_t* __restrict__ escape_counts,
    const int16_t* __restrict__ escape_vals,
    float* __restrict__ out0, float* __restrict__ out1,
    float* __restrict__ out2, float* __restrict__ out3,
    float* __restrict__ out4, float* __restrict__ out5,
    float* __restrict__ out6, float* __restrict__ out7,
    int M, int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float warp_sums[NUM_WARPS * 8];
    __shared__ int warp_esc_totals[NUM_WARPS];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int prefix = my_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int v = __shfl_up(prefix, off, WARP_SIZE);
        if (lane >= off) prefix += v;
    }
    int warp_total = __shfl(prefix, WARP_SIZE - 1, WARP_SIZE);
    prefix -= my_count;
    if (lane == WARP_SIZE - 1) warp_esc_totals[warp_id] = warp_total;
    __syncthreads();
    int cross = 0;
    for (int w = 0; w < warp_id; w++) cross += warp_esc_totals[w];
    int esc_ptr = escape_row_base[row] + cross + prefix;

    // Pointer setup
    const uint8_t* sm_ptr  = sign_mantissa + (int64_t)row * K + tid;
    const uint8_t* gr_ptr  = groups + (int64_t)row * K / 2 + tid / 2;
    const int is_odd = tid & 1;

    // Single accumulators to save VGPRs (critical for B=8 register pressure)
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;

    int col = tid;
    for (; col + WORKGROUP_SIZE < K; col += WORKGROUP_SIZE * 2) {
        // First element
        uint8_t sm0 = sm_ptr[0];
        uint8_t gb0 = gr_ptr[0];
        uint32_t group0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);

        float wt0;
        if (__builtin_expect(group0 != 0, 1)) {
            uint32_t sign = sm0 >> 7;
            uint32_t mantissa = sm0 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group0) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt0 = c.f;
        } else {
            wt0 = bf16_to_float(escape_vals[esc_ptr++]);
        }

        // Second element (col + WORKGROUP_SIZE)
        uint8_t sm1 = sm_ptr[256];
        uint8_t gb1 = gr_ptr[128];
        uint32_t group1 = is_odd ? (gb1 >> 4) : (gb1 & 0xF);

        float wt1;
        if (__builtin_expect(group1 != 0, 1)) {
            uint32_t sign = sm1 >> 7;
            uint32_t mantissa = sm1 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group1) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt1 = c.f;
        } else {
            wt1 = bf16_to_float(escape_vals[esc_ptr++]);
        }

        s0 += wt0 * bf16_to_float(act0[col]) + wt1 * bf16_to_float(act0[col + WORKGROUP_SIZE]);
        s1 += wt0 * bf16_to_float(act1[col]) + wt1 * bf16_to_float(act1[col + WORKGROUP_SIZE]);
        s2 += wt0 * bf16_to_float(act2[col]) + wt1 * bf16_to_float(act2[col + WORKGROUP_SIZE]);
        s3 += wt0 * bf16_to_float(act3[col]) + wt1 * bf16_to_float(act3[col + WORKGROUP_SIZE]);
        s4 += wt0 * bf16_to_float(act4[col]) + wt1 * bf16_to_float(act4[col + WORKGROUP_SIZE]);
        s5 += wt0 * bf16_to_float(act5[col]) + wt1 * bf16_to_float(act5[col + WORKGROUP_SIZE]);
        s6 += wt0 * bf16_to_float(act6[col]) + wt1 * bf16_to_float(act6[col + WORKGROUP_SIZE]);
        s7 += wt0 * bf16_to_float(act7[col]) + wt1 * bf16_to_float(act7[col + WORKGROUP_SIZE]);

        sm_ptr += 512;   // 256 * 2
        gr_ptr += 256;   // 256 * 2 / 2
    }
    // Tail
    if (col < K) {
        uint8_t sm0 = sm_ptr[0];
        uint8_t gb0 = gr_ptr[0];
        uint32_t group0 = is_odd ? (gb0 >> 4) : (gb0 & 0xF);

        float wt;
        if (__builtin_expect(group0 != 0, 1)) {
            uint32_t sign = sm0 >> 7;
            uint32_t mantissa = sm0 & 0x7F;
            uint32_t bf16 = (sign << 15) | ((base_exp + group0) << 7) | mantissa;
            union { uint32_t u; float f; } c; c.u = bf16 << 16; wt = c.f;
        } else {
            wt = bf16_to_float(escape_vals[esc_ptr++]);
        }

        s0 += wt * bf16_to_float(act0[col]);
        s1 += wt * bf16_to_float(act1[col]);
        s2 += wt * bf16_to_float(act2[col]);
        s3 += wt * bf16_to_float(act3[col]);
        s4 += wt * bf16_to_float(act4[col]);
        s5 += wt * bf16_to_float(act5[col]);
        s6 += wt * bf16_to_float(act6[col]);
        s7 += wt * bf16_to_float(act7[col]);
    }

    // Warp reduction + output
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        s0 += __shfl_down(s0, off, WARP_SIZE);
        s1 += __shfl_down(s1, off, WARP_SIZE);
        s2 += __shfl_down(s2, off, WARP_SIZE);
        s3 += __shfl_down(s3, off, WARP_SIZE);
        s4 += __shfl_down(s4, off, WARP_SIZE);
        s5 += __shfl_down(s5, off, WARP_SIZE);
        s6 += __shfl_down(s6, off, WARP_SIZE);
        s7 += __shfl_down(s7, off, WARP_SIZE);
    }
    int wid = tid / WARP_SIZE;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]              = s0;
        warp_sums[wid + NUM_WARPS]   = s1;
        warp_sums[wid + NUM_WARPS*2] = s2;
        warp_sums[wid + NUM_WARPS*3] = s3;
        warp_sums[wid + NUM_WARPS*4] = s4;
        warp_sums[wid + NUM_WARPS*5] = s5;
        warp_sums[wid + NUM_WARPS*6] = s6;
        warp_sums[wid + NUM_WARPS*7] = s7;
    }
    __syncthreads();
    if (tid == 0) {
        float r[8] = {0};
        for (int w = 0; w < NUM_WARPS; w++)
            for (int o = 0; o < 8; o++) r[o] += warp_sums[w + NUM_WARPS * o];
        out0[row]=r[0]; out1[row]=r[1]; out2[row]=r[2]; out3[row]=r[3];
        out4[row]=r[4]; out5[row]=r[5]; out6[row]=r[6]; out7[row]=r[7];
    }
}

extern "C" {

int launch_split12_batch4_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_batch4,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3, M, K);
    return 0;
}

int launch_split12_batch8_async(
    const void* sign_mantissa, const void* groups, int base_exp,
    const void* a0, const void* a1, const void* a2, const void* a3,
    const void* a4, const void* a5, const void* a6, const void* a7,
    const void* esc_row_base, const void* esc_counts, const void* esc_vals,
    void* o0, void* o1, void* o2, void* o3,
    void* o4, void* o5, void* o6, void* o7,
    int M, int K, void* stream)
{
    hipLaunchKernelGGL(split12_matvec_batch8,
        dim3(M), dim3(WORKGROUP_SIZE), 0, (hipStream_t)stream,
        (const uint8_t*)sign_mantissa, (const uint8_t*)groups, base_exp,
        (const int16_t*)a0, (const int16_t*)a1,
        (const int16_t*)a2, (const int16_t*)a3,
        (const int16_t*)a4, (const int16_t*)a5,
        (const int16_t*)a6, (const int16_t*)a7,
        (const int32_t*)esc_row_base, (const uint8_t*)esc_counts,
        (const int16_t*)esc_vals,
        (float*)o0, (float*)o1, (float*)o2, (float*)o3,
        (float*)o4, (float*)o5, (float*)o6, (float*)o7,
        M, K);
    return 0;
}

}
