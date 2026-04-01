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

#include <hip/hip_runtime.h>
#include <cstdint>

#define WARP_SIZE      64   // MI50 wavefront width
#define WORKGROUP_SIZE 256  // 4 wavefronts per block

// BF16 → float32: BF16 is upper 16 bits of IEEE float32
__device__ __forceinline__
float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
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
    __shared__ float warp_sums[4];
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    const uint32_t* row_packed = packed + (int64_t)row * K * 12 / 32;
    int bp = (int)((int64_t)row * K * 12 % 32) + tid * 12;
    const int shift = bp & 31;
    const char* pack_ptr = (const char*)(row_packed + (bp >> 5));
    const char* act_ptr = (const char*)(activations + tid);

    int col = tid;
    for (; col + WORKGROUP_SIZE * 3 < K; col += WORKGROUP_SIZE * 4) {
        uint32_t d0 = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        uint32_t d1 = (uint32_t)(*(const uint64_t*)(pack_ptr + 384) >> shift) & 0xFFF;
        uint32_t d2 = (uint32_t)(*(const uint64_t*)(pack_ptr + 768) >> shift) & 0xFFF;
        uint32_t d3 = (uint32_t)(*(const uint64_t*)(pack_ptr + 1152) >> shift) & 0xFFF;
        sum0 += structured12_decode(d0, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        sum1 += structured12_decode(d1, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 512));
        sum2 += structured12_decode(d2, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1024));
        sum3 += structured12_decode(d3, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr + 1536));
        pack_ptr += 1536;
        act_ptr += 2048;
    }
    for (; col < K; col += WORKGROUP_SIZE) {
        uint32_t d = (uint32_t)(*(const uint64_t*)(pack_ptr) >> shift) & 0xFFF;
        sum0 += structured12_decode(d, base_exp) * bf16_to_float(*(const int16_t*)(act_ptr));
        pack_ptr += 384;
        act_ptr += 512;
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) sum += __shfl_down(sum, off, WARP_SIZE);
    if ((tid & (WARP_SIZE-1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
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
    __shared__ float warp_sums_a[4];
    __shared__ float warp_sums_b[4];

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
        uint32_t da0 = (uint32_t)(*(const uint64_t*)(pa_ptr) >> shift) & 0xFFF;
        uint32_t da1 = (uint32_t)(*(const uint64_t*)(pa_ptr + 384) >> shift) & 0xFFF;
        uint32_t da2 = (uint32_t)(*(const uint64_t*)(pa_ptr + 768) >> shift) & 0xFFF;
        uint32_t da3 = (uint32_t)(*(const uint64_t*)(pa_ptr + 1152) >> shift) & 0xFFF;
        sa0 += structured12_decode(da0, base_exp_a) * af0;
        sa1 += structured12_decode(da1, base_exp_a) * af1;
        sa2 += structured12_decode(da2, base_exp_a) * af2;
        sa3 += structured12_decode(da3, base_exp_a) * af3;

        // Decode weights from tensor B (up)
        uint32_t db0 = (uint32_t)(*(const uint64_t*)(pb_ptr) >> shift) & 0xFFF;
        uint32_t db1 = (uint32_t)(*(const uint64_t*)(pb_ptr + 384) >> shift) & 0xFFF;
        uint32_t db2 = (uint32_t)(*(const uint64_t*)(pb_ptr + 768) >> shift) & 0xFFF;
        uint32_t db3 = (uint32_t)(*(const uint64_t*)(pb_ptr + 1152) >> shift) & 0xFFF;
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
        uint32_t da = (uint32_t)(*(const uint64_t*)(pa_ptr) >> shift) & 0xFFF;
        uint32_t db = (uint32_t)(*(const uint64_t*)(pb_ptr) >> shift) & 0xFFF;
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
        warp_sums_a[tid >> 6] = sum_a;
        warp_sums_b[tid >> 6] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        output_a[row] = warp_sums_a[0] + warp_sums_a[1] + warp_sums_a[2] + warp_sums_a[3];
        output_b[row] = warp_sums_b[0] + warp_sums_b[1] + warp_sums_b[2] + warp_sums_b[3];
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

    // NO LDS codebook — only warp_sums (64 bytes) + esc_totals (16 bytes)
    __shared__ float warp_sums[16];
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles (same as before)
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
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
        uint32_t raw0 = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t raw1 = (uint32_t)(*(const uint64_t*)(base_ptr + 384) >> shift) & 0xFFF;

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
        uint32_t raw = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
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
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]=s0; warp_sums[wid+4]=s1;
        warp_sums[wid+8]=s2; warp_sums[wid+12]=s3;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row]=warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row]=warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row]=warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row]=warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
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

    // NO LDS codebook — only warp_sums (128 bytes) + esc_totals (16 bytes)
    __shared__ float warp_sums[32];  // 4 warps x 8 outputs
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
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
        uint32_t raw0 = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
        uint32_t raw1 = (uint32_t)(*(const uint64_t*)(base_ptr + 384) >> shift) & 0xFFF;

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
        uint32_t raw = (uint32_t)(*(const uint64_t*)(base_ptr) >> shift) & 0xFFF;
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
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[wid]       = s0;
        warp_sums[wid + 4]   = s1;
        warp_sums[wid + 8]   = s2;
        warp_sums[wid + 12]  = s3;
        warp_sums[wid + 16]  = s4;
        warp_sums[wid + 20]  = s5;
        warp_sums[wid + 24]  = s6;
        warp_sums[wid + 28]  = s7;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
        out4[row] = warp_sums[16]+warp_sums[17]+warp_sums[18]+warp_sums[19];
        out5[row] = warp_sums[20]+warp_sums[21]+warp_sums[22]+warp_sums[23];
        out6[row] = warp_sums[24]+warp_sums[25]+warp_sums[26]+warp_sums[27];
        out7[row] = warp_sums[28]+warp_sums[29]+warp_sums[30]+warp_sums[31];
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
    __shared__ float warp_sums[4];
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
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid >> 6] = sum;
    __syncthreads();
    if (tid == 0) output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
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
    __shared__ float warp_sums[8]; // 4 per row

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
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[tid >> 6] = sum0;

    // Warp reduction for row 1
    float sum1 = sum_b0 + sum_b1 + sum_b2 + sum_b3;
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum1 += __shfl_down(sum1, off, WARP_SIZE);
    if ((tid & (WARP_SIZE - 1)) == 0) warp_sums[4 + (tid >> 6)] = sum1;

    __syncthreads();
    if (tid == 0) {
        output[row0] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        if (has_row1)
            output[row1] = warp_sums[4] + warp_sums[5] + warp_sums[6] + warp_sums[7];
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
    __shared__ float warp_sums_a[4];
    __shared__ float warp_sums_b[4];

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
        warp_sums_a[tid >> 6] = sum_a;
        warp_sums_b[tid >> 6] = sum_b;
    }
    __syncthreads();
    if (tid == 0) {
        output_a[row] = warp_sums_a[0] + warp_sums_a[1] + warp_sums_a[2] + warp_sums_a[3];
        output_b[row] = warp_sums_b[0] + warp_sums_b[1] + warp_sums_b[2] + warp_sums_b[3];
    }
}

extern "C" {

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

    __shared__ float warp_sums[16];
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
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
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]=s0; warp_sums[wid+4]=s1;
        warp_sums[wid+8]=s2; warp_sums[wid+12]=s3;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row]=warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row]=warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row]=warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row]=warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
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

    __shared__ float warp_sums[32];  // 4 warps x 8 outputs
    __shared__ int warp_esc_totals[4];

    // Escape count prefix sum via warp shuffles
    int my_count = (int)escape_counts[row * WORKGROUP_SIZE + tid];
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid >> 6;
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
    int wid = tid >> 6;
    if ((tid & (WARP_SIZE-1)) == 0) {
        warp_sums[wid]       = s0;
        warp_sums[wid + 4]   = s1;
        warp_sums[wid + 8]   = s2;
        warp_sums[wid + 12]  = s3;
        warp_sums[wid + 16]  = s4;
        warp_sums[wid + 20]  = s5;
        warp_sums[wid + 24]  = s6;
        warp_sums[wid + 28]  = s7;
    }
    __syncthreads();
    if (tid == 0) {
        out0[row] = warp_sums[0]+warp_sums[1]+warp_sums[2]+warp_sums[3];
        out1[row] = warp_sums[4]+warp_sums[5]+warp_sums[6]+warp_sums[7];
        out2[row] = warp_sums[8]+warp_sums[9]+warp_sums[10]+warp_sums[11];
        out3[row] = warp_sums[12]+warp_sums[13]+warp_sums[14]+warp_sums[15];
        out4[row] = warp_sums[16]+warp_sums[17]+warp_sums[18]+warp_sums[19];
        out5[row] = warp_sums[20]+warp_sums[21]+warp_sums[22]+warp_sums[23];
        out6[row] = warp_sums[24]+warp_sums[25]+warp_sums[26]+warp_sums[27];
        out7[row] = warp_sums[28]+warp_sums[29]+warp_sums[30]+warp_sums[31];
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
