/**
 * Turbo Lossless Engine — HIP Kernels
 * All non-matvec transformer operations for MI50 (gfx906)
 */

#include "../gpu_compat.h"
#include <cstdint>
#include <cmath>

#define WARP_SIZE TURBO_WARP_SIZE

// ============================================================
// BF16 ↔ Float conversion helpers
// ============================================================
__device__ __forceinline__ float bf16_to_float(int16_t raw) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)(uint16_t)raw) << 16;
    return c.f;
}

__device__ __forceinline__ int16_t float_to_bf16(float val) {
    union { float f; uint32_t u; } c;
    c.f = val;
    // Round to nearest even
    uint32_t rounding = (c.u + 0x7FFF + ((c.u >> 16) & 1)) >> 16;
    return (int16_t)rounding;
}

// ============================================================
// RoPE: Rotary Position Embeddings (with precomputed frequencies)
// ============================================================

// Precomputed frequency table on GPU (set once during init)
static float* d_rope_freqs = nullptr;
static int d_rope_freqs_dim = 0;


// ============================================================
// SiLU(gate) * up — SwiGLU activation for MLP
// ============================================================
__global__ void silu_mul_kernel(
    const float* __restrict__ gate,   // [n]
    const float* __restrict__ up,     // [n]
    float* __restrict__ out,          // [n]
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));  // SiLU = x * sigmoid(x)
    out[idx] = silu * up[idx];
}

// ============================================================
// Embedding lookup: token_id → embedding vector
// ============================================================
__global__ void embed_lookup_kernel(
    const int16_t* __restrict__ embd_table,  // [n_vocab, n_embd] BF16
    const int* __restrict__ token_ids,        // [batch_size]
    float* __restrict__ output,               // [batch_size, n_embd]
    int n_embd, int batch_size, int n_vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (seq >= batch_size || idx >= n_embd) return;

    int token_id = token_ids[seq];
    if (token_id < 0 || token_id >= n_vocab) {
        output[seq * n_embd + idx] = 0.0f;
        return;
    }
    output[seq * n_embd + idx] = bf16_to_float(embd_table[token_id * n_embd + idx]);
}

// ============================================================
// Residual add: y = x + residual
// ============================================================
__global__ void add_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += x[idx];
}

// ============================================================
// Argmax: find token with highest logit
// ============================================================
__global__ void argmax_kernel(
    const float* __restrict__ logits,  // [batch, n_vocab] (one block per batch element)
    int* __restrict__ result,          // [batch]
    int n)
{
    extern __shared__ float sdata[];
    int* sidx = (int*)(sdata + blockDim.x);
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    const float* my_logits = logits + (size_t)bid * n;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (my_logits[i] > max_val) {
            max_val = my_logits[i];
            max_idx = i;
        }
    }
    sdata[tid] = max_val;
    sidx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[bid] = sidx[0];
}

// ============================================================
// C launch wrappers
// ============================================================
extern "C" {

void launch_rope_init(int head_dim, float theta_base) {
    if (d_rope_freqs && d_rope_freqs_dim == head_dim) return;
    int half_dim = head_dim / 2;
    float* h_freqs = new float[half_dim];
    for (int d = 0; d < half_dim; d++)
        h_freqs[d] = 1.0f / powf(theta_base, (float)(2 * d) / head_dim);
    if (d_rope_freqs) hipFree(d_rope_freqs);
    GPU_CHECK(hipMalloc(&d_rope_freqs, half_dim * sizeof(float)));
    hipMemcpy(d_rope_freqs, h_freqs, half_dim * sizeof(float), hipMemcpyHostToDevice);
    d_rope_freqs_dim = head_dim;
    delete[] h_freqs;
}

void launch_embed_lookup(const int16_t* embd_table, const int* token_ids,
                         float* output, int n_embd, int batch_size, int n_vocab, hipStream_t stream) {
    dim3 threads(256);
    dim3 blocks((n_embd + 255) / 256, batch_size);
    embed_lookup_kernel<<<blocks, threads, 0, stream>>>(embd_table, token_ids, output, n_embd, batch_size, n_vocab);
}

void launch_argmax(const float* logits, int* result, int n, hipStream_t stream) {
    int threads = 256;
    argmax_kernel<<<1, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(logits, result, n);
}

void launch_argmax_batch(const float* logits, int* results, int n_vocab, int batch_size, hipStream_t stream) {
    int threads = 256;
    // One block per sequence in the batch
    argmax_kernel<<<batch_size, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(logits, results, n_vocab);
}

}

// ============================================================
// Attention: Q×K^T → softmax → ×V with GQA support
// ============================================================

// ============================================================
// Fused RoPE + INT8 KV Store (B=1 only)
// Two-phase approach:
//   Phase 1: RoPE rotation on Q and K (flat dispatch, no cooperation needed)
//   Phase 2: INT8 quantized KV store (one block per kv_head, cooperative absmax)
// ============================================================

// Phase 1: RoPE on Q and K (flat dispatch, same structure as before)
__global__ void rope_qk_kernel(
    float* __restrict__ q,              // [n_head, head_dim]
    float* __restrict__ k,              // [n_head_kv, head_dim]
    const float* __restrict__ freqs,    // [half_dim] precomputed
    int head_dim, int n_head, int n_head_kv,
    int position)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    // RoPE on Q heads
    if (idx < n_head * half_dim) {
        int head = idx / half_dim;
        int d = idx % half_dim;
        float angle = position * freqs[d];
        float cos_a, sin_a;
        __sincosf(angle, &sin_a, &cos_a);

        int base = head * head_dim;
        float x0 = q[base + d];
        float x1 = q[base + d + half_dim];
        q[base + d]            = x0 * cos_a - x1 * sin_a;
        q[base + d + half_dim] = x0 * sin_a + x1 * cos_a;
    }

    // RoPE on K heads (rotate in-place; cache store handled by phase 2)
    if (idx < n_head_kv * half_dim) {
        int head = idx / half_dim;
        int d = idx % half_dim;
        float angle = position * freqs[d];
        float cos_a, sin_a;
        __sincosf(angle, &sin_a, &cos_a);

        int base = head * head_dim;
        float x0 = k[base + d];
        float x1 = k[base + d + half_dim];
        k[base + d]            = x0 * cos_a - x1 * sin_a;
        k[base + d + half_dim] = x0 * sin_a + x1 * cos_a;
    }
}

// Phase 2: INT8 quantized KV store
// Grid: (n_head_kv, 2) -- x=kv_head, y=0(K)/1(V)
// Threads: head_dim (rounded up to warp multiple)
// Each block cooperatively finds absmax via warp shuffle, computes scale,
// then quantizes all head_dim values to int8.
__global__ void store_kv_int8_kernel(
    const float* __restrict__ k_in,     // [n_head_kv * head_dim] rotated K (FP32)
    const float* __restrict__ v_in,     // [n_head_kv * head_dim] V (FP32)
    int8_t* __restrict__ kv_cache_k,    // [max_seq, n_head_kv, head_dim] INT8
    int8_t* __restrict__ kv_cache_v,
    float* __restrict__ kv_scale_k,     // [max_seq, n_head_kv] per-head scale
    float* __restrict__ kv_scale_v,
    int head_dim, int n_head_kv,
    int position, int max_seq)
{
    int kv_head = blockIdx.x;
    int is_v = blockIdx.y;  // 0 = K, 1 = V
    int tid = threadIdx.x;

    if (kv_head >= n_head_kv) return;

    // Shared memory for warp-level absmax reduction
    extern __shared__ float smem[];

    const float* src = is_v ? v_in : k_in;
    int8_t* dst = is_v ? kv_cache_v : kv_cache_k;
    float* scale_dst = is_v ? kv_scale_v : kv_scale_k;

    int src_offset = kv_head * head_dim;
    int kv_total = n_head_kv * head_dim;
    int cache_offset = position * kv_total + kv_head * head_dim;
    int scale_offset = position * n_head_kv + kv_head;

    // Step 1: Load value and compute thread-local absmax
    float val = 0.0f;
    float thread_absmax = 0.0f;
    if (tid < head_dim) {
        val = src[src_offset + tid];
        thread_absmax = fabsf(val);
    }

    // Step 2: Warp-level absmax reduction via shuffle
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    float warp_max = thread_absmax;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(warp_max, offset, WARP_SIZE);
        warp_max = fmaxf(warp_max, other);
    }

    // Step 3: Write per-warp max to shared memory
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (lane == 0) {
        smem[warp_id] = warp_max;
    }
    __syncthreads();

    // Step 4: First warp reduces across all warps to get global absmax
    float global_max = 0.0f;
    if (tid < num_warps) {
        global_max = smem[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(global_max, offset, WARP_SIZE);
        global_max = fmaxf(global_max, other);
    }

    // Step 5: Broadcast global_max to all threads via shared memory
    if (tid == 0) {
        smem[0] = global_max;
    }
    __syncthreads();
    global_max = smem[0];

    // scale = absmax / 127; inv_scale = 127 / absmax (for quantization)
    float scale = global_max / 127.0f;
    float inv_scale = (global_max > 0.0f) ? 127.0f / global_max : 0.0f;

    // Step 6: Quantize to int8 and store (symmetric: clamp to [-127, 127])
    if (tid < head_dim) {
        float q_val = rintf(val * inv_scale);
        q_val = fminf(fmaxf(q_val, -127.0f), 127.0f);
        dst[cache_offset + tid] = (int8_t)((int)q_val);
    }

    // Step 7: Store scale factor (one per kv_head per position)
    if (tid == 0) {
        scale_dst[scale_offset] = scale;
    }
}

extern "C" {

void launch_rope_store_kv(float* q, float* k, const float* v,
    int8_t* kv_k, int8_t* kv_v,
    float* kv_scale_k, float* kv_scale_v,
    int head_dim, int n_head, int n_head_kv, int position,
    int max_seq, float theta, hipStream_t stream)
{
    launch_rope_init(head_dim, theta);
    int half_dim = head_dim / 2;

    // Phase 1: RoPE on Q and K (flat dispatch)
    int rope_threads = max(n_head, n_head_kv) * half_dim;
    int threads = 256;
    int blocks = (rope_threads + threads - 1) / threads;
    rope_qk_kernel<<<blocks, threads, 0, stream>>>(
        q, k, d_rope_freqs, head_dim, n_head, n_head_kv, position);

    // Phase 2: INT8 quantized KV store
    // One block per kv_head x 2 (K and V), head_dim threads per block
    int store_threads = ((head_dim + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int num_warps = (store_threads + WARP_SIZE - 1) / WARP_SIZE;
    int smem_bytes = num_warps * sizeof(float);
    dim3 grid(n_head_kv, 2);  // x=kv_head, y=0(K)/1(V)
    store_kv_int8_kernel<<<grid, store_threads, smem_bytes, stream>>>(
        k, v, kv_k, kv_v, kv_scale_k, kv_scale_v,
        head_dim, n_head_kv, position, max_seq);
}

}

// ============================================================
// Batched add: y[i] += x[i] for batch_size * n elements
// ============================================================
__global__ void add_batch_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int n, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;
    if (idx >= total) return;
    y[idx] += x[idx];
}


// ============================================================
// Fused RMSNorm → BF16 output (eliminates separate fp32_to_bf16 launch)
// ============================================================
__global__ void rms_norm_bf16_batch_kernel(
    const float* __restrict__ x,      // [batch_size * n] FP32 input
    const float* __restrict__ weight,  // [n]
    int16_t* __restrict__ y,          // [batch_size * n] BF16 output
    int n, float eps, int batch_size)
{
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = batch * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[offset + i];
        sum_sq += v * v;
    }
    // Warp-level reduction (no sync needed within warp)
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
    int warp_id = tid / WARP_SIZE;
    int lane = tid & (WARP_SIZE - 1);
    if (lane == 0) sdata[warp_id] = sum_sq;
    __syncthreads();
    // Cross-warp reduction
    if (tid < (blockDim.x / WARP_SIZE)) {
        sum_sq = sdata[tid];
        for (int off = (blockDim.x / (WARP_SIZE * 2)); off > 0; off >>= 1)
            sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
        if (tid == 0) sdata[0] = sum_sq;
    }
    __syncthreads();
    float rms = rsqrtf(sdata[0] / n + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        y[offset + i] = float_to_bf16(x[offset + i] * rms * weight[i]);
    }
}

// ============================================================
// Fused Add + RMSNorm → BF16: y = bf16(rms_norm(x + residual, weight))
// Replaces add_batch + rms_norm_bf16_batch (saves 1 launch + 1 read/write pass)
// Also writes the FP32 sum back to x for the residual path
// ============================================================
__global__ void add_rms_norm_bf16_batch_kernel(
    float* __restrict__ x,             // [batch_size * n] FP32 — gets x += residual IN PLACE
    const float* __restrict__ residual, // [batch_size * n] FP32
    const float* __restrict__ weight,   // [n]
    int16_t* __restrict__ y,           // [batch_size * n] BF16 output
    int n, float eps, int batch_size)
{
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = batch * n;

    // Pass 1: add residual, write back, compute sum of squares
    // Use warp shuffles for first reduction level (no __syncthreads needed)
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[offset + i] + residual[offset + i];
        x[offset + i] = v;  // write back sum for residual path
        sum_sq += v * v;
    }
    // Warp-level reduction first (no sync needed within warp)
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
    int warp_id = tid / WARP_SIZE;
    int lane = tid & (WARP_SIZE - 1);
    if (lane == 0) sdata[warp_id] = sum_sq;
    __syncthreads();
    // Final reduction across warps (max 16 warps for 1024 threads)
    if (tid < (blockDim.x / WARP_SIZE)) {
        sum_sq = sdata[tid];
        for (int off = (blockDim.x / (WARP_SIZE * 2)); off > 0; off >>= 1)
            sum_sq += __shfl_xor(sum_sq, off, WARP_SIZE);
        if (tid == 0) sdata[0] = sum_sq;
    }
    __syncthreads();
    float rms = rsqrtf(sdata[0] / n + eps);

    // Pass 2: normalize, scale, convert to BF16
    // x[offset + i] was written in pass 1 — re-read from L2 (should be cached)
    for (int i = tid; i < n; i += blockDim.x) {
        y[offset + i] = float_to_bf16(x[offset + i] * rms * weight[i]);
    }
}

// ============================================================
// Fused SiLU(gate) * up → BF16 output
// ============================================================
__global__ void silu_mul_bf16_batch_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int16_t* __restrict__ out,       // BF16 output
    int n, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;
    if (idx >= total) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = float_to_bf16(silu * up[idx]);
}

extern "C" {

void launch_rms_norm_bf16_batch(const float* x, const float* weight, int16_t* y,
                                int n, float eps, int batch_size, hipStream_t stream) {
    int threads = min(1024, n);
    int smem = (threads / WARP_SIZE + 1) * sizeof(float);  // warp-shuffle: only need warp count floats
    rms_norm_bf16_batch_kernel<<<batch_size, threads, smem, stream>>>(
        x, weight, y, n, eps, batch_size);
}

void launch_add_rms_norm_bf16_batch(float* x, const float* residual, const float* weight,
                                     int16_t* y, int n, float eps, int batch_size, hipStream_t stream) {
    int threads = min(1024, n);
    int smem = (threads / WARP_SIZE + 1) * sizeof(float);
    add_rms_norm_bf16_batch_kernel<<<batch_size, threads, smem, stream>>>(
        x, residual, weight, y, n, eps, batch_size);
}

void launch_silu_mul_bf16_batch(const float* gate, const float* up, int16_t* out,
                                int n, int batch_size, hipStream_t stream) {
    int threads = 256;
    int total = batch_size * n;
    int blocks = (total + threads - 1) / threads;
    silu_mul_bf16_batch_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n, batch_size);
}

void launch_add_batch(float* y, const float* x, int n, int batch_size, hipStream_t stream) {
    int threads = 256;
    int total = batch_size * n;
    int blocks = (total + threads - 1) / threads;
    add_batch_kernel<<<blocks, threads, 0, stream>>>(y, x, n, batch_size);
}

}

// ============================================================================
// Fused Multi-Head Attention (all heads in one launch)
// ============================================================================
__global__ void attention_all_heads_kernel(
    const float* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const float* __restrict__ kv_scale_k,
    const float* __restrict__ kv_scale_v,
    int16_t* __restrict__ output,
    int seq_len, int max_seq, int n_head_kv, int head_dim,
    int gqa_ratio, float scale)
{
    int head = blockIdx.x;
    int kv_head = head / gqa_ratio;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores = smem;
    float* q_shared = smem + seq_len;

    for (int i = tid; i < head_dim; i += blockDim.x)
        q_shared[i] = q[head * head_dim + i];
    __syncthreads();

    // Q·K^T with INT8 dequantization
    for (int t = tid; t < seq_len; t += blockDim.x) {
        int k_base = t * n_head_kv * head_dim + kv_head * head_dim;
        float sk = kv_scale_k[t * n_head_kv + kv_head];
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_shared[d] * ((float)k_cache[k_base + d] * sk);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    __shared__ float warp_reduce[16];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off, WARP_SIZE));
    if (lane == 0) warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = warp_reduce[0];
        for (int i = 1; i < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE; i++)
            m = fmaxf(m, warp_reduce[i]);
        warp_reduce[0] = m;
    }
    __syncthreads();
    float gmax = warp_reduce[0];

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - gmax);
        scores[t] = e;
        local_sum += e;
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off, WARP_SIZE);
    if (lane == 0) warp_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE; i++)
            s += warp_reduce[i];
        warp_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_reduce[0];

    for (int t = tid; t < seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Weighted sum of V with INT8 dequantization
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        int v_off = kv_head * head_dim + d;
        for (int t = 0; t < seq_len; t++) {
            float sv = kv_scale_v[t * n_head_kv + kv_head];
            acc += scores[t] * ((float)v_cache[t * n_head_kv * head_dim + v_off] * sv);
        }
        output[head * head_dim + d] = float_to_bf16(acc);
    }
}

extern "C" {
void launch_attention_all_heads(const float* q, const int8_t* k_cache, const int8_t* v_cache,
    const float* kv_scale_k, const float* kv_scale_v,
    int16_t* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq,
    float scale, hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = 128;  // per head block
    int smem = (seq_len + head_dim) * sizeof(float) + 16 * sizeof(float);
    attention_all_heads_kernel<<<n_head, threads, smem, stream>>>(
        q, k_cache, v_cache, kv_scale_k, kv_scale_v, output, seq_len, max_seq, n_head_kv, head_dim, gqa_ratio, scale);
}
}

// ============================================================================
// Batched Multi-Head Attention — B sequences × N heads in one launch
// Grid: (n_head, batch_size), each block handles one head of one sequence
// ============================================================================
__global__ void attention_all_heads_batch_kernel(
    const float* __restrict__ q,           // [batch_size * n_head * head_dim]
    const int8_t* __restrict__ kv_cache_k,  // [n_layer * max_seq * n_head_kv * head_dim] INT8
    const int8_t* __restrict__ kv_cache_v,
    const float* __restrict__ kv_scale_k,  // [max_seq * n_head_kv] per-(position, kv_head) scale
    const float* __restrict__ kv_scale_v,
    int16_t* __restrict__ output,          // [batch_size * n_head * head_dim] BF16
    const int* __restrict__ positions,     // [batch_size] current position per sequence
    int n_head, int n_head_kv, int head_dim,
    int max_seq, int gqa_ratio, float scale,
    int kv_stride)                         // stride between sequences in KV cache (0 = shared cache)
{
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int kv_head = head / gqa_ratio;
    int tid = threadIdx.x;
    int pos = positions[seq];
    int seq_len = pos + 1;

    int n_total = n_head * head_dim;
    
    extern __shared__ float smem[];
    float* scores = smem;
    float* q_shared = smem + seq_len;

    // Load Q for this head of this sequence
    for (int i = tid; i < head_dim; i += blockDim.x)
        q_shared[i] = q[seq * n_total + head * head_dim + i];
    __syncthreads();

    // Q·K^T with INT8 dequantization (KV cache offset depends on whether cache is shared or per-sequence)
    const int8_t* k = kv_cache_k + (size_t)seq * kv_stride;
    const int8_t* v = kv_cache_v + (size_t)seq * kv_stride;
    // Scale arrays have stride max_seq * n_head_kv per sequence
    const float* sk_base = kv_scale_k + (size_t)seq * (kv_stride / head_dim);
    const float* sv_base = kv_scale_v + (size_t)seq * (kv_stride / head_dim);

    for (int t = tid; t < seq_len; t += blockDim.x) {
        int k_off = t * n_head_kv * head_dim + kv_head * head_dim;
        float sk = sk_base[t * n_head_kv + kv_head];
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_shared[d] * ((float)k[k_off + d] * sk);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    __shared__ float warp_reduce[16];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off, WARP_SIZE));
    if (lane == 0) warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) { float m = warp_reduce[0]; for (int i=1;i<(blockDim.x + WARP_SIZE - 1) / WARP_SIZE;i++) m=fmaxf(m,warp_reduce[i]); warp_reduce[0]=m; }
    __syncthreads();
    float gmax = warp_reduce[0];

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - gmax); scores[t] = e; local_sum += e;
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) local_sum += __shfl_xor(local_sum, off, WARP_SIZE);
    if (lane == 0) warp_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) { float s=0; for (int i=0;i<(blockDim.x + WARP_SIZE - 1) / WARP_SIZE;i++) s+=warp_reduce[i]; warp_reduce[0]=s; }
    __syncthreads();
    float inv_sum = 1.0f / warp_reduce[0];
    for (int t = tid; t < seq_len; t += blockDim.x) scores[t] *= inv_sum;
    __syncthreads();

    // Weighted sum of V with INT8 dequantization
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        int v_off = kv_head * head_dim + d;
        for (int t = 0; t < seq_len; t++) {
            float sv = sv_base[t * n_head_kv + kv_head];
            acc += scores[t] * ((float)v[t * n_head_kv * head_dim + v_off] * sv);
        }
        output[seq * n_total + head * head_dim + d] = float_to_bf16(acc);
    }
}

extern "C" {
void launch_attention_all_heads_batch(
    const float* q, const int8_t* kv_k, const int8_t* kv_v,
    const float* kv_scale_k, const float* kv_scale_v,
    int16_t* output, const int* positions,
    int n_head, int n_head_kv, int head_dim,
    int max_seq, float scale, int batch_size, int kv_stride,
    hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = 128;
    // max_seq is passed as the actual max position+1 needed for smem allocation
    int smem = (max_seq + head_dim) * sizeof(float) + 16 * sizeof(float);
    dim3 grid(n_head, batch_size);
    attention_all_heads_batch_kernel<<<grid, threads, smem, stream>>>(
        q, kv_k, kv_v, kv_scale_k, kv_scale_v, output, positions, n_head, n_head_kv, head_dim,
        max_seq, gqa_ratio, scale, kv_stride);
}
}

// ============================================================================
// Batched RoPE: process B sequences in one launch
// ============================================================================
__global__ void rope_batch_kernel(
    float* __restrict__ q,   // [batch_size * n_head * head_dim]
    float* __restrict__ k,   // [batch_size * n_head_kv * head_dim]
    const int* __restrict__ positions,  // [batch_size]
    const float* __restrict__ freqs,    // [half_dim] precomputed
    int head_dim, int n_head, int n_head_kv,
    int q_stride, int k_stride)
{
    int seq = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int position = positions[seq];

    // Q heads
    if (idx < n_head * half_dim) {
        int head = idx / half_dim;
        int d = idx % half_dim;
        float angle = position * freqs[d];
        float cos_a, sin_a;
        __sincosf(angle, &sin_a, &cos_a);
        int base = seq * q_stride + head * head_dim;
        float x0 = q[base + d], x1 = q[base + d + half_dim];
        q[base + d]            = x0 * cos_a - x1 * sin_a;
        q[base + d + half_dim] = x0 * sin_a + x1 * cos_a;
    }
    // K heads
    if (idx < n_head_kv * half_dim) {
        int head = idx / half_dim;
        int d = idx % half_dim;
        float angle = position * freqs[d];
        float cos_a, sin_a;
        __sincosf(angle, &sin_a, &cos_a);
        int base = seq * k_stride + head * head_dim;
        float x0 = k[base + d], x1 = k[base + d + half_dim];
        k[base + d]            = x0 * cos_a - x1 * sin_a;
        k[base + d + half_dim] = x0 * sin_a + x1 * cos_a;
    }
}

// Batched KV store with INT8 quantization: store B sequences' K,V into cache
// Grid: (n_head_kv, 2, batch_size) -- x=kv_head, y=0(K)/1(V), z=sequence
// Threads: head_dim (rounded up to warp multiple)
// Each block cooperatively quantizes one (sequence, kv_head) group.
__global__ void store_kv_batch_int8_kernel(
    const float* __restrict__ k_in,    // [batch_size * kv_dim]
    const float* __restrict__ v_in,
    int8_t* __restrict__ kv_k,         // [max_seq * kv_dim] INT8
    int8_t* __restrict__ kv_v,
    float* __restrict__ kv_scale_k,    // [max_seq * n_head_kv]
    float* __restrict__ kv_scale_v,
    const int* __restrict__ positions,
    int head_dim, int n_head_kv,
    int max_seq, int batch_size)
{
    int kv_head = blockIdx.x;
    int is_v = blockIdx.y;  // 0 = K, 1 = V
    int seq = blockIdx.z;
    int tid = threadIdx.x;

    if (kv_head >= n_head_kv || seq >= batch_size) return;

    extern __shared__ float smem[];

    const float* src = is_v ? v_in : k_in;
    int8_t* dst = is_v ? kv_v : kv_k;
    float* scale_dst = is_v ? kv_scale_v : kv_scale_k;

    int kv_dim = n_head_kv * head_dim;
    int pos = positions[seq];
    int src_offset = seq * kv_dim + kv_head * head_dim;
    int cache_offset = pos * kv_dim + kv_head * head_dim;
    int scale_offset = pos * n_head_kv + kv_head;

    // Step 1: Load value and compute thread-local absmax
    float val = 0.0f;
    float thread_absmax = 0.0f;
    if (tid < head_dim) {
        val = src[src_offset + tid];
        thread_absmax = fabsf(val);
    }

    // Step 2: Warp-level absmax reduction via shuffle
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    float warp_max = thread_absmax;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(warp_max, offset, WARP_SIZE);
        warp_max = fmaxf(warp_max, other);
    }

    // Step 3: Write per-warp max to shared memory
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (lane == 0) {
        smem[warp_id] = warp_max;
    }
    __syncthreads();

    // Step 4: First warp reduces across all warps
    float global_max = 0.0f;
    if (tid < num_warps) {
        global_max = smem[tid];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor(global_max, offset, WARP_SIZE);
        global_max = fmaxf(global_max, other);
    }

    // Step 5: Broadcast global_max
    if (tid == 0) {
        smem[0] = global_max;
    }
    __syncthreads();
    global_max = smem[0];

    float scale = global_max / 127.0f;
    float inv_scale = (global_max > 0.0f) ? 127.0f / global_max : 0.0f;

    // Step 6: Quantize and store
    if (tid < head_dim) {
        float q_val = rintf(val * inv_scale);
        q_val = fminf(fmaxf(q_val, -127.0f), 127.0f);
        dst[cache_offset + tid] = (int8_t)((int)q_val);
    }

    // Step 7: Store scale factor
    if (tid == 0) {
        scale_dst[scale_offset] = scale;
    }
}

extern "C" {
void launch_rope_batch(float* q, float* k, const int* positions,
    int head_dim, int n_head, int n_head_kv,
    int q_stride, int k_stride,
    float theta, int batch_size, hipStream_t stream) {
    launch_rope_init(head_dim, theta);
    int half_dim = head_dim / 2;
    int max_threads = max(n_head, n_head_kv) * half_dim;
    dim3 grid((max_threads + 255) / 256, batch_size);
    rope_batch_kernel<<<grid, 256, 0, stream>>>(
        q, k, positions, d_rope_freqs, head_dim, n_head, n_head_kv, q_stride, k_stride);
}

void launch_store_kv_batch(const float* k, const float* v,
    int8_t* kv_k, int8_t* kv_v,
    float* kv_scale_k, float* kv_scale_v,
    const int* positions,
    int kv_dim, int max_seq, int n_head_kv, int head_dim,
    int batch_size, hipStream_t stream) {
    // kv_dim = n_head_kv * head_dim (passed for compatibility, derived internally)
    int store_threads = ((head_dim + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int num_warps = (store_threads + WARP_SIZE - 1) / WARP_SIZE;
    int smem_bytes = num_warps * sizeof(float);
    dim3 grid(n_head_kv, 2, batch_size);  // x=kv_head, y=0(K)/1(V), z=seq
    store_kv_batch_int8_kernel<<<grid, store_threads, smem_bytes, stream>>>(
        k, v, kv_k, kv_v, kv_scale_k, kv_scale_v,
        positions, head_dim, n_head_kv, max_seq, batch_size);
}
}

// ============================================================================
// Flash Attention v2 — tiled KV, O(1) shared memory, BF16 KV cache
// Tc=128 tile, 128 threads = head_dim, ~34 KB LDS (constant, no seq_len dep)
// ============================================================================
#define FA_TC 128  // KV tile size

__global__ void flash_attention_kernel(
    const float* __restrict__ q,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    const float* __restrict__ kv_scale_k,
    const float* __restrict__ kv_scale_v,
    int16_t* __restrict__ output,
    int seq_len, int n_head_kv, int head_dim,
    int gqa_ratio, float scale)
{
    const int head = blockIdx.x;
    const int kv_head = head / gqa_ratio;
    const int tid = threadIdx.x;  // 0..head_dim-1 (128)
    const int kv_stride = n_head_kv * head_dim;

    extern __shared__ float smem_fa_raw[];
    char* smem = (char*)smem_fa_raw;
    float*   q_s     = (float*)smem;                                     // [head_dim]
    float*   kv_tile = (float*)(smem + head_dim * sizeof(float));        // [FA_TC][head_dim] floats
    float*   scores  = (float*)(smem + head_dim * 4 + FA_TC * head_dim * 4); // [FA_TC]
    float*   warp_r  = scores + FA_TC;                                    // [4]

    // Load Q once
    q_s[tid] = q[head * head_dim + tid];
    __syncthreads();

    // Online softmax accumulators (per-thread, in registers)
    float O_acc = 0.0f;
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    int n_tiles = (seq_len + FA_TC - 1) / FA_TC;

    for (int tile = 0; tile < n_tiles; tile++) {
        int t_start = tile * FA_TC;
        int t_len = min(FA_TC, seq_len - t_start);

        // Phase A: Load K tile [t_len][head_dim] — dequantize INT8 to float during load
        int total = t_len * head_dim;
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            float sk = kv_scale_k[(t_start + row) * n_head_kv + kv_head];
            kv_tile[i] = (float)k_cache[(t_start + row) * kv_stride + kv_head * head_dim + col] * sk;
        }
        __syncthreads();

        // Phase A: Compute Q·K^T scores — 1 thread per KV position
        float s = -1e30f;
        if (tid < t_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_s[d] * kv_tile[tid * head_dim + d];
            s = dot * scale;
        }
        scores[tid] = s;

        // Tile max via warp shuffle
        float t_max = s;
        int lane = tid & (WARP_SIZE - 1), wid = tid / WARP_SIZE;
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            t_max = fmaxf(t_max, __shfl_xor(t_max, off, WARP_SIZE));
        if (lane == 0) warp_r[wid] = t_max;
        __syncthreads();
        if (tid == 0) { for (int w = 1; w < blockDim.x / WARP_SIZE; w++) warp_r[0] = fmaxf(warp_r[0], warp_r[w]); }
        __syncthreads();
        t_max = warp_r[0];

        float m_new = fmaxf(m_prev, t_max);

        // Exp + sum
        float p = (tid < t_len) ? expf(scores[tid] - m_new) : 0.0f;
        scores[tid] = p;  // store for V accumulation

        float p_sum = p;
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            p_sum += __shfl_xor(p_sum, off, WARP_SIZE);
        if (lane == 0) warp_r[wid] = p_sum;
        __syncthreads();
        if (tid == 0) { for (int w = 1; w < blockDim.x / WARP_SIZE; w++) warp_r[0] += warp_r[w]; }
        __syncthreads();
        p_sum = warp_r[0];

        // Rescale previous accumulator
        float correction = expf(m_prev - m_new);
        O_acc *= correction;
        float l_new = correction * l_prev + p_sum;

        __syncthreads();

        // Phase B: Load V tile (reuses kv_tile LDS region) — dequantize INT8 to float
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            float sv = kv_scale_v[(t_start + row) * n_head_kv + kv_head];
            kv_tile[i] = (float)v_cache[(t_start + row) * kv_stride + kv_head * head_dim + col] * sv;
        }
        __syncthreads();

        // Phase B: Accumulate P·V — thread tid handles output dim tid
        float pv = 0.0f;
        for (int t = 0; t < t_len; t++)
            pv += scores[t] * kv_tile[t * head_dim + tid];
        O_acc += pv;

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Final normalization
    output[head * head_dim + tid] = float_to_bf16(O_acc / l_prev);
}

// Flash Attention Batch — B sequences × N heads
__global__ void flash_attention_batch_kernel(
    const float* __restrict__ q,
    const int8_t* __restrict__ kv_cache_k,
    const int8_t* __restrict__ kv_cache_v,
    const float* __restrict__ kv_scale_k,
    const float* __restrict__ kv_scale_v,
    int16_t* __restrict__ output,
    const int* __restrict__ positions,
    int n_head, int n_head_kv, int head_dim,
    int max_seq, int gqa_ratio, float scale,
    int kv_stride_batch)
{
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int kv_head = head / gqa_ratio;
    const int tid = threadIdx.x;
    const int kv_stride = n_head_kv * head_dim;
    const int seq_len = positions[seq] + 1;
    const int n_total = n_head * head_dim;

    extern __shared__ float smem_fa_raw[];
    char* smem = (char*)smem_fa_raw;
    float*   q_s     = (float*)smem;
    float*   kv_tile = (float*)(smem + head_dim * sizeof(float));        // [FA_TC][head_dim] floats
    float*   scores  = (float*)(smem + head_dim * 4 + FA_TC * head_dim * 4);
    float*   warp_r  = scores + FA_TC;

    // KV cache offset for this sequence
    const int8_t* k_base = kv_cache_k + (size_t)seq * kv_stride_batch;
    const int8_t* v_base = kv_cache_v + (size_t)seq * kv_stride_batch;
    // Scale arrays: stride is kv_stride_batch / head_dim = max_seq * n_head_kv per sequence
    const float* sk_base = kv_scale_k + (size_t)seq * (kv_stride_batch / head_dim);
    const float* sv_base = kv_scale_v + (size_t)seq * (kv_stride_batch / head_dim);

    q_s[tid] = q[seq * n_total + head * head_dim + tid];
    __syncthreads();

    float O_acc = 0.0f, m_prev = -1e30f, l_prev = 0.0f;
    int n_tiles = (seq_len + FA_TC - 1) / FA_TC;

    for (int tile = 0; tile < n_tiles; tile++) {
        int t_start = tile * FA_TC;
        int t_len = min(FA_TC, seq_len - t_start);
        int total = t_len * head_dim;

        // Load K tile — dequantize INT8 to float during load
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            float sk = sk_base[(t_start + row) * n_head_kv + kv_head];
            kv_tile[i] = (float)k_base[(t_start + row) * kv_stride + kv_head * head_dim + col] * sk;
        }
        __syncthreads();

        float s = -1e30f;
        if (tid < t_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_s[d] * kv_tile[tid * head_dim + d];
            s = dot * scale;
        }
        scores[tid] = s;

        float t_max = s;
        int lane = tid & (WARP_SIZE - 1), wid = tid / WARP_SIZE;
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            t_max = fmaxf(t_max, __shfl_xor(t_max, off, WARP_SIZE));
        if (lane == 0) warp_r[wid] = t_max;
        __syncthreads();
        if (tid == 0) { for (int w = 1; w < blockDim.x / WARP_SIZE; w++) warp_r[0] = fmaxf(warp_r[0], warp_r[w]); }
        __syncthreads();
        t_max = warp_r[0];
        float m_new = fmaxf(m_prev, t_max);

        float p = (tid < t_len) ? expf(scores[tid] - m_new) : 0.0f;
        scores[tid] = p;
        float p_sum = p;
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            p_sum += __shfl_xor(p_sum, off, WARP_SIZE);
        if (lane == 0) warp_r[wid] = p_sum;
        __syncthreads();
        if (tid == 0) { for (int w = 1; w < blockDim.x / WARP_SIZE; w++) warp_r[0] += warp_r[w]; }
        __syncthreads();
        p_sum = warp_r[0];

        float correction = expf(m_prev - m_new);
        O_acc *= correction;
        float l_new = correction * l_prev + p_sum;
        __syncthreads();

        // Load V tile — dequantize INT8 to float during load
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            float sv = sv_base[(t_start + row) * n_head_kv + kv_head];
            kv_tile[i] = (float)v_base[(t_start + row) * kv_stride + kv_head * head_dim + col] * sv;
        }
        __syncthreads();

        float pv = 0.0f;
        for (int t = 0; t < t_len; t++)
            pv += scores[t] * kv_tile[t * head_dim + tid];
        O_acc += pv;

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    output[seq * n_total + head * head_dim + tid] = float_to_bf16(O_acc / l_prev);
}

extern "C" {
void launch_flash_attention(const float* q, const int8_t* k_cache, const int8_t* v_cache,
    const float* kv_scale_k, const float* kv_scale_v,
    int16_t* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq,
    float scale, hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = head_dim;  // 128 for Mistral
    // kv_tile is now float (4 bytes) instead of int16_t (2 bytes)
    int smem = head_dim * 4 + FA_TC * head_dim * 4 + FA_TC * 4 + 16;
    flash_attention_kernel<<<n_head, threads, smem, stream>>>(
        q, k_cache, v_cache, kv_scale_k, kv_scale_v, output, seq_len, n_head_kv, head_dim, gqa_ratio, scale);
}

void launch_flash_attention_batch(const float* q, const int8_t* kv_k, const int8_t* kv_v,
    const float* kv_scale_k, const float* kv_scale_v,
    int16_t* output, const int* positions,
    int n_head, int n_head_kv, int head_dim,
    int max_seq, float scale, int batch_size, int kv_stride,
    hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = head_dim;
    // kv_tile is now float (4 bytes) instead of int16_t (2 bytes)
    int smem = head_dim * 4 + FA_TC * head_dim * 4 + FA_TC * 4 + 16;
    dim3 grid(n_head, batch_size);
    flash_attention_batch_kernel<<<grid, threads, smem, stream>>>(
        q, kv_k, kv_v, kv_scale_k, kv_scale_v, output, positions, n_head, n_head_kv, head_dim,
        max_seq, gqa_ratio, scale, kv_stride);
}
}
