/**
 * Turbo Lossless Engine — HIP Kernels
 * All non-matvec transformer operations for MI50 (gfx906)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

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
// RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight
// ============================================================
__global__ void rms_norm_kernel(
    const float* __restrict__ x,      // [n]
    const float* __restrict__ weight,  // [n]
    float* __restrict__ y,            // [n]
    int n, float eps)
{
    // One block per row, threads cooperate on reduction
    // Safe for in-place (x == y): reads all x first, syncs, then writes
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }

    // Block reduction
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / n + eps);
    __syncthreads();

    // Pass 2: read x, compute, write y (safe because all threads have rms now)
    // For in-place safety: each thread reads its elements, computes, writes
    // This is safe because thread i only writes to indices i, i+blockDim, i+2*blockDim...
    // and reads the same indices — no cross-thread read/write conflict
    for (int i = tid; i < n; i += blockDim.x) {
        y[i] = x[i] * rms * weight[i];
    }
}

// ============================================================
// RoPE: Rotary Position Embeddings (with precomputed frequencies)
// ============================================================

// Precomputed frequency table on GPU (set once during init)
static float* d_rope_freqs = nullptr;
static int d_rope_freqs_dim = 0;

__global__ void rope_kernel(
    float* __restrict__ q,   // [n_head, head_dim]
    float* __restrict__ k,   // [n_head_kv, head_dim]
    const float* __restrict__ freqs,  // [half_dim] precomputed
    int head_dim, int n_head, int n_head_kv,
    int position)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    // Apply to Q heads
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

    // Apply to K heads
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
    int n_embd, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (seq >= batch_size || idx >= n_embd) return;

    int token_id = token_ids[seq];
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
    const float* __restrict__ logits,  // [n_vocab]
    int* __restrict__ result,          // [1]
    int n)
{
    extern __shared__ float sdata[];
    int* sidx = (int*)(sdata + blockDim.x);
    int tid = threadIdx.x;

    float max_val = -INFINITY;
    int max_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > max_val) {
            max_val = logits[i];
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

    if (tid == 0) result[0] = sidx[0];
}

// ============================================================
// C launch wrappers
// ============================================================
extern "C" {

void launch_rms_norm(const float* x, const float* weight, float* y,
                     int n, float eps, hipStream_t stream) {
    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(x, weight, y, n, eps);
}

void launch_rope_init(int head_dim, float theta_base) {
    if (d_rope_freqs && d_rope_freqs_dim == head_dim) return;
    int half_dim = head_dim / 2;
    float* h_freqs = new float[half_dim];
    for (int d = 0; d < half_dim; d++)
        h_freqs[d] = 1.0f / powf(theta_base, (float)(2 * d) / head_dim);
    if (d_rope_freqs) hipFree(d_rope_freqs);
    hipMalloc(&d_rope_freqs, half_dim * sizeof(float));
    hipMemcpy(d_rope_freqs, h_freqs, half_dim * sizeof(float), hipMemcpyHostToDevice);
    d_rope_freqs_dim = head_dim;
    delete[] h_freqs;
}

void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv,
                 int position, float theta, hipStream_t stream) {
    launch_rope_init(head_dim, theta);
    int half_dim = head_dim / 2;
    int max_threads = max(n_head, n_head_kv) * half_dim;
    int threads = 256;
    int blocks = (max_threads + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(q, k, d_rope_freqs, head_dim, n_head, n_head_kv, position);
}

void launch_embed_lookup(const int16_t* embd_table, const int* token_ids,
                         float* output, int n_embd, int batch_size, hipStream_t stream) {
    dim3 threads(256);
    dim3 blocks((n_embd + 255) / 256, batch_size);
    embed_lookup_kernel<<<blocks, threads, 0, stream>>>(embd_table, token_ids, output, n_embd, batch_size);
}

void launch_argmax(const float* logits, int* result, int n, hipStream_t stream) {
    int threads = 256;
    argmax_kernel<<<1, threads, threads * (sizeof(float) + sizeof(int)), stream>>>(logits, result, n);
}

}

// ============================================================
// Attention: Q×K^T → softmax → ×V with GQA support
// ============================================================

// Store K and V into the KV cache at the current position
__global__ void store_kv_kernel(
    const float* __restrict__ k_in,     // [n_head_kv * head_dim] FP32
    const float* __restrict__ v_in,     // [n_head_kv * head_dim] FP32
    int16_t* __restrict__ kv_cache_k,   // [max_seq, n_head_kv, head_dim] BF16
    int16_t* __restrict__ kv_cache_v,
    int pos, int n_head_kv, int head_dim, int max_seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_head_kv * head_dim;
    if (idx >= total) return;

    int cache_idx = pos * total + idx;
    kv_cache_k[cache_idx] = float_to_bf16(k_in[idx]);
    kv_cache_v[cache_idx] = float_to_bf16(v_in[idx]);
}

extern "C" {

void launch_store_kv(const float* k, const float* v,
                     int16_t* kv_k, int16_t* kv_v,
                     int pos, int n_head_kv, int head_dim, int max_seq,
                     hipStream_t stream) {
    int total = n_head_kv * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    store_kv_kernel<<<blocks, threads, 0, stream>>>(k, v, kv_k, kv_v, pos, n_head_kv, head_dim, max_seq);
}

}

// ============================================================
// FP32 → BF16 conversion kernel (GPU, replaces CPU conversion)
// ============================================================
__global__ void fp32_to_bf16_kernel(
    const float* __restrict__ input,
    int16_t* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = float_to_bf16(input[idx]);
}

extern "C" {
void launch_fp32_to_bf16(const float* input, int16_t* output, int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}
}

// ============================================================
// Batched RMSNorm: process B vectors of length n in one launch
// x,y are [batch_size * n], weight is [n] (shared across batch)
// One block per batch item, threads cooperate on reduction
// ============================================================
__global__ void rms_norm_batch_kernel(
    const float* __restrict__ x,      // [batch_size * n]
    const float* __restrict__ weight,  // [n]
    float* __restrict__ y,            // [batch_size * n]
    int n, float eps, int batch_size)
{
    int batch = blockIdx.x;
    if (batch >= batch_size) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = batch * n;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[offset + i];
        sum_sq += v * v;
    }

    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / n + eps);
    __syncthreads();

    // Pass 2: normalize and scale
    for (int i = tid; i < n; i += blockDim.x) {
        y[offset + i] = x[offset + i] * rms * weight[i];
    }
}

// ============================================================
// Batched SiLU multiply: SiLU(gate) * up for B*n elements
// ============================================================
__global__ void silu_mul_batch_kernel(
    const float* __restrict__ gate,   // [batch_size * n]
    const float* __restrict__ up,     // [batch_size * n]
    float* __restrict__ out,          // [batch_size * n]
    int n, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;
    if (idx >= total) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * up[idx];
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
// Batched memcpy: GPU-side copy of batch_size * n floats
// ============================================================
__global__ void memcpy_batch_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * n;
    if (idx >= total) return;
    dst[idx] = src[idx];
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
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / n + eps);
    __syncthreads();

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

    // Pass 1: add residual and compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[offset + i] + residual[offset + i];
        x[offset + i] = v;  // write back sum for residual path
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / n + eps);
    __syncthreads();

    // Pass 2: normalize, scale, convert to BF16
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

void launch_rms_norm_batch(const float* x, const float* weight, float* y,
                           int n, float eps, int batch_size, hipStream_t stream) {
    int threads = 256;
    rms_norm_batch_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        x, weight, y, n, eps, batch_size);
}

void launch_rms_norm_bf16_batch(const float* x, const float* weight, int16_t* y,
                                int n, float eps, int batch_size, hipStream_t stream) {
    int threads = 256;
    rms_norm_bf16_batch_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        x, weight, y, n, eps, batch_size);
}

void launch_add_rms_norm_bf16_batch(float* x, const float* residual, const float* weight,
                                     int16_t* y, int n, float eps, int batch_size, hipStream_t stream) {
    int threads = 256;
    add_rms_norm_bf16_batch_kernel<<<batch_size, threads, threads * sizeof(float), stream>>>(
        x, residual, weight, y, n, eps, batch_size);
}

void launch_silu_mul_batch(const float* gate, const float* up, float* out,
                           int n, int batch_size, hipStream_t stream) {
    int threads = 256;
    int total = batch_size * n;
    int blocks = (total + threads - 1) / threads;
    silu_mul_batch_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n, batch_size);
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

void launch_memcpy_batch(float* dst, const float* src, int n, int batch_size, hipStream_t stream) {
    int threads = 256;
    int total = batch_size * n;
    int blocks = (total + threads - 1) / threads;
    memcpy_batch_kernel<<<blocks, threads, 0, stream>>>(dst, src, n, batch_size);
}

}

// ============================================================================
// Fused Multi-Head Attention (all heads in one launch)
// ============================================================================
__global__ void attention_all_heads_kernel(
    const float* __restrict__ q,
    const int16_t* __restrict__ k_cache,
    const int16_t* __restrict__ v_cache,
    float* __restrict__ output,
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

    // Q·K^T
    for (int t = tid; t < seq_len; t += blockDim.x) {
        int k_base = t * n_head_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_shared[d] * bf16_to_float(k_cache[k_base + d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    __shared__ float warp_reduce[16];
    int lane = tid % 64;
    int warp_id = tid / 64;

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    for (int off = 32; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off, 64));
    if (lane == 0) warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = warp_reduce[0];
        for (int i = 1; i < (blockDim.x + 63) / 64; i++)
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
    for (int off = 32; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off, 64);
    if (lane == 0) warp_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < (blockDim.x + 63) / 64; i++)
            s += warp_reduce[i];
        warp_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_reduce[0];

    for (int t = tid; t < seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Weighted sum of V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        int v_off = kv_head * head_dim + d;
        for (int t = 0; t < seq_len; t++)
            acc += scores[t] * bf16_to_float(v_cache[t * n_head_kv * head_dim + v_off]);
        output[head * head_dim + d] = acc;
    }
}

extern "C" {
void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache,
    float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq,
    float scale, hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = 128;  // per head block
    int smem = (seq_len + head_dim) * sizeof(float) + 16 * sizeof(float);
    attention_all_heads_kernel<<<n_head, threads, smem, stream>>>(
        q, k_cache, v_cache, output, seq_len, max_seq, n_head_kv, head_dim, gqa_ratio, scale);
}
}

// ============================================================================
// Batched Multi-Head Attention — B sequences × N heads in one launch
// Grid: (n_head, batch_size), each block handles one head of one sequence
// ============================================================================
__global__ void attention_all_heads_batch_kernel(
    const float* __restrict__ q,           // [batch_size * n_head * head_dim]
    const int16_t* __restrict__ kv_cache_k, // [n_layer * max_seq * n_head_kv * head_dim] (shared)
    const int16_t* __restrict__ kv_cache_v,
    float* __restrict__ output,            // [batch_size * n_head * head_dim]
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

    // Q·K^T  (KV cache offset depends on whether cache is shared or per-sequence)
    const int16_t* k = kv_cache_k + (size_t)seq * kv_stride;
    const int16_t* v = kv_cache_v + (size_t)seq * kv_stride;

    for (int t = tid; t < seq_len; t += blockDim.x) {
        int k_base = t * n_head_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_shared[d] * bf16_to_float(k[k_base + d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    __shared__ float warp_reduce[16];
    int lane = tid % 64;
    int warp_id = tid / 64;

    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    for (int off = 32; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off, 64));
    if (lane == 0) warp_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) { float m = warp_reduce[0]; for (int i=1;i<(blockDim.x+63)/64;i++) m=fmaxf(m,warp_reduce[i]); warp_reduce[0]=m; }
    __syncthreads();
    float gmax = warp_reduce[0];

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - gmax); scores[t] = e; local_sum += e;
    }
    for (int off = 32; off > 0; off >>= 1) local_sum += __shfl_xor(local_sum, off, 64);
    if (lane == 0) warp_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) { float s=0; for (int i=0;i<(blockDim.x+63)/64;i++) s+=warp_reduce[i]; warp_reduce[0]=s; }
    __syncthreads();
    float inv_sum = 1.0f / warp_reduce[0];
    for (int t = tid; t < seq_len; t += blockDim.x) scores[t] *= inv_sum;
    __syncthreads();

    // Weighted sum of V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        int v_off = kv_head * head_dim + d;
        for (int t = 0; t < seq_len; t++)
            acc += scores[t] * bf16_to_float(v[t * n_head_kv * head_dim + v_off]);
        output[seq * n_total + head * head_dim + d] = acc;
    }
}

extern "C" {
void launch_attention_all_heads_batch(
    const float* q, const int16_t* kv_k, const int16_t* kv_v,
    float* output, const int* positions,
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
        q, kv_k, kv_v, output, positions, n_head, n_head_kv, head_dim,
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

// Batched KV store: store B sequences' K,V into cache
__global__ void store_kv_batch_kernel(
    const float* __restrict__ k_in,    // [batch_size * kv_dim]
    const float* __restrict__ v_in,
    int16_t* __restrict__ kv_k,        // [max_seq * kv_dim]
    int16_t* __restrict__ kv_v,
    const int* __restrict__ positions,
    int kv_dim, int max_seq, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq = blockIdx.y;
    if (seq >= batch_size || idx >= kv_dim) return;
    int pos = positions[seq];
    int cache_idx = pos * kv_dim + idx;
    int src_idx = seq * kv_dim + idx;
    kv_k[cache_idx] = float_to_bf16(k_in[src_idx]);
    kv_v[cache_idx] = float_to_bf16(v_in[src_idx]);
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
    int16_t* kv_k, int16_t* kv_v, const int* positions,
    int kv_dim, int max_seq, int batch_size, hipStream_t stream) {
    dim3 grid((kv_dim + 255) / 256, batch_size);
    store_kv_batch_kernel<<<grid, 256, 0, stream>>>(
        k, v, kv_k, kv_v, positions, kv_dim, max_seq, batch_size);
}
}

// ============================================================================
// Flash Attention v2 — tiled KV, O(1) shared memory, BF16 KV cache
// Tc=128 tile, 128 threads = head_dim, ~34 KB LDS (constant, no seq_len dep)
// ============================================================================
#define FA_TC 128  // KV tile size

__global__ void flash_attention_kernel(
    const float* __restrict__ q,
    const int16_t* __restrict__ k_cache,
    const int16_t* __restrict__ v_cache,
    float* __restrict__ output,
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
    int16_t* kv_tile = (int16_t*)(smem + head_dim * sizeof(float));      // [FA_TC][head_dim]
    float*   scores  = (float*)(smem + head_dim * 4 + FA_TC * head_dim * 2); // [FA_TC]
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

        // Phase A: Load K tile [t_len][head_dim] as BF16 into LDS
        int total = t_len * head_dim;
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            kv_tile[i] = k_cache[(t_start + row) * kv_stride + kv_head * head_dim + col];
        }
        __syncthreads();

        // Phase A: Compute Q·K^T scores — 1 thread per KV position
        float s = -1e30f;
        if (tid < t_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_s[d] * bf16_to_float(kv_tile[tid * head_dim + d]);
            s = dot * scale;
        }
        scores[tid] = s;

        // Tile max via warp shuffle (2 wavefronts of 64)
        float t_max = s;
        int lane = tid & 63, wid = tid >> 6;
        for (int off = 32; off > 0; off >>= 1)
            t_max = fmaxf(t_max, __shfl_xor(t_max, off, 64));
        if (lane == 0) warp_r[wid] = t_max;
        __syncthreads();
        if (tid == 0) warp_r[0] = fmaxf(warp_r[0], warp_r[1]);
        __syncthreads();
        t_max = warp_r[0];

        float m_new = fmaxf(m_prev, t_max);

        // Exp + sum
        float p = (tid < t_len) ? expf(scores[tid] - m_new) : 0.0f;
        scores[tid] = p;  // store for V accumulation

        float p_sum = p;
        for (int off = 32; off > 0; off >>= 1)
            p_sum += __shfl_xor(p_sum, off, 64);
        if (lane == 0) warp_r[wid] = p_sum;
        __syncthreads();
        if (tid == 0) warp_r[0] += warp_r[1];
        __syncthreads();
        p_sum = warp_r[0];

        // Rescale previous accumulator
        float correction = expf(m_prev - m_new);
        O_acc *= correction;
        float l_new = correction * l_prev + p_sum;

        __syncthreads();

        // Phase B: Load V tile (reuses kv_tile LDS region)
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            kv_tile[i] = v_cache[(t_start + row) * kv_stride + kv_head * head_dim + col];
        }
        __syncthreads();

        // Phase B: Accumulate P·V — thread tid handles output dim tid
        float pv = 0.0f;
        for (int t = 0; t < t_len; t++)
            pv += scores[t] * bf16_to_float(kv_tile[t * head_dim + tid]);
        O_acc += pv;

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    // Final normalization
    output[head * head_dim + tid] = O_acc / l_prev;
}

// Flash Attention Batch — B sequences × N heads
__global__ void flash_attention_batch_kernel(
    const float* __restrict__ q,
    const int16_t* __restrict__ kv_cache_k,
    const int16_t* __restrict__ kv_cache_v,
    float* __restrict__ output,
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
    int16_t* kv_tile = (int16_t*)(smem + head_dim * sizeof(float));
    float*   scores  = (float*)(smem + head_dim * 4 + FA_TC * head_dim * 2);
    float*   warp_r  = scores + FA_TC;

    // KV cache offset for this sequence
    const int16_t* k_base = kv_cache_k + (size_t)seq * kv_stride_batch;
    const int16_t* v_base = kv_cache_v + (size_t)seq * kv_stride_batch;

    q_s[tid] = q[seq * n_total + head * head_dim + tid];
    __syncthreads();

    float O_acc = 0.0f, m_prev = -1e30f, l_prev = 0.0f;
    int n_tiles = (seq_len + FA_TC - 1) / FA_TC;

    for (int tile = 0; tile < n_tiles; tile++) {
        int t_start = tile * FA_TC;
        int t_len = min(FA_TC, seq_len - t_start);
        int total = t_len * head_dim;

        // Load K tile
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            kv_tile[i] = k_base[(t_start + row) * kv_stride + kv_head * head_dim + col];
        }
        __syncthreads();

        float s = -1e30f;
        if (tid < t_len) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += q_s[d] * bf16_to_float(kv_tile[tid * head_dim + d]);
            s = dot * scale;
        }
        scores[tid] = s;

        float t_max = s;
        int lane = tid & 63, wid = tid >> 6;
        for (int off = 32; off > 0; off >>= 1)
            t_max = fmaxf(t_max, __shfl_xor(t_max, off, 64));
        if (lane == 0) warp_r[wid] = t_max;
        __syncthreads();
        if (tid == 0) warp_r[0] = fmaxf(warp_r[0], warp_r[1]);
        __syncthreads();
        t_max = warp_r[0];
        float m_new = fmaxf(m_prev, t_max);

        float p = (tid < t_len) ? expf(scores[tid] - m_new) : 0.0f;
        scores[tid] = p;
        float p_sum = p;
        for (int off = 32; off > 0; off >>= 1)
            p_sum += __shfl_xor(p_sum, off, 64);
        if (lane == 0) warp_r[wid] = p_sum;
        __syncthreads();
        if (tid == 0) warp_r[0] += warp_r[1];
        __syncthreads();
        p_sum = warp_r[0];

        float correction = expf(m_prev - m_new);
        O_acc *= correction;
        float l_new = correction * l_prev + p_sum;
        __syncthreads();

        // Load V tile
        for (int i = tid; i < total; i += blockDim.x) {
            int row = i / head_dim, col = i % head_dim;
            kv_tile[i] = v_base[(t_start + row) * kv_stride + kv_head * head_dim + col];
        }
        __syncthreads();

        float pv = 0.0f;
        for (int t = 0; t < t_len; t++)
            pv += scores[t] * bf16_to_float(kv_tile[t * head_dim + tid]);
        O_acc += pv;

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    output[seq * n_total + head * head_dim + tid] = O_acc / l_prev;
}

extern "C" {
void launch_flash_attention(const float* q, const int16_t* k_cache, const int16_t* v_cache,
    float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq,
    float scale, hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = head_dim;  // 128 for Mistral
    int smem = head_dim * 4 + FA_TC * head_dim * 2 + FA_TC * 4 + 16;
    flash_attention_kernel<<<n_head, threads, smem, stream>>>(
        q, k_cache, v_cache, output, seq_len, n_head_kv, head_dim, gqa_ratio, scale);
}

void launch_flash_attention_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v,
    float* output, const int* positions,
    int n_head, int n_head_kv, int head_dim,
    int max_seq, float scale, int batch_size, int kv_stride,
    hipStream_t stream)
{
    int gqa_ratio = n_head / n_head_kv;
    int threads = head_dim;
    int smem = head_dim * 4 + FA_TC * head_dim * 2 + FA_TC * 4 + 16;
    dim3 grid(n_head, batch_size);
    flash_attention_batch_kernel<<<grid, threads, smem, stream>>>(
        q, kv_k, kv_v, output, positions, n_head, n_head_kv, head_dim,
        max_seq, gqa_ratio, scale, kv_stride);
}
}
