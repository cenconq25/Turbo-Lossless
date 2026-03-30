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
// RoPE: Rotary Position Embeddings
// ============================================================
__global__ void rope_kernel(
    float* __restrict__ q,   // [n_head, head_dim]
    float* __restrict__ k,   // [n_head_kv, head_dim]
    int head_dim, int n_head, int n_head_kv,
    int position, float theta_base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;

    // Apply to Q heads
    if (idx < n_head * half_dim) {
        int head = idx / half_dim;
        int d = idx % half_dim;
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / head_dim);
        float angle = position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

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
        float freq = 1.0f / powf(theta_base, (float)(2 * d) / head_dim);
        float angle = position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

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
// Softmax over attention scores
// ============================================================
__global__ void softmax_kernel(
    float* __restrict__ x,  // [seq_len] — one row of attention scores
    int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];

    // Exp and sum
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];

    // Normalize
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] /= sum;
    }
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

void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv,
                 int position, float theta, hipStream_t stream) {
    int half_dim = head_dim / 2;
    int max_threads = max(n_head, n_head_kv) * half_dim;
    int threads = 256;
    int blocks = (max_threads + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(q, k, head_dim, n_head, n_head_kv, position, theta);
}

void launch_silu_mul(const float* gate, const float* up, float* out,
                     int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void launch_embed_lookup(const int16_t* embd_table, const int* token_ids,
                         float* output, int n_embd, int batch_size, hipStream_t stream) {
    dim3 threads(256);
    dim3 blocks((n_embd + 255) / 256, batch_size);
    embed_lookup_kernel<<<blocks, threads, 0, stream>>>(embd_table, token_ids, output, n_embd, batch_size);
}

void launch_add(float* y, const float* x, int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(y, x, n);
}

void launch_softmax(float* x, int n, hipStream_t stream) {
    int threads = 256;
    softmax_kernel<<<1, threads, threads * sizeof(float), stream>>>(x, n);
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

// Single-head attention: compute scores for one query head against cached keys
// With GQA: multiple Q heads share one KV head
__global__ void attention_scores_kernel(
    const float* __restrict__ q,           // [head_dim] for one Q head
    const int16_t* __restrict__ k_cache,   // [max_seq, n_head_kv, head_dim]
    float* __restrict__ scores,            // [seq_len] output attention scores
    int kv_head_idx, int head_dim, int n_head_kv,
    int seq_len, int max_seq, float scale)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= seq_len) return;

    // Dot product: q · k[pos]
    float dot = 0.0f;
    int base = pos * n_head_kv * head_dim + kv_head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        dot += q[d] * bf16_to_float(k_cache[base + d]);
    }
    scores[pos] = dot * scale;
}

// Weighted sum of V vectors using attention weights
__global__ void attention_value_kernel(
    const float* __restrict__ weights,     // [seq_len] softmax attention weights
    const int16_t* __restrict__ v_cache,   // [max_seq, n_head_kv, head_dim]
    float* __restrict__ output,            // [head_dim] output
    int kv_head_idx, int head_dim, int n_head_kv,
    int seq_len, int max_seq)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float sum = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        int base = pos * n_head_kv * head_dim + kv_head_idx * head_dim;
        sum += weights[pos] * bf16_to_float(v_cache[base + d]);
    }
    output[d] = sum;
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

void launch_attn_scores(const float* q, const int16_t* k_cache, float* scores,
                        int kv_head, int head_dim, int n_head_kv,
                        int seq_len, int max_seq, float scale,
                        hipStream_t stream) {
    int threads = 256;
    int blocks = (seq_len + threads - 1) / threads;
    attention_scores_kernel<<<blocks, threads, 0, stream>>>(
        q, k_cache, scores, kv_head, head_dim, n_head_kv, seq_len, max_seq, scale);
}

void launch_attn_value(const float* weights, const int16_t* v_cache, float* output,
                       int kv_head, int head_dim, int n_head_kv,
                       int seq_len, int max_seq, hipStream_t stream) {
    int threads = min(head_dim, 256);
    int blocks = (head_dim + threads - 1) / threads;
    attention_value_kernel<<<blocks, threads, 0, stream>>>(
        weights, v_cache, output, kv_head, head_dim, n_head_kv, seq_len, max_seq);
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
