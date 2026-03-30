#include "inference.h"
#include "sampler.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <hip/hip_runtime.h>

extern "C" {
    // Single-item kernels
    void launch_rms_norm(const float* x, const float* weight, float* y, int n, float eps, hipStream_t stream);
    void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv, int position, float theta, hipStream_t stream);
    void launch_embed_lookup(const int16_t* embd_table, const int* token_ids, float* output, int n_embd, int batch_size, hipStream_t stream);
    void launch_fp32_to_bf16(const float* input, int16_t* output, int n, hipStream_t stream);
    void launch_store_kv(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v,
        int pos, int n_head_kv, int head_dim, int max_seq, hipStream_t stream);
    void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache,
        float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);

    // Batched kernels (process batch_size items in one launch)
    void launch_rms_norm_batch(const float* x, const float* weight, float* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_silu_mul_batch(const float* gate, const float* up, float* out, int n, int batch_size, hipStream_t stream);
    void launch_add_batch(float* y, const float* x, int n, int batch_size, hipStream_t stream);
    void launch_memcpy_batch(float* dst, const float* src, int n, int batch_size, hipStream_t stream);
    void launch_attention_all_heads_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v,
        float* output, const int* positions, int n_head, int n_head_kv, int head_dim,
        int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_rope_batch(float* q, float* k, const int* positions,
        int head_dim, int n_head, int n_head_kv, int q_stride, int k_stride,
        float theta, int batch_size, hipStream_t stream);
    void launch_store_kv_batch(const float* k, const float* v,
        int16_t* kv_k, int16_t* kv_v, const int* positions,
        int kv_dim, int max_seq, int batch_size, hipStream_t stream);

    // FP32 direct input kernels (no BF16 conversion needed)
    int launch_fixed12_v2_f32(const void* packed, const void* codebook,
        const void* activations, void* output, int M, int K, void* stream);
    int launch_fixed12_batch4_f32(const void* packed, const void* codebook,
        const void* a0, const void* a1, const void* a2, const void* a3,
        void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
    // Patch correction (still needs BF16 activations for now — TODO: make FP32)
    int launch_patches_v2_async(const void* row_offsets, const void* patch_cols,
        const void* correct_vals, const void* wrong_vals,
        const void* activations, void* output, int M, void* stream);
}

static int16_t* s_bf16_bufs[4] = {};
static int s_bf16_buf_size = 0;
static void ensure_bf16_bufs(int K, int batch_size) {
    if (K > s_bf16_buf_size) {
        for (int i = 0; i < 4; i++) {
            if (s_bf16_bufs[i]) hipFree(s_bf16_bufs[i]);
            hipMalloc(&s_bf16_bufs[i], K * sizeof(int16_t));
        }
        s_bf16_buf_size = K;
    }
}

#define B 4
#define SEQ(buf, s, sz) ((buf) + (s) * (sz))

// Skip patches for tensors with <0.01% escape rate (< 1 in 10000)
// Error is undetectable in practice but saves ~900 kernel launches at B=4
#define PATCH_THRESHOLD 0  // 0 = true lossless (correct ALL escapes)

// B=1 FP32 direct matvec
static void matvec_b1(const CompressedWeight& w, const float* x, float* out, hipStream_t stream) {
    launch_fixed12_v2_f32(w.packed, w.codebook, x, out, w.M, w.K, stream);
    if (w.num_patches > PATCH_THRESHOLD && w.row_offsets) {
        // Patch kernel still needs BF16 activations — convert just for patches
        ensure_bf16_bufs(w.K, 1);
        launch_fp32_to_bf16(x, s_bf16_bufs[0], w.K, stream);
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong,
                                s_bf16_bufs[0], out, w.M, stream);
    }
}

// B=4 FP32 direct batch matvec (no BF16 conversion for main kernel!)
static void matvec_b4(const CompressedWeight& w,
    float* x, float* out, int n_in, int n_out, hipStream_t stream) {
    launch_fixed12_batch4_f32(w.packed, w.codebook,
        SEQ(x,0,n_in), SEQ(x,1,n_in), SEQ(x,2,n_in), SEQ(x,3,n_in),
        SEQ(out,0,n_out), SEQ(out,1,n_out), SEQ(out,2,n_out), SEQ(out,3,n_out),
        w.M, w.K, stream);
    if (w.num_patches > PATCH_THRESHOLD && w.row_offsets) {
        ensure_bf16_bufs(w.K, 4);
        for (int s = 0; s < B; s++) {
            launch_fp32_to_bf16(SEQ(x,s,n_in), s_bf16_bufs[s], w.K, stream);
            launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong,
                                    s_bf16_bufs[s], SEQ(out,s,n_out), w.M, stream);
        }
    }
}

InferenceState* create_inference_state(Model* model, int batch_size, int max_seq_len) {
    auto* s = new InferenceState();
    s->model = model;
    s->batch_size = batch_size;
    s->positions = new int[batch_size]();
    int n = model->config.n_embd;
    int n_ff = model->config.n_ff;
    int n_vocab = model->config.n_vocab;
    int kv_dim = model->config.n_head_kv * (n / model->config.n_head);
    int bs = batch_size;

    hipMalloc(&s->hidden,   bs * n * sizeof(float));
    hipMalloc(&s->hidden2,  bs * n * sizeof(float));
    hipMalloc(&s->attn_out, bs * n * sizeof(float));
    hipMalloc(&s->q_buf,    bs * n * sizeof(float));
    hipMalloc(&s->k_buf,    bs * kv_dim * sizeof(float));
    hipMalloc(&s->v_buf,    bs * kv_dim * sizeof(float));
    hipMalloc(&s->ffn_gate, bs * n_ff * sizeof(float));
    hipMalloc(&s->ffn_up,   bs * n_ff * sizeof(float));
    hipMalloc(&s->ffn_down, bs * n * sizeof(float));
    hipMalloc(&s->logits,   bs * n_vocab * sizeof(float));
    hipMalloc(&s->attn_scores_buf, max_seq_len * sizeof(float));
    // Pre-allocate temp buffers for B=4 forward
    hipMalloc(&s->d_positions, bs * sizeof(int));
    hipMalloc(&s->d_tokens, bs * sizeof(int));
    s->stream = 0;
    return s;
}

void free_inference_state(InferenceState* state) {
    if (!state) return;
    delete[] state->positions;
    hipFree(state->hidden); hipFree(state->hidden2); hipFree(state->attn_out);
    hipFree(state->q_buf); hipFree(state->k_buf); hipFree(state->v_buf);
    hipFree(state->ffn_gate); hipFree(state->ffn_up); hipFree(state->ffn_down);
    hipFree(state->logits); hipFree(state->attn_scores_buf);
    delete state;
}

// B=1 forward (single sequence)
static void forward_b1(InferenceState* state, const int* token_ids) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int n = cfg.n_embd;
    int head_dim = n / cfg.n_head;
    int pos = state->positions[0];

    hipMemcpyAsync(state->logits, token_ids, sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, (int*)state->logits, state->hidden, n, 1, stream);

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];
        hipMemcpyAsync(state->hidden2, state->hidden, n * sizeof(float), hipMemcpyDeviceToDevice, stream);
        launch_rms_norm(state->hidden, L.attn_norm, state->hidden, n, cfg.rms_norm_eps, stream);
        matvec_b1(L.wq, state->hidden, state->q_buf, stream);
        matvec_b1(L.wk, state->hidden, state->k_buf, stream);
        matvec_b1(L.wv, state->hidden, state->v_buf, stream);
        launch_rope(state->q_buf, state->k_buf, head_dim, cfg.n_head, cfg.n_head_kv, pos, cfg.rope_theta, stream);
        size_t kv_off = (size_t)layer * m->max_seq_len * cfg.n_head_kv * head_dim;
        launch_store_kv(state->k_buf, state->v_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                        pos, cfg.n_head_kv, head_dim, m->max_seq_len, stream);
        float scale = 1.0f / sqrtf((float)head_dim);
        launch_attention_all_heads(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                                   state->attn_out, cfg.n_head, cfg.n_head_kv, head_dim, pos+1, m->max_seq_len, scale, stream);
        matvec_b1(L.wo, state->attn_out, state->ffn_down, stream);
        hipMemcpyAsync(state->attn_out, state->ffn_down, n * sizeof(float), hipMemcpyDeviceToDevice, stream);
        launch_add_batch(state->attn_out, state->hidden2, n, 1, stream);
        hipMemcpyAsync(state->hidden2, state->attn_out, n * sizeof(float), hipMemcpyDeviceToDevice, stream);
        launch_rms_norm(state->attn_out, L.ffn_norm, state->attn_out, n, cfg.rms_norm_eps, stream);
        matvec_b1(L.w_gate, state->attn_out, state->ffn_gate, stream);
        matvec_b1(L.w_up, state->attn_out, state->ffn_up, stream);
        launch_silu_mul_batch(state->ffn_gate, state->ffn_up, state->ffn_gate, cfg.n_ff, 1, stream);
        matvec_b1(L.w_down, state->ffn_gate, state->hidden, stream);
        launch_add_batch(state->hidden, state->hidden2, n, 1, stream);
    }
    launch_rms_norm(state->hidden, m->output_norm, state->hidden, n, cfg.rms_norm_eps, stream);
    matvec_b1(m->output_proj, state->hidden, state->logits, stream);
    hipDeviceSynchronize();
    state->positions[0] = pos + 1;
}

// B=4 fully batched forward — minimal kernel launches per layer
static void forward_b4(InferenceState* state, const int token_ids[4]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int n = cfg.n_embd;
    int n_ff = cfg.n_ff;
    int head_dim = n / cfg.n_head;
    int kv_dim = cfg.n_head_kv * head_dim;

    // Upload positions and tokens to pre-allocated GPU buffers
    hipMemcpyAsync(state->d_positions, state->positions, B * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(state->d_tokens, token_ids, B * sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, B, stream);

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // 1 launch: save residual (B items)
        launch_memcpy_batch(state->hidden2, state->hidden, n, B, stream);

        // 1 launch: RMSNorm (B items)
        launch_rms_norm_batch(state->hidden, L.attn_norm, state->hidden, n, cfg.rms_norm_eps, B, stream);

        // 3 launches: Q,K,V batch matvec
        matvec_b4(L.wq, state->hidden, state->q_buf, n, n, stream);
        matvec_b4(L.wk, state->hidden, state->k_buf, n, kv_dim, stream);
        matvec_b4(L.wv, state->hidden, state->v_buf, n, kv_dim, stream);

        // 1 launch: batched RoPE for all B sequences
        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, cfg.n_head, cfg.n_head_kv, n, kv_dim,
                          cfg.rope_theta, B, stream);

        // 1 launch: batched KV store for all B sequences
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, B, stream);

        // 1 launch: batched attention (B×n_head blocks)
        float scale = 1.0f / sqrtf((float)head_dim);
        launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
            state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
            m->max_seq_len, scale, B, 0, stream);

        // 1 launch: output projection batch
        matvec_b4(L.wo, state->attn_out, state->ffn_down, n, n, stream);

        // 1 launch: copy + residual add
        launch_memcpy_batch(state->attn_out, state->ffn_down, n, B, stream);
        launch_add_batch(state->attn_out, state->hidden2, n, B, stream);

        // 1 launch: save residual + FFN norm
        launch_memcpy_batch(state->hidden2, state->attn_out, n, B, stream);
        launch_rms_norm_batch(state->attn_out, L.ffn_norm, state->attn_out, n, cfg.rms_norm_eps, B, stream);

        // 2 launches: MLP gate + up batch
        matvec_b4(L.w_gate, state->attn_out, state->ffn_gate, n, n_ff, stream);
        matvec_b4(L.w_up, state->attn_out, state->ffn_up, n, n_ff, stream);

        // 1 launch: SiLU batch
        launch_silu_mul_batch(state->ffn_gate, state->ffn_up, state->ffn_gate, n_ff, B, stream);

        // 1 launch: MLP down batch
        matvec_b4(L.w_down, state->ffn_gate, state->hidden, n_ff, n, stream);

        // 1 launch: residual add
        launch_add_batch(state->hidden, state->hidden2, n, B, stream);
    }

    // Final norm + output projection
    launch_rms_norm_batch(state->hidden, m->output_norm, state->hidden, n, cfg.rms_norm_eps, B, stream);
    matvec_b4(m->output_proj, state->hidden, state->logits, n, cfg.n_vocab, stream);

    hipDeviceSynchronize();
    for (int s = 0; s < B; s++) state->positions[s]++;
}

void forward(InferenceState* state, const int* token_ids) {
    if (state->batch_size == 4) forward_b4(state, token_ids);
    else forward_b1(state, token_ids);
}

std::vector<int> generate(InferenceState* state, const std::vector<int>& prompt_tokens, int max_tokens) {
    std::vector<int> output;
    if (state->batch_size == 1) {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            state->positions[0] = i;
            forward(state, &prompt_tokens[i]);
        }
        for (int t = 0; t < max_tokens; t++) {
            int next = sample_greedy(state->logits, state->model->config.n_vocab, state->stream);
            if (next == 2) break;
            output.push_back(next);
            forward(state, &next);
        }
    } else {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            for (int s = 0; s < B; s++) state->positions[s] = i;
            int tokens[B]; for (int s=0;s<B;s++) tokens[s] = prompt_tokens[i];
            forward(state, tokens);
        }
        int n_vocab = state->model->config.n_vocab;
        for (int t = 0; t < max_tokens; t++) {
            int tokens[B];
            for (int s = 0; s < B; s++)
                tokens[s] = sample_greedy(state->logits + s * n_vocab, n_vocab, state->stream);
            if (tokens[0] == 2) break;
            output.push_back(tokens[0]);
            forward(state, tokens);
        }
    }
    return output;
}
