#include "inference.h"
#include "sampler.h"
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

extern "C" {
    void launch_rms_norm(const float* x, const float* weight, float* y, int n, float eps, hipStream_t stream);
    void launch_embed_lookup(const int16_t* embd_table, const int* token_ids, float* output, int n_embd, int batch_size, hipStream_t stream);
    // Batched ops
    void launch_rms_norm_batch(const float* x, const float* weight, float* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_silu_mul_batch(const float* gate, const float* up, float* out, int n, int batch_size, hipStream_t stream);
    void launch_add_batch(float* y, const float* x, int n, int batch_size, hipStream_t stream);
    void launch_memcpy_batch(float* dst, const float* src, int n, int batch_size, hipStream_t stream);
    void launch_rope_batch(float* q, float* k, const int* positions, int head_dim, int n_head, int n_head_kv, int q_stride, int k_stride, float theta, int batch_size, hipStream_t stream);
    void launch_store_kv_batch(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v, const int* positions, int kv_dim, int max_seq, int batch_size, hipStream_t stream);
    void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache, float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);
    void launch_attention_all_heads_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v, float* output, const int* positions, int n_head, int n_head_kv, int head_dim, int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv, int position, float theta, hipStream_t stream);
    void launch_store_kv(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v, int pos, int n_head_kv, int head_dim, int max_seq, hipStream_t stream);

    // FP32→BF16 conversion
    void launch_fp32_to_bf16(const float* input, int16_t* output, int n, hipStream_t stream);

    // Two-pass BF16: v2 matvec (no escape) + BF16 patch correction
    int launch_fixed12_v2_async(const void* packed, const void* codebook, const void* activations, void* output, int M, int K, void* stream);
    int launch_patches_v2_async(const void* row_offsets, const void* patch_cols, const void* correct_vals, const void* wrong_vals, const void* activations, void* output, int M, void* stream);

    // Fused BF16-input kernels (single-pass, O(1) escape)
    int launch_fixed12_fused_async(const void* packed, const void* codebook, const void* activations, const void* escape_offsets, const void* escape_vals, void* output, int M, int K, void* stream);
    int launch_fixed12_batch4_async(const void* packed, const void* codebook, const void* a0, const void* a1, const void* a2, const void* a3, const void* esc_off, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
}

#define B 4
#define SEQ(buf, s, sz) ((buf) + (s) * (sz))
#define SEQI(buf, s, sz) ((buf) + (s) * (sz))  // int16_t version

// B=1: convert FP32→BF16, then use fused BF16 kernel (max bandwidth)
static void matvec_b1(const CompressedWeight& w, const float* x, float* out,
                      int16_t* bf16_buf, hipStream_t stream) {
    launch_fp32_to_bf16(x, bf16_buf, w.K, stream);
    launch_fixed12_fused_async(w.packed, w.codebook, bf16_buf,
        w.escape_offsets, w.escape_vals, out, w.M, w.K, stream);
}

// B=4: convert FP32→BF16, then use fused BF16 batch4 kernel
static void matvec_b4(const CompressedWeight& w, float* x, float* out,
                      int16_t* bf16_buf, int n_in, int n_out, hipStream_t stream) {
    for (int s = 0; s < B; s++)
        launch_fp32_to_bf16(SEQ(x,s,n_in), SEQI(bf16_buf,s,n_in), n_in, stream);
    launch_fixed12_batch4_async(w.packed, w.codebook,
        SEQI(bf16_buf,0,n_in), SEQI(bf16_buf,1,n_in),
        SEQI(bf16_buf,2,n_in), SEQI(bf16_buf,3,n_in),
        w.escape_offsets, w.escape_vals,
        SEQ(out,0,n_out), SEQ(out,1,n_out), SEQ(out,2,n_out), SEQ(out,3,n_out),
        w.M, w.K, stream);
}

InferenceState* create_inference_state(Model* model, int batch_size, int max_seq_len) {
    auto* s = new InferenceState();
    s->model = model;
    s->batch_size = batch_size;
    s->positions = new int[batch_size]();
    int n = model->config.n_embd, n_ff = model->config.n_ff, n_vocab = model->config.n_vocab;
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
    hipMalloc(&s->d_positions, bs * sizeof(int));
    hipMalloc(&s->d_tokens, bs * sizeof(int));
    int max_act = std::max(n, n_ff);
    hipMalloc(&s->bf16_act,  bs * max_act * sizeof(int16_t));
    hipMalloc(&s->bf16_act2, bs * max_act * sizeof(int16_t));
    s->stream = 0;
    init_sampler();
    return s;
}

void free_inference_state(InferenceState* state) {
    if (!state) return;
    delete[] state->positions;
    hipFree(state->hidden); hipFree(state->hidden2); hipFree(state->attn_out);
    hipFree(state->q_buf); hipFree(state->k_buf); hipFree(state->v_buf);
    hipFree(state->ffn_gate); hipFree(state->ffn_up); hipFree(state->ffn_down);
    hipFree(state->logits); hipFree(state->attn_scores_buf);
    hipFree(state->d_positions); hipFree(state->d_tokens);
    hipFree(state->bf16_act); hipFree(state->bf16_act2);
    delete state;
}

// B=1 forward — optimized: no redundant memcpy, single sync at end
static void forward_b1(InferenceState* state, const int* token_ids) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = 0;
    int n = cfg.n_embd, head_dim = n / cfg.n_head;
    int kv_dim = cfg.n_head_kv * head_dim;
    int pos = state->positions[0];

    hipMemcpyAsync(state->d_tokens, token_ids, sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, 1, stream);

    // Ping-pong between hidden and hidden2 to avoid unnecessary copies
    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;   // BF16 activation buffer
    int16_t* bf16_b = state->bf16_act2;  // second BF16 buffer

    // Two-pass BF16 matvec: v2 (no escape) + patch correction — avoids escape_offsets read
    #define MATVEC_B1(w, bf16_in, fp32_out) do { \
        launch_fixed12_v2_async((w).packed, (w).codebook, bf16_in, fp32_out, (w).M, (w).K, stream); \
        if ((w).num_patches > 0 && (w).row_offsets) \
            launch_patches_v2_async((w).row_offsets, (w).patch_cols, (w).patch_correct, (w).patch_wrong, \
                                    bf16_in, fp32_out, (w).M, stream); \
    } while(0)

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        launch_rms_norm(cur, L.attn_norm, state->attn_out, n, cfg.rms_norm_eps, stream);

        // Convert attn_out to BF16 once, reuse for Q/K/V
        launch_fp32_to_bf16(state->attn_out, bf16_a, n, stream);
        MATVEC_B1(L.wq, bf16_a, state->q_buf);
        MATVEC_B1(L.wk, bf16_a, state->k_buf);
        MATVEC_B1(L.wv, bf16_a, state->v_buf);

        launch_rope(state->q_buf, state->k_buf, head_dim, cfg.n_head, cfg.n_head_kv, pos, cfg.rope_theta, stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_store_kv(state->k_buf, state->v_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                        pos, cfg.n_head_kv, head_dim, m->max_seq_len, stream);
        launch_attention_all_heads(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                                   state->attn_out, cfg.n_head, cfg.n_head_kv, head_dim, pos+1, m->max_seq_len, scale, stream);

        // wo: new attn_out → BF16
        launch_fp32_to_bf16(state->attn_out, bf16_a, n, stream);
        MATVEC_B1(L.wo, bf16_a, res);
        launch_add_batch(res, cur, n, 1, stream);

        launch_rms_norm(res, L.ffn_norm, state->attn_out, n, cfg.rms_norm_eps, stream);

        // gate/up: convert once, reuse
        launch_fp32_to_bf16(state->attn_out, bf16_a, n, stream);
        MATVEC_B1(L.w_gate, bf16_a, state->ffn_gate);
        MATVEC_B1(L.w_up, bf16_a, state->ffn_up);
        launch_silu_mul_batch(state->ffn_gate, state->ffn_up, state->ffn_gate, cfg.n_ff, 1, stream);

        // w_down: ffn_gate → BF16
        launch_fp32_to_bf16(state->ffn_gate, bf16_b, cfg.n_ff, stream);
        MATVEC_B1(L.w_down, bf16_b, cur);
        launch_add_batch(cur, res, n, 1, stream);
    }

    launch_rms_norm(cur, m->output_norm, cur, n, cfg.rms_norm_eps, stream);
    launch_fp32_to_bf16(cur, bf16_a, n, stream);
    MATVEC_B1(m->output_proj, bf16_a, state->logits);
    state->positions[0] = pos + 1;
    #undef MATVEC_B1
}

// B=4 forward — fully batched, minimal kernel launches
static void forward_b4(InferenceState* state, const int token_ids[4]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = 0;
    int n = cfg.n_embd, n_ff = cfg.n_ff, head_dim = n / cfg.n_head;
    int kv_dim = cfg.n_head_kv * head_dim;

    hipMemcpyAsync(state->d_positions, state->positions, B * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(state->d_tokens, token_ids, B * sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, B, stream);

    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;
    int16_t* bf16_b = state->bf16_act2;

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        launch_rms_norm_batch(cur, L.attn_norm, state->attn_out, n, cfg.rms_norm_eps, B, stream);

        // Convert attn_out to BF16 once for Q/K/V (B*n elements)
        launch_fp32_to_bf16(state->attn_out, bf16_a, B * n, stream);
        launch_fixed12_batch4_async(L.wq.packed, L.wq.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wq.escape_offsets, L.wq.escape_vals,
            SEQ(state->q_buf,0,n), SEQ(state->q_buf,1,n), SEQ(state->q_buf,2,n), SEQ(state->q_buf,3,n),
            L.wq.M, L.wq.K, stream);
        launch_fixed12_batch4_async(L.wk.packed, L.wk.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wk.escape_offsets, L.wk.escape_vals,
            SEQ(state->k_buf,0,kv_dim), SEQ(state->k_buf,1,kv_dim), SEQ(state->k_buf,2,kv_dim), SEQ(state->k_buf,3,kv_dim),
            L.wk.M, L.wk.K, stream);
        launch_fixed12_batch4_async(L.wv.packed, L.wv.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wv.escape_offsets, L.wv.escape_vals,
            SEQ(state->v_buf,0,kv_dim), SEQ(state->v_buf,1,kv_dim), SEQ(state->v_buf,2,kv_dim), SEQ(state->v_buf,3,kv_dim),
            L.wv.M, L.wv.K, stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, cfg.n_head, cfg.n_head_kv, n, kv_dim, cfg.rope_theta, B, stream);
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, B, stream);

        int max_pos = 0;
        for (int s = 0; s < B; s++) max_pos = std::max(max_pos, state->positions[s]);
        launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
            state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
            max_pos + 1, scale, B, 0, stream);

        // wo: convert attn_out to BF16
        launch_fp32_to_bf16(state->attn_out, bf16_a, B * n, stream);
        launch_fixed12_batch4_async(L.wo.packed, L.wo.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wo.escape_offsets, L.wo.escape_vals,
            SEQ(res,0,n), SEQ(res,1,n), SEQ(res,2,n), SEQ(res,3,n),
            L.wo.M, L.wo.K, stream);
        launch_add_batch(res, cur, n, B, stream);

        launch_rms_norm_batch(res, L.ffn_norm, state->attn_out, n, cfg.rms_norm_eps, B, stream);

        // gate/up: convert attn_out once for both
        launch_fp32_to_bf16(state->attn_out, bf16_a, B * n, stream);
        launch_fixed12_batch4_async(L.w_gate.packed, L.w_gate.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_gate.escape_offsets, L.w_gate.escape_vals,
            SEQ(state->ffn_gate,0,n_ff), SEQ(state->ffn_gate,1,n_ff), SEQ(state->ffn_gate,2,n_ff), SEQ(state->ffn_gate,3,n_ff),
            L.w_gate.M, L.w_gate.K, stream);
        launch_fixed12_batch4_async(L.w_up.packed, L.w_up.codebook,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_up.escape_offsets, L.w_up.escape_vals,
            SEQ(state->ffn_up,0,n_ff), SEQ(state->ffn_up,1,n_ff), SEQ(state->ffn_up,2,n_ff), SEQ(state->ffn_up,3,n_ff),
            L.w_up.M, L.w_up.K, stream);
        launch_silu_mul_batch(state->ffn_gate, state->ffn_up, state->ffn_gate, n_ff, B, stream);

        // w_down: convert ffn_gate to BF16 (size n_ff)
        launch_fp32_to_bf16(state->ffn_gate, bf16_b, B * n_ff, stream);
        launch_fixed12_batch4_async(L.w_down.packed, L.w_down.codebook,
            SEQI(bf16_b,0,n_ff), SEQI(bf16_b,1,n_ff), SEQI(bf16_b,2,n_ff), SEQI(bf16_b,3,n_ff),
            L.w_down.escape_offsets, L.w_down.escape_vals,
            SEQ(cur,0,n), SEQ(cur,1,n), SEQ(cur,2,n), SEQ(cur,3,n),
            L.w_down.M, L.w_down.K, stream);
        launch_add_batch(cur, res, n, B, stream);
    }

    launch_rms_norm_batch(cur, m->output_norm, cur, n, cfg.rms_norm_eps, B, stream);
    launch_fp32_to_bf16(cur, bf16_a, B * n, stream);
    launch_fixed12_batch4_async(m->output_proj.packed, m->output_proj.codebook,
        SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
        m->output_proj.escape_offsets, m->output_proj.escape_vals,
        SEQ(state->logits,0,cfg.n_vocab), SEQ(state->logits,1,cfg.n_vocab),
        SEQ(state->logits,2,cfg.n_vocab), SEQ(state->logits,3,cfg.n_vocab),
        m->output_proj.M, m->output_proj.K, stream);
    for (int s = 0; s < B; s++) state->positions[s]++;
}

void forward(InferenceState* state, const int* token_ids) {
    if (state->batch_size == 4) forward_b4(state, token_ids);
    else forward_b1(state, token_ids);
}

std::vector<int> generate(InferenceState* state, const std::vector<int>& prompt_tokens, int max_tokens) {
    std::vector<int> output;
    int n_vocab = state->model->config.n_vocab;

    if (state->batch_size == 1) {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            state->positions[0] = i;
            forward(state, &prompt_tokens[i]);
        }
        for (int t = 0; t < max_tokens; t++) {
            int next = sample_greedy(state->logits, n_vocab, state->stream);
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
