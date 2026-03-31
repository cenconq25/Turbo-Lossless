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
    void launch_rms_norm_bf16_batch(const float* x, const float* weight, int16_t* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_add_rms_norm_bf16_batch(float* x, const float* residual, const float* weight, int16_t* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_silu_mul_batch(const float* gate, const float* up, float* out, int n, int batch_size, hipStream_t stream);
    void launch_silu_mul_bf16_batch(const float* gate, const float* up, int16_t* out, int n, int batch_size, hipStream_t stream);
    void launch_add_batch(float* y, const float* x, int n, int batch_size, hipStream_t stream);
    void launch_memcpy_batch(float* dst, const float* src, int n, int batch_size, hipStream_t stream);
    void launch_rope_batch(float* q, float* k, const int* positions, int head_dim, int n_head, int n_head_kv, int q_stride, int k_stride, float theta, int batch_size, hipStream_t stream);
    void launch_store_kv_batch(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v, const int* positions, int kv_dim, int max_seq, int batch_size, hipStream_t stream);
    void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache, float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);
    void launch_attention_all_heads_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v, float* output, const int* positions, int n_head, int n_head_kv, int head_dim, int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_flash_attention(const float* q, const int16_t* k_cache, const int16_t* v_cache, float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);
    void launch_flash_attention_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v, float* output, const int* positions, int n_head, int n_head_kv, int head_dim, int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv, int position, float theta, hipStream_t stream);
    void launch_store_kv(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v, int pos, int n_head_kv, int head_dim, int max_seq, hipStream_t stream);

    // FP32→BF16 conversion
    void launch_fp32_to_bf16(const float* input, int16_t* output, int n, hipStream_t stream);

    // Two-pass BF16: v2 matvec (no escape) + BF16 patch correction
    int launch_fixed12_v2_async(const void* packed, const void* codebook, const void* activations, void* output, int M, int K, void* stream);
    int launch_patches_v2_async(const void* row_offsets, const void* patch_cols, const void* correct_vals, const void* wrong_vals, const void* activations, void* output, int M, void* stream);

    // Fused BF16 batch4 (single-pass, O(1) escape with split offset table)
    int launch_fixed12_batch4_async(const void* packed, const void* codebook, const void* a0, const void* a1, const void* a2, const void* a3, const void* esc_row_base, const void* esc_thread_off, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);

    // Fused BF16 batch8 (single-pass, O(1) escape with split offset table)
    int launch_fixed12_batch8_v2_async(const void* packed, const void* codebook, const void* a0, const void* a1, const void* a2, const void* a3, const void* a4, const void* a5, const void* a6, const void* a7, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, void* o4, void* o5, void* o6, void* o7, int M, int K, void* stream);

    // Structured 12-bit: arithmetic decode (no LDS codebook)
    int launch_structured12_v2_async(const void* packed, int base_exp, const void* activations, void* output, int M, int K, void* stream);
    int launch_structured12_batch4_async(const void* packed, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
    int launch_structured12_batch8_async(const void* packed, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* a4, const void* a5, const void* a6, const void* a7, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, void* o4, void* o5, void* o6, void* o7, int M, int K, void* stream);
}

#define B 4
#define SEQ(buf, s, sz) ((buf) + (s) * (sz))
#define SEQI(buf, s, sz) ((buf) + (s) * (sz))  // int16_t version

// B=4 batch4 matvec helper: structured 12-bit (no LDS codebook)
#define BATCH4_MATVEC(w, bf16_in, out_buf, n_in, n_out, strm) do { \
    launch_structured12_batch4_async((w).packed, (w).base_exp, \
        SEQI(bf16_in,0,n_in), SEQI(bf16_in,1,n_in), \
        SEQI(bf16_in,2,n_in), SEQI(bf16_in,3,n_in), \
        (w).escape_row_base, (w).escape_counts, (w).escape_vals, \
        SEQ(out_buf,0,n_out), SEQ(out_buf,1,n_out), \
        SEQ(out_buf,2,n_out), SEQ(out_buf,3,n_out), \
        (w).M, (w).K, strm); \
} while(0)

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
    s->ffn_down = nullptr;  // unused — w_down writes directly to cur
    hipMalloc(&s->logits,   bs * n_vocab * sizeof(float));
    s->attn_scores_buf = nullptr;  // unused — flash attention uses tiled LDS
    hipMalloc(&s->d_positions, bs * sizeof(int));
    hipMalloc(&s->d_tokens, bs * sizeof(int));
    int max_act = std::max(n, n_ff);
    hipMalloc(&s->bf16_act,  bs * max_act * sizeof(int16_t));
    hipMalloc(&s->bf16_act2, bs * max_act * sizeof(int16_t));
    s->stream = 0;
    hipStreamCreateWithFlags(&s->stream2, hipStreamNonBlocking);
    // Event for safe cross-stream synchronization (lighter than hipStreamSynchronize)
    hipEventCreateWithFlags(&s->sync_event, hipEventDisableTiming);
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

    // Two-pass BF16 matvec: structured 12-bit v2 (no escape) + patch correction
    #define MATVEC_B1(w, bf16_in, fp32_out) do { \
        launch_structured12_v2_async((w).packed, (w).base_exp, bf16_in, fp32_out, (w).M, (w).K, stream); \
        if ((w).num_patches > 0 && (w).row_offsets) \
            launch_patches_v2_async((w).row_offsets, (w).patch_cols, (w).patch_correct, (w).patch_wrong, \
                                    bf16_in, fp32_out, (w).M, stream); \
    } while(0)

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // Fused RMSNorm → BF16 for Q/K/V input (saves 1 launch)
        launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        MATVEC_B1(L.wq, bf16_a, state->q_buf);
        MATVEC_B1(L.wk, bf16_a, state->k_buf);
        MATVEC_B1(L.wv, bf16_a, state->v_buf);

        launch_rope(state->q_buf, state->k_buf, head_dim, cfg.n_head, cfg.n_head_kv, pos, cfg.rope_theta, stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_store_kv(state->k_buf, state->v_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                        pos, cfg.n_head_kv, head_dim, m->max_seq_len, stream);
        // Adaptive: naive attention for short context, flash for long
        if (pos < 1024)
            launch_attention_all_heads(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, cfg.n_head, cfg.n_head_kv, head_dim, pos+1, m->max_seq_len, scale, stream);
        else
            launch_flash_attention(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, cfg.n_head, cfg.n_head_kv, head_dim, pos+1, m->max_seq_len, scale, stream);

        // wo: attn_out → BF16 (can't fuse — attention outputs FP32)
        launch_fp32_to_bf16(state->attn_out, bf16_a, n, stream);
        MATVEC_B1(L.wo, bf16_a, res);

        // Fused add + RMSNorm → BF16 (res += cur, then bf16_a = norm(res))
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        MATVEC_B1(L.w_gate, bf16_a, state->ffn_gate);
        MATVEC_B1(L.w_up, bf16_a, state->ffn_up);

        // Fused SiLU*mul → BF16 for w_down input (saves 1 launch)
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, cfg.n_ff, 1, stream);
        MATVEC_B1(L.w_down, bf16_b, cur);
        launch_add_batch(cur, res, n, 1, stream);
    }

    // Fused RMSNorm → BF16 for output projection (saves 1 launch)
    launch_rms_norm_bf16_batch(cur, m->output_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
    MATVEC_B1(m->output_proj, bf16_a, state->logits);
    state->positions[0] = pos + 1;
    #undef MATVEC_B1
}

// B=8 batch8 matvec helper: structured 12-bit (no LDS codebook)
#define BATCH8_MATVEC(w, bf16_in, out_buf, n_in, n_out, strm) do { \
    launch_structured12_batch8_async((w).packed, (w).base_exp, \
        SEQI(bf16_in,0,n_in), SEQI(bf16_in,1,n_in), \
        SEQI(bf16_in,2,n_in), SEQI(bf16_in,3,n_in), \
        SEQI(bf16_in,4,n_in), SEQI(bf16_in,5,n_in), \
        SEQI(bf16_in,6,n_in), SEQI(bf16_in,7,n_in), \
        (w).escape_row_base, (w).escape_counts, (w).escape_vals, \
        SEQ(out_buf,0,n_out), SEQ(out_buf,1,n_out), \
        SEQ(out_buf,2,n_out), SEQ(out_buf,3,n_out), \
        SEQ(out_buf,4,n_out), SEQ(out_buf,5,n_out), \
        SEQ(out_buf,6,n_out), SEQ(out_buf,7,n_out), \
        (w).M, (w).K, strm); \
} while(0)

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

        // Fused RMSNorm → BF16 for Q/K/V (saves 1 launch)
        launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, B, stream);
        // Q on stream0, K+V on stream2 (K,V are 4x smaller — can overlap with Q's tail)
        launch_structured12_batch4_async(L.wq.packed, L.wq.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wq.escape_row_base, L.wq.escape_counts, L.wq.escape_vals,
            SEQ(state->q_buf,0,n), SEQ(state->q_buf,1,n), SEQ(state->q_buf,2,n), SEQ(state->q_buf,3,n),
            L.wq.M, L.wq.K, stream);
        launch_structured12_batch4_async(L.wk.packed, L.wk.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wk.escape_row_base, L.wk.escape_counts, L.wk.escape_vals,
            SEQ(state->k_buf,0,kv_dim), SEQ(state->k_buf,1,kv_dim), SEQ(state->k_buf,2,kv_dim), SEQ(state->k_buf,3,kv_dim),
            L.wk.M, L.wk.K, stream);
        launch_structured12_batch4_async(L.wv.packed, L.wv.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wv.escape_row_base, L.wv.escape_counts, L.wv.escape_vals,
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
        if (max_pos < 1024)
            launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
                max_pos + 1, scale, B, 0, stream);
        else
            launch_flash_attention_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
                max_pos + 1, scale, B, 0, stream);

        // wo: convert attn_out to BF16
        launch_fp32_to_bf16(state->attn_out, bf16_a, B * n, stream);
        launch_structured12_batch4_async(L.wo.packed, L.wo.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wo.escape_row_base, L.wo.escape_counts, L.wo.escape_vals,
            SEQ(res,0,n), SEQ(res,1,n), SEQ(res,2,n), SEQ(res,3,n),
            L.wo.M, L.wo.K, stream);
        // Fused add + RMSNorm → BF16 (res += cur, then bf16_a = norm(res))
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, B, stream);
        launch_structured12_batch4_async(L.w_gate.packed, L.w_gate.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_gate.escape_row_base, L.w_gate.escape_counts, L.w_gate.escape_vals,
            SEQ(state->ffn_gate,0,n_ff), SEQ(state->ffn_gate,1,n_ff), SEQ(state->ffn_gate,2,n_ff), SEQ(state->ffn_gate,3,n_ff),
            L.w_gate.M, L.w_gate.K, stream);
        launch_structured12_batch4_async(L.w_up.packed, L.w_up.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_up.escape_row_base, L.w_up.escape_counts, L.w_up.escape_vals,
            SEQ(state->ffn_up,0,n_ff), SEQ(state->ffn_up,1,n_ff), SEQ(state->ffn_up,2,n_ff), SEQ(state->ffn_up,3,n_ff),
            L.w_up.M, L.w_up.K, stream);
        // Fused SiLU*mul → BF16 for w_down (saves 1 launch)
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, n_ff, B, stream);
        launch_structured12_batch4_async(L.w_down.packed, L.w_down.base_exp,
            SEQI(bf16_b,0,n_ff), SEQI(bf16_b,1,n_ff), SEQI(bf16_b,2,n_ff), SEQI(bf16_b,3,n_ff),
            L.w_down.escape_row_base, L.w_down.escape_counts, L.w_down.escape_vals,
            SEQ(cur,0,n), SEQ(cur,1,n), SEQ(cur,2,n), SEQ(cur,3,n),
            L.w_down.M, L.w_down.K, stream);
        launch_add_batch(cur, res, n, B, stream);
    }

    // Fused RMSNorm → BF16 for output projection (saves 1 launch)
    launch_rms_norm_bf16_batch(cur, m->output_norm, bf16_a, n, cfg.rms_norm_eps, B, stream);
    launch_structured12_batch4_async(m->output_proj.packed, m->output_proj.base_exp,
        SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
        m->output_proj.escape_row_base, m->output_proj.escape_counts, m->output_proj.escape_vals,
        SEQ(state->logits,0,cfg.n_vocab), SEQ(state->logits,1,cfg.n_vocab),
        SEQ(state->logits,2,cfg.n_vocab), SEQ(state->logits,3,cfg.n_vocab),
        m->output_proj.M, m->output_proj.K, stream);
    for (int s = 0; s < B; s++) state->positions[s]++;
}

// B=8 forward — fully batched, minimal kernel launches
static void forward_b8(InferenceState* state, const int token_ids[8]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = 0;
    int n = cfg.n_embd, n_ff = cfg.n_ff, head_dim = n / cfg.n_head;
    int kv_dim = cfg.n_head_kv * head_dim;
    const int BS = 8;

    hipMemcpyAsync(state->d_positions, state->positions, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(state->d_tokens, token_ids, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, BS, stream);

    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;
    int16_t* bf16_b = state->bf16_act2;

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // Fused RMSNorm -> BF16 for Q/K/V
        launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        launch_structured12_batch8_async(L.wq.packed, L.wq.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.wq.escape_row_base, L.wq.escape_counts, L.wq.escape_vals,
            SEQ(state->q_buf,0,n), SEQ(state->q_buf,1,n), SEQ(state->q_buf,2,n), SEQ(state->q_buf,3,n),
            SEQ(state->q_buf,4,n), SEQ(state->q_buf,5,n), SEQ(state->q_buf,6,n), SEQ(state->q_buf,7,n),
            L.wq.M, L.wq.K, stream);
        launch_structured12_batch8_async(L.wk.packed, L.wk.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.wk.escape_row_base, L.wk.escape_counts, L.wk.escape_vals,
            SEQ(state->k_buf,0,kv_dim), SEQ(state->k_buf,1,kv_dim), SEQ(state->k_buf,2,kv_dim), SEQ(state->k_buf,3,kv_dim),
            SEQ(state->k_buf,4,kv_dim), SEQ(state->k_buf,5,kv_dim), SEQ(state->k_buf,6,kv_dim), SEQ(state->k_buf,7,kv_dim),
            L.wk.M, L.wk.K, stream);
        launch_structured12_batch8_async(L.wv.packed, L.wv.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.wv.escape_row_base, L.wv.escape_counts, L.wv.escape_vals,
            SEQ(state->v_buf,0,kv_dim), SEQ(state->v_buf,1,kv_dim), SEQ(state->v_buf,2,kv_dim), SEQ(state->v_buf,3,kv_dim),
            SEQ(state->v_buf,4,kv_dim), SEQ(state->v_buf,5,kv_dim), SEQ(state->v_buf,6,kv_dim), SEQ(state->v_buf,7,kv_dim),
            L.wv.M, L.wv.K, stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, cfg.n_head, cfg.n_head_kv, n, kv_dim, cfg.rope_theta, BS, stream);
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, BS, stream);

        int max_pos = 0;
        for (int s = 0; s < BS; s++) max_pos = std::max(max_pos, state->positions[s]);
        if (max_pos < 1024)
            launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
                max_pos + 1, scale, BS, 0, stream);
        else
            launch_flash_attention_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                state->attn_out, state->d_positions, cfg.n_head, cfg.n_head_kv, head_dim,
                max_pos + 1, scale, BS, 0, stream);

        // wo: convert attn_out to BF16
        launch_fp32_to_bf16(state->attn_out, bf16_a, BS * n, stream);
        launch_structured12_batch8_async(L.wo.packed, L.wo.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.wo.escape_row_base, L.wo.escape_counts, L.wo.escape_vals,
            SEQ(res,0,n), SEQ(res,1,n), SEQ(res,2,n), SEQ(res,3,n),
            SEQ(res,4,n), SEQ(res,5,n), SEQ(res,6,n), SEQ(res,7,n),
            L.wo.M, L.wo.K, stream);
        // Fused add + RMSNorm -> BF16
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        launch_structured12_batch8_async(L.w_gate.packed, L.w_gate.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.w_gate.escape_row_base, L.w_gate.escape_counts, L.w_gate.escape_vals,
            SEQ(state->ffn_gate,0,n_ff), SEQ(state->ffn_gate,1,n_ff), SEQ(state->ffn_gate,2,n_ff), SEQ(state->ffn_gate,3,n_ff),
            SEQ(state->ffn_gate,4,n_ff), SEQ(state->ffn_gate,5,n_ff), SEQ(state->ffn_gate,6,n_ff), SEQ(state->ffn_gate,7,n_ff),
            L.w_gate.M, L.w_gate.K, stream);
        launch_structured12_batch8_async(L.w_up.packed, L.w_up.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
            L.w_up.escape_row_base, L.w_up.escape_counts, L.w_up.escape_vals,
            SEQ(state->ffn_up,0,n_ff), SEQ(state->ffn_up,1,n_ff), SEQ(state->ffn_up,2,n_ff), SEQ(state->ffn_up,3,n_ff),
            SEQ(state->ffn_up,4,n_ff), SEQ(state->ffn_up,5,n_ff), SEQ(state->ffn_up,6,n_ff), SEQ(state->ffn_up,7,n_ff),
            L.w_up.M, L.w_up.K, stream);
        // Fused SiLU*mul -> BF16 for w_down
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, n_ff, BS, stream);
        launch_structured12_batch8_async(L.w_down.packed, L.w_down.base_exp,
            SEQI(bf16_b,0,n_ff), SEQI(bf16_b,1,n_ff), SEQI(bf16_b,2,n_ff), SEQI(bf16_b,3,n_ff),
            SEQI(bf16_b,4,n_ff), SEQI(bf16_b,5,n_ff), SEQI(bf16_b,6,n_ff), SEQI(bf16_b,7,n_ff),
            L.w_down.escape_row_base, L.w_down.escape_counts, L.w_down.escape_vals,
            SEQ(cur,0,n), SEQ(cur,1,n), SEQ(cur,2,n), SEQ(cur,3,n),
            SEQ(cur,4,n), SEQ(cur,5,n), SEQ(cur,6,n), SEQ(cur,7,n),
            L.w_down.M, L.w_down.K, stream);
        launch_add_batch(cur, res, n, BS, stream);
    }

    // Fused RMSNorm -> BF16 for output projection
    launch_rms_norm_bf16_batch(cur, m->output_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
    launch_structured12_batch8_async(m->output_proj.packed, m->output_proj.base_exp,
        SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
        SEQI(bf16_a,4,n), SEQI(bf16_a,5,n), SEQI(bf16_a,6,n), SEQI(bf16_a,7,n),
        m->output_proj.escape_row_base, m->output_proj.escape_counts, m->output_proj.escape_vals,
        SEQ(state->logits,0,cfg.n_vocab), SEQ(state->logits,1,cfg.n_vocab),
        SEQ(state->logits,2,cfg.n_vocab), SEQ(state->logits,3,cfg.n_vocab),
        SEQ(state->logits,4,cfg.n_vocab), SEQ(state->logits,5,cfg.n_vocab),
        SEQ(state->logits,6,cfg.n_vocab), SEQ(state->logits,7,cfg.n_vocab),
        m->output_proj.M, m->output_proj.K, stream);
    for (int s = 0; s < BS; s++) state->positions[s]++;
}

void forward(InferenceState* state, const int* token_ids) {
    if (state->batch_size == 8) forward_b8(state, token_ids);
    else if (state->batch_size == 4) forward_b4(state, token_ids);
    else if (state->batch_size > 8 && state->batch_size % 8 == 0) {
        // Process as multiple B=8 passes (e.g., B=16 = 2× B=8, B=32 = 4× B=8)
        int orig_bs = state->batch_size;
        int n = state->model->config.n_embd;
        int n_ff = state->model->config.n_ff;
        int n_vocab = state->model->config.n_vocab;
        int kv_dim = state->model->config.n_head_kv * (n / state->model->config.n_head);

        for (int chunk = 0; chunk < orig_bs; chunk += 8) {
            // Temporarily adjust state to point to this chunk's slice
            InferenceState slice = *state;
            slice.batch_size = 8;
            slice.positions = state->positions + chunk;
            slice.hidden = state->hidden + chunk * n;
            slice.hidden2 = state->hidden2 + chunk * n;
            slice.attn_out = state->attn_out + chunk * n;
            slice.q_buf = state->q_buf + chunk * n;
            slice.k_buf = state->k_buf + chunk * kv_dim;
            slice.v_buf = state->v_buf + chunk * kv_dim;
            slice.ffn_gate = state->ffn_gate + chunk * n_ff;
            slice.ffn_up = state->ffn_up + chunk * n_ff;
            slice.logits = state->logits + chunk * n_vocab;
            slice.d_positions = state->d_positions + chunk;
            slice.d_tokens = state->d_tokens + chunk;
            slice.bf16_act = state->bf16_act + chunk * std::max(n, n_ff);
            slice.bf16_act2 = state->bf16_act2 + chunk * std::max(n, n_ff);

            forward_b8(&slice, token_ids + chunk);
        }
    } else {
        forward_b1(state, token_ids);
    }
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
        int bs = state->batch_size;
        std::vector<int> tokens(bs);
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            for (int s = 0; s < bs; s++) state->positions[s] = i;
            for (int s = 0; s < bs; s++) tokens[s] = prompt_tokens[i];
            forward(state, tokens.data());
        }
        for (int t = 0; t < max_tokens; t++) {
            for (int s = 0; s < bs; s++)
                tokens[s] = sample_greedy(state->logits + s * n_vocab, n_vocab, state->stream);
            if (tokens[0] == 2) break;
            output.push_back(tokens[0]);
            forward(state, tokens.data());
        }
    }
    return output;
}
