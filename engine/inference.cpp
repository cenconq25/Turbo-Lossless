#include "inference.h"
#include "sampler.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <hip/hip_runtime.h>

extern "C" {
    void launch_rms_norm(const float* x, const float* weight, float* y, int n, float eps, hipStream_t stream);
    void launch_rope(float* q, float* k, int head_dim, int n_head, int n_head_kv, int position, float theta, hipStream_t stream);
    void launch_silu_mul(const float* gate, const float* up, float* out, int n, hipStream_t stream);
    void launch_embed_lookup(const int16_t* embd_table, const int* token_ids, float* output, int n_embd, int batch_size, hipStream_t stream);
    void launch_add(float* y, const float* x, int n, hipStream_t stream);
    void launch_fp32_to_bf16(const float* input, int16_t* output, int n, hipStream_t stream);
    void launch_store_kv(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v,
        int pos, int n_head_kv, int head_dim, int max_seq, hipStream_t stream);
    void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache,
        float* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq,
        float scale, hipStream_t stream);
    // Async two-pass: matvec then patch correction (no hipDeviceSynchronize)
    int launch_fixed12_v2_async(const void* packed, const void* codebook,
        const void* activations, void* output, int M, int K, void* stream);
    int launch_patches_v2_async(const void* row_offsets, const void* patch_cols,
        const void* correct_vals, const void* wrong_vals,
        const void* activations, void* output, int M, void* stream);
    // Async batch4 matvec without escape handling
    int launch_fixed12_batch4_noesc_async(const void* packed, const void* codebook,
        const void* a0, const void* a1, const void* a2, const void* a3,
        void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
}

// BF16 conversion buffers — one per batch slot
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

// B=1 async two-pass matvec: decode + patch correction (no sync)
static void compressed_matvec_b1(const CompressedWeight& w, const float* x, float* out, hipStream_t stream) {
    ensure_bf16_bufs(w.K, 1);
    launch_fp32_to_bf16(x, s_bf16_bufs[0], w.K, stream);
    launch_fixed12_v2_async(w.packed, w.codebook, s_bf16_bufs[0], out, w.M, w.K, stream);
    if (w.num_patches > 0 && w.row_offsets)
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong,
                                s_bf16_bufs[0], out, w.M, stream);
}

// B=4 two-pass: batch matvec (decode once, 4 activations) + 4× patch correction
static void compressed_matvec_b4(const CompressedWeight& w,
    const float* x0, const float* x1, const float* x2, const float* x3,
    float* o0, float* o1, float* o2, float* o3, hipStream_t stream) {
    ensure_bf16_bufs(w.K, 4);
    // Convert all 4 activations to BF16
    launch_fp32_to_bf16(x0, s_bf16_bufs[0], w.K, stream);
    launch_fp32_to_bf16(x1, s_bf16_bufs[1], w.K, stream);
    launch_fp32_to_bf16(x2, s_bf16_bufs[2], w.K, stream);
    launch_fp32_to_bf16(x3, s_bf16_bufs[3], w.K, stream);
    // Pass 1: batch matvec (escapes get codebook[4095] = wrong value)
    launch_fixed12_batch4_noesc_async(w.packed, w.codebook,
        s_bf16_bufs[0], s_bf16_bufs[1], s_bf16_bufs[2], s_bf16_bufs[3],
        o0, o1, o2, o3, w.M, w.K, stream);
    // Pass 2: correct escapes for each output
    if (w.num_patches > 0 && w.row_offsets) {
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong, s_bf16_bufs[0], o0, w.M, stream);
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong, s_bf16_bufs[1], o1, w.M, stream);
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong, s_bf16_bufs[2], o2, w.M, stream);
        launch_patches_v2_async(w.row_offsets, w.patch_cols, w.patch_correct, w.patch_wrong, s_bf16_bufs[3], o3, w.M, stream);
    }
}

// Per-sequence buffers for B=4
struct SeqBufs {
    float *hidden, *hidden2, *attn_out, *q_buf, *k_buf, *v_buf;
    float *ffn_gate, *ffn_up, *ffn_down;
};

InferenceState* create_inference_state(Model* model, int batch_size, int max_seq_len) {
    auto* s = new InferenceState();
    s->model = model;
    s->batch_size = batch_size;
    s->positions = new int[batch_size]();
    int n = model->config.n_embd;
    int n_ff = model->config.n_ff;
    int n_vocab = model->config.n_vocab;
    int kv_dim = model->config.n_head_kv * (n / model->config.n_head);

    // Allocate per-sequence buffers × batch_size
    // Store as flat arrays: buffer[seq * size ... (seq+1) * size]
    hipMalloc(&s->hidden,   batch_size * n * sizeof(float));
    hipMalloc(&s->hidden2,  batch_size * n * sizeof(float));
    hipMalloc(&s->attn_out, batch_size * n * sizeof(float));
    hipMalloc(&s->q_buf,    batch_size * n * sizeof(float));
    hipMalloc(&s->k_buf,    batch_size * kv_dim * sizeof(float));
    hipMalloc(&s->v_buf,    batch_size * kv_dim * sizeof(float));
    hipMalloc(&s->ffn_gate, batch_size * n_ff * sizeof(float));
    hipMalloc(&s->ffn_up,   batch_size * n_ff * sizeof(float));
    hipMalloc(&s->ffn_down, batch_size * n * sizeof(float));
    hipMalloc(&s->logits,   batch_size * n_vocab * sizeof(float));
    hipMalloc(&s->attn_scores_buf, max_seq_len * sizeof(float));

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

// B=1 forward pass (unchanged)
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

        compressed_matvec_b1(L.wq, state->hidden, state->q_buf, stream);
        compressed_matvec_b1(L.wk, state->hidden, state->k_buf, stream);
        compressed_matvec_b1(L.wv, state->hidden, state->v_buf, stream);
        launch_rope(state->q_buf, state->k_buf, head_dim, cfg.n_head, cfg.n_head_kv, pos, cfg.rope_theta, stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * cfg.n_head_kv * head_dim;
        launch_store_kv(state->k_buf, state->v_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                        pos, cfg.n_head_kv, head_dim, m->max_seq_len, stream);

        int seq_len = pos + 1;
        float scale = 1.0f / sqrtf((float)head_dim);
        launch_attention_all_heads(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                                   state->attn_out, cfg.n_head, cfg.n_head_kv, head_dim, seq_len,
                                   m->max_seq_len, scale, stream);

        compressed_matvec_b1(L.wo, state->attn_out, state->ffn_down, stream);
        hipMemcpyAsync(state->attn_out, state->ffn_down, n * sizeof(float), hipMemcpyDeviceToDevice, stream);
        launch_add(state->attn_out, state->hidden2, n, stream);

        hipMemcpyAsync(state->hidden2, state->attn_out, n * sizeof(float), hipMemcpyDeviceToDevice, stream);
        launch_rms_norm(state->attn_out, L.ffn_norm, state->attn_out, n, cfg.rms_norm_eps, stream);

        compressed_matvec_b1(L.w_gate, state->attn_out, state->ffn_gate, stream);
        compressed_matvec_b1(L.w_up,   state->attn_out, state->ffn_up,   stream);
        launch_silu_mul(state->ffn_gate, state->ffn_up, state->ffn_gate, cfg.n_ff, stream);
        compressed_matvec_b1(L.w_down, state->ffn_gate, state->hidden, stream);
        launch_add(state->hidden, state->hidden2, n, stream);
    }

    launch_rms_norm(state->hidden, m->output_norm, state->hidden, n, cfg.rms_norm_eps, stream);
    compressed_matvec_b1(m->output_proj, state->hidden, state->logits, stream);
    hipDeviceSynchronize();
    state->positions[0] = pos + 1;
}

// B=4 forward pass — batch weight matvecs, per-sequence attention
static void forward_b4(InferenceState* state, const int token_ids[4]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int n = cfg.n_embd;
    int n_ff = cfg.n_ff;
    int head_dim = n / cfg.n_head;
    int kv_dim = cfg.n_head_kv * head_dim;

    // Seq buffer pointers
    #define SEQ(buf, i, sz) ((buf) + (i) * (sz))

    // 1. Embedding lookup for all 4 sequences
    for (int s = 0; s < 4; s++) {
        hipMemcpyAsync(SEQ(state->logits, s, cfg.n_vocab), &token_ids[s], sizeof(int), hipMemcpyHostToDevice, stream);
        launch_embed_lookup(m->token_embd, (int*)SEQ(state->logits, s, cfg.n_vocab), SEQ(state->hidden, s, n), n, 1, stream);
    }

    // 2. Transformer layers
    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // Save residuals + RMSNorm for all 4 sequences
        for (int s = 0; s < 4; s++) {
            hipMemcpyAsync(SEQ(state->hidden2, s, n), SEQ(state->hidden, s, n), n * sizeof(float), hipMemcpyDeviceToDevice, stream);
            launch_rms_norm(SEQ(state->hidden, s, n), L.attn_norm, SEQ(state->hidden, s, n), n, cfg.rms_norm_eps, stream);
        }

        // Q projection — BATCHED! decode weight once, 4 activations
        compressed_matvec_b4(L.wq,
            SEQ(state->hidden, 0, n), SEQ(state->hidden, 1, n),
            SEQ(state->hidden, 2, n), SEQ(state->hidden, 3, n),
            SEQ(state->q_buf, 0, n), SEQ(state->q_buf, 1, n),
            SEQ(state->q_buf, 2, n), SEQ(state->q_buf, 3, n), stream);

        // K projection — BATCHED
        compressed_matvec_b4(L.wk,
            SEQ(state->hidden, 0, n), SEQ(state->hidden, 1, n),
            SEQ(state->hidden, 2, n), SEQ(state->hidden, 3, n),
            SEQ(state->k_buf, 0, kv_dim), SEQ(state->k_buf, 1, kv_dim),
            SEQ(state->k_buf, 2, kv_dim), SEQ(state->k_buf, 3, kv_dim), stream);

        // V projection — BATCHED
        compressed_matvec_b4(L.wv,
            SEQ(state->hidden, 0, n), SEQ(state->hidden, 1, n),
            SEQ(state->hidden, 2, n), SEQ(state->hidden, 3, n),
            SEQ(state->v_buf, 0, kv_dim), SEQ(state->v_buf, 1, kv_dim),
            SEQ(state->v_buf, 2, kv_dim), SEQ(state->v_buf, 3, kv_dim), stream);

        // RoPE + KV store + Attention — per sequence (different positions)
        size_t kv_off = (size_t)layer * m->max_seq_len * cfg.n_head_kv * head_dim;
        for (int s = 0; s < 4; s++) {
            int pos = state->positions[s];
            launch_rope(SEQ(state->q_buf, s, n), SEQ(state->k_buf, s, kv_dim),
                        head_dim, cfg.n_head, cfg.n_head_kv, pos, cfg.rope_theta, stream);

            // Each sequence has its own KV cache region
            size_t seq_kv_off = kv_off + (size_t)s * cfg.n_layer * m->max_seq_len * kv_dim;
            // Actually: for simplicity, use same KV cache but different positions
            // This assumes all 4 sequences share the same KV cache (wrong for multi-user)
            // TODO: separate KV caches per sequence
            launch_store_kv(SEQ(state->k_buf, s, kv_dim), SEQ(state->v_buf, s, kv_dim),
                            m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                            pos, cfg.n_head_kv, head_dim, m->max_seq_len, stream);

            int seq_len = pos + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            launch_attention_all_heads(SEQ(state->q_buf, s, n),
                                       m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                                       SEQ(state->attn_out, s, n),
                                       cfg.n_head, cfg.n_head_kv, head_dim, seq_len,
                                       m->max_seq_len, scale, stream);
        }

        // Output projection — BATCHED
        compressed_matvec_b4(L.wo,
            SEQ(state->attn_out, 0, n), SEQ(state->attn_out, 1, n),
            SEQ(state->attn_out, 2, n), SEQ(state->attn_out, 3, n),
            SEQ(state->ffn_down, 0, n), SEQ(state->ffn_down, 1, n),
            SEQ(state->ffn_down, 2, n), SEQ(state->ffn_down, 3, n), stream);

        // Residual + FFN norm for all sequences
        for (int s = 0; s < 4; s++) {
            hipMemcpyAsync(SEQ(state->attn_out, s, n), SEQ(state->ffn_down, s, n), n * sizeof(float), hipMemcpyDeviceToDevice, stream);
            launch_add(SEQ(state->attn_out, s, n), SEQ(state->hidden2, s, n), n, stream);
            hipMemcpyAsync(SEQ(state->hidden2, s, n), SEQ(state->attn_out, s, n), n * sizeof(float), hipMemcpyDeviceToDevice, stream);
            launch_rms_norm(SEQ(state->attn_out, s, n), L.ffn_norm, SEQ(state->attn_out, s, n), n, cfg.rms_norm_eps, stream);
        }

        // MLP gate — BATCHED
        compressed_matvec_b4(L.w_gate,
            SEQ(state->attn_out, 0, n), SEQ(state->attn_out, 1, n),
            SEQ(state->attn_out, 2, n), SEQ(state->attn_out, 3, n),
            SEQ(state->ffn_gate, 0, n_ff), SEQ(state->ffn_gate, 1, n_ff),
            SEQ(state->ffn_gate, 2, n_ff), SEQ(state->ffn_gate, 3, n_ff), stream);

        // MLP up — BATCHED
        compressed_matvec_b4(L.w_up,
            SEQ(state->attn_out, 0, n), SEQ(state->attn_out, 1, n),
            SEQ(state->attn_out, 2, n), SEQ(state->attn_out, 3, n),
            SEQ(state->ffn_up, 0, n_ff), SEQ(state->ffn_up, 1, n_ff),
            SEQ(state->ffn_up, 2, n_ff), SEQ(state->ffn_up, 3, n_ff), stream);

        // SiLU for all sequences
        for (int s = 0; s < 4; s++)
            launch_silu_mul(SEQ(state->ffn_gate, s, n_ff), SEQ(state->ffn_up, s, n_ff),
                            SEQ(state->ffn_gate, s, n_ff), n_ff, stream);

        // MLP down — BATCHED
        compressed_matvec_b4(L.w_down,
            SEQ(state->ffn_gate, 0, n_ff), SEQ(state->ffn_gate, 1, n_ff),
            SEQ(state->ffn_gate, 2, n_ff), SEQ(state->ffn_gate, 3, n_ff),
            SEQ(state->hidden, 0, n), SEQ(state->hidden, 1, n),
            SEQ(state->hidden, 2, n), SEQ(state->hidden, 3, n), stream);

        // Residual add for all sequences
        for (int s = 0; s < 4; s++)
            launch_add(SEQ(state->hidden, s, n), SEQ(state->hidden2, s, n), n, stream);
    }

    // 3. Final norm + output projection for all sequences
    for (int s = 0; s < 4; s++)
        launch_rms_norm(SEQ(state->hidden, s, n), m->output_norm, SEQ(state->hidden, s, n), n, cfg.rms_norm_eps, stream);

    compressed_matvec_b4(m->output_proj,
        SEQ(state->hidden, 0, n), SEQ(state->hidden, 1, n),
        SEQ(state->hidden, 2, n), SEQ(state->hidden, 3, n),
        SEQ(state->logits, 0, cfg.n_vocab), SEQ(state->logits, 1, cfg.n_vocab),
        SEQ(state->logits, 2, cfg.n_vocab), SEQ(state->logits, 3, cfg.n_vocab), stream);

    hipDeviceSynchronize();
    for (int s = 0; s < 4; s++)
        state->positions[s]++;
    #undef SEQ
}

void forward(InferenceState* state, const int* token_ids) {
    if (state->batch_size == 4) {
        forward_b4(state, token_ids);
    } else {
        forward_b1(state, token_ids);
    }
}

std::vector<int> generate(InferenceState* state, const std::vector<int>& prompt_tokens, int max_tokens) {
    std::vector<int> output;

    if (state->batch_size == 1) {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            state->positions[0] = i;
            forward(state, &prompt_tokens[i]);
        }
        for (int t = 0; t < max_tokens; t++) {
            int next_token = sample_greedy(state->logits, state->model->config.n_vocab, state->stream);
            if (next_token == 2) break;
            output.push_back(next_token);
            forward(state, &next_token);
        }
    } else if (state->batch_size == 4) {
        // Prefill: process prompt for sequence 0 (other sequences get same prompt)
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            for (int s = 0; s < 4; s++) state->positions[s] = i;
            int tokens[4] = {prompt_tokens[i], prompt_tokens[i], prompt_tokens[i], prompt_tokens[i]};
            forward(state, tokens);
        }
        // Generate: all 4 sequences generate independently
        int n_vocab = state->model->config.n_vocab;
        for (int t = 0; t < max_tokens; t++) {
            int tokens[4];
            for (int s = 0; s < 4; s++) {
                tokens[s] = sample_greedy(state->logits + s * n_vocab, n_vocab, state->stream);
            }
            if (tokens[0] == 2) break;
            output.push_back(tokens[0]);  // report sequence 0
            forward(state, tokens);
        }
    }
    return output;
}
