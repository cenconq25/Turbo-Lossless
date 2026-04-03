#include "inference.h"
#include "sampler.h"
#include "multi_gpu.h"
#include <cstdio>
#include <cmath>
#include "../gpu_compat.h"

extern "C" {
    // Kernel launch wrappers (kernels.hip)
    void launch_embed_lookup(const int16_t* embd_table, const int* token_ids, float* output, int n_embd, int batch_size, int n_vocab, hipStream_t stream);
    void launch_rms_norm_bf16_batch(const float* x, const float* weight, int16_t* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_add_rms_norm_bf16_batch(float* x, const float* residual, const float* weight, int16_t* y, int n, float eps, int batch_size, hipStream_t stream);
    void launch_silu_mul_bf16_batch(const float* gate, const float* up, int16_t* out, int n, int batch_size, hipStream_t stream);
    void launch_add_batch(float* y, const float* x, int n, int batch_size, hipStream_t stream);
    void launch_rope_batch(float* q, float* k, const int* positions, int head_dim, int n_head, int n_head_kv, int q_stride, int k_stride, float theta, int batch_size, hipStream_t stream);
    void launch_store_kv_batch(const float* k, const float* v, int16_t* kv_k, int16_t* kv_v, const int* positions, int kv_dim, int max_seq, int batch_size, hipStream_t stream);
    void launch_attention_all_heads(const float* q, const int16_t* k_cache, const int16_t* v_cache, int16_t* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);
    void launch_attention_all_heads_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v, int16_t* output, const int* positions, int n_head, int n_head_kv, int head_dim, int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_flash_attention(const float* q, const int16_t* k_cache, const int16_t* v_cache, int16_t* output, int n_head, int n_head_kv, int head_dim, int seq_len, int max_seq, float scale, hipStream_t stream);
    void launch_flash_attention_batch(const float* q, const int16_t* kv_k, const int16_t* kv_v, int16_t* output, const int* positions, int n_head, int n_head_kv, int head_dim, int max_seq, float scale, int batch_size, int kv_stride, hipStream_t stream);
    void launch_rope_store_kv(float* q, float* k, const float* v, int16_t* kv_k, int16_t* kv_v, int head_dim, int n_head, int n_head_kv, int position, int max_seq, float theta, hipStream_t stream);

    // Matvec launch wrappers (decompress_v2.hip)
    int launch_patches_v2_async(const void* row_offsets, const void* patch_cols, const void* correct_vals, const void* wrong_vals, const void* activations, void* output, int M, void* stream);
    int launch_structured12_v2_async(const void* packed, int base_exp, const void* activations, void* output, int M, int K, void* stream, const void* patch_row_offsets, const void* patch_cols, const void* patch_correct_vals, const void* patch_wrong_vals);
    int launch_structured12_v2_dual_async(const void* packed_a, int base_exp_a, const void* packed_b, int base_exp_b, const void* activations, void* output_a, void* output_b, int M, int K, void* stream);
    int launch_structured12_batch4_async(const void* packed, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
    int launch_structured12_batch8_async(const void* packed, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* a4, const void* a5, const void* a6, const void* a7, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, void* o4, void* o5, void* o6, void* o7, int M, int K, void* stream);
    int launch_split12_v2_async(const void* sign_mantissa, const void* groups, int base_exp, const void* activations, void* output, int M, int K, void* stream);
    int launch_split12_v2_dual_async(const void* sm_a, const void* gr_a, int base_exp_a, const void* sm_b, const void* gr_b, int base_exp_b, const void* activations, void* output_a, void* output_b, int M, int K, void* stream);
    int launch_split12_batch4_async(const void* sm, const void* gr, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, int M, int K, void* stream);
    int launch_split12_batch8_async(const void* sm, const void* gr, int base_exp, const void* a0, const void* a1, const void* a2, const void* a3, const void* a4, const void* a5, const void* a6, const void* a7, const void* esc_row_base, const void* esc_counts, const void* esc_vals, void* o0, void* o1, void* o2, void* o3, void* o4, void* o5, void* o6, void* o7, int M, int K, void* stream);

#ifdef TURBO_NVIDIA
    // NVIDIA kernels (nvidia_kernels.cu)
    int nv_launch_split12_v2_async(const void* sm, const void* gr, int base_exp, const void* act, void* out, int M, int K, void* stream);
    int nv_launch_patches_async(const void* row_off, const void* cols, const void* correct, const void* wrong, const void* act, void* out, int M, void* stream);
    int nv_launch_split12_fused_gemm_async(const void* sm, const void* gr, int base_exp, const void* act, int act_stride, void* out, int out_stride, int M, int K, int B, void* stream, const void* patch_row_off, const void* patch_cols, const void* patch_correct, const void* patch_wrong);
    int nv_launch_patches_batch_async(const void* row_off, const void* cols, const void* correct, const void* wrong, const void* nonempty_rows, int num_nonempty, const void* act, int act_stride, void* out, int out_stride, int B, void* stream);
    int nv_launch_split12_cublas_batch_async(const void* sm, const void* gr, int base_exp, const void* act, int act_stride, void* out, int out_stride, void* bf16_weight_buf, int buf_half_elems, int M, int K, int B, void* stream);
#endif
}

#define SEQ(buf, s, sz) ((buf) + (s) * (sz))
#define SEQI(buf, s, sz) ((buf) + (s) * (sz))  // int16_t version

// B=4 batch4 matvec: prefer split12 (zero amplification) when available
#define BATCH4_MATVEC(w, bf16_in, out_buf, n_in, n_out, strm) do { \
    if ((w).split_sm) { \
        launch_split12_batch4_async((w).split_sm, (w).split_gr, (w).base_exp, \
            SEQI(bf16_in,0,n_in), SEQI(bf16_in,1,n_in), \
            SEQI(bf16_in,2,n_in), SEQI(bf16_in,3,n_in), \
            (w).escape_row_base, (w).escape_counts, (w).escape_vals, \
            SEQ(out_buf,0,n_out), SEQ(out_buf,1,n_out), \
            SEQ(out_buf,2,n_out), SEQ(out_buf,3,n_out), \
            (w).M, (w).K, strm); \
    } else { \
        launch_structured12_batch4_async((w).packed, (w).base_exp, \
            SEQI(bf16_in,0,n_in), SEQI(bf16_in,1,n_in), \
            SEQI(bf16_in,2,n_in), SEQI(bf16_in,3,n_in), \
            (w).escape_row_base, (w).escape_counts, (w).escape_vals, \
            SEQ(out_buf,0,n_out), SEQ(out_buf,1,n_out), \
            SEQ(out_buf,2,n_out), SEQ(out_buf,3,n_out), \
            (w).M, (w).K, strm); \
    } \
} while(0)

// NVIDIA tensor core GEMM: decode → shared mem → WMMA (sm_80+ Ampere/Hopper/Ada/Blackwell)
#ifdef TURBO_NVIDIA
static int s_force_cublas = -1;
static void check_force_cublas() {
    if (s_force_cublas < 0) {
        const char* e = getenv("TURBO_CUBLAS");
        s_force_cublas = (e && e[0] == '1') ? 1 : 0;
        if (s_force_cublas) printf("  TURBO_CUBLAS=1: all tensors use cuBLAS path\n");
    }
}
#define FUSED_MATVEC(w, bf16_in, out_buf, n_in, n_out, bs, strm) do { \
    if ((w).split_sm) { \
        /* Route through fused or cuBLAS based on TURBO_CUBLAS env */ \
        if ((w).M >= 4096 && !s_force_cublas) { \
            nv_launch_split12_fused_gemm_async((w).split_sm, (w).split_gr, (w).base_exp, \
                bf16_in, n_in, out_buf, n_out, (w).M, (w).K, bs, strm, \
                nullptr, nullptr, nullptr, nullptr); \
        } else { \
            nv_launch_split12_cublas_batch_async((w).split_sm, (w).split_gr, (w).base_exp, \
                bf16_in, n_in, out_buf, n_out, state->weight_buf, state->weight_buf_half, \
                (w).M, (w).K, bs, strm); \
        } \
        if ((w).num_nonempty_rows > 0) \
            nv_launch_patches_batch_async((w).row_offsets, (w).patch_cols, \
                (w).patch_correct, (w).patch_wrong, \
                (w).patch_nonempty_rows, (w).num_nonempty_rows, \
                bf16_in, n_in, out_buf, n_out, bs, strm); \
    } \
} while(0)

// CUBLAS_MATVEC: decode split12 → BF16 buffer → cuBLAS GEMM + batched patches
#define CUBLAS_MATVEC(w, bf16_in, out_buf, n_in, n_out, bs, strm, wbuf, wbuf_half) do { \
    if ((w).split_sm) { \
        nv_launch_split12_cublas_batch_async((w).split_sm, (w).split_gr, (w).base_exp, \
            bf16_in, n_in, out_buf, n_out, wbuf, wbuf_half, (w).M, (w).K, bs, strm); \
        if ((w).num_nonempty_rows > 0) \
            nv_launch_patches_batch_async((w).row_offsets, (w).patch_cols, \
                (w).patch_correct, (w).patch_wrong, \
                (w).patch_nonempty_rows, (w).num_nonempty_rows, \
                bf16_in, n_in, out_buf, n_out, bs, strm); \
    } \
} while(0)
#endif

InferenceState* create_inference_state(Model* model, int batch_size, int max_seq_len, int tp_size) {
    auto* s = new InferenceState();
    s->model = model;
    s->batch_size = batch_size;
    s->positions = new int[batch_size]();
    s->tp = nullptr;
    s->tp_rank = 0;
    int n = model->config.n_embd, n_ff = model->config.n_ff, n_vocab = model->config.n_vocab;
    int kv_dim = model->config.n_head_kv * (n / model->config.n_head);
    int bs = batch_size;

    // When TP is active, row-split weights produce smaller local outputs
    int local_n = n / tp_size;           // local head dim for q (row-split wq)
    int local_kv = kv_dim / tp_size;     // local kv dim (row-split wk/wv)
    int local_nff = n_ff / tp_size;      // local FFN dim (row-split w_gate/w_up)
    int local_vocab = n_vocab / tp_size;  // local vocab (row-split output_proj)

    // hidden/hidden2/res stay full size (hold all-reduced results)
    GPU_CHECK(hipMalloc(&s->hidden,   bs * n * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->hidden2,  bs * n * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->attn_out, bs * n * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->q_buf,    bs * local_n * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->k_buf,    bs * local_kv * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->v_buf,    bs * local_kv * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->ffn_gate, bs * local_nff * sizeof(float)));
    GPU_CHECK(hipMalloc(&s->ffn_up,   bs * local_nff * sizeof(float)));
    s->ffn_down = nullptr;  // unused — w_down writes directly to cur
    GPU_CHECK(hipMalloc(&s->logits,   bs * local_vocab * sizeof(float)));
    s->attn_scores_buf = nullptr;  // unused — flash attention uses tiled LDS
    GPU_CHECK(hipMalloc(&s->d_positions, bs * sizeof(int)));
    GPU_CHECK(hipMalloc(&s->d_tokens, bs * sizeof(int)));
    int max_act = std::max(n, local_nff);
    // Pad batch dimension to 64 for GEMM tile alignment: the fused decode+GEMM
    // kernel loads full TILE_N-wide activation tiles via cp.async without bounds
    // checking. When bs is not a multiple of TILE_N (e.g., B=48 with TILE_N=32),
    // the last tile reads past the allocation. Padding to 64 (max TILE_N) avoids OOB.
    int bs_padded = ((bs + 63) / 64) * 64;
    GPU_CHECK(hipMalloc(&s->bf16_act,  bs_padded * max_act * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&s->bf16_act2, bs_padded * max_act * sizeof(int16_t)));
    GPU_CHECK(hipStreamCreateWithFlags(&s->stream, hipStreamNonBlocking));
    GPU_CHECK(hipStreamCreateWithFlags(&s->stream2, hipStreamNonBlocking));
    GPU_CHECK(hipEventCreateWithFlags(&s->sync_event, hipEventDisableTiming));
#ifdef TURBO_NVIDIA
    // Ping-pong BF16 buffers for decode+cuBLAS: only needed for weights with M < 4096
    // Fused GEMM handles M >= 4096 without materializing BF16 weights
    // Scan all weights to find the largest M*K that routes to cuBLAS
    {
        size_t max_cublas_elems = 0;
        auto check = [&](const CompressedWeight& w) {
            if (w.split_sm && w.M < 4096)
                max_cublas_elems = std::max(max_cublas_elems, (size_t)w.M * w.K);
        };
        for (int i = 0; i < model->config.n_layer; i++) {
            auto& L = model->layers[i];
            check(L.wq); check(L.wk); check(L.wv); check(L.wo);
            check(L.w_gate); check(L.w_up); check(L.w_down);
        }
        check(model->output_proj);
        s->weight_buf_half = max_cublas_elems;
        if (max_cublas_elems > 0)
            GPU_CHECK(hipMalloc(&s->weight_buf, max_cublas_elems * 2 * sizeof(int16_t)));
        else
            s->weight_buf = nullptr;
    }
#endif
    init_sampler(batch_size);
    return s;
}

void free_inference_state(InferenceState* state) {
    if (!state) return;
    delete[] state->positions;
    hipFree(state->hidden); hipFree(state->hidden2); hipFree(state->attn_out);
    hipFree(state->q_buf); hipFree(state->k_buf); hipFree(state->v_buf);
    hipFree(state->ffn_gate); hipFree(state->ffn_up);
    if (state->ffn_down) hipFree(state->ffn_down);
    hipFree(state->logits);
    if (state->attn_scores_buf) hipFree(state->attn_scores_buf);
    hipFree(state->d_positions); hipFree(state->d_tokens);
    hipFree(state->bf16_act); hipFree(state->bf16_act2);
    if (state->stream2) hipStreamDestroy(state->stream2);
    if (state->sync_event) hipEventDestroy(state->sync_event);
    delete state;
}

// Profiling support (TURBO_PROFILE=1 to enable)
static int s_profile = -1;
static int s_profile_tokens = 0;
static float s_prof_matvec = 0, s_prof_nonmv = 0;

#define PROF_INIT() do { \
    if (s_profile < 0) { const char* e = getenv("TURBO_PROFILE"); s_profile = (e && e[0]=='1') ? 1 : 0; } \
} while(0)

#define PROF_EVENT_DECL() hipEvent_t _pE0, _pE1
#define PROF_EVENT_CREATE() do { if (s_profile) { GPU_CHECK(hipEventCreate(&_pE0)); GPU_CHECK(hipEventCreate(&_pE1)); } } while(0)
#define PROF_EVENT_DESTROY() do { if (s_profile) { hipEventDestroy(_pE0); hipEventDestroy(_pE1); } } while(0)
#define PROF_START(strm) do { if (s_profile) hipEventRecord(_pE0, strm); } while(0)
#define PROF_END_MV(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_matvec += _ms; } } while(0)
#define PROF_END_NM(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_nonmv += _ms; } } while(0)
static float s_prof_attn = 0, s_prof_norm = 0, s_prof_silu = 0, s_prof_misc = 0, s_prof_nccl = 0;
#define PROF_END_ATTN(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_attn += _ms; s_prof_nonmv += _ms; } } while(0)
#define PROF_END_NORM(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_norm += _ms; s_prof_nonmv += _ms; } } while(0)
#define PROF_END_SILU(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_silu += _ms; s_prof_nonmv += _ms; } } while(0)
#define PROF_END_NCCL(strm) do { if (s_profile) { hipEventRecord(_pE1, strm); hipEventSynchronize(_pE1); \
    float _ms; hipEventElapsedTime(&_ms, _pE0, _pE1); s_prof_nccl += _ms; s_prof_nonmv += _ms; } } while(0)
#define PROF_TOKEN() do { if (s_profile && ++s_profile_tokens % 10 == 0) { \
    float total = s_prof_matvec + s_prof_nonmv; \
    printf("  [PROFILE] %d tok: mv %.1f (%.0f%%) | attn %.1f norm %.1f silu %.1f nccl %.1f misc %.1f | total %.1f (%.1f t/s)\n", \
        s_profile_tokens, s_prof_matvec, 100*s_prof_matvec/total, s_prof_attn, s_prof_norm, s_prof_silu, \
        s_prof_nccl, s_prof_nonmv - s_prof_attn - s_prof_norm - s_prof_silu - s_prof_nccl, total, 10000.0f/total); \
    s_prof_matvec = s_prof_nonmv = s_prof_attn = s_prof_norm = s_prof_silu = s_prof_misc = s_prof_nccl = 0; } } while(0)

// B=1 forward — optimized: no redundant memcpy, single sync at end
static void forward_b1(InferenceState* state, const int* token_ids) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int tp = m->tp_size > 1 ? m->tp_size : 1;
    int n = cfg.n_embd, head_dim = n / cfg.n_head;
    int n_head_local = cfg.n_head / tp;
    int n_head_kv_local = cfg.n_head_kv / tp;
    int kv_dim = n_head_kv_local * head_dim;
    int pos = state->positions[0];

    hipMemcpyAsync(state->d_tokens, token_ids, sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, 1, cfg.n_vocab, stream);


    // Ping-pong between hidden and hidden2 to avoid unnecessary copies
    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;   // BF16 activation buffer
    int16_t* bf16_b = state->bf16_act2;  // second BF16 buffer

    PROF_INIT();
    PROF_EVENT_DECL();
    PROF_EVENT_CREATE();

    // Split12 matvec + separate patches (inline patches was slower due to extra args)
    #define MATVEC_B1(w, bf16_in, fp32_out) do { \
        if ((w).split_sm) { \
            launch_split12_v2_async((w).split_sm, (w).split_gr, (w).base_exp, bf16_in, fp32_out, (w).M, (w).K, stream); \
            if ((w).num_patches > 0 && (w).row_offsets) \
                launch_patches_v2_async((w).row_offsets, (w).patch_cols, (w).patch_correct, (w).patch_wrong, \
                                        bf16_in, fp32_out, (w).M, stream); \
        } else { \
            launch_structured12_v2_async((w).packed, (w).base_exp, bf16_in, fp32_out, (w).M, (w).K, stream, \
                (w).row_offsets, (w).patch_cols, (w).patch_correct, (w).patch_wrong); \
        } \
    } while(0)

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // Fused RMSNorm → BF16 for Q/K/V input
        PROF_START(stream);
        if (layer == 0)
            launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        PROF_END_NORM(stream);


        PROF_START(stream);
        MATVEC_B1(L.wq, bf16_a, state->q_buf);

        MATVEC_B1(L.wk, bf16_a, state->k_buf);

        MATVEC_B1(L.wv, bf16_a, state->v_buf);

        PROF_END_MV(stream);

        PROF_START(stream);
        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_store_kv(state->q_buf, state->k_buf, state->v_buf,
                             m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                             head_dim, n_head_local, n_head_kv_local, pos,
                             m->max_seq_len, cfg.rope_theta, stream);

        if (pos < 1024)
            launch_attention_all_heads(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, n_head_local, n_head_kv_local, head_dim, pos+1, m->max_seq_len, scale, stream);
        else
            launch_flash_attention(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, n_head_local, n_head_kv_local, head_dim, pos+1, m->max_seq_len, scale, stream);


        PROF_END_ATTN(stream);

        PROF_START(stream);
        MATVEC_B1(L.wo, bf16_a, res);

        PROF_END_MV(stream);
        // TP: wo is column-split, output is partial sum — all-reduce to get full result
        if (state->tp) {
            PROF_START(stream);
            tp_allreduce_sum(res, n * 1, state->tp, state->tp_rank, (void*)stream, 1);
            PROF_END_NCCL(stream);
        }

        PROF_START(stream);
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        PROF_END_NORM(stream);

        PROF_START(stream);
        // Fused dual kernel: gate + up share activation reads (1 launch instead of 2)
        if (L.w_gate.split_sm && L.w_up.split_sm) {
            launch_split12_v2_dual_async(
                L.w_gate.split_sm, L.w_gate.split_gr, L.w_gate.base_exp,
                L.w_up.split_sm, L.w_up.split_gr, L.w_up.base_exp,
                bf16_a,
                state->ffn_gate, state->ffn_up,
                L.w_gate.M, L.w_gate.K, stream);
        } else {
            launch_structured12_v2_dual_async(
                L.w_gate.packed, L.w_gate.base_exp,
                L.w_up.packed, L.w_up.base_exp,
                bf16_a,
                state->ffn_gate, state->ffn_up,
                L.w_gate.M, L.w_gate.K, stream);
        }
        // Patch corrections — still separate for dual (inline would need 2 sets of patch args)
        if (L.w_gate.num_patches > 0 && L.w_gate.row_offsets)
            launch_patches_v2_async(L.w_gate.row_offsets, L.w_gate.patch_cols, L.w_gate.patch_correct, L.w_gate.patch_wrong,
                                    bf16_a, state->ffn_gate, L.w_gate.M, stream);
        if (L.w_up.num_patches > 0 && L.w_up.row_offsets)
            launch_patches_v2_async(L.w_up.row_offsets, L.w_up.patch_cols, L.w_up.patch_correct, L.w_up.patch_wrong,
                                    bf16_a, state->ffn_up, L.w_up.M, stream);
        PROF_END_MV(stream);

        PROF_START(stream);
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, L.w_gate.M, 1, stream);
        PROF_END_SILU(stream);

        PROF_START(stream);
        MATVEC_B1(L.w_down, bf16_b, cur);
        PROF_END_MV(stream);
        // TP: w_down is column-split, output is partial sum — all-reduce to get full result
        if (state->tp) {
            PROF_START(stream);
            tp_allreduce_sum(cur, n * 1, state->tp, state->tp_rank, (void*)stream, 0);
            PROF_END_NCCL(stream);
        }

        PROF_START(stream);
        if (layer + 1 < cfg.n_layer) {
            auto& nextL = m->layers[layer + 1];
            launch_add_rms_norm_bf16_batch(cur, res, nextL.attn_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        } else {
            launch_add_rms_norm_bf16_batch(cur, res, m->output_norm, bf16_a, n, cfg.rms_norm_eps, 1, stream);
        }
        PROF_END_NORM(stream);
    }

    // Output projection (bf16_a already set by fused add+norm above)
    PROF_START(stream);
    MATVEC_B1(m->output_proj, bf16_a, state->logits);
    PROF_END_MV(stream);
    PROF_EVENT_DESTROY();
    PROF_TOKEN();
    state->positions[0] = pos + 1;
    #undef MATVEC_B1
}

// B=8 batch8 matvec: prefer split12 when available
#define BATCH8_MATVEC(w, bf16_in, out_buf, n_in, n_out, strm) do { \
    if ((w).split_sm) { \
        launch_split12_batch8_async((w).split_sm, (w).split_gr, (w).base_exp, \
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
    } else { \
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
    } \
} while(0)

// B=4 forward — fully batched, minimal kernel launches
static void forward_b4(InferenceState* state, const int token_ids[4]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int tp = m->tp_size > 1 ? m->tp_size : 1;
    int n = cfg.n_embd, head_dim = n / cfg.n_head;
    int n_head_local = cfg.n_head / tp;
    int n_head_kv_local = cfg.n_head_kv / tp;
    int kv_dim = n_head_kv_local * head_dim;
    const int BS = 4;

    hipMemcpyAsync(state->d_positions, state->positions, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(state->d_tokens, token_ids, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, BS, cfg.n_vocab, stream);

    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;
    int16_t* bf16_b = state->bf16_act2;

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        // Fused RMSNorm -> BF16 for Q/K/V (layer 0 only; subsequent fused with prev add)
        if (layer == 0)
            launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);

#ifdef TURBO_NVIDIA
        // NVIDIA: auto-select split12 when available (saves 10 GB VRAM)
        BATCH4_MATVEC(L.wq, bf16_a, state->q_buf, n, L.wq.M, stream);
        BATCH4_MATVEC(L.wk, bf16_a, state->k_buf, n, kv_dim, stream);
        BATCH4_MATVEC(L.wv, bf16_a, state->v_buf, n, kv_dim, stream);
#else
        // AMD: original structured12 path (unchanged)
        launch_structured12_batch4_async(L.wq.packed, L.wq.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.wq.escape_row_base, L.wq.escape_counts, L.wq.escape_vals,
            SEQ(state->q_buf,0,L.wq.M), SEQ(state->q_buf,1,L.wq.M), SEQ(state->q_buf,2,L.wq.M), SEQ(state->q_buf,3,L.wq.M),
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
#endif

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, n_head_local, n_head_kv_local, n_head_local * head_dim, kv_dim, cfg.rope_theta, BS, stream);
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, BS, stream);

        int max_pos = 0;
        for (int s = 0; s < BS; s++) max_pos = std::max(max_pos, state->positions[s]);
        if (max_pos < 1024)
            launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, stream);
        else
            launch_flash_attention_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, stream);

#ifdef TURBO_NVIDIA
        BATCH4_MATVEC(L.wo, bf16_a, res, n_head_local * head_dim, n, stream);
        // TP: wo is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(res, n * BS, state->tp, state->tp_rank, (void*)stream, 1);
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        BATCH4_MATVEC(L.w_gate, bf16_a, state->ffn_gate, n, L.w_gate.M, stream);
        BATCH4_MATVEC(L.w_up, bf16_a, state->ffn_up, n, L.w_gate.M, stream);
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, L.w_gate.M, BS, stream);
        BATCH4_MATVEC(L.w_down, bf16_b, cur, L.w_gate.M, n, stream);
        // TP: w_down is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(cur, n * BS, state->tp, state->tp_rank, (void*)stream, 0);
#else
        launch_structured12_batch4_async(L.wo.packed, L.wo.base_exp,
            SEQI(bf16_a,0,n_head_local * head_dim), SEQI(bf16_a,1,n_head_local * head_dim), SEQI(bf16_a,2,n_head_local * head_dim), SEQI(bf16_a,3,n_head_local * head_dim),
            L.wo.escape_row_base, L.wo.escape_counts, L.wo.escape_vals,
            SEQ(res,0,n), SEQ(res,1,n), SEQ(res,2,n), SEQ(res,3,n),
            L.wo.M, L.wo.K, stream);
        // TP: wo is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(res, n * BS, state->tp, state->tp_rank, (void*)stream, 1);
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        launch_structured12_batch4_async(L.w_gate.packed, L.w_gate.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_gate.escape_row_base, L.w_gate.escape_counts, L.w_gate.escape_vals,
            SEQ(state->ffn_gate,0,L.w_gate.M), SEQ(state->ffn_gate,1,L.w_gate.M), SEQ(state->ffn_gate,2,L.w_gate.M), SEQ(state->ffn_gate,3,L.w_gate.M),
            L.w_gate.M, L.w_gate.K, stream);
        launch_structured12_batch4_async(L.w_up.packed, L.w_up.base_exp,
            SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
            L.w_up.escape_row_base, L.w_up.escape_counts, L.w_up.escape_vals,
            SEQ(state->ffn_up,0,L.w_gate.M), SEQ(state->ffn_up,1,L.w_gate.M), SEQ(state->ffn_up,2,L.w_gate.M), SEQ(state->ffn_up,3,L.w_gate.M),
            L.w_up.M, L.w_up.K, stream);
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, L.w_gate.M, BS, stream);
        launch_structured12_batch4_async(L.w_down.packed, L.w_down.base_exp,
            SEQI(bf16_b,0,L.w_gate.M), SEQI(bf16_b,1,L.w_gate.M), SEQI(bf16_b,2,L.w_gate.M), SEQI(bf16_b,3,L.w_gate.M),
            L.w_down.escape_row_base, L.w_down.escape_counts, L.w_down.escape_vals,
            SEQ(cur,0,n), SEQ(cur,1,n), SEQ(cur,2,n), SEQ(cur,3,n),
            L.w_down.M, L.w_down.K, stream);
        // TP: w_down is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(cur, n * BS, state->tp, state->tp_rank, (void*)stream, 0);
#endif

        // Fuse add + next layer's RMSNorm (saves 1 kernel launch per layer)
        if (layer + 1 < cfg.n_layer) {
            auto& nextL = m->layers[layer + 1];
            launch_add_rms_norm_bf16_batch(cur, res, nextL.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        } else {
            launch_add_rms_norm_bf16_batch(cur, res, m->output_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        }
    }

    // Output projection
#ifdef TURBO_NVIDIA
    BATCH4_MATVEC(m->output_proj, bf16_a, state->logits, n, m->output_proj.M, stream);
#else
    launch_structured12_batch4_async(m->output_proj.packed, m->output_proj.base_exp,
        SEQI(bf16_a,0,n), SEQI(bf16_a,1,n), SEQI(bf16_a,2,n), SEQI(bf16_a,3,n),
        m->output_proj.escape_row_base, m->output_proj.escape_counts, m->output_proj.escape_vals,
        SEQ(state->logits,0,m->output_proj.M), SEQ(state->logits,1,m->output_proj.M),
        SEQ(state->logits,2,m->output_proj.M), SEQ(state->logits,3,m->output_proj.M),
        m->output_proj.M, m->output_proj.K, stream);
#endif
    for (int s = 0; s < BS; s++) state->positions[s]++;
}

// B=8 forward — fully batched, minimal kernel launches
static void forward_b8(InferenceState* state, const int token_ids[8]) {
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t stream = state->stream;
    int tp = m->tp_size > 1 ? m->tp_size : 1;
    int n = cfg.n_embd, head_dim = n / cfg.n_head;
    int n_head_local = cfg.n_head / tp;
    int n_head_kv_local = cfg.n_head_kv / tp;
    int kv_dim = n_head_kv_local * head_dim;
    const int BS = 8;

    PROF_INIT();
    PROF_EVENT_DECL();
    PROF_EVENT_CREATE();

    hipMemcpyAsync(state->d_positions, state->positions, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(state->d_tokens, token_ids, BS * sizeof(int), hipMemcpyHostToDevice, stream);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, BS, cfg.n_vocab, stream);

    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);

    int16_t* bf16_a = state->bf16_act;
    int16_t* bf16_b = state->bf16_act2;

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        PROF_START(stream);
        // Fused RMSNorm -> BF16 for Q/K/V (layer 0 only; subsequent fused with prev add)
        if (layer == 0)
            launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        PROF_END_NORM(stream);

        PROF_START(stream);
        BATCH8_MATVEC(L.wq, bf16_a, state->q_buf, n, L.wq.M, stream);
        BATCH8_MATVEC(L.wk, bf16_a, state->k_buf, n, kv_dim, stream);
        BATCH8_MATVEC(L.wv, bf16_a, state->v_buf, n, kv_dim, stream);
        PROF_END_MV(stream);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, n_head_local, n_head_kv_local, n_head_local * head_dim, kv_dim, cfg.rope_theta, BS, stream);
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, BS, stream);

        PROF_START(stream);
        int max_pos = 0;
        for (int s = 0; s < BS; s++) max_pos = std::max(max_pos, state->positions[s]);
        if (max_pos < 1024)
            launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, stream);
        else
            launch_flash_attention_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, stream);
        PROF_END_ATTN(stream);

        // wo: attention now writes BF16 directly to bf16_a
        PROF_START(stream);
        BATCH8_MATVEC(L.wo, bf16_a, res, n_head_local * head_dim, n, stream);
        PROF_END_MV(stream);
        // TP: wo is column-split — all-reduce partial sums
        if (state->tp) {
            PROF_START(stream);
            tp_allreduce_sum(res, n * BS, state->tp, state->tp_rank, (void*)stream, 1);
            PROF_END_NCCL(stream);
        }

        PROF_START(stream);
        // Fused add + RMSNorm -> BF16
        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        PROF_END_NORM(stream);

        PROF_START(stream);
        BATCH8_MATVEC(L.w_gate, bf16_a, state->ffn_gate, n, L.w_gate.M, stream);
        BATCH8_MATVEC(L.w_up, bf16_a, state->ffn_up, n, L.w_gate.M, stream);
        PROF_END_MV(stream);

        PROF_START(stream);
        // Fused SiLU*mul -> BF16 for w_down
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, L.w_gate.M, BS, stream);
        PROF_END_SILU(stream);

        PROF_START(stream);
        BATCH8_MATVEC(L.w_down, bf16_b, cur, L.w_gate.M, n, stream);
        PROF_END_MV(stream);
        // TP: w_down is column-split — all-reduce partial sums
        if (state->tp) {
            PROF_START(stream);
            tp_allreduce_sum(cur, n * BS, state->tp, state->tp_rank, (void*)stream, 0);
            PROF_END_NCCL(stream);
        }

        PROF_START(stream);
        // Fuse add + next layer's RMSNorm (saves 1 kernel launch per layer)
        if (layer + 1 < cfg.n_layer) {
            auto& nextL = m->layers[layer + 1];
            launch_add_rms_norm_bf16_batch(cur, res, nextL.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        } else {
            // Fuse last layer's add with output RMSNorm
            launch_add_rms_norm_bf16_batch(cur, res, m->output_norm, bf16_a, n, cfg.rms_norm_eps, BS, stream);
        }
        PROF_END_NORM(stream);
    }

    // Output projection (bf16_a already set by fused add+norm above)
    PROF_START(stream);
    BATCH8_MATVEC(m->output_proj, bf16_a, state->logits, n, m->output_proj.M, stream);
    PROF_END_MV(stream);
    PROF_EVENT_DESTROY();
    PROF_TOKEN();
    for (int s = 0; s < BS; s++) state->positions[s]++;
}

// NVIDIA tiled batch forward with stream overlap:
// GEMM on stream1, patches on stream2 (patches hidden behind next GEMM)
#ifdef TURBO_NVIDIA

// Launch GEMM only (no patches) on the main stream
#define GEMM_ONLY(w, bf16_in, out_buf, n_in, n_out, bs, strm) do { \
    if ((w).split_sm) { \
        if ((w).M >= 4096) { \
            nv_launch_split12_fused_gemm_async((w).split_sm, (w).split_gr, (w).base_exp, \
                bf16_in, n_in, out_buf, n_out, (w).M, (w).K, bs, strm, \
                nullptr, nullptr, nullptr, nullptr); \
        } else { \
            nv_launch_split12_cublas_batch_async((w).split_sm, (w).split_gr, (w).base_exp, \
                bf16_in, n_in, out_buf, n_out, state->weight_buf, state->weight_buf_half, \
                (w).M, (w).K, bs, strm); \
        } \
    } \
} while(0)

// Launch patches on stream2, after GEMM finishes on stream1
#define PATCHES_ASYNC(w, bf16_in, out_buf, n_in, n_out, bs, s1, s2, ev) do { \
    if ((w).split_sm && (w).num_nonempty_rows > 0) { \
        hipEventRecord(ev, s1); \
        hipStreamWaitEvent(s2, ev, 0); \
        nv_launch_patches_batch_async((w).row_offsets, (w).patch_cols, \
            (w).patch_correct, (w).patch_wrong, \
            (w).patch_nonempty_rows, (w).num_nonempty_rows, \
            bf16_in, n_in, out_buf, n_out, bs, s2); \
    } \
} while(0)

// Sync: ensure stream2 patches are done before consuming the output
#define SYNC_PATCHES(s1, s2, ev) do { \
    hipEventRecord(ev, s2); \
    hipStreamWaitEvent(s1, ev, 0); \
} while(0)

static void forward_batch_tiled(InferenceState* state, const int* token_ids) {
    check_force_cublas();
    Model* m = state->model;
    auto& cfg = m->config;
    hipStream_t s1 = state->stream;      // main compute stream
    hipStream_t s2 = state->stream2;     // patch correction stream
    hipEvent_t ev = state->sync_event;
    int tp = m->tp_size > 1 ? m->tp_size : 1;
    int n = cfg.n_embd, head_dim = n / cfg.n_head;
    int n_head_local = cfg.n_head / tp;
    int n_head_kv_local = cfg.n_head_kv / tp;
    int kv_dim = n_head_kv_local * head_dim;
    int BS = state->batch_size;

    hipMemcpyAsync(state->d_positions, state->positions, BS * sizeof(int), hipMemcpyHostToDevice, s1);
    hipMemcpyAsync(state->d_tokens, token_ids, BS * sizeof(int), hipMemcpyHostToDevice, s1);
    launch_embed_lookup(m->token_embd, state->d_tokens, state->hidden, n, BS, cfg.n_vocab, s1);

    float* cur = state->hidden;
    float* res = state->hidden2;
    float scale = 1.0f / sqrtf((float)head_dim);
    int16_t* bf16_a = state->bf16_act;
    int16_t* bf16_b = state->bf16_act2;

    for (int layer = 0; layer < cfg.n_layer; layer++) {
        auto& L = m->layers[layer];

        if (layer == 0)
            launch_rms_norm_bf16_batch(cur, L.attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, s1);

        // wq/wk/wv: GEMM on s1, patches on s2 (overlapped with next GEMM)
        GEMM_ONLY(L.wq, bf16_a, state->q_buf, n, L.wq.M, BS, s1);
        PATCHES_ASYNC(L.wq, bf16_a, state->q_buf, n, L.wq.M, BS, s1, s2, ev);
        GEMM_ONLY(L.wk, bf16_a, state->k_buf, n, kv_dim, BS, s1);
        PATCHES_ASYNC(L.wk, bf16_a, state->k_buf, n, kv_dim, BS, s1, s2, ev);
        GEMM_ONLY(L.wv, bf16_a, state->v_buf, n, kv_dim, BS, s1);
        PATCHES_ASYNC(L.wv, bf16_a, state->v_buf, n, kv_dim, BS, s1, s2, ev);

        // Sync patches before consuming q/k/v
        SYNC_PATCHES(s1, s2, ev);

        size_t kv_off = (size_t)layer * m->max_seq_len * kv_dim;
        launch_rope_batch(state->q_buf, state->k_buf, state->d_positions,
                          head_dim, n_head_local, n_head_kv_local, n_head_local * head_dim, kv_dim, cfg.rope_theta, BS, s1);
        launch_store_kv_batch(state->k_buf, state->v_buf,
                              m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                              state->d_positions, kv_dim, m->max_seq_len, BS, s1);

        int max_pos = 0;
        for (int s = 0; s < BS; s++) max_pos = std::max(max_pos, state->positions[s]);
        if (max_pos < 1024)
            launch_attention_all_heads_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, s1);
        else
            launch_flash_attention_batch(state->q_buf, m->kv_cache_k + kv_off, m->kv_cache_v + kv_off,
                bf16_a, state->d_positions, n_head_local, n_head_kv_local, head_dim,
                max_pos + 1, scale, BS, 0, s1);

        // wo: GEMM + patches (patches must finish before add_rms_norm reads res)
        GEMM_ONLY(L.wo, bf16_a, res, n_head_local * head_dim, n, BS, s1);
        PATCHES_ASYNC(L.wo, bf16_a, res, n_head_local * head_dim, n, BS, s1, s2, ev);
        SYNC_PATCHES(s1, s2, ev);
        // TP: wo is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(res, n * BS, state->tp, state->tp_rank, (void*)s1, 1);

        launch_add_rms_norm_bf16_batch(res, cur, L.ffn_norm, bf16_a, n, cfg.rms_norm_eps, BS, s1);

        // w_gate + w_up: patches overlap with next GEMM
        GEMM_ONLY(L.w_gate, bf16_a, state->ffn_gate, n, L.w_gate.M, BS, s1);
        PATCHES_ASYNC(L.w_gate, bf16_a, state->ffn_gate, n, L.w_gate.M, BS, s1, s2, ev);
        GEMM_ONLY(L.w_up, bf16_a, state->ffn_up, n, L.w_gate.M, BS, s1);
        PATCHES_ASYNC(L.w_up, bf16_a, state->ffn_up, n, L.w_gate.M, BS, s1, s2, ev);

        // Sync patches before silu_mul consumes gate+up
        SYNC_PATCHES(s1, s2, ev);
        launch_silu_mul_bf16_batch(state->ffn_gate, state->ffn_up, bf16_b, L.w_gate.M, BS, s1);

        // w_down: patches must finish before add_rms_norm
        GEMM_ONLY(L.w_down, bf16_b, cur, L.w_gate.M, n, BS, s1);
        PATCHES_ASYNC(L.w_down, bf16_b, cur, L.w_gate.M, n, BS, s1, s2, ev);
        SYNC_PATCHES(s1, s2, ev);
        // TP: w_down is column-split — all-reduce partial sums
        if (state->tp)
            tp_allreduce_sum(cur, n * BS, state->tp, state->tp_rank, (void*)s1, 0);

        if (layer + 1 < cfg.n_layer)
            launch_add_rms_norm_bf16_batch(cur, res, m->layers[layer + 1].attn_norm, bf16_a, n, cfg.rms_norm_eps, BS, s1);
        else
            launch_add_rms_norm_bf16_batch(cur, res, m->output_norm, bf16_a, n, cfg.rms_norm_eps, BS, s1);
    }

    // output_proj: patches must finish before argmax
    GEMM_ONLY(m->output_proj, bf16_a, state->logits, n, m->output_proj.M, BS, s1);
    PATCHES_ASYNC(m->output_proj, bf16_a, state->logits, n, m->output_proj.M, BS, s1, s2, ev);
    SYNC_PATCHES(s1, s2, ev);

    for (int s = 0; s < BS; s++) state->positions[s]++;
}
#undef GEMM_ONLY
#undef PATCHES_ASYNC
#undef SYNC_PATCHES
#endif

void forward(InferenceState* state, const int* token_ids) {
#ifdef TURBO_NVIDIA
    // NVIDIA: decode+cuBLAS for B>8 (tensor cores win at high batch)
    // B<=8 uses hand-optimized split12 per-row kernels (faster for small batch)
    // TP dimension bugs in forward_batch_tiled fixed — re-enabled for TP
    if (state->batch_size > 8 && state->model->layers[0].wq.split_sm) {
        forward_batch_tiled(state, token_ids); return;
    }
#endif
    if (state->batch_size == 8) forward_b8(state, token_ids);
    else if (state->batch_size == 4) forward_b4(state, token_ids);
    else if (state->batch_size > 8 && state->batch_size % 8 == 0) {
        // Process as multiple B=8 passes (e.g., B=16 = 2× B=8, B=32 = 4× B=8)
        int orig_bs = state->batch_size;
        int n = state->model->config.n_embd;
        int tp = state->model->tp_size > 1 ? state->model->tp_size : 1;
        int head_dim = n / state->model->config.n_head;
        int n_head_local = state->model->config.n_head / tp;
        int n_head_kv_local = state->model->config.n_head_kv / tp;
        int local_n = n_head_local * head_dim;
        int kv_dim = n_head_kv_local * head_dim;
        int local_nff = state->model->config.n_ff / tp;
        int local_vocab = state->model->config.n_vocab / tp;

        for (int chunk = 0; chunk < orig_bs; chunk += 8) {
            // Temporarily adjust state to point to this chunk's slice
            InferenceState slice = *state;
            slice.batch_size = 8;
            slice.positions = state->positions + chunk;
            slice.hidden = state->hidden + chunk * n;
            slice.hidden2 = state->hidden2 + chunk * n;
            slice.attn_out = state->attn_out + chunk * n;
            slice.q_buf = state->q_buf + chunk * local_n;
            slice.k_buf = state->k_buf + chunk * kv_dim;
            slice.v_buf = state->v_buf + chunk * kv_dim;
            slice.ffn_gate = state->ffn_gate + chunk * local_nff;
            slice.ffn_up = state->ffn_up + chunk * local_nff;
            slice.logits = state->logits + chunk * local_vocab;
            slice.d_positions = state->d_positions + chunk;
            slice.d_tokens = state->d_tokens + chunk;
            slice.bf16_act = state->bf16_act + chunk * std::max(n, local_nff);
            slice.bf16_act2 = state->bf16_act2 + chunk * std::max(n, local_nff);

            forward_b8(&slice, token_ids + chunk);
        }
    } else {
        forward_b1(state, token_ids);
    }
}

std::vector<int> generate(InferenceState* state, const std::vector<int>& prompt_tokens, int max_tokens) {
    std::vector<int> output;
    int n_vocab = state->model->config.n_vocab;

    // TP: each GPU has local_vocab logits; distributed argmax finds global max
    int local_vocab = state->tp ? (n_vocab / state->tp->tp_size) : n_vocab;

    if (state->batch_size == 1) {
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            state->positions[0] = i;
            forward(state, &prompt_tokens[i]);
        }
        for (int t = 0; t < max_tokens; t++) {
            int next;
            if (state->tp) {
                next = tp_distributed_argmax(state->logits, local_vocab, n_vocab,
                                             state->tp, state->tp_rank, (void*)state->stream);
            } else {
                next = sample_greedy(state->logits, n_vocab, state->stream);
            }
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
            if (state->tp) {
                // TP batched: distributed argmax for each sequence
                for (int s = 0; s < bs; s++) {
                    tokens[s] = tp_distributed_argmax(state->logits + s * local_vocab,
                                                      local_vocab, n_vocab,
                                                      state->tp, state->tp_rank, (void*)state->stream);
                }
            } else {
                // Single batched argmax launch + one sync (was: bs separate launches + syncs)
                sample_greedy_batch(state->logits, tokens.data(), n_vocab, bs, state->stream);
            }
            if (tokens[0] == 2) break;
            output.push_back(tokens[0]);
            forward(state, tokens.data());
        }
    }
    return output;
}

// TP forward: run both GPUs in parallel using 2 threads.
// NCCL allreduce inside forward() naturally synchronizes both ranks.
#include <thread>

static void forward_rank(InferenceState* state, const int* token_ids, int device_id) {
    GPU_CHECK(hipSetDevice(device_id));
    forward(state, token_ids);
    GPU_CHECK(hipStreamSynchronize(state->stream));
}

static void forward_tp(TPState* tp, const int* token_ids, int* device_ids) {
    std::thread t1(forward_rank, tp->states[1], token_ids, device_ids[1]);
    forward_rank(tp->states[0], token_ids, device_ids[0]);
    t1.join();
}

std::vector<int> generate_tp(TPState* tp, const std::vector<int>& prompt_tokens, int max_tokens, int* device_ids) {
    std::vector<int> output;
    InferenceState* s0 = tp->states[0];
    int n_vocab = s0->model->config.n_vocab;
    int local_vocab = n_vocab / tp->tp_size;

    if (s0->batch_size == 1) {
        // Prefill: process each prompt token on both GPUs
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            for (int r = 0; r < tp->tp_size; r++)
                tp->states[r]->positions[0] = i;
            forward_tp(tp, &prompt_tokens[i], device_ids);
        }
        // Decode: generate tokens
        for (int t = 0; t < max_tokens; t++) {
            // Distributed argmax on rank 0 (it coordinates with rank 1 via NCCL)
            GPU_CHECK(hipSetDevice(device_ids[0]));
            int next = tp_distributed_argmax(s0->logits, local_vocab, n_vocab,
                                              tp, 0, (void*)s0->stream);
            if (next == 2 || next == s0->model->config.n_vocab - 1) break; // EOS
            output.push_back(next);
            forward_tp(tp, &next, device_ids);
        }
    } else {
        int bs = s0->batch_size;
        std::vector<int> tokens(bs);
        // Prefill
        for (int i = 0; i < (int)prompt_tokens.size(); i++) {
            for (int r = 0; r < tp->tp_size; r++)
                for (int s = 0; s < bs; s++)
                    tp->states[r]->positions[s] = i;
            for (int s = 0; s < bs; s++) tokens[s] = prompt_tokens[i];
            forward_tp(tp, tokens.data(), device_ids);
        }
        // Decode
        for (int t = 0; t < max_tokens; t++) {
            GPU_CHECK(hipSetDevice(device_ids[0]));
            for (int s = 0; s < bs; s++) {
                tokens[s] = tp_distributed_argmax(s0->logits + s * local_vocab,
                                                   local_vocab, n_vocab,
                                                   tp, 0, (void*)s0->stream);
            }
            if (tokens[0] == 2) break;
            output.push_back(tokens[0]);
            forward_tp(tp, tokens.data(), device_ids);
        }
    }
    return output;
}
