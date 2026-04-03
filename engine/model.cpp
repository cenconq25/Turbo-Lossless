#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>


// Read entire binary file into host memory
static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); return {}; }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read((char*)data.data(), size);
    return data;
}

// Upload raw data to GPU, return device pointer
template<typename T>
static T* upload_gpu(const void* host_data, size_t count) {
    T* d_ptr;
    size_t bytes = count * sizeof(T);
    GPU_CHECK(hipMalloc(&d_ptr, bytes));
    hipMemcpy(d_ptr, host_data, bytes, hipMemcpyHostToDevice);
    return d_ptr;
}

// Load a compressed weight tensor
static bool load_compressed(const std::string& dir, const std::string& prefix, CompressedWeight& w) {
    // Read dimensions
    std::string dims_path = dir + "/" + prefix + ".dims";
    std::ifstream df(dims_path);
    if (!df) { fprintf(stderr, "Missing: %s\n", dims_path.c_str()); return false; }
    df >> w.M >> w.K >> w.num_patches;
    // Read base_exp for structured 12-bit (4th field, optional for backward compat)
    w.base_exp = 0;
    if (df >> w.base_exp) {
        // structured 12-bit mode
    }
    df.close();

    // Check if split12 format is available (skip packed12 upload to save VRAM)
    bool has_split12 = false;
    {
        std::string sm_path = dir + "/" + prefix + ".sm.bin";
        std::string gr_path = dir + "/" + prefix + ".gr.bin";
        std::ifstream sf(sm_path, std::ios::binary), gf(gr_path, std::ios::binary);
        has_split12 = sf.good() && gf.good();
    }

    // Read and upload packed data (skip if split12 available — saves ~10 GB VRAM)
    auto packed = read_file(dir + "/" + prefix + ".packed.bin");
    if (packed.empty() && !has_split12) return false;
    if (!packed.empty() && !has_split12)
        w.packed = upload_gpu<int32_t>(packed.data(), packed.size() / sizeof(int32_t));
    else
        w.packed = nullptr;

    // Codebook (not needed for structured 12-bit, but load if present for backward compat)
    {
        std::string cb_path = dir + "/" + prefix + ".codebook.bin";
        std::ifstream cbf(cb_path, std::ios::binary);
        if (cbf) {
            auto cb = read_file(cb_path);
            w.codebook = upload_gpu<int16_t>(cb.data(), cb.size() / sizeof(int16_t));
        } else {
            w.codebook = nullptr;  // structured12 mode — no codebook needed
        }
    }

    // CSR escape data
    auto ro = read_file(dir + "/" + prefix + ".row_off.bin");
    if (!ro.empty())
        w.row_offsets = upload_gpu<int32_t>(ro.data(), ro.size() / sizeof(int32_t));
    else
        w.row_offsets = nullptr;

    auto pc = read_file(dir + "/" + prefix + ".patch_cols.bin");
    if (!pc.empty())
        w.patch_cols = upload_gpu<int32_t>(pc.data(), pc.size() / sizeof(int32_t));
    else
        w.patch_cols = nullptr;

    auto pcv = read_file(dir + "/" + prefix + ".patch_correct.bin");
    if (!pcv.empty())
        w.patch_correct = upload_gpu<int16_t>(pcv.data(), pcv.size() / sizeof(int16_t));
    else
        w.patch_correct = nullptr;

    auto pw = read_file(dir + "/" + prefix + ".patch_wrong.bin");
    if (!pw.empty())
        w.patch_wrong = upload_gpu<int16_t>(pw.data(), pw.size() / sizeof(int16_t));
    else
        w.patch_wrong = nullptr;

    // Load split12 format if available (byte-aligned, zero read amplification)
    w.split_sm = nullptr;
    w.split_gr = nullptr;
    auto sm_data = read_file(dir + "/" + prefix + ".sm.bin");
    auto gr_data = read_file(dir + "/" + prefix + ".gr.bin");
    if (!sm_data.empty() && !gr_data.empty()) {
        w.split_sm = upload_gpu<uint8_t>(sm_data.data(), sm_data.size());
        w.split_gr = upload_gpu<uint8_t>(gr_data.data(), gr_data.size());
    }

    // Build fused escape table from CSR data
    // escape_offsets[row*256+tid] = exclusive prefix sum of escapes for thread tid in row
    // escape_vals = correct BF16 values in (row, tid, encounter_order) order
    w.escape_row_base = nullptr;
    w.escape_counts = nullptr;
    w.escape_vals = nullptr;
    if (w.num_patches > 0 && !ro.empty() && !pc.empty() && !pcv.empty()) {
        const int WG = 256;
        const int32_t* h_row_off = (const int32_t*)ro.data();
        const int32_t* h_cols = (const int32_t*)pc.data();
        const int16_t* h_correct = (const int16_t*)pcv.data();

        // Pass 1: count escapes per (row, tid)
        std::vector<int32_t> counts(w.M * WG, 0);
        for (int r = 0; r < w.M; r++) {
            for (int p = h_row_off[r]; p < h_row_off[r + 1]; p++) {
                int tid = h_cols[p] % WG;
                counts[r * WG + tid]++;
            }
        }

        // Pass 2: exclusive prefix sum → split into row_base + thread_off
        std::vector<int32_t> row_base(w.M);
        
        std::vector<int32_t> abs_off(w.M * WG);
        int total = 0;
        for (int r = 0; r < w.M; r++) {
            row_base[r] = total;
            for (int t = 0; t < WG; t++) {
                int idx = r * WG + t;
                abs_off[idx] = total;
                total += counts[idx];
            }
        }

        // Pass 3: fill escape_vals in thread-stride order
        std::vector<int16_t> esc_vals(total);
        std::vector<int32_t> fill_ptr(w.M * WG);
        for (int i = 0; i < w.M * WG; i++) fill_ptr[i] = abs_off[i];
        for (int r = 0; r < w.M; r++) {
            for (int p = h_row_off[r]; p < h_row_off[r + 1]; p++) {
                int tid = h_cols[p] % WG;
                int idx = r * WG + tid;
                esc_vals[fill_ptr[idx]++] = h_correct[p];
            }
        }

        w.escape_row_base = upload_gpu<int32_t>(row_base.data(), row_base.size());

        // Optional: pre-compute escape counts for fast path (costs M*256 bytes VRAM)
        static int s_fast_mode = -1;
        if (s_fast_mode < 0) {
            const char* env = getenv("TURBO_FAST");
            s_fast_mode = (env && env[0] == '1') ? 1 : 0;
            if (s_fast_mode) printf("  TURBO_FAST=1: escape count table (uses ~361 MB extra VRAM)\n");
            else printf("  TURBO_FAST=0: on-the-fly escape scan (saves VRAM)\n");
        }
        if (s_fast_mode) {
            std::vector<uint8_t> esc_counts(w.M * WG);
            for (int i = 0; i < w.M * WG; i++)
                esc_counts[i] = (uint8_t)counts[i];
            w.escape_counts = upload_gpu<uint8_t>(esc_counts.data(), esc_counts.size());
        }

        w.escape_vals = upload_gpu<int16_t>(esc_vals.data(), esc_vals.size());

        // Build sparse nonempty row list for efficient patch correction
        std::vector<int32_t> nonempty;
        for (int r = 0; r < w.M; r++) {
            if (h_row_off[r + 1] > h_row_off[r])
                nonempty.push_back(r);
        }
        w.num_nonempty_rows = (int)nonempty.size();
        if (!nonempty.empty())
            w.patch_nonempty_rows = upload_gpu<int32_t>(nonempty.data(), nonempty.size());
        else
            w.patch_nonempty_rows = nullptr;
    } else {
        w.patch_nonempty_rows = nullptr;
        w.num_nonempty_rows = 0;
    }

    return true;
}

// Try to load a TP-sharded compressed weight; fall back to full weight if not found
static bool load_compressed_tp(const std::string& dir, const std::string& prefix,
                               CompressedWeight& w, int tp_rank, int tp_size) {
    if (tp_size > 1) {
        // Try TP shard file first
        char tp_prefix[256];
        snprintf(tp_prefix, sizeof(tp_prefix), "%s.tp%d", prefix.c_str(), tp_rank);
        std::string tp_dims = dir + "/" + tp_prefix + ".dims";
        std::ifstream tf(tp_dims);
        if (tf.good()) {
            tf.close();
            return load_compressed(dir, tp_prefix, w);
        }
        // Fall back to full weight (backward compatible)
    }
    return load_compressed(dir, prefix, w);
}

Model* load_model(const std::string& model_path, int device_id, int tp_rank, int tp_size) {
    hipSetDevice(device_id);

    Model* m = new Model();
    m->tp_rank = tp_rank;
    m->tp_size = tp_size;
    std::string dir = model_path;

    // Check if this is a turbo-converted directory
    auto config_data = read_file(dir + "/config.bin");
    if (config_data.empty()) {
        fprintf(stderr, "No config.bin found in %s\n", dir.c_str());
        fprintf(stderr, "Run: python3 engine/convert_model.py <model_dir>\n");
        delete m;
        return nullptr;
    }

    // Parse config — check for v2 magic ("TLv2") at start
    bool is_v2 = (config_data.size() >= 4 &&
                  config_data[0] == 'T' && config_data[1] == 'L' &&
                  config_data[2] == 'v' && config_data[3] == '2');

    // Zero-init Gemma4 extensions
    m->config.head_dim_sliding = 0;
    m->config.head_dim_full = 0;
    m->config.sliding_window = 0;
    m->config.logit_softcap = 0;
    m->config.num_kv_shared = 0;
    m->config.rope_theta_full = 0;
    m->config.partial_rotary = 1.0f;
    m->config.activation_type = 0;
    m->config.tie_embeddings = 0;
    memset(m->config.layer_types, 0, sizeof(m->config.layer_types));

    if (is_v2) {
        // V2 config format: "TLv2" + legacy fields + extended fields
        const uint8_t* p = config_data.data() + 4;  // skip magic
        struct { int n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ff, n_ctx; float rope_theta, rms_eps; } cfg;
        memcpy(&cfg, p, sizeof(cfg));
        p += sizeof(cfg);
        m->config.n_vocab = cfg.n_vocab;
        m->config.n_embd = cfg.n_embd;
        m->config.n_head = cfg.n_head;
        m->config.n_head_kv = cfg.n_head_kv;
        m->config.n_layer = cfg.n_layer;
        m->config.n_ff = cfg.n_ff;
        m->config.n_ctx = cfg.n_ctx;
        m->config.rope_theta = cfg.rope_theta;
        m->config.rms_norm_eps = cfg.rms_eps;

        // Extended v2 fields
        struct {
            int head_dim_sliding, head_dim_full, sliding_window;
            float logit_softcap;
            int num_kv_shared;
            float rope_theta_full, partial_rotary;
            int activation_type, tie_embeddings;
        } ext;
        size_t remaining = config_data.size() - 4 - sizeof(cfg);
        if (remaining >= sizeof(ext)) {
            memcpy(&ext, p, sizeof(ext));
            p += sizeof(ext);
            m->config.head_dim_sliding = ext.head_dim_sliding;
            m->config.head_dim_full = ext.head_dim_full;
            m->config.sliding_window = ext.sliding_window;
            m->config.logit_softcap = ext.logit_softcap;
            m->config.num_kv_shared = ext.num_kv_shared;
            m->config.rope_theta_full = ext.rope_theta_full;
            m->config.partial_rotary = ext.partial_rotary;
            m->config.activation_type = ext.activation_type;
            m->config.tie_embeddings = ext.tie_embeddings;

            // Layer types array (up to 64 bytes)
            size_t lt_remaining = config_data.size() - (p - config_data.data());
            int lt_count = std::min((int)lt_remaining, std::min(cfg.n_layer, 64));
            if (lt_count > 0) {
                memcpy(m->config.layer_types, p, lt_count);
            }
        }

        printf("  Config v2: vocab=%d embd=%d heads=%d/%d layers=%d ff=%d\n",
               cfg.n_vocab, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.n_layer, cfg.n_ff);
        if (m->config.head_dim_sliding > 0) {
            printf("  Gemma4: sliding_hd=%d full_hd=%d window=%d softcap=%.1f shared=%d act=%s\n",
                   m->config.head_dim_sliding, m->config.head_dim_full,
                   m->config.sliding_window, m->config.logit_softcap,
                   m->config.num_kv_shared,
                   m->config.activation_type == 1 ? "gelu_tanh" : "silu");
        }
    } else {
        // Legacy v1 config format (backward compatible)
        struct { int n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ff, n_ctx; float rope_theta, rms_eps; } cfg;
        memcpy(&cfg, config_data.data(), sizeof(cfg));
        m->config.n_vocab = cfg.n_vocab;
        m->config.n_embd = cfg.n_embd;
        m->config.n_head = cfg.n_head;
        m->config.n_head_kv = cfg.n_head_kv;
        m->config.n_layer = cfg.n_layer;
        m->config.n_ff = cfg.n_ff;
        m->config.n_ctx = cfg.n_ctx;
        m->config.rope_theta = cfg.rope_theta;
        m->config.rms_norm_eps = cfg.rms_eps;

        printf("  Config: vocab=%d embd=%d heads=%d/%d layers=%d ff=%d\n",
               cfg.n_vocab, cfg.n_embd, cfg.n_head, cfg.n_head_kv, cfg.n_layer, cfg.n_ff);
    }
    if (tp_size > 1)
        printf("  TP: rank %d/%d\n", tp_rank, tp_size);

    // Token embeddings (BF16 on GPU)
    auto embd = read_file(dir + "/tok_embd.bin");
    if (embd.empty()) { fprintf(stderr, "Missing tok_embd.bin\n"); delete m; return nullptr; }
    m->token_embd = upload_gpu<int16_t>(embd.data(), embd.size() / sizeof(int16_t));
    printf("  Embeddings: %.1f MB\n", embd.size() / 1e6);

    // Output norm
    auto norm = read_file(dir + "/output_norm.bin");
    if (!norm.empty()) {
        m->output_norm = upload_gpu<float>(norm.data(), norm.size() / sizeof(float));
    }

    // Output projection (for tie_embeddings, output_proj may not exist — handled in inference)
    if (!load_compressed_tp(dir, "output_proj", m->output_proj, tp_rank, tp_size)) {
        if (m->config.tie_embeddings)
            printf("  Tied embeddings: output_proj uses transposed embed_tokens\n");
        else
            fprintf(stderr, "Warning: no output_proj\n");
    }

    // Layers
    int n_layer = m->config.n_layer;
    m->layers.resize(n_layer);
    bool is_gemma4 = (m->config.head_dim_sliding > 0);
    int legacy_head_dim = m->config.n_embd / m->config.n_head;

    for (int i = 0; i < n_layer; i++) {
        auto& layer = m->layers[i];
        char prefix[64];

        // Initialize Gemma4 extensions to defaults
        layer.q_norm = nullptr;
        layer.k_norm = nullptr;
        layer.pre_ffn_norm = nullptr;
        layer.post_ffn_norm = nullptr;
        layer.layer_scalar = 1.0f;
        layer.kv_cache_layer = i;  // default: own KV slot
        layer.is_full_attn = false;

        // Compute per-layer dimensions
        if (is_gemma4) {
            layer.is_full_attn = (m->config.layer_types[i] == 1);
            layer.head_dim = layer.is_full_attn ? m->config.head_dim_full : m->config.head_dim_sliding;
            layer.kv_dim = m->config.n_head_kv * layer.head_dim;

            // KV sharing: layers num_kv_shared..n_layer-1 share with layers 0..n_layer-num_kv_shared-1
            if (m->config.num_kv_shared > 0 && i >= (n_layer - m->config.num_kv_shared)) {
                int donor = i - (n_layer - m->config.num_kv_shared);
                layer.kv_cache_layer = donor;
            }
        } else {
            layer.head_dim = legacy_head_dim;
            layer.kv_dim = (m->config.n_head_kv / tp_size) * legacy_head_dim;
        }

        // Norms
        snprintf(prefix, sizeof(prefix), "layer.%d.attn_norm.bin", i);
        auto an = read_file(dir + "/" + prefix);
        if (!an.empty()) layer.attn_norm = upload_gpu<float>(an.data(), an.size() / sizeof(float));

        snprintf(prefix, sizeof(prefix), "layer.%d.ffn_norm.bin", i);
        auto fn = read_file(dir + "/" + prefix);
        if (!fn.empty()) layer.ffn_norm = upload_gpu<float>(fn.data(), fn.size() / sizeof(float));

        // Gemma4 extra norms (silently skip if files don't exist)
        snprintf(prefix, sizeof(prefix), "layer.%d.q_norm.bin", i);
        auto qn = read_file(dir + "/" + prefix);
        if (!qn.empty()) layer.q_norm = upload_gpu<float>(qn.data(), qn.size() / sizeof(float));

        snprintf(prefix, sizeof(prefix), "layer.%d.k_norm.bin", i);
        auto kn = read_file(dir + "/" + prefix);
        if (!kn.empty()) layer.k_norm = upload_gpu<float>(kn.data(), kn.size() / sizeof(float));

        snprintf(prefix, sizeof(prefix), "layer.%d.pre_ffn_norm.bin", i);
        auto pfn = read_file(dir + "/" + prefix);
        if (!pfn.empty()) layer.pre_ffn_norm = upload_gpu<float>(pfn.data(), pfn.size() / sizeof(float));

        snprintf(prefix, sizeof(prefix), "layer.%d.post_ffn_norm.bin", i);
        auto pofn = read_file(dir + "/" + prefix);
        if (!pofn.empty()) layer.post_ffn_norm = upload_gpu<float>(pofn.data(), pofn.size() / sizeof(float));

        // Compressed weights (TP-aware: tries .tp{rank} files first)
        snprintf(prefix, sizeof(prefix), "layer.%d.wq", i);
        load_compressed_tp(dir, prefix, layer.wq, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.wk", i);
        load_compressed_tp(dir, prefix, layer.wk, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.wv", i);
        load_compressed_tp(dir, prefix, layer.wv, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.wo", i);
        load_compressed_tp(dir, prefix, layer.wo, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_gate", i);
        load_compressed_tp(dir, prefix, layer.w_gate, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_up", i);
        load_compressed_tp(dir, prefix, layer.w_up, tp_rank, tp_size);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_down", i);
        load_compressed_tp(dir, prefix, layer.w_down, tp_rank, tp_size);

        if ((i + 1) % 8 == 0 || i == n_layer - 1)
            printf("  Loaded layer %d/%d\n", i + 1, n_layer);
    }

    // Allocate KV cache (TP: each rank handles n_head_kv/tp_size heads)
    // Max context length — configurable via TURBO_CTX env var (default 2048)
    const char* ctx_env = getenv("TURBO_CTX");
    m->max_seq_len = ctx_env ? atoi(ctx_env) : 2048;
    if (m->max_seq_len < 128) m->max_seq_len = 128;
    printf("  Context length: %d\n", m->max_seq_len);

    if (is_gemma4) {
        // Gemma4: per-layer KV cache with variable head_dim and KV sharing
        int num_kv_shared = m->config.num_kv_shared;
        m->num_kv_slots = n_layer - num_kv_shared;
        m->kv_k_ptrs.resize(n_layer);
        m->kv_v_ptrs.resize(n_layer);

        // Allocate unique KV slots (non-shared layers only)
        size_t total_kv_bytes = 0;
        std::vector<int16_t*> slot_k(m->num_kv_slots), slot_v(m->num_kv_slots);
        for (int i = 0; i < m->num_kv_slots; i++) {
            auto& layer = m->layers[i];
            int kv_heads_local = m->config.n_head_kv / tp_size;
            size_t slot_size = (size_t)m->max_seq_len * kv_heads_local * layer.head_dim;
            GPU_CHECK(hipMalloc(&slot_k[i], slot_size * sizeof(int16_t)));
            GPU_CHECK(hipMalloc(&slot_v[i], slot_size * sizeof(int16_t)));
            hipMemset(slot_k[i], 0, slot_size * sizeof(int16_t));
            hipMemset(slot_v[i], 0, slot_size * sizeof(int16_t));
            total_kv_bytes += slot_size * 2 * sizeof(int16_t);
        }

        // Assign per-layer pointers (shared layers point to donor's slot)
        for (int i = 0; i < n_layer; i++) {
            int slot = m->layers[i].kv_cache_layer;
            m->kv_k_ptrs[i] = slot_k[slot];
            m->kv_v_ptrs[i] = slot_v[slot];
        }

        // Legacy flat pointers unused for Gemma4
        m->kv_cache_k = nullptr;
        m->kv_cache_v = nullptr;
        printf("  KV cache (Gemma4): %.1f MB total, %d unique slots, %d shared\n",
               total_kv_bytes / 1e6, m->num_kv_slots, num_kv_shared);
    } else {
        // Legacy: single flat KV cache allocation
        int head_dim_kv = m->config.n_embd / m->config.n_head;
        int kv_heads_local = m->config.n_head_kv / tp_size;
        if (tp_size > 1)
            printf("  KV heads per rank: %d (of %d total)\n", kv_heads_local, m->config.n_head_kv);
        size_t kv_size = (size_t)n_layer * m->max_seq_len * kv_heads_local * head_dim_kv;
        GPU_CHECK(hipMalloc(&m->kv_cache_k, kv_size * sizeof(int16_t)));
        GPU_CHECK(hipMalloc(&m->kv_cache_v, kv_size * sizeof(int16_t)));
        hipMemset(m->kv_cache_k, 0, kv_size * sizeof(int16_t));
        hipMemset(m->kv_cache_v, 0, kv_size * sizeof(int16_t));
        printf("  KV cache: %.1f MB per K/V\n", kv_size * 2 / 1e6);

        // Build per-layer pointers for uniform access in inference
        int kv_dim_local = kv_heads_local * head_dim_kv;
        m->kv_k_ptrs.resize(n_layer);
        m->kv_v_ptrs.resize(n_layer);
        for (int i = 0; i < n_layer; i++) {
            size_t off = (size_t)i * m->max_seq_len * kv_dim_local;
            m->kv_k_ptrs[i] = m->kv_cache_k + off;
            m->kv_v_ptrs[i] = m->kv_cache_v + off;
        }
        m->num_kv_slots = n_layer;
    }

    size_t gpu_mem = 0;
    hipMemGetInfo(nullptr, &gpu_mem);
    printf("  Model loaded.\n");

    return m;
}

static void free_compressed_weight(CompressedWeight& w) {
    if (w.packed)           hipFree(w.packed);
    if (w.codebook)         hipFree(w.codebook);
    if (w.row_offsets)      hipFree(w.row_offsets);
    if (w.patch_cols)       hipFree(w.patch_cols);
    if (w.patch_correct)    hipFree(w.patch_correct);
    if (w.patch_wrong)      hipFree(w.patch_wrong);
    if (w.escape_row_base)  hipFree(w.escape_row_base);
    if (w.escape_counts)    hipFree(w.escape_counts);
    if (w.escape_vals)      hipFree(w.escape_vals);
    if (w.split_sm)         hipFree(w.split_sm);
    if (w.split_gr)         hipFree(w.split_gr);
}

void free_model(Model* model) {
    if (!model) return;

    if (model->token_embd)  hipFree(model->token_embd);
    if (model->output_norm) hipFree(model->output_norm);
    free_compressed_weight(model->output_proj);

    for (auto& layer : model->layers) {
        if (layer.attn_norm) hipFree(layer.attn_norm);
        if (layer.ffn_norm)  hipFree(layer.ffn_norm);
        if (layer.q_norm)       hipFree(layer.q_norm);
        if (layer.k_norm)       hipFree(layer.k_norm);
        if (layer.pre_ffn_norm) hipFree(layer.pre_ffn_norm);
        if (layer.post_ffn_norm) hipFree(layer.post_ffn_norm);
        free_compressed_weight(layer.wq);
        free_compressed_weight(layer.wk);
        free_compressed_weight(layer.wv);
        free_compressed_weight(layer.wo);
        free_compressed_weight(layer.w_gate);
        free_compressed_weight(layer.w_up);
        free_compressed_weight(layer.w_down);
    }

    // Free KV cache
    if (model->config.head_dim_sliding > 0) {
        // Gemma4: per-slot allocation (only free unique slots, not shared pointers)
        for (int i = 0; i < model->num_kv_slots; i++) {
            if (model->kv_k_ptrs[i]) hipFree(model->kv_k_ptrs[i]);
            if (model->kv_v_ptrs[i]) hipFree(model->kv_v_ptrs[i]);
        }
    } else {
        if (model->kv_cache_k) hipFree(model->kv_cache_k);
        if (model->kv_cache_v) hipFree(model->kv_cache_v);
    }

    delete model;
}
