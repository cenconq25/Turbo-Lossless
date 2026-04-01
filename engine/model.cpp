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
    hipMalloc(&d_ptr, bytes);
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

    // Read and upload packed data
    auto packed = read_file(dir + "/" + prefix + ".packed.bin");
    if (packed.empty()) return false;
    w.packed = upload_gpu<int32_t>(packed.data(), packed.size() / sizeof(int32_t));

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
    }

    return true;
}

Model* load_model(const std::string& model_path, int device_id) {
    hipSetDevice(device_id);

    Model* m = new Model();
    std::string dir = model_path;

    // Check if this is a turbo-converted directory
    auto config_data = read_file(dir + "/config.bin");
    if (config_data.empty()) {
        fprintf(stderr, "No config.bin found in %s\n", dir.c_str());
        fprintf(stderr, "Run: python3 engine/convert_model.py <model_dir>\n");
        delete m;
        return nullptr;
    }

    // Parse config
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

    // Output projection
    if (!load_compressed(dir, "output_proj", m->output_proj)) {
        fprintf(stderr, "Warning: no output_proj\n");
    }

    // Layers
    m->layers.resize(cfg.n_layer);
    for (int i = 0; i < cfg.n_layer; i++) {
        auto& layer = m->layers[i];
        char prefix[64];

        // Norms
        snprintf(prefix, sizeof(prefix), "layer.%d.attn_norm.bin", i);
        auto an = read_file(dir + "/" + prefix);
        if (!an.empty()) layer.attn_norm = upload_gpu<float>(an.data(), an.size() / sizeof(float));

        snprintf(prefix, sizeof(prefix), "layer.%d.ffn_norm.bin", i);
        auto fn = read_file(dir + "/" + prefix);
        if (!fn.empty()) layer.ffn_norm = upload_gpu<float>(fn.data(), fn.size() / sizeof(float));

        // Compressed weights
        snprintf(prefix, sizeof(prefix), "layer.%d.wq", i);
        load_compressed(dir, prefix, layer.wq);
        snprintf(prefix, sizeof(prefix), "layer.%d.wk", i);
        load_compressed(dir, prefix, layer.wk);
        snprintf(prefix, sizeof(prefix), "layer.%d.wv", i);
        load_compressed(dir, prefix, layer.wv);
        snprintf(prefix, sizeof(prefix), "layer.%d.wo", i);
        load_compressed(dir, prefix, layer.wo);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_gate", i);
        load_compressed(dir, prefix, layer.w_gate);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_up", i);
        load_compressed(dir, prefix, layer.w_up);
        snprintf(prefix, sizeof(prefix), "layer.%d.w_down", i);
        load_compressed(dir, prefix, layer.w_down);

        if ((i + 1) % 8 == 0 || i == cfg.n_layer - 1)
            printf("  Loaded layer %d/%d\n", i + 1, cfg.n_layer);
    }

    // Allocate KV cache
    int head_dim = cfg.n_embd / cfg.n_head;
    // Max context length — configurable via TURBO_CTX env var (default 2048)
    const char* ctx_env = getenv("TURBO_CTX");
    m->max_seq_len = ctx_env ? atoi(ctx_env) : 2048;
    if (m->max_seq_len < 128) m->max_seq_len = 128;
    printf("  Context length: %d\n", m->max_seq_len);
    size_t kv_size = (size_t)cfg.n_layer * m->max_seq_len * cfg.n_head_kv * head_dim;
    hipMalloc(&m->kv_cache_k, kv_size * sizeof(int16_t));
    hipMalloc(&m->kv_cache_v, kv_size * sizeof(int16_t));
    hipMemset(m->kv_cache_k, 0, kv_size * sizeof(int16_t));
    hipMemset(m->kv_cache_v, 0, kv_size * sizeof(int16_t));
    printf("  KV cache: %.1f MB per K/V\n", kv_size * 2 / 1e6);

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
        free_compressed_weight(layer.wq);
        free_compressed_weight(layer.wk);
        free_compressed_weight(layer.wv);
        free_compressed_weight(layer.wo);
        free_compressed_weight(layer.w_gate);
        free_compressed_weight(layer.w_up);
        free_compressed_weight(layer.w_down);
    }

    if (model->kv_cache_k) hipFree(model->kv_cache_k);
    if (model->kv_cache_v) hipFree(model->kv_cache_v);

    delete model;
}
