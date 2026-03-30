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
    df.close();

    // Read and upload packed data
    auto packed = read_file(dir + "/" + prefix + ".packed.bin");
    if (packed.empty()) return false;
    w.packed = upload_gpu<int32_t>(packed.data(), packed.size() / sizeof(int32_t));

    // Codebook
    auto cb = read_file(dir + "/" + prefix + ".codebook.bin");
    if (cb.empty()) return false;
    w.codebook = upload_gpu<int16_t>(cb.data(), cb.size() / sizeof(int16_t));

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
    m->max_seq_len = 2048;  // default, can be changed
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

void free_model(Model* model) {
    if (!model) return;
    // TODO: free all GPU allocations
    delete model;
}
