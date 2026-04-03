#pragma once
#include "../gpu_compat.h"
#include <string>
#include <vector>
#include <cstdint>

// Model hyperparameters (from config.json)
struct ModelConfig {
    int n_vocab;        // vocabulary size (32000 for Mistral)
    int n_embd;         // embedding dimension (4096)
    int n_head;         // number of attention heads (32)
    int n_head_kv;      // number of KV heads (8 for GQA)
    int n_layer;        // number of transformer layers (32)
    int n_ff;           // feed-forward hidden dim (14336)
    int n_ctx;          // max context length
    float rope_theta;   // RoPE base frequency
    float rms_norm_eps; // RMSNorm epsilon

    // Gemma 4 extensions (all zero = legacy Llama/Mistral path)
    int head_dim_sliding;    // 256, or 0 = use n_embd/n_head (legacy)
    int head_dim_full;       // 512, or 0
    int sliding_window;      // 512, or 0 = disabled
    float logit_softcap;     // 30.0, or 0 = disabled
    int num_kv_shared;       // 18, or 0
    float rope_theta_full;   // 1000000
    float partial_rotary;    // 0.25, or 1.0 default
    int activation_type;     // 0=silu, 1=gelu_tanh
    int tie_embeddings;      // 0 or 1
    uint8_t layer_types[64]; // 0=sliding, 1=full (max 64 layers)
};

// Compressed weight tensor on GPU (CSR escape format + fused escape table)
struct CompressedWeight {
    int M, K;                     // weight dimensions [M, K]
    int32_t* packed;              // 12-bit packed indices on GPU
    int16_t* codebook;            // 4096-entry codebook on GPU
    // CSR escape patches (tiny — ~few KB per tensor)
    int32_t* row_offsets;         // [M+1] CSR row pointers
    int32_t* patch_cols;          // [num_patches] column indices
    int16_t* patch_correct;       // [num_patches] correct BF16 values
    int16_t* patch_wrong;         // [num_patches] wrong BF16 values
    int num_patches;
    int base_exp;                     // BaseExp for structured 12-bit decode
    // Fused escape table (built at load time from CSR)
    int32_t* escape_row_base;     // [M] absolute start of row's escapes (5.6 MB total)
    uint8_t* escape_counts;       // [M*256] optional: per-thread escape count (361 MB, max val 12)
    int16_t* escape_vals;         // [num_patches] correct BF16 values in thread-stride order
    // Split12 format: byte-aligned arrays (zero read amplification)
    uint8_t* split_sm;            // [M*K] sign+mantissa bytes (NULL if not loaded)
    uint8_t* split_gr;            // [M*K/2] nibble-packed groups
    // Sparse patch row list (NVIDIA: reduces patch grid from M to num_nonempty_rows)
    int32_t* patch_nonempty_rows; // [num_nonempty_rows] indices of rows with patches
    int num_nonempty_rows;
};

// One transformer layer
struct TransformerLayer {
    // Attention weights (compressed)
    CompressedWeight wq, wk, wv, wo;

    // MLP weights (compressed)
    CompressedWeight w_gate, w_up, w_down;

    // Norm weights (small, stored as FP32 on GPU)
    float* attn_norm;   // [n_embd]
    float* ffn_norm;    // [n_embd]

    // Gemma 4 extensions (nullptr for Llama/Mistral)
    float* q_norm;        // [head_dim] per-head Q norm
    float* k_norm;        // [head_dim] per-head K norm
    float* pre_ffn_norm;  // [n_embd]
    float* post_ffn_norm; // [n_embd]
    float layer_scalar;   // 1.0 default
    int head_dim;         // per-layer (256 or 512 for Gemma4, n_embd/n_head for legacy)
    int kv_dim;           // per-layer (n_kv_heads * head_dim)
    int kv_cache_layer;   // which layer's KV slot to use (-1 = own slot)
    bool is_full_attn;    // true for full attention layers
};

// Full model
struct Model {
    ModelConfig config;

    // Tensor parallelism
    int tp_rank;    // 0 or 1 (default 0)
    int tp_size;    // 1 (no TP) or 2

    // Embedding (BF16 on GPU — used for lookup, not matvec)
    int16_t* token_embd;  // [n_vocab, n_embd] as BF16

    // Transformer layers
    std::vector<TransformerLayer> layers;

    // Output
    float* output_norm;           // [n_embd] FP32
    CompressedWeight output_proj; // [n_vocab, n_embd] compressed

    // KV cache [n_layer, max_seq, n_head_kv, head_dim] as FP16
    int16_t* kv_cache_k;
    int16_t* kv_cache_v;
    int max_seq_len;

    // Per-layer KV cache pointers (Gemma4: shared layers point to donor's cache)
    // For legacy models, kv_k_ptrs[i] = kv_cache_k + i * max_seq * kv_dim
    std::vector<int16_t*> kv_k_ptrs;
    std::vector<int16_t*> kv_v_ptrs;
    int num_kv_slots;  // actual allocated KV slots (n_layer - num_kv_shared)
};

// Load model from safetensors directory
Model* load_model(const std::string& model_path, int device_id = 0, int tp_rank = 0, int tp_size = 1);

// Free model
void free_model(Model* model);
