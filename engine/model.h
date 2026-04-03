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
};

// Full model
struct Model {
    ModelConfig config;

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
};

// Load model from safetensors directory
Model* load_model(const std::string& model_path, int device_id = 0);

// Free model
void free_model(Model* model);
