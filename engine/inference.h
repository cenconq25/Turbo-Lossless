#pragma once
#include "model.h"

// Inference state for one or more concurrent sequences
struct InferenceState {
    Model* model;
    int batch_size;       // number of concurrent sequences (1-8)
    int* positions;       // current position per sequence [batch_size]

    // Scratch buffers on GPU (shared across layers)
    float* hidden;        // [batch_size, n_embd]
    float* hidden2;       // [batch_size, n_embd] (second buffer for residual)
    float* attn_out;      // [batch_size, n_embd]
    float* q_buf;         // [batch_size, n_head * head_dim]
    float* k_buf;         // [batch_size, n_head_kv * head_dim]
    float* v_buf;         // [batch_size, n_head_kv * head_dim]
    float* ffn_gate;      // [batch_size, n_ff]
    float* ffn_up;        // [batch_size, n_ff]
    float* ffn_down;      // [batch_size, n_embd]
    float* logits;        // [batch_size, n_vocab]
    float* attn_scores_buf; // [max_seq_len] temp for attention scores
    int* d_positions;       // [batch_size] positions on GPU
    int* d_tokens;          // [batch_size] token IDs on GPU

    hipStream_t stream;
    hipStream_t stream2;    // secondary stream for concurrent kernel execution
    hipEvent_t sync_event;  // lightweight cross-stream sync

    // BF16 activation buffers for matvec (halves L2 activation bandwidth)
    int16_t* bf16_act;      // [batch_size * max(n_embd, n_ff)] BF16 temp
    int16_t* bf16_act2;     // second buffer for ffn (gate input differs from attn input)
};

// Create inference state
InferenceState* create_inference_state(Model* model, int batch_size, int max_seq_len);
void free_inference_state(InferenceState* state);

// Run one forward pass: token_ids → logits
// token_ids: [batch_size] input tokens for this step
// Returns logits: [batch_size, n_vocab]
void forward(InferenceState* state, const int* token_ids);

// Generate tokens
// prompt_tokens: input token IDs
// max_tokens: max tokens to generate
// Returns: generated token IDs
std::vector<int> generate(InferenceState* state, const std::vector<int>& prompt_tokens, int max_tokens);
