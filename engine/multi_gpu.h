#pragma once
#include "model.h"
#include "inference.h"

struct TPState {
    int tp_size;               // 1 (disabled) or 2
    int device_ids[2];         // CUDA device IDs

    // NCCL communicators per device (void* to avoid nccl.h dependency in header)
    void* nccl_comms[2];

    Model* models[2];          // one model shard per GPU
    InferenceState* states[2]; // one inference state per GPU

    // Scratch buffer for distributed argmax (one per GPU)
    float* argmax_scratch[2];  // [2] floats: {max_val, max_idx}
};

// Initialize tensor parallelism: creates NCCL comms, enables peer access.
// device_ids: array of tp_size CUDA device IDs.
// Returns NULL on failure.
TPState* init_tp(int tp_size, int* device_ids);

// All-reduce sum across GPUs (in-place). buf is on device `rank`.
// Synchronized on the given stream.
void tp_allreduce_sum(float* buf, int count, TPState* tp, int rank, void* stream);

// Distributed argmax: each GPU has local_logits[local_vocab].
// Returns the global best token ID across all shards.
int tp_distributed_argmax(float* local_logits, int local_vocab, int total_vocab,
                          TPState* tp, int rank, void* stream);

// Cleanup: destroy NCCL comms, free scratch buffers.
void free_tp(TPState* tp);
