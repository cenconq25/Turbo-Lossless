#pragma once
#include <hip/hip_runtime.h>

// Pre-allocate sampler buffers
void init_sampler(int max_batch = 1);

// Greedy sampling: argmax of logits
int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream);

// Batched greedy: one argmax launch for all sequences, one sync
void sample_greedy_batch(const float* logits_gpu, int* tokens_out, int n_vocab, int batch_size, hipStream_t stream);
