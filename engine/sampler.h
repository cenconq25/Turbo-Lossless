#pragma once
#include <hip/hip_runtime.h>

// Pre-allocate sampler buffers
void init_sampler();

// Greedy sampling: argmax of logits
int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream);
