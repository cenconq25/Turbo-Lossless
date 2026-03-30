#pragma once
#include <hip/hip_runtime.h>

// Greedy sampling: argmax of logits
int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream);
