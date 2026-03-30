#include "sampler.h"
#include <hip/hip_runtime.h>

extern "C" void launch_argmax(const float* logits, int* result, int n, hipStream_t stream);

// Pre-allocated result buffer (set during inference state creation)
static int* s_d_result = nullptr;

void init_sampler() {
    if (!s_d_result) hipMalloc(&s_d_result, sizeof(int));
}

int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream) {
    if (!s_d_result) init_sampler();
    launch_argmax(logits_gpu, s_d_result, n_vocab, stream);
    int result;
    hipMemcpyAsync(&result, s_d_result, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    return result;
}
