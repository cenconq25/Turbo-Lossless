#include "sampler.h"
#include <hip/hip_runtime.h>

extern "C" void launch_argmax(const float* logits, int* result, int n, hipStream_t stream);

int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream) {
    int* d_result;
    hipMalloc(&d_result, sizeof(int));
    launch_argmax(logits_gpu, d_result, n_vocab, stream);
    int result;
    hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_result);
    return result;
}
