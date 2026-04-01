#include "sampler.h"
#include "../gpu_compat.h"

extern "C" void launch_argmax(const float* logits, int* result, int n, hipStream_t stream);
extern "C" void launch_argmax_batch(const float* logits, int* results, int n_vocab, int batch_size, hipStream_t stream);

// Pre-allocated result buffer
static int* s_d_result = nullptr;
static int  s_d_result_size = 0;

void init_sampler(int max_batch) {
    if (max_batch <= s_d_result_size) return;
    if (s_d_result) hipFree(s_d_result);
    hipMalloc(&s_d_result, max_batch * sizeof(int));
    s_d_result_size = max_batch;
}

int sample_greedy(const float* logits_gpu, int n_vocab, hipStream_t stream) {
    if (!s_d_result) init_sampler(1);
    launch_argmax(logits_gpu, s_d_result, n_vocab, stream);
    int result;
    hipMemcpyAsync(&result, s_d_result, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    return result;
}

void sample_greedy_batch(const float* logits_gpu, int* tokens_out, int n_vocab, int batch_size, hipStream_t stream) {
    if (batch_size > s_d_result_size) init_sampler(batch_size);
    launch_argmax_batch(logits_gpu, s_d_result, n_vocab, batch_size, stream);
    hipMemcpyAsync(tokens_out, s_d_result, batch_size * sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
}
