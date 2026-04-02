#include "multi_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// NCCL is required for multi-GPU (TP>1) on NVIDIA.
// Define TURBO_NCCL at compile time to enable: -DTURBO_NCCL
// Without it, init_tp() will reject tp_size>1 at runtime.
#if defined(TURBO_NVIDIA) && defined(TURBO_NCCL)
#include <nccl.h>

#define NCCL_CHECK(call) do { \
    ncclResult_t _nccl_err = (call); \
    if (_nccl_err != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                ncclGetErrorString(_nccl_err), (int)_nccl_err); \
        exit(1); \
    } \
} while(0)
#define HAS_NCCL 1
#else
#define HAS_NCCL 0
#endif

TPState* init_tp(int tp_size, int* device_ids) {
    if (tp_size < 1 || tp_size > 2) {
        fprintf(stderr, "TP size must be 1 or 2, got %d\n", tp_size);
        return nullptr;
    }

    TPState* tp = new TPState();
    memset(tp, 0, sizeof(TPState));
    tp->tp_size = tp_size;

    for (int i = 0; i < tp_size; i++) {
        tp->device_ids[i] = device_ids[i];
    }

    if (tp_size == 1) {
        // No NCCL needed for single GPU
        return tp;
    }

#if HAS_NCCL
    // Initialize NCCL communicators (single-process multi-GPU)
    printf("Initializing NCCL for TP=%d (devices %d, %d)...\n",
           tp_size, device_ids[0], device_ids[1]);

    ncclComm_t comms[2];
    NCCL_CHECK(ncclCommInitAll(comms, tp_size, device_ids));

    for (int i = 0; i < tp_size; i++) {
        tp->nccl_comms[i] = (void*)comms[i];
    }

    // Enable peer access between GPUs (best-effort, not all topologies support it)
    for (int i = 0; i < tp_size; i++) {
        GPU_CHECK(hipSetDevice(device_ids[i]));
        for (int j = 0; j < tp_size; j++) {
            if (i == j) continue;
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]);
            if (can_access) {
                // Ignore error if already enabled
                cudaDeviceEnablePeerAccess(device_ids[j], 0);
                printf("  GPU %d -> GPU %d: peer access enabled\n", device_ids[i], device_ids[j]);
            } else {
                printf("  GPU %d -> GPU %d: peer access not available (will use staging)\n",
                       device_ids[i], device_ids[j]);
            }
        }
    }

    // Allocate argmax scratch buffers on each GPU: {max_val, max_idx_as_float}
    for (int i = 0; i < tp_size; i++) {
        GPU_CHECK(hipSetDevice(device_ids[i]));
        GPU_CHECK(hipMalloc(&tp->argmax_scratch[i], 2 * sizeof(float)));
    }

    printf("NCCL TP=%d initialized successfully\n", tp_size);
#else
    fprintf(stderr, "Multi-GPU (TP>1) requires NCCL. Compile with -DTURBO_NCCL and link -lnccl.\n");
    delete tp;
    return nullptr;
#endif

    return tp;
}

void tp_allreduce_sum(float* buf, int count, TPState* tp, int rank, void* stream, int /*buf_id*/) {
    if (tp->tp_size <= 1) return;  // no-op for single GPU

#if HAS_NCCL
    GPU_CHECK(hipSetDevice(tp->device_ids[rank]));
    ncclComm_t comm = (ncclComm_t)tp->nccl_comms[rank];
    NCCL_CHECK(ncclAllReduce(
        (const void*)buf, (void*)buf, (size_t)count,
        ncclFloat, ncclSum, comm,
        (cudaStream_t)stream
    ));
#endif
}

// Batch all-reduce: call for ALL ranks at once (avoids deadlock from single thread)
void tp_allreduce_sum_all(float** bufs, int count, TPState* tp) {
    if (tp->tp_size <= 1) return;

#if HAS_NCCL
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < tp->tp_size; r++) {
        ncclComm_t comm = (ncclComm_t)tp->nccl_comms[r];
        NCCL_CHECK(ncclAllReduce(
            (const void*)bufs[r], (void*)bufs[r], (size_t)count,
            ncclFloat, ncclSum, comm,
            (cudaStream_t)tp->states[r]->stream
        ));
    }
    NCCL_CHECK(ncclGroupEnd());
#endif
}

int tp_distributed_argmax(float* local_logits, int local_vocab, int total_vocab,
                          TPState* tp, int rank, void* stream) {
    // Single GPU: CPU-side argmax after sync
    if (tp->tp_size <= 1) {
        GPU_CHECK(hipSetDevice(tp->device_ids[rank]));
        GPU_CHECK(hipStreamSynchronize((hipStream_t)stream));

        std::vector<float> host_logits(local_vocab);
        GPU_CHECK(hipMemcpy(host_logits.data(), local_logits,
                            local_vocab * sizeof(float), hipMemcpyDeviceToHost));

        int best = 0;
        float best_val = host_logits[0];
        for (int i = 1; i < local_vocab; i++) {
            if (host_logits[i] > best_val) {
                best_val = host_logits[i];
                best = i;
            }
        }
        return best;
    }

#if HAS_NCCL
    // Multi-GPU: each rank finds local max, then exchange via host.
    // Step 1: local argmax on CPU (simple, correct -- optimize later if needed)
    GPU_CHECK(hipSetDevice(tp->device_ids[rank]));
    GPU_CHECK(hipStreamSynchronize((hipStream_t)stream));

    std::vector<float> host_logits(local_vocab);
    GPU_CHECK(hipMemcpy(host_logits.data(), local_logits,
                        local_vocab * sizeof(float), hipMemcpyDeviceToHost));

    int local_best = 0;
    float local_best_val = host_logits[0];
    for (int i = 1; i < local_vocab; i++) {
        if (host_logits[i] > local_best_val) {
            local_best_val = host_logits[i];
            local_best = i;
        }
    }

    // Step 2: Pack {val, global_index} into scratch buffer on GPU
    int global_idx = rank * local_vocab + local_best;
    float scratch_host[2] = { local_best_val, (float)global_idx };
    GPU_CHECK(hipMemcpy(tp->argmax_scratch[rank], scratch_host,
                        2 * sizeof(float), hipMemcpyHostToDevice));

    // Step 3: With only 2 GPUs, gather both results on host and compare.
    // (AllReduce is not suitable for argmax; this is simple and correct.)
    float all_results[2][2];  // [rank][{val, idx}]
    for (int r = 0; r < tp->tp_size; r++) {
        GPU_CHECK(hipSetDevice(tp->device_ids[r]));
        GPU_CHECK(hipMemcpy(all_results[r], tp->argmax_scratch[r],
                            2 * sizeof(float), hipMemcpyDeviceToHost));
    }

    // Find global best
    int winner = 0;
    for (int r = 1; r < tp->tp_size; r++) {
        if (all_results[r][0] > all_results[winner][0]) {
            winner = r;
        }
    }

    return (int)all_results[winner][1];
#else
    return 0;  // unreachable: init_tp fails for TP>1 without NCCL
#endif
}

void free_tp(TPState* tp) {
    if (!tp) return;

    for (int i = 0; i < tp->tp_size; i++) {
        // Free scratch buffers
        if (tp->argmax_scratch[i]) {
            GPU_CHECK(hipSetDevice(tp->device_ids[i]));
            hipFree(tp->argmax_scratch[i]);
            tp->argmax_scratch[i] = nullptr;
        }

#if HAS_NCCL
        // Destroy NCCL communicators
        if (tp->nccl_comms[i]) {
            ncclCommDestroy((ncclComm_t)tp->nccl_comms[i]);
            tp->nccl_comms[i] = nullptr;
        }
#endif
    }

    delete tp;
}
