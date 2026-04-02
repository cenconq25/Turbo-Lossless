#include <cstdio>
#include <ctime>
#include <csignal>
#include <cstdlib>
#include <string>
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "sampler.h"
#include "multi_gpu.h"

// Global pointers for signal handler GPU cleanup
static volatile Model* g_model = nullptr;
static volatile InferenceState* g_state = nullptr;
static volatile Tokenizer* g_tok = nullptr;
static volatile TPState* g_tp = nullptr;

static void signal_handler(int sig) {
    if (g_tp && ((TPState*)g_tp)->tp_size > 1) {
        // Multi-GPU cleanup
        TPState* tp = const_cast<TPState*>((const TPState*)g_tp);
        for (int r = 0; r < tp->tp_size; r++) {
            if (tp->states[r]) free_inference_state(tp->states[r]);
            if (tp->models[r]) free_model(tp->models[r]);
        }
        free_tp(tp);
    } else {
        // Single-GPU cleanup (original path)
        if (g_state) free_inference_state(const_cast<InferenceState*>(g_state));
        if (g_model) free_model(const_cast<Model*>(g_model));
    }
    if (g_tok) free_tokenizer(const_cast<Tokenizer*>(g_tok));
    _exit(1);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_path> <prompt> [max_tokens] [batch_size]\n", argv[0]);
        fprintf(stderr, "  env TURBO_TP=2  to enable 2-GPU tensor parallelism\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 50;
    int batch_size = argc > 4 ? atoi(argv[4]) : 1;

    // Parse TP size from environment
    int tp_size = 1;
    const char* tp_env = getenv("TURBO_TP");
    if (tp_env) {
        tp_size = atoi(tp_env);
        if (tp_size != 1 && tp_size != 2) {
            fprintf(stderr, "TURBO_TP must be 1 or 2, got %s\n", tp_env);
            return 1;
        }
    }

    printf("Turbo Lossless Engine\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n", max_tokens);
    if (tp_size > 1) printf("Tensor Parallelism: TP=%d\n", tp_size);
    printf("\n");

    // Load tokenizer (shared, CPU-only)
    Tokenizer* tok = load_tokenizer(model_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }
    g_tok = tok;

    // Tokenize prompt
    auto prompt_tokens = tokenize(tok, prompt);
    printf("Prompt tokens: %zu\n", prompt_tokens.size());

    // Register signal handlers for graceful GPU cleanup on kill
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    if (tp_size > 1) {
        // =============================================
        // Multi-GPU tensor parallel path (TP=2)
        // =============================================
        int device_ids[2] = {0, 1};

        // Check if HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES restricts us
        // (NCCL uses physical device IDs after visibility filtering)
        printf("Initializing TP=%d on GPUs %d, %d...\n", tp_size, device_ids[0], device_ids[1]);

        TPState* tp = init_tp(tp_size, device_ids);
        if (!tp) { fprintf(stderr, "Failed to initialize tensor parallelism\n"); return 1; }
        g_tp = tp;

        // Load model shard on each GPU
        for (int rank = 0; rank < tp_size; rank++) {
            printf("Loading model shard %d/%d on GPU %d...\n", rank, tp_size, device_ids[rank]);
            GPU_CHECK(hipSetDevice(device_ids[rank]));

            tp->models[rank] = load_model(model_path, device_ids[rank], rank, tp_size);
            if (!tp->models[rank]) {
                fprintf(stderr, "Failed to load model on GPU %d\n", device_ids[rank]);
                return 1;
            }

            tp->states[rank] = create_inference_state(
                tp->models[rank], batch_size, tp->models[rank]->max_seq_len, tp_size);
            if (!tp->states[rank]) {
                fprintf(stderr, "Failed to create inference state on GPU %d\n", device_ids[rank]);
                return 1;
            }
            // Wire up TP pointers so forward pass can call all-reduce
            tp->states[rank]->tp = tp;
            tp->states[rank]->tp_rank = rank;
        }

        printf("All model shards loaded. Generating...\n\n");

        // TP generation: both GPUs run forward in lockstep (NCCL requires both ranks)
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        auto output_tokens = generate_tp(tp, prompt_tokens, max_tokens, device_ids);

        clock_gettime(CLOCK_MONOTONIC, &t1);

        double gen_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        int n_prompt = prompt_tokens.size();
        int n_gen = output_tokens.size();

        // Decode and print
        for (int tok_id : output_tokens) {
            printf("%s", detokenize_one(tok, tok_id).c_str());
            fflush(stdout);
        }
        printf("\n");

        // Stats
        printf("\n--- Stats (TP=%d) ---\n", tp_size);
        printf("Prompt tokens: %d\n", n_prompt);
        printf("Generated tokens: %d\n", n_gen);
        printf("Total time: %.2f s\n", gen_secs);
        if (n_gen > 0) {
            double prefill_est = gen_secs * n_prompt / (n_prompt + n_gen);
            double decode_est = gen_secs - prefill_est;
            printf("Decode speed: %.1f tok/s per user\n", n_gen / decode_est);
            if (batch_size > 1) {
                printf("Total throughput: %.1f tok/s (%d users)\n",
                       batch_size * n_gen / decode_est, batch_size);
            }
        }

        // Cleanup
        g_tp = nullptr;
        g_tok = nullptr;
        for (int rank = 0; rank < tp_size; rank++) {
            GPU_CHECK(hipSetDevice(device_ids[rank]));
            free_inference_state(tp->states[rank]);
            free_model(tp->models[rank]);
        }
        free_tp(tp);
        free_tokenizer(tok);

    } else {
        // =============================================
        // Single-GPU path (original, unchanged logic)
        // =============================================
        printf("Loading model...\n");
        Model* model = load_model(model_path);
        if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
        g_model = model;

        // Create inference state
        InferenceState* state = create_inference_state(model, batch_size, model->max_seq_len);
        if (batch_size > 1) printf("Batch size: %d\n", batch_size);
        if (!state) { fprintf(stderr, "Failed to create inference state\n"); return 1; }
        g_state = state;

        // Generate with timing
        printf("Generating...\n\n");

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        auto output_tokens = generate(state, prompt_tokens, max_tokens);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double gen_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        int n_prompt = prompt_tokens.size();
        int n_gen = output_tokens.size();

        // Decode and print
        for (int tok_id : output_tokens) {
            printf("%s", detokenize_one(tok, tok_id).c_str());
            fflush(stdout);
        }
        printf("\n");

        // Stats
        printf("\n--- Stats ---\n");
        printf("Prompt tokens: %d\n", n_prompt);
        printf("Generated tokens: %d\n", n_gen);
        printf("Total time: %.2f s\n", gen_secs);
        if (n_gen > 0) {
            double prefill_est = gen_secs * n_prompt / (n_prompt + n_gen);
            double decode_est = gen_secs - prefill_est;
            printf("Decode speed: %.1f tok/s per user\n", n_gen / decode_est);
            if (batch_size > 1) {
                printf("Total throughput: %.1f tok/s (%d users)\n",
                       batch_size * n_gen / decode_est, batch_size);
            }
        }

        // Cleanup (clear globals first to prevent signal handler double-free)
        g_state = nullptr;
        g_tok = nullptr;
        g_model = nullptr;
        free_inference_state(state);
        free_tokenizer(tok);
        free_model(model);
    }

    return 0;
}
