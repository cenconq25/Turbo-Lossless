#include <cstdio>
#include <ctime>
#include <csignal>
#include <string>
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "sampler.h"

// Global pointers for signal handler GPU cleanup
static volatile Model* g_model = nullptr;
static volatile InferenceState* g_state = nullptr;
static volatile Tokenizer* g_tok = nullptr;

static void signal_handler(int sig) {
    if (g_state) free_inference_state(const_cast<InferenceState*>(g_state));
    if (g_tok) free_tokenizer(const_cast<Tokenizer*>(g_tok));
    if (g_model) free_model(const_cast<Model*>(g_model));
    _exit(1);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_path> <prompt> [max_tokens]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 50;
    int batch_size = argc > 4 ? atoi(argv[4]) : 1;

    printf("Turbo Lossless Engine\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n\n", max_tokens);

    // Load
    printf("Loading model...\n");
    Model* model = load_model(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    g_model = model;

    Tokenizer* tok = load_tokenizer(model_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }
    g_tok = tok;

    // Tokenize prompt
    auto prompt_tokens = tokenize(tok, prompt);
    printf("Prompt tokens: %zu\n", prompt_tokens.size());

    // Create inference state
    InferenceState* state = create_inference_state(model, batch_size, model->max_seq_len);
    if (batch_size > 1) printf("Batch size: %d\n", batch_size);
    if (!state) { fprintf(stderr, "Failed to create inference state\n"); return 1; }
    g_state = state;

    // Register signal handlers for graceful GPU cleanup on kill
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

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
    return 0;
}
