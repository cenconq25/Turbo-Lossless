#include <cstdio>
#include <ctime>
#include <csignal>
#include <cstring>
#include <string>
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "sampler.h"

static volatile Model* g_model = nullptr;
static volatile InferenceState* g_state = nullptr;
static volatile Tokenizer* g_tok = nullptr;

static void signal_handler(int sig) {
    if (g_state) free_inference_state(const_cast<InferenceState*>(g_state));
    if (g_tok) free_tokenizer(const_cast<Tokenizer*>(g_tok));
    if (g_model) free_model(const_cast<Model*>(g_model));
    _exit(1);
}

static void run_prompt(InferenceState* state, Tokenizer* tok, const std::string& prompt,
                       int max_tokens, int batch_size) {
    // Reset positions for new prompt
    memset(state->positions, 0, batch_size * sizeof(int));

    auto prompt_tokens = tokenize(tok, prompt);

    printf("Generating...\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    auto output_tokens = generate(state, prompt_tokens, max_tokens);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double gen_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    int n_prompt = prompt_tokens.size();
    int n_gen = output_tokens.size();

    for (int tok_id : output_tokens) {
        printf("%s", detokenize_one(tok, tok_id).c_str());
        fflush(stdout);
    }
    printf("\n");

    printf("\n--- Stats ---\n");
    printf("Prompt tokens: %d\n", n_prompt);
    printf("Generated tokens: %d\n", n_gen);
    printf("Total time: %.2f s\n", gen_secs);
    if (n_gen > 0) {
        double prefill_est = gen_secs * n_prompt / (n_prompt + n_gen);
        double decode_est = gen_secs - prefill_est;
        printf("Decode speed: %.1f tok/s per user\n", n_gen / decode_est);
        if (batch_size > 1)
            printf("Total throughput: %.1f tok/s (%d users)\n",
                   batch_size * n_gen / decode_est, batch_size);
    }
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [prompt] [max_tokens] [batch_size]\n", argv[0]);
        fprintf(stderr, "       %s <model_path> --interactive [max_tokens]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];

    // Check for interactive mode
    bool interactive = false;
    std::string prompt;
    int max_tokens = 200;
    int batch_size = 1;

    if (argc >= 3 && std::string(argv[2]) == "--interactive") {
        interactive = true;
        if (argc > 3) max_tokens = atoi(argv[3]);
    } else if (argc >= 3) {
        prompt = argv[2];
        if (argc > 3) max_tokens = atoi(argv[3]);
        if (argc > 4) batch_size = atoi(argv[4]);
    } else {
        interactive = true;  // no prompt = interactive
    }

    if (!interactive) {
        printf("Turbo Lossless Engine\n");
        printf("Model: %s\n", model_path.c_str());
        printf("Prompt: %s\n", prompt.c_str());
        printf("Max tokens: %d\n\n", max_tokens);
    }

    printf("Loading model...\n");
    Model* model = load_model(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    g_model = model;

    Tokenizer* tok = load_tokenizer(model_path);
    if (!tok) { fprintf(stderr, "Failed to load tokenizer\n"); return 1; }
    g_tok = tok;

    InferenceState* state = create_inference_state(model, batch_size, model->max_seq_len);
    if (!state) { fprintf(stderr, "Failed to create inference state\n"); return 1; }
    g_state = state;

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    printf("Model loaded.\n");
    fflush(stdout);

    if (interactive) {
        // Interactive: read prompts from stdin, keep KV cache across turns
        int current_pos = 0;
        char line[4096];
        while (fgets(line, sizeof(line), stdin)) {
            // Strip trailing newline
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
            if (len == 0) continue;
            if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;

            auto tokens = tokenize(tok, std::string(line));
            // tokenize() always prepends BOS — only keep it on the first turn
            if (current_pos > 0 && !tokens.empty() && tokens[0] == tok->bos_id) {
                tokens.erase(tokens.begin());
            }

            printf("Generating...\n\n");
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            // Prefill: feed each prompt token, continuing from current_pos
            state->positions[0] = current_pos;
            for (int i = 0; i < (int)tokens.size(); i++) {
                forward(state, &tokens[i]);
                // forward_b1 auto-increments positions[0]
            }
            current_pos = state->positions[0];

            // Decode: generate new tokens
            int n_gen = 0;
            int n_vocab = model->config.n_vocab;
            int eos_id = tok->eos_id;

            // Find stop token IDs (tokens that start a new instruction turn)
            // These are model-specific but we check common ones
            std::vector<int> stop_ids;
            auto probe = [&](const std::string& s) {
                auto ids = tokenize(tok, s);
                if (!ids.empty()) stop_ids.push_back(ids.back());
            };
            probe("[INST]"); probe("[/INST]"); probe("</s>");

            clock_gettime(CLOCK_MONOTONIC, &t0);  // restart timer for decode only

            for (int t = 0; t < max_tokens; t++) {
                int next = sample_greedy(state->logits, n_vocab, state->stream);
                if (next == eos_id || next == 2 || next == 1) break;

                // Check stop tokens (model trying to start new turn)
                bool is_stop = false;
                for (int sid : stop_ids) {
                    if (next == sid) { is_stop = true; break; }
                }
                if (is_stop && n_gen > 2) break;

                printf("%s", detokenize_one(tok, next).c_str());
                fflush(stdout);

                forward(state, &next);
                current_pos = state->positions[0];
                n_gen++;
            }

            clock_gettime(CLOCK_MONOTONIC, &t1);  // stop timer before EOS overhead

            // Feed EOS/turn separator so model knows this turn ended
            {
                int eos_tok = tok->eos_id > 0 ? tok->eos_id : 2;
                state->positions[0] = current_pos;
                forward(state, &eos_tok);
                current_pos = state->positions[0];
            }
            double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

            printf("\n\n--- Stats ---\n");
            printf("Prompt tokens: %zu\n", tokens.size());
            printf("Generated tokens: %d\n", n_gen);
            printf("Context position: %d / %d\n", current_pos, model->max_seq_len);
            printf("Total time: %.2f s\n", secs);
            if (n_gen > 0) printf("Decode speed: %.1f tok/s\n", n_gen / secs);

            // Check context limit
            if (current_pos > model->max_seq_len - 100) {
                printf("Context nearly full (%d/%d). Resetting.\n", current_pos, model->max_seq_len);
                current_pos = 0;
            }

            printf("\n--- READY ---\n");
            fflush(stdout);
        }
    } else {
        // Single-shot mode (original behavior)
        printf("Prompt tokens: %zu\n", tokenize(tok, prompt).size());
        if (batch_size > 1) printf("Batch size: %d\n", batch_size);
        run_prompt(state, tok, prompt, max_tokens, batch_size);
    }

    g_state = nullptr;
    g_tok = nullptr;
    g_model = nullptr;
    free_inference_state(state);
    free_tokenizer(tok);
    free_model(model);
    return 0;
}
