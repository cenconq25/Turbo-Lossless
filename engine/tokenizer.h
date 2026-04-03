#pragma once
#include <string>
#include <vector>

// Special tokens loaded from tokenizer.json (e.g., [INST], [/INST])
struct SpecialToken {
    std::string text;
    int id;
};

struct Tokenizer {
    void* sp_model;    // sentencepiece or HFTokenizer handle
    int bos_id, eos_id;
    int vocab_size;
    int type;          // 0=sentencepiece, 1=HF BPE
    std::vector<SpecialToken> special_tokens;  // sorted longest-first for greedy match
};

Tokenizer* load_tokenizer(const std::string& model_path);
void free_tokenizer(Tokenizer* tok);
std::vector<int> tokenize(Tokenizer* tok, const std::string& text);
std::string detokenize(Tokenizer* tok, const std::vector<int>& tokens);
std::string detokenize_one(Tokenizer* tok, int token);
