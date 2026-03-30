#include "tokenizer.h"
#include <sentencepiece_processor.h>
#include <cstdio>

Tokenizer* load_tokenizer(const std::string& model_path) {
    auto* tok = new Tokenizer();
    auto* sp = new sentencepiece::SentencePieceProcessor();

    std::string sp_path = model_path + "/tokenizer.model";
    auto status = sp->Load(sp_path);
    if (!status.ok()) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", sp_path.c_str());
        delete sp;
        delete tok;
        return nullptr;
    }

    tok->sp_model = sp;
    tok->bos_id = sp->bos_id();
    tok->eos_id = sp->eos_id();

    printf("  Tokenizer: vocab=%d bos=%d eos=%d\n", sp->GetPieceSize(), tok->bos_id, tok->eos_id);
    return tok;
}

void free_tokenizer(Tokenizer* tok) {
    if (!tok) return;
    delete (sentencepiece::SentencePieceProcessor*)tok->sp_model;
    delete tok;
}

std::vector<int> tokenize(Tokenizer* tok, const std::string& text) {
    auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
    std::vector<int> ids;
    sp->Encode(text, &ids);
    // Prepend BOS
    ids.insert(ids.begin(), tok->bos_id);
    return ids;
}

std::string detokenize(Tokenizer* tok, const std::vector<int>& tokens) {
    auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
    std::string text;
    sp->Decode(tokens, &text);
    return text;
}

std::string detokenize_one(Tokenizer* tok, int token) {
    auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
    return sp->IdToPiece(token);
}
