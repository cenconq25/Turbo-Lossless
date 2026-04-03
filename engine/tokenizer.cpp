#include "tokenizer.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cstring>

// ============================================================
// Sentencepiece backend
// ============================================================
#include <sentencepiece_processor.h>

static bool load_sentencepiece(Tokenizer* tok, const std::string& path) {
    auto* sp = new sentencepiece::SentencePieceProcessor();
    auto status = sp->Load(path);
    if (!status.ok()) {
        delete sp;
        return false;
    }
    tok->sp_model = sp;
    tok->bos_id = sp->bos_id();
    tok->eos_id = sp->eos_id();
    tok->vocab_size = sp->GetPieceSize();
    tok->type = 0;  // sentencepiece
    printf("  Tokenizer: sentencepiece, vocab=%d bos=%d eos=%d\n", tok->vocab_size, tok->bos_id, tok->eos_id);
    return true;
}

// ============================================================
// HuggingFace tokenizer.json backend (BPE)
// Minimal implementation: vocab lookup for decode, Python fallback for encode
// ============================================================

struct HFTokenizer {
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;
    std::string byte_encoder[256];  // byte → unicode string mapping (GPT-2 style)
    int bos_id, eos_id;
    int tok_type;  // 1=GPT-2 byte-level BPE, 2=sentencepiece-style BPE (Gemma)
};

// Simple JSON string extraction (no external lib needed)
static std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find("\"", pos + 1);
    // Handle escaped quotes
    while (end != std::string::npos && json[end - 1] == '\\') end = json.find("\"", end + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static int extract_json_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return -1;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return -1;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return atoi(json.c_str() + pos);
}

// Decode UTF-8 escape sequences like \u0120 → space prefix (Ġ)
static std::string unescape_token(const std::string& s) {
    std::string result;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i+1] == 'n') { result += '\n'; i++; }
            else if (s[i+1] == 't') { result += '\t'; i++; }
            else if (s[i+1] == '\\') { result += '\\'; i++; }
            else if (s[i+1] == '"') { result += '"'; i++; }
            else if (s[i+1] == 'u' && i + 5 < s.size()) {
                char hex[5] = {s[i+2], s[i+3], s[i+4], s[i+5], 0};
                unsigned cp = strtoul(hex, nullptr, 16);
                if (cp < 0x80) result += (char)cp;
                else if (cp < 0x800) { result += (char)(0xC0 | (cp >> 6)); result += (char)(0x80 | (cp & 0x3F)); }
                else { result += (char)(0xE0 | (cp >> 12)); result += (char)(0x80 | ((cp >> 6) & 0x3F)); result += (char)(0x80 | (cp & 0x3F)); }
                i += 5;
            } else {
                result += s[i];
            }
        } else {
            result += s[i];
        }
    }
    return result;
}

static bool load_hf_tokenizer(Tokenizer* tok, const std::string& model_dir) {
    // Prefer pre-extracted binary format (vocab.bin + merges.bin) for speed and correctness
    std::string vocab_path = model_dir + "/vocab.bin";
    std::string merges_path = model_dir + "/merges.bin";
    std::ifstream vf(vocab_path, std::ios::binary);
    std::ifstream mf(merges_path, std::ios::binary);
    if (vf && mf) {
        auto* hf = new HFTokenizer();
        int n_vocab, bos, eos;
        vf.read((char*)&n_vocab, 4); vf.read((char*)&bos, 4); vf.read((char*)&eos, 4);
        hf->bos_id = bos; hf->eos_id = eos;

        // Try to read tok_type (new field, may not exist in older vocab.bin files)
        int tok_type_val = 1;  // default: GPT-2 byte-level BPE
        auto pos_before = vf.tellg();
        // Peek: if next 4 bytes look like a small int (1 or 2), it's tok_type
        // Otherwise it's the first token length (uint16), so we rewind
        int32_t maybe_type;
        if (vf.read((char*)&maybe_type, 4)) {
            if (maybe_type == 1 || maybe_type == 2) {
                tok_type_val = maybe_type;
            } else {
                // Old format without tok_type — rewind
                vf.seekg(pos_before);
            }
        }
        hf->tok_type = tok_type_val;

        for (int i = 0; i < n_vocab; i++) {
            uint16_t len; vf.read((char*)&len, 2);
            std::string t(len, 0); vf.read(&t[0], len);
            hf->token_to_id[t] = i;
            hf->id_to_token[i] = t;
        }
        int n_merges; mf.read((char*)&n_merges, 4);
        hf->merges.reserve(n_merges);
        for (int i = 0; i < n_merges; i++) {
            uint16_t la, lb; mf.read((char*)&la, 2); mf.read((char*)&lb, 2);
            std::string a(la, 0), b(lb, 0);
            mf.read(&a[0], la); mf.read(&b[0], lb);
            hf->merges.push_back({a, b});
        }
        // Load byte encoder mapping (only used for GPT-2 style)
        if (tok_type_val == 1) {
            std::string be_path = model_dir + "/byte_encoder.bin";
            std::ifstream bef(be_path, std::ios::binary);
            if (bef) {
                for (int i = 0; i < 256; i++) {
                    uint8_t len; bef.read((char*)&len, 1);
                    std::string s(len, 0); bef.read(&s[0], len);
                    hf->byte_encoder[i] = s;
                }
            } else {
                for (int i = 0; i < 256; i++) hf->byte_encoder[i] = std::string(1, (char)i);
            }
        }

        tok->sp_model = hf; tok->bos_id = bos; tok->eos_id = eos;
        tok->vocab_size = n_vocab; tok->type = 1;
        const char* style = (tok_type_val == 2) ? "SP-BPE" : "GPT2-BPE";
        printf("  Tokenizer: HF %s (binary), vocab=%d bos=%d eos=%d merges=%zu\n",
               style, n_vocab, bos, eos, hf->merges.size());
        return true;
    }

    // Fallback: parse tokenizer.json directly
    std::string path = model_dir + "/tokenizer.json";
    std::ifstream f(path);
    if (!f) return false;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    auto* hf = new HFTokenizer();

    // Parse vocab from the "model" → "vocab" section
    // Format: {"token": id, ...}
    size_t vocab_start = json.find("\"vocab\"");
    if (vocab_start == std::string::npos) { delete hf; return false; }
    vocab_start = json.find("{", vocab_start + 7);
    if (vocab_start == std::string::npos) { delete hf; return false; }

    size_t pos = vocab_start + 1;
    int max_id = 0;
    while (pos < json.size()) {
        // Find next key
        size_t key_start = json.find("\"", pos);
        if (key_start == std::string::npos || json[key_start - 1] == '}') break;
        size_t key_end = json.find("\"", key_start + 1);
        while (key_end != std::string::npos && json[key_end - 1] == '\\')
            key_end = json.find("\"", key_end + 1);
        if (key_end == std::string::npos) break;

        std::string token = json.substr(key_start + 1, key_end - key_start - 1);
        token = unescape_token(token);

        // Find value
        size_t colon = json.find(":", key_end);
        if (colon == std::string::npos) break;
        int id = atoi(json.c_str() + colon + 1);

        hf->token_to_id[token] = id;
        hf->id_to_token[id] = token;
        if (id > max_id) max_id = id;

        // Move to next entry
        pos = json.find(",", colon);
        if (pos == std::string::npos || json.find("}", colon) < pos) break;
        pos++;
    }

    // Detect tokenizer style: check for sentencepiece-style ▁ tokens
    int sp_count = 0;
    for (auto& [k, v] : hf->token_to_id) {
        if (k.find("\xe2\x96\x81") != std::string::npos) sp_count++;
        if (sp_count > 100) break;
    }
    bool is_sp_style = (sp_count > 100);
    hf->tok_type = is_sp_style ? 2 : 1;

    // Find BOS/EOS from added_tokens
    hf->bos_id = -1;
    hf->eos_id = -1;

    // Check added_tokens for bos/eos
    size_t added = json.find("\"added_tokens\"");
    if (added != std::string::npos) {
        // Llama 3: <|begin_of_text|> / <|end_of_text|>
        size_t bos_pos = json.find("begin_of_text", added);
        if (bos_pos != std::string::npos) {
            size_t id_pos = json.rfind("\"id\"", bos_pos);
            if (id_pos != std::string::npos && id_pos > added)
                hf->bos_id = extract_json_int(json.substr(id_pos - 1), "id");
        }
        size_t eos_pos = json.find("eot_id", added);
        if (eos_pos != std::string::npos) {
            size_t id_pos = json.rfind("\"id\"", eos_pos);
            if (id_pos != std::string::npos && id_pos > added)
                hf->eos_id = extract_json_int(json.substr(id_pos - 1), "id");
        }
        // Gemma: <bos> / <eos>
        if (hf->bos_id == -1) {
            size_t bp = json.find("\"<bos>\"", added);
            if (bp != std::string::npos) {
                size_t id_pos = json.rfind("\"id\"", bp);
                if (id_pos != std::string::npos && id_pos > added)
                    hf->bos_id = extract_json_int(json.substr(id_pos - 1), "id");
            }
        }
        if (hf->eos_id == -1) {
            size_t ep = json.find("\"<eos>\"", added);
            if (ep != std::string::npos) {
                size_t id_pos = json.rfind("\"id\"", ep);
                if (id_pos != std::string::npos && id_pos > added)
                    hf->eos_id = extract_json_int(json.substr(id_pos - 1), "id");
            }
        }
    }

    // Fallback defaults
    if (hf->bos_id == -1) hf->bos_id = is_sp_style ? 2 : 128000;
    if (hf->eos_id == -1) hf->eos_id = is_sp_style ? 1 : 128009;

    // Parse merges for BPE encode
    size_t merges_start = json.find("\"merges\"");
    if (merges_start != std::string::npos) {
        merges_start = json.find("[", merges_start);
        if (merges_start != std::string::npos) {
            pos = merges_start + 1;
            while (pos < json.size()) {
                size_t ms = json.find("\"", pos);
                if (ms == std::string::npos) break;
                size_t me = json.find("\"", ms + 1);
                if (me == std::string::npos) break;
                std::string merge = json.substr(ms + 1, me - ms - 1);
                size_t sp = merge.find(" ");
                if (sp != std::string::npos)
                    hf->merges.push_back({merge.substr(0, sp), merge.substr(sp + 1)});
                pos = me + 1;
                if (json.find("]", pos) < json.find("\"", pos)) break;
            }
        }
    }

    tok->sp_model = hf;
    tok->bos_id = hf->bos_id;
    tok->eos_id = hf->eos_id;
    tok->vocab_size = max_id + 1;
    tok->type = 1;  // HF BPE
    printf("  Tokenizer: HF BPE, vocab=%d bos=%d eos=%d merges=%zu\n",
           tok->vocab_size, tok->bos_id, tok->eos_id, hf->merges.size());
    return true;
}

// Split a UTF-8 string into individual unicode characters
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int len = 1;
        if      ((c & 0x80) == 0)    len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) len = 1;  // safety
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// BPE encode: GPT-2 byte-level BPE (Llama 3, GPT-style)
static std::vector<int> bpe_encode_gpt2(HFTokenizer* hf, const std::string& text) {
    // Byte-level: map each byte through the GPT-2/Llama byte encoder
    // This maps bytes to unicode characters (e.g., 0x20→Ġ, 0x57→W)
    std::vector<std::string> tokens;
    for (unsigned char c : text) {
        std::string mapped = hf->byte_encoder[c];
        tokens.push_back(mapped);
    }

    // Apply BPE merges
    for (auto& [left, right] : hf->merges) {
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == left && tokens[i + 1] == right) {
                tokens[i] = left + right;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                i++;
            }
        }
    }

    // Convert to IDs
    std::vector<int> ids;
    for (auto& t : tokens) {
        auto it = hf->token_to_id.find(t);
        if (it != hf->token_to_id.end())
            ids.push_back(it->second);
        else
            ids.push_back(0);  // unknown
    }
    return ids;
}

// BPE encode: sentencepiece-style BPE (Gemma)
// Normalizer: replace ' ' with ▁ (U+2581)
// Pre-tokenizer: split on ' ' with MergedWithPrevious behavior
// Byte fallback: unknown bytes → <0xNN> tokens
static std::vector<int> bpe_encode_sp(HFTokenizer* hf, const std::string& text) {
    // Step 1: Normalize — replace spaces with ▁
    std::string normalized;
    // Gemma pre-tokenizer: split on space, merge with previous
    // The normalizer replaces " " with "▁", then pre-tokenizer splits on " "
    // In practice for input text: prepend ▁ to start, replace all spaces with ▁
    normalized = "\xe2\x96\x81";  // ▁ prefix (sentencepiece convention)
    for (char c : text) {
        if (c == ' ')
            normalized += "\xe2\x96\x81";  // ▁
        else
            normalized += c;
    }

    // Step 2: Split into unicode characters
    std::vector<std::string> tokens = utf8_chars(normalized);

    // Step 3: Apply BPE merges
    for (auto& [left, right] : hf->merges) {
        for (size_t i = 0; i + 1 < tokens.size(); ) {
            if (tokens[i] == left && tokens[i + 1] == right) {
                tokens[i] = left + right;
                tokens.erase(tokens.begin() + i + 1);
            } else {
                i++;
            }
        }
    }

    // Step 4: Convert to IDs, with byte fallback for unknown tokens
    std::vector<int> ids;
    for (auto& t : tokens) {
        auto it = hf->token_to_id.find(t);
        if (it != hf->token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback: encode each byte as <0xNN>
            for (unsigned char c : t) {
                char buf[8];
                snprintf(buf, sizeof(buf), "<0x%02X>", c);
                auto bit = hf->token_to_id.find(buf);
                if (bit != hf->token_to_id.end())
                    ids.push_back(bit->second);
                else
                    ids.push_back(3);  // <unk>
            }
        }
    }
    return ids;
}

static std::vector<int> bpe_encode(HFTokenizer* hf, const std::string& text) {
    if (hf->tok_type == 2)
        return bpe_encode_sp(hf, text);
    return bpe_encode_gpt2(hf, text);
}

// Reverse byte-level BPE encoding for display
static std::string bpe_token_to_text(HFTokenizer* hf, const std::string& token) {
    // Build reverse byte map (unicode char → original byte) on first call
    static std::unordered_map<std::string, uint8_t> byte_decoder;
    if (byte_decoder.empty()) {
        for (int i = 0; i < 256; i++)
            if (!hf->byte_encoder[i].empty())
                byte_decoder[hf->byte_encoder[i]] = (uint8_t)i;
    }
    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        // Try matching multi-byte UTF-8 sequences first
        bool found = false;
        for (int len = 4; len >= 1; len--) {
            if (i + len <= token.size()) {
                std::string sub = token.substr(i, len);
                auto it = byte_decoder.find(sub);
                if (it != byte_decoder.end()) {
                    result += (char)it->second;
                    i += len;
                    found = true;
                    break;
                }
            }
        }
        if (!found) { result += token[i]; i++; }
    }
    return result;
}

// Decode a sentencepiece-style BPE token to text
// Handles ▁ → space and <0xNN> → raw byte
static std::string sp_bpe_token_to_text(const std::string& token) {
    // Handle <0xNN> byte fallback tokens
    if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
        char hex[3] = {token[3], token[4], 0};
        unsigned byte_val = strtoul(hex, nullptr, 16);
        return std::string(1, (char)byte_val);
    }
    // Replace ▁ (U+2581, 3 bytes: E2 96 81) with space
    std::string result;
    for (size_t i = 0; i < token.size(); ) {
        if (i + 2 < token.size() &&
            (unsigned char)token[i] == 0xE2 &&
            (unsigned char)token[i+1] == 0x96 &&
            (unsigned char)token[i+2] == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += token[i];
            i++;
        }
    }
    return result;
}

// ============================================================
// Public API — auto-detect tokenizer type
// ============================================================

Tokenizer* load_tokenizer(const std::string& model_path) {
    auto* tok = new Tokenizer();
    tok->sp_model = nullptr;
    tok->type = -1;

    // Try sentencepiece first
    std::string sp_path = model_path + "/tokenizer.model";
    if (load_sentencepiece(tok, sp_path))
        return tok;

    // Try HuggingFace tokenizer (vocab.bin or tokenizer.json)
    if (load_hf_tokenizer(tok, model_path))
        return tok;

    fprintf(stderr, "No tokenizer found in %s (tried tokenizer.model and tokenizer.json)\n", model_path.c_str());
    delete tok;
    return nullptr;
}

void free_tokenizer(Tokenizer* tok) {
    if (!tok) return;
    if (tok->type == 0)
        delete (sentencepiece::SentencePieceProcessor*)tok->sp_model;
    else if (tok->type == 1)
        delete (HFTokenizer*)tok->sp_model;
    delete tok;
}

std::vector<int> tokenize(Tokenizer* tok, const std::string& text) {
    if (tok->type == 0) {
        auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
        std::vector<int> ids;
        sp->Encode(text, &ids);
        ids.insert(ids.begin(), tok->bos_id);
        return ids;
    } else {
        auto* hf = (HFTokenizer*)tok->sp_model;
        auto ids = bpe_encode(hf, text);
        ids.insert(ids.begin(), tok->bos_id);
        return ids;
    }
}

std::string detokenize(Tokenizer* tok, const std::vector<int>& tokens) {
    if (tok->type == 0) {
        auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
        std::string text;
        sp->Decode(tokens, &text);
        return text;
    } else {
        auto* hf = (HFTokenizer*)tok->sp_model;
        std::string text;
        for (int id : tokens) {
            auto it = hf->id_to_token.find(id);
            if (it != hf->id_to_token.end()) {
                if (hf->tok_type == 2)
                    text += sp_bpe_token_to_text(it->second);
                else
                    text += it->second;
            }
        }
        return text;
    }
}

std::string detokenize_one(Tokenizer* tok, int token) {
    if (tok->type == 0) {
        auto* sp = (sentencepiece::SentencePieceProcessor*)tok->sp_model;
        return sp->IdToPiece(token);
    } else {
        auto* hf = (HFTokenizer*)tok->sp_model;
        auto it = hf->id_to_token.find(token);
        if (it == hf->id_to_token.end()) return "";
        if (hf->tok_type == 2)
            return sp_bpe_token_to_text(it->second);
        return bpe_token_to_text(hf, it->second);
    }
}
