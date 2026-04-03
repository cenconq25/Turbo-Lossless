#!/usr/bin/env python3
"""Extract HuggingFace BPE tokenizer to binary format for Turbo engine.

Reads tokenizer.json, writes:
  vocab.bin:        [n_vocab:i32][bos_id:i32][eos_id:i32][tok_type:i32] + [len:u16][bytes] per token
  merges.bin:       [n_merges:i32] + [len_a:u16][len_b:u16][bytes_a][bytes_b] per merge
  byte_encoder.bin: 256 entries of [utf8_len:u8][utf8_bytes]

tok_type in vocab.bin header:
  1 = GPT-2 byte-level BPE (Llama 3, GPT-style) — uses byte_encoder.bin
  2 = Sentencepiece-style BPE (Gemma) — uses ▁ for spaces, <0xNN> byte fallback
"""

import json, struct, sys, os

def bytes_to_unicode():
    """GPT-2 byte encoder: maps bytes 0-255 to unicode characters."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def detect_tokenizer_style(tok):
    """Detect whether this is GPT-2 byte-level BPE or sentencepiece-style BPE.
    Returns 1 for GPT-2, 2 for sentencepiece-style (Gemma)."""
    model = tok.get("model", {})
    vocab = model.get("vocab", {})

    # Sentencepiece-style: uses ▁ prefix, has byte_fallback, <0xNN> tokens
    has_sp_prefix = sum(1 for k in vocab if "▁" in k) > 1000
    has_byte_fallback = model.get("byte_fallback", False)
    has_hex_tokens = any(k.startswith("<0x") for k in vocab)

    if has_sp_prefix and has_byte_fallback and has_hex_tokens:
        return 2  # sentencepiece-style BPE (Gemma)

    return 1  # GPT-2 byte-level BPE (Llama 3, GPT-style)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        sys.exit(1)

    model_dir = sys.argv[1]
    # Auto-detect turbo output dir
    turbo_dir = model_dir.rstrip("/") + "-turbo"
    if not os.path.isdir(turbo_dir):
        turbo_dir = model_dir

    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = os.path.join(turbo_dir, "tokenizer.json")

    print(f"Reading {tok_path}...")
    with open(tok_path) as f:
        tok = json.load(f)

    # Detect tokenizer style
    tok_type = detect_tokenizer_style(tok)
    style_name = "GPT-2 byte-level BPE" if tok_type == 1 else "sentencepiece-style BPE (Gemma)"
    print(f"  Detected style: {style_name} (type={tok_type})")

    # Extract vocab
    model = tok.get("model", {})
    vocab = model.get("vocab", {})
    n_vocab = len(vocab)

    # Sort by ID to ensure correct order
    id_to_token = {v: k for k, v in vocab.items()}

    # Find BOS/EOS from added_tokens
    bos_id, eos_id = -1, -1
    for at in tok.get("added_tokens", []):
        content = at.get("content", "")
        tid = at.get("id", -1)
        if not at.get("special", False):
            continue
        # Llama 3 style
        if content == "<|begin_of_text|>":
            bos_id = tid
        elif content == "<|end_of_text|>":
            eos_id = tid
        # Sentencepiece / Gemma style
        if content == "<bos>":
            bos_id = tid
        elif content == "<eos>":
            eos_id = tid
        # Generic
        if content == "<s>":
            bos_id = tid
        if content == "</s>":
            eos_id = tid

    # Fallback defaults if not found
    if bos_id == -1:
        bos_id = 128000 if tok_type == 1 else 2
    if eos_id == -1:
        eos_id = 128001 if tok_type == 1 else 1

    print(f"  vocab={n_vocab}, bos={bos_id}, eos={eos_id}")

    # Write vocab.bin (with tok_type field)
    vocab_out = os.path.join(turbo_dir, "vocab.bin")
    with open(vocab_out, "wb") as f:
        f.write(struct.pack("<iiii", n_vocab, bos_id, eos_id, tok_type))
        for i in range(n_vocab):
            token = id_to_token.get(i, "")
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
    print(f"  Wrote {vocab_out}")

    # Extract merges
    merges = model.get("merges", [])
    merges_out = os.path.join(turbo_dir, "merges.bin")
    with open(merges_out, "wb") as f:
        f.write(struct.pack("<i", len(merges)))
        for merge in merges:
            if isinstance(merge, str):
                parts = merge.split(" ", 1)
            else:
                parts = merge
            a, b = parts[0], parts[1]
            a_bytes = a.encode("utf-8")
            b_bytes = b.encode("utf-8")
            f.write(struct.pack("<HH", len(a_bytes), len(b_bytes)))
            f.write(a_bytes)
            f.write(b_bytes)
    print(f"  Wrote {merges_out} ({len(merges)} merges)")

    # Write byte_encoder.bin (GPT-2 byte-to-unicode mapping)
    # Only meaningful for tok_type==1, but write it anyway for compatibility
    byte_enc = bytes_to_unicode()
    be_out = os.path.join(turbo_dir, "byte_encoder.bin")
    with open(be_out, "wb") as f:
        for i in range(256):
            s = byte_enc[i].encode("utf-8")
            f.write(struct.pack("<B", len(s)))
            f.write(s)
    print(f"  Wrote {be_out}")
    print("Done!")

if __name__ == "__main__":
    main()
