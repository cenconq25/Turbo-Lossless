#!/usr/bin/env python3
"""Convert BF16 safetensors model → Turbo Lossless engine format.

Outputs a directory with:
  config.bin      — model hyperparameters
  tok_embd.bin    — token embeddings (BF16 raw)
  output_norm.bin — output norm weights (FP32)
  layer.N.{attn_norm,ffn_norm}.bin — norm weights (FP32)
  layer.N.{wq,wk,wv,wo,w_gate,w_up,w_down}.{packed,codebook,esc_off,esc_val}.bin
"""

import sys, os, json, struct, time
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ctypes, glob

# Load C packer for CSR format
_dir = os.path.join(os.path.dirname(__file__), "..")
_pack_lib = ctypes.CDLL(os.path.join(_dir, "fixed12_pack.so"))
_pack_lib.build_codebook_12bit.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint32),
]
_pack_lib.build_codebook_12bit.restype = ctypes.c_int
_pack_lib.pack_fixed12_csr.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int16, ctypes.c_int32, ctypes.c_int32,
]
_pack_lib.pack_fixed12_csr.restype = ctypes.c_int64

def convert(model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    n_embd = cfg.get("hidden_size", cfg.get("dim", 4096))
    n_head = cfg.get("num_attention_heads", cfg.get("n_heads", 32))
    n_head_kv = cfg.get("num_key_value_heads", cfg.get("n_kv_heads", n_head))
    n_layer = cfg.get("num_hidden_layers", cfg.get("n_layers", 32))
    n_ff = cfg.get("intermediate_size", cfg.get("hidden_dim", n_embd * 4))
    n_vocab = cfg.get("vocab_size", 32000)
    n_ctx = cfg.get("max_position_embeddings", cfg.get("max_seq_len", 32768))
    rope_theta = cfg.get("rope_theta", 10000.0)
    rms_eps = cfg.get("rms_norm_eps", 1e-5)

    # Write config
    config_data = struct.pack("iiiiiiiff",
        n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ff, n_ctx,
        rope_theta, rms_eps)
    with open(os.path.join(output_dir, "config.bin"), "wb") as f:
        f.write(config_data)
    print(f"Config: vocab={n_vocab} embd={n_embd} heads={n_head}/{n_head_kv} layers={n_layer} ff={n_ff}")

    # Load all safetensors shards
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    tensors = {}
    for shard in shards:
        f = safe_open(shard, framework="pt")
        for name in f.keys():
            tensors[name] = (shard, name)

    # Name mapping: HF → our format
    def find_tensor(pattern):
        for name in tensors:
            if pattern in name:
                return name
        return None

    def load_tensor(name):
        shard, key = tensors[name]
        f = safe_open(shard, framework="pt")
        return f.get_tensor(key)

    def save_raw(filename, data):
        path = os.path.join(output_dir, filename)
        if isinstance(data, torch.Tensor):
            data = data.contiguous().numpy()
        data.tofile(path)
        return os.path.getsize(path)

    def save_compressed(prefix, W):
        """Compress BF16 weight and save packed + CSR escape data."""
        M, K = W.shape
        n = M * K

        # GPU frequency sort
        raw_gpu = W.contiguous().view(torch.int16).to("cuda:0")
        unique_gpu, counts_gpu = torch.unique(raw_gpu.view(-1), return_counts=True)
        si_gpu = torch.argsort(counts_gpu, descending=True)
        sorted_vals = unique_gpu[si_gpu].cpu().numpy().astype(np.uint16)
        del raw_gpu, unique_gpu, counts_gpu, si_gpu
        torch.cuda.empty_cache()

        raw = W.contiguous().view(torch.int16).numpy().flatten().astype(np.uint16)

        # Build codebook
        codebook = np.zeros(4096, dtype=np.int16)
        reverse_map = np.zeros(65536, dtype=np.uint32)
        _pack_lib.build_codebook_12bit(
            sorted_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            len(sorted_vals),
            codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            reverse_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )

        # Pack 12-bit + CSR escape data
        num_words = (n * 12 + 31) // 32 + 4
        packed = np.zeros(num_words, dtype=np.uint32)
        max_patches = max(n // 10, 1024)
        row_offsets = np.zeros(M + 1, dtype=np.int32)
        patch_cols = np.zeros(max_patches, dtype=np.int32)
        patch_correct = np.zeros(max_patches, dtype=np.int16)
        patch_wrong = np.zeros(max_patches, dtype=np.int16)
        wrong_val = codebook[4095]

        num_patches = _pack_lib.pack_fixed12_csr(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            reverse_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_correct.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            patch_wrong.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_int16(wrong_val),
            ctypes.c_int32(M), ctypes.c_int32(K),
        )

        # Save files
        save_raw(f"{prefix}.packed.bin", packed[:num_words])
        save_raw(f"{prefix}.codebook.bin", codebook)
        save_raw(f"{prefix}.row_off.bin", row_offsets)
        save_raw(f"{prefix}.patch_cols.bin", patch_cols[:num_patches])
        save_raw(f"{prefix}.patch_correct.bin", patch_correct[:num_patches])
        save_raw(f"{prefix}.patch_wrong.bin", patch_wrong[:num_patches])
        with open(os.path.join(output_dir, f"{prefix}.dims"), "w") as f:
            f.write(f"{M} {K} {num_patches}")
        return int(num_patches)

    total_size = 0
    t0 = time.time()

    # Token embeddings (keep as BF16 — used for lookup not matvec)
    embd_name = find_tensor("embed_tokens")
    if embd_name:
        W = load_tensor(embd_name)
        sz = save_raw("tok_embd.bin", W.contiguous().view(torch.int16))
        total_size += sz
        print(f"  tok_embd: {W.shape} → {sz/1e6:.1f} MB")

    # Output norm — must match exactly "model.norm.weight" not layer norms
    norm_name = find_tensor("model.norm.weight")
    if norm_name:
        W = load_tensor(norm_name).float()
        save_raw("output_norm.bin", W)

    # Output projection (lm_head)
    out_name = find_tensor("lm_head")
    if out_name:
        W = load_tensor(out_name)
        if W.dtype == torch.bfloat16 and W.ndim == 2:
            esc = save_compressed("output_proj", W)
            print(f"  output_proj: {W.shape} escapes={esc}")

    # Layers
    for layer in range(n_layer):
        print(f"  Layer {layer}/{n_layer}...", end="", flush=True)

        # Attention norms
        for norm_type, suffix in [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")]:
            name = find_tensor(f"layers.{layer}.{norm_type}")
            if name:
                W = load_tensor(name).float()
                save_raw(f"layer.{layer}.{suffix}.bin", W)

        # Weight tensors
        weight_map = {
            "wq": f"layers.{layer}.self_attn.q_proj",
            "wk": f"layers.{layer}.self_attn.k_proj",
            "wv": f"layers.{layer}.self_attn.v_proj",
            "wo": f"layers.{layer}.self_attn.o_proj",
            "w_gate": f"layers.{layer}.mlp.gate_proj",
            "w_up": f"layers.{layer}.mlp.up_proj",
            "w_down": f"layers.{layer}.mlp.down_proj",
        }

        for our_name, hf_pattern in weight_map.items():
            hf_name = find_tensor(hf_pattern)
            if hf_name:
                W = load_tensor(hf_name)
                if W.dtype == torch.bfloat16 and W.ndim == 2:
                    esc = save_compressed(f"layer.{layer}.{our_name}", W)

        print(f" done")

    elapsed = time.time() - t0
    print(f"\nConversion complete in {elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [output_dir]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else model_dir + "-turbo"
    convert(model_dir, output_dir)
