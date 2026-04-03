#!/usr/bin/env python3
"""Convert BF16 safetensors model → Turbo Lossless engine format.

Outputs a directory with:
  config.bin      — model hyperparameters
  tok_embd.bin    — token embeddings (BF16 raw)
  output_norm.bin — output norm weights (FP32)
  layer.N.{attn_norm,ffn_norm}.bin — norm weights (FP32)
  layer.N.{wq,wk,wv,wo,w_gate,w_up,w_down}.{sm,gr,row_off,patch_cols,patch_correct,patch_wrong,dims}.bin
"""

import sys, os, json, struct, time
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ctypes, glob

# Load C packer for split 12-bit format
_dir = os.path.join(os.path.dirname(__file__), "..")
_pack_lib = ctypes.CDLL(os.path.join(_dir, "split12_pack.so"))
_pack_lib.split12_find_base_exp.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int,
]
_pack_lib.split12_find_base_exp.restype = ctypes.c_int
_pack_lib.pack_split12.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # bf16_data
    ctypes.c_int,                     # M
    ctypes.c_int,                     # K
    ctypes.c_int,                     # base_exp
    ctypes.POINTER(ctypes.c_uint8),   # sign_mantissa
    ctypes.POINTER(ctypes.c_uint8),   # groups
    ctypes.POINTER(ctypes.c_int32),   # row_offsets
    ctypes.POINTER(ctypes.c_int32),   # patch_cols
    ctypes.POINTER(ctypes.c_int16),   # patch_correct
    ctypes.POINTER(ctypes.c_int16),   # patch_wrong
]
_pack_lib.pack_split12.restype = ctypes.c_int

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
        """Compress BF16 weight using split 12-bit and save .sm.bin/.gr.bin + CSR escape data."""
        M, K = W.shape
        n = M * K

        raw = W.contiguous().view(torch.int16).numpy().flatten().astype(np.uint16)

        # Find optimal BaseExp for this tensor
        base_exp = _pack_lib.split12_find_base_exp(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int(n),
        )

        # Allocate output arrays
        sign_mantissa = np.zeros(n, dtype=np.uint8)
        groups = np.zeros((n + 1) // 2, dtype=np.uint8)
        max_patches = max(n // 10, 1024)
        row_offsets = np.zeros(M + 1, dtype=np.int32)
        patch_cols = np.zeros(max_patches, dtype=np.int32)
        patch_correct = np.zeros(max_patches, dtype=np.int16)
        patch_wrong = np.zeros(max_patches, dtype=np.int16)

        num_patches = _pack_lib.pack_split12(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int(M), ctypes.c_int(K),
            ctypes.c_int(base_exp),
            sign_mantissa.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            groups.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_correct.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            patch_wrong.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )

        # Save files
        save_raw(f"{prefix}.sm.bin", sign_mantissa)
        save_raw(f"{prefix}.gr.bin", groups)
        save_raw(f"{prefix}.row_off.bin", row_offsets)
        save_raw(f"{prefix}.patch_cols.bin", patch_cols[:num_patches])
        save_raw(f"{prefix}.patch_correct.bin", patch_correct[:num_patches])
        save_raw(f"{prefix}.patch_wrong.bin", patch_wrong[:num_patches])
        with open(os.path.join(output_dir, f"{prefix}.dims"), "w") as f:
            f.write(f"{M} {K} {num_patches} {base_exp}")
        return int(num_patches)

    total_size = 0
    t0 = time.time()

    # Token embeddings (keep as BF16 — used for lookup not matvec)
    embd_name = find_tensor("embed_tokens")
    if embd_name:
        W = load_tensor(embd_name).to(torch.bfloat16)
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
        if W.ndim == 2 and W.dtype in (torch.bfloat16, torch.float16, torch.float32):
            W = W.to(torch.bfloat16)
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
                if W.ndim == 2 and W.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    W = W.to(torch.bfloat16)
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
