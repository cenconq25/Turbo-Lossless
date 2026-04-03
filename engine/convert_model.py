#!/usr/bin/env python3
"""Convert BF16 safetensors model → Turbo Lossless engine format.

Outputs a directory with:
  config.bin      — model hyperparameters
  tok_embd.bin    — token embeddings (BF16 raw)
  output_norm.bin — output norm weights (FP32)
  layer.N.{attn_norm,ffn_norm}.bin — norm weights (FP32)
  layer.N.{wq,wk,wv,wo,w_gate,w_up,w_down}.{packed,codebook,esc_off,esc_val}.bin
"""

import sys, os, json, struct, time, argparse
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ctypes, glob

# Load C packer for structured 12-bit format
_dir = os.path.join(os.path.dirname(__file__), "..")
_pack_lib = ctypes.CDLL(os.path.join(_dir, "structured12_pack.so"))
_pack_lib.find_base_exp.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),  # exp_rmap_out[256]
]
_pack_lib.find_base_exp.restype = ctypes.c_int
_pack_lib.pack_structured12_csr.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),   # exp_rmap[256]
    ctypes.POINTER(ctypes.c_uint32),  # packed_out
    ctypes.POINTER(ctypes.c_int32),   # row_offsets_out
    ctypes.POINTER(ctypes.c_int32),   # patch_cols_out
    ctypes.POINTER(ctypes.c_int16),   # patch_correct_out
    ctypes.POINTER(ctypes.c_int16),   # patch_wrong_out
    ctypes.c_int32,                   # wrong_value
    ctypes.c_int32, ctypes.c_int32,   # M, K
]
_pack_lib.pack_structured12_csr.restype = ctypes.c_int64

# Load C split12 packer
_split12_lib_path = os.path.join(_dir, "split12_pack.so")
if not os.path.exists(_split12_lib_path):
    os.system(f"gcc -O3 -shared -fPIC -o {_split12_lib_path} {os.path.join(_dir, 'split12_pack.c')}")
_split12_lib = ctypes.CDLL(_split12_lib_path)
_split12_lib.split12_find_base_exp.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
_split12_lib.split12_find_base_exp.restype = ctypes.c_int
_split12_lib.pack_split12.argtypes = [
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int16),
]
_split12_lib.pack_split12.restype = ctypes.c_int

# Tensors that are row-split (split M axis) for tensor parallelism
ROW_SPLIT_NAMES = {"wq", "wk", "wv", "w_gate", "w_up", "output_proj"}
# Tensors that are column-split (split K axis) for tensor parallelism
COL_SPLIT_NAMES = {"wo", "w_down"}

def convert(model_dir, output_dir, tp=1):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    # Detect Gemma 4: text model config is nested under text_config
    is_gemma4 = cfg.get("model_type") == "gemma4"
    tcfg = cfg.get("text_config", cfg) if is_gemma4 else cfg

    n_embd = tcfg.get("hidden_size", tcfg.get("dim", 4096))
    n_head = tcfg.get("num_attention_heads", tcfg.get("n_heads", 32))
    n_head_kv = tcfg.get("num_key_value_heads", tcfg.get("n_kv_heads", n_head))
    n_layer = tcfg.get("num_hidden_layers", tcfg.get("n_layers", 32))
    n_ff = tcfg.get("intermediate_size", tcfg.get("hidden_dim", n_embd * 4))
    n_vocab = tcfg.get("vocab_size", 32000)
    n_ctx = tcfg.get("max_position_embeddings", tcfg.get("max_seq_len", 32768))
    rope_theta = tcfg.get("rope_theta", 10000.0)
    rms_eps = tcfg.get("rms_norm_eps", 1e-5)

    # For Gemma 4, rope_theta comes from sliding_attention sub-dict
    if is_gemma4:
        rope_params = tcfg.get("rope_parameters", {})
        rope_theta = rope_params.get("sliding_attention", {}).get("rope_theta", rope_theta)

    # Write config
    if is_gemma4:
        # v2 format with Gemma4-specific fields
        head_dim_sliding = tcfg.get("head_dim", 256)
        head_dim_full = tcfg.get("global_head_dim", 512)
        sliding_window = tcfg.get("sliding_window", 512)
        logit_softcap = tcfg.get("final_logit_softcapping", 0.0)
        num_kv_shared = tcfg.get("num_kv_shared_layers", 0)
        rope_params = tcfg.get("rope_parameters", {})
        rope_theta_full = rope_params.get("full_attention", {}).get("rope_theta", 1000000.0)
        partial_rotary_factor = rope_params.get("full_attention", {}).get("partial_rotary_factor", 1.0)
        hidden_act = tcfg.get("hidden_activation", "silu")
        activation_type = 1 if "gelu" in hidden_act else 0
        tie_embeddings = 1 if tcfg.get("tie_word_embeddings", False) else 0

        # Layer types: 0=sliding, 1=full_attention
        layer_types_str = tcfg.get("layer_types", [])
        layer_types = []
        for lt in layer_types_str:
            layer_types.append(1 if lt == "full_attention" else 0)
        # Pad to n_layer if needed
        while len(layer_types) < n_layer:
            layer_types.append(0)

        config_data = b"TLv2"
        config_data += struct.pack("iiiiiiiff",
            n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ff, n_ctx,
            rope_theta, rms_eps)
        config_data += struct.pack("iiififf",
            head_dim_sliding, head_dim_full, sliding_window, logit_softcap,
            num_kv_shared, rope_theta_full, partial_rotary_factor)
        config_data += struct.pack("ii", activation_type, tie_embeddings)
        config_data += bytes(layer_types[:n_layer])

        with open(os.path.join(output_dir, "config.bin"), "wb") as f:
            f.write(config_data)
        print(f"Config (TLv2/Gemma4): vocab={n_vocab} embd={n_embd} heads={n_head}/{n_head_kv} "
              f"layers={n_layer} ff={n_ff}")
        print(f"  head_dim: sliding={head_dim_sliding} full={head_dim_full} "
              f"sliding_window={sliding_window} softcap={logit_softcap}")
        print(f"  rope: sliding={rope_theta} full={rope_theta_full} "
              f"partial_rotary={partial_rotary_factor}")
        print(f"  activation={hidden_act}({activation_type}) tie_embd={tie_embeddings} "
              f"kv_shared={num_kv_shared}")
        n_full = sum(layer_types[:n_layer])
        print(f"  layer_types: {n_full} full_attention, {n_layer - n_full} sliding")
    else:
        config_data = struct.pack("iiiiiiiff",
            n_vocab, n_embd, n_head, n_head_kv, n_layer, n_ff, n_ctx,
            rope_theta, rms_eps)
        with open(os.path.join(output_dir, "config.bin"), "wb") as f:
            f.write(config_data)
        print(f"Config: vocab={n_vocab} embd={n_embd} heads={n_head}/{n_head_kv} layers={n_layer} ff={n_ff}")
    if tp > 1:
        print(f"Tensor Parallelism: tp={tp}, row-split={ROW_SPLIT_NAMES}, col-split={COL_SPLIT_NAMES}")

    # Load all safetensors shards
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    tensors = {}
    for shard in shards:
        f = safe_open(shard, framework="pt")
        for name in f.keys():
            # For Gemma 4: filter out audio_tower/vision_tower tensors
            if is_gemma4 and ("audio_tower" in name or "vision_tower" in name):
                continue
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
        """Compress BF16 weight using structured 12-bit and save packed + split12 + CSR escape data."""
        M, K = W.shape
        n = M * K

        raw = W.contiguous().view(torch.int16).numpy().flatten().astype(np.uint16)

        # Find optimal BaseExp for this tensor
        exp_rmap = np.zeros(256, dtype=np.uint8)
        base_exp = _pack_lib.find_base_exp(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            exp_rmap.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        # Pack structured 12-bit + CSR escape data
        num_words = (n * 12 + 31) // 32 + 4
        packed = np.zeros(num_words, dtype=np.uint32)
        max_patches = max(n // 10, 1024)
        row_offsets = np.zeros(M + 1, dtype=np.int32)
        patch_cols = np.zeros(max_patches, dtype=np.int32)
        patch_correct = np.zeros(max_patches, dtype=np.int16)
        patch_wrong = np.zeros(max_patches, dtype=np.int16)

        num_patches = _pack_lib.pack_structured12_csr(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            exp_rmap.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch_correct.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            patch_wrong.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_int32(0),  # wrong_value unused (group=0 encodes escapes)
            ctypes.c_int32(M), ctypes.c_int32(K),
        )

        # Save packed12 files
        save_raw(f"{prefix}.packed.bin", packed[:num_words])
        save_raw(f"{prefix}.row_off.bin", row_offsets)
        save_raw(f"{prefix}.patch_cols.bin", patch_cols[:num_patches])
        save_raw(f"{prefix}.patch_correct.bin", patch_correct[:num_patches])
        save_raw(f"{prefix}.patch_wrong.bin", patch_wrong[:num_patches])
        with open(os.path.join(output_dir, f"{prefix}.dims"), "w") as f:
            f.write(f"{M} {K} {num_patches} {base_exp}")

        # Also generate split12 format (byte-aligned, zero read amplification)
        sm_out = np.zeros(n, dtype=np.uint8)
        gr_out = np.zeros((n + 1) // 2, dtype=np.uint8)
        s12_row_offsets = np.zeros(M + 1, dtype=np.int32)
        s12_patch_cols = np.zeros(max_patches, dtype=np.int32)
        s12_patch_correct = np.zeros(max_patches, dtype=np.int16)
        s12_patch_wrong = np.zeros(max_patches, dtype=np.int16)

        _split12_lib.pack_split12(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(base_exp),
            sm_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            gr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            s12_row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            s12_patch_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            s12_patch_correct.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            s12_patch_wrong.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
        save_raw(f"{prefix}.sm.bin", sm_out)
        save_raw(f"{prefix}.gr.bin", gr_out)

        return int(num_patches)

    def save_compressed_shard(prefix, W):
        """Compress a TP shard — same as save_compressed but for a shard tensor."""
        return save_compressed(prefix, W)

    def shard_and_save(our_name, prefix, W):
        """For tp > 1: shard a weight tensor and save each shard."""
        M, K = W.shape
        is_row_split = our_name in ROW_SPLIT_NAMES
        is_col_split = our_name in COL_SPLIT_NAMES

        for rank in range(tp):
            if is_row_split:
                shard_m_start = rank * M // tp
                shard_m_end = (rank + 1) * M // tp
                shard = W[shard_m_start:shard_m_end, :].contiguous()
            elif is_col_split:
                shard_k_start = rank * K // tp
                shard_k_end = (rank + 1) * K // tp
                shard = W[:, shard_k_start:shard_k_end].contiguous()
            else:
                # Replicate (should not happen for weight tensors, but safety)
                shard = W.contiguous()

            tp_prefix = f"{prefix}.tp{rank}"
            save_compressed_shard(tp_prefix, shard)

        return tp  # number of shards saved

    total_size = 0
    t0 = time.time()

    # Tensor name prefix differs for Gemma 4
    lang_prefix = "model.language_model." if is_gemma4 else ""

    # Token embeddings (keep as BF16 — used for lookup not matvec)
    embd_key = f"{lang_prefix}embed_tokens.weight" if is_gemma4 else None
    embd_name = embd_key if (embd_key and embd_key in tensors) else find_tensor("embed_tokens")
    embd_tensor = None
    if embd_name:
        W = load_tensor(embd_name)
        sz = save_raw("tok_embd.bin", W.contiguous().view(torch.int16))
        total_size += sz
        print(f"  tok_embd: {W.shape} -> {sz/1e6:.1f} MB")
        if is_gemma4:
            embd_tensor = W  # Keep for tied output_proj

    # Output norm — must match exactly "model.norm.weight" (or language_model.norm.weight for Gemma4)
    norm_key = f"{lang_prefix}norm.weight" if is_gemma4 else "model.norm.weight"
    norm_name = norm_key if norm_key in tensors else find_tensor("model.norm.weight")
    if norm_name:
        W = load_tensor(norm_name).float()
        save_raw("output_norm.bin", W)

    # Output projection (lm_head) — or tied embeddings for Gemma 4
    if is_gemma4 and tcfg.get("tie_word_embeddings", False) and embd_tensor is not None:
        # Tied embeddings: embed_tokens IS the output projection
        W = embd_tensor
        if W.dtype == torch.bfloat16 and W.ndim == 2:
            if tp > 1:
                shard_and_save("output_proj", "output_proj", W)
                print(f"  output_proj (tied): {W.shape} -> {tp} TP shards (row-split)")
            else:
                esc = save_compressed("output_proj", W)
                print(f"  output_proj (tied from embed_tokens): {W.shape} escapes={esc}")
    else:
        out_name = find_tensor("lm_head")
        if out_name:
            W = load_tensor(out_name)
            if W.dtype == torch.bfloat16 and W.ndim == 2:
                if tp > 1:
                    shard_and_save("output_proj", "output_proj", W)
                    print(f"  output_proj: {W.shape} -> {tp} TP shards (row-split)")
                else:
                    esc = save_compressed("output_proj", W)
                    print(f"  output_proj: {W.shape} escapes={esc}")

    # Gemma 4: save per-layer input embedding and projection tensors
    if is_gemma4:
        for extra_name, out_file in [
            (f"{lang_prefix}embed_tokens_per_layer.weight", "embed_tokens_per_layer.bin"),
            (f"{lang_prefix}per_layer_model_projection.weight", "per_layer_model_projection.bin"),
            (f"{lang_prefix}per_layer_projection_norm.weight", "per_layer_projection_norm.bin"),
        ]:
            if extra_name in tensors:
                W = load_tensor(extra_name)
                sz = save_raw(out_file, W.contiguous().view(torch.int16) if W.dtype == torch.bfloat16 and W.ndim >= 2 else W.float())
                print(f"  {out_file}: {W.shape} -> {sz/1e6:.1f} MB")

    # Layers
    for layer in range(n_layer):
        print(f"  Layer {layer}/{n_layer}...", end="", flush=True)

        # Attention norms
        norm_map = [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")]
        if is_gemma4:
            norm_map += [
                ("pre_feedforward_layernorm", "pre_ffn_norm"),
                ("post_feedforward_layernorm", "post_ffn_norm"),
                ("post_per_layer_input_norm", "post_pli_norm"),
            ]
        for norm_type, suffix in norm_map:
            name = find_tensor(f"layers.{layer}.{norm_type}")
            if name:
                W = load_tensor(name).float()
                save_raw(f"layer.{layer}.{suffix}.bin", W)

        # Gemma 4: q_norm, k_norm (save as FP32)
        if is_gemma4:
            for qk_name, qk_suffix in [("q_norm", "q_norm"), ("k_norm", "k_norm")]:
                name = find_tensor(f"layers.{layer}.self_attn.{qk_name}")
                if name:
                    W = load_tensor(name).float()
                    save_raw(f"layer.{layer}.{qk_suffix}.bin", W)

        # Gemma 4: layer_scalar (save as FP32)
        if is_gemma4:
            scalar_name = find_tensor(f"layers.{layer}.layer_scalar")
            if scalar_name:
                W = load_tensor(scalar_name).float()
                save_raw(f"layer.{layer}.layer_scalar.bin", W)

        # Gemma 4: per_layer_input_gate and per_layer_projection (compress as BF16 2D)
        if is_gemma4:
            for pli_name, pli_suffix in [
                ("per_layer_input_gate", "pli_gate"),
                ("per_layer_projection", "pli_proj"),
            ]:
                name = find_tensor(f"layers.{layer}.{pli_name}")
                if name:
                    W = load_tensor(name)
                    if W.dtype == torch.bfloat16 and W.ndim == 2:
                        prefix = f"layer.{layer}.{pli_suffix}"
                        if tp > 1:
                            shard_and_save(pli_suffix, prefix, W)
                        else:
                            save_compressed(prefix, W)

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
                    prefix = f"layer.{layer}.{our_name}"
                    if tp > 1:
                        shard_and_save(our_name, prefix, W)
                    else:
                        save_compressed(prefix, W)

        print(f" done")

    # Copy tokenizer files
    import shutil
    tokenizer_files = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json"]
    for tf in tokenizer_files:
        src = os.path.join(model_dir, tf)
        if os.path.exists(src):
            dst = os.path.join(output_dir, tf)
            shutil.copy2(src, dst)
            print(f"  Copied {tf}")

    elapsed = time.time() - t0
    print(f"\nConversion complete in {elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BF16 safetensors to Turbo Lossless format")
    parser.add_argument("model_dir", help="Path to HuggingFace model directory")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory (default: <model_dir>-turbo)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree (default: 1)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.model_dir + "-turbo"
    if args.tp > 1:
        print(f"Tensor Parallelism: splitting weights for {args.tp} GPUs")
    convert(args.model_dir, output_dir, tp=args.tp)
