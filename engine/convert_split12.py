#!/usr/bin/env python3
"""Generate split12 format files from original BF16 safetensors.

Much faster than converting from packed12 — loads BF16 directly and calls split12_pack.so.
"""

import sys, os, struct, ctypes, glob, time
import numpy as np
import torch
from safetensors import safe_open

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hf_model_dir>")
        print("  Generates .sm.bin and .gr.bin files in the turbo output directory")
        sys.exit(1)

    hf_dir = sys.argv[1]
    turbo_dir = hf_dir.rstrip("/") + "-turbo"

    if not os.path.isdir(turbo_dir):
        print(f"Turbo dir not found: {turbo_dir}")
        print("Run convert_model.py first.")
        sys.exit(1)

    # Load C split12 packer
    pack_dir = os.path.join(os.path.dirname(__file__), "..")
    lib_path = os.path.join(pack_dir, "split12_pack.so")
    if not os.path.exists(lib_path):
        os.system(f"gcc -O3 -shared -fPIC -o {lib_path} {os.path.join(pack_dir, 'split12_pack.c')}")

    lib = ctypes.CDLL(lib_path)
    lib.split12_find_base_exp.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int]
    lib.split12_find_base_exp.restype = ctypes.c_int
    lib.pack_split12.argtypes = [
        ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int16),
    ]
    lib.pack_split12.restype = ctypes.c_int

    # Load HF config for tensor name mapping
    import json
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    n_layer = config.get("num_hidden_layers", 32)

    # Map HF tensor names to turbo names
    name_map = {}
    for i in range(n_layer):
        pfx = f"model.layers.{i}"
        name_map[f"{pfx}.self_attn.q_proj.weight"] = f"layer.{i}.wq"
        name_map[f"{pfx}.self_attn.k_proj.weight"] = f"layer.{i}.wk"
        name_map[f"{pfx}.self_attn.v_proj.weight"] = f"layer.{i}.wv"
        name_map[f"{pfx}.self_attn.o_proj.weight"] = f"layer.{i}.wo"
        name_map[f"{pfx}.mlp.gate_proj.weight"] = f"layer.{i}.w_gate"
        name_map[f"{pfx}.mlp.up_proj.weight"] = f"layer.{i}.w_up"
        name_map[f"{pfx}.mlp.down_proj.weight"] = f"layer.{i}.w_down"
    name_map["lm_head.weight"] = "output_proj"

    # Find safetensors shards
    shards = sorted(glob.glob(os.path.join(hf_dir, "model-*.safetensors")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if not shards:
        print(f"No safetensors files in {hf_dir}")
        sys.exit(1)

    print(f"Converting {len(name_map)} weight tensors to split12 format...")
    t0 = time.time()
    converted = 0

    for shard_path in shards:
        with safe_open(shard_path, framework="pt") as f:
            for hf_name in f.keys():
                if hf_name not in name_map:
                    continue
                turbo_name = name_map[hf_name]
                sm_path = os.path.join(turbo_dir, f"{turbo_name}.sm.bin")
                gr_path = os.path.join(turbo_dir, f"{turbo_name}.gr.bin")

                if os.path.exists(sm_path) and os.path.exists(gr_path):
                    converted += 1
                    continue

                # Load BF16 tensor
                tensor = f.get_tensor(hf_name)
                if tensor.dtype != torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                bf16_flat = tensor.contiguous().view(-1).view(torch.int16).numpy().astype(np.uint16)
                M, K = tensor.shape[0], tensor.shape[1]

                # Read base_exp from .dims file
                dims_path = os.path.join(turbo_dir, f"{turbo_name}.dims")
                if os.path.exists(dims_path):
                    with open(dims_path) as df:
                        base_exp = int(df.read().strip().split()[3])
                else:
                    base_exp = lib.split12_find_base_exp(
                        bf16_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                        len(bf16_flat))

                count = M * K
                sm_out = np.zeros(count, dtype=np.uint8)
                gr_out = np.zeros((count + 1) // 2, dtype=np.uint8)
                ro = np.zeros(M + 1, dtype=np.int32)
                max_patches = max(int(count * 0.05), 1024)  # 5% headroom for high-escape tensors like wq
                pc = np.zeros(max_patches, dtype=np.int32)
                pcorr = np.zeros(max_patches, dtype=np.int16)
                pwrong = np.zeros(max_patches, dtype=np.int16)

                lib.pack_split12(
                    bf16_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                    M, K, base_exp,
                    sm_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    gr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    ro.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    pc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    pcorr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                    pwrong.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                )

                sm_out.tofile(sm_path)
                gr_out.tofile(gr_path)
                converted += 1

                if converted % 20 == 0:
                    print(f"  {converted}/{len(name_map)}: {turbo_name} ({M}x{K}) [{time.time()-t0:.1f}s]")

    print(f"\nDone! Converted {converted} tensors in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
