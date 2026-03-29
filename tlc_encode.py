#!/usr/bin/env python3
"""
TLC Encoder — converts BF16 safetensors to compressed .tlc files.

Uses 8-tier prefix coding with per-tensor codebooks and escape fallback
for guaranteed bit-perfect lossless compression.
"""

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ctypes
import numpy as np
import torch
from safetensors import safe_open

# Load fast C bit packer if available
_FAST_PACKER = None
_fast_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitpack_fast.so")
if os.path.exists(_fast_lib_path):
    try:
        _FAST_PACKER = ctypes.CDLL(_fast_lib_path)
        _FAST_PACKER.pack_values.restype = ctypes.c_uint64
        _FAST_PACKER.pack_values.argtypes = [
            ctypes.POINTER(ctypes.c_int32),   # entries
            ctypes.POINTER(ctypes.c_uint16),  # raw_uint16
            ctypes.c_int,                      # index_bits
            ctypes.c_int64,                    # num_values
            ctypes.POINTER(ctypes.c_uint32),  # packed_out
            ctypes.POINTER(ctypes.c_uint16),  # escapes_out
            ctypes.POINTER(ctypes.c_uint32),  # block_bit_offsets
            ctypes.POINTER(ctypes.c_uint32),  # block_esc_indices
            ctypes.c_int,                      # block_size
            ctypes.POINTER(ctypes.c_int64),   # escape_count_out
        ]
        print("Using fast C bit packer")
    except Exception as e:
        print(f"Warning: Could not load fast bit packer: {e}")
        _FAST_PACKER = None

from tlc_format import (
    MAGIC,
    VERSION,
    FILE_HEADER_SIZE,
    TENSOR_HEADER_SIZE,
    DTYPE_BF16_COMPRESSED,
    DTYPE_F32_PASSTHROUGH,
    NUM_CODED_TIERS,
    ESCAPE_PREFIX_BITS,
    ESCAPE_RAW_BITS,
    ESCAPE_TOTAL_BITS,
    MAX_SHAPE_DIMS,
    BLOCK_SIZE,
    BLOCK_PTR_ENTRY_SIZE,
    TensorHeader,
    write_file_header,
    write_name_table,
    write_tensor_header,
    write_bits,
)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def discover_shards(path: str) -> List[Tuple[str, List[str]]]:
    """
    Discover safetensors shards and their tensor names.

    If path is a .safetensors file: return [(path, all_tensor_names)]
    If path is a directory: check for model.safetensors.index.json, group
    tensors by shard. Fallback: glob for *.safetensors files.

    Returns list of (shard_path, [tensor_names]) tuples.
    """
    path = os.path.abspath(path)

    if os.path.isfile(path) and path.endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            names = list(f.keys())
        return [(path, names)]

    if os.path.isdir(path):
        # Try index file first
        index_path = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r") as fp:
                index = json.load(fp)
            weight_map = index.get("weight_map", {})
            # Group tensor names by shard filename
            shard_to_names: Dict[str, List[str]] = {}
            for tensor_name, shard_file in weight_map.items():
                shard_path = os.path.join(path, shard_file)
                if shard_path not in shard_to_names:
                    shard_to_names[shard_path] = []
                shard_to_names[shard_path].append(tensor_name)
            result = []
            for shard_path in sorted(shard_to_names.keys()):
                result.append((shard_path, shard_to_names[shard_path]))
            return result

        # Check for single model.safetensors
        single_path = os.path.join(path, "model.safetensors")
        if os.path.isfile(single_path):
            with safe_open(single_path, framework="pt", device="cpu") as f:
                names = list(f.keys())
            return [(single_path, names)]

        # Fallback: glob for *.safetensors
        safetensor_files = sorted(Path(path).glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {path}"
            )
        result = []
        for sf in safetensor_files:
            sf_str = str(sf)
            with safe_open(sf_str, framework="pt", device="cpu") as f:
                names = list(f.keys())
            result.append((sf_str, names))
        return result

    raise FileNotFoundError(f"Path not found or not a safetensors source: {path}")


# ---------------------------------------------------------------------------
# Index bits optimization
# ---------------------------------------------------------------------------

def optimize_index_bits(sorted_counts: np.ndarray, total: int) -> int:
    """
    Sweep index_bits 6-12, compute effective bits/value for each configuration.
    Returns the optimal index_bits that minimizes effective bits.

    For each candidate index_bits:
      tier_size = 2^index_bits
      For tiers 0..6: prefix_bits = tier+1, count values in that tier's range
      Escape: 7 + 16 = 23 bits for remaining values
      effective = total_bits / total_values
    """
    best_bits = float("inf")
    best_index_bits = 9  # default

    for index_bits in range(6, 13):
        tier_size = 1 << index_bits
        total_bits = 0
        offset = 0

        for tier in range(NUM_CODED_TIERS):  # tiers 0..6
            prefix_bits = tier + 1
            bits_per_value = prefix_bits + index_bits
            end = min(offset + tier_size, len(sorted_counts))
            tier_count = int(np.sum(sorted_counts[offset:end])) if offset < len(sorted_counts) else 0
            total_bits += tier_count * bits_per_value
            offset = end

        # Escape tier: remaining values
        if offset < len(sorted_counts):
            escape_count = int(np.sum(sorted_counts[offset:]))
        else:
            escape_count = 0
        total_bits += escape_count * ESCAPE_TOTAL_BITS

        effective = total_bits / total if total > 0 else 16.0
        if effective < best_bits:
            best_bits = effective
            best_index_bits = index_bits

    return best_index_bits


# ---------------------------------------------------------------------------
# Single tensor encoding
# ---------------------------------------------------------------------------

def encode_tensor(
    raw_int16: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Encode a single BF16 tensor (given as int16 view of its raw bytes).

    Steps:
      1. GPU unique+counts, sort by frequency descending
      2. Optimize index_bits
      3. Build codebook (sorted_vals[:NUM_CODED_TIERS * tier_size])
      4. Build reverse_map: int32 array of size 65536
      5. Vectorized tier/index lookup
      6. Bit packing with block pointers
      7. Return encoded data dict

    Returns dict with keys:
      codebook      - int16 array of codebook entries
      packed_data   - uint32 array of bit-packed data
      escapes       - uint16 array of escape values
      block_ptrs    - list of (bit_offset, escape_index) tuples
      index_bits    - chosen index_bits
      num_values    - total number of values encoded
      stats         - dict with compression statistics
    """
    num_values = len(raw_int16)

    # --- Step 1: GPU frequency counting ---
    raw_tensor = torch.from_numpy(raw_int16.astype(np.int16)).to(device)
    # View as int16 for unique counting
    unique_vals, counts = torch.unique(raw_tensor, return_counts=True)

    # Sort by count descending
    sort_idx = torch.argsort(counts, descending=True)
    unique_vals = unique_vals[sort_idx].cpu().numpy().astype(np.int16)
    sorted_counts = counts[sort_idx].cpu().numpy().astype(np.int64)
    del raw_tensor  # free GPU memory

    # --- Step 2: Optimize index bits ---
    index_bits = optimize_index_bits(sorted_counts, num_values)
    tier_size = 1 << index_bits

    # --- Step 3: Build codebook ---
    max_coded = NUM_CODED_TIERS * tier_size
    codebook_size = min(max_coded, len(unique_vals))
    codebook = unique_vals[:codebook_size].copy()

    # --- Step 4: Build reverse map ---
    # reverse_map[value & 0xFFFF] = (tier << 16) | index, or -1 for escape
    reverse_map = np.full(65536, -1, dtype=np.int32)
    for i in range(codebook_size):
        tier = i // tier_size
        index = i % tier_size
        val_uint16 = int(codebook[i].view(np.uint16))
        reverse_map[val_uint16] = (tier << 16) | index

    # --- Step 5: Vectorized tier/index lookup ---
    raw_uint16 = raw_int16.astype(np.uint16)
    entries = reverse_map[raw_uint16]  # vectorized lookup

    # Pre-compute tier and index arrays for vectorized stats
    is_escape = entries == -1
    escape_count = int(np.sum(is_escape))

    # Compute per-value bit lengths for stats
    tiers = np.where(is_escape, -1, entries >> 16)
    # prefix bits = tier + 1 for coded values
    coded_mask = ~is_escape
    bit_lengths = np.where(
        is_escape,
        ESCAPE_TOTAL_BITS,
        (tiers + 1 + index_bits),
    )
    total_bits_used = int(np.sum(bit_lengths))

    # Pre-compute cumulative bit positions for block pointer estimation
    cumulative_bits = np.cumsum(bit_lengths)

    # --- Step 6: Bit packing ---
    max_uint32s = (num_values * ESCAPE_TOTAL_BITS + 31) // 32 + 1
    packed = np.zeros(max_uint32s, dtype=np.uint32)
    num_blocks = (num_values + BLOCK_SIZE - 1) // BLOCK_SIZE

    if _FAST_PACKER is not None:
        # Fast C path
        escapes_buf = np.zeros(max(escape_count, 1), dtype=np.uint16)
        block_bit_off = np.zeros(num_blocks, dtype=np.uint32)
        block_esc_idx = np.zeros(num_blocks, dtype=np.uint32)
        esc_count_out = ctypes.c_int64(0)

        entries_c = np.ascontiguousarray(entries, dtype=np.int32)
        raw_c = np.ascontiguousarray(raw_uint16, dtype=np.uint16)

        bit_pos = _FAST_PACKER.pack_values(
            entries_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            raw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int(index_bits),
            ctypes.c_int64(num_values),
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            escapes_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            block_bit_off.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            block_esc_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_int(BLOCK_SIZE),
            ctypes.byref(esc_count_out),
        )
        bit_pos = int(bit_pos)
        actual_esc = int(esc_count_out.value)
        escapes = escapes_buf[:actual_esc]
        block_ptrs = list(zip(block_bit_off[:num_blocks].tolist(), block_esc_idx[:num_blocks].tolist()))
    else:
        # Slow Python fallback
        block_ptrs: List[Tuple[int, int]] = []
        escapes_list: List[int] = []
        bit_pos = 0
        esc_idx = 0

        for i in range(num_values):
            if i % BLOCK_SIZE == 0:
                block_ptrs.append((bit_pos, esc_idx))

            entry = int(entries[i])
            if entry == -1:
                write_bits(packed, bit_pos, (1 << ESCAPE_PREFIX_BITS) - 1, ESCAPE_PREFIX_BITS)
                bit_pos += ESCAPE_PREFIX_BITS
                raw_val = int(raw_uint16[i])
                write_bits(packed, bit_pos, raw_val, ESCAPE_RAW_BITS)
                bit_pos += ESCAPE_RAW_BITS
                escapes_list.append(raw_val)
                esc_idx += 1
            else:
                tier = entry >> 16
                index = entry & 0xFFFF
                prefix_bits = tier + 1
                if tier == 0:
                    prefix_val = 0
                else:
                    prefix_val = ((1 << tier) - 1) << 1
                write_bits(packed, bit_pos, prefix_val, prefix_bits)
                bit_pos += prefix_bits
                write_bits(packed, bit_pos, index, index_bits)
                bit_pos += index_bits

        escapes = np.array(escapes_list, dtype=np.uint16) if escapes_list else np.empty(0, dtype=np.uint16)

    # Trim packed buffer
    used_uint32s = (bit_pos + 31) // 32
    packed = packed[:used_uint32s]

    # --- Step 7: Stats ---
    original_bytes = num_values * 2
    packed_bytes = used_uint32s * 4
    escape_bytes = len(escapes) * 2
    compressed_bytes = packed_bytes + escape_bytes + codebook_size * 2
    effective_bpv = total_bits_used / num_values if num_values > 0 else 0.0
    cr = original_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

    stats = {
        "num_values": num_values,
        "num_unique": len(unique_vals),
        "index_bits": index_bits,
        "tier_size": tier_size,
        "codebook_entries": codebook_size,
        "escape_count": escape_count,
        "escape_rate": escape_count / num_values if num_values > 0 else 0.0,
        "total_packed_bits": bit_pos,
        "effective_bpv": effective_bpv,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": cr,
    }

    return {
        "codebook": codebook,
        "packed_data": packed,
        "escapes": escapes,
        "block_ptrs": block_ptrs,
        "index_bits": index_bits,
        "num_values": num_values,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Model encoding — main entry point
# ---------------------------------------------------------------------------

def encode_model(input_path: str, output_path: str) -> None:
    """
    Encode a BF16 safetensors model (file or directory) into a .tlc file.

    Layout of the .tlc file:
      1. File header
      2. Name table (all tensor names, null-separated)
      3. Tensor headers (with computed offsets)
      4. All codebooks sequentially
      5. All packed data sequentially
      6. All block pointer tables sequentially
      7. All escape sections sequentially
      8. All F32 passthrough data
    """
    total_start = time.time()

    # --- Discover shards ---
    shards = discover_shards(input_path)
    print(f"Discovered {len(shards)} shard(s)")

    # Determine GPU device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, "hip") or (hasattr(torch.__config__, "show") and "rocm" in torch.__config__.show().lower()):
        device = torch.device("cuda")  # ROCm uses cuda device in PyTorch
        print("Using ROCm device")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, using CPU (will be slower)")

    # --- First pass: collect all tensor metadata ---
    tensor_meta: List[dict] = []  # [{name, shape, dtype, shard_path}]
    for shard_path, names in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                t = f.get_tensor(name)
                tensor_meta.append({
                    "name": name,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "shard_path": shard_path,
                })
    num_tensors = len(tensor_meta)
    print(f"Total tensors: {num_tensors}")

    # --- Encode each tensor ---
    encoded_tensors: List[dict] = []
    passthrough_tensors: List[dict] = []

    total_original = 0
    total_compressed = 0

    for idx, meta in enumerate(tensor_meta):
        name = meta["name"]
        shard_path = meta["shard_path"]
        t_start = time.time()

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)

        if tensor.dtype == torch.bfloat16:
            # BF16 -> encode
            raw_bytes = tensor.contiguous().view(torch.int16).numpy().flatten()
            result = encode_tensor(raw_bytes, device)
            result["name"] = name
            result["shape"] = meta["shape"]
            result["dtype_flag"] = DTYPE_BF16_COMPRESSED
            encoded_tensors.append(result)

            elapsed = time.time() - t_start
            s = result["stats"]
            total_original += s["original_bytes"]
            total_compressed += s["compressed_bytes"]
            print(
                f"  [{idx+1}/{num_tensors}] {name}: "
                f"{s['original_bytes']/1024/1024:.2f} MB -> {s['compressed_bytes']/1024/1024:.2f} MB "
                f"(CR {s['compression_ratio']:.3f}x, escape {s['escape_rate']*100:.3f}%, "
                f"bpv {s['effective_bpv']:.2f}, idx_bits={s['index_bits']}) "
                f"[{elapsed:.2f}s]"
            )
        elif tensor.dtype == torch.float32:
            # F32 -> passthrough (norms, etc.)
            raw = tensor.contiguous().numpy().tobytes()
            passthrough_tensors.append({
                "name": name,
                "shape": meta["shape"],
                "dtype_flag": DTYPE_F32_PASSTHROUGH,
                "raw_bytes": raw,
            })
            elapsed = time.time() - t_start
            total_original += len(raw)
            total_compressed += len(raw)
            print(
                f"  [{idx+1}/{num_tensors}] {name}: "
                f"F32 passthrough {len(raw)/1024/1024:.2f} MB [{elapsed:.2f}s]"
            )
        else:
            # Treat other dtypes as BF16 if they're 2 bytes, otherwise error
            elem_size = tensor.element_size()
            if elem_size == 2:
                raw_bytes = tensor.contiguous().view(torch.int16).numpy().flatten()
                result = encode_tensor(raw_bytes, device)
                result["name"] = name
                result["shape"] = meta["shape"]
                result["dtype_flag"] = DTYPE_BF16_COMPRESSED
                encoded_tensors.append(result)

                elapsed = time.time() - t_start
                s = result["stats"]
                total_original += s["original_bytes"]
                total_compressed += s["compressed_bytes"]
                print(
                    f"  [{idx+1}/{num_tensors}] {name}: "
                    f"{s['original_bytes']/1024/1024:.2f} MB -> {s['compressed_bytes']/1024/1024:.2f} MB "
                    f"(CR {s['compression_ratio']:.3f}x, escape {s['escape_rate']*100:.3f}%) "
                    f"[{elapsed:.2f}s]"
                )
            else:
                raise ValueError(
                    f"Unsupported dtype {tensor.dtype} for tensor {name} "
                    f"(element size {elem_size})"
                )

    # Combine all tensors in order: encoded first, then passthrough
    all_tensors = encoded_tensors + passthrough_tensors

    # --- Build name table ---
    all_names = [t["name"] for t in all_tensors]
    # Compute name table size using the format module's convention: uint16 len + UTF-8 bytes per name
    name_table_size = sum(2 + len(n.encode("utf-8")) for n in all_names)

    # --- Compute section offsets ---
    # Layout:
    #   [File Header]
    #   [Name Table]
    #   [Tensor Headers]
    #   [Codebook Data]
    #   [Packed Data]
    #   [Block Pointer Tables]
    #   [Escape Data]
    #   [F32 Passthrough Data]

    name_table_offset = FILE_HEADER_SIZE
    tensor_headers_offset = name_table_offset + name_table_size
    # Align tensor headers
    name_table_padding = 0
    if tensor_headers_offset % 4 != 0:
        name_table_padding = 4 - tensor_headers_offset % 4
        tensor_headers_offset += name_table_padding

    data_start = tensor_headers_offset + len(all_tensors) * TENSOR_HEADER_SIZE

    # Compute per-tensor offsets for each section
    current_offset = data_start

    # Codebook offsets
    for t in all_tensors:
        t["codebook_offset"] = current_offset
        if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
            cb_size = len(t["codebook"]) * 2  # int16 = 2 bytes each
            current_offset += cb_size
        # passthrough tensors have no codebook

    # Align to 4 bytes
    if current_offset % 4 != 0:
        current_offset += 4 - (current_offset % 4)

    # Packed data offsets
    for t in all_tensors:
        t["data_offset"] = current_offset
        if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
            pd_size = len(t["packed_data"]) * 4  # uint32 = 4 bytes each
            current_offset += pd_size

    # Align to 4 bytes
    if current_offset % 4 != 0:
        current_offset += 4 - (current_offset % 4)

    # Block pointer offsets
    for t in all_tensors:
        t["block_ptr_offset"] = current_offset
        if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
            bp_size = len(t["block_ptrs"]) * BLOCK_PTR_ENTRY_SIZE
            current_offset += bp_size

    # Align to 4 bytes
    if current_offset % 4 != 0:
        current_offset += 4 - (current_offset % 4)

    # Escape data offsets
    for t in all_tensors:
        t["escape_offset"] = current_offset
        if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
            esc_size = len(t["escapes"]) * 2  # uint16 = 2 bytes each
            current_offset += esc_size

    # Align to 4 bytes
    if current_offset % 4 != 0:
        current_offset += 4 - (current_offset % 4)

    # F32 passthrough data offsets
    for t in all_tensors:
        if t.get("dtype_flag") == DTYPE_F32_PASSTHROUGH:
            t["data_offset"] = current_offset
            current_offset += len(t["raw_bytes"])

    total_file_size = current_offset

    # --- Write the .tlc file ---
    print(f"\nWriting {output_path} ({total_file_size / 1024 / 1024:.2f} MB)...")
    write_start = time.time()

    with open(output_path, "wb") as out:
        # 1. File header
        write_file_header(out, len(all_tensors))

        # 2. Name table
        name_offsets = write_name_table(out, all_names)

        # Pad to alignment
        if name_table_padding > 0:
            out.write(b"\x00" * name_table_padding)

        # 3. Tensor headers
        for ti, t in enumerate(all_tensors):
            shape = t["shape"]
            # Pad shape to MAX_SHAPE_DIMS
            padded_shape = shape + [0] * (MAX_SHAPE_DIMS - len(shape))
            ndim = len(shape)

            name_bytes = t["name"].encode("utf-8")
            if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
                packed_size = len(t["packed_data"]) * 4  # uint32 array -> bytes
                hdr = TensorHeader(
                    name_offset=name_offsets[ti],
                    name_len=len(name_bytes),
                    dtype=DTYPE_BF16_COMPRESSED,
                    ndim=ndim,
                    shape=tuple(padded_shape),
                    num_elements=t["num_values"],
                    codebook_offset=t["codebook_offset"],
                    data_offset=t["data_offset"],
                    data_size_bytes=packed_size,
                    block_ptr_offset=t["block_ptr_offset"],
                    block_ptr_count=len(t["block_ptrs"]),
                    escape_offset=t["escape_offset"],
                    escape_count=len(t["escapes"]),
                    index_bits=t["index_bits"],
                )
            else:
                # F32 passthrough
                num_el = int(np.prod(shape))
                hdr = TensorHeader(
                    name_offset=name_offsets[ti],
                    name_len=len(name_bytes),
                    dtype=DTYPE_F32_PASSTHROUGH,
                    ndim=ndim,
                    shape=tuple(padded_shape),
                    num_elements=num_el,
                    codebook_offset=0,
                    data_offset=t["data_offset"],
                    data_size_bytes=num_el * 4,
                    block_ptr_offset=0,
                    block_ptr_count=0,
                    escape_offset=0,
                    escape_count=0,
                    index_bits=0,
                )

            write_tensor_header(out, hdr)

            # Advance name offset (name bytes + null terminator)
            # name offset handled by name_offsets list

        # 4. Codebook data
        for t in all_tensors:
            if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
                pos = out.tell()
                assert pos == t["codebook_offset"], \
                    f"Codebook offset mismatch: {pos} != {t['codebook_offset']}"
                out.write(t["codebook"].tobytes())

        # Align to 4 bytes
        pos = out.tell()
        if pos % 4 != 0:
            out.write(b"\x00" * (4 - pos % 4))

        # 5. Packed data
        for t in all_tensors:
            if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
                pos = out.tell()
                assert pos == t["data_offset"], \
                    f"Data offset mismatch for {t['name']}: {pos} != {t['data_offset']}"
                out.write(t["packed_data"].tobytes())

        # Align to 4 bytes
        pos = out.tell()
        if pos % 4 != 0:
            out.write(b"\x00" * (4 - pos % 4))

        # 6. Block pointer tables
        for t in all_tensors:
            if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
                pos = out.tell()
                assert pos == t["block_ptr_offset"], \
                    f"Block ptr offset mismatch: {pos} != {t['block_ptr_offset']}"
                for bit_offset, escape_index in t["block_ptrs"]:
                    out.write(struct.pack("<II", bit_offset, escape_index))

        # Align to 4 bytes
        pos = out.tell()
        if pos % 4 != 0:
            out.write(b"\x00" * (4 - pos % 4))

        # 7. Escape data
        for t in all_tensors:
            if t.get("dtype_flag") == DTYPE_BF16_COMPRESSED:
                pos = out.tell()
                assert pos == t["escape_offset"], \
                    f"Escape offset mismatch: {pos} != {t['escape_offset']}"
                if len(t["escapes"]) > 0:
                    out.write(t["escapes"].tobytes())

        # Align to 4 bytes
        pos = out.tell()
        if pos % 4 != 0:
            out.write(b"\x00" * (4 - pos % 4))

        # 8. F32 passthrough data
        for t in all_tensors:
            if t.get("dtype_flag") == DTYPE_F32_PASSTHROUGH:
                pos = out.tell()
                assert pos == t["data_offset"], \
                    f"F32 data offset mismatch: {pos} != {t['data_offset']}"
                out.write(t["raw_bytes"])

    write_elapsed = time.time() - write_start
    total_elapsed = time.time() - total_start

    # --- Summary ---
    output_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"Encoding complete!")
    print(f"  Input:       {total_original / 1024 / 1024:.2f} MB")
    print(f"  Output:      {output_size / 1024 / 1024:.2f} MB")
    print(f"  CR:          {total_original / output_size:.3f}x" if output_size > 0 else "  CR: N/A")
    print(f"  BF16 tensors: {len(encoded_tensors)}")
    print(f"  F32 tensors:  {len(passthrough_tensors)} (passthrough)")
    print(f"  Write time:  {write_elapsed:.2f}s")
    print(f"  Total time:  {total_elapsed:.2f}s")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tlc_encode.py <input_safetensors_path> <output.tlc>")
        sys.exit(1)
    encode_model(sys.argv[1], sys.argv[2])
