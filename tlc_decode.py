"""TLC Decoder — reads .tlc files and reconstructs tensors."""

import ctypes
import os
import time
import numpy as np
import torch

from tlc_format import (
    MAGIC, VERSION, DTYPE_BF16_COMPRESSED, DTYPE_F32_PASSTHROUGH,
    NUM_CODED_TIERS, BLOCK_SIZE,
    TensorHeader,
    read_file_header, read_name_table, read_tensor_header,
    read_bits, read_bit,
)

# Load fast C bit unpacker if available
_FAST_UNPACKER = None
_fast_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitunpack_fast.so")
if os.path.exists(_fast_lib_path):
    try:
        _FAST_UNPACKER = ctypes.CDLL(_fast_lib_path)
        _FAST_UNPACKER.unpack_values.restype = ctypes.c_uint64
        _FAST_UNPACKER.unpack_values.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # packed
            ctypes.POINTER(ctypes.c_int16),   # codebook
            ctypes.c_int,                      # index_bits
            ctypes.c_int,                      # tier_size
            ctypes.c_int64,                    # num_values
            ctypes.POINTER(ctypes.c_int16),   # output
        ]
    except Exception:
        _FAST_UNPACKER = None


def decode_tensor(f, header: TensorHeader) -> torch.Tensor:
    """Decode a single BF16 compressed tensor from the TLC bitstream."""
    # 1. Read codebook
    f.seek(header.codebook_offset)
    tier_size = 1 << header.index_bits
    codebook_bytes = f.read(NUM_CODED_TIERS * tier_size * 2)
    codebook = np.frombuffer(codebook_bytes, dtype=np.int16).copy()

    # 2. Read packed bitstream
    f.seek(header.data_offset)
    packed = np.frombuffer(f.read(header.data_size_bytes), dtype=np.uint32).copy()

    # 3. Decode
    output = np.empty(header.num_elements, dtype=np.int16)

    if _FAST_UNPACKER is not None:
        _FAST_UNPACKER.unpack_values(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_int(header.index_bits),
            ctypes.c_int(tier_size),
            ctypes.c_int64(header.num_elements),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        )
    else:
        # Slow Python fallback
        bit_pos = 0
        for i in range(header.num_elements):
            tier = 0
            while tier < 7:
                bit_val, bit_pos = read_bit(packed, bit_pos)
                if bit_val == 0:
                    break
                tier += 1
            if tier < 7:
                index, bit_pos = read_bits(packed, bit_pos, header.index_bits)
                output[i] = codebook[tier * tier_size + index]
            else:
                raw_val, bit_pos = read_bits(packed, bit_pos, 16)
                if raw_val >= 32768:
                    raw_val -= 65536
                output[i] = raw_val

    # 4. Reshape and convert to bfloat16 tensor
    shape = header.shape[:header.ndim]
    tensor = torch.from_numpy(output.copy()).view(torch.bfloat16).reshape(shape)
    return tensor


def decode_f32_tensor(f, header: TensorHeader) -> torch.Tensor:
    """Read an F32 passthrough tensor (stored uncompressed)."""
    f.seek(header.data_offset)
    raw = np.frombuffer(f.read(header.data_size_bytes), dtype=np.float32)
    return torch.from_numpy(raw.copy()).reshape(header.shape[:header.ndim])


def decode_tlc(tlc_path: str) -> dict[str, torch.Tensor]:
    """Decode a .tlc file and return a dict mapping tensor names to tensors."""
    tensors = {}

    with open(tlc_path, "rb") as f:
        # 1. Read and validate file header
        version, num_tensors = read_file_header(f)

        # 2. Read name table
        names = read_name_table(f, num_tensors)

        # Align to 4-byte boundary before tensor headers
        pos = f.tell()
        pad = (4 - pos % 4) % 4
        if pad:
            f.read(pad)

        # 3. Read all tensor headers
        headers = []
        for _ in range(num_tensors):
            hdr = read_tensor_header(f)
            headers.append(hdr)

        # 4. Decode each tensor
        for idx, hdr in enumerate(headers):
            name = names[idx]
            t0 = time.perf_counter()

            if hdr.dtype == DTYPE_BF16_COMPRESSED:
                tensor = decode_tensor(f, hdr)
            elif hdr.dtype == DTYPE_F32_PASSTHROUGH:
                tensor = decode_f32_tensor(f, hdr)
            else:
                raise ValueError(f"Unknown dtype {hdr.dtype} for tensor '{name}'")

            elapsed = time.perf_counter() - t0
            print(f"  [{idx+1}/{len(headers)}] {name}: "
                  f"{hdr.num_elements:,} elements, {elapsed:.3f}s")

            tensors[name] = tensor

    return tensors


def save_as_safetensors(tensors: dict[str, torch.Tensor], output_path: str) -> None:
    """Save decoded tensors back to safetensors format."""
    from safetensors.torch import save_file
    save_file(tensors, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tlc_decode.py <input.tlc> [output.safetensors]")
        sys.exit(1)

    tlc_path = sys.argv[1]
    print(f"Decoding {tlc_path} ...")
    t_start = time.perf_counter()

    tensors = decode_tlc(tlc_path)
    t_total = time.perf_counter() - t_start
    print(f"Decoded {len(tensors)} tensors in {t_total:.2f}s")

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
        save_as_safetensors(tensors, out_path)
        print(f"Saved to {out_path}")
