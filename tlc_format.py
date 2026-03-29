"""
Turbo Lossless Compression (.tlc) binary format definitions.
Shared by encoder and decoder.
"""
import struct
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# File format constants
# ---------------------------------------------------------------------------
MAGIC = b"TLC8"
VERSION = 1
FILE_HEADER_SIZE = 9  # 4 (magic) + 1 (version) + 4 (num_tensors)

# Dtype codes
DTYPE_BF16_COMPRESSED = 0
DTYPE_F32_PASSTHROUGH = 1

# 8-tier coding constants
NUM_CODED_TIERS = 7
ESCAPE_PREFIX_BITS = 7   # 1111111
ESCAPE_RAW_BITS = 16
ESCAPE_TOTAL_BITS = 23   # 7 + 16
MAX_SHAPE_DIMS = 8
BLOCK_SIZE = 256          # block pointer every 256 values
BLOCK_PTR_ENTRY_SIZE = 8  # uint32 bit_offset + uint32 escape_index

# Tensor header struct format (little-endian, packed)
#   name_offset    : I  (uint32)   4
#   name_len       : H  (uint16)   2
#   dtype          : B  (uint8)    1
#   ndim           : B  (uint8)    1
#   shape[8]       : 8I (uint32x8) 32
#   num_elements   : Q  (uint64)   8
#   codebook_offset: Q  (uint64)   8
#   data_offset    : Q  (uint64)   8
#   data_size_bytes: Q  (uint64)   8
#   block_ptr_offset: Q (uint64)   8
#   block_ptr_count: I  (uint32)   4
#   escape_offset  : Q  (uint64)   8
#   escape_count   : I  (uint32)   4
#   index_bits     : B  (uint8)    1
#   padding        : 3x            3
# Total = 100 bytes
_TENSOR_HEADER_FMT = "<IHBBIIIIIIIIQQQQQIQIBxxx"
TENSOR_HEADER_SIZE = struct.calcsize(_TENSOR_HEADER_FMT)  # 100


# ---------------------------------------------------------------------------
# TensorHeader dataclass
# ---------------------------------------------------------------------------
@dataclass
class TensorHeader:
    name_offset: int        # uint32 - offset into name table
    name_len: int           # uint16 - byte length of name
    dtype: int              # uint8  - DTYPE_* code
    ndim: int               # uint8  - number of dimensions
    shape: tuple            # up to MAX_SHAPE_DIMS uint32 values
    num_elements: int       # uint64
    codebook_offset: int    # uint64 - byte offset of codebook in file
    data_offset: int        # uint64 - byte offset of packed bitstream
    data_size_bytes: int    # uint64 - size of packed bitstream in bytes
    block_ptr_offset: int   # uint64 - byte offset of block pointer table
    block_ptr_count: int    # uint32 - number of block pointer entries
    escape_offset: int      # uint64 - byte offset of overflow stream
    escape_count: int       # uint32 - number of escape values
    index_bits: int         # uint8  - bits per sub-dictionary index


# ---------------------------------------------------------------------------
# File header
# ---------------------------------------------------------------------------
def write_file_header(f, num_tensors: int) -> None:
    """Write the 9-byte file header: magic + version + num_tensors."""
    f.write(MAGIC)
    f.write(struct.pack("<B", VERSION))
    f.write(struct.pack("<I", num_tensors))


def read_file_header(f) -> Tuple[int, int]:
    """Read and validate magic, return (version, num_tensors)."""
    magic = f.read(4)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: expected {MAGIC!r}, got {magic!r}")
    version = struct.unpack("<B", f.read(1))[0]
    num_tensors = struct.unpack("<I", f.read(4))[0]
    return version, num_tensors


# ---------------------------------------------------------------------------
# Name table
# ---------------------------------------------------------------------------
def write_name_table(f, names: List[str]) -> List[int]:
    """Write tensor names sequentially (uint16 len + UTF-8 bytes).

    Returns a list of byte offsets for each name, relative to the start
    of the name table.
    """
    offsets = []
    pos = 0
    for name in names:
        offsets.append(pos)
        encoded = name.encode("utf-8")
        length = len(encoded)
        f.write(struct.pack("<H", length))
        f.write(encoded)
        pos += 2 + length
    return offsets


def read_name_table(f, num_tensors: int) -> List[str]:
    """Read all tensor names from the name table."""
    names = []
    for _ in range(num_tensors):
        length = struct.unpack("<H", f.read(2))[0]
        name = f.read(length).decode("utf-8")
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Tensor header
# ---------------------------------------------------------------------------
def write_tensor_header(f, header: TensorHeader) -> None:
    """Pack and write the tensor header."""
    # Pad shape to MAX_SHAPE_DIMS entries
    padded_shape = list(header.shape) + [0] * (MAX_SHAPE_DIMS - len(header.shape))
    f.write(struct.pack(
        _TENSOR_HEADER_FMT,
        header.name_offset,
        header.name_len,
        header.dtype,
        header.ndim,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        padded_shape[4],
        padded_shape[5],
        padded_shape[6],
        padded_shape[7],
        header.num_elements,
        header.codebook_offset,
        header.data_offset,
        header.data_size_bytes,
        header.block_ptr_offset,
        header.block_ptr_count,
        header.escape_offset,
        header.escape_count,
        header.index_bits,
    ))


def read_tensor_header(f) -> TensorHeader:
    """Unpack the tensor header from the current file position."""
    data = f.read(TENSOR_HEADER_SIZE)
    if len(data) != TENSOR_HEADER_SIZE:
        raise ValueError(
            f"Incomplete tensor header: expected {TENSOR_HEADER_SIZE} bytes, "
            f"got {len(data)}"
        )
    fields = struct.unpack(_TENSOR_HEADER_FMT, data)
    name_offset = fields[0]
    name_len = fields[1]
    dtype = fields[2]
    ndim = fields[3]
    raw_shape = fields[4:12]
    shape = tuple(raw_shape[:ndim])
    num_elements = fields[12]
    codebook_offset = fields[13]
    data_offset = fields[14]
    data_size_bytes = fields[15]
    block_ptr_offset = fields[16]
    block_ptr_count = fields[17]
    escape_offset = fields[18]
    escape_count = fields[19]
    index_bits = fields[20]
    return TensorHeader(
        name_offset=name_offset,
        name_len=name_len,
        dtype=dtype,
        ndim=ndim,
        shape=shape,
        num_elements=num_elements,
        codebook_offset=codebook_offset,
        data_offset=data_offset,
        data_size_bytes=data_size_bytes,
        block_ptr_offset=block_ptr_offset,
        block_ptr_count=block_ptr_count,
        escape_offset=escape_offset,
        escape_count=escape_count,
        index_bits=index_bits,
    )


# ---------------------------------------------------------------------------
# Bitstream read/write helpers (MSB-first within each uint32 word)
#
# Bit ordering: bit_pos 0 is bit 31 of word 0, bit_pos 1 is bit 30 of
# word 0, ..., bit_pos 31 is bit 0 of word 0, bit_pos 32 is bit 31 of
# word 1, and so on.
# ---------------------------------------------------------------------------

def write_bits(buf: np.ndarray, bit_pos: int, value: int, num_bits: int) -> int:
    """Write *num_bits* from *value* into a uint32 buffer at *bit_pos*.

    MSB-first packing: the most-significant bit of *value* is written to
    the earliest bit position.  Returns the new bit_pos after writing.
    """
    for i in range(num_bits):
        # Extract bits from value MSB-first
        bit = (value >> (num_bits - 1 - i)) & 1
        word_idx = (bit_pos + i) >> 5          # // 32
        bit_idx = 31 - ((bit_pos + i) & 31)   # MSB-first within word
        if bit:
            buf[word_idx] = np.uint32(int(buf[word_idx]) | (1 << bit_idx))
        else:
            buf[word_idx] = np.uint32(int(buf[word_idx]) & ~(1 << bit_idx))
    return bit_pos + num_bits


def read_bits(buf: np.ndarray, bit_pos: int, num_bits: int) -> Tuple[int, int]:
    """Read *num_bits* from a uint32 buffer at *bit_pos*.

    MSB-first ordering.  Returns (value, new_bit_pos).
    """
    value = 0
    for i in range(num_bits):
        word_idx = (bit_pos + i) >> 5
        bit_idx = 31 - ((bit_pos + i) & 31)
        bit = (int(buf[word_idx]) >> bit_idx) & 1
        value = (value << 1) | bit
    return value, bit_pos + num_bits


def read_bit(buf: np.ndarray, bit_pos: int) -> Tuple[int, int]:
    """Read a single bit from a uint32 buffer at *bit_pos*.

    Returns (bit_value, new_bit_pos).
    """
    word_idx = bit_pos >> 5
    bit_idx = 31 - (bit_pos & 31)
    bit = (int(buf[word_idx]) >> bit_idx) & 1
    return bit, bit_pos + 1
