"""
TLC Runtime — loads .tlc compressed models and dispatches GPU kernels
for fused decompression + matrix-vector multiply via HIP/CUDA.
"""

import ctypes
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from tlc_format import (
    DTYPE_BF16_COMPRESSED,
    DTYPE_F32_PASSTHROUGH,
    NUM_CODED_TIERS,
    BLOCK_SIZE,
    BLOCK_PTR_ENTRY_SIZE,
    read_file_header,
    read_name_table,
    read_tensor_header,
    TensorHeader,
)


# ---------------------------------------------------------------------------
# TLCTensor — per-tensor GPU data
# ---------------------------------------------------------------------------

@dataclass
class TLCTensor:
    name: str
    shape: tuple                    # e.g. (4096, 4096)
    dtype_flag: int                 # DTYPE_BF16_COMPRESSED or DTYPE_F32_PASSTHROUGH
    index_bits: int
    num_elements: int
    packed_data_gpu: torch.Tensor   # int32 on device (bit-identical to uint32)
    codebook_gpu: torch.Tensor      # int16 on device
    block_ptrs_gpu: torch.Tensor    # int32 on device (pairs: bit_offset, escape_index)
    # For F32 passthrough tensors:
    f32_data_gpu: Optional[torch.Tensor] = None  # float32 on device


# ---------------------------------------------------------------------------
# TLCModel — loads a .tlc file and provides GPU inference primitives
# ---------------------------------------------------------------------------

class TLCModel:
    def __init__(self, tlc_path: str, device: str = "cuda:0"):
        """Load a .tlc file and transfer all tensor data to GPU.

        Parameters
        ----------
        tlc_path : str
            Path to the .tlc compressed model file.
        device : str
            PyTorch device string (e.g. "cuda:0").
        """
        self.tlc_path = os.path.abspath(tlc_path)
        self.device = torch.device(device)
        self.tensors: Dict[str, TLCTensor] = {}
        self._lib = None

        t_start = time.perf_counter()
        self._load_file()
        t_load = time.perf_counter() - t_start

        self._load_kernel()

        print(f"TLCModel: loaded {len(self.tensors)} tensors from "
              f"{self.tlc_path} in {t_load:.2f}s")

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_file(self) -> None:
        """Parse the .tlc file and upload every tensor's data to the GPU."""
        with open(self.tlc_path, "rb") as f:
            # 1. File header
            version, num_tensors = read_file_header(f)

            # 2. Name table
            names = read_name_table(f, num_tensors)

            # 3. Align to 4-byte boundary before tensor headers
            pos = f.tell()
            pad = (4 - pos % 4) % 4
            if pad:
                f.read(pad)

            # 4. Read all tensor headers
            headers = []
            for _ in range(num_tensors):
                hdr = read_tensor_header(f)
                headers.append(hdr)

            # 5. Load each tensor's data to GPU
            for idx, hdr in enumerate(headers):
                name = names[idx]
                shape = hdr.shape[:hdr.ndim]

                if hdr.dtype == DTYPE_BF16_COMPRESSED:
                    tlc_t = self._load_compressed_tensor(f, name, shape, hdr)
                elif hdr.dtype == DTYPE_F32_PASSTHROUGH:
                    tlc_t = self._load_f32_tensor(f, name, shape, hdr)
                else:
                    raise ValueError(
                        f"Unknown dtype flag {hdr.dtype} for tensor '{name}'"
                    )
                self.tensors[name] = tlc_t

    def _load_compressed_tensor(
        self,
        f,
        name: str,
        shape: tuple,
        hdr: TensorHeader,
    ) -> TLCTensor:
        """Read codebook, packed bitstream, and block pointers for one
        BF16-compressed tensor, upload to GPU."""

        # Codebook: NUM_CODED_TIERS * tier_size entries, each int16
        f.seek(hdr.codebook_offset)
        tier_size = 1 << hdr.index_bits
        codebook_entries = NUM_CODED_TIERS * tier_size
        codebook_bytes = f.read(codebook_entries * 2)
        codebook_np = np.frombuffer(codebook_bytes, dtype=np.int16).copy()
        codebook_gpu = torch.from_numpy(codebook_np).to(self.device)

        # Packed bitstream: uint32 words
        f.seek(hdr.data_offset)
        packed_bytes = f.read(hdr.data_size_bytes)
        packed_np = np.frombuffer(packed_bytes, dtype=np.uint32).copy()
        # torch has no uint32 — store as int32; the bits are identical
        packed_gpu = torch.from_numpy(packed_np.view(np.int32)).to(self.device)

        # Block pointer table: pairs of (uint32 bit_offset, uint32 escape_index)
        f.seek(hdr.block_ptr_offset)
        bp_byte_count = hdr.block_ptr_count * BLOCK_PTR_ENTRY_SIZE
        bp_bytes = f.read(bp_byte_count)
        bp_np = np.frombuffer(bp_bytes, dtype=np.uint32).copy()
        bp_gpu = torch.from_numpy(bp_np.view(np.int32)).to(self.device)

        return TLCTensor(
            name=name,
            shape=shape,
            dtype_flag=DTYPE_BF16_COMPRESSED,
            index_bits=hdr.index_bits,
            num_elements=hdr.num_elements,
            packed_data_gpu=packed_gpu,
            codebook_gpu=codebook_gpu,
            block_ptrs_gpu=bp_gpu,
        )

    def _load_f32_tensor(
        self,
        f,
        name: str,
        shape: tuple,
        hdr: TensorHeader,
    ) -> TLCTensor:
        """Read an F32 passthrough tensor and upload to GPU."""
        f.seek(hdr.data_offset)
        raw_bytes = f.read(hdr.data_size_bytes)
        f32_np = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        f32_gpu = torch.from_numpy(f32_np).to(self.device)

        return TLCTensor(
            name=name,
            shape=shape,
            dtype_flag=DTYPE_F32_PASSTHROUGH,
            index_bits=0,
            num_elements=hdr.num_elements,
            packed_data_gpu=torch.empty(0, dtype=torch.int32, device=self.device),
            codebook_gpu=torch.empty(0, dtype=torch.int16, device=self.device),
            block_ptrs_gpu=torch.empty(0, dtype=torch.int32, device=self.device),
            f32_data_gpu=f32_gpu,
        )

    # ------------------------------------------------------------------
    # Kernel loading via ctypes
    # ------------------------------------------------------------------

    def _load_kernel(self) -> None:
        """Load the compiled HIP/CUDA decompression kernel (.so)."""
        lib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "decompress_matmul.so",
        )
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Kernel shared library not found at {lib_path}. "
                f"Build it with:\n"
                f"  hipcc -O3 --offload-arch=gfx906 -shared -fPIC "
                f"-o decompress_matmul.so decompress_matmul.hip\n"
                f"or:\n"
                f"  nvcc -O3 -shared -Xcompiler -fPIC "
                f"-o decompress_matmul.so decompress_matmul.cu"
            )

        self._lib = ctypes.CDLL(lib_path)

        # launch_decompress_only(packed, codebook, block_ptrs, output,
        #                        num_elements, index_bits) -> int
        self._lib.launch_decompress_only.argtypes = [
            ctypes.c_void_p,   # packed_data
            ctypes.c_void_p,   # codebook
            ctypes.c_void_p,   # block_ptrs
            ctypes.c_void_p,   # output (int16)
            ctypes.c_int,      # num_elements
            ctypes.c_int,      # index_bits
        ]
        self._lib.launch_decompress_only.restype = ctypes.c_int

        # launch_decompress_matvec(packed, codebook, block_ptrs, activations,
        #                          output, M, K, index_bits, blocks_per_row) -> int
        self._lib.launch_decompress_matvec.argtypes = [
            ctypes.c_void_p,   # packed_data
            ctypes.c_void_p,   # codebook
            ctypes.c_void_p,   # block_ptrs
            ctypes.c_void_p,   # activations (int16 view of bf16)
            ctypes.c_void_p,   # output (float32)
            ctypes.c_int,      # M (rows)
            ctypes.c_int,      # K (cols)
            ctypes.c_int,      # index_bits
            ctypes.c_int,      # blocks_per_row
        ]
        self._lib.launch_decompress_matvec.restype = ctypes.c_int

    # ------------------------------------------------------------------
    # Kernel launch wrappers
    # ------------------------------------------------------------------

    def _launch_decompress_only(
        self,
        packed_ptr: int,
        codebook_ptr: int,
        block_ptrs_ptr: int,
        output_ptr: int,
        num_elements: int,
        index_bits: int,
    ) -> None:
        """Call the decompress-only GPU kernel."""
        ret = self._lib.launch_decompress_only(
            ctypes.c_void_p(packed_ptr),
            ctypes.c_void_p(codebook_ptr),
            ctypes.c_void_p(block_ptrs_ptr),
            ctypes.c_void_p(output_ptr),
            ctypes.c_int(num_elements),
            ctypes.c_int(index_bits),
        )
        if ret != 0:
            raise RuntimeError(
                f"launch_decompress_only returned error code {ret}"
            )

    def _launch_matvec(
        self,
        packed_ptr: int,
        codebook_ptr: int,
        block_ptrs_ptr: int,
        activations_ptr: int,
        output_ptr: int,
        M: int,
        K: int,
        index_bits: int,
        blocks_per_row: int,
    ) -> None:
        """Call the fused decompress + matvec GPU kernel."""
        ret = self._lib.launch_decompress_matvec(
            ctypes.c_void_p(packed_ptr),
            ctypes.c_void_p(codebook_ptr),
            ctypes.c_void_p(block_ptrs_ptr),
            ctypes.c_void_p(activations_ptr),
            ctypes.c_void_p(output_ptr),
            ctypes.c_int(M),
            ctypes.c_int(K),
            ctypes.c_int(index_bits),
            ctypes.c_int(blocks_per_row),
        )
        if ret != 0:
            raise RuntimeError(
                f"launch_decompress_matvec returned error code {ret}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def matvec(self, tensor_name: str, x: torch.Tensor) -> torch.Tensor:
        """Compute W @ x where W is a compressed BF16 tensor.

        Parameters
        ----------
        tensor_name : str
            Name of the tensor (must be DTYPE_BF16_COMPRESSED).
        x : torch.Tensor
            Activation vector on the same device.  Accepted dtypes:
            bfloat16, float32, int16 (raw bf16 bits).

        Returns
        -------
        torch.Tensor
            Result vector of shape (M,) in float32.
        """
        t = self.tensors[tensor_name]
        assert t.dtype_flag == DTYPE_BF16_COMPRESSED, (
            f"matvec requires a BF16-compressed tensor, "
            f"but '{tensor_name}' has dtype_flag={t.dtype_flag}"
        )

        M, K = t.shape
        blocks_per_row = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

        # The kernel reads activations as int16 (raw BF16 bits)
        if x.dtype == torch.bfloat16:
            x_int16 = x.view(torch.int16)
        elif x.dtype == torch.float32:
            x_int16 = x.to(torch.bfloat16).view(torch.int16)
        else:
            x_int16 = x

        output = torch.empty(M, dtype=torch.float32, device=self.device)

        self._launch_matvec(
            t.packed_data_gpu.data_ptr(),
            t.codebook_gpu.data_ptr(),
            t.block_ptrs_gpu.data_ptr(),
            x_int16.data_ptr(),
            output.data_ptr(),
            M,
            K,
            t.index_bits,
            blocks_per_row,
        )
        return output

    def decompress(self, tensor_name: str) -> torch.Tensor:
        """Decompress a tensor fully on GPU.

        Parameters
        ----------
        tensor_name : str
            Name of the tensor to decompress.

        Returns
        -------
        torch.Tensor
            For BF16-compressed tensors: bfloat16 tensor with original shape.
            For F32 passthrough tensors: float32 tensor with original shape.
        """
        t = self.tensors[tensor_name]

        if t.dtype_flag == DTYPE_F32_PASSTHROUGH:
            return t.f32_data_gpu.reshape(t.shape)

        # BF16 compressed — decompress via GPU kernel
        output_int16 = torch.empty(
            t.num_elements, dtype=torch.int16, device=self.device
        )

        self._launch_decompress_only(
            t.packed_data_gpu.data_ptr(),
            t.codebook_gpu.data_ptr(),
            t.block_ptrs_gpu.data_ptr(),
            output_int16.data_ptr(),
            t.num_elements,
            t.index_bits,
        )
        return output_int16.view(torch.bfloat16).reshape(t.shape)

    def memory_report(self) -> dict:
        """Report GPU memory usage across all loaded tensors.

        Returns
        -------
        dict
            Keys: total_gpu_mb (float), num_tensors (int),
            per_tensor (dict mapping name -> size in bytes).
        """
        total = 0
        per_tensor = {}
        for name, t in self.tensors.items():
            if t.dtype_flag == DTYPE_BF16_COMPRESSED:
                size = (
                    t.packed_data_gpu.nbytes
                    + t.codebook_gpu.nbytes
                    + t.block_ptrs_gpu.nbytes
                )
            else:
                size = t.f32_data_gpu.nbytes if t.f32_data_gpu is not None else 0
            per_tensor[name] = size
            total += size
        return {
            "total_gpu_mb": total / (1024 * 1024),
            "num_tensors": len(self.tensors),
            "per_tensor": per_tensor,
        }

    def tensor_names(self) -> list:
        """Return list of all tensor names in load order."""
        return list(self.tensors.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tlc_runtime.py <model.tlc> [device]")
        sys.exit(1)

    tlc_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"

    model = TLCModel(tlc_path, device=device)
    report = model.memory_report()
    print(f"Loaded {report['num_tensors']} tensors, "
          f"{report['total_gpu_mb']:.1f} MB on GPU")

    # Print per-tensor summary
    for name in model.tensor_names():
        t = model.tensors[name]
        size_bytes = report["per_tensor"][name]
        dtype_str = ("BF16-compressed" if t.dtype_flag == DTYPE_BF16_COMPRESSED
                     else "F32-passthrough")
        print(f"  {name}: {t.shape} {dtype_str} "
              f"({size_bytes / 1024:.1f} KB on GPU)")
