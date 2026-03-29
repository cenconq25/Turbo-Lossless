"""Turbo Lossless: Round-trip verification script.

Encodes a safetensors model to .tlc, decodes it back, and verifies
bit-perfect lossless reconstruction of every tensor.
"""

from tlc_encode import encode_model, discover_shards
from tlc_decode import decode_tlc
from safetensors import safe_open
import torch
import os
import sys
import time


def _dtype_label(dtype: torch.dtype) -> str:
    """Return a short human-readable label for a torch dtype."""
    labels = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    return labels.get(dtype, str(dtype))


def _format_bytes(n: int) -> str:
    """Format byte count as a human-readable string."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    elif n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    elif n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KB"
    return f"{n} B"


def verify(input_path: str, tlc_path: str = None) -> bool:
    """Encode, decode, and verify bit-perfect round-trip.

    Args:
        input_path: Path to a .safetensors file or directory with shards.
        tlc_path: Output .tlc path. If None, derived from input_path.

    Returns:
        True if every tensor is bit-perfect, False otherwise.
    """
    # ------------------------------------------------------------------ paths
    if tlc_path is None:
        if os.path.isfile(input_path):
            tlc_path = input_path.rsplit(".", 1)[0] + ".tlc"
        else:
            tlc_path = os.path.join(input_path, "model.tlc")

    # --------------------------------------------------------------- encode
    print("[Encoding...]")
    encode_model(input_path, tlc_path)
    print()

    # --------------------------------------------------------------- decode
    print("[Decoding...]")
    decoded_tensors = decode_tlc(tlc_path)
    print()

    # ------------------------------------------------- load originals & compare
    print("[Verifying...]")
    shards = discover_shards(input_path)

    # Collect all original tensor names and shard mapping
    original_names: dict[str, str] = {}  # tensor_name -> shard_path
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as st:
            for name in st.keys():
                original_names[name] = shard_path

    orig_name_set = set(original_names.keys())
    dec_name_set = set(decoded_tensors.keys())

    all_ok = True
    total_tensors = 0
    total_params = 0
    mismatch_tensors = 0

    # Check for name mismatches
    missing_in_decoded = orig_name_set - dec_name_set
    extra_in_decoded = dec_name_set - orig_name_set

    if missing_in_decoded:
        all_ok = False
        for name in sorted(missing_in_decoded):
            print(f"FAIL {name}  -- MISSING in decoded output")
            mismatch_tensors += 1

    if extra_in_decoded:
        all_ok = False
        for name in sorted(extra_in_decoded):
            print(f"FAIL {name}  -- EXTRA in decoded output (not in original)")
            mismatch_tensors += 1

    # Verify each tensor that exists in both sets, loading one shard at a time
    # to control memory usage
    verified_names: set[str] = set()
    for shard_path in shards:
        with safe_open(shard_path, framework="pt", device="cpu") as st:
            for name in st.keys():
                if name in verified_names:
                    continue
                verified_names.add(name)
                total_tensors += 1

                if name not in dec_name_set:
                    # Already reported above
                    continue

                orig = st.get_tensor(name)
                dec = decoded_tensors[name]
                total_params += orig.numel()

                dtype_str = _dtype_label(orig.dtype)
                shape_str = list(orig.shape)

                # Check shape
                if orig.shape != dec.shape:
                    print(f"FAIL {name:<55s} {str(shape_str):<18s} {dtype_str}  "
                          f"-- shape mismatch: orig {orig.shape} vs dec {dec.shape}")
                    all_ok = False
                    mismatch_tensors += 1
                    continue

                # Check dtype
                if orig.dtype != dec.dtype:
                    print(f"FAIL {name:<55s} {str(shape_str):<18s} {dtype_str}  "
                          f"-- dtype mismatch: orig {orig.dtype} vs dec {dec.dtype}")
                    all_ok = False
                    mismatch_tensors += 1
                    continue

                # Bit-level comparison
                if orig.dtype == torch.bfloat16:
                    orig_bits = orig.view(torch.int16)
                    dec_bits = dec.view(torch.int16)
                    match = torch.equal(orig_bits, dec_bits)
                elif orig.dtype == torch.float32:
                    orig_bits = orig.view(torch.int32)
                    dec_bits = dec.view(torch.int32)
                    match = torch.equal(orig_bits, dec_bits)
                elif orig.dtype == torch.float16:
                    orig_bits = orig.view(torch.int16)
                    dec_bits = dec.view(torch.int16)
                    match = torch.equal(orig_bits, dec_bits)
                elif orig.dtype == torch.float64:
                    orig_bits = orig.view(torch.int64)
                    dec_bits = dec.view(torch.int64)
                    match = torch.equal(orig_bits, dec_bits)
                else:
                    # Integer or other types: direct comparison
                    match = torch.equal(orig, dec)

                if match:
                    print(f"OK   {name:<55s} {str(shape_str):<18s} {dtype_str}")
                else:
                    # Count mismatches for diagnostic info
                    if orig.dtype in (torch.bfloat16, torch.float16):
                        diff_mask = orig.view(torch.int16) != dec.view(torch.int16)
                    elif orig.dtype == torch.float32:
                        diff_mask = orig.view(torch.int32) != dec.view(torch.int32)
                    elif orig.dtype == torch.float64:
                        diff_mask = orig.view(torch.int64) != dec.view(torch.int64)
                    else:
                        diff_mask = orig != dec

                    num_mismatches = diff_mask.sum().item()
                    pct = 100.0 * num_mismatches / orig.numel()
                    print(f"FAIL {name:<55s} {str(shape_str):<18s} {dtype_str}  "
                          f"-- {num_mismatches:,} mismatches ({pct:.4f}%)")
                    all_ok = False
                    mismatch_tensors += 1

    # ------------------------------------------------------------- file sizes
    print()
    print("[File Size]")

    original_total = 0
    for shard_path in shards:
        original_total += os.path.getsize(shard_path)

    tlc_size = os.path.getsize(tlc_path)

    ratio = original_total / tlc_size if tlc_size > 0 else float("inf")
    saved = original_total - tlc_size
    saved_pct = 100.0 * saved / original_total if original_total > 0 else 0.0

    print(f"Original:   {_format_bytes(original_total)} (safetensors)")
    print(f"Compressed: {_format_bytes(tlc_size)} (tlc)")
    print(f"Ratio:      {ratio:.3f}x")
    print(f"Saved:      {_format_bytes(saved)} ({saved_pct:.1f}%)")

    # ------------------------------------------------------------- summary
    print()
    print(f"Total tensors: {total_tensors}")
    print(f"Total params:  {total_params:,}")
    if mismatch_tensors > 0:
        print(f"Mismatched:    {mismatch_tensors}")

    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tlc_verify.py <safetensors_path> [output.tlc]")
        print()
        print("Encodes the model, decodes it, and verifies bit-perfect match.")
        print("safetensors_path: path to .safetensors file or directory with shards")
        sys.exit(1)

    input_path = sys.argv[1]
    tlc_path = sys.argv[2] if len(sys.argv) >= 3 else None

    if tlc_path is None:
        # Generate output path
        if os.path.isfile(input_path):
            tlc_path = input_path.rsplit(".", 1)[0] + ".tlc"
        else:
            tlc_path = os.path.join(input_path, "model.tlc")

    print("=" * 70)
    print("Turbo Lossless: Round-Trip Verification")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {tlc_path}")
    print()

    t0 = time.time()
    success = verify(input_path, tlc_path)
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    if success:
        print(f"RESULT: ALL TENSORS BIT-PERFECT  ({elapsed:.1f}s)")
    else:
        print(f"RESULT: VERIFICATION FAILED  ({elapsed:.1f}s)")
    print("=" * 70)

    sys.exit(0 if success else 1)
