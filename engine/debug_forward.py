#!/usr/bin/env python3
"""
Debug forward pass: compare reference (torch BF16) vs compressed kernel
for layer 0 of Mistral 7B, step by step.

Identifies WHERE values first diverge between reference and engine.
"""

import os, sys, struct, math
import numpy as np
import torch

os.environ["HIP_VISIBLE_DEVICES"] = "1"

# ---- paths ----
MODEL_DIR  = "/home/ubuntu/AI/ compression/models/mistral-7b"
TURBO_DIR  = "/home/ubuntu/AI/ compression/models/mistral-7b-turbo"

# ---- load compressed matvec kernel via ctypes ----
sys.path.insert(0, "/home/ubuntu/AI/ compression")
from bench_fixed12 import _hip_lib

DEVICE = "cuda:0"  # maps to GPU 1 via HIP_VISIBLE_DEVICES
WORKGROUP_SIZE = 256


# ==============================================================
# Utility functions
# ==============================================================

def show(name, t, n=5):
    """Print summary of a tensor: norm, first n values."""
    t_f = t.float() if t.dtype != torch.float32 else t
    norm = t_f.norm().item()
    vals = t_f.flatten()[:n].tolist()
    print(f"  {name:30s}  norm={norm:12.4f}  first{n}={[f'{v:.6f}' for v in vals]}")
    return norm


def compare(name, ref, ours, rtol=1e-3, atol=1e-4):
    """Compare two tensors, report max diff."""
    ref_f = ref.float().cpu() if ref.dtype != torch.float32 else ref.cpu()
    ours_f = ours.float().cpu() if ours.dtype != torch.float32 else ours.cpu()

    if ref_f.shape != ours_f.shape:
        print(f"  *** SHAPE MISMATCH at {name}: ref={ref_f.shape} ours={ours_f.shape}")
        return False

    diff = (ref_f - ours_f).abs()
    max_diff = diff.max().item()
    max_idx = diff.argmax().item()
    ref_norm = ref_f.norm().item()
    ours_norm = ours_f.norm().item()
    rel_err = max_diff / (ref_f.abs().max().item() + 1e-30)

    ok = max_diff < atol or rel_err < rtol
    status = "OK" if ok else "*** DIVERGED ***"

    print(f"  {name:30s}  max_diff={max_diff:.6e}  rel_err={rel_err:.6e}  "
          f"ref_norm={ref_norm:.4f}  ours_norm={ours_norm:.4f}  {status}")
    if not ok:
        print(f"    worst at index {max_idx}: ref={ref_f.flatten()[max_idx]:.6f} "
              f"ours={ours_f.flatten()[max_idx]:.6f}")
        # Show first few elements
        n = min(8, ref_f.numel())
        print(f"    ref  first{n}: {ref_f.flatten()[:n].tolist()}")
        print(f"    ours first{n}: {ours_f.flatten()[:n].tolist()}")
    return ok


# ==============================================================
# Load reference weights from safetensors
# ==============================================================

def load_safetensors():
    """Load layer 0 + embeddings from original Mistral 7B safetensors."""
    from safetensors import safe_open
    import glob

    tensors = {}
    shards = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    for shard in shards:
        f = safe_open(shard, framework="pt")
        for name in f.keys():
            # Only load what we need
            if ("layers.0." in name or "embed_tokens" in name or
                "model.norm" in name or "lm_head" in name):
                tensors[name] = f.get_tensor(name)
    return tensors


# ==============================================================
# Load compressed weight data from turbo directory
# ==============================================================

def load_compressed_weight(prefix):
    """Load a compressed weight tensor from the turbo directory."""
    dims_path = os.path.join(TURBO_DIR, f"{prefix}.dims")
    with open(dims_path) as f:
        parts = f.read().split()
        M, K, num_patches = int(parts[0]), int(parts[1]), int(parts[2])

    packed = np.fromfile(os.path.join(TURBO_DIR, f"{prefix}.packed.bin"), dtype=np.int32)
    codebook = np.fromfile(os.path.join(TURBO_DIR, f"{prefix}.codebook.bin"), dtype=np.int16)
    esc_off = np.fromfile(os.path.join(TURBO_DIR, f"{prefix}.esc_off.bin"), dtype=np.int32)
    esc_val = np.fromfile(os.path.join(TURBO_DIR, f"{prefix}.esc_val.bin"), dtype=np.int16)

    # Upload to GPU
    packed_g = torch.from_numpy(packed).to(torch.int32).to(DEVICE)
    codebook_g = torch.from_numpy(codebook).to(torch.int16).to(DEVICE)
    esc_off_g = torch.from_numpy(esc_off).to(torch.int32).to(DEVICE)
    esc_val_g = torch.from_numpy(esc_val).to(torch.int16).to(DEVICE)

    return {
        "packed": packed_g, "codebook": codebook_g,
        "escape_offsets": esc_off_g, "escape_vals": esc_val_g,
        "M": M, "K": K, "num_patches": num_patches
    }


def compressed_matvec(w, x_fp32):
    """Run our fused 12-bit matvec kernel.
    w: dict from load_compressed_weight
    x_fp32: [K] float32 tensor on GPU
    Returns: [M] float32 tensor on GPU
    """
    M, K = w["M"], w["K"]
    # Convert FP32 activation to BF16 (matching what inference.cpp does)
    x_bf16 = x_fp32.to(torch.bfloat16).view(torch.int16)

    output = torch.zeros(M, dtype=torch.float32, device=DEVICE)

    ret = _hip_lib.launch_fixed12_fused(
        w["packed"].data_ptr(),
        w["codebook"].data_ptr(),
        x_bf16.data_ptr(),
        w["escape_offsets"].data_ptr(),
        w["escape_vals"].data_ptr(),
        output.data_ptr(),
        M, K
    )
    if ret != 0:
        print(f"  *** KERNEL LAUNCH FAILED (ret={ret})")
    return output


# ==============================================================
# Reference operations (pure torch)
# ==============================================================

def rms_norm(x, weight, eps=1e-5):
    """RMSNorm in float32."""
    x_f = x.float()
    w_f = weight.float()
    rms = torch.rsqrt(x_f.pow(2).mean() + eps)
    return (x_f * rms * w_f)


def apply_rope(x, head_dim, n_heads, position, theta=10000.0):
    """Apply RoPE to [n_heads, head_dim] tensor at given position."""
    x = x.clone()
    half_dim = head_dim // 2
    for h in range(n_heads):
        for d in range(half_dim):
            freq = 1.0 / (theta ** (2.0 * d / head_dim))
            angle = position * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            base = h * head_dim
            x0 = x[base + d].item()
            x1 = x[base + d + half_dim].item()
            x[base + d] = x0 * cos_a - x1 * sin_a
            x[base + d + half_dim] = x0 * sin_a + x1 * cos_a
    return x


def silu(x):
    return x * torch.sigmoid(x)


# ==============================================================
# Main debug routine
# ==============================================================

def main():
    print("=" * 90)
    print("Debug Forward Pass: Mistral 7B Layer 0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)
    print()

    # Model config
    n_embd = 4096
    n_head = 32
    n_head_kv = 8
    head_dim = n_embd // n_head  # 128
    n_ff = 14336
    rope_theta = 10000.0
    rms_eps = 1e-5
    position = 0
    token_id = 1  # BOS

    # ---- Load reference weights ----
    print("Loading reference weights (safetensors)...")
    ref = load_safetensors()
    print(f"  Loaded {len(ref)} tensors")

    # ---- Load compressed weights ----
    print("Loading compressed weights (turbo)...")
    comp = {}
    for name in ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]:
        comp[name] = load_compressed_weight(f"layer.0.{name}")
        print(f"  {name}: {comp[name]['M']}x{comp[name]['K']}, patches={comp[name]['num_patches']}")

    # Load norm weights from turbo dir
    attn_norm_turbo = torch.from_numpy(
        np.fromfile(os.path.join(TURBO_DIR, "layer.0.attn_norm.bin"), dtype=np.float32)
    ).to(DEVICE)
    ffn_norm_turbo = torch.from_numpy(
        np.fromfile(os.path.join(TURBO_DIR, "layer.0.ffn_norm.bin"), dtype=np.float32)
    ).to(DEVICE)

    # Load embedding table from turbo dir
    tok_embd_raw = np.fromfile(os.path.join(TURBO_DIR, "tok_embd.bin"), dtype=np.int16)
    tok_embd_turbo = torch.from_numpy(tok_embd_raw).view(32000, n_embd)

    print()
    print("=" * 90)
    print("Step-by-step comparison (layer 0, BOS token, position 0)")
    print("=" * 90)

    # ================================================================
    # (a) Embedding lookup
    # ================================================================
    print("\n--- (a) Embedding lookup (token_id=1, BOS) ---")

    ref_embd = ref["model.embed_tokens.weight"][token_id].to(torch.bfloat16).float().to(DEVICE)

    # Turbo: same as C++ embed_lookup_kernel - bf16_to_float on stored int16
    turbo_embd_i16 = tok_embd_turbo[token_id].to(DEVICE)
    # Convert BF16 stored as int16 to float32 (same as bf16_to_float in kernels.hip)
    turbo_embd = turbo_embd_i16.view(torch.bfloat16).float()

    show("ref_embd", ref_embd)
    show("turbo_embd", turbo_embd)
    compare("embedding", ref_embd, turbo_embd)

    # Use ref embedding as starting hidden state (both should be identical)
    hidden_ref = ref_embd.clone()
    hidden_ours = turbo_embd.clone()

    # ================================================================
    # (b) RMSNorm (input_layernorm)
    # ================================================================
    print("\n--- (b) RMSNorm (input_layernorm) ---")

    ref_norm_w = ref["model.layers.0.input_layernorm.weight"].float().to(DEVICE)

    normed_ref = rms_norm(hidden_ref, ref_norm_w, rms_eps)
    normed_ours = rms_norm(hidden_ours, attn_norm_turbo, rms_eps)

    show("normed_ref", normed_ref)
    show("normed_ours", normed_ours)
    compare("RMSNorm", normed_ref, normed_ours)

    # ================================================================
    # (c) Q projection
    # ================================================================
    print("\n--- (c) Q projection ---")

    wq_ref = ref["model.layers.0.self_attn.q_proj.weight"].to(DEVICE)
    q_ref = (wq_ref.float() @ normed_ref.float())
    q_ours = compressed_matvec(comp["wq"], normed_ours)

    show("q_ref", q_ref)
    show("q_ours", q_ours)
    compare("Q projection", q_ref, q_ours)

    # ================================================================
    # (d) K projection
    # ================================================================
    print("\n--- (d) K projection ---")

    wk_ref = ref["model.layers.0.self_attn.k_proj.weight"].to(DEVICE)
    k_ref = (wk_ref.float() @ normed_ref.float())
    k_ours = compressed_matvec(comp["wk"], normed_ours)

    show("k_ref", k_ref)
    show("k_ours", k_ours)
    compare("K projection", k_ref, k_ours)

    # ================================================================
    # (e) V projection
    # ================================================================
    print("\n--- (e) V projection ---")

    wv_ref = ref["model.layers.0.self_attn.v_proj.weight"].to(DEVICE)
    v_ref = (wv_ref.float() @ normed_ref.float())
    v_ours = compressed_matvec(comp["wv"], normed_ours)

    show("v_ref", v_ref)
    show("v_ours", v_ours)
    compare("V projection", v_ref, v_ours)

    # ================================================================
    # (f) RoPE at position 0
    # ================================================================
    print("\n--- (f) RoPE (position=0) ---")

    q_ref_rope = apply_rope(q_ref.cpu(), head_dim, n_head, position, rope_theta).to(DEVICE)
    k_ref_rope = apply_rope(k_ref.cpu(), head_dim, n_head_kv, position, rope_theta).to(DEVICE)
    q_ours_rope = apply_rope(q_ours.cpu(), head_dim, n_head, position, rope_theta).to(DEVICE)
    k_ours_rope = apply_rope(k_ours.cpu(), head_dim, n_head_kv, position, rope_theta).to(DEVICE)

    show("q_ref_rope", q_ref_rope)
    show("q_ours_rope", q_ours_rope)
    compare("Q after RoPE", q_ref_rope, q_ours_rope)

    show("k_ref_rope", k_ref_rope)
    show("k_ours_rope", k_ours_rope)
    compare("K after RoPE", k_ref_rope, k_ours_rope)

    # Note: at position=0, RoPE with cos(0)=1, sin(0)=0 is identity,
    # so values shouldn't change for d=0
    print("  (Note: RoPE at pos=0 is identity for freq=0 components)")

    # ================================================================
    # (g) Attention (seq_len=1, trivial softmax)
    # ================================================================
    print("\n--- (g) Attention (pos=0, seq_len=1) ---")
    print("  With seq_len=1, attention is trivial: softmax([score]) = [1.0]")
    print("  So attn_output = V directly (each head's V slice).")

    gqa_ratio = n_head // n_head_kv
    scale = 1.0 / math.sqrt(head_dim)

    # Reference attention
    attn_ref = torch.zeros(n_embd, dtype=torch.float32, device=DEVICE)
    for h in range(n_head):
        kv_h = h // gqa_ratio
        q_h = q_ref_rope[h * head_dim:(h + 1) * head_dim]
        k_h = k_ref_rope[kv_h * head_dim:(kv_h + 1) * head_dim]
        v_h = v_ref[kv_h * head_dim:(kv_h + 1) * head_dim]
        score = (q_h @ k_h) * scale
        # softmax of single element = 1.0
        attn_ref[h * head_dim:(h + 1) * head_dim] = v_h

    # Ours attention
    attn_ours = torch.zeros(n_embd, dtype=torch.float32, device=DEVICE)
    for h in range(n_head):
        kv_h = h // gqa_ratio
        q_h = q_ours_rope[h * head_dim:(h + 1) * head_dim]
        k_h = k_ours_rope[kv_h * head_dim:(kv_h + 1) * head_dim]
        v_h = v_ours[kv_h * head_dim:(kv_h + 1) * head_dim]
        score = (q_h @ k_h) * scale
        attn_ours[h * head_dim:(h + 1) * head_dim] = v_h

    show("attn_ref", attn_ref)
    show("attn_ours", attn_ours)
    compare("Attention output", attn_ref, attn_ours)

    # ================================================================
    # (h) Output projection (wo)
    # ================================================================
    print("\n--- (h) Output projection (wo) ---")

    wo_ref = ref["model.layers.0.self_attn.o_proj.weight"].to(DEVICE)
    wo_out_ref = (wo_ref.float() @ attn_ref.float())
    wo_out_ours = compressed_matvec(comp["wo"], attn_ours)

    show("wo_out_ref", wo_out_ref)
    show("wo_out_ours", wo_out_ours)
    compare("Output projection", wo_out_ref, wo_out_ours)

    # ================================================================
    # (i) Residual add
    # ================================================================
    print("\n--- (i) Residual add (post-attention) ---")

    resid_ref = wo_out_ref + hidden_ref
    resid_ours = wo_out_ours + hidden_ours

    show("resid_ref", resid_ref)
    show("resid_ours", resid_ours)
    compare("Post-attn residual", resid_ref, resid_ours)

    # ================================================================
    # (j) FFN norm (post_attention_layernorm)
    # ================================================================
    print("\n--- (j) FFN norm (post_attention_layernorm) ---")

    ffn_norm_ref_w = ref["model.layers.0.post_attention_layernorm.weight"].float().to(DEVICE)
    ffn_normed_ref = rms_norm(resid_ref, ffn_norm_ref_w, rms_eps)
    ffn_normed_ours = rms_norm(resid_ours, ffn_norm_turbo, rms_eps)

    show("ffn_normed_ref", ffn_normed_ref)
    show("ffn_normed_ours", ffn_normed_ours)
    compare("FFN norm", ffn_normed_ref, ffn_normed_ours)

    # ================================================================
    # (k) MLP: gate, up, silu*gate, down
    # ================================================================
    print("\n--- (k) MLP ---")

    # Gate projection
    wgate_ref = ref["model.layers.0.mlp.gate_proj.weight"].to(DEVICE)
    gate_ref = (wgate_ref.float() @ ffn_normed_ref.float())
    gate_ours = compressed_matvec(comp["w_gate"], ffn_normed_ours)

    show("gate_ref", gate_ref)
    show("gate_ours", gate_ours)
    compare("MLP gate_proj", gate_ref, gate_ours)

    # Up projection
    wup_ref = ref["model.layers.0.mlp.up_proj.weight"].to(DEVICE)
    up_ref = (wup_ref.float() @ ffn_normed_ref.float())
    up_ours = compressed_matvec(comp["w_up"], ffn_normed_ours)

    show("up_ref", up_ref)
    show("up_ours", up_ours)
    compare("MLP up_proj", up_ref, up_ours)

    # SiLU(gate) * up
    swiglu_ref = silu(gate_ref) * up_ref
    swiglu_ours = silu(gate_ours) * up_ours

    show("swiglu_ref", swiglu_ref)
    show("swiglu_ours", swiglu_ours)
    compare("SwiGLU", swiglu_ref, swiglu_ours)

    # Down projection
    wdown_ref = ref["model.layers.0.mlp.down_proj.weight"].to(DEVICE)
    down_ref = (wdown_ref.float() @ swiglu_ref.float())
    down_ours = compressed_matvec(comp["w_down"], swiglu_ours)

    show("down_ref", down_ref)
    show("down_ours", down_ours)
    compare("MLP down_proj", down_ref, down_ours)

    # ================================================================
    # (l) Final residual add
    # ================================================================
    print("\n--- (l) Final residual add (post-MLP) ---")

    final_ref = down_ref + resid_ref
    final_ours = down_ours + resid_ours

    show("final_ref (layer 0 output)", final_ref)
    show("final_ours (layer 0 output)", final_ours)
    compare("Layer 0 output", final_ref, final_ours)

    # ================================================================
    # Summary: also check the C++ engine's actual behavior
    # ================================================================
    print("\n" + "=" * 90)
    print("ADDITIONAL CHECKS")
    print("=" * 90)

    # Check: does the compressed kernel output match ref BF16 matvec?
    # The kernel takes BF16 activations, so the ref should also use BF16.
    print("\n--- BF16-precision reference vs kernel (Q proj) ---")
    normed_bf16 = normed_ref.to(torch.bfloat16)
    q_ref_bf16 = (wq_ref.float() @ normed_bf16.float())
    compare("Q (BF16 input ref vs kernel)", q_ref_bf16, q_ours, rtol=5e-3, atol=0.5)

    # Check: does fp32->bf16 conversion lose significant precision?
    print("\n--- FP32 vs BF16 activation precision ---")
    normed_f32_vals = normed_ref[:5].tolist()
    normed_bf16_vals = normed_bf16.float()[:5].tolist()
    print(f"  FP32: {[f'{v:.8f}' for v in normed_f32_vals]}")
    print(f"  BF16: {[f'{v:.8f}' for v in normed_bf16_vals]}")

    # Check: what does the kernel output look like raw?
    print("\n--- Kernel output statistics ---")
    print(f"  q_ours: min={q_ours.min().item():.6f} max={q_ours.max().item():.6f} "
          f"mean={q_ours.mean().item():.6f} std={q_ours.std().item():.6f}")
    print(f"  q_ref:  min={q_ref.min().item():.6f} max={q_ref.max().item():.6f} "
          f"mean={q_ref.mean().item():.6f} std={q_ref.std().item():.6f}")

    # Check if output is near zero (the reported bug)
    if q_ours.abs().max().item() < 0.01:
        print("\n  *** BUG CONFIRMED: kernel output is near-zero! ***")
        print("  The compressed matvec kernel is producing near-zero results.")
        print("  Possible causes:")
        print("    1. Codebook not loaded into shared memory correctly")
        print("    2. Packed data pointer offset wrong")
        print("    3. Escape offsets corrupted")
        print("    4. Activation BF16 conversion wrong")
    elif q_ours.abs().max().item() > 0.1:
        print("\n  Kernel output has reasonable magnitude -- matvec itself is working.")
        if not compare("Q proj (final check)", q_ref, q_ours, rtol=0.05, atol=1.0):
            print("  Differences may be due to BF16 precision loss in the kernel path.")

    print("\n" + "=" * 90)
    print("Debug complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
