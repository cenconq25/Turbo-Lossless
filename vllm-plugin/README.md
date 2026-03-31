# Turbo Lossless vLLM Plugin

Drop-in replacement for BF16 GEMV in vLLM using structured 12-bit compressed weights.

## How it works

1. Model weights are pre-compressed to structured 12-bit format (offline)
2. This plugin replaces vLLM's `torch.matmul` / rocBLAS GEMV with our fused decode-matvec kernel
3. vLLM handles everything else: batching, KV cache, attention, scheduling
4. Output is bit-exact with BF16 — zero quality loss

## Status: Proof of Concept

This is a minimal integration to validate the approach. Not production-ready yet.
