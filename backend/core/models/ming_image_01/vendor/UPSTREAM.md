# Ming-Image provenance

Pinned inclusionAI/Ming-Image commit `3515cd7b1e3049fc517eed7d35020251cd06519a`.
Source files retain their original notices. Local adaptations use package-relative imports, native Torch FX instead of a removed Transformers helper, and device-aware autocast while preserving native CUDA BF16/FP32 behavior. See [file ledger](../../../../../docs/legal/MING_IMAGE_01_PROVENANCE.md) and Git history for subsequent compatibility changes. No weights or CLI external-service integrations are included.

Additional adaptations preserve native default/partial RoPE, expose an explicit live-conditioning graph flag, map tied-weight declarations, and initialize tokenizer BOS state after the base constructor. Whitespace cleanup is checked for AST equivalence. The external loader regenerates nonpersistent rotary buffers after low-memory loading.

- Vision declares `supports_gradient_checkpointing=True` so Transformers can configure its existing block checkpoint path during full TE training.

- Vision activation-dtype discovery and language FlashAttention recasting use a packed Linear's `compute_dtype`, preserving the floating fallback for ordinary checkpoints.

- DiT accepts optional full-canvas target spatial coordinates; that path packs target tiles and full reference grids independently and reconstructs target frames only. The ordinary native path retains its original reference-output behavior.

- Main-stream gradient checkpointing accepts an explicit prefix block limit; refiners retain the native global switch.

- An optional global input residual is shape-checked and added only to target patch embeddings before refinement, leaving reference and padding embeddings intact.
