"""
Custom Attention Processor for accelerated inference (SDXL / SD1.5 UNet).

A single :class:`UnifiedAttnProcessor` handles every backend by routing the core
attention region through the unified conduit (:func:`core.attention.dispatch_attention`).
The processor keeps only the diffusers norm / residual / reshape boilerplate; the
backend selection, capability guards (head_dim / mask / dtype / GQA), and native
fallback all live in the conduit.

Backends (selected by the ``attention_type`` string, normalized inside the conduit):
- "normal"/"none"/"sdpa"/None: PyTorch scaled_dot_product_attention (native)
- "flash":                     FlashAttention-2 (falls back to native on any failure)
- "sage":                      SageAttention INT8 (auto-downgrades to native when the
                               head_dim is unsupported -- e.g. SD1.5 40/80/160)
"""

import torch
from typing import Optional
from diffusers.models.attention_processor import Attention

from core.attention import AttentionMode, dispatch_attention


class UnifiedAttnProcessor:
    """
    Unified attention processor for the diffusers UNet (SDXL / SD1.5).

    Replaces the former hand-written SageAttnProcessor / FlashAttnProcessor and
    the default AttnProcessor2_0. The QKV projections are reshaped to
    ``[batch, heads, seq_len, head_dim]`` (BHSD) and handed to the conduit with
    ``layout="BHSD"``; the conduit adapts the layout and dispatches to the
    selected kernel, falling back to native SDPA when the kernel is unavailable
    or unsupported for the given shapes.

    Args:
        backend: Backend selector string ("normal", "sage", or "flash"). The
            string is normalized and capability-gated inside the conduit, so no
            per-processor availability probing is needed here.
        mode: Conduit dispatch mode. Defaults to ``AttentionMode.INFERENCE`` so
            inference callers are unchanged; SDXL/SD1.5 training installs this
            same processor with ``AttentionMode.TRAINING`` (autograd-safe path,
            training-only backend guards).
    """

    def __init__(self, backend: str = "normal", mode: AttentionMode = AttentionMode.INFERENCE):
        self.backend = backend
        self.mode = mode
        # --- Training-free reference-style transfer (StyleAligned/VSP-style KV
        # injection, see core.inference.reference_style) ---
        # `block_idx` is the self-attention layer's ordinal position (assigned by
        # `ensure_style_block_indices`); stays `None` for cross-attention ("attn2")
        # processors, which are never touched by style transfer (SD1.5/SDXL U-Net
        # self-attention has no text tokens, so injection only targets self-attn).
        # `_style_ctx` is a `StyleContext` or `None`; both default to `None` so a
        # processor that is never stamped takes the byte-identical original path.
        self.block_idx: Optional[int] = None
        self._style_ctx = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        # Self-attention: diffusers UNet attn1 is always called with
        # encoder_hidden_states=None (cross-attention/attn2 always passes the text
        # embeddings). Must be captured BEFORE the "encoder_hidden_states = hidden_states"
        # fallback below overwrites it.
        is_self_attn = encoder_hidden_states is None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # BSHD == [batch, seq_len, heads, head_dim] (pre-transpose) -- this is the
        # layout `core.inference.reference_style` expects for the KV-injection hook.
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # --- Reference-style KV injection (training-free, self-attention only) ---
        # The whole self-attention sequence IS the image-token sequence (no text
        # prefix, unlike DiT archs), so img_start=0 / img_end=sequence_length always
        # -- computed locally per call since every U-Net resolution (down/mid/up
        # block) has a DIFFERENT sequence_length, unlike a DiT's fixed token count.
        # No RoPE in this U-Net: the frequency-scale vector is a constant `ones`
        # (no per-frequency content suppression), relying on block selection +
        # AdaIN + ref_k_strength for content/style control instead (StyleAligned's
        # original recipe).
        ctx = self._style_ctx
        if is_self_attn and self.block_idx is not None and ctx is not None and ctx.active_for_block(self.block_idx):
            from core.inference.reference_style import inject_kv, make_ref_value

            img_start, img_end = 0, query.shape[1]
            if ctx.mode == "capture":
                ctx.store[self.block_idx] = (
                    query[:, img_start:img_end].detach().clone(),
                    key[:, img_start:img_end].detach().clone(),
                    value[:, img_start:img_end].detach().clone(),
                )
            elif ctx.mode == "inject":
                ref_qkv = ctx.store.get(self.block_idx)
                if ref_qkv is not None:
                    ref_q, ref_k, ref_v = ref_qkv
                    cfg = ctx.config
                    if cfg.ref_k_strength != 0.0 or cfg.adain_strength > 0.0:
                        freq_vec = torch.ones(head_dim, device=key.device, dtype=key.dtype)
                        target_v_img = value[:, img_start:img_end]
                        ref_v_final = make_ref_value(
                            target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
                        )
                        key, value, query = inject_kv(
                            key, value, ref_k, ref_v_final, img_start, img_end,
                            cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=query, ref_q=ref_q,
                        )
                        # attention_mask is not extended: SD1.5/SDXL U-Net self-attention
                        # is called with attention_mask=None (no padding to account for).

        # Reshape to BHSD == [batch, heads, seq_len, head_dim].
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Core attention region -> unified conduit (BHSD in/out).
        hidden_states = dispatch_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self.backend,
            mode=self.mode,
            layout="BHSD",
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(
    unet,
    attention_type: str = "normal",
    mode: AttentionMode = AttentionMode.INFERENCE,
):
    """
    Set the unified attention processor on every attention layer of the UNet.

    Args:
        unet: The UNet model
        attention_type: Backend selector ("normal", "sage", "flash"). Normalized
            inside the conduit.
        mode: Conduit dispatch mode forwarded to each processor. Defaults to
            ``AttentionMode.INFERENCE`` so inference callers are unchanged; SDXL/
            SD1.5 training passes ``AttentionMode.TRAINING``.

    Returns:
        dict: Original processors for restoration
    """
    # Store original processors
    original_processors = unet.attn_processors.copy()

    num_processors = len(unet.attn_processors)

    print(f"[AttentionProcessor] Setting UnifiedAttnProcessor (backend={attention_type}, mode={mode.name}) for {num_processors} attention layers")
    new_processors = {name: UnifiedAttnProcessor(attention_type, mode=mode) for name in unet.attn_processors.keys()}
    unet.set_attn_processor(new_processors)
    print(f"[AttentionProcessor] [OK] UnifiedAttnProcessor ACTIVE for all {num_processors} layers")

    return original_processors


def restore_processors(unet, original_processors: dict):
    """Restore original attention processors"""
    if original_processors:
        unet.set_attn_processor(original_processors)
        print("[AttentionProcessor] Restored original processors")


def ensure_style_block_indices(unet) -> int:
    """Assign a stable self-attention block index (0..N-1) to every self-attn
    ("attn1") processor, in ``unet.attn_processors`` traversal order (whatever
    order this diffusers version's UNet2DConditionModel registers its blocks in
    -- e.g. down_blocks / up_blocks / mid_block, each in module-registration
    order). This is the layer-index space that a style-transfer request's
    ``style_blocks`` (``StyleTransferConfig.block_range``) selects into.

    Idempotent: safe to call every generation (re-numbers identically as long
    as the processor set/order hasn't changed -- e.g. after
    ``set_attention_processor`` swaps in new instances). Cross-attention
    ("attn2") processors are left with ``block_idx=None`` -- style transfer
    only targets self-attention, so they are silently never touched.

    Returns the total self-attention layer count (the valid ``[0, N-1]`` range
    for ``style_blocks``); ``None``/unset ``block_range`` (the default) applies
    style transfer to ALL of them, matching StyleAligned's original full-U-Net
    shared-self-attention setting.
    """
    idx = 0
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor") and hasattr(proc, "block_idx"):
            proc.block_idx = idx
            idx += 1
    return idx


def set_style_context(unet, ctx) -> None:
    """Stamp a ``core.inference.reference_style.StyleContext`` (or ``None`` to
    clear) onto every self-attention processor that has already been assigned a
    ``block_idx`` by ``ensure_style_block_indices``. No-op (nothing to stamp)
    if that hasn't been called yet."""
    for name, proc in unet.attn_processors.items():
        if hasattr(proc, "block_idx") and proc.block_idx is not None:
            proc._style_ctx = ctx
