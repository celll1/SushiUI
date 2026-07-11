"""Training-free reference-style transfer (StyleAligned/VSP-style KV injection) for
MiniT2I's ``DoubleStreamDiTBlock``, wired through the arch-agnostic
``core.inference.reference_style`` module (see that module's docstring for the shared
math: ``inject_kv``, ``cross_batch_adain_qk``, ``make_ref_value``,
``frequency_scale_vector``).

Design (mirrors ``core.inference.nag_minit2i``, the closest architectural precedent):
MiniT2I's dual-stream attention is INLINED in ``DoubleStreamDiTBlock.forward`` (unlike
Krea2/FLUX.2, which route through a separate ``Attention``/processor object this repo
already stamps a ``_style_ctx`` onto), so there is no per-block attention module to hook
into non-invasively. Exactly like NAG's ``MiniT2INAGWrapper``, this module monkey-patches
``block.forward`` (via ``types.MethodType``) with a byte-identical duplicate of the
vendored forward, plus one extra call to ``_apply_style_hook`` right after RoPE and right
before ``mem_efficient_sdpa`` -- the same insertion point Krea2Attention/FLUX.2's style
processors use (after qk-RMSNorm + RoPE, before the attention kernel). The patch is
installed ONLY for the duration of a style-active generation and restored afterwards
(``install_minit2i_style_blocks`` / ``restore_minit2i_style_blocks``), so a generation
without a style reference never executes this module's forward variant at all.

Image-token region: MiniT2I's joint sequence is ``cat([txt, img])`` -- the IMAGE tokens
are the SUFFIX (``[lt : lt+li]``), the opposite convention from FLUX.2/Z-Image's
``[img, txt]`` prefix layout but the SAME convention Krea2 uses for its own joint
sequence in the text-fusion blocks. ``lt`` = text sequence length (``txt.shape[1]``,
FLAN-T5's fixed ``prompt_length``), ``li`` = image token count (``grid_h*grid_w``).

RoPE note: MiniT2I uses INTERLEAVED (rotate_half) RoPE, NOT the
``repeat_interleave_real=True`` real-cos/sin-pair layout ``frequency_scale_vector``
assumes for Krea2/Z-Image (whose axis-split channel mapping only holds for that specific
pairing). Re-deriving the equivalent per-channel frequency curve for ``rotate_half``
(where the two halves of each pair are NOT adjacent head-dim slots, they're
``head_dim//2`` apart) is out of scope for v1 -- so this wiring passes an all-ones vector
straight to ``inject_kv`` instead of calling ``StyleTransferConfig.get_freq_scale_vector``
(which requires ``axes_dims`` to be set; it never is here). Frequency suppression is a
quality knob only -- ``ref_k_strength`` and AdaIN (the load-bearing mechanisms) apply in
full regardless.

head_dim: b16 = 64, l16 = 52 (mem_efficient_sdpa pads to 56 internally for the SDPA-fast
path, AFTER this hook runs) -- capture/inject/ones-vector all use ``block.head_dim``
(the TRUE, unpadded dim), so shapes always match what the hook sees pre-padding.

Interop: mutually exclusive with NAG and NegPip (both ALSO monkey-patch
``block.forward``; combining would have one wrapper silently clobber the other's patch)
-- the pipeline_backends/minit2i.py caller gates NAG/NegPip installation off whenever
style transfer is active for a generation. FBCache is disabled for the WHOLE generation
whenever style transfer is requested (mirrors Z-Image/FLUX.2: a cache hit skips
``double_blocks[1:]``, which would desync the per-block style capture/inject store across
steps) -- enforced by ``minit2i_pipeline_ops._build_minit2i_fbcache``'s caller. Block Swap
composes unchanged (it only changes WHERE compute happens, not what attention sees).
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional, Tuple

import torch

from core.attention import AttentionMode


def _apply_style_hook(
    block, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, img_start: int, img_end: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared capture/inject body, mirrors ``style_flux2.py``'s ``_apply_style_hook``.

    ``q``/``k``/``v`` are already post-qk-RMSNorm, post-RoPE, and the full
    ``[txt, img]`` concatenated sequence (BSHD: ``[B, S, H, D]``). No-op when
    ``_style_ctx`` is ``None`` (default / non-style generation), the block was never
    assigned a ``_style_block_idx``, or the context's ``block_range`` excludes this
    block.
    """
    ctx = getattr(block, "_style_ctx", None)
    block_idx = getattr(block, "_style_block_idx", None)
    if ctx is None or block_idx is None or not ctx.active_for_block(block_idx):
        return q, k, v

    if ctx.mode == "capture":
        ctx.store[block_idx] = (
            q[:, img_start:img_end].detach().clone(),
            k[:, img_start:img_end].detach().clone(),
            v[:, img_start:img_end].detach().clone(),
        )
        return q, k, v

    # mode == "inject"
    ref_qkv = ctx.store.get(block_idx)
    if ref_qkv is None:
        return q, k, v

    from core.inference.reference_style import inject_kv, make_ref_value

    ref_q, ref_k, ref_v = ref_qkv
    cfg = ctx.config
    if cfg.ref_k_strength == 0.0 and cfg.adain_strength <= 0.0:
        return q, k, v

    # Interleaved (rotate_half) RoPE: the Krea2/Z-Image per-axis frequency-suppression
    # channel mapping does not cleanly apply (see module docstring). Frequency
    # suppression is a quality knob only, off (ones = no scaling) for v1; ref_k_strength
    # + AdaIN (the load-bearing mechanisms) still apply in full.
    freq_vec = torch.ones(block.head_dim, device=k.device, dtype=k.dtype)

    target_v_img = v[:, img_start:img_end]
    ref_v_final = make_ref_value(
        target_v_img, ref_v, cfg.value_mode, cfg.value_adain_strength, cfg.ref_value_mix
    )
    k, v, q = inject_kv(
        k, v, ref_k, ref_v_final, img_start, img_end,
        cfg.ref_k_strength, freq_vec, cfg.adain_strength, q=q, ref_q=ref_q,
    )
    return q, k, v


def _style_double_block_forward(block, x, txt, vec, grid_h: int, grid_w: int):
    """``DoubleStreamDiTBlock.forward`` with a style capture/inject hook on the joint
    Q/K/V, inserted right after RoPE (same position Krea2Attention/FLUX.2's style
    processors use) and right before ``mem_efficient_sdpa``. Byte-identical to the
    vendored forward whenever ``block._style_ctx`` is ``None`` (``_apply_style_hook``'s
    fast no-op path)."""
    b, li, _ = x.shape
    lt = txt.shape[1]
    x_norm = block.img_norm1(x)
    txt_norm = block.txt_norm1(txt)
    qkv_i = block.img_qkv(x_norm).reshape(b, li, 3, block.num_heads, block.head_dim)
    qkv_t = block.txt_qkv(txt_norm).reshape(b, lt, 3, block.num_heads, block.head_dim)
    q_i, k_i, v_i = qkv_i[:, :, 0], qkv_i[:, :, 1], qkv_i[:, :, 2]
    q_t, k_t, v_t = qkv_t[:, :, 0], qkv_t[:, :, 1], qkv_t[:, :, 2]
    q_i, k_i = block.q_norm(q_i), block.k_norm(k_i)
    q_t, k_t = block.q_norm(q_t), block.k_norm(k_t)
    q = block.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    k = block.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    v = torch.cat([v_t, v_i], dim=1)

    # Image tokens are the SUFFIX of the joint [txt, img] sequence: [lt : lt+li].
    q, k, v = _apply_style_hook(block, q, k, v, img_start=lt, img_end=lt + li)

    from core.models.minit2i.vendor.mmjit import mem_efficient_sdpa
    out = mem_efficient_sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        backend=getattr(block, "_attn_backend", "native"),
        mode=getattr(block, "_attn_mode", AttentionMode.INFERENCE),
    ).transpose(1, 2).contiguous()  # [b, seq, heads, hd]

    x = x + block.img_attn_proj(out[:, lt:].reshape(b, li, -1))
    txt = txt + block.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
    x = x + block.img_mlp(block.img_norm2(x))
    txt = txt + block.txt_mlp(block.txt_norm2(txt))
    return x, txt


def install_minit2i_style_blocks(net) -> Dict[int, Any]:
    """Monkey-patch every ``net.double_blocks[i].forward`` with the style-aware
    variant, stamping a per-block unified index (``block_range`` gates against this
    same index space -- double_blocks only, ``txt_preamble_blocks`` are untouched,
    matching Z-Image/FLUX.2's "main DiT self-attention only" scope). Returns a
    ``{id(block): original_forward}`` map for ``restore_minit2i_style_blocks``."""
    saved: Dict[int, Any] = {}
    for idx, block in enumerate(net.double_blocks):
        saved[id(block)] = block.forward
        block._style_block_idx = idx
        block._style_ctx = None
        block.forward = types.MethodType(_style_double_block_forward, block)
    return saved


def restore_minit2i_style_blocks(net, saved: Dict[int, Any]) -> None:
    """Undo ``install_minit2i_style_blocks``: restores the original bound forward and
    strips the style attributes so a subsequent non-style generation is unaffected."""
    for block in net.double_blocks:
        fwd = saved.get(id(block))
        if fwd is not None:
            block.forward = fwd
        for attr in ("_style_block_idx", "_style_ctx"):
            if hasattr(block, attr):
                delattr(block, attr)


def set_minit2i_style_context(net, ctx: Optional[Any]) -> None:
    """Stamp the SAME ``StyleContext`` (or ``None`` to disarm) onto every patched
    double block. Called once per capture/inject/disarm phase per active step."""
    if ctx is not None:
        ctx.config.resolve_default_block_range(len(net.double_blocks))
    for block in net.double_blocks:
        block._style_ctx = ctx
