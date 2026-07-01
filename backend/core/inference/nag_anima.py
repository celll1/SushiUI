"""NAG (Normalized Attention Guidance) for the Anima DiT.

Anima is a UViT-style DiT whose transformer ``Block``s each contain an image
self-attention followed by a *cross-attention* where the image (query) tokens
attend to the text context (``crossattn_emb``, produced by the LLM adapter).
NAG operates exactly on this cross-attention output: for the image queries we
compute the attention output against the POSITIVE text context and against a
separate NAG-negative text context, then extrapolate in attention-OUTPUT space
via ``core.inference.nag_dit.nag_guidance`` (L2, norm_p=2 — the official DiT
variant).

CFG handling
------------
Anima runs CFG as a SEPARATE forward pass with batch size 1 (see
``anima_pipeline_ops.sample_txt2img``: ``v_cond = transformer(...)`` and, if
``do_cfg``, a second ``v_uncond = transformer(...)``). NAG must apply to the
COND/positive image tokens only (SDXL / FLUX.2 parity). Because the two passes
are separate, we simply enable the NAG cross-attention branch on the COND pass
and leave the UNCOND pass untouched — the guidance therefore lands only on the
positive/conditional prediction, matching how SDXL composes CFG+NAG.

Mechanism
---------
``AnimaNAGWrapper`` wraps the transformer:
  * It pre-computes the NAG-negative cross-attention context using the *same*
    ``_preprocess_text_embeds`` path the model already uses for the positive /
    negative prompt (LLM adapter + t5 conditioning), so the negative context is
    threaded identically to the positive one.
  * It monkey-patches every ``Block.cross_attn.forward`` (the vendored
    ``Attention``) with a NAG-aware forward that, when armed, runs the cross
    attention twice (image-vs-pos-text, image-vs-nagneg-text) and blends with
    ``nag_guidance``. When NOT armed the patched forward is byte-identical to
    the original (it calls the exact same code path).
  * ``forward(...)`` arms the branch, pushes the negative context onto each
    cross-attn module, runs the wrapped transformer's real forward, then
    disarms — so only the cond pass is guided and nothing leaks.

This composes with Spectrum acceleration: NAG runs inside the real attention
forward, so Spectrum records the NAG-modified ``v`` on anchor steps and
forecasts it on skip steps unchanged.

Everything is OFF by default: the wrapper is only installed when
``nag_active(...)`` is true (nag_enable AND nag_scale>1 AND a negative context).
When NAG is off the generation path is byte-identical to before.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from core.inference.nag_dit import nag_guidance


def _apply_negpip_v(v, token_weights):
    """Signed per-token scale of the cross-attention VALUE (NegPip).

    ``v`` is ``[B, L_text, H, D]`` (bshd); ``token_weights`` is a 1-D vector of
    length ``L_text`` (aligned to the T5 token positions that build the cross
    context). Scales ``v[:, t, :, :] *= weights[t]`` so negative-weight tokens
    have their contribution SUBTRACTED. When ``token_weights`` is None this is a
    no-op — the default path is untouched.
    """
    if token_weights is None:
        return v
    w = token_weights.to(device=v.device, dtype=v.dtype)
    L = v.shape[1]
    if w.shape[0] != L:
        # Align to the actual text-token axis: pad with 1.0 (identity) or
        # truncate. Padding/truncation may differ from the built length.
        if w.shape[0] < L:
            pad = torch.ones(L - w.shape[0], device=w.device, dtype=w.dtype)
            w = torch.cat([w, pad], dim=0)
        else:
            w = w[:L]
    return v * w[None, :, None, None]


def _patched_cross_attn_forward(self, x, attn_params, context=None, rope_emb=None):
    """Drop-in replacement for ``Attention.forward`` on cross-attention modules.

    When ``self._nag_armed`` is False (the default) and ``self._negpip_weights``
    is None this is exactly the original forward. When NAG is armed it also
    attends the image queries to the stored NAG-negative context and blends the
    two outputs with nag_guidance. When NegPip weights are armed it applies a
    signed per-token scale to the text-context V (both the positive and the
    NAG-negative context), composing with NAG in this single patch.
    """
    # Import here to avoid a circular import at module load.
    from core.models.anima import anima_attention as _attn

    negpip_w = getattr(self, "_negpip_weights", None)

    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
    if q.dtype != v.dtype and torch.is_autocast_enabled():
        q = q.to(v.dtype)
        k = k.to(v.dtype)
    # NegPip: signed per-token V scale on the positive text context.
    v = _apply_negpip_v(v, negpip_w)
    z_pos = _attn.attention([q, k, v], attn_params=attn_params)

    if getattr(self, "_nag_armed", False):
        neg_context = getattr(self, "_nag_neg_context", None)
        if neg_context is not None:
            # k/v against the negative context; the query is unchanged. RoPE is
            # not applied on cross-attention (is_selfattn is False), so we only
            # need fresh k/v from the negative context.
            _, k_neg, v_neg = self.compute_qkv(x, neg_context, rope_emb=rope_emb)
            if q.dtype != v_neg.dtype and torch.is_autocast_enabled():
                k_neg = k_neg.to(v_neg.dtype)
            # NegPip also scales the NAG-negative context V. This uses the SAME
            # per-context weight vector currently armed (the NAG cond pass runs
            # with the positive prompt's signed weights).
            v_neg = _apply_negpip_v(v_neg, negpip_w)
            q_neg = q
            z_neg = _attn.attention([q_neg, k_neg, v_neg], attn_params=attn_params)
            z_pos = nag_guidance(
                z_pos, z_neg,
                scale=self._nag_scale, tau=self._nag_tau, alpha=self._nag_alpha,
                norm_p=2,
            )

    return self.output_dropout(self.output_proj(z_pos))


class AnimaNAGWrapper(nn.Module):
    """Wraps an Anima transformer to add an OFF-by-default NAG cross-attention
    branch. ``forward`` has the same signature as the underlying transformer.

    The wrapper patches every block's ``cross_attn.forward`` once at construction
    and restores the originals via :meth:`restore`.
    """

    def __init__(self, transformer, neg_embeds: Dict[str, torch.Tensor],
                 nag_scale: float = 5.0, nag_tau: float = 2.5, nag_alpha: float = 0.25):
        super().__init__()
        self.transformer = transformer
        self.neg_embeds = neg_embeds
        self.nag_scale = float(nag_scale)
        self.nag_tau = float(nag_tau)
        self.nag_alpha = float(nag_alpha)

        # Collect the cross-attention modules of every block.
        self._cross_attns = [block.cross_attn for block in transformer.blocks]
        self._originals = []
        import types
        for ca in self._cross_attns:
            self._originals.append(ca.forward)
            ca._nag_armed = False
            ca._nag_scale = self.nag_scale
            ca._nag_tau = self.nag_tau
            ca._nag_alpha = self.nag_alpha
            ca._nag_neg_context = None
            ca._negpip_weights = None
            ca._negpip_or_nag_patched = True
            ca.forward = types.MethodType(_patched_cross_attn_forward, ca)

    def restore(self):
        """Restore the original cross-attention forwards and clear NAG state."""
        for ca, orig in zip(self._cross_attns, self._originals):
            ca.forward = orig
            for attr in ("_nag_armed", "_nag_scale", "_nag_tau",
                         "_nag_alpha", "_nag_neg_context",
                         "_negpip_weights", "_negpip_or_nag_patched"):
                if hasattr(ca, attr):
                    delattr(ca, attr)

    def _compute_neg_context(self) -> torch.Tensor:
        """Run the model's own text-embed preprocessing (LLM adapter + t5
        conditioning) on the NAG-negative embeds — identical to the positive /
        negative prompt path — producing the per-block cross-attention context.
        """
        t = self.transformer
        return t._preprocess_text_embeds(
            self.neg_embeds["prompt_embeds"],
            self.neg_embeds["t5_input_ids"],
            target_attention_mask=self.neg_embeds["t5_attn_mask"],
            source_attention_mask=self.neg_embeds["source_mask"],
        )

    def forward(self, *args, **kwargs):
        """Arm NAG (cond pass), run the real forward, then disarm.

        The positive context flows in via the normal ``context`` / ``target_*``
        kwargs and is processed by the transformer's own forward; we only supply
        the parallel NAG-negative context.
        """
        neg_context = self._compute_neg_context()
        for ca in self._cross_attns:
            ca._nag_neg_context = neg_context
            ca._nag_armed = True
        try:
            out = self.transformer(*args, **kwargs)
        finally:
            for ca in self._cross_attns:
                ca._nag_armed = False
                ca._nag_neg_context = None
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
