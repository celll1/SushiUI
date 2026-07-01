"""NegPip (negative-emphasis prompting) for the Lens transformer (DiT).

NegPip lets a token written with a NEGATIVE emphasis weight (e.g.
``(worst quality:-1)``) have its attention VALUE (V) negated, so the concept is
SUBTRACTED rather than added; positive weights scale V up. It is a single
elementwise, per-token signed scale of the TEXT-token V inside attention — Q and K
are left untouched, so the attention pattern is the normal one and no extra forward
pass is added (iter speed unchanged, unlike NAG which computes a second attention).

It AUTO-ACTIVATES (no toggle) when the prompt contains any negative weight, and is
applied per-context: in CFG the positive-context text V uses the positive prompt's
signed weights and the negative-context text V uses the negative prompt's signed
weights. A negative weight in the negative prompt is a double-negative that
re-affirms the concept.

Lens specifics (differs from the SDXL/CLIP reference):
  * Text encoder is GPT-OSS (not CLIP), so CLIP 77-token chunking does NOT apply.
    Weights are built via ``_build_emphasis_lens`` (progressive tokenization on the
    GPT-OSS tokenizer), the SAME path ``encode_prompt`` already uses.
  * The Lens text features are trimmed to start at ``DEFAULT_TXT_OFFSET`` (the chat
    template preamble is dropped), so position 0 of the feature/weight vector is the
    first user-prompt token. There are therefore NO BOS/EOS tokens inside the
    sequence; any position past the parsed content tokens (padding) gets weight 1.0.
  * Lens uses a single ``LensJointAttention`` (joint image+text attention), not a
    separate cross-attention. The text V (``txt_v``) is the one scaled; the image V
    is left untouched.

The hook is installed only when the prompt has a negative weight, so the
positive-only default path is byte-identical to before.

Reuses (read-only): core.prompts.prompt_parser (parse via _build_emphasis_lens).
"""

from __future__ import annotations

from typing import List, Optional

import torch


def build_lens_signed_weights(
    prompt: str,
    tokenizer,
    seq_txt: int,
    device,
    dtype,
    max_length: int = 512,
) -> torch.Tensor:
    """Build the per-token signed weight vector aligned to the Lens text sequence.

    Returns a 1-D tensor of length ``seq_txt`` with 1.0 on every position by default
    and the parsed (possibly negative) emphasis weight on each content token. The
    mapping uses ``_build_emphasis_lens`` (progressive tokenization on the GPT-OSS
    tokenizer), so it lines up exactly with the offset-trimmed text features (position
    0 = first user-prompt token). Padding positions past the content tokens stay 1.0.

    An empty / positive-only prompt yields an all-ones vector (identity).
    """
    # Reuse the exact same parse+tokenize path encode_prompt uses to place weights.
    from core.models.lens.lens_pipeline_ops import _build_emphasis_lens

    weights = torch.ones(seq_txt, dtype=dtype, device=device)
    if not prompt:
        return weights

    _clean, token_weights = _build_emphasis_lens(prompt, tokenizer, max_length)
    if not token_weights:
        return weights

    n = min(len(token_weights), seq_txt)
    if n <= 0:
        return weights
    weights[:n] = torch.tensor(token_weights[:n], dtype=dtype, device=device)
    return weights


def build_lens_signed_weight_batch(
    prompts: List[Optional[str]],
    tokenizer,
    seq_txt: int,
    device,
    dtype,
    max_length: int = 512,
) -> torch.Tensor:
    """Stack per-context signed weight vectors into a [batch, seq_txt] tensor.

    ``prompts`` is ordered to match the model's text batch (e.g. [positive, negative]
    for CFG, or [positive, negative, nag_neg] when NAG is also active). Each entry is
    the raw (emphasis-bearing) prompt string for that context; None/"" yields ones.
    """
    rows = [
        build_lens_signed_weights(p or "", tokenizer, seq_txt, device, dtype, max_length)
        for p in prompts
    ]
    return torch.stack(rows, dim=0)  # [batch, seq_txt]


def install_negpip(transformer, token_weights: torch.Tensor) -> List:
    """Enable the OFF-by-default NegPip branch on every Lens attention module.

    ``token_weights`` is a [batch, seq_txt] signed weight tensor whose batch order
    matches the text batch the transformer receives ([cond, uncond] for plain CFG, or
    [cond, uncond, nag_neg] when NAG is active). Each attention module scales its
    ``txt_v`` per-token by the row matching its group. Returns the list of touched
    modules so the caller can restore them.
    """
    touched = []
    # Unwrap the NAG wrapper if present (it forwards attribute access to .transformer,
    # but we want the underlying module list either way).
    base = getattr(transformer, "transformer", transformer)
    for module in base.modules():
        if module.__class__.__name__ == "LensJointAttention":
            module._negpip_enabled = True
            module._negpip_weights = token_weights
            touched.append(module)
    return touched


def restore_negpip(modules: List) -> None:
    """Disable the NegPip branch on every module (restore the default path)."""
    for module in modules:
        module._negpip_enabled = False
        module._negpip_weights = None


def scale_text_value(txt_v: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    """Signed per-token scale of the text V (called from LensJointAttention.forward).

    txt_v: [bsz, seq_txt, heads, dim_head] (pre-transpose, per the vendored attention).
    token_weights: [batch, seq_txt] signed weights whose batch order matches txt_v's.

    Handles batch / sequence-length mismatches defensively (pad with 1.0 = identity),
    so a partially-built weight tensor can never corrupt an unrelated group.
    """
    bsz, seq_txt = txt_v.shape[0], txt_v.shape[1]
    w = token_weights.to(device=txt_v.device, dtype=txt_v.dtype)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    # Align batch.
    if w.shape[0] != bsz:
        if w.shape[0] == 1:
            w = w.expand(bsz, -1)
        elif w.shape[0] > bsz:
            w = w[:bsz]
        else:
            pad = torch.ones(bsz - w.shape[0], w.shape[1], device=w.device, dtype=w.dtype)
            w = torch.cat([w, pad], dim=0)
    # Align sequence length (padding differs between contexts): pad with 1.0 / truncate.
    if w.shape[1] != seq_txt:
        if w.shape[1] < seq_txt:
            pad = torch.ones(w.shape[0], seq_txt - w.shape[1], device=w.device, dtype=w.dtype)
            w = torch.cat([w, pad], dim=1)
        else:
            w = w[:, :seq_txt]
    return txt_v * w[:, :, None, None]  # [b, s, h, d] *= [b, s, 1, 1]
