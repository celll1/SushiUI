"""NegPip (negative-emphasis / signed-value attention) for the Z-Image transformer.

NegPip lets a token written with a NEGATIVE emphasis weight (e.g. ``(worst quality:-1)``)
have its attention VALUE (V) NEGATED, so the concept is SUBTRACTED from the output rather
than added; a positive weight scales V up. Q and K are left untouched -- only the sign and
magnitude of the value change -- so it is a single elementwise multiply with NO extra forward
pass (iter speed unchanged, unlike NAG which computes a second attention).

It AUTO-ACTIVATES (no UI toggle) when the prompt contains any negative weight. When no
negative weight is present, ``ZImageAttention._negpip_ctx`` is ``None`` and the attention
forward is byte-identical to before.

Z-Image specifics
-----------------
``ZImageTransformer2DModel`` runs self-attention (``ZImageAttention``) over a per-item unified
sequence ``[image_tokens (PREFIX) ; caption_tokens (SUFFIX)]``. There is no processor
abstraction (unlike SDXL's ``NegPipAttnProcessor2_0``), so NegPip scales V by an OFF-by-default
branch inside ``ZImageAttention.forward`` gated by a module-level context, exactly mirroring the
NAG integration in ``nag_zimage.py``.

The caption tokens are produced by the Qwen text encoder using a CHAT TEMPLATE (NOT CLIP), so
CLIP's 77-token chunking (``build_signed_weight_vector_chunked``) does not apply. The per-item
signed weight vector is built by :func:`build_zimage_caption_weights`, which tokenizes the
chat-formatted prompt the same way ``_zimage_encode_single`` does and lays each parsed emphasis
fragment's weight onto the tokens that fragment produces. Chat-template / role / BOS / EOS /
padding tokens keep weight 1.0.

CFG per-context handling
------------------------
The denoising loop passes captions in the batch order it uses:
  - CFG on, NAG on:  [cfg_neg ; cfg_pos ; nag_neg]
  - CFG on, NAG off: [cfg_neg ; cfg_pos]
  - CFG off, NAG on: [pos ; nag_neg]
  - CFG off, NAG off: [pos]
Each context gets its OWN signed weight vector: a negative weight in the positive prompt
subtracts the concept; a negative weight in the negative (uncond) prompt is a double-negative
that re-affirms it. Because it is applied per-item to the caption V, it composes with NAG (which
edits the image-prefix attention OUTPUT) and with Spectrum (which forecasts post-CFG velocity):
NegPip only touches the caption V of real transformer-evaluated steps.
"""

from typing import List, Optional

import torch


class NegPipContext:
    """Per-forward NegPip configuration shared with every ``ZImageAttention`` via a class attr.

    Args:
        weight_rows: list (length == unified batch, i.e. the number of caption items in the
            batch order the transformer received) of 1-D signed weight tensors. Each row's
            length equals the number of REAL caption tokens for that item (the encoder's
            masked/non-padded token count). Rows may be shorter than the item's post-patchify
            caption length (inner padding); the extra caption positions and all image-prefix
            positions keep weight 1.0.
        image_len: number of IMAGE (prefix) tokens per unified row (same for every row because
            all groups denoise identical latents). Caption tokens occupy ``[image_len:]``.
    """

    def __init__(self, weight_rows: List[torch.Tensor], image_len: int):
        self.weight_rows = weight_rows
        self.image_len = image_len


def build_zimage_caption_weights(prompt: str, tokenizer, formatted_token_len: int,
                                 device, dtype) -> torch.Tensor:
    """Signed per-token weight vector aligned to ONE item's Qwen caption token sequence.

    ``formatted_token_len`` is the number of real (non-padded) tokens the encoder produced for
    this prompt (i.e. ``masks[i].sum()`` in ``_zimage_encode_single``). The returned 1-D tensor
    has that length: 1.0 on every chat-template / role / special token and the parsed (possibly
    negative) emphasis weight on the tokens that belong to the prompt content.

    Alignment strategy (correct-by-construction, no reliance on special-token ids):
      1. Re-create the exact chat-formatted string the encoder tokenizes.
      2. Progressively tokenize the formatted string as its PREFIX grows with each parsed
         fragment inserted in order, so token positions of each fragment are discovered the same
         way ``build_signed_weight_vector`` does -- but on the FORMATTED text, so template tokens
         are naturally skipped (they live in the fixed prefix/suffix that no fragment touches).
    """
    from core.prompts.prompt_parser import parse_prompt_attention

    weights = torch.ones(formatted_token_len, dtype=dtype, device=device)

    parsed = parse_prompt_attention(prompt) if prompt else []
    # Fast path: no non-neutral weight -> identity (also covers empty prompt).
    if not parsed or all(abs(w - 1.0) <= 1e-6 for _, w in parsed):
        return weights

    # Rebuild the chat template around the content. The concatenation of the fragment texts
    # equals the original prompt, so ``apply_chat_template`` reproduces exactly the string the
    # encoder tokenized.
    def format_with(content: str) -> str:
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )

    def tok_ids(text: str):
        # The encoder tokenizes the ALREADY-formatted string, so the chat template's own special
        # tokens are literal text here; add_special_tokens=False so we don't prepend an extra BOS
        # the encoder path would not have added.
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

    # Locate the content region inside the formatted token stream by token-level common-prefix
    # diffing between the empty-content and full-content formattings. Prefix tokens are shared at
    # the front; suffix tokens are shared at the back; the content tokens sit between them. This
    # is robust to whatever role/BOS/EOS tokens the template inserts (they all live in the fixed
    # prefix/suffix and keep weight 1.0).
    empty_ids = tok_ids(format_with(""))
    full_ids = tok_ids(format_with(prompt))

    # Common prefix length.
    prefix_len = 0
    max_pref = min(len(empty_ids), len(full_ids))
    while prefix_len < max_pref and empty_ids[prefix_len] == full_ids[prefix_len]:
        prefix_len += 1
    # Common suffix length (not overlapping the counted prefix).
    suffix_len = 0
    while (suffix_len < (min(len(empty_ids), len(full_ids)) - prefix_len)
           and empty_ids[len(empty_ids) - 1 - suffix_len] == full_ids[len(full_ids) - 1 - suffix_len]):
        suffix_len += 1

    content_start = prefix_len
    content_len = len(full_ids) - prefix_len - suffix_len
    if content_len <= 0:
        return weights  # nothing localisable (unexpected) -> identity, safe.

    # Walk the parsed fragments in order; progressively tokenize the growing content (formatted)
    # and diff token counts to get each fragment's token span WITHIN the content region.
    def content_tok_count(content: str) -> int:
        ids = tok_ids(format_with(content))
        # Re-derive content length for this partial content using the same prefix/suffix.
        return max(0, len(ids) - prefix_len - suffix_len)

    prev = 0
    current_content = ""
    for text, weight in parsed:
        if not text:
            continue
        current_content += text
        cur = content_tok_count(current_content)
        if abs(weight - 1.0) > 1e-6:
            for j in range(prev, cur):
                pos = content_start + j
                if 0 <= pos < formatted_token_len:
                    weights[pos] = weight
        prev = cur

    return weights


def apply_negpip_to_value(value: torch.Tensor, ctx: NegPipContext) -> torch.Tensor:
    """Scale the caption (suffix) portion of the attention VALUE by signed per-token weights.

    ``value`` has shape [bsz, seq, n_kv_heads, head_dim] (before RoPE/norm, straight from
    ``to_v``). Caption tokens occupy ``[image_len:]`` in each row. Row ``b`` is scaled by
    ``ctx.weight_rows[b]`` over its first ``len(weight_rows[b])`` caption positions; image-prefix
    tokens and inner-pad caption tokens keep weight 1.0. Byte-safe no-op if the batch layout does
    not match (returns ``value`` unchanged).
    """
    rows = ctx.weight_rows
    il = ctx.image_len
    bsz, seq = value.shape[0], value.shape[1]

    if rows is None or len(rows) != bsz or il < 0 or il > seq:
        return value

    for b in range(bsz):
        w = rows[b]
        if w is None or w.numel() == 0:
            continue
        n = min(w.shape[0], seq - il)
        if n <= 0:
            continue
        wv = w[:n].to(device=value.device, dtype=value.dtype)
        # value[b, il:il+n] : [n, n_kv_heads, head_dim] *= [n, 1, 1]
        value[b, il:il + n] = value[b, il:il + n] * wv[:, None, None]
    return value
