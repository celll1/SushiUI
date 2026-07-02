"""NegPip (negative-weight prompt emphasis) for the Ideogram 4 transformer.

NegPip lets a token written with a NEGATIVE emphasis weight (e.g. ``(worst quality:-1)``)
have its attention VALUE (V) negated so the concept is SUBTRACTED rather than added;
positive weights scale V up. It is a single per-token signed elementwise multiply of the
TEXT-token V inside the model's self-attention -- Q and K are untouched (the attention
pattern is the normal one), so there is NO extra forward and iter speed is unchanged.

This mirrors the SDXL reference (``negpip_processor.py`` /
``pipeline._build_negpip_weights``) but adapts it to Ideogram 4:

  * Ideogram 4 has NO CLIP text encoder. Text is encoded via ``encode_prompt`` (Qwen3-VL
    behind a chat template) and the packed sequence is ``[left-pad][text][image]`` with the
    text region LEFT-PADDED to ``max_sequence_length``. So the CLIP 77-token chunking of
    ``build_signed_weight_vector_chunked`` does NOT apply -- we build the signed per-token
    weight vector aligned to Ideogram 4's own token sequence (same tokenizer + chat template
    that ``encode_prompt`` uses), via progressive tokenization of cumulative prompt
    fragments (like ``build_signed_weight_vector``). Chat-template prefix/suffix tokens and
    all padding get weight 1.0 (identity).

  * Attention is SELF-attention over the packed sequence, so V is scaled at the TEXT
    positions only. A full-sequence ``[batch, total_seq]`` weight tensor (1.0 everywhere
    except the text region, which carries the signed weights at the left-padded offset)
    selects exactly the text-token V.

  * CFG is DUAL-BRANCH: the unconditional transformer runs the image-only sequence with
    ZEROED text features (``neg_llm_features``). There are no negative-prompt text tokens in
    that branch to weight, so NegPip's "double-negative in the negative prompt" case does not
    apply to Ideogram 4 -- only the positive (conditional) branch's text V is scaled. This is
    a documented limitation (see the module note in the report), consistent with Ideogram 4's
    asymmetric CFG having no textual unconditional context.

Composition:
  * With NAG active, the conditional transformer is wrapped by ``Ideogram4NAGWrapper`` and the
    packed batch is doubled to ``[positive; nag_negative]``. NegPip's per-sample weight tensor
    is doubled the same way: half 0 uses the positive prompt's signed weights, half 1 uses the
    nag-negative prompt's signed weights (a negative weight in the nag-negative prompt is a
    double-negative that re-affirms). The NegPip processors read the current per-sample weight
    tensor set by the NAG wrapper each forward.
  * Without NAG, a single-sample weight tensor is installed once on the (unwrapped)
    transformer's processors.
  * Spectrum only skips whole steps; NegPip lives inside each real forward, so they compose.

Auto-activation: NegPip is installed ONLY when the prompt (or nag-negative prompt) carries a
negative emphasis weight. Positive-only prompts never install these processors and are
byte-identical to before.
"""

from typing import Dict, List, Optional

import torch

from core.models.ideogram4.vendor.transformer import (
    LLM_TOKEN_INDICATOR,
    _rotate_half,
    ideogram4_dispatch_attention,
)


# ---------------------------------------------------------------------------
# Signed per-token weight vector aligned to Ideogram 4's packed sequence
# ---------------------------------------------------------------------------

def build_ideogram4_text_weights(
    prompt: str,
    tokenizer,
    max_sequence_length: int,
    device,
    dtype,
) -> torch.Tensor:
    """Signed per-token weight vector for the LEFT-PADDED text region.

    Returns a 1-D tensor of length ``max_sequence_length`` with 1.0 on padding and on
    chat-template prefix/suffix tokens, and the parsed (possibly negative) emphasis weight
    on each token produced by an emphasized prompt fragment.

    The tokenization exactly mirrors ``encode_prompt``: the prompt is wrapped by the chat
    template (``apply_chat_template(..., add_generation_prompt=True)``) and tokenized with
    ``add_special_tokens=False``; if it exceeds ``max_sequence_length`` the most-recent
    tokens are kept (matching ``encode_prompt``'s ``toks[-max_sequence_length:]``). The text
    is then left-padded so real tokens occupy ``[offset:]``.

    Fragment -> token mapping (progressive tokenization, like
    ``prompt_parser.build_signed_weight_vector``): we tokenize the chat template applied to
    the cumulative prompt text after each fragment and assign that fragment's weight to the
    newly-added tokens. The fixed chat-template prefix and the ``add_generation_prompt``
    suffix therefore keep weight 1.0 because they do not grow with the prompt content.
    """
    from core.prompts.prompt_parser import parse_prompt_attention

    out = torch.ones(max_sequence_length, dtype=dtype, device=device)
    parsed = parse_prompt_attention(prompt) if prompt else []
    if len(parsed) == 0:
        return out

    def _chat_token_count(text: str) -> int:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        toks = tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        return int(toks.shape[0])

    # Full (pre-truncation, pre-pad) chat token count, matching encode_prompt's tokenization
    # (chat template + add_generation_prompt=True). Layout: [prefix][content][suffix].
    full_count = _chat_token_count("".join(t for t, _ in parsed))

    # All positions default to 1.0 (identity): the fixed template prefix, the generation-prompt
    # suffix, and any token a fragment does not cover. Emphasized fragments overwrite their
    # content tokens, which sit at [prefix_len : prefix_len + content_count].
    full_weights = [1.0] * full_count
    prefix_len = _chat_prefix_len(tokenizer)
    cumulative_text = ""
    prev_count = prefix_len
    for text, weight in parsed:
        if not text:
            continue
        cumulative_text += text
        cur = prefix_len + _content_token_count(tokenizer, cumulative_text, prefix_len)
        for pos in range(prev_count, cur):
            if 0 <= pos < full_count:
                full_weights[pos] = weight
        prev_count = cur

    # Truncate to the most-recent max_sequence_length tokens (encode_prompt behavior), then
    # left-pad to max_sequence_length with 1.0.
    n = full_count
    if n > max_sequence_length:
        full_weights = full_weights[-max_sequence_length:]
        n = max_sequence_length
    offset = max_sequence_length - n
    for j, w in enumerate(full_weights):
        out[offset + j] = w
    return out


# Cache of chat-template prefix length per tokenizer instance (id-keyed) to avoid
# re-tokenizing the fixed prefix on every fragment.
_PREFIX_LEN_CACHE: Dict[int, int] = {}


def _chat_prefix_len(tokenizer) -> int:
    """Number of leading chat-template tokens that precede the user content.

    Determined by tokenizing WITHOUT ``add_generation_prompt`` for empty content and taking
    the token count; for the standard Qwen template this is the ``<|im_start|>user\\n`` prefix
    (the content itself contributes 0 tokens for the empty string). This is a fixed prefix, so
    fragment weights are written starting at this position.
    """
    key = id(tokenizer)
    cached = _PREFIX_LEN_CACHE.get(key)
    if cached is not None:
        return cached
    messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    chat_no_gen = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    toks = tokenizer(chat_no_gen, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    prefix = int(toks.shape[0])
    _PREFIX_LEN_CACHE[key] = prefix
    return prefix


def _content_token_count(tokenizer, content_text: str, prefix_len: int) -> int:
    """Number of tokens the given user content contributes, above the fixed prefix.

    Tokenizes the chat template (without the generation-prompt suffix) for this content and
    subtracts the fixed prefix length, giving a monotonically-growing content-token count
    used to map cumulative prompt fragments onto token positions.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": content_text}]}]
    chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    toks = tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    return max(0, int(toks.shape[0]) - prefix_len)


def build_ideogram4_negpip_weights(
    prompt: str,
    tokenizer,
    max_sequence_length: int,
    grid_h: int,
    grid_w: int,
    device,
    dtype,
    nag_negative_prompt: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Build the full-sequence signed V-scaling weights for the packed ``[text][image]`` layout.

    Returns ``{"pos": tensor[total_seq]}`` and, when a nag-negative prompt is supplied,
    additionally ``{"nag_neg": tensor[total_seq]}``. Each tensor is 1.0 on the image region
    and on text padding, carrying the signed emphasis weights at the left-padded text tokens.
    ``total_seq = max_sequence_length + grid_h * grid_w``.
    """
    num_image = grid_h * grid_w

    def _full(text: str) -> torch.Tensor:
        text_w = build_ideogram4_text_weights(
            text or "", tokenizer, max_sequence_length, device, dtype
        )
        image_w = torch.ones(num_image, dtype=dtype, device=device)
        return torch.cat([text_w, image_w], dim=0)

    weights = {"pos": _full(prompt)}
    if nag_negative_prompt is not None:
        weights["nag_neg"] = _full(nag_negative_prompt)
    return weights


# ---------------------------------------------------------------------------
# NegPip attention processor (signed V scaling on text tokens)
# ---------------------------------------------------------------------------

class Ideogram4NegPipAttnProcessor:
    """Ideogram4 self-attention processor that scales V by signed per-token weights.

    ``token_weights`` is a ``[batch, total_seq]`` (or ``[total_seq]``) tensor aligned with the
    packed sequence. When ``None`` this processor is byte-identical to the vendored one.

    The wrapper (NAG active) or the installer (NAG inactive) sets ``token_weights`` to match
    the current per-sample text context so the correct signed weights scale each half's V.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, token_weights: Optional[torch.Tensor] = None):
        self.token_weights: Optional[torch.Tensor] = token_weights

    def __call__(self, attn, hidden_states, attention_mask, image_rotary_emb, segment_ids=None):
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        cos, sin = image_rotary_emb
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)

        # NegPip: signed per-token scaling of V (text tokens carry the signed weights; image
        # tokens and padding carry 1.0). value shape: (B, L, num_heads, head_dim).
        w = self.token_weights
        if w is not None:
            w = w.to(device=value.device, dtype=value.dtype)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            b = value.shape[0]
            seq = value.shape[1]
            if w.shape[0] != b:
                w = w[:1].expand(b, -1) if w.shape[0] == 1 else w[:b]
            if w.shape[1] != seq:
                if w.shape[1] < seq:
                    pad = torch.ones(w.shape[0], seq - w.shape[1], device=w.device, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                else:
                    w = w[:, :seq]
            value = value * w[:, :, None, None]  # (B, L, 1, 1) broadcast over heads/head_dim

        hidden_states = ideogram4_dispatch_attention(
            query,
            key,
            value,
            attention_mask,
            self._attention_backend,
            self._parallel_config,
            segment_ids,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


def set_negpip_ideogram4_processors(transformer, token_weights: Optional[torch.Tensor]):
    """Install NegPip processors on every Ideogram4Attention module.

    Returns ``(originals, procs)`` for restoration and for later per-forward weight updates.
    """
    originals = {}
    procs: List[Ideogram4NegPipAttnProcessor] = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ == "Ideogram4Attention":
            originals[name] = module.processor
            p = Ideogram4NegPipAttnProcessor(token_weights=token_weights)
            module.set_processor(p)
            procs.append(p)
    print(f"[NegPip/Ideogram4] Installed {len(procs)} processors (signed V weighting)")
    return originals, procs


def restore_ideogram4_processors(transformer, originals):
    for name, module in transformer.named_modules():
        if name in originals:
            module.set_processor(originals[name])


# ---------------------------------------------------------------------------
# Combined NAG + NegPip processor (composition)
# ---------------------------------------------------------------------------

def make_negpip_nag_processor_class():
    """Build a NAG-processor subclass that also applies NegPip signed V scaling.

    Imported lazily so the NAG module is only required when NAG is active. The subclass runs
    identical NAG logic (doubled ``[pos; nag_neg]`` batch, image-token extrapolation) but scales
    V by the per-sample signed weights first, so both features compose in one attention pass.
    """
    from core.inference.nag_ideogram4 import Ideogram4NAGAttnProcessor

    class Ideogram4NegPipNAGAttnProcessor(Ideogram4NAGAttnProcessor):
        def __init__(self, nag_scale, nag_tau, nag_alpha, token_weights=None):
            super().__init__(nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha)
            self.token_weights = token_weights

        def __call__(self, attn, hidden_states, attention_mask, image_rotary_emb, segment_ids=None):
            query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
            key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            cos, sin = image_rotary_emb
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
            query = (query * cos) + (_rotate_half(query) * sin)
            key = (key * cos) + (_rotate_half(key) * sin)

            w = self.token_weights
            if w is not None:
                w = w.to(device=value.device, dtype=value.dtype)
                if w.dim() == 1:
                    w = w.unsqueeze(0)
                b = value.shape[0]
                seq = value.shape[1]
                if w.shape[0] != b:
                    w = w[:1].expand(b, -1) if w.shape[0] == 1 else w[:b]
                if w.shape[1] != seq:
                    if w.shape[1] < seq:
                        pad = torch.ones(w.shape[0], seq - w.shape[1], device=w.device, dtype=w.dtype)
                        w = torch.cat([w, pad], dim=1)
                    else:
                        w = w[:, :seq]
                value = value * w[:, :, None, None]

            hidden_states = ideogram4_dispatch_attention(
                query, key, value, attention_mask,
                self._attention_backend, self._parallel_config, segment_ids,
            )
            hidden_states = hidden_states.flatten(2, 3)

            # NAG guidance on the image tokens of the doubled batch (identical to the NAG proc).
            from core.inference.nag_dit import nag_guidance
            guidance = (
                self.nag_scale > 1.0
                and self.image_token_index is not None
                and hidden_states.shape[0] == 2
            )
            if guidance:
                idx = self.image_token_index
                img_out = hidden_states[:, idx, :]
                z_pos = img_out[0:1]
                z_neg = img_out[1:2]
                guided = nag_guidance(
                    z_pos, z_neg, self.nag_scale, self.nag_tau, self.nag_alpha, norm_p=2
                )
                hidden_states = hidden_states.clone()
                hidden_states[:, idx, :] = guided

            return attn.to_out[0](hidden_states)

    return Ideogram4NegPipNAGAttnProcessor


def install_negpip(transformer, token_weights: torch.Tensor):
    """Install NegPip on an Ideogram 4 transformer, composing with NAG if it is wrapped.

    ``transformer`` may be the bare transformer (NAG off) or an ``Ideogram4NAGWrapper``
    (NAG on). Returns a handle dict with ``restore`` (callable) so the caller can undo the
    processor swap after the denoise loop.

    NAG off:  installs plain ``Ideogram4NegPipAttnProcessor`` on the transformer's attention
              modules with a single-sample ``[1, total_seq]`` weight tensor.
    NAG on:   replaces the wrapper's NAG processors with the combined NegPip+NAG processor.
              ``token_weights`` must be ``[2, total_seq]`` = stack([pos, nag_neg]); the
              wrapper keeps driving ``image_token_index`` on the new ``_procs``.
    """
    is_wrapper = transformer.__class__.__name__ == "Ideogram4NAGWrapper"

    if not is_wrapper:
        if token_weights.dim() == 1:
            token_weights = token_weights.unsqueeze(0)
        originals, procs = set_negpip_ideogram4_processors(transformer, token_weights)

        def _restore():
            restore_ideogram4_processors(transformer, originals)

        return {"restore": _restore, "procs": procs}

    # NAG wrapper: swap NAG procs for combined NegPip+NAG procs on the underlying transformer.
    base = transformer.transformer
    combined_cls = make_negpip_nag_processor_class()
    nag_scale, nag_tau, nag_alpha = transformer.nag_scale, transformer.nag_tau, transformer.nag_alpha
    originals = {}
    new_procs = []
    for name, module in base.named_modules():
        if module.__class__.__name__ == "Ideogram4Attention":
            originals[name] = module.processor
            p = combined_cls(nag_scale, nag_tau, nag_alpha, token_weights=token_weights)
            module.set_processor(p)
            new_procs.append(p)
    # Redirect the wrapper's per-forward image_token_index driver to the new procs.
    transformer._procs = new_procs
    print(f"[NegPip/Ideogram4] Installed {len(new_procs)} combined NegPip+NAG processors")

    def _restore():
        restore_ideogram4_processors(base, originals)

    return {"restore": _restore, "procs": new_procs}
