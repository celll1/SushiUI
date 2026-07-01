"""NegPip (signed attention-value weighting) for the FLUX.2 transformer.

NegPip lets you push *away* from a concept written inside an ordinary prompt by
giving it a NEGATIVE emphasis weight, e.g. ``(worst quality:-1)``. A token with a
negative weight has its attention VALUE (V) negated, so attending to it SUBTRACTS
its concept instead of adding it. Positive weights scale V up. This is a single
elementwise multiply of the TEXT-token V inside the existing attention -> NO extra
forward pass, iter speed unchanged. Q and K are left untouched (the attention
pattern stays the normal one for the unweighted token embedding).

FLUX.2 specifics
----------------
* Text encoder is Qwen3 (NOT CLIP). ``_flux2_encode_prompt`` wraps the prompt in a
  chat template, tokenizes to a fixed ``max_sequence_length`` and takes hidden
  states from layers (9, 18, 27). The embedding is already "clean" (no A1111
  emphasis scaling is ever applied to it), so requirement 5 (let the signed V carry
  ALL the weight) is satisfied by construction -- we only add the signed V scale.
* The transformer attention is joint: text tokens are a PREFIX of the sequence,
  followed by the image (and reference) tokens. NegPip scales only the text prefix
  V (``num_txt_tokens`` positions), never the image V.
* Batch layout matches the NAG hook / the flux2 backend CFG assembly:
    - CFG (no NAG):        text batch [cfg_neg, cfg_pos]
    - CFG + NAG:           text batch [cfg_neg, cfg_pos, nag_neg]
    - distilled (no NAG):  text batch [pos]
    - distilled + NAG:     text batch [pos, nag_neg]
  Each row gets its own signed weight vector (per-context), so a negative weight in
  the negative prompt is the double-negative that re-affirms the concept.

Token -> weight alignment
-------------------------
``build_flux2_signed_weight_vector`` parses the prompt with the shared
``parse_prompt_attention`` (read-only) into (fragment, weight) pairs, then finds the
token positions each fragment occupies *inside the rendered chat template* by
progressive tokenization: it re-renders the template with the cumulative prompt text
and diffs the token count against the previous cumulative text. Positions outside the
prompt content (template prefix/suffix, BOS/EOS, padding) stay 1.0. This is the
FLUX.2 analogue of ``build_signed_weight_vector`` (which assumes CLIP 77-token
chunking and does NOT apply here).

Reuses read-only: core.prompts.prompt_parser (parse_prompt_attention),
core.inference.nag_flux2 (NAG processors, subclassed to compose).
"""

from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn

from core.prompts.prompt_parser import parse_prompt_attention
from core.inference.nag_flux2 import (
    NAGFlux2AttnProcessor,
    NAGFlux2ParallelSelfAttnProcessor,
    Flux2NAGWrapper,
    _sdpa,
)

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    _get_qkv_projections,
    apply_rotary_emb,
)


# ---------------------------------------------------------------------------
# Weight-vector construction (Qwen3 chat-template aware)
# ---------------------------------------------------------------------------

def _render_template(tokenizer, prompt: str) -> str:
    """Render the exact templated text ``_flux2_encode_prompt`` feeds the tokenizer."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _template_token_len(tokenizer, prompt: str, max_length: int) -> int:
    """Number of real (pre-padding) tokens for the templated ``prompt``."""
    text = _render_template(tokenizer, prompt)
    ids = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return ids.shape[1]


def build_flux2_signed_weight_vector(prompt: str, embed_seq_len: int, tokenizer,
                                     device, dtype, max_length: int = 512):
    """Signed per-token weight vector aligned to the Qwen3 chat-template sequence.

    Returns a 1-D tensor of length ``embed_seq_len`` (== the encoded text length,
    fixed padding to ``max_sequence_length``). BOS/EOS, template prefix/suffix and
    padding stay 1.0; each prompt fragment's tokens carry its (possibly negative)
    parsed emphasis weight.

    The prompt content sits between a template prefix and suffix. We locate the
    fragment tokens by progressive tokenization of ``template(cumulative_text)`` and
    diffing token counts; the per-fragment delta is attributed to the token slice
    that fragment adds. Because the tokenizer is not perfectly prefix-monotonic
    across a chat template, positions are clamped to [prefix_len, prefix_len+body].
    """
    out = torch.ones(embed_seq_len, dtype=dtype, device=device)
    parsed = parse_prompt_attention(prompt) if prompt else []
    # No content, or all-neutral weights -> identity (fast path; positive-only prompts
    # never reach here because NegPip is gated on a negative weight upstream).
    if not parsed or all(abs(w - 1.0) < 1e-6 for _, w in parsed):
        return out

    # Template token length with an EMPTY body -> gives (prefix + suffix) length.
    empty_len = _template_token_len(tokenizer, "", max_length)
    # Full body token length (prompt content only, = full - (prefix+suffix)).
    full_len = _template_token_len(tokenizer, prompt, max_length)
    body_tokens = max(full_len - empty_len, 0)
    if body_tokens == 0:
        return out

    # prefix_len: tokens before the body. The chat template appends a fixed suffix
    # (generation prompt) after the content, so the body occupies
    # [prefix_len, prefix_len + body_tokens). We approximate prefix_len as
    # (empty_len - suffix_len). suffix_len is unknown independently, but the delta
    # method below assigns each fragment relative to the running body offset, so we
    # only need where the body STARTS. Tokenize template("") and template(first_frag)
    # is unreliable; instead use the standard assumption that the generation-prompt
    # suffix is what remains after the content. We derive prefix_len from the first
    # non-empty cumulative render.
    #
    # Progressive: cumulative body offset starts at 0; for each fragment we render
    # template(cumulative_text) and take the token length, subtract (prefix+? ) to get
    # cumulative body length. Since prefix+suffix is constant (empty_len) and suffix
    # sits AFTER the body, template(cumulative)_len - suffix_len - prefix_len =
    # cumulative_body_len. We eliminate the unknowns by referencing deltas against the
    # empty render and clamping into the known body window.
    prefix_len = empty_len - _suffix_len(tokenizer, prompt, max_length, empty_len, full_len)
    if prefix_len < 0:
        prefix_len = 0

    body_start = prefix_len
    body_end = min(prefix_len + body_tokens, embed_seq_len)

    cumulative = ""
    prev_body = 0
    for text, weight in parsed:
        if not text:
            continue
        cumulative += text
        cur_full = _template_token_len(tokenizer, cumulative, max_length)
        cur_body = max(cur_full - empty_len, 0)
        cur_body = min(cur_body, body_tokens)
        for j in range(prev_body, cur_body):
            pos = body_start + j
            if body_start <= pos < body_end:
                out[pos] = weight
        prev_body = cur_body

    return out


def _suffix_len(tokenizer, prompt, max_length, empty_len, full_len):
    """Estimate the template suffix (generation-prompt) token length.

    Rendered as: PREFIX + body + SUFFIX. With an empty body the render is
    PREFIX + SUFFIX (length ``empty_len``). We find SUFFIX by tokenizing the
    template of a single-char body and seeing how many trailing tokens are shared
    with the empty render. Falls back to a conservative estimate that keeps the
    body window valid.
    """
    try:
        empty_ids = tokenizer(_render_template(tokenizer, ""), padding=False,
                              truncation=True, max_length=max_length,
                              return_tensors="pt")["input_ids"][0].tolist()
        probe_ids = tokenizer(_render_template(tokenizer, prompt), padding=False,
                              truncation=True, max_length=max_length,
                              return_tensors="pt")["input_ids"][0].tolist()
        # Count shared trailing tokens between the two renders (the suffix).
        s = 0
        while (s < len(empty_ids) and s < len(probe_ids)
               and empty_ids[-1 - s] == probe_ids[-1 - s]):
            s += 1
        # Suffix cannot exceed empty_len; body must be non-negative.
        s = min(s, empty_len)
        # Body length implied by this suffix estimate:
        implied_prefix = empty_len - s
        if implied_prefix < 0:
            s = empty_len
        return s
    except Exception:
        # Conservative fallback: no suffix (body ends at full_len). This still lands
        # weights on real prompt tokens (prefix_len == empty_len would push the body
        # out of range), so clamp to keep body within [0, embed).
        return 0


def build_flux2_negpip_weights(prompt, negative_prompt, tokenizer, device, dtype,
                               embed_seq_len, nag_negative_prompt=None,
                               do_cfg=True, nag_active=False, max_length=512):
    """Build the batched signed weight tensor matching the transformer's text batch.

    Batch order mirrors the flux2 backend / nag_flux2 hook:
      * CFG + NAG:          [cfg_neg, cfg_pos, nag_neg]
      * CFG (no NAG):       [cfg_neg, cfg_pos]
      * distilled + NAG:    [pos, nag_neg]
      * distilled (no NAG): [pos]

    Returns a [txt_b, embed_seq_len] float tensor (1.0 == identity).
    """
    def _wv(text):
        return build_flux2_signed_weight_vector(
            text or "", embed_seq_len, tokenizer, device, dtype, max_length=max_length
        )

    pos = _wv(prompt)
    rows: List[torch.Tensor] = []
    if do_cfg:
        rows.append(_wv(negative_prompt))  # cfg_neg
        rows.append(pos)                    # cfg_pos
    else:
        rows.append(pos)                    # distilled pos
    if nag_active:
        rows.append(_wv(nag_negative_prompt if nag_negative_prompt is not None
                        else (negative_prompt or "")))  # nag_neg
    return torch.stack(rows, dim=0)  # [txt_b, seq]


# ---------------------------------------------------------------------------
# Signed-V helper (shared by all processors)
# ---------------------------------------------------------------------------

def _scale_text_value(enc_v, weights, txt_b):
    """Scale text-token value ``enc_v`` [B, seq, heads, dim] by per-context signed
    ``weights`` [W, seq]. Batch is aligned to the value's leading dim; if the value
    batch was NAG-expanded (txt_b) the weights (also txt_b rows) line up 1:1.
    """
    if weights is None:
        return enc_v
    b = enc_v.shape[0]
    w = weights.to(device=enc_v.device, dtype=enc_v.dtype)
    if w.shape[0] != b:
        if w.shape[0] == 1:
            w = w.expand(b, -1)
        elif w.shape[0] > b:
            w = w[:b]
        else:
            # Pad missing rows with identity (should not happen given batch checks).
            pad = torch.ones(b - w.shape[0], w.shape[1], device=w.device, dtype=w.dtype)
            w = torch.cat([w, pad], dim=0)
    seq = enc_v.shape[1]
    if w.shape[1] != seq:
        if w.shape[1] < seq:
            pad = torch.ones(w.shape[0], seq - w.shape[1], device=w.device, dtype=w.dtype)
            w = torch.cat([w, pad], dim=1)
        else:
            w = w[:, :seq]
    return enc_v * w[:, :, None, None]  # [B, seq, heads, dim] *= [B, seq, 1, 1]


# ---------------------------------------------------------------------------
# Standalone NegPip processors (NAG OFF)
# ---------------------------------------------------------------------------

class NegPipFlux2AttnProcessor:
    """Signed-V NegPip for the dual-stream Flux2Attention (text = encoder stream)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, token_weights: Optional[torch.Tensor] = None):
        self.token_weights = token_weights  # [txt_b, num_txt_tokens]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        query, key, value, enc_q, enc_k, enc_v = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        enc_q = attn.norm_added_q(enc_q.unflatten(-1, (attn.heads, -1)))
        enc_k = attn.norm_added_k(enc_k.unflatten(-1, (attn.heads, -1)))
        enc_v = enc_v.unflatten(-1, (attn.heads, -1))

        # NegPip: signed per-token V on the TEXT (encoder) stream only.
        enc_v = _scale_text_value(enc_v, self.token_weights, enc_v.shape[0])

        query = torch.cat([enc_q, query], dim=1)
        key = torch.cat([enc_k, key], dim=1)
        value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend,
                   self._parallel_config)
        hs = hs.to(query.dtype)

        enc_len = encoder_hidden_states.shape[1]
        txt_hs, img_hs = hs.split_with_sizes([enc_len, hs.shape[1] - enc_len], dim=1)

        img_out = attn.to_out[1](attn.to_out[0](img_hs))
        enc_out = attn.to_add_out(txt_hs)
        return img_out, enc_out


class NegPipFlux2ParallelSelfAttnProcessor:
    """Signed-V NegPip for the single-stream Flux2ParallelSelfAttention.

    The unified sequence is [text; image]; text length ``encoder_hidden_states_length``
    is set by the wrapper. NegPip scales only the first ``n`` (text) V positions.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, token_weights: Optional[torch.Tensor] = None,
                 encoder_hidden_states_length: Optional[int] = None):
        self.token_weights = token_weights
        self.encoder_hidden_states_length = encoder_hidden_states_length

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        fused = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            fused, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        # NegPip: signed V on the text prefix only.
        n = self.encoder_hidden_states_length
        if self.token_weights is not None and n is not None and n > 0:
            txt_v = value[:, :n]
            txt_v = _scale_text_value(txt_v, self.token_weights, txt_v.shape[0])
            value = torch.cat([txt_v, value[:, n:]], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend,
                   self._parallel_config)
        hs = hs.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hs = torch.cat([hs, mlp_hidden_states], dim=-1)
        hs = attn.to_out(hs)
        return hs


# ---------------------------------------------------------------------------
# NAG-composing NegPip processors (NAG ON): subclass the NAG processors and fold
# the signed V into their per-context V before the NAG attention/guidance runs.
# ---------------------------------------------------------------------------

class NegPipNAGFlux2AttnProcessor(NAGFlux2AttnProcessor):
    """NAG dual-stream processor that also applies NegPip signed V on text tokens."""

    def __init__(self, nag_scale, nag_tau, nag_alpha, token_weights=None):
        super().__init__(nag_scale, nag_tau, nag_alpha)
        self.token_weights = token_weights

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from core.inference.nag_flux2 import _expand, _reduce_image
        img_b = hidden_states.shape[0]
        txt_b = encoder_hidden_states.shape[0] if encoder_hidden_states is not None else img_b
        guidance = self.nag_scale > 1.0 and encoder_hidden_states is not None and txt_b > img_b

        query, key, value, enc_q, enc_k, enc_v = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if guidance:
            query = _expand(query, img_b, txt_b)
            key = _expand(key, img_b, txt_b)
            value = _expand(value, img_b, txt_b)

        enc_q = attn.norm_added_q(enc_q.unflatten(-1, (attn.heads, -1)))
        enc_k = attn.norm_added_k(enc_k.unflatten(-1, (attn.heads, -1)))
        enc_v = enc_v.unflatten(-1, (attn.heads, -1))

        # NegPip: signed per-context V on the TEXT stream (rows align with txt_b).
        enc_v = _scale_text_value(enc_v, self.token_weights, enc_v.shape[0])

        query = torch.cat([enc_q, query], dim=1)
        key = torch.cat([enc_k, key], dim=1)
        value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend,
                   self._parallel_config)
        hs = hs.to(query.dtype)

        enc_len = encoder_hidden_states.shape[1]
        txt_hs, img_hs = hs.split_with_sizes([enc_len, hs.shape[1] - enc_len], dim=1)

        if guidance:
            img_hs = _reduce_image(img_hs, img_b, txt_b, self.nag_scale,
                                   self.nag_tau, self.nag_alpha)

        img_out = attn.to_out[1](attn.to_out[0](img_hs))
        enc_out = attn.to_add_out(txt_hs)
        return img_out, enc_out


class NegPipNAGFlux2ParallelSelfAttnProcessor(NAGFlux2ParallelSelfAttnProcessor):
    """NAG single-stream processor that also applies NegPip signed V on text tokens."""

    def __init__(self, nag_scale, nag_tau, nag_alpha,
                 encoder_hidden_states_length=None, origin_img_batch=None,
                 token_weights=None):
        super().__init__(nag_scale, nag_tau, nag_alpha,
                         encoder_hidden_states_length, origin_img_batch)
        self.token_weights = token_weights

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from core.inference.nag_flux2 import _reduce_image, _writeback_image
        guidance = (
            self.nag_scale > 1.0
            and self.encoder_hidden_states_length is not None
            and self.origin_img_batch is not None
            and hidden_states.shape[0] > self.origin_img_batch
        )

        fused = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            fused, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        # NegPip: signed V on the text prefix (rows align with the tiled txt_b batch).
        n = self.encoder_hidden_states_length
        if self.token_weights is not None and n is not None and n > 0:
            txt_v = value[:, :n]
            txt_v = _scale_text_value(txt_v, self.token_weights, txt_v.shape[0])
            value = torch.cat([txt_v, value[:, n:]], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend,
                   self._parallel_config)
        hs = hs.to(query.dtype)

        if guidance:
            n = self.encoder_hidden_states_length
            img_b = self.origin_img_batch
            txt_b = hs.shape[0]
            img_hs = hs[:, n:]
            guided = _reduce_image(img_hs, img_b, txt_b, self.nag_scale,
                                   self.nag_tau, self.nag_alpha)
            hs = hs.clone()
            hs[:, n:] = _writeback_image(img_hs, guided, img_b, txt_b)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hs = torch.cat([hs, mlp_hidden_states], dim=-1)
        hs = attn.to_out(hs)
        return hs


# ---------------------------------------------------------------------------
# Install / restore
# ---------------------------------------------------------------------------

def set_negpip_flux2_processors(transformer, token_weights):
    """Install standalone NegPip processors (NAG OFF). Returns (originals, single_procs)."""
    originals = {}
    single_procs = []
    for name, module in transformer.named_modules():
        cls = module.__class__.__name__
        if cls == "Flux2Attention" and getattr(module, "added_kv_proj_dim", None) is not None:
            originals[name] = module.processor
            module.set_processor(NegPipFlux2AttnProcessor(token_weights=token_weights))
        elif cls == "Flux2ParallelSelfAttention":
            originals[name] = module.processor
            p = NegPipFlux2ParallelSelfAttnProcessor(token_weights=token_weights)
            module.set_processor(p)
            single_procs.append(p)
    n = len(originals)
    print(f"[NegPip-FLUX.2] Installed {n} processors (signed text-V weighting)")
    return originals, single_procs


def restore_processors(transformer, originals):
    for name, module in transformer.named_modules():
        if name in originals:
            module.set_processor(originals[name])


def set_negpip_nag_flux2_processors(transformer, nag_scale, nag_tau, nag_alpha, token_weights):
    """Install NAG+NegPip processors (NAG ON). Returns (originals, single_procs)."""
    originals = {}
    single_procs = []
    for name, module in transformer.named_modules():
        cls = module.__class__.__name__
        if cls == "Flux2Attention" and getattr(module, "added_kv_proj_dim", None) is not None:
            originals[name] = module.processor
            module.set_processor(NegPipNAGFlux2AttnProcessor(
                nag_scale, nag_tau, nag_alpha, token_weights=token_weights))
        elif cls == "Flux2ParallelSelfAttention":
            originals[name] = module.processor
            p = NegPipNAGFlux2ParallelSelfAttnProcessor(
                nag_scale, nag_tau, nag_alpha, token_weights=token_weights)
            module.set_processor(p)
            single_procs.append(p)
    print(f"[NegPip-FLUX.2] Installed {len(originals)} NAG+NegPip processors")
    return originals, single_procs


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class Flux2NegPipWrapper(nn.Module):
    """Reimplements the Flux.2 forward for the NAG-OFF NegPip case.

    Identical batch handling to the plain transformer (no NAG expansion): the only
    reason a wrapper is needed is to set ``encoder_hidden_states_length`` on the
    single-stream NegPip processors each forward (so they know where the text prefix
    ends). Mirrors Flux2NAGWrapper's forward without the NAG image expansion/reduce.

    Independent of block swap (installs its own processors on the raw transformer);
    used only when block swap is off, matching the NAG wrapper's constraint.
    """

    def __init__(self, transformer, token_weights):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = next(transformer.parameters()).device
        self._originals, self._single_procs = set_negpip_flux2_processors(
            transformer, token_weights
        )

    def restore(self):
        restore_processors(self.transformer, self._originals)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        import numpy as np
        transformer = self.transformer
        num_txt_tokens = encoder_hidden_states.shape[1]

        ts = timestep[:1] if timestep.ndim >= 1 else timestep
        ts = ts.to(hidden_states.dtype) * 1000
        g = None
        if guidance is not None:
            g = (guidance[:1] if guidance.ndim >= 1 else guidance).to(hidden_states.dtype) * 1000
        temb = transformer.time_guidance_embed(ts, g)

        double_stream_mod_img = transformer.double_stream_modulation_img(temb)
        double_stream_mod_txt = transformer.double_stream_modulation_txt(temb)
        single_stream_mod = transformer.single_stream_modulation(temb)[0]

        hidden_states = transformer.x_embedder(hidden_states)
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = transformer.pos_embed(img_ids)
        text_rotary_emb = transformer.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for p in self._single_procs:
            p.encoder_hidden_states_length = num_txt_tokens

        for index_block, block in enumerate(transformer.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_block_samples is not None:
                interval = int(np.ceil(len(transformer.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(transformer.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            if controlnet_single_block_samples is not None:
                interval = int(np.ceil(len(transformer.single_transformer_blocks) / len(controlnet_single_block_samples)))
                sample = controlnet_single_block_samples[index_block // interval]
                hidden_states[:, num_txt_tokens:, ...] = hidden_states[:, num_txt_tokens:, ...] + sample

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = transformer.norm_out(hidden_states, temb)
        output = transformer.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)


class Flux2NegPipNAGWrapper(Flux2NAGWrapper):
    """NAG + NegPip: reuses the NAG wrapper's forward (batch expansion, per-context
    image reduce) but installs NAG processors that ALSO apply the signed text-V.

    The NAG wrapper sets ``encoder_hidden_states_length``/``origin_img_batch`` on the
    single-stream processors each forward; our subclassed processors read those plus
    ``token_weights``. Overrides only processor installation.
    """

    def __init__(self, transformer, token_weights, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        nn.Module.__init__(self)
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = next(transformer.parameters()).device
        self.nag_scale, self.nag_tau, self.nag_alpha = nag_scale, nag_tau, nag_alpha
        self._originals, self._single_procs = set_negpip_nag_flux2_processors(
            transformer, nag_scale, nag_tau, nag_alpha, token_weights
        )
