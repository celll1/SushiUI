"""NegPip (negative-emphasis attention) for the MiniT2I MM-JiT transformer.

NegPip lets a token written with a NEGATIVE emphasis weight -- e.g.
``(worst quality:-1)`` -- have its attention VALUE (V) NEGATED, so the concept is
SUBTRACTED from the output rather than added; positive weights scale V up. It is a
single elementwise, per-token SIGNED scale of the TEXT-token V inside the model's
joint attention. Q and K are left untouched (the attention pattern is unchanged);
only the value sign/magnitude changes. No extra forward pass -> iter speed is
unchanged. It AUTO-ACTIVATES (no toggle) whenever the prompt contains any negative
weight, and applies per-token in BOTH the positive prompt (subtracts) and, under
CFG, the negative prompt (a negative weight there is a double-negative that
re-affirms).

MiniT2I is a pixel/latent-space dual-stream MM-DiT (``DoubleStreamDiTBlock``): every
block runs joint attention over ``[text; image]``. Inside that block the text value
is ``v_t`` (the first ``lt`` positions of the concatenated V). NegPip scales ``v_t``
per token per CFG-batch element BEFORE the joint SDPA. This is the SAME attention
interception seam the Phase-2 NAG hook uses (``nag_minit2i._nag_double_block_forward``);
NegPip reuses that seam rather than adding a second interception mechanism, and its
patched forward reproduces the NAG branch (read-only import of ``nag_guidance``) so
the two COMPOSE.

Text tokenizer: MiniT2I encodes prompts with FLAN-T5 (``minit2i_pipeline_ops.encode_prompt``,
``padding="max_length"`` to ``prompt_length``). T5 has NO BOS; it appends a single EOS
(``</s>``) after the content tokens, then pads. So the signed weight vector places the
parsed (possibly negative) weight on each content-token position ``[0, n_content)`` and
leaves EOS + padding = 1.0, matching the text-token V positions in attention.

CFG batch order (see ``minit2i_pipeline_ops._predict_x0_cfg`` and
``nag_minit2i.MiniT2INAGWrapper``): the transformer sees text rows
``[pos, uncond(, nag_neg)]``. NegPip supplies one signed weight row per context:
``pos`` uses the positive prompt's weights, the ``uncond`` row uses the negative
prompt's weights (or 1.0 when there is no separate negative prompt), and the optional
``nag_neg`` scratch row uses the NAG-negative prompt's weights.

Reference (SDXL, merged): core.inference.negpip_processor / pipeline._build_negpip_weights.
"""

from __future__ import annotations

import torch

from core.inference.nag_dit import nag_guidance


# ---------------------------------------------------------------------------
# Signed per-token weight vector (FLAN-T5 aligned)
# ---------------------------------------------------------------------------
def build_signed_weight_vector_t5(prompt, tokenizer, seq_len, device, dtype):
    """Signed per-token weight vector aligned to the FLAN-T5 text sequence.

    ``prompt`` is the ORIGINAL prompt (with emphasis syntax); parse_prompt_attention
    yields clean text fragments + weights, and the fragments' progressive tokenization
    is identical to tokenizing clean_prompt(prompt) -- so this vector lines up with the
    embeddings T5 produced from the cleaned prompt.

    Returns a 1-D tensor of length ``seq_len`` (== ``prompt_length``) with the parsed
    (possibly negative) emphasis weight on each content-token position and 1.0 on the
    EOS + padding positions. Uses progressive tokenization (``add_special_tokens=False``)
    so a fragment's weight lands on exactly the tokens that fragment produces -- the same
    mapping strategy as prompt_parser.build_signed_weight_vector, minus the CLIP 77-token
    chunking (T5 is a flat, BOS-less sequence).
    """
    from core.prompts.prompt_parser import parse_prompt_attention

    weights = torch.ones(seq_len, dtype=dtype, device=device)
    if not prompt:
        return weights
    try:
        parsed = parse_prompt_attention(prompt)
    except Exception:
        return weights
    if not parsed:
        return weights

    current = ""
    prev = 0
    for text, weight in parsed:
        if not text:
            continue
        current += text
        cnt = tokenizer(current, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
        for pos in range(prev, cnt):
            if pos < seq_len:   # never touch EOS / padding tail
                weights[pos] = weight
        prev = cnt
    return weights


def clean_prompt(prompt):
    """Strip A1111 emphasis syntax, leaving only the raw text T5 should encode.

    NegPip carries ALL signed weights on the attention value, so the text encoder must
    see the un-emphasised prompt (e.g. ``(worst quality:-1)`` -> ``worst quality``). The
    fragment order/content here matches build_signed_weight_vector_t5's progressive
    tokenization, so the weight vector lines up with the encoded token sequence.
    """
    from core.prompts.prompt_parser import parse_prompt_attention
    if not prompt:
        return prompt
    try:
        parsed = parse_prompt_attention(prompt)
    except Exception:
        return prompt
    return "".join(text for text, _ in parsed) or prompt


def negpip_eligible(prompt, negative_prompt):
    """Auto-activation gate: True iff either prompt carries a negative weight."""
    from core.prompts.prompt_parser import prompt_has_negative_weight
    return bool(prompt_has_negative_weight(prompt or "")
                or prompt_has_negative_weight(negative_prompt or ""))


# ---------------------------------------------------------------------------
# Patched double-block forward: NegPip V-scale (+ optional NAG branch)
# ---------------------------------------------------------------------------
def _negpip_double_block_forward(block, x, txt, vec, grid_h: int, grid_w: int):
    """DoubleStreamDiTBlock.forward with signed per-token scaling of the TEXT value
    ``v_t`` (NegPip) and, when NAG is also active, the NAG guidance branch on the image
    attention output. Identical to the vendored forward when neither is active.

    Set by MiniT2INegPipWrapper:
      _negpip_active   : bool
      _negpip_weights  : [B_ctx, lt] signed weights, one row per CFG-batch element
                         (row order matches the text batch [pos, uncond(, nag_neg)]).
    Set by MiniT2INAGWrapper (read-only here, reproduced so the two compose):
      _nag_active, _nag_pos_idx, _nag_neg_idx, _nag_scale, _nag_tau, _nag_alpha
    """
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

    # NegPip: signed per-token scale of the TEXT value only (Q, K untouched).
    if getattr(block, "_negpip_active", False):
        w = block._negpip_weights  # [B_ctx, lt]
        if w is not None:
            w = w.to(device=v_t.device, dtype=v_t.dtype)
            # Align batch rows to v_t's batch (broadcast single row if needed).
            if w.shape[0] != b:
                w = w[:1].expand(b, -1) if w.shape[0] == 1 else w[:b]
            # Align seq length to lt (pad tail / truncate with identity 1.0).
            if w.shape[1] != lt:
                if w.shape[1] < lt:
                    pad = torch.ones(w.shape[0], lt - w.shape[1], device=w.device, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                else:
                    w = w[:, :lt]
            v_t = v_t * w[:, :, None, None]   # [b, lt, heads, hd] *= [b, lt, 1, 1]

    q = block.rope(torch.cat([q_t, q_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    k = block.rope(torch.cat([k_t, k_i], dim=1), txt_len=lt, grid_h=grid_h, grid_w=grid_w)
    v = torch.cat([v_t, v_i], dim=1)

    from core.models.minit2i.vendor.mmjit import mem_efficient_sdpa
    out = mem_efficient_sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2).contiguous()  # [b, seq, heads, hd]

    img_out = out[:, lt:].reshape(b, li, -1)
    txt_out = out[:, :lt].reshape(b, lt, -1)

    # Optional NAG branch (reproduced from nag_minit2i so NegPip composes with NAG).
    if getattr(block, "_nag_active", False):
        pos = block._nag_pos_idx
        neg = block._nag_neg_idx
        z_pos = img_out[pos:pos + 1]
        z_neg = img_out[neg:neg + 1]
        guided = nag_guidance(
            z_pos, z_neg,
            scale=block._nag_scale, tau=block._nag_tau, alpha=block._nag_alpha, norm_p=2,
        )
        img_out = img_out.clone()
        img_out[pos:pos + 1] = guided

    x = x + block.img_attn_proj(img_out)
    txt = txt + block.txt_attn_proj(txt_out)
    x = x + block.img_mlp(block.img_norm2(x))
    txt = txt + block.txt_mlp(block.txt_norm2(txt))
    return x, txt


class MiniT2INegPipWrapper:
    """Threads signed per-token text-V weights through the MiniT2I transformer.

    Auto-activated only when the prompt has a negative weight (see negpip_eligible).
    Presents the same call signature the euler loop uses -- ``wrapper(x, t, text, mask)``
    -> predicted x0 -- and, when NAG is also active, delegates the batch layout to the
    installed MiniT2INAGWrapper (whose patched forward is REPLACED here by the unified
    NegPip+NAG forward). Otherwise it reproduces _predict_x0_cfg's batching and sets the
    per-context weight rows to match [pos, uncond].

    ``restore()`` un-patches the block forwards (delegating to the NAG wrapper's restore
    when NAG owns the original forwards).
    """

    def __init__(self, transformer, pos_weights, neg_weights=None, nag_neg_weights=None,
                 nag_wrapper=None):
        self.transformer = transformer
        self.net = transformer.model.net  # MMJiT
        self.pos_weights = pos_weights            # [lt]
        self.neg_weights = neg_weights            # [lt] or None
        self.nag_neg_weights = nag_neg_weights    # [lt] or None
        self.nag_wrapper = nag_wrapper            # MiniT2INAGWrapper or None
        self._orig_forwards = {}
        self._install()

    # ---- block patching -------------------------------------------------
    def _install(self):
        import types
        for block in self.net.double_blocks:
            # If NAG already patched this block, its original is stored in the NAG
            # wrapper; we only stash the *current* forward to restore our layer.
            self._orig_forwards[id(block)] = block.forward
            block._negpip_active = False
            block._negpip_weights = None
            block.forward = types.MethodType(_negpip_double_block_forward, block)

    def restore(self):
        for block in self.net.double_blocks:
            fwd = self._orig_forwards.get(id(block))
            if fwd is not None:
                block.forward = fwd
            for attr in ("_negpip_active", "_negpip_weights"):
                if hasattr(block, attr):
                    delattr(block, attr)

    def _set_weights(self, active, weights):
        """weights: [B_ctx, lt] tensor (row order == text batch) or None."""
        for block in self.net.double_blocks:
            block._negpip_active = active
            block._negpip_weights = weights

    @staticmethod
    def _row(w, lt, device, dtype):
        if w is None:
            return torch.ones(lt, device=device, dtype=dtype)
        return w.to(device=device, dtype=dtype)

    # ---- forward --------------------------------------------------------
    def __call__(self, x, t, text, mask):
        """Runs like the transformer/NAG-wrapper the euler loop uses:
        ``call_target(x, t, text, mask)`` -> x0. The batch is built by the CALLER:

          * Without NAG, _predict_x0_cfg calls this with either
              [cond]         (plain / cfg-off this step) -> rows [pos]
              [cond, uncond] (CFG)                        -> rows [pos, neg]
            matching yy = cat([text, u_text]) == [pos, uncond].
          * With NAG installed, this IS the NAG wrapper's call target: the NAG wrapper
            builds [pos, uncond, nag_neg] (CFG) or [pos, nag_neg] (no-CFG) and calls
            self.net(...) through our patched block forward. We defer to it, setting the
            weight rows for the maximal layout; the NAG forward selects by batch index.

        NegPip attaches one signed weight row per text-context element, so v_t is scaled
        per token per context inside the joint attention.
        """
        device = x.device
        dtype = text.dtype
        lt = text.shape[1]
        pos_row = self._row(self.pos_weights, lt, device, dtype)
        neg_row = self._row(self.neg_weights, lt, device, dtype)
        nag_row = self._row(self.nag_neg_weights, lt, device, dtype)

        if self.nag_wrapper is not None:
            b_in = x.shape[0]
            if b_in == 2:
                rows = torch.stack([pos_row, neg_row, nag_row], dim=0)   # [pos, uncond, nag_neg]
            else:
                rows = torch.stack([pos_row, nag_row], dim=0)            # [pos, nag_neg]
            self._set_weights(True, rows)
            try:
                return self.nag_wrapper(x, t, text, mask)
            finally:
                self._set_weights(False, None)

        # No NAG: the caller (_predict_x0_cfg) has already built the batch. Attach the
        # matching rows by batch size and pass straight through to the net.
        b = x.shape[0]
        if b == 2:
            rows = torch.stack([pos_row, neg_row], dim=0)   # [cond=pos, uncond=neg]
        else:
            rows = pos_row.unsqueeze(0)                      # [pos]
        self._set_weights(True, rows)
        try:
            return self.net(x, t, text, mask)
        finally:
            self._set_weights(False, None)
