"""NegPip (negative-weight prompt emphasis) for the Anima DiT.

NegPip lets a token written with a NEGATIVE emphasis weight — e.g.
``(worst quality:-1)`` — have its cross-attention VALUE (V) negated, so the
concept is SUBTRACTED from the image instead of added. Positive weights scale V
up as usual. This is a single elementwise, per-token, signed scale of the
TEXT-context V inside the existing cross-attention: Q and K are left untouched
(the attention *pattern* is the normal one), so there is NO extra forward pass —
iter speed is unchanged (unlike NAG, which runs a second attention).

Where the scale lands (Anima specifics)
---------------------------------------
Every Anima ``Block`` runs an image self-attention followed by a *cross*
attention where image (query) tokens attend to the text context
``crossattn_emb``. That context is the OUTPUT of the model's ``LLMAdapter``,
whose target sequence is built from the **T5 tokens** (``target_input_ids``):
``x = embed(target_input_ids)`` then cross-attends onto the Qwen3 source hidden
states. So the cross-attention K/V sequence is aligned 1:1 with the **T5 token
positions**, NOT the Qwen3 tokens. NegPip therefore builds its signed per-token
weight vector against the **T5 tokenizer** and aligns it to ``t5_input_ids``.

The vendored cross-attention (``anima_models.Attention``) produces q/k/v as
``[B, L, H, D]`` (bshd). NegPip scales ``v`` along the L (text-token) axis:
``v[:, t, :, :] *= weight[t]``. BOS/EOS/padding weights are 1.0 (identity).

CFG per-context handling
------------------------
Anima runs CFG as two SEPARATE forward passes (cond, then uncond), each batch=1
with a single text context. So the positive-context V uses the positive prompt's
signed weights and the negative-context V uses the negative prompt's signed
weights, simply by arming the relevant weight vector around each pass. A negative
weight in the negative prompt is a double-negative that re-affirms the concept —
this falls out naturally because the same signed-scale rule is applied per
context.

Composition with NAG
--------------------
The signed scale is applied by ``core.inference.nag_anima``'s patched
cross-attention forward, which reads ``self._negpip_weights`` and folds it into
BOTH the positive-context V and (when NAG is armed) the NAG-negative-context V.
That single patch is the only cross-attention interception point, so NAG and
NegPip compose without a second mechanism. When NAG is inactive this wrapper
installs the SAME patched forward (armed only for NegPip), so behaviour is
identical whether or not NAG is present.

Everything is OFF by default: a wrapper is only built when the prompt actually
contains a negative weight (see ``negpip_active`` / ``build_anima_negpip``).
Positive-only prompts never touch this module and are byte-identical to before.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# The patched cross-attention forward lives in nag_anima and already knows how
# to read ``self._negpip_weights`` (folded into both the positive and the
# NAG-negative context V). We reuse it so there is a single interception point.
from core.inference.nag_anima import _patched_cross_attn_forward


def build_anima_negpip_weights(prompt: str, t5_input_ids: torch.Tensor,
                               t5_tokenizer, device, dtype) -> Optional[torch.Tensor]:
    """Build the signed per-token weight vector aligned to the T5 token
    sequence that reaches the cross-attention V.

    Returns a 1-D tensor of length ``t5_input_ids.shape[-1]`` (== the cross-attn
    text-token axis) with 1.0 on every position except the content tokens, which
    carry the parsed (possibly negative) emphasis weight. Returns ``None`` when
    the prompt has no non-unit weight (caller then leaves that context unscaled).

    Alignment (progressive tokenization, mirrors
    ``prompt_parser.build_signed_weight_vector`` but for the T5 tokenizer):
      * ``parse_prompt_attention`` yields (text_fragment, weight) pairs.
      * We tokenize the accumulated clean text with ``add_special_tokens=False``
        and assign each newly-produced token the current fragment's weight.
      * T5 has NO BOS; it appends a single EOS (``</s>``) after the content and
        then pads. Those trailing positions keep weight 1.0. So content tokens
        occupy positions ``[0, n_content)`` of ``t5_input_ids`` and the mapping
        is position-for-position from index 0.
    """
    from core.prompts.prompt_parser import parse_prompt_attention

    # Caller (``_anima_negpip_active``) already gated on a negative weight being
    # present in the overall prompt/negative_prompt. Here we build the signed
    # vector for THIS context whenever it carries any non-unit weight, so a
    # NegPip-active prompt honours BOTH its negative weights (subtract) and its
    # positive weights (scale up) on the cross-attention V. Contexts with only
    # unit weights return None and are left untouched.
    parsed = parse_prompt_attention(prompt) if prompt else []
    parsed = [(t, w) for (t, w) in parsed if t and t != "BREAK"]
    if not parsed or not any(abs(w - 1.0) > 1e-4 for (_t, w) in parsed):
        return None

    seq_len = int(t5_input_ids.shape[-1])
    weights = torch.ones(seq_len, dtype=dtype, device=device)

    current_text = ""
    prev_count = 0
    for text, weight in parsed:
        current_text += text
        try:
            ids = t5_tokenizer(
                current_text, add_special_tokens=False, return_tensors="pt",
            ).input_ids
            cur_count = int(ids.shape[1])
        except Exception:
            continue
        for pos in range(prev_count, cur_count):
            if pos < seq_len:
                weights[pos] = float(weight)
        prev_count = cur_count

    return weights


def negpip_active(prompt: str, negative_prompt: str = "") -> bool:
    """True when NegPip should auto-activate: either the positive OR the negative
    prompt carries a negative emphasis weight. OFF by default otherwise."""
    from core.prompts.prompt_parser import prompt_has_negative_weight
    return bool(prompt_has_negative_weight(prompt or "")
                or prompt_has_negative_weight(negative_prompt or ""))


class AnimaNegPipWrapper(nn.Module):
    """Wraps an Anima transformer to arm a signed per-token V scale on its
    cross-attention for the duration of one forward pass.

    ``forward`` has the same signature as the underlying transformer. Around the
    real forward it sets ``_negpip_weights`` on every block's cross-attention and
    clears it afterwards, so nothing leaks between passes / contexts.

    If the transformer's cross-attentions are already patched (e.g. by an
    ``AnimaNAGWrapper`` — the NAG cond pass), this wrapper does NOT re-patch them;
    it only arms the weights on the existing patched modules. Otherwise it
    installs the SAME patched forward (``nag_anima._patched_cross_attn_forward``)
    with NAG disarmed, so only the NegPip V scale is applied.
    """

    def __init__(self, transformer, token_weights: Optional[torch.Tensor]):
        super().__init__()
        self.transformer = transformer
        self.token_weights = token_weights

        # Unwrap an AnimaNAGWrapper to reach the real transformer whose blocks
        # hold the cross-attention modules. The NAG wrapper already patched them.
        real = transformer
        while hasattr(real, "transformer") and not hasattr(real, "blocks"):
            real = real.transformer
        self._cross_attns = [block.cross_attn for block in real.blocks]

        import types
        self._patched_here = []
        for ca in self._cross_attns:
            already_patched = getattr(ca, "_negpip_or_nag_patched", False)
            if not already_patched:
                self._originals_map = getattr(self, "_originals_map", {})
                self._originals_map[id(ca)] = ca.forward
                ca._nag_armed = getattr(ca, "_nag_armed", False)
                ca._nag_neg_context = getattr(ca, "_nag_neg_context", None)
                ca._negpip_or_nag_patched = True
                ca.forward = types.MethodType(_patched_cross_attn_forward, ca)
                self._patched_here.append(ca)
            # Ensure the attribute exists so the patched forward can read it.
            if not hasattr(ca, "_negpip_weights"):
                ca._negpip_weights = None

    def restore(self):
        """Restore any forwards this wrapper installed and clear NegPip state.

        Only forwards patched *by this wrapper* are restored — a co-active NAG
        wrapper restores its own. NegPip weight state is always cleared.
        """
        originals = getattr(self, "_originals_map", {})
        for ca in self._patched_here:
            orig = originals.get(id(ca))
            if orig is not None:
                ca.forward = orig
            for attr in ("_negpip_or_nag_patched", "_negpip_weights"):
                if hasattr(ca, attr):
                    delattr(ca, attr)
        # For modules we did NOT patch (NAG owns them), just clear our weights.
        for ca in self._cross_attns:
            if ca not in self._patched_here and hasattr(ca, "_negpip_weights"):
                ca._negpip_weights = None

    def forward(self, *args, **kwargs):
        w = self.token_weights
        for ca in self._cross_attns:
            ca._negpip_weights = w
        try:
            out = self.transformer(*args, **kwargs)
        finally:
            for ca in self._cross_attns:
                ca._negpip_weights = None
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
