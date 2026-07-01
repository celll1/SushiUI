"""NAG (Normalized Attention Guidance) for the Ideogram 4 transformer.

Ports the official MIT method (ChenDarYen/Normalized-Attention-Guidance) to Ideogram 4,
generalized to match how SDXL / FLUX.2 already do NAG. Ideogram 4 has *dual-branch* CFG
(a separate ``unconditional_transformer``), so NAG is applied only to the **conditional
(positive) branch's image tokens** — exactly the SDXL/FLUX.2 "NAG on COND only" parity.

Mechanism (mirrors Flux2NAGWrapper, adapted to Ideogram 4's single self-attention stack):
  * The conditional transformer runs a single packed ``[text][image]`` sequence with a
    block-diagonal segment mask (each sample attends only within itself).
  * We double the *whole* packed batch to ``[positive; nag_negative]``: both halves carry
    the SAME image latent, differing only in the text region. Because the mask is
    block-diagonal, sample 0's image tokens attend to the positive text and sample 1's
    image tokens attend to the nag-negative text — giving ``z_pos`` and ``z_neg`` for the
    image queries in every attention layer, at no extra sequence cost.
  * In each attention layer, after SDPA, the image-token outputs are extrapolated with the
    shared ``nag_guidance`` (L2, norm_p=2) and the guided result is written back into BOTH
    halves' image tokens (so the guided image evolves through the blocks). Text stays
    per-half so each context keeps developing.
  * The transformer's velocity output is taken from half 0 (positive) by the wrapper.

All vendored forward logic (adaln, MRoPE, masks, block-swap offloader) is preserved — only
the attention processor is swapped, and only while NAG is active (gated by the caller).

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from diffusers.models.attention_dispatch import dispatch_attention_fn

from core.inference.nag_dit import nag_guidance
from core.models.ideogram4.vendor.transformer import (
    OUTPUT_IMAGE_INDICATOR,
    _rotate_half,
)


class Ideogram4NAGAttnProcessor:
    """NAG variant of Ideogram4AttnProcessor.

    Identical to the vendored processor except that, when NAG is active (batch doubled to
    ``[pos; nag_neg]``), the image-token attention outputs are extrapolated with
    ``nag_guidance`` and written back into both halves before ``to_out``.

    ``image_token_index`` (1-D LongTensor of image-token positions in the packed sequence,
    shared by both halves) is set by the wrapper each forward. When it is ``None`` the
    processor is byte-identical to the vendored one.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.image_token_index: Optional[torch.Tensor] = None

    def __call__(self, attn, hidden_states, attention_mask, image_rotary_emb):
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

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)

        # NAG: extrapolate the image tokens of the positive half away from the nag-negative
        # half in attention-output space, then write the guided result back into both halves.
        guidance = (
            self.nag_scale > 1.0
            and self.image_token_index is not None
            and hidden_states.shape[0] == 2
        )
        if guidance:
            idx = self.image_token_index
            img_out = hidden_states[:, idx, :]           # (2, num_img, hidden)
            z_pos = img_out[0:1]
            z_neg = img_out[1:2]
            guided = nag_guidance(
                z_pos, z_neg, self.nag_scale, self.nag_tau, self.nag_alpha, norm_p=2
            )                                            # (1, num_img, hidden)
            hidden_states = hidden_states.clone()
            hidden_states[:, idx, :] = guided            # broadcast (1,...) into both halves

        return attn.to_out[0](hidden_states)


def set_nag_ideogram4_processors(transformer, nag_scale, nag_tau, nag_alpha):
    """Install NAG processors on every Ideogram4Attention module. Returns (originals, procs)."""
    originals = {}
    procs = []
    for name, module in transformer.named_modules():
        if module.__class__.__name__ == "Ideogram4Attention":
            originals[name] = module.processor
            p = Ideogram4NAGAttnProcessor(nag_scale, nag_tau, nag_alpha)
            module.set_processor(p)
            procs.append(p)
    return originals, procs


def restore_processors(transformer, originals):
    for name, module in transformer.named_modules():
        if name in originals:
            module.set_processor(originals[name])


class Ideogram4NAGWrapper(nn.Module):
    """Wrap the Ideogram 4 conditional transformer so a forward with a doubled
    ``[positive; nag_negative]`` text batch applies NAG to the image tokens and returns the
    guided **positive** velocity (batch 1), matching the single-branch cond output shape.

    The wrapper only intercepts ``forward``; it delegates every other attribute (dtype,
    ``.to``, ``layers``, ``_block_offloader``, LoRA-wrapped Linears, ...) to the underlying
    transformer, so block swap and LoRA keep working unchanged.
    """

    def __init__(self, transformer, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        super().__init__()
        self.transformer = transformer
        self.nag_scale, self.nag_tau, self.nag_alpha = nag_scale, nag_tau, nag_alpha
        self._originals, self._procs = set_nag_ideogram4_processors(
            transformer, nag_scale, nag_tau, nag_alpha
        )

    def restore(self):
        restore_processors(self.transformer, self._originals)

    @property
    def dtype(self):
        return self.transformer.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attention_kwargs: dict = None,
        return_dict: bool = True,
    ):
        # The caller passes the POSITIVE packed inputs (batch 1) plus the nag-negative text
        # region via ``nag_llm_features`` on the wrapper (set each step). Build the doubled
        # ``[pos; nag_neg]`` batch here so all vendored forward logic runs unchanged.
        nag_llm = getattr(self, "_nag_llm_features", None)
        b = hidden_states.shape[0]
        do_nag = self.nag_scale > 1.0 and nag_llm is not None and b == 1

        if not do_nag:
            out = self.transformer(
                hidden_states=hidden_states, timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                position_ids=position_ids, segment_ids=segment_ids, indicator=indicator,
                attention_kwargs=attention_kwargs, return_dict=False,
            )[0]
            if not return_dict:
                return (out,)
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=out)

        # Double every packed input; half 0 = positive text, half 1 = nag-negative text.
        # Only the text (LLM) region of encoder_hidden_states differs between the halves;
        # image latents, positions, segment ids, indicator and timestep are identical.
        enc_pos = encoder_hidden_states
        enc_neg = nag_llm.to(dtype=enc_pos.dtype, device=enc_pos.device)
        enc = torch.cat([enc_pos, enc_neg], dim=0)
        hs = hidden_states.repeat(2, *([1] * (hidden_states.ndim - 1)))
        pos_ids = position_ids.repeat(2, *([1] * (position_ids.ndim - 1)))
        seg = segment_ids.repeat(2, *([1] * (segment_ids.ndim - 1)))
        ind = indicator.repeat(2, *([1] * (indicator.ndim - 1)))
        if timestep.dim() >= 1 and timestep.shape[0] == 1:
            ts = timestep.repeat(2)
        elif timestep.dim() >= 1:
            ts = torch.cat([timestep, timestep], dim=0)
        else:
            ts = timestep

        # Image-token positions (identical across the batch); drive the NAG processors.
        img_index = (indicator[0] == OUTPUT_IMAGE_INDICATOR).nonzero(as_tuple=True)[0]
        for p in self._procs:
            p.image_token_index = img_index
        try:
            out2 = self.transformer(
                hidden_states=hs, timestep=ts,
                encoder_hidden_states=enc,
                position_ids=pos_ids, segment_ids=seg, indicator=ind,
                attention_kwargs=attention_kwargs, return_dict=False,
            )[0]
        finally:
            for p in self._procs:
                p.image_token_index = None

        # Half 0 carries the NAG-guided (positive) image tokens.
        out = out2[0:1]
        if not return_dict:
            return (out,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=out)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)
