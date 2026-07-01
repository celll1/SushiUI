"""NAG (Normalized Attention Guidance) for the FLUX.2 transformer.

Ports the official MIT method (ChenDarYen/Normalized-Attention-Guidance,
nag/attention_flux_nag.py + nag/transformer_flux.py) to our Flux.2 (dual-stream
`Flux2Attention` + single-stream `Flux2ParallelSelfAttention`).

Mechanism (canonical NAG on MM-DiT, no separate evolving image stream):
  - Carry the TEXT as a doubled batch [positive_text; nag_negative_text]; both evolve
    through each block's own projections.
  - The IMAGE stays single: in each attention it is tiled x2 so the image queries attend
    to (positive text + image) and (nag-negative text + image); the two IMAGE outputs are
    extrapolated by nag_guidance (L2), then the SAME guided image is written to both
    halves. Text outputs keep their pos/neg halves.
  - do_nag is detected by image_batch != text_batch. The forward tiles the image before
    the single-stream section and takes one half at the end.

This wrapper reimplements the Flux.2 forward (mirrors Flux2BlockSwapWrapper) so NAG works
independently of block swap. v1 targets the DISTILLED path (guidance vector, no CFG
batch); NAG is mutually exclusive with block swap and with CFG batching for now.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

import inspect
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    _get_qkv_projections,
    apply_rotary_emb,
    dispatch_attention_fn,
)

from core.inference.nag_dit import nag_guidance


def _sdpa(query, key, value, attention_mask, backend, parallel_config):
    out = dispatch_attention_fn(
        query, key, value, attn_mask=attention_mask,
        backend=backend, parallel_config=parallel_config,
    )
    return out.flatten(2, 3)


class NAGFlux2AttnProcessor:
    """NAG variant of Flux2AttnProcessor (dual-stream). Image batch B, text batch 2B."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        apply_guidance = (
            self.nag_scale > 1.0
            and encoder_hidden_states is not None
            and encoder_hidden_states.shape[0] == 2 * hidden_states.shape[0]
        )

        query, key, value, enc_q, enc_k, enc_v = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if apply_guidance:
            # image -> 2x batch (same image in both halves)
            query = query.repeat(2, 1, 1, 1)
            key = key.repeat(2, 1, 1, 1)
            value = value.repeat(2, 1, 1, 1)

        enc_q = enc_q.unflatten(-1, (attn.heads, -1))
        enc_k = enc_k.unflatten(-1, (attn.heads, -1))
        enc_v = enc_v.unflatten(-1, (attn.heads, -1))
        enc_q = attn.norm_added_q(enc_q)
        enc_k = attn.norm_added_k(enc_k)

        # text prefix + image (concat along the sequence dim=1)
        query = torch.cat([enc_q, query], dim=1)
        key = torch.cat([enc_k, key], dim=1)
        value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        enc_len = encoder_hidden_states.shape[1]

        if not apply_guidance:
            hs = _sdpa(query, key, value, attention_mask, self._attention_backend, self._parallel_config)
            hs = hs.to(query.dtype)
            enc_out, img_out = hs.split_with_sizes([enc_len, hs.shape[1] - enc_len], dim=1)
        else:
            q_pos, q_neg = query.chunk(2, dim=0)
            k_pos, k_neg = key.chunk(2, dim=0)
            v_pos, v_neg = value.chunk(2, dim=0)
            hs_pos = _sdpa(q_pos, k_pos, v_pos, attention_mask, self._attention_backend, self._parallel_config).to(query.dtype)
            hs_neg = _sdpa(q_neg, k_neg, v_neg, attention_mask, self._attention_backend, self._parallel_config).to(query.dtype)
            txt_pos, img_pos = hs_pos.split_with_sizes([enc_len, hs_pos.shape[1] - enc_len], dim=1)
            txt_neg, img_neg = hs_neg.split_with_sizes([enc_len, hs_neg.shape[1] - enc_len], dim=1)
            img_out = nag_guidance(img_pos, img_neg, self.nag_scale, self.nag_tau, self.nag_alpha, norm_p=2)
            enc_out = torch.cat([txt_pos, txt_neg], dim=0)  # keep pos/neg text halves (2B)

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out)
        return img_out, enc_out


class NAGFlux2ParallelSelfAttnProcessor:
    """NAG variant of Flux2ParallelSelfAttnProcessor (single-stream, fused QKV+MLP).

    The unified sequence is [text; image] at batch 2B (set up by the wrapper). Only the
    image-token attention output is NAG-guided; the text/MLP paths are untouched.
    ``encoder_hidden_states_length`` (text length) is set by the wrapper each forward.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25, encoder_hidden_states_length=None):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.encoder_hidden_states_length = encoder_hidden_states_length

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        apply_guidance = self.nag_scale > 1.0 and self.encoder_hidden_states_length is not None

        fused = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            fused, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend, self._parallel_config)
        hs = hs.to(query.dtype)

        if apply_guidance:
            n = self.encoder_hidden_states_length
            b = hs.shape[0] // 2
            img = hs[:, n:]                       # [2B, img_seq, dim]
            img_pos = img[:b]
            img_neg = img[b:]
            img_g = nag_guidance(img_pos, img_neg, self.nag_scale, self.nag_tau, self.nag_alpha, norm_p=2)
            # write the SAME guided image into both batch halves
            hs = hs.clone()
            hs[:b, n:] = img_g
            hs[b:, n:] = img_g

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hs = torch.cat([hs, mlp_hidden_states], dim=-1)
        hs = attn.to_out(hs)
        return hs


def set_nag_flux2_processors(transformer, nag_scale, nag_tau, nag_alpha):
    """Install NAG processors on the Flux.2 dual + single attention modules.
    Returns (original_processors_dict, single_processor_list) for restore + per-forward
    text-length update.
    """
    originals = {}
    single_procs = []
    for name, module in transformer.named_modules():
        cls = module.__class__.__name__
        if cls == "Flux2Attention" and getattr(module, "added_kv_proj_dim", None) is not None:
            originals[name] = module.processor
            module.set_processor(NAGFlux2AttnProcessor(nag_scale, nag_tau, nag_alpha))
        elif cls == "Flux2ParallelSelfAttention":
            originals[name] = module.processor
            p = NAGFlux2ParallelSelfAttnProcessor(nag_scale, nag_tau, nag_alpha)
            module.set_processor(p)
            single_procs.append(p)
    return originals, single_procs


def restore_processors(transformer, originals):
    for name, module in transformer.named_modules():
        if name in originals:
            module.set_processor(originals[name])


class Flux2NAGWrapper(nn.Module):
    """Reimplements the Flux.2 forward with the do_nag batch logic, so NAG works
    regardless of block swap. Text arrives doubled [pos; nag_neg]; the image stays single.
    v1: distilled path only (no CFG batch), mutually exclusive with block swap.
    """

    def __init__(self, transformer, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = next(transformer.parameters()).device
        self._originals, self._single_procs = set_nag_flux2_processors(
            transformer, nag_scale, nag_tau, nag_alpha
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
        transformer = self.transformer
        num_txt_tokens = encoder_hidden_states.shape[1]
        do_nag = hidden_states.shape[0] != encoder_hidden_states.shape[0]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = transformer.time_guidance_embed(timestep, guidance)

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

        # tell the single-stream NAG processors the text length for this forward
        for p in self._single_procs:
            p.encoder_hidden_states_length = num_txt_tokens

        # dual stream (image stays single/guided, text doubled)
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
                import numpy as np
                interval_control = int(np.ceil(len(transformer.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # image -> doubled to pair with [pos; neg] text for the single stream
        if do_nag:
            hidden_states = hidden_states.repeat(2, 1, 1)
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
                import numpy as np
                interval_control = int(np.ceil(len(transformer.single_transformer_blocks) / len(controlnet_single_block_samples)))
                sample = controlnet_single_block_samples[index_block // interval_control]
                if do_nag:
                    sample = sample.repeat(2, 1, 1)
                hidden_states[:, num_txt_tokens:, ...] = hidden_states[:, num_txt_tokens:, ...] + sample

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        if do_nag:
            hidden_states = hidden_states.chunk(2, dim=0)[0]  # both halves share the guided image

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
