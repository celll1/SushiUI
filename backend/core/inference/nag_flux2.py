"""NAG (Normalized Attention Guidance) for the FLUX.2 transformer.

Ports the official MIT method (ChenDarYen/Normalized-Attention-Guidance) to our Flux.2
(dual-stream `Flux2Attention` + single-stream `Flux2ParallelSelfAttention`), generalized
to match how SDXL already does CFG+NAG.

Batch layout (one processor handles both):
  - distilled (no CFG): image batch B, text batch 2B = [positive; nag_negative].
        image is duplicated -> 2B; NAG on all: guided = nag(img_pos, img_neg). image out = B.
  - CFG:                 image batch 2k = [uncond, cond], text batch 3k =
        [cfg_negative, cfg_positive, nag_negative]. image [uncond, cond] -> [uncond, cond,
        cond] (3k); NAG on COND only: A_cond = nag(cond->cfg_pos, cond->nag_neg),
        A_uncond = uncond->cfg_neg. image out = [A_uncond, A_cond] (2k).
Detection: txt_b == 2*img_b -> distilled; 2*txt_b == 3*img_b -> CFG. One SDPA over the
full text batch (each element attends within its own [text; image]); the image output is
sliced per group and reduced back to img_b. The image (and its modulation temb, batch 1)
stay single; only the text batch carries the pos/neg(/uncond) contexts through the blocks.

Flux2NAGWrapper reimplements the Flux.2 forward (mirrors Flux2BlockSwapWrapper) so NAG
works independently of block swap. Mutually exclusive with block swap in v1.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

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


def _expand(x, img_b, txt_b):
    """Expand an image tensor (batch img_b) to the text batch (txt_b) along dim 0."""
    if txt_b == 2 * img_b:                       # distilled: duplicate
        return x.repeat(*([2] + [1] * (x.ndim - 1)))
    origin = img_b // 2                           # cfg: [uncond, cond] -> [uncond, cond, cond]
    return torch.cat([x, x[origin:2 * origin]], dim=0)


def _reduce_image(img_hs, img_b, txt_b, scale, tau, alpha):
    """Reduce the per-group image attention output (batch txt_b) to img_b via NAG."""
    if txt_b == 2 * img_b:                        # distilled: 2 groups [pos, neg]
        return nag_guidance(img_hs[:img_b], img_hs[img_b:2 * img_b], scale, tau, alpha, norm_p=2)
    k = img_b // 2                                # cfg: 3 groups [uncond, cond_pos, cond_neg]
    a_uncond = img_hs[0:k]
    a_cond = nag_guidance(img_hs[k:2 * k], img_hs[2 * k:3 * k], scale, tau, alpha, norm_p=2)
    return torch.cat([a_uncond, a_cond], dim=0)


def _writeback_image(img_hs, guided, img_b, txt_b):
    """Write the reduced/guided image (batch img_b) back into all txt_b image groups."""
    out = img_hs.clone()
    if txt_b == 2 * img_b:                        # distilled: guided[B] -> both halves
        out[:img_b] = guided
        out[img_b:2 * img_b] = guided
    else:                                         # cfg: guided=[A_uncond(k), A_cond(k)]
        k = img_b // 2
        out[0:k] = guided[:k]                     # uncond group
        out[k:2 * k] = guided[k:2 * k]            # cfg_pos group -> A_cond
        out[2 * k:3 * k] = guided[k:2 * k]        # nag_neg group -> A_cond
    return out


def _sdpa(query, key, value, attention_mask, backend, parallel_config):
    return dispatch_attention_fn(
        query, key, value, attn_mask=attention_mask, backend=backend, parallel_config=parallel_config
    ).flatten(2, 3)


class NAGFlux2AttnProcessor:
    """NAG variant of Flux2AttnProcessor (dual-stream)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
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

        query = torch.cat([enc_q, query], dim=1)
        key = torch.cat([enc_k, key], dim=1)
        value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend, self._parallel_config)
        hs = hs.to(query.dtype)

        enc_len = encoder_hidden_states.shape[1]
        txt_hs, img_hs = hs.split_with_sizes([enc_len, hs.shape[1] - enc_len], dim=1)

        if guidance:
            img_hs = _reduce_image(img_hs, img_b, txt_b, self.nag_scale, self.nag_tau, self.nag_alpha)

        img_out = attn.to_out[1](attn.to_out[0](img_hs))
        enc_out = attn.to_add_out(txt_hs)
        return img_out, enc_out


class NAGFlux2ParallelSelfAttnProcessor:
    """NAG variant of Flux2ParallelSelfAttnProcessor (single-stream, fused QKV+MLP).

    The unified sequence is [text; image] at batch txt_b (image tiled by the wrapper).
    ``encoder_hidden_states_length`` (text length) and ``origin_img_batch`` (img_b) are
    set by the wrapper each forward.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25,
                 encoder_hidden_states_length=None, origin_img_batch=None):
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.encoder_hidden_states_length = encoder_hidden_states_length
        self.origin_img_batch = origin_img_batch

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
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

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hs = _sdpa(query, key, value, attention_mask, self._attention_backend, self._parallel_config)
        hs = hs.to(query.dtype)

        if guidance:
            n = self.encoder_hidden_states_length
            img_b = self.origin_img_batch
            txt_b = hs.shape[0]
            img_hs = hs[:, n:]
            guided = _reduce_image(img_hs, img_b, txt_b, self.nag_scale, self.nag_tau, self.nag_alpha)
            hs = hs.clone()
            hs[:, n:] = _writeback_image(img_hs, guided, img_b, txt_b)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hs = torch.cat([hs, mlp_hidden_states], dim=-1)
        hs = attn.to_out(hs)
        return hs


def set_nag_flux2_processors(transformer, nag_scale, nag_tau, nag_alpha):
    """Install NAG processors on the Flux.2 dual + single attention modules.
    Returns (originals, single_processor_list)."""
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
    """Reimplements the Flux.2 forward with NAG's batch handling, independent of block
    swap. The image batch is img_b, the text (encoder) batch is txt_b (2*img_b distilled,
    or 3k for CFG with img_b=2k)."""

    def __init__(self, transformer, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = next(transformer.parameters()).device
        self.nag_scale, self.nag_tau, self.nag_alpha = nag_scale, nag_tau, nag_alpha
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
        import numpy as np
        transformer = self.transformer
        num_txt_tokens = encoder_hidden_states.shape[1]
        img_b = hidden_states.shape[0]
        txt_b = encoder_hidden_states.shape[0]
        do_nag = txt_b > img_b

        # temb from a single timestep value (same t for all) so it broadcasts to both
        # the image (img_b) and text (txt_b) modulation.
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
            p.origin_img_batch = img_b

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

        # pair the image with the text batch for the single stream
        if do_nag:
            hidden_states = _expand(hidden_states, img_b, txt_b)
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
                if do_nag:
                    sample = _expand(sample, img_b, txt_b)
                hidden_states[:, num_txt_tokens:, ...] = hidden_states[:, num_txt_tokens:, ...] + sample

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        if do_nag:
            hidden_states = hidden_states[:img_b]  # [A_uncond, A_cond] (CFG) or guided (distilled)

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
