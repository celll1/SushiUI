"""NAG (Normalized Attention Guidance) for the Lens transformer (DiT).

Ports the official MIT method (ChenDarYen/Normalized-Attention-Guidance) to Lens's
single-stream joint attention (``LensJointAttention``), generalized to match how SDXL
and FLUX.2 already do CFG + NAG.

Lens CFG layout (see lens_pipeline_ops.denoise_loop):
    hidden_states = latents.repeat(2, 1, 1)         # image batch 2 = [cond, uncond]
    encoder_features[i]                              # text  batch 2 = [cond, uncond]
    noise_out = transformer(...); cond, uncond = noise_out.chunk(2)

NAG layout (this module):
    text  batch 3 = [cond, uncond, nag_neg]         # nag_neg appended last
    image batch 2 = [cond, uncond] -> 3 = [cond, uncond, cond]   (cond duplicated)
Each batch element attends jointly within its own [image; text] (the joint attention
mask / RoPE are per-element-agnostic and shared), so:
    group 0 (cond image + cond text)      -> z_pos   (image attention output vs positive)
    group 2 (cond image + nag_neg text)   -> z_neg   (image attention output vs nag-negative)
The cond image attention output is replaced by nag_guidance(z_pos, z_neg) and written back
into BOTH the cond (0) and nag_neg (2) image slots so the residual streams stay identical
through the remaining blocks. group 1 (uncond) is untouched. After the final block the
image batch is sliced back to [cond, uncond] (2), giving byte-identical CFG downstream.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from core.inference.nag_dit import nag_guidance


def _nag_reduce_and_writeback_image(
    img_attn: torch.Tensor, scale: float, tau: float, alpha: float
) -> torch.Tensor:
    """Given the per-group image attention output (batch 3 = [cond, uncond, nag_neg]),
    apply NAG on the cond group and write the guided result into both cond and nag_neg.

    Returns a tensor of the same shape (batch 3). The uncond group (index 1) is left as-is.
    """
    z_pos = img_attn[0:1]        # cond image vs cond (positive) text
    z_neg = img_attn[2:3]        # cond image vs nag-negative text
    guided = nag_guidance(z_pos, z_neg, scale=scale, tau=tau, alpha=alpha, norm_p=2)
    out = img_attn.clone()
    out[0:1] = guided            # cond group -> guided
    out[2:3] = guided            # nag_neg group -> guided (keep residual streams identical)
    return out


class LensNAGWrapper(nn.Module):
    """Wraps a LensTransformer2DModel forward to run NAG.

    The image (hidden_states) arrives at batch 2 = [cond, uncond] from the denoise loop.
    The text (encoder_features/mask) arrives at batch 3 = [cond, uncond, nag_neg] (the
    backend appends the nag-negative encoding). This wrapper expands the image to batch 3
    (= [cond, uncond, cond]), enables an OFF-by-default NAG branch on every attention
    module, runs the underlying forward, then slices the image output back to batch 2.

    When NAG is inactive the wrapper is never installed, so the default path is unchanged.
    """

    def __init__(self, transformer, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25):
        super().__init__()
        self.transformer = transformer
        self.nag_scale = float(nag_scale)
        self.nag_tau = float(nag_tau)
        self.nag_alpha = float(nag_alpha)
        # Attach the NAG parameters to each attention module and enable its NAG branch.
        self._attn_modules = []
        for module in transformer.modules():
            if module.__class__.__name__ == "LensJointAttention":
                module._nag_enabled = True
                module._nag_scale = self.nag_scale
                module._nag_tau = self.nag_tau
                module._nag_alpha = self.nag_alpha
                self._attn_modules.append(module)

    def restore(self):
        """Disable the NAG branch on every attention module (restore default path)."""
        for module in self._attn_modules:
            module._nag_enabled = False

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                timestep, img_shapes, attention_kwargs=None):
        img_b = hidden_states.shape[0]
        txt_b = encoder_hidden_states[0].shape[0] if isinstance(
            encoder_hidden_states, (list, tuple)
        ) else encoder_hidden_states.shape[0]

        do_nag = txt_b == img_b + (img_b // 2)   # 2 -> 3 (img_b=2, k=1)

        if do_nag:
            # Expand image [cond, uncond] -> [cond, uncond, cond]: append cond (first half).
            k = img_b // 2
            hidden_states = torch.cat([hidden_states, hidden_states[0:k]], dim=0)
            # timestep is per-image-batch; expand the same way.
            if timestep.ndim >= 1 and timestep.shape[0] == img_b:
                timestep = torch.cat([timestep, timestep[0:k]], dim=0)

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
        )

        if do_nag:
            # Slice image output back to [cond, uncond] (drop the appended nag_neg group).
            output = output[:img_b]
        return output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)


def build_nag_text_batch(
    encoder_features: List[torch.Tensor], encoder_mask: torch.Tensor,
    nag_features: List[torch.Tensor], nag_mask: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Append the nag-negative text encoding to the [cond, uncond] CFG text batch.

    encoder_features: list of [2, S, H] (cond, uncond). nag_features: list of [1, S, H].
    Pads to a common sequence length, then returns list of [3, S', H] and mask [3, S'].
    """
    seq_cfg = encoder_features[0].shape[1]
    seq_nag = nag_features[0].shape[1]
    target = max(seq_cfg, seq_nag)

    def pad_feats(feats, cur):
        if cur == target:
            return feats
        pad = target - cur
        return [torch.cat([f, f.new_zeros((f.shape[0], pad, f.shape[-1]))], dim=1) for f in feats]

    def pad_mask(mask, cur):
        if cur == target:
            return mask
        return torch.cat(
            [mask, torch.zeros((mask.shape[0], target - cur), dtype=torch.bool, device=mask.device)],
            dim=1,
        )

    cfg_feats = pad_feats(encoder_features, seq_cfg)
    nag_feats = pad_feats(nag_features, seq_nag)
    out_feats = [torch.cat([cf, nf], dim=0) for cf, nf in zip(cfg_feats, nag_feats)]
    out_mask = torch.cat([pad_mask(encoder_mask.bool(), seq_cfg), pad_mask(nag_mask.bool(), seq_nag)], dim=0)
    return out_feats, out_mask
