"""NAG (Normalized Attention Guidance) for the Z-Image transformer.

Z-Image's ``ZImageTransformer2DModel`` runs self-attention (``ZImageAttention``) over a
per-item unified sequence ``[image_tokens ; caption_tokens]`` where the IMAGE tokens are a
PREFIX and the TEXT (caption) tokens are a SUFFIX. There is no processor abstraction, so
NAG is applied by an OFF-by-default branch inside ``ZImageAttention.forward`` (gated by a
module-level flag set only while NAG is active).

Mechanism (mirrors the FLUX.2 / SDXL parity in ``nag_flux2.py``):
  The image latents are IDENTICAL across the batch groups (the same image is denoised
  against the positive text and against the nag-negative text). So the image query tokens
  entering every attention block are the same for the pos group and the nag_neg group.
  After each attention we take the image attention output of the pos group (z_pos) and of
  the nag_neg group (z_neg), extrapolate in attention-OUTPUT space with ``nag_guidance``
  (L2, norm_p=2), and WRITE THE GUIDED RESULT BACK into BOTH the pos and nag_neg image
  groups. Writing back to both keeps the image sub-sequences identical at the next block's
  input, so editing only the attention forward (no full transformer-forward rewrite) is
  correct-by-construction.

Batch layout (set by the denoising loop via ``configure_nag`` / ``clear_nag``):
  - CFG on:  group_size = B, groups = [cfg_neg (uncond), cfg_pos (cond), nag_neg];
             NAG on COND only -> z_pos = cond group image out, z_neg = nag_neg group image
             out. The uncond group is left untouched (standard CFG uncond branch).
  - CFG off: group_size = B, groups = [pos, nag_neg]; NAG on the pos group.

When NAG is off, ``ZImageAttention._nag_ctx`` is ``None`` and the forward is byte-identical.
Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance (MIT).
"""

from typing import Optional

import torch

from core.inference.nag_dit import nag_guidance


class NAGContext:
    """Per-forward NAG configuration shared with every ``ZImageAttention`` via a class attr.

    ``image_len`` is the number of IMAGE tokens (prefix) per unified sequence row; it is the
    same for every batch row because all groups denoise the identical image latents.
    ``group_size`` is B (the distinct-prompt batch); ``has_cfg`` selects the 3-group (CFG)
    vs 2-group (distilled/no-CFG) layout.
    """

    def __init__(self, group_size: int, has_cfg: bool, image_len: int,
                 scale: float, tau: float, alpha: float):
        self.group_size = group_size
        self.has_cfg = has_cfg
        self.image_len = image_len
        self.scale = scale
        self.tau = tau
        self.alpha = alpha


def apply_nag_to_attention_output(hidden_states: torch.Tensor, ctx: NAGContext) -> torch.Tensor:
    """Apply NAG to the image (prefix) tokens of a ``ZImageAttention`` output.

    ``hidden_states`` is the raw attention output of shape [bsz, seq, heads, head_dim]
    (before ``to_out``). NAG operates over the feature axis: heads and head_dim are the
    last two dims, so we treat the flattened per-token feature vector as the norm axis by
    reshaping the image slice. Returns a tensor of the same shape with the pos (and, for
    CFG, only the cond) image tokens replaced by the guided output, written back into both
    the pos and nag_neg groups so their image sub-sequences stay identical.
    """
    b = ctx.group_size
    il = ctx.image_len
    bsz = hidden_states.shape[0]

    # Sanity: batch must match the expected group layout, else skip (byte-safe no-op).
    expected = (3 * b) if ctx.has_cfg else (2 * b)
    if bsz != expected or il <= 0 or il > hidden_states.shape[1]:
        return hidden_states

    # Norm over the full per-token feature vector (heads * head_dim). Flatten the last two
    # dims for the NAG math, then restore.
    def guide(z_pos_4d: torch.Tensor, z_neg_4d: torch.Tensor) -> torch.Tensor:
        shp = z_pos_4d.shape
        zp = z_pos_4d.reshape(shp[0], shp[1], -1)
        zn = z_neg_4d.reshape(shp[0], shp[1], -1)
        out = nag_guidance(zp, zn, scale=ctx.scale, tau=ctx.tau, alpha=ctx.alpha, norm_p=2)
        return out.reshape(shp)

    out = hidden_states.clone()
    if ctx.has_cfg:
        # groups: [uncond (0:b), cond (b:2b), nag_neg (2b:3b)]
        cond = hidden_states[b:2 * b, :il]
        nneg = hidden_states[2 * b:3 * b, :il]
        guided = guide(cond, nneg)
        out[b:2 * b, :il] = guided
        out[2 * b:3 * b, :il] = guided  # keep image sub-sequences identical for next block
    else:
        # groups: [pos (0:b), nag_neg (b:2b)]
        pos = hidden_states[:b, :il]
        nneg = hidden_states[b:2 * b, :il]
        guided = guide(pos, nneg)
        out[:b, :il] = guided
        out[b:2 * b, :il] = guided
    return out
