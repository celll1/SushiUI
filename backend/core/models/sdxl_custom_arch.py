"""Custom-architecture helpers for SDXL (high-spec VAE migration).

SDXL's U-Net I/F is fixed (cross_attention_dim=2048, add_embedding in=2816), but the
latent channel count is tied only to conv_in/conv_out. Migrating SDXL to a higher-spec
VAE (e.g. FLUX.1, 16 latent channels) therefore only requires resizing those two conv
layers; the whole transformer body is inherited unchanged.

The VAE registry + normalisation is shared with the latent-MiniT2I path
(core.models.minit2i.minit2i_vae): both are diffusers AutoencoderKL, so
load/normalize/denormalize are identical regardless of the host architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# Reuse the generic AutoencoderKL registry (sdxl 4ch / flux1 16ch; scale/shift from
# the VAE config). The "minit2i" name is historical — the helpers are arch-agnostic.
from core.models.minit2i.minit2i_vae import (  # noqa: F401  (re-exported for callers)
    VAE_REGISTRY,
    VAE_SCALE_FACTOR,
    vae_latent_channels,
    load_minit2i_vae as load_alt_vae,
    normalize_latent,
    denormalize_latent,
)


def resize_unet_in_out(unet, in_channels: int, out_channels: Optional[int] = None) -> None:
    """In-place: resize a diffusers UNet2DConditionModel conv_in/conv_out to a new latent
    channel count, channel-partial copying the overlapping weights (warm start).

    - conv_in:  Conv2d(old_in, hidden, k, ...) -> Conv2d(in_channels, hidden, k, ...);
      copy the overlapping INPUT channels, leave the rest at fresh init.
    - conv_out: Conv2d(hidden, old_out, k, ...) -> Conv2d(hidden, out_channels, k, ...);
      copy the overlapping OUTPUT channels (weight rows + bias).
    - unet.config.in_channels / out_channels updated via register_to_config so downstream
      code that reads them (latent shape, custom sampler) stays consistent.

    No-op when channels already match. The body (all blocks) is untouched.
    """
    out_channels = int(out_channels if out_channels is not None else in_channels)
    in_channels = int(in_channels)

    conv_in = unet.conv_in
    conv_out = unet.conv_out
    cur_in = conv_in.in_channels
    cur_out = conv_out.out_channels
    if cur_in == in_channels and cur_out == out_channels:
        return

    dev = conv_in.weight.device
    dtype = conv_in.weight.dtype

    # --- conv_in: change input channels, keep hidden (out) ---
    if cur_in != in_channels:
        new_in = nn.Conv2d(
            in_channels, conv_in.out_channels,
            kernel_size=conv_in.kernel_size, stride=conv_in.stride,
            padding=conv_in.padding, dilation=conv_in.dilation,
            groups=conv_in.groups, bias=conv_in.bias is not None,
            padding_mode=conv_in.padding_mode,
        ).to(device=dev, dtype=dtype)
        with torch.no_grad():
            n = min(cur_in, in_channels)
            new_in.weight[:, :n] = conv_in.weight[:, :n]
            if conv_in.bias is not None and new_in.bias is not None:
                new_in.bias.copy_(conv_in.bias)
        unet.conv_in = new_in

    # --- conv_out: change output channels, keep hidden (in) ---
    if cur_out != out_channels:
        new_out = nn.Conv2d(
            conv_out.in_channels, out_channels,
            kernel_size=conv_out.kernel_size, stride=conv_out.stride,
            padding=conv_out.padding, dilation=conv_out.dilation,
            groups=conv_out.groups, bias=conv_out.bias is not None,
            padding_mode=conv_out.padding_mode,
        ).to(device=dev, dtype=dtype)
        with torch.no_grad():
            m = min(cur_out, out_channels)
            new_out.weight[:m] = conv_out.weight[:m]
            if conv_out.bias is not None and new_out.bias is not None:
                new_out.bias[:m] = conv_out.bias[:m]
        unet.conv_out = new_out

    # Keep config in sync (read by latent-shape checks and the custom sampler).
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=in_channels, out_channels=out_channels)
    else:  # fallback: best-effort mutate
        try:
            unet.config.in_channels = in_channels
            unet.config.out_channels = out_channels
        except Exception:
            pass

    print(f"[SDXLCustomArch] Resized U-Net conv_in->{in_channels}ch, conv_out->{out_channels}ch "
          f"(channel-partial copy; body unchanged)")
