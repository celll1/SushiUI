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

from dataclasses import replace as _dc_replace
from types import SimpleNamespace
from typing import Optional

import torch

from core.models.components.latent_io import resize_latent_io
from core.models.components.wiring import SD_UNET_LATENT_IO

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
      copy the overlapping INPUT channels, new channels are ZERO.
    - conv_out: Conv2d(hidden, old_out, k, ...) -> Conv2d(hidden, out_channels, k, ...);
      copy the overlapping OUTPUT channels (weight rows + bias), new channels are ZERO.
    - unet.config.in_channels / out_channels updated via register_to_config so downstream
      code that reads them (latent shape, custom sampler) stays consistent.

    No-op when channels already match. The body (all blocks) is untouched.

    The channel algebra lives in ``components.latent_io`` (design §6), shared with
    every other arch. New channels used to be Kaiming-initialised here; they are
    zero as of the shared helper, so a rerun of an old SDXL VAE-swap run does not
    reproduce its earlier result. Reloading a saved swapped checkpoint is
    unaffected: the saved convs overwrite these after the resize.
    """
    out_channels = int(out_channels if out_channels is not None else in_channels)
    in_channels = int(in_channels)

    if unet.conv_in.in_channels == in_channels and unet.conv_out.out_channels == out_channels:
        return

    # SD_UNET_LATENT_IO's paths are "unet.conv_*", so the root is the U-Net's owner.
    root = SimpleNamespace(unet=unet)
    if in_channels == out_channels:
        resize_latent_io(root, SD_UNET_LATENT_IO, in_channels)
    else:
        resize_latent_io(root, _dc_replace(SD_UNET_LATENT_IO, out_module=""), in_channels)
        resize_latent_io(root, _dc_replace(SD_UNET_LATENT_IO, in_module=""), out_channels)

    print(f"[SDXLCustomArch] Resized U-Net conv_in->{in_channels}ch, conv_out->{out_channels}ch "
          f"(channel-partial copy, new channels zero; body unchanged)")


# CompVis/LDM keys for the two channel-dependent conv layers (the save side uses
# convert_unet_state_dict_to_original, so a custom SDXL single-file carries these).
_LDM_CONV_KEYS = {
    "conv_in.weight": "model.diffusion_model.input_blocks.0.0.weight",
    "conv_in.bias": "model.diffusion_model.input_blocks.0.0.bias",
    "conv_out.weight": "model.diffusion_model.out.2.weight",
    "conv_out.bias": "model.diffusion_model.out.2.bias",
}


def load_custom_convs_from_single_file(unet, file_path: str) -> bool:
    """Copy the trained conv_in/conv_out from a custom SDXL single-file into `unet`.

    diffusers from_single_file can override in_channels (num_in_channels=) but NOT
    out_channels, so the file's channel-resized convs are not loaded reliably. After
    resize_unet_in_out, this assigns both convs directly from the CompVis-format file,
    guaranteeing the trained weights regardless of from_single_file's behavior.
    Raises when the file cannot supply either conv (design §8.6.3).
    """
    from safetensors import safe_open

    found = {}
    with safe_open(file_path, framework="pt") as f:
        keys = set(f.keys())
        for diff_k, ldm_k in _LDM_CONV_KEYS.items():
            if ldm_k in keys:
                found[diff_k] = f.get_tensor(ldm_k)

    missing = []
    with torch.no_grad():
        for name, conv in (("conv_in", unet.conv_in), ("conv_out", unet.conv_out)):
            wk, bk = f"{name}.weight", f"{name}.bias"
            if wk in found and tuple(found[wk].shape) == tuple(conv.weight.shape):
                conv.weight.copy_(found[wk].to(conv.weight.device, conv.weight.dtype))
            else:
                shape = tuple(found[wk].shape) if wk in found else None
                missing.append(f"{name}.weight ({'shape ' + str(shape) if shape else 'absent'}, "
                               f"expected {tuple(conv.weight.shape)})")
            if conv.bias is not None and bk in found and tuple(found[bk].shape) == tuple(conv.bias.shape):
                conv.bias.copy_(found[bk].to(conv.bias.device, conv.bias.dtype))
    if missing:
        # Leaving these at the resize's zero init produces a model that loads
        # and generates noise; that failure used to be a print (design §8.6.3).
        raise RuntimeError(
            f"custom-arch checkpoint {file_path} does not supply its trained "
            f"latent convs: {', '.join(missing)}")
    print("[SDXLCustomArch] Loaded trained conv_in/conv_out from single-file")
    return True

