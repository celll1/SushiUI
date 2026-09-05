"""SenseNova's generation-side grid: pixels natively, a VAE latent after a swap.

The single reader of the geometry the design fixes in §10.2 of
`docs/guides/VAE_SWAP_MIGRATION_DESIGN.md`, and the single writer of the two
tensors a swap changes shape. Everything else about this architecture --
including the whole understanding tower and the reference-conditioning path --
is unaffected by a swap and must not read from here.

The geometry, in one place:

* the generation patch is ``P = 4`` latent cells, for EVERY compression ratio.
  The fm_head's ``ps1(2) -> conv1 -> ps2(2) -> ps3(k)`` has total gain ``4k``
  with ``k`` a positive integer, so 4 is the smallest legal patch;
* one token therefore covers ``P * vae_scale_factor`` PIXELS -- 32 with an 8x
  VAE, which is the pixel model's own geometry, and 64 with a 16x one. The
  token COUNT is preserved at a resolution that scales with the VAE;
* the gen ViT's patch embed faces ``P / merge_size = 2`` latent cells, and the
  fm_head's final PixelShuffle factor is ``k = P / 4 = 1``. Neither depends on
  the compression ratio, so no 16x VAE needs extra weight surgery.

This module makes no claim about whether a swapped model trains or generates
well; §10.6 leaves that to measurement on real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from core.models.components.latent_io import ResizeReport

#: The generation patch, on the LATENT grid. Fixed for every ``vae_scale_factor``
#: (§10.2); the pixel model's own patch is 32 and is read off the checkpoint.
GEN_LATENT_PATCH = 4


@dataclass(frozen=True)
class GenGeometry:
    """What the generation branch faces. ``vae_scale_factor == 1`` is pixel space."""

    channels: int
    patch: int
    vit_patch: int
    vae_scale_factor: int

    @property
    def is_latent(self) -> bool:
        return self.vae_scale_factor > 1 or self.patch != 32 or self.channels != 3

    @property
    def token_pixel_width(self) -> int:
        """Pixels one token covers: 32 natively, ``4 * scale`` after a swap."""
        return self.patch * self.vae_scale_factor

    @property
    def head_shuffle(self) -> int:
        """``k``, the fm_head's final PixelShuffle factor."""
        return self.patch // 4


def gen_geometry(transformer) -> GenGeometry:
    """This tree's generation geometry, read from the model the loader built."""
    merge = int(1 / transformer.downsample_ratio)
    patch = int(getattr(transformer, "gen_patch_size", 0)
                or transformer.patch_size * merge)
    return GenGeometry(
        channels=int(getattr(transformer, "gen_in_channels", 0) or 3),
        patch=patch,
        vit_patch=int(getattr(transformer, "gen_vit_patch_size", 0) or patch // merge),
        vae_scale_factor=int(getattr(transformer, "gen_vae_scale_factor", 0) or 1),
    )


def token_pixel_width(transformer) -> int:
    """The pixel grid every canvas dimension must be a multiple of."""
    return gen_geometry(transformer).token_pixel_width


def assert_pixel_aligned(transformer, width: int, height: int, *,
                         label: str = "SenseNova") -> int:
    """Refuse a canvas that is not a whole number of tokens. Returns the width."""
    align = token_pixel_width(transformer)
    if width % align or height % align:
        raise ValueError(
            f"{label}: {width}x{height} is not aligned to the {align}px token "
            f"grid (generation patch {gen_geometry(transformer).patch} x VAE "
            f"compression {gen_geometry(transformer).vae_scale_factor})")
    return align


#: The documented ~4 MP token band of the PIXEL model, in megapixels. Not a
#: parameter and not a bound: an informational range the generation backend
#: warns outside of. `resolution_band_mp` moves it with the token width.
_PIXEL_BAND_MP = (3.0, 5.0)


def resolution_band_mp(token_pixel_width: int) -> tuple:
    """The recommended band in megapixels for this token width.

    The band is a TOKEN-COUNT band quoted in pixels, so it scales with the
    square of the token width -- otherwise every in-range generation on a 16x
    VAE would be reported as out of range (§10.2).
    """
    ratio = (int(token_pixel_width) / 32.0) ** 2
    return (_PIXEL_BAND_MP[0] * ratio, _PIXEL_BAND_MP[1] * ratio)


def _replace_conv(old: nn.Conv2d, *, in_channels: int, out_channels: int,
                  kernel_size: int, stride: Optional[int] = None,
                  padding: int = 0) -> nn.Conv2d:
    new = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride if stride is not None else kernel_size,
                    padding=padding, bias=old.bias is not None)
    return new.to(device=old.weight.device, dtype=old.weight.dtype)


def apply_latent_geometry(
    transformer,
    *,
    channels: int,
    vae_scale_factor: int,
    patch: int = GEN_LATENT_PATCH,
    head_init: str = "zero",
    generator: Optional[torch.Generator] = None,
) -> ResizeReport:
    """Move this tree's generation branch onto a ``channels``-wide latent grid.

    Rebuilds the only two tensors whose SHAPE changes (§10.1): the gen ViT's
    patch embed and the fm_head's ``conv2``. Every other tensor -- the 588
    decoder Linears, ``conv1``, ``dense_embedding``, both embedders, both RoPE
    mechanisms and the entire understanding tower -- is left untouched, which is
    what fixing ``P = 4`` buys.

    Initialisation is §10.3's: the patch embed from a truncated normal at
    ``std = 1/sqrt(fan_in)`` (anima's ``PatchEmbed.init_weights`` convention) so
    the body sees content-dependent features from step 0, and ``conv2`` zeroed so
    ``x_pred`` starts at a defined value that does not depend on the input. A
    zero head also makes the gradient to everything upstream of it zero at step 0
    -- that is a consequence, not a mitigation, and it does NOT tame the
    ``v = -z/(1-t)`` divergence as ``t -> 1``; only ``(1-t).clamp_min(t_eps)``
    does.

    Call BEFORE the optimizer is built: this rebinds Parameters.
    """
    if head_init != "zero":
        # "encoder_pinv" is reserved by §10.3 for a later experiment.
        raise ValueError(
            f"SenseNova latent head init {head_init!r} is not implemented; "
            f"only 'zero' is accepted")
    if channels <= 0:
        raise ValueError(f"latent channel count must be positive, got {channels}")
    if patch <= 0 or patch % 4:
        raise ValueError(
            f"generation patch {patch} must be a positive multiple of 4: the "
            f"fm_head's ps1(2)/ps2(2) leave ps3 a factor of patch/4, which has "
            f"to be a positive integer")
    if not getattr(transformer, "use_pixel_head", False):
        raise RuntimeError(
            "SenseNova's latent migration rebuilds the ConvDecoder (pixel-head) "
            "fm_head; this tree was built with another head layout")

    merge = int(1 / transformer.downsample_ratio)
    vit_patch, remainder = divmod(patch, merge)
    if remainder:
        raise ValueError(
            f"generation patch {patch} is not divisible by the ViT merge size "
            f"{merge}")

    embeddings = transformer.fm_modules["vision_model_mot_gen"].embeddings
    old_embed = embeddings.patch_embedding
    new_embed = _replace_conv(old_embed, in_channels=channels,
                              out_channels=old_embed.out_channels,
                              kernel_size=vit_patch)
    std = 1.0 / math.sqrt(channels * vit_patch * vit_patch)
    with torch.no_grad():
        weight = torch.empty(new_embed.weight.shape, dtype=torch.float32,
                             device="cpu")
        nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std,
                              generator=generator)
        new_embed.weight.copy_(weight.to(new_embed.weight.dtype))
        if new_embed.bias is not None:
            new_embed.bias.zero_()
    embeddings.patch_embedding = new_embed
    embeddings.patch_size = vit_patch
    embeddings.config.patch_size = vit_patch
    embeddings.config.num_channels = channels

    head = transformer.fm_modules["fm_head"]
    shuffle = patch // 4
    old_conv2 = head.conv2
    new_conv2 = _replace_conv(old_conv2, in_channels=old_conv2.in_channels,
                              out_channels=channels * shuffle * shuffle,
                              kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        new_conv2.weight.zero_()
        if new_conv2.bias is not None:
            new_conv2.bias.zero_()
    head.conv2 = new_conv2
    head.ps3 = nn.PixelShuffle(shuffle)

    transformer.gen_in_channels = int(channels)
    transformer.gen_patch_size = int(patch)
    transformer.gen_vit_patch_size = int(vit_patch)
    transformer.gen_vae_scale_factor = int(vae_scale_factor)
    transformer.config.gen_in_channels = int(channels)
    transformer.config.gen_patch_size = int(patch)

    print(f"[SenseNova] generation grid -> {channels}ch latent, patch {patch} "
          f"({patch * vae_scale_factor}px per token at {vae_scale_factor}x): "
          f"patch_embedding {tuple(old_embed.weight.shape)} -> "
          f"{tuple(new_embed.weight.shape)} (trunc normal, std={std:.5f}), "
          f"fm_head.conv2 {tuple(old_conv2.weight.shape)} -> "
          f"{tuple(new_conv2.weight.shape)} (zero)")
    return ResizeReport(
        replaced=("fm_modules.vision_model_mot_gen.embeddings.patch_embedding",
                  "fm_modules.fm_head.conv2"),
        old_in_channels=int(old_embed.in_channels),
        old_out_channels=int(old_conv2.out_channels),
        new_channels=int(channels),
        # Zero COPIED is the whole difference from every other architecture's
        # swap: this is a rebuild, not a channel-axis slice (§10.6-1).
        copied_elements=0,
        new_elements=int(new_embed.weight.numel() + new_conv2.weight.numel()),
    )


def latent_config_dict(config_dict: Optional[Dict[str, Any]], *, channels: int,
                       patch: int = GEN_LATENT_PATCH) -> Dict[str, Any]:
    """The checkpoint's geometry block, carrying this run's generation grid.

    The export re-embeds the block THIS load accepted verbatim
    (``loader._embeddable_sensenova_config``), so a swapped run has to write its
    two keys into it or the saved file rebuilds as a pixel model and fails its
    strict load.
    """
    out = dict(config_dict or {})
    out["gen_in_channels"] = int(channels)
    out["gen_patch_size"] = int(patch)
    return out


def stamp_vae_scale_factor(transformer, vae_scale_factor: int) -> None:
    """Record the compression of the VAE this tree's latents come from.

    A stamp rather than a config key: the number's home is
    ``component.vae.scale_factor`` (§5.2), and two homes could disagree.
    """
    transformer.gen_vae_scale_factor = int(vae_scale_factor or 1)


def _module_to(module) -> Dict[str, Any]:
    """The device/dtype kwargs for a tensor crossing into ``module``."""
    parameter = next(module.parameters())
    return {"device": parameter.device, "dtype": parameter.dtype}


def encode(vae, images: torch.Tensor, *, spec=None,
           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """``[-1,1]`` RGB ``[B,3,H,W]`` -> normalised latent ``[B,C,H/s,W/s]``.

    Sampling, then the shared normalisation layer (§8.4) -- which owns the three
    methods and the packing domain, so nothing here knows which one this VAE uses.
    """
    from core.models.components.vae_registry import normalize

    images = images.to(**_module_to(vae))
    posterior = vae.encode(images)
    dist = getattr(posterior, "latent_dist", None)
    if dist is not None:
        sample = dist.sample(generator=generator) if generator is not None else dist.sample()
    else:
        sample = getattr(posterior, "latent", None)
        if sample is None:
            sample = posterior[0] if isinstance(posterior, (tuple, list)) else None
    if sample is None:
        raise RuntimeError("SenseNova VAE encode returned no latent this path understands")
    return normalize(sample, vae, spec)


def decode(vae, latents: torch.Tensor, *, spec=None) -> torch.Tensor:
    """Normalised latent -> ``[-1,1]`` RGB ``[B,3,H,W]`` (unclamped)."""
    from core.models.components.vae_registry import denormalize

    latents = latents.to(**_module_to(vae))
    decoded = vae.decode(denormalize(latents, vae, spec))
    sample = getattr(decoded, "sample", None)
    if sample is None:
        sample = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
    return sample
