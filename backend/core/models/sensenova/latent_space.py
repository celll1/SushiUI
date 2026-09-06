"""SenseNova's generation-side grid: pixels natively, a VAE latent after a swap.

The single reader of the geometry the design fixes in §10.2 of
`docs/guides/VAE_SWAP_MIGRATION_DESIGN.md`, and the single writer of the two
tensors a swap changes shape. Everything else about this architecture --
including the whole understanding tower and the reference-conditioning path --
is unaffected by a swap and must not read from here.

The geometry, in one place:

* the generation patch ``P`` is measured in LATENT cells and is a per-run
  choice, not a structure. The fm_head's ``ps1(2) -> conv1 -> ps2(2) -> ps3(k)``
  has total gain ``4k`` with ``k`` a positive integer, so the only rule is that
  ``P`` is a positive multiple of 4 -- tokens can be made coarser, never finer.
  ``TRAINING_DEFAULTS["sensenova_gen_patch"]`` is ``0``, which means INHERIT
  (``resolve_gen_patch``), not ``P = 4``;
* one token covers ``P * vae_scale_factor`` PIXELS -- 32 at ``P=4`` on an 8x
  VAE, which is the pixel model's own geometry, so a swap there preserves the
  token count exactly and the transformer does IDENTICAL work (§10.6 measured
  it 8% slower for the added encode). The token count falls as ``1/P**2``:
  ``P=8`` on an 8x VAE is 64px per token, a quarter of the tokens;
* the gen ViT's patch embed faces ``P / merge_size`` latent cells and the
  fm_head's final PixelShuffle factor is ``k = P / 4``. Neither depends on the
  compression ratio, so no 16x VAE needs extra weight surgery.

This module makes no claim about whether any ``P`` or any ``vae_scale_factor``
trains or generates well; §10.6 leaves that to measurement on real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from core.models.components.latent_io import ResizeReport

#: The smallest legal generation patch on the LATENT grid: ``ps1(2)*ps2(2)``
#: leaves ``ps3 = P/4``, which has to be a positive integer. NOT a default --
#: the run's patch is ``resolve_gen_patch(TRAINING_DEFAULTS["sensenova_gen_patch"])``
#: and the checkpoint's is ``config.gen_patch_size``.
MIN_GEN_LATENT_PATCH = 4

#: ``sensenova_gen_patch = 0`` means INHERIT: keep the patch the base checkpoint
#: was built at, or take ``NATIVE_GEN_LATENT_PATCH`` when the base is
#: pixel-space. It is the served default because rebuilding the two latent I/O
#: layers must be something a caller ASKS for: ``update_training_run``
#: materialises every Pydantic default, so any positive default would make a
#: plain UI edit of a coarse-patch run read as a request to replace them.
INHERIT_GEN_PATCH = 0

#: The patch a pixel-space base is migrated at when the run inherits: one token
#: on 32px at an 8x VAE, which is the pixel model's own grid. Not an API
#: default (that is the sentinel above) -- the architecture's own value.
NATIVE_GEN_LATENT_PATCH = 4


def validate_gen_patch(patch: Any, *, label: str = "generation patch") -> int:
    """The one rule on ``P``, asked in one place. Returns it as an int.

    Takes a RESOLVED patch: ``INHERIT_GEN_PATCH`` is refused here, so a caller
    holding a raw config value goes through ``resolve_gen_patch`` first.
    """
    value = int(patch)
    if value <= 0 or value % MIN_GEN_LATENT_PATCH:
        raise ValueError(
            f"{label} {patch!r} must be a positive multiple of "
            f"{MIN_GEN_LATENT_PATCH}: the fm_head's ps1(2)/ps2(2) leave ps3 a "
            f"factor of patch/4, which has to be a positive integer")
    return value


def resolve_gen_patch(patch: Any, *, base_patch: Optional[int] = None,
                      label: str = "generation patch") -> int:
    """A raw ``sensenova_gen_patch`` as a real patch (``INHERIT_GEN_PATCH`` = 0).

    ``0``/``None`` resolves to ``base_patch`` -- the patch the loaded checkpoint
    was BUILT at -- or to ``NATIVE_GEN_LATENT_PATCH`` when the caller has none
    (a pixel-space base, or a listing that does not know the base yet). Any
    other value is an explicit request and must pass ``validate_gen_patch``.
    """
    if patch is None or int(patch) == INHERIT_GEN_PATCH:
        return validate_gen_patch(base_patch or NATIVE_GEN_LATENT_PATCH,
                                  label=label)
    return validate_gen_patch(patch, label=label)


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
        """Pixels one token covers: 32 natively, ``patch * scale`` after a swap."""
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


LATENT_HEAD_TARGET_STD = 1.0
"""What ``x0_pred`` should measure at step 0.

The head predicts the clean latent directly (``sensenova_ops`` scores it as
``mse(x0_pred, x0)/(1-t)**2``), and a VAE's normalisation -- ``scaling_factor``,
``latents_mean``/``latents_std`` or the batchnorm form -- exists precisely to put
that latent at unit variance. Measured across 720 dataset images through the SDXL
VAE: per-image std p1 0.743, p50 0.987, p99 1.235.
"""


def _calibrate_output_scale(conv: nn.Conv2d, target_std: float) -> None:
    """Rescale ``conv`` once, on its first input, so its output std hits target.

    conv2's input is whatever ``act1(conv1(ps1(hidden)))`` happens to produce, and
    nothing at build time knows that scale, so a fan-in init lands the head's
    output at an arbitrary multiple of the latent it is predicting. Measuring one
    real input settles it exactly.

    The measurement runs in a pre-forward hook rather than at build time because
    the activations only exist once the body runs. It uses ``conv2d`` directly, not
    ``conv(...)``, so the probe cannot re-enter this hook, and it rescales BEFORE
    the module computes the output that autograd will see -- a post-forward rescale
    would leave one step's gradient scaled against weights that no longer exist.
    """
    state = {"handle": None}

    def probe(module, args):
        handle, state["handle"] = state["handle"], None
        if handle is None:
            return None
        handle.remove()
        with torch.no_grad():
            out = F.conv2d(args[0].to(module.weight.dtype), module.weight,
                           module.bias, module.stride, module.padding,
                           module.dilation, module.groups)
            observed = float(out.float().std())
            if not (observed > 0.0 and math.isfinite(observed)):
                print(f"[SenseNova] head calibration skipped: output std "
                      f"measured {observed!r}")
                return None
            factor = float(target_std) / observed
            module.weight.mul_(factor)
            if module.bias is not None:
                module.bias.mul_(factor)
        print(f"[SenseNova] head calibrated: x0_pred std {observed:.5f} -> "
              f"{target_std:.5f} (conv2 weights x{factor:.5f})")
        return None

    state["handle"] = conv.register_forward_pre_hook(probe)


def apply_latent_geometry(
    transformer,
    *,
    channels: int,
    vae_scale_factor: int,
    patch: int,
    head_init: str = "zero",
    generator: Optional[torch.Generator] = None,
) -> ResizeReport:
    """Move this tree's generation branch onto a ``channels``-wide latent grid.

    Rebuilds the only two tensors whose SHAPE changes (§10.1): the gen ViT's
    patch embed and the fm_head's ``conv2``. Every other tensor -- the 588
    decoder Linears, ``conv1``, ``dense_embedding``, both embedders, both RoPE
    mechanisms and the entire understanding tower -- is left untouched, at any
    ``patch``: only these two face the grid.

    ``patch`` has no default. A caller that let one apply would build a tree
    whose geometry disagrees with the config block it writes, which loads clean
    and generates noise.

    The patch embed comes from a truncated normal at ``std = 1/sqrt(fan_in)``
    (anima's ``PatchEmbed.init_weights`` convention) so the body sees
    content-dependent features from step 0. ``head_init`` chooses what the output
    convolution starts at:

    ``"zero"`` (§10.3) makes ``x0_pred`` start at a defined value that does not
    depend on the input -- the constant zero, i.e. "predict the mean latent for
    every input". It also makes the gradient to everything UPSTREAM of the head
    zero at step 0, since that gradient is ``W`` times something and ``W`` is zero.

    ``"scaled"`` gives the head a fan-in truncated normal and calibrates it on its
    first real input so ``x0_pred`` measures ``LATENT_HEAD_TARGET_STD``. The
    decoder maps latent amplitude to output contrast near-linearly (measured:
    output std 6.8 / 21.3 / 79.2 at latent amplitude 0.05 / 0.2 / 1.0 of a real
    latent, and the same curve for structureless noise), so a zero head decodes to
    exactly ``decode(0)`` -- one flat colour -- whatever the body has learnt.

    Neither choice tames the ``v = -z/(1-t)`` divergence as ``t -> 1``; only
    ``(1-t).clamp_min(t_eps)`` does.

    Call BEFORE the optimizer is built: this rebinds Parameters.
    """
    if head_init not in ("zero", "scaled"):
        # "encoder_pinv" is reserved by §10.3 for a later experiment.
        raise ValueError(
            f"SenseNova latent head init {head_init!r} is not implemented; "
            f"'zero' and 'scaled' are accepted")
    if channels <= 0:
        raise ValueError(f"latent channel count must be positive, got {channels}")
    patch = validate_gen_patch(patch)
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
    if head_init == "scaled":
        head_std = 1.0 / math.sqrt(new_conv2.in_channels * 3 * 3)
        with torch.no_grad():
            weight = torch.empty(new_conv2.weight.shape, dtype=torch.float32,
                                 device="cpu")
            nn.init.trunc_normal_(weight, std=head_std, a=-3 * head_std,
                                  b=3 * head_std, generator=generator)
            new_conv2.weight.copy_(weight.to(new_conv2.weight.dtype))
        _calibrate_output_scale(new_conv2, LATENT_HEAD_TARGET_STD)
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
          f"{tuple(new_conv2.weight.shape)} ({head_init})")
    return ResizeReport(
        replaced=("fm_modules.vision_model_mot_gen.embeddings.patch_embedding",
                  "fm_modules.fm_head.conv2"),
        old_in_channels=int(old_embed.in_channels),
        old_out_channels=int(old_conv2.out_channels),
        new_channels=int(channels),
        # Zero COPIED is the whole difference from every other architecture's
        # swap: this is a rebuild, not a channel-axis slice (§10.6-1). That holds
        # for both head inits -- "scaled" seeds the head, it does not carry the
        # pixel head's weights across.
        copied_elements=0,
        new_elements=int(new_embed.weight.numel() + new_conv2.weight.numel()),
    )


def latent_config_dict(config_dict: Optional[Dict[str, Any]], *, channels: int,
                       patch: int) -> Dict[str, Any]:
    """The checkpoint's geometry block, carrying this run's generation grid.

    The export re-embeds the block THIS load accepted verbatim
    (``loader._embeddable_sensenova_config``), so a swapped run has to write its
    two keys into it or the saved file rebuilds as a pixel model and fails its
    strict load.

    ``patch`` has no default and must come from the tree that was actually
    built: a defaulted 4 written by a ``P=8`` run would rebuild as the wrong
    geometry on the next load, silently.
    """
    out = dict(config_dict or {})
    out["gen_in_channels"] = int(channels)
    out["gen_patch_size"] = validate_gen_patch(patch)
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
