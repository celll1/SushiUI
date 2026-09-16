"""Production SDXL donor loading for Chimera bootstrap builds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from core.models.common.vae_source import ResolvedVAE, content_hash_for_state_dict, resolve_vae_source
from core.models.common.vae_store import VAE_REGISTRY


def _validate_sdxl_unet(unet: torch.nn.Module, source: str) -> None:
    config = unet.config
    if int(getattr(config, "in_channels", 0)) != 4:
        raise ValueError(f"Chimera SDXL donor must have four input channels: {source}")
    if int(getattr(config, "out_channels", 0)) != 4:
        raise ValueError(f"Chimera SDXL donor must have four output channels: {source}")
    if int(getattr(config, "cross_attention_dim", 0)) != 2048:
        raise ValueError(f"Chimera SDXL donor must use cross_attention_dim=2048: {source}")
    if getattr(config, "addition_embed_type", None) != "text_time":
        raise ValueError(f"Chimera SDXL donor must use SDXL text_time conditioning: {source}")


def _resolved_vae_from_module(module: torch.nn.Module, source: str) -> ResolvedVAE:
    registry = VAE_REGISTRY["sdxl"]
    scaling = float(registry["scaling_factor"])
    module.register_to_config(scaling_factor=scaling, shift_factor=registry["shift_factor"])
    state = {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}
    config = dict(module.config)
    config["scaling_factor"] = scaling
    config["shift_factor"] = registry["shift_factor"]
    return ResolvedVAE(
        source=f"model:{source}",
        form="model",
        family="sdxl",
        latent_channels=int(registry["latent_channels"]),
        scale_factor=int(registry["scale_factor"]),
        scale_temporal=int(registry["scale_temporal"]),
        ndim=int(registry["latent_ndim"]),
        norm=str(registry["norm"]),
        norm_pack=int(registry["norm_pack"]),
        vae_class="AutoencoderKL",
        config=config,
        content_hash=content_hash_for_state_dict(state),
        provenance=f"extracted:{Path(source).name}",
        locator=None,
        struct_native=True,
        identity_native=None,
        scaling_factor=scaling,
        shift_factor=registry["shift_factor"],
        path=None,
        prefix="first_stage_model." if Path(source).is_file() else "vae.",
        state_dict=state,
    )


def load_sdxl_donor_components(
    source: str,
    *,
    torch_dtype: torch.dtype = torch.float32,
    local_files_only: bool = True,
) -> tuple[torch.nn.Module, ResolvedVAE]:
    """Load only the donor U-Net and VAE from a diffusers directory or LDM file."""
    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SDXL donor not found: {path}")

    from diffusers import AutoencoderKL, UNet2DConditionModel

    if path.is_dir():
        unet_dir = path / "unet"
        vae_dir = path / "vae"
        if not (unet_dir / "config.json").is_file():
            raise FileNotFoundError(f"diffusers SDXL donor has no unet/config.json: {path}")
        if not (vae_dir / "config.json").is_file():
            raise FileNotFoundError(f"diffusers SDXL donor has no vae/config.json: {path}")
        unet = UNet2DConditionModel.from_pretrained(
            str(unet_dir), torch_dtype=torch_dtype, local_files_only=local_files_only
        )
        resolved = resolve_vae_source(
            f"file:{vae_dir}", arch="sdxl", download=False, load_weights=True,
            require_backbone=False,
        )
        # A directory config is authoritative, but the builder requires owned tensors.
        if resolved.state_dict is None:
            module = resolved.load_module(torch_dtype=torch_dtype)
            state = {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}
            resolved = replace(resolved, state_dict=state, content_hash=content_hash_for_state_dict(state))
    elif path.is_file():
        config_repo = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_single_file(
            str(path), config=config_repo, subfolder="unet", torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        vae_module = AutoencoderKL.from_single_file(
            str(path), config=config_repo, subfolder="vae", torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        resolved = _resolved_vae_from_module(vae_module, str(path))
        del vae_module
    else:
        raise ValueError(f"SDXL donor must be a file or diffusers directory: {path}")

    _validate_sdxl_unet(unet, str(path))
    if resolved.latent_channels != 4 or resolved.scale_factor != 8 or resolved.ndim != 4:
        raise ValueError(
            "Chimera requires an SDXL-compatible 4-channel, 8x, 2-D donor VAE; "
            f"got {resolved.latent_channels}ch/{resolved.scale_factor}x/ndim={resolved.ndim}"
        )
    return unet, resolved
