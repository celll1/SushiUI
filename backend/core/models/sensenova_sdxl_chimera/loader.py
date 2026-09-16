"""Production reconstruction and preflight for a Chimera directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from core.models.common.single_file_format import read_state_dict, strip_prefix
from core.models.common.vae_source import content_hash_for_state_dict

from .artifact import (
    ChimeraArtifactError,
    checkpoint_headers,
    config_hash,
    find_weights_entry,
    read_artifact_documents,
    resolve_understanding_source,
)
from .conditioning_bridge import ChimeraBridgeConfig, ConditioningBridge
from .unet import parameter_census


def preflight_chimera_artifact(
    directory: str,
    *,
    understanding_override: str | None = None,
) -> dict[str, Any]:
    """Validate documents, pinned understanding source, and weight headers only."""
    root = Path(directory).resolve()
    manifest, config = read_artifact_documents(root)
    understanding_path = resolve_understanding_source(
        manifest["understanding"], override=understanding_override
    )
    weights_path = find_weights_entry(root)
    shapes, metadata = checkpoint_headers(weights_path)
    if metadata.get("model_type") != "sensenova_sdxl_chimera":
        raise ChimeraArtifactError("weights metadata does not declare sensenova_sdxl_chimera")
    if any("_mot_gen" in key for key in shapes):
        raise ChimeraArtifactError("Chimera artifact must not bundle SenseNova generation tensors")
    for prefix in ("condition_bridge.", "unet.", "vae."):
        if not any(key.startswith(prefix) for key in shapes):
            raise ChimeraArtifactError(f"Chimera weights carry no {prefix} tensors")
    if config_hash(config["unet"]) != manifest["unet"]["config_hash"]:
        raise ChimeraArtifactError("U-Net config hash differs from chimera.json")
    if config_hash(config["vae"]) != manifest["vae"]["config_hash"]:
        raise ChimeraArtifactError("VAE config hash differs from chimera.json")
    return {
        "root": root,
        "manifest": manifest,
        "config": config,
        "understanding_path": understanding_path,
        "weights_path": weights_path,
        "shapes": shapes,
    }


def _strict_component(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    expected = set(module.state_dict())
    actual = set(state)
    if expected != actual:
        raise ChimeraArtifactError(
            f"{label} tensor census mismatch: missing={sorted(expected-actual)[:5]}, "
            f"unexpected={sorted(actual-expected)[:5]}"
        )
    module.load_state_dict(state, strict=True)


def load_chimera_artifact(
    directory: str,
    *,
    torch_dtype: torch.dtype | None = None,
    vae_dtype: torch.dtype | None = None,
    understanding_override: str | None = None,
    load_understanding: bool = False,
) -> dict[str, Any]:
    preflight = preflight_chimera_artifact(
        directory, understanding_override=understanding_override
    )
    root = preflight["root"]
    manifest = preflight["manifest"]
    config = preflight["config"]
    understanding_path = preflight["understanding_path"]
    weights_path = preflight["weights_path"]
    shapes = preflight["shapes"]

    from diffusers import AutoencoderKL, UNet2DConditionModel

    unet = UNet2DConditionModel.from_config(config["unet"])
    count = sum(parameter.numel() for parameter in unet.parameters())
    if count != int(manifest["unet"]["parameter_count"]):
        raise ChimeraArtifactError(
            f"U-Net parameter count {count} differs from manifest {manifest['unet']['parameter_count']}"
        )
    bridge_dict = dict(config["conditioning_bridge"])
    bridge_dict["selected_layers"] = tuple(bridge_dict["selected_layers"])
    bridge = ConditioningBridge(ChimeraBridgeConfig(**bridge_dict))
    vae = AutoencoderKL.from_config(config["vae"])

    state, _ = read_state_dict(weights_path)
    _strict_component(bridge, strip_prefix(state, "condition_bridge."), "conditioning bridge")
    _strict_component(unet, strip_prefix(state, "unet."), "U-Net")
    vae_state = strip_prefix(state, "vae.")
    if content_hash_for_state_dict(vae_state) != manifest["vae"]["content_hash"]:
        raise ChimeraArtifactError("bundled VAE content hash differs from chimera.json")
    _strict_component(vae, vae_state, "VAE")
    if parameter_census(unet) != {
        key[len("unet.") :]: shape for key, shape in shapes.items() if key.startswith("unet.")
    }:
        raise ChimeraArtifactError("loaded U-Net census differs from weight headers")
    if torch_dtype is not None:
        unet.to(dtype=torch_dtype)
        bridge.to(dtype=torch_dtype)
    effective_vae_dtype = vae_dtype if vae_dtype is not None else torch_dtype
    if effective_vae_dtype is not None:
        vae.to(dtype=effective_vae_dtype)
    for module in (unet, bridge, vae):
        module.eval()
    vae.requires_grad_(False)

    understanding = None
    if load_understanding:
        from .understanding import load_understanding_only

        understanding = load_understanding_only(understanding_path, torch_dtype=torch_dtype)
    return {
        "type": "sensenova_sdxl_chimera",
        "unet": unet,
        "condition_bridge": bridge,
        "vae": vae,
        # Checkpoint saves must preserve the donor VAE bit-for-bit even when the
        # runtime module is cast to a lower precision for encoding.
        "frozen_vae_state": vae_state,
        "understanding": understanding,
        "understanding_path": understanding_path,
        "manifest": manifest,
        "config": config,
        "model_path": str(root),
    }
