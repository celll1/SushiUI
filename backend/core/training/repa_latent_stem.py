"""Distilled latent-to-SigLIP embedding stem for REPA.

The stem replaces only SigLIP's patch embedding.  The frozen position embedding,
encoder trunk and post-layernorm remain the teacher, so an artifact is valid only
for the VAE encoder/normalisation and teacher checkpoint it was distilled with.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STEM_FORMAT = "sushi.repa_latent_stem.v1"
STEM_GRID = 27


class LatentRepaStem(nn.Module):
    """Small local latent stem producing a fixed 27x27 SigLIP token grid."""

    def __init__(self, in_channels: int, out_dim: int, width: int = 256,
                 grid: int = STEM_GRID) -> None:
        super().__init__()
        if width % 32:
            raise ValueError("REPA latent stem width must be divisible by 32")
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.width = int(width)
        self.grid = int(grid)
        self.in_proj = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.GroupNorm(32, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(32, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.out_proj = nn.Conv2d(width, out_dim, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            latents, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        x = self.in_proj(x)
        x = x + self.blocks(x)
        return self.out_proj(x).flatten(2).transpose(1, 2)


def _config_dict(module) -> Dict[str, Any]:
    config = getattr(module, "config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif not hasattr(config, "get"):
        config = vars(config)
    return dict(config)


def _unwrap_vae(vae):
    inner = getattr(vae, "vae", None)
    return inner if isinstance(inner, nn.Module) else vae


def vae_encoder_identity(vae) -> Tuple[str, Dict[str, Any]]:
    """Hash only weights capable of changing encode output plus normalisation."""
    from core.models.common.vae_source import (
        content_hash_for_state_dict, latent_space_hash, normalization_config,
    )

    vae = _unwrap_vae(vae)
    selected = {
        key: tensor for key, tensor in vae.state_dict().items()
        if key.startswith("encoder.") or key.startswith("quant_conv.")
    }
    if not selected:
        raise ValueError("VAE has no encoder.* or quant_conv.* weights to identify")
    config = _config_dict(vae)
    norm = normalization_config(config)
    return latent_space_hash(content_hash_for_state_dict(selected), config), norm


def file_content_identity(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def tagger_teacher_identity(model_dir: str) -> Tuple[str, str, str]:
    """Return checkpoint-file identity, path and structural base repo."""
    from core.training.repa import _resolve_tagger_checkpoint
    checkpoint, repo = _resolve_tagger_checkpoint(model_dir)
    return file_content_identity(checkpoint), checkpoint, repo


def teacher_content_identity(encoder: nn.Module) -> str:
    """Identity of the materialized teacher, including a LoRA checkpoint's base."""
    from core.models.common.vae_source import content_hash_for_state_dict
    return content_hash_for_state_dict(encoder.state_dict())


def _metadata_strings(metadata: Mapping[str, Any]) -> Dict[str, str]:
    return {
        str(key): (value if isinstance(value, str) else json.dumps(value, sort_keys=True))
        for key, value in metadata.items()
    }


def save_latent_stem(path: str | os.PathLike[str], stem: LatentRepaStem,
                     metadata: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file
    payload = {
        "format": STEM_FORMAT,
        "in_channels": stem.in_channels,
        "out_dim": stem.out_dim,
        "width": stem.width,
        "grid": stem.grid,
        **dict(metadata),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    save_file(
        {key: value.detach().cpu().contiguous()
         for key, value in stem.state_dict().items()},
        str(temporary), metadata=_metadata_strings(payload))
    os.replace(temporary, target)


def read_latent_stem_metadata(path: str | os.PathLike[str]) -> Dict[str, Any]:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw = handle.metadata() or {}
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        try:
            result[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            result[key] = value
    return result


def load_latent_stem(path: str | os.PathLike[str], *, vae, teacher_identity: str,
                     encoder_dim: int, device, dtype) -> Tuple[LatentRepaStem, Dict[str, Any]]:
    """Load an artifact after exact VAE/teacher/dimension validation."""
    from safetensors.torch import load_file

    path = str(path or "").strip().strip('"').strip("'")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"REPA latent stem not found: {path!r}")
    metadata = read_latent_stem_metadata(path)
    if metadata.get("format") != STEM_FORMAT:
        raise ValueError(f"{path}: unsupported REPA latent stem format {metadata.get('format')!r}")
    actual_vae, actual_norm = vae_encoder_identity(vae)
    expected = {
        "vae_encoder_identity": actual_vae,
        "vae_normalization": actual_norm,
        "teacher_identity": teacher_identity,
        "out_dim": int(encoder_dim),
        "grid": STEM_GRID,
    }
    mismatches = [
        f"{key}: artifact={metadata.get(key)!r}, run={value!r}"
        for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"{path}: REPA latent stem identity mismatch; refusing stale targets ("
            + "; ".join(mismatches) + ")")
    stem = LatentRepaStem(
        int(metadata["in_channels"]), int(metadata["out_dim"]),
        int(metadata["width"]), int(metadata["grid"]),
    )
    stem.load_state_dict(load_file(path))
    stem = stem.to(device=device, dtype=dtype).eval()
    stem.requires_grad_(False)
    return stem, metadata


def encode_latent_targets(encoder: nn.Module, stem: LatentRepaStem,
                          latents: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """Run stem -> frozen SigLIP position embedding/trunk -> target token grid."""
    embeddings = getattr(encoder, "embeddings", None)
    trunk = getattr(encoder, "encoder", None)
    post = getattr(encoder, "post_layernorm", None)
    if embeddings is None or trunk is None or post is None:
        raise TypeError("REPA latent stem requires a SigLIP vision transformer")
    position_embedding = getattr(embeddings, "position_embedding", None)
    position_ids = getattr(embeddings, "position_ids", None)
    if position_embedding is None or position_ids is None:
        raise TypeError("REPA latent stem requires fixed SigLIP position embeddings")

    enc_dtype = next(encoder.parameters()).dtype
    hidden = stem(latents.to(dtype=enc_dtype))
    if hidden.shape[1] != position_ids.shape[-1]:
        raise RuntimeError(
            f"REPA latent stem produced {hidden.shape[1]} tokens but teacher expects "
            f"{position_ids.shape[-1]}")
    hidden = hidden + position_embedding(position_ids).to(dtype=hidden.dtype)
    encoded = trunk(inputs_embeds=hidden, return_dict=True)
    feat = post(encoded.last_hidden_state)
    batch, count, dim = feat.shape
    grid = int(round(count ** 0.5))
    if grid * grid != count:
        raise RuntimeError(f"REPA latent teacher produced non-square token count {count}")
    if (grid, grid) != (gh, gw):
        feat = feat.reshape(batch, grid, grid, dim).permute(0, 3, 1, 2)
        feat = F.interpolate(feat, size=(gh, gw), mode="bilinear", align_corners=False)
        feat = feat.permute(0, 2, 3, 1).reshape(batch, gh * gw, dim)
    return feat


def economic_gate(*, redistill_items: int, distill_ms_per_item: float,
                  online_steps: int, batch_size: int, replaced_ms_per_item: float,
                  stem_ms_per_item: float) -> Dict[str, Any]:
    """Phase 5-2 gate 3, with a break-even step count for planning."""
    if int(batch_size) <= 0:
        raise ValueError("economic gate batch_size must be positive")
    saved = float(replaced_ms_per_item) - float(stem_ms_per_item)
    distill_ms = int(redistill_items) * float(distill_ms_per_item)
    online_ms = int(online_steps) * int(batch_size) * saved
    break_even = None if saved <= 0 else distill_ms / (int(batch_size) * saved)
    return {
        "redistill_cost_ms": distill_ms,
        "projected_online_saving_ms": online_ms,
        "break_even_steps": break_even,
        "passes": saved > 0 and distill_ms < online_ms,
    }
