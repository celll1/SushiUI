"""VAE registry + helpers for latent-space MiniT2I (SDXL / FLUX.1).

MiniT2I can train/infer either in pixel space (vae_type="none") or in a VAE latent
space. The latent variants are hardcoded (the usable VAE set is small):

  sdxl  : madebyollin/sdxl-vae-fp16-fix  — AutoencoderKL, 4 latent channels
  flux1 : diffusers/FLUX.1-vae           — AutoencoderKL, 16 latent channels
                                           (same weights as FLUX.1-schnell, Apache-2.0)

Both are diffusers AutoencoderKL, so a single normalisation handles both — read the
scaling/shift from the loaded VAE config:
  latent = (sample - shift_factor) * scaling_factor      (SDXL: 0.13025/0, FLUX.1: 0.3611/0.1159)
  sample = latent / scaling_factor + shift_factor

VAE weights are NOT bundled into MiniT2I checkpoints; they are resolved here from a
local directory (preferred, offline) or the HF repo (cached). FLUX.2 (32ch,
BatchNorm) is intentionally out of scope.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# vae_type -> (HF repo, subfolder, latent channels, preview-decoder kind)
VAE_REGISTRY = {
    "sdxl": {
        "repo": "madebyollin/sdxl-vae-fp16-fix",
        "subfolder": None,
        "channels": 4,
        "preview": "taesdxl",   # taesd.py is_sdxl path
    },
    "flux1": {
        "repo": "diffusers/FLUX.1-vae",
        "subfolder": None,
        "channels": 16,
        "preview": "taef1",     # taesd.py is_zimage (FLUX VAE) path
    },
}

VAE_SCALE_FACTOR = 8  # spatial downsample of these AutoencoderKLs


def is_latent_vae(vae_type: Optional[str]) -> bool:
    return bool(vae_type) and vae_type != "none" and vae_type in VAE_REGISTRY


def vae_latent_channels(vae_type: str) -> int:
    return VAE_REGISTRY[vae_type]["channels"]


def preview_decoder_for(vae_type: str) -> str:
    return VAE_REGISTRY.get(vae_type, {}).get("preview", "matrix")


def _local_candidates(vae_type: str, local_dir: Optional[str]) -> list:
    cands = []
    if local_dir:
        cands.append(local_dir)
    # Common local layouts under the model tree (offline-friendly).
    for root in (os.environ.get("MINIT2I_VAE_DIR"), r"M:\model\minit2i\vae"):
        if root:
            cands.append(os.path.join(root, vae_type))
    return [c for c in cands if c and os.path.isdir(c)]


def load_minit2i_vae(vae_type: str, torch_dtype: torch.dtype = torch.float16,
                     local_dir: Optional[str] = None):
    """Load the AutoencoderKL for the given vae_type (local dir preferred, else HF)."""
    from diffusers import AutoencoderKL

    if vae_type not in VAE_REGISTRY:
        raise ValueError(f"Unknown MiniT2I vae_type '{vae_type}' (expected one of {list(VAE_REGISTRY)})")

    for cand in _local_candidates(vae_type, local_dir):
        print(f"[MiniT2I VAE] Loading {vae_type} VAE from local dir: {cand}")
        vae = AutoencoderKL.from_pretrained(cand, torch_dtype=torch_dtype)
        vae.eval()
        return vae

    # Shared VAE store (sdxl/flux1). MINIT2I_VAE_DIR is preserved via _local_candidates
    # above (takes precedence); the store downloads the default once and reuses it.
    try:
        from core.models.common.vae_store import resolve_vae_dir
        store_dir = resolve_vae_dir(vae_type)
        if store_dir and os.path.isdir(store_dir):
            print(f"[MiniT2I VAE] Loading {vae_type} VAE from shared store: {store_dir}")
            vae = AutoencoderKL.from_pretrained(store_dir, torch_dtype=torch_dtype)
            vae.eval()
            return vae
    except Exception as e:
        print(f"[MiniT2I VAE] shared store resolution failed ({e}); falling back to hub")

    entry = VAE_REGISTRY[vae_type]
    repo, sub = entry["repo"], entry["subfolder"]
    print(f"[MiniT2I VAE] Loading {vae_type} VAE from HF: {repo}" + (f" (subfolder={sub})" if sub else ""))
    kwargs = {"torch_dtype": torch_dtype}
    if sub:
        kwargs["subfolder"] = sub
    vae = AutoencoderKL.from_pretrained(repo, **kwargs)
    vae.eval()
    return vae


def _scale_shift(vae) -> tuple:
    scale = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    shift = float(shift) if shift is not None else 0.0
    return scale, shift


def normalize_latent(sample: torch.Tensor, vae) -> torch.Tensor:
    """Raw VAE sample -> normalised latent: (sample - shift) * scale."""
    scale, shift = _scale_shift(vae)
    return (sample - shift) * scale


def denormalize_latent(latent: torch.Tensor, vae) -> torch.Tensor:
    """Normalised latent -> raw VAE sample: latent / scale + shift."""
    scale, shift = _scale_shift(vae)
    return latent / scale + shift
