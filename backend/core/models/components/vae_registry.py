"""VAE component registry — arch-independent VAE component layer (plan A.4).

Generalizes ``core/models/minit2i/minit2i_vae.py``: it hosts the canonical
implementation of the AutoencoderKL registry (``VAE_REGISTRY``, ``VAE_SCALE_FACTOR``,
``is_latent_vae``, ``vae_latent_channels``, ``preview_decoder_for``,
``load_minit2i_vae``, ``normalize_latent`` / ``denormalize_latent``). The old
``core.models.minit2i.minit2i_vae`` module is a thin re-export shim of this one, so
existing imports keep resolving to the SAME objects (identity preserved).

BEHAVIOR FREEZE: the minit2i VAE functions are moved byte-identically (including
the ``[MiniT2I VAE]`` log strings and the ``"none"`` pixel-space sentinel semantics
via ``is_latent_vae``). The shared default-VAE resolver still lives in
``core/models/common/vae_store.py`` (already arch-independent and consumed by many
archs); this module re-exports ``resolve_vae_dir`` / ``vae_identity`` for a unified
component surface and does not move it (least-churn, R6 cache stability).

Spec-driven ``load_vae`` / ``encode`` / ``normalize`` are ADDITIVE (plan A.4): the
pixel-space branch is ``spec.latent_channels == 0`` (mirrors minit2i's
``vae_type=="none"`` / ``is_latent_vae`` pattern), not ``is_minit2i``.
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
    # Common local layouts under the model tree (offline-friendly): an explicit
    # override, else the conventional place in a configured external tree.
    from ..common.model_root import external_model_path

    for root in (os.environ.get("MINIT2I_VAE_DIR"),
                 external_model_path("minit2i", "vae")):
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


# --- Shared resolver re-export (canonical impl stays in common/vae_store) -------
try:  # optional import guard — vae_store has no hard deps but keep resolution lazy-safe
    from core.models.common.vae_store import resolve_vae_dir, vae_identity  # noqa: F401
except Exception:  # pragma: no cover - defensive
    resolve_vae_dir = None
    vae_identity = None


# --- Generalized, spec-driven entry points (ADDITIVE — plan A.4) ----------------

def load_vae(spec_or_type, *, torch_dtype: torch.dtype = torch.float16,
             local_dir: Optional[str] = None):
    """Arch-independent VAE loader. Pixel-space (``spec.latent_channels == 0`` or
    a ``"none"`` vae_type string) returns ``None``; otherwise delegates to the frozen
    ``load_minit2i_vae``. Accepts a vae_type string OR a wiring spec carrying a
    ``vae_type`` attribute (with ``latent_channels`` used for the pixel-space branch).
    """
    if isinstance(spec_or_type, str):
        vae_type = spec_or_type
        if not is_latent_vae(vae_type):
            return None
    else:
        if getattr(spec_or_type, "latent_channels", 0) == 0:
            return None
        vae_type = getattr(spec_or_type, "vae_type", None)
        if vae_type is None:
            raise ValueError(
                "load_vae requires a vae_type string or a spec with a 'vae_type' attribute"
            )
        if not is_latent_vae(vae_type):
            return None
    return load_minit2i_vae(vae_type, torch_dtype=torch_dtype, local_dir=local_dir)


def normalize(latent: torch.Tensor, vae, spec=None) -> torch.Tensor:
    """Spec-aware normalize wrapper (delegates to the frozen ``normalize_latent``)."""
    return normalize_latent(latent, vae)
