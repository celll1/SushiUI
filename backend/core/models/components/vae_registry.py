"""VAE component registry — arch-independent VAE component layer (plan A.4).

Generalizes ``core/models/minit2i/minit2i_vae.py``: it hosts the AutoencoderKL
loader surface (``VAE_REGISTRY``, ``VAE_SCALE_FACTOR``,
``is_latent_vae``, ``vae_latent_channels``, ``preview_decoder_for``,
``load_minit2i_vae``, ``normalize_latent`` / ``denormalize_latent``). The old
``core.models.minit2i.minit2i_vae`` module is a thin re-export shim of this one, so
existing imports keep resolving to the SAME objects (identity preserved).

BEHAVIOR FREEZE: the minit2i VAE functions are moved byte-identically (including
the ``[MiniT2I VAE]`` log strings and the ``"none"`` pixel-space sentinel semantics
via ``is_latent_vae``). ``VAE_REGISTRY`` is now a PROJECTION of the family table in
``core/models/common/vae_store.py`` rather than a second copy of it; the projected
membership and every field it exposes are unchanged. That module also stays the
home of the shared resolver, re-exported here for a unified component surface.

Spec-driven ``load_vae`` / ``encode`` / ``normalize`` are ADDITIVE (plan A.4): the
pixel-space branch is ``spec.latent_channels == 0`` (mirrors minit2i's
``vae_type=="none"`` / ``is_latent_vae`` pattern), not ``is_minit2i``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from core.models.common.vae_store import VAE_REGISTRY as _VAE_FAMILIES


def _generic_kl_entry(entry: dict) -> Optional[dict]:
    """Project a family-table entry onto this loader's view, or None if it can't
    load it: ``load_minit2i_vae`` hardcodes ``AutoencoderKL.from_pretrained``, so
    an entry needs both that class and a diffusers-layout default repo.
    """
    if entry.get("class") != "AutoencoderKL" or not entry.get("diffusers_repo"):
        return None
    return {
        "repo": entry["default_repo"],
        "subfolder": entry.get("default_subfolder"),
        "channels": entry["latent_channels"],
        "preview": entry.get("preview"),
    }


# vae_type -> (HF repo, subfolder, latent channels, preview-decoder kind).
# A PROJECTION of the family table in common/vae_store.py, not a second table
# (design doc VAE_SWAP_MIGRATION_DESIGN.md §7.1): sdxl + flux1, as before.
VAE_REGISTRY = {
    key: projected
    for key, projected in (
        (key, _generic_kl_entry(entry)) for key, entry in _VAE_FAMILIES.items()
    )
    if projected is not None
}

VAE_SCALE_FACTOR = 8  # spatial downsample of these AutoencoderKLs


def is_latent_vae(vae_type: Optional[str]) -> bool:
    return bool(vae_type) and vae_type != "none" and vae_type in VAE_REGISTRY


def vae_latent_channels(vae_type: str) -> int:
    return VAE_REGISTRY[vae_type]["channels"]


#: Linear latent->RGB projection tables `taesd.py` carries, by channel count.
#: The fallback for a VAE that names no family (a ``file:``/``model:`` source
#: resolves to family "custom"), where the channel count is all there is to go on.
_PROJECTION_PREVIEW_BY_CHANNELS = {16: "matrix16", 32: "matrix32"}


def preview_decoder_for(vae_type: Optional[str],
                        latent_channels: Optional[int] = None) -> str:
    """Live-preview decoder kind for a VAE family, or "" when none can decode it.

    Reads the family table rather than this module's loader projection, which
    drops the families it cannot ``from_pretrained`` (sd15, flux2, qwen_image)
    — those still have a preview decoder.
    """
    entry = _VAE_FAMILIES.get(vae_type or "")
    if entry is not None:
        return str(entry.get("preview") or "")
    if latent_channels is None:
        return ""
    return _PROJECTION_PREVIEW_BY_CHANNELS.get(int(latent_channels), "")


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
