"""Shared default-VAE registry + resolver.

Several architectures fall back to the same handful of default VAEs (SDXL 4ch,
SD1.5 4ch, FLUX.1 16ch, FLUX.2 32ch, Qwen-Image 16ch). Downloading and caching
each one once per arch wastes disk; this module resolves a *local directory* for
a given VAE type, using a shared on-disk store under
``<models_dir>/vae/<store_subdir>`` so the download is reused by every arch of
that type.

License note
------------
The FLUX.2 VAE MUST resolve from the Apache-2.0 ``black-forest-labs/FLUX.2-klein-4B``
repo (subfolder ``vae``) — NEVER from ``FLUX.2-klein-9B`` (FLUX Non-Commercial),
even when the local transformer checkpoint is a 9B variant. All registry default
repos below are redistributable (MIT / Apache-2.0).

Resolution precedence (``resolve_vae_dir``)
-------------------------------------------
  1. ``explicit`` argument (caller-supplied path)
  2. environment alias (``env_var`` name, e.g. ``KREA2_VAE_DIR``)
  3. the model's own ``vae/`` subfolder (``model_own_vae``)
  4. the shared store ``<models_dir>/vae/<store_subdir>`` (if already populated)
  5. Hugging Face Hub download INTO the shared store (so it is fetched once)

Never moves or deletes existing local files; the store is opportunistic.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


# vae_type -> registry entry.
#   class            : diffusers class name that loads this VAE (documentation only)
#   latent_channels  : latent channel count
#   store_subdir     : subdirectory under <models_dir>/vae/ for the shared store
#   default_repo     : HF repo id for the default weights (redistributable)
#   default_subfolder: subfolder within the repo holding the VAE (None = repo root)
#   license          : SPDX-ish license string of the default repo
VAE_REGISTRY: Dict[str, Dict] = {
    "sdxl": {
        "class": "AutoencoderKL",
        "latent_channels": 4,
        "store_subdir": "sdxl",
        "default_repo": "madebyollin/sdxl-vae-fp16-fix",
        "default_subfolder": None,
        "license": "MIT",
    },
    "sd15": {
        "class": "AutoencoderKL",
        "latent_channels": 4,
        "store_subdir": "sd15",
        "default_repo": "stabilityai/sd-vae-ft-mse-original",
        "default_subfolder": None,
        "license": "MIT",
    },
    "flux1": {
        "class": "AutoencoderKL",
        "latent_channels": 16,
        "store_subdir": "flux1",
        "default_repo": "diffusers/FLUX.1-vae",
        "default_subfolder": None,
        "license": "Apache-2.0",
    },
    "flux2": {
        "class": "AutoencoderKLFlux2",
        "latent_channels": 32,
        "store_subdir": "flux2",
        # MUST be the Apache-2.0 4B repo — NEVER FLUX.2-klein-9B (Non-Commercial).
        "default_repo": "black-forest-labs/FLUX.2-klein-4B",
        "default_subfolder": "vae",
        "license": "Apache-2.0",
    },
    "qwen_image": {
        "class": "AutoencoderKLQwenImage",
        "latent_channels": 16,
        "store_subdir": "qwen_image",
        "default_repo": "Qwen/Qwen-Image",
        "default_subfolder": "vae",
        "license": "Apache-2.0",
    },
}


def _models_dir() -> Optional[str]:
    try:
        from config.settings import settings
        return getattr(settings, "models_dir", None)
    except Exception:
        return None


def _has_vae_config(directory: Optional[str]) -> bool:
    return bool(directory) and os.path.isdir(directory) and os.path.isfile(
        os.path.join(directory, "config.json")
    )


def store_dir_for(vae_type: str) -> Optional[str]:
    """Return the shared-store inner directory for ``vae_type`` (no download).

    This is ``<models_dir>/vae/<store_subdir>[/<default_subfolder>]`` — the path a
    populated store would load from. Returns None when models_dir is unknown.
    """
    if vae_type not in VAE_REGISTRY:
        raise ValueError(f"Unknown vae_type '{vae_type}' (known: {list(VAE_REGISTRY)})")
    models_dir = _models_dir()
    if not models_dir:
        return None
    entry = VAE_REGISTRY[vae_type]
    store_root = os.path.join(models_dir, "vae", entry["store_subdir"])
    sub = entry.get("default_subfolder")
    return os.path.join(store_root, sub) if sub else store_root


def _download_into_store(vae_type: str) -> Optional[str]:
    """Download the default VAE for ``vae_type`` into the shared store; return dir."""
    entry = VAE_REGISTRY[vae_type]
    models_dir = _models_dir()
    if not models_dir:
        return None
    store_root = os.path.join(models_dir, "vae", entry["store_subdir"])
    sub = entry.get("default_subfolder")
    inner = os.path.join(store_root, sub) if sub else store_root

    if _has_vae_config(inner):
        return inner

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[VAEStore] huggingface_hub unavailable, cannot fetch {vae_type} VAE: {e}")
        return None

    allow = [f"{sub}/*"] if sub else ["*.json", "*.safetensors", "*.bin"]
    os.makedirs(store_root, exist_ok=True)
    print(f"[VAEStore] Downloading {vae_type} VAE ({entry['default_repo']}"
          + (f" subfolder={sub}" if sub else "")
          + f", {entry['license']}) into shared store: {store_root}")
    snapshot_download(entry["default_repo"], allow_patterns=allow, local_dir=store_root)
    return inner if _has_vae_config(inner) else None


def resolve_vae_dir(
    vae_type: str,
    explicit: Optional[str] = None,
    env_var: Optional[str] = None,
    model_own_vae: Optional[str] = None,
    download: bool = True,
) -> Optional[str]:
    """Resolve a local directory for the ``vae_type`` default VAE.

    See module docstring for the precedence order. Returns a directory loadable by
    ``<class>.from_pretrained(dir)`` (no ``subfolder`` needed), or None when nothing
    resolved and ``download`` is False (or the download failed / models_dir unknown).
    """
    if vae_type not in VAE_REGISTRY:
        raise ValueError(f"Unknown vae_type '{vae_type}' (known: {list(VAE_REGISTRY)})")

    # 1. explicit
    if _has_vae_config(explicit):
        return explicit

    # 2. environment alias
    if env_var:
        env_val = os.environ.get(env_var)
        if _has_vae_config(env_val):
            return env_val

    # 3. model's own vae/ subfolder
    if _has_vae_config(model_own_vae):
        return model_own_vae

    # 4. shared store (already populated)
    store_inner = store_dir_for(vae_type)
    if _has_vae_config(store_inner):
        return store_inner

    # 5. download into the shared store
    if download:
        return _download_into_store(vae_type)
    return None
