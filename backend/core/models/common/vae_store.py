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
  4. the existing Hugging Face hub cache (offline probe — reuses VAEs already
     downloaded by pre-store code instead of fetching a second copy)
  5. the shared store ``<models_dir>/vae/<store_subdir>`` (if already populated)
  6. Hugging Face Hub download INTO the shared store (so it is fetched once)

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


def _allow_patterns(entry: Dict) -> list:
    sub = entry.get("default_subfolder")
    return [f"{sub}/*"] if sub else ["*.json", "*.safetensors", "*.bin"]


def _probe_hf_cache(vae_type: str) -> Optional[str]:
    """Return the default VAE dir from the existing HF hub cache, or None.

    Offline probe only (``local_files_only=True``) — never downloads. Reuses
    copies fetched by pre-store code so the store does not duplicate them.
    """
    entry = VAE_REGISTRY[vae_type]
    for repo_id in _cache_repo_id_candidates(entry["default_repo"]):
        try:
            from huggingface_hub import snapshot_download
            snapshot_root = snapshot_download(
                repo_id,
                allow_patterns=_allow_patterns(entry),
                local_files_only=True,
            )
        except Exception:
            continue
        sub = entry.get("default_subfolder")
        inner = os.path.join(snapshot_root, sub) if sub else snapshot_root
        if _has_vae_dir(inner):
            print(f"[VAEStore] Reusing {vae_type} VAE from HF hub cache: {inner}")
            return inner
    # The main ref may point to a snapshot without the VAE files (partial
    # downloads leave older snapshots behind) — scan all cached snapshots.
    inner = _scan_cached_snapshots(vae_type)
    if inner:
        print(f"[VAEStore] Reusing {vae_type} VAE from HF hub cache snapshot: {inner}")
    return inner


def _has_vae_dir(directory: Optional[str]) -> bool:
    """True when ``directory`` holds a loadable VAE (config + weights)."""
    if not _has_vae_config(directory):
        return False
    return any(
        name.endswith((".safetensors", ".bin"))
        for name in os.listdir(directory)
    )


def _scan_cached_snapshots(vae_type: str) -> Optional[str]:
    """Scan every cached snapshot of the default repo for a loadable VAE dir."""
    entry = VAE_REGISTRY[vae_type]
    sub = entry.get("default_subfolder")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    for repo_id in _cache_repo_id_candidates(entry["default_repo"]):
        snap_root = os.path.join(
            HF_HUB_CACHE, f"models--{repo_id.replace('/', '--')}", "snapshots"
        )
        if not os.path.isdir(snap_root):
            continue
        for snap in sorted(os.listdir(snap_root)):
            inner = os.path.join(snap_root, snap, sub) if sub else os.path.join(snap_root, snap)
            if _has_vae_dir(inner):
                return inner
    return None


def _cache_repo_id_candidates(repo_id: str) -> list:
    """Cache-folder repo-id candidates for ``repo_id``, case-insensitively.

    HF resolves repo ids case-insensitively server-side, but the hub cache
    folder name follows the string the ORIGINAL download used — e.g. a cache
    populated via ``FLUX.2-Klein-4B`` is missed by an offline probe for
    ``FLUX.2-klein-4B``. Scan the cache dir for folders matching the id
    case-insensitively and probe each spelling found.
    """
    candidates = [repo_id]
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        want = f"models--{repo_id.replace('/', '--')}".lower()
        for name in os.listdir(HF_HUB_CACHE):
            if name.lower() == want and name != f"models--{repo_id.replace('/', '--')}":
                candidates.append(name[len("models--"):].replace("--", "/"))
    except Exception:
        pass
    return candidates


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

    allow = _allow_patterns(entry)
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

    # 4. existing HF hub cache (offline probe; avoids re-downloading VAEs
    #    already fetched by pre-store code)
    cached = _probe_hf_cache(vae_type)
    if cached:
        return cached

    # 5. shared store (already populated)
    store_inner = store_dir_for(vae_type)
    if _has_vae_config(store_inner):
        return store_inner

    # 6. download into the shared store
    if download:
        return _download_into_store(vae_type)
    return None
