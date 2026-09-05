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


# vae_type -> registry entry. THE VAE FAMILY TABLE: the component-side registry
# (``models/components/vae_registry.py``) is a projection of this one, not a
# second table (design doc VAE_SWAP_MIGRATION_DESIGN.md §7.1).
#   class            : diffusers class name that loads this VAE
#   latent_channels  : latent channel count
#   latent_ndim      : rank of the latent tensor this VAE produces (4 = [B,C,H,W]
#                      2D conv stack; 5 = [B,C,T,H,W] causal 3D stack)
#   scale_factor     : spatial compression ratio
#   scale_temporal   : temporal compression ratio (1 for a 2D image VAE)
#   norm / norm_pack : how latents are normalised, and the spatial pack factor of
#                      the domain the statistics are defined on (§5.2)
#   store_subdir     : subdirectory under <models_dir>/vae/ for the shared store
#   default_repo     : HF repo id for the default weights (redistributable)
#   default_subfolder: subfolder within the repo holding the VAE (None = repo root)
#   diffusers_repo   : default_repo is a diffusers-layout directory (config.json +
#                      weights). False = original/LDM single-file release, which
#                      `from_pretrained` cannot open.
#   preview          : live-preview decoder kind for this latent space
#                      ("taesd"/"taesdxl"/"taef1" tiny autoencoders, "matrix16"/
#                      "matrix32" linear latent->RGB projections), or None when
#                      nothing in core/utils/taesd.py can decode it
#   license          : SPDX-ish license string of the default repo
#   scaling_factor   : the family's canonical latent scaling factor, or None when
#                      the family does not have a single scalar one (see below)
#   shift_factor     : the family's canonical latent shift, or None for "absent"
#
# The structural fields are read off the diffusers classes: AutoencoderKL and
# AutoencoderKLFlux2 are 2D stacks whose spatial ratio is
# 2 ** (len(block_out_channels) - 1) = 8 (the same expression the diffusers
# pipelines compute their own vae_scale_factor with); AutoencoderKLQwenImage is a
# causal 3D stack with spatial 2 ** len(temperal_downsample) = 8 and one temporal
# halving per True in it = 4. AutoencoderKLFlux2's BatchNorm is declared over
# prod(patch_size) * latent_channels = 4 * 32 channels, i.e. on the 2x2-packed
# domain -> norm_pack 2.
#
# scaling_factor / shift_factor are the values in the default repo's own
# config.json, verified against it. They exist here so that a VAE loaded WITHOUT
# a config.json (a bare `.safetensors`) can be given the right number instead of
# diffusers' single-file fallback: `AutoencoderKL.from_single_file` cannot tell
# an SDXL VAE from an SD1.5 one (the architectures are identical) and falls back
# to LDM_VAE_DEFAULT_SCALING_FACTOR = 0.18215, which is a 1.40x error on SDXL.
# This table is the ONLY place those numbers are written down.
#
# `None` means "this family has no single scalar scaling factor": AutoencoderKLFlux2
# and AutoencoderKLQwenImage normalise with per-channel latents_mean/latents_std
# and their config.json carries no scaling_factor at all. None must therefore be
# read as "cannot be determined from the architecture alone", never as 1.0.
VAE_REGISTRY: Dict[str, Dict] = {
    "sdxl": {
        "class": "AutoencoderKL",
        "latent_channels": 4,
        "store_subdir": "sdxl",
        "default_repo": "madebyollin/sdxl-vae-fp16-fix",
        "default_subfolder": None,
        "license": "MIT",
        "scaling_factor": 0.13025,
        "shift_factor": None,
        "latent_ndim": 4,
        "scale_factor": 8,
        "scale_temporal": 1,
        "norm": "shift_scale",
        "norm_pack": 1,
        "diffusers_repo": True,
        "preview": "taesdxl",   # taesd.py is_sdxl path
    },
    "sd15": {
        "class": "AutoencoderKL",
        "latent_channels": 4,
        "store_subdir": "sd15",
        "default_repo": "stabilityai/sd-vae-ft-mse-original",
        "default_subfolder": None,
        "license": "MIT",
        "scaling_factor": 0.18215,
        "shift_factor": None,
        "latent_ndim": 4,
        "scale_factor": 8,
        "scale_temporal": 1,
        "norm": "shift_scale",
        "norm_pack": 1,
        # Original/LDM single-file release (a bare .safetensors, no config.json).
        "diffusers_repo": False,
        "preview": "taesd",     # taesd.py generic (SD1.5) path
    },
    "flux1": {
        "class": "AutoencoderKL",
        "latent_channels": 16,
        "store_subdir": "flux1",
        "default_repo": "diffusers/FLUX.1-vae",
        "default_subfolder": None,
        "license": "Apache-2.0",
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "latent_ndim": 4,
        "scale_factor": 8,
        "scale_temporal": 1,
        "norm": "shift_scale",
        "norm_pack": 1,
        "diffusers_repo": True,
        "preview": "taef1",     # taesd.py is_zimage (FLUX VAE) path
    },
    "flux2": {
        "class": "AutoencoderKLFlux2",
        "latent_channels": 32,
        "store_subdir": "flux2",
        # MUST be the Apache-2.0 4B repo — NEVER FLUX.2-klein-9B (Non-Commercial).
        "default_repo": "black-forest-labs/FLUX.2-klein-4B",
        "default_subfolder": "vae",
        "license": "Apache-2.0",
        "scaling_factor": None,   # latents_mean / latents_std, no scalar
        "shift_factor": None,
        "latent_ndim": 4,
        "scale_factor": 8,
        "scale_temporal": 1,
        "norm": "batchnorm",
        "norm_pack": 2,
        "diffusers_repo": True,
        "preview": "matrix32",  # taesd.py FLUX.2 32ch latent->RGB projection
    },
    "qwen_image": {
        "class": "AutoencoderKLQwenImage",
        "latent_channels": 16,
        "store_subdir": "qwen_image",
        "default_repo": "Qwen/Qwen-Image",
        "default_subfolder": "vae",
        "license": "Apache-2.0",
        "scaling_factor": None,   # latents_mean / latents_std, no scalar
        "shift_factor": None,
        "latent_ndim": 5,
        "scale_factor": 8,
        "scale_temporal": 4,
        "norm": "per_channel",
        "norm_pack": 1,
        "diffusers_repo": True,
        "preview": "matrix16",  # taesd.py Wan21 16ch latent->RGB projection
    },
}

# What `AutoencoderKL.from_single_file` falls back to when the file it is given
# carries no architectural evidence (diffusers
# `loaders/single_file_utils.LDM_VAE_DEFAULT_SCALING_FACTOR`). Named here so the
# "this value is a guess, not a measurement" test reads as such at its call site.
LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR = 0.18215


def canonical_latent_scaling(vae_type: str):
    """Return ``(scaling_factor, shift_factor, latent_channels)`` for ``vae_type``.

    ``scaling_factor`` is None when the family has no single scalar one, and the
    whole tuple is None when ``vae_type`` is not a known registry key. Callers
    must treat both as "cannot determine" and leave whatever they loaded alone
    rather than substituting a number.
    """
    entry = VAE_REGISTRY.get(vae_type)
    if entry is None:
        return None
    return (entry.get("scaling_factor"), entry.get("shift_factor"),
            entry.get("latent_channels"))


def vae_identity(vae, embedded: bool = False, pixel_space: bool = False) -> tuple:
    """Return ``(vae_source, vae_path)`` describing the VAE used at decode time.

    This is a lightweight, string-only reporter for generation metadata — it never
    downloads or hashes. ``vae_source`` is a human-readable description (a repo id,
    a resolved directory, ``"embedded (checkpoint)"`` or ``"none (pixel-space)"``);
    ``vae_path`` is a concrete local directory/file the caller may hash (or None).

    - ``pixel_space=True`` -> ("none (pixel-space)", None); no VAE participates.
    - ``embedded=True`` -> the VAE weights were bundled in / extracted from the base
      checkpoint, whose hash is already recorded, so no separate path is reported.
    - otherwise the effective source is read from ``vae.config._name_or_path``
      (set by ``from_pretrained``); a local path there is also returned as
      ``vae_path`` so a concrete weight hash can be computed.
    """
    if pixel_space or vae is None:
        return "none (pixel-space)", None
    if embedded:
        return "embedded (checkpoint)", None
    src = ""
    cfg = getattr(vae, "config", None)
    if cfg is not None:
        try:
            src = getattr(cfg, "_name_or_path", "") or ""
        except Exception:
            src = ""
    path = src if (src and os.path.exists(src)) else None
    if not src:
        # No recorded source (rebuilt module); fall back to the class name.
        src = type(vae).__name__
    return src, path


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
