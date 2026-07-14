"""Shared default text-encoder registry + resolver.

Mirrors ``vae_store.py``'s resolution precedence for the (currently single) case
of an architecture that needs an optional, on-demand text encoder it does not
ship with its own checkpoint: PiD's opt-in runtime Gemma-2-2b-it captioner
(``pid_use_gemma``). No gated-approval machinery is needed here — PiD was
trained against the UNGATED ``Efficient-Large-Model/gemma-2-2b-it`` mirror (byte
-identical embeddings to ``google/gemma-2-2b-it``, but redistributable without a
license-acceptance gate), so a download failure just warns and the caller falls
back to the null-caption default.

Resolution precedence (``resolve_te_dir``) — identical shape to
``vae_store.resolve_vae_dir``:
  1. ``explicit`` argument (caller-supplied path)
  2. environment alias (``env_var`` name)
  3. the model's own ``te/`` subfolder (``model_own_te``)
  4. the existing Hugging Face hub cache (offline probe)
  5. the shared store ``<models_dir>/text_encoders/<store_subdir>`` (if populated)
  6. Hugging Face Hub download INTO the shared store (so it is fetched once)

Never moves or deletes existing local files; the store is opportunistic.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


# te_type -> registry entry.
#   class            : transformers class name that loads this TE (documentation only)
#   te_type          : canonical identifier (matches the dict key; used by callers/compat checks)
#   out_dim          : hidden size of the encoder output (documentation + compat checks)
#   store_subdir     : subdirectory under <models_dir>/text_encoders/ for the shared store
#   default_repo     : HF repo id for the default weights
#   license          : SPDX-ish / human license string of the default repo
TE_REGISTRY: Dict[str, Dict] = {
    "gemma-2-2b-it": {
        "class": "Gemma2Model",
        "te_type": "gemma-2-2b-it",
        "out_dim": 2304,
        "store_subdir": "gemma-2-2b-it",
        # UNGATED mirror PiD was actually trained against (NOT google/gemma-2-2b-it,
        # which is gated) — same weights, byte-identical embeddings, redistributable.
        "default_repo": "Efficient-Large-Model/gemma-2-2b-it",
        "license": "Gemma Terms",
    },
}


def _models_dir() -> Optional[str]:
    try:
        from config.settings import settings
        return getattr(settings, "models_dir", None)
    except Exception:
        return None


def _has_te_config(directory: Optional[str]) -> bool:
    return bool(directory) and os.path.isdir(directory) and os.path.isfile(
        os.path.join(directory, "config.json")
    )


def _has_te_dir(directory: Optional[str]) -> bool:
    """True when ``directory`` holds a loadable text encoder (config + weights)."""
    if not _has_te_config(directory):
        return False
    return any(
        name.endswith((".safetensors", ".bin"))
        for name in os.listdir(directory)
    )


def store_dir_for(te_type: str) -> Optional[str]:
    """Return the shared-store inner directory for ``te_type`` (no download)."""
    if te_type not in TE_REGISTRY:
        raise ValueError(f"Unknown te_type '{te_type}' (known: {list(TE_REGISTRY)})")
    models_dir = _models_dir()
    if not models_dir:
        return None
    entry = TE_REGISTRY[te_type]
    return os.path.join(models_dir, "text_encoders", entry["store_subdir"])


_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.model"]


def _cache_repo_id_candidates(repo_id: str) -> list:
    """Cache-folder repo-id candidates for ``repo_id``, case-insensitively.

    See ``vae_store._cache_repo_id_candidates`` — same rationale (HF resolves
    repo ids case-insensitively server-side, but the hub cache folder name
    follows the string the ORIGINAL download used).
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


def _probe_hf_cache(te_type: str) -> Optional[str]:
    """Return the default TE dir from the existing HF hub cache, or None.

    Offline probe only (``local_files_only=True``) — never downloads.
    """
    entry = TE_REGISTRY[te_type]
    for repo_id in _cache_repo_id_candidates(entry["default_repo"]):
        try:
            from huggingface_hub import snapshot_download
            snapshot_root = snapshot_download(
                repo_id,
                allow_patterns=_ALLOW_PATTERNS,
                local_files_only=True,
            )
        except Exception:
            continue
        if _has_te_dir(snapshot_root):
            print(f"[TEStore] Reusing {te_type} text encoder from HF hub cache: {snapshot_root}")
            return snapshot_root
    inner = _scan_cached_snapshots(te_type)
    if inner:
        print(f"[TEStore] Reusing {te_type} text encoder from HF hub cache snapshot: {inner}")
    return inner


def _scan_cached_snapshots(te_type: str) -> Optional[str]:
    """Scan every cached snapshot of the default repo for a loadable TE dir."""
    entry = TE_REGISTRY[te_type]
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
            inner = os.path.join(snap_root, snap)
            if _has_te_dir(inner):
                return inner
    return None


def _download_into_store(te_type: str) -> Optional[str]:
    """Download the default TE for ``te_type`` into the shared store; return dir.

    On any failure (network, no models_dir, huggingface_hub unavailable, gated
    approval — not expected for this ungated entry but handled defensively
    anyway) this just warns and returns None; callers fall back to the
    null-caption default rather than crashing generation.
    """
    entry = TE_REGISTRY[te_type]
    models_dir = _models_dir()
    if not models_dir:
        print(f"[TEStore] models_dir unknown, cannot fetch {te_type} text encoder")
        return None
    store_root = os.path.join(models_dir, "text_encoders", entry["store_subdir"])

    if _has_te_dir(store_root):
        return store_root

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[TEStore] huggingface_hub unavailable, cannot fetch {te_type} text encoder: {e}")
        return None

    os.makedirs(store_root, exist_ok=True)
    print(f"[TEStore] Downloading {te_type} text encoder ({entry['default_repo']}, "
          f"{entry['license']}) into shared store: {store_root}")
    try:
        snapshot_download(entry["default_repo"], allow_patterns=_ALLOW_PATTERNS, local_dir=store_root)
    except Exception as e:
        print(f"[TEStore] Download failed for {te_type} text encoder: {e}. Falling back to null-caption.")
        return None
    return store_root if _has_te_dir(store_root) else None


def resolve_te_dir(
    te_type: str,
    explicit: Optional[str] = None,
    env_var: Optional[str] = None,
    model_own_te: Optional[str] = None,
    download: bool = True,
) -> Optional[str]:
    """Resolve a local directory for the ``te_type`` default text encoder.

    See module docstring for the precedence order. Returns a directory loadable
    by ``AutoModelForCausalLM.from_pretrained(dir)`` /
    ``AutoTokenizer.from_pretrained(dir)``, or None when nothing resolved (or
    ``download`` is False, or the download failed / models_dir unknown). Never
    raises — on any resolution failure the caller should fall back to a
    precomputed default (e.g. PiD's null-caption asset) rather than crash.
    """
    if te_type not in TE_REGISTRY:
        raise ValueError(f"Unknown te_type '{te_type}' (known: {list(TE_REGISTRY)})")

    # 1. explicit
    if _has_te_dir(explicit):
        return explicit

    # 2. environment alias
    if env_var:
        env_val = os.environ.get(env_var)
        if _has_te_dir(env_val):
            return env_val

    # 3. model's own te/ subfolder
    if _has_te_dir(model_own_te):
        return model_own_te

    # 4. existing HF hub cache (offline probe)
    cached = _probe_hf_cache(te_type)
    if cached:
        return cached

    # 5. shared store (already populated) — require config + weights so a partial
    #    download (config.json only) doesn't permanently short-circuit the retry.
    store_inner = store_dir_for(te_type)
    if _has_te_dir(store_inner):
        return store_inner

    # 6. download into the shared store
    if download:
        return _download_into_store(te_type)
    return None
