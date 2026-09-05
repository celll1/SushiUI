"""sushiUI single-file format (v2): shared writer / reader / dedup / sharding.

This module centralises the on-disk conventions that were previously duplicated
verbatim across ``minit2i`` and ``krea2`` ``vendor/single_file.py`` and the
sharded-component reader that lived in ``ideogram4_loader``.

Format summary
--------------
Key prefixes (arch-native diffusers-style layout underneath):
    ``transformer.``   main DiT/UNet-equivalent model
    ``text_encoder.``  optional bundled text encoder
    ``vae.``           reserved for a bundled VAE

Metadata (safetensors ``__metadata__``, all values ``str``):
    ``model_type``  arch string (single source for detection)
    ``format``      "pt"
    plus arch-specific config JSON, ``variant``, ``has_text_encoder``,
    ``tied_weights_dropped`` and optional ``component.*`` hints.

Sharding (threshold-switched, diffusers convention)
---------------------------------------------------
``save_single_file_state`` writes a single ``<stem>.safetensors`` when the total
(deduplicated) tensor byte size is ``<= max_shard_bytes`` (default 10 GB), else
greedy-splits into ``<stem>-00001-of-000NN.safetensors`` shards and writes a
``<stem>.safetensors.index.json`` of the form::

    {
      "metadata": {**our_string_metadata, "total_size": <int bytes>},
      "weight_map": {"<tensor key>": "<shard filename>", ...}
    }

``read_state_dict`` accepts either a ``<stem>.safetensors`` path or a
``<stem>.safetensors.index.json`` path and returns ``(state_dict, metadata)``.
For a sharded save the returned metadata is the index's ``"metadata"`` block.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSFORMER_PREFIX = "transformer."
TEXT_ENCODER_PREFIX = "text_encoder."
VAE_PREFIX = "vae."

# Threshold above which a save is split into diffusers-convention shards.
DEFAULT_MAX_SHARD_BYTES = 10 * 1024 ** 3  # 10 GB

_INDEX_SUFFIX = ".safetensors.index.json"
_SHARD_SUFFIX = ".safetensors"


def is_index_path(path: str) -> bool:
    """True for a ``<stem>.safetensors.index.json`` path."""
    return isinstance(path, str) and path.endswith(_INDEX_SUFFIX)


# ---------------------------------------------------------------------------
# Tied-weight dedup
# ---------------------------------------------------------------------------

def dedup_tensors(named_tensors: Iterable[Tuple[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Deduplicate tied tensors by storage pointer.

    safetensors rejects tensors that share memory (e.g. FLAN-T5 ties
    ``shared.weight`` with ``encoder.embed_tokens.weight``); the loader re-ties
    them. Returns ``(state, dropped_tied)`` where ``state`` maps each retained
    key to a detached CPU-contiguous tensor and ``dropped_tied`` lists the keys
    that aliased an already-seen tensor (recorded in metadata for transparency).
    """
    state: Dict[str, torch.Tensor] = {}
    seen_ptrs: Dict[int, str] = {}
    dropped_tied: List[str] = []
    for key, v in named_tensors:
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            dropped_tied.append(key)  # tied to seen_ptrs[ptr]; re-tied on load
            continue
        seen_ptrs[ptr] = key
        state[key] = v.detach().to("cpu").contiguous()
    return state, dropped_tied


# ---------------------------------------------------------------------------
# Prefix helpers
# ---------------------------------------------------------------------------

def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Return the sub-dict of keys starting with ``prefix``, prefix stripped."""
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def split_prefixed_state_dict(
    raw: Dict[str, torch.Tensor], prefixes: Iterable[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split ``raw`` by the given key prefixes.

    Returns ``{prefix: {stripped_key: tensor}}`` for each prefix, plus a ``""``
    bucket for keys matching no prefix. Prefixes are tested in the given order;
    the first match wins.
    """
    prefixes = list(prefixes)
    out: Dict[str, Dict[str, torch.Tensor]] = {p: {} for p in prefixes}
    out[""] = {}
    for k, v in raw.items():
        for p in prefixes:
            if p and k.startswith(p):
                out[p][k[len(p):]] = v
                break
        else:
            out[""][k] = v
    return out


# ---------------------------------------------------------------------------
# component.* metadata hints (declarative; unknown keys ignored on read)
# ---------------------------------------------------------------------------

def build_component_metadata(
    *,
    vae_type: Optional[str] = None,
    vae_channels=None,
    vae_embedded: Optional[bool] = None,
    vae_prefix: Optional[str] = None,
    vae_class: Optional[str] = None,
    vae_config=None,
    vae_scale_factor=None,
    vae_scale_temporal=None,
    vae_norm: Optional[str] = None,
    vae_norm_pack=None,
    vae_provenance: Optional[str] = None,
    vae_locator: Optional[str] = None,
    vae_hash: Optional[str] = None,
    vae_struct_native: Optional[bool] = None,
    vae_identity_native: Optional[bool] = None,
    te_type: Optional[str] = None,
    te_dim=None,
    te_embedded: Optional[bool] = None,
    te_adapter: Optional[bool] = None,
) -> Dict[str, str]:
    """Build the optional ``component.*`` metadata hint block (all str values).

    The VAE block describes the latent space the checkpoint was trained in — see
    ``docs/guides/VAE_SWAP_MIGRATION_DESIGN.md`` §5.2 for each key's meaning.
    ``struct_native`` (channels/scale/ndim/class equal the arch default) and
    ``identity_native`` (same VAE weights as the arch default) are separate
    questions and are never interchangeable: a fine-tuned copy of the native VAE
    is ``struct_native=True, identity_native=False``.

    ``vae_provenance`` is display-only; ``vae_locator`` (``registry:<key>`` or
    ``path:<abs>``) is what a reader resolves a non-embedded VAE through, and is
    only meaningful together with ``vae_hash``.
    """
    if vae_struct_native is False and vae_identity_native is True:
        raise ValueError(
            "component.vae: identity_native=True requires struct_native=True "
            "(a structurally different VAE cannot be the same latent space)")
    md: Dict[str, str] = {}
    if vae_type is not None:
        md["component.vae.type"] = str(vae_type)
    if vae_channels is not None:
        md["component.vae.channels"] = str(vae_channels)
    if vae_embedded is not None:
        md["component.vae.embedded"] = "1" if vae_embedded else "0"
    if vae_prefix:
        md["component.vae.prefix"] = str(vae_prefix)
    if vae_class:
        md["component.vae.class"] = str(vae_class)
    if vae_config is not None:
        md["component.vae.config"] = (
            vae_config if isinstance(vae_config, str)
            else json.dumps(vae_config, sort_keys=True))
    if vae_scale_factor is not None:
        md["component.vae.scale_factor"] = str(int(vae_scale_factor))
    if vae_scale_temporal is not None:
        md["component.vae.scale_temporal"] = str(int(vae_scale_temporal))
    if vae_norm:
        md["component.vae.norm"] = str(vae_norm)
    if vae_norm_pack is not None:
        md["component.vae.norm_pack"] = str(int(vae_norm_pack))
    if vae_provenance:
        md["component.vae.provenance"] = str(vae_provenance)
    if vae_locator:
        md["component.vae.locator"] = str(vae_locator)
    if vae_hash:
        md["component.vae.hash"] = str(vae_hash)
    if vae_struct_native is not None:
        md["component.vae.struct_native"] = "1" if vae_struct_native else "0"
    if vae_identity_native is not None:
        md["component.vae.identity_native"] = "1" if vae_identity_native else "0"
    if te_type is not None:
        md["component.te.type"] = str(te_type)
    if te_dim is not None:
        md["component.te.dim"] = str(te_dim)
    if te_embedded is not None:
        md["component.te.embedded"] = "1" if te_embedded else "0"
    if te_adapter is not None:
        md["component.te_adapter"] = "1" if te_adapter else "0"
    return md


def parse_component_metadata(metadata: Optional[dict]) -> Dict[str, dict]:
    """Parse the ``component.*`` hints back into a nested dict.

    Only keys present are returned. Unknown ``component.*`` keys are ignored.
    """
    metadata = metadata or {}
    out: Dict[str, dict] = {}
    vae: Dict[str, Any] = {}
    if "component.vae.type" in metadata:
        vae["type"] = metadata["component.vae.type"]
    if "component.vae.channels" in metadata:
        vae["channels"] = metadata["component.vae.channels"]
    if "component.vae.embedded" in metadata:
        vae["embedded"] = str(metadata["component.vae.embedded"]).strip().lower() in ("1", "true", "yes")
    for key in ("prefix", "class", "norm", "provenance", "locator", "hash"):
        raw = metadata.get(f"component.vae.{key}")
        if raw:
            vae[key] = str(raw)
    for key in ("scale_factor", "scale_temporal", "norm_pack"):
        if f"component.vae.{key}" in metadata:
            try:
                vae[key] = int(str(metadata[f"component.vae.{key}"]).strip())
            except (TypeError, ValueError):
                pass
    if "component.vae.config" in metadata:
        try:
            vae["config"] = json.loads(metadata["component.vae.config"])
        except (TypeError, ValueError):
            pass
    for key in ("struct_native", "identity_native"):
        if f"component.vae.{key}" in metadata:
            vae[key] = str(metadata[f"component.vae.{key}"]).strip().lower() in ("1", "true", "yes")
    # A file may declare the impossible pair; the structural answer wins so a
    # reader can never treat a differently-shaped VAE as the same latent space.
    if vae.get("struct_native") is False:
        vae["identity_native"] = False
    if vae:
        out["vae"] = vae
    te: Dict[str, str] = {}
    if "component.te.type" in metadata:
        te["type"] = metadata["component.te.type"]
    if "component.te.dim" in metadata:
        te["dim"] = metadata["component.te.dim"]
    if "component.te.embedded" in metadata:
        te["embedded"] = str(metadata["component.te.embedded"]).strip().lower() in ("1", "true", "yes")
    if te:
        out["te"] = te
    if "component.te_adapter" in metadata:
        out["te_adapter"] = str(metadata["component.te_adapter"]).strip().lower() in ("1", "true", "yes")
    return out


# ---------------------------------------------------------------------------
# Save (single file or diffusers-convention shards)
# ---------------------------------------------------------------------------

def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _plan_shards(state: Dict[str, torch.Tensor], max_shard_bytes: int) -> List[List[str]]:
    """Greedy split keys into shards each ``<= max_shard_bytes`` where possible.

    A single tensor larger than ``max_shard_bytes`` occupies its own shard.
    Preserves ``state`` iteration order for deterministic shard assignment.
    """
    shards: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for key, v in state.items():
        nbytes = _tensor_nbytes(v)
        if current and current_bytes + nbytes > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def save_single_file_state(
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, str],
    path: str,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> str:
    """Write ``state_dict`` + ``metadata`` to ``path``, sharding above threshold.

    ``state_dict`` must already be deduplicated (see ``dedup_tensors``) and
    CPU-contiguous. ``metadata`` values must all be ``str``. ``path`` is the
    ``<stem>.safetensors`` target.

    Returns the path that was written: ``path`` itself for a single file, or the
    ``<stem>.safetensors.index.json`` path for a sharded save.
    """
    metadata = {k: str(v) for k, v in (metadata or {}).items()}

    total_size = sum(_tensor_nbytes(v) for v in state_dict.values())
    if total_size <= max_shard_bytes:
        save_file(state_dict, path, metadata=metadata)
        return path

    # Sharded write (diffusers convention).
    if path.endswith(_SHARD_SUFFIX):
        stem = os.path.basename(path[: -len(_SHARD_SUFFIX)])
    else:
        stem = os.path.basename(path)
    directory = os.path.dirname(path)

    shards = _plan_shards(state_dict, max_shard_bytes)
    n = len(shards)
    weight_map: Dict[str, str] = {}
    for i, keys in enumerate(shards, start=1):
        shard_name = f"{stem}-{i:05d}-of-{n:05d}.safetensors"
        shard_state = {k: state_dict[k] for k in keys}
        # Embed the same metadata in each shard so tensors stay self-describing.
        save_file(shard_state, os.path.join(directory, shard_name), metadata=metadata)
        for k in keys:
            weight_map[k] = shard_name

    index = {
        "metadata": {**metadata, "total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(directory, f"{stem}{_INDEX_SUFFIX}")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index_path


# ---------------------------------------------------------------------------
# Read (single file or shard index)
# ---------------------------------------------------------------------------

def read_state_dict(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Read a sushiUI single file or a shard index into ``(state_dict, metadata)``.

    Accepts:
      * ``<stem>.safetensors``          — single file
      * ``<stem>.safetensors.index.json`` — sharded (loads all shards)

    For a sharded save, ``metadata`` is the index's ``"metadata"`` block (which
    also carries ``total_size``). If a ``.safetensors`` path is given but only a
    sibling shard index exists, the index is used.
    """
    if is_index_path(path):
        return _read_index(path)

    if path.endswith(_SHARD_SUFFIX) and not os.path.exists(path):
        sibling_index = path[: -len(_SHARD_SUFFIX)] + _INDEX_SUFFIX
        if os.path.exists(sibling_index):
            return _read_index(sibling_index)

    raw: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    return raw, metadata


def _read_index(index_path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    directory = os.path.dirname(index_path)
    weight_map = index.get("weight_map", {}) or {}
    metadata = dict(index.get("metadata", {}) or {})
    shard_files = sorted(set(weight_map.values()))
    state_dict: Dict[str, torch.Tensor] = {}
    for shard in shard_files:
        state_dict.update(load_file(os.path.join(directory, shard)))
    return state_dict, metadata


def load_component_state_dict(component_dir: str, basename: str) -> Dict[str, torch.Tensor]:
    """Load a diffusers component's weights, whether sharded (index) or single file.

    Promoted from ``ideogram4_loader`` (kept as a re-export there for
    compatibility). Returns the assembled state dict only (no metadata), matching
    diffusers' per-component layout ``<basename>.safetensors`` /
    ``<basename>.safetensors.index.json`` inside ``component_dir``.
    """
    index_path = os.path.join(component_dir, f"{basename}{_INDEX_SUFFIX}")
    single_path = os.path.join(component_dir, f"{basename}{_SHARD_SUFFIX}")

    if os.path.exists(index_path):
        state_dict, _ = _read_index(index_path)
        return state_dict

    if os.path.exists(single_path):
        return load_file(single_path)

    raise FileNotFoundError(f"No safetensors found for '{basename}' in {component_dir}")


# ---------------------------------------------------------------------------
# Embedded-weight reattach (zero-match-raises)
# ---------------------------------------------------------------------------

def reattach_embedded_weights(module, state_dict: Dict[str, torch.Tensor], label: str) -> None:
    """Load embedded (trained) weights into a freshly built base component.

    Mirrors ``ModelLoader._reattach_embedded_weights`` but lives here so loaders
    that cannot import ``ModelLoader`` (circular) can reuse it. Raises when NOTHING
    matched — silently keeping the untrained base weights would reintroduce a lossy
    roundtrip and hide a key-layout mismatch.
    """
    print(f"[SingleFile] Reattaching embedded {label} weights ({len(state_dict)} tensors)")
    info = module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(info, "missing_keys", []) or [])
    unexpected = list(getattr(info, "unexpected_keys", []) or [])
    matched = len(state_dict) - len(unexpected)
    if missing:
        print(f"[SingleFile]   embedded {label} missing: {len(missing)}")
    if unexpected:
        print(f"[SingleFile]   embedded {label} unexpected: {len(unexpected)}")
    if matched <= 0:
        raise RuntimeError(
            f"Embedded {label} weights in the checkpoint did not match the base "
            f"{label} at all ({len(state_dict)} tensors, 0 matched). The checkpoint's "
            f"{label} section uses an incompatible key layout; refusing to silently "
            f"fall back to the untrained base {label}."
        )
