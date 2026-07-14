"""Model Component Registry (RP1).

Persistent, per-model registry of components + their I/O dims, produced by CHEAP
header/config reads only (NEVER loading weights, NEVER open()'ing a directory).
Keyed by a dir-aware content hash so a backend restart does not force a rescan.

See the shared contract in ``model_registry_spec.md`` for the record schema. This
module PRODUCES ``ModelComponentRecord`` dicts; RP2-RP4 consume them.

Design notes:
  * Extraction is best-effort and defensive: any per-component failure yields that
    component's dims as ``null`` (never raises). A model that fails entirely yields
    a minimal record ``{arch, content_hash, ...}`` rather than an exception.
  * The persistent cache mirrors ``backend/utils/hash_cache.py``: a JSON file under
    ``settings.cache_dir`` mapping ``path -> {content_hash, size, mtime, record}``.
    Invalidation is by a cheap (size, mtime) stat aggregate — dir-aware — so a cache
    hit performs NO header reads.
  * Expected baselines come from the per-arch ``ComponentWiringSpec`` (wiring.py);
    OBSERVED dims are stored, and mismatches are flagged in ``record["mismatches"]``.
"""

from __future__ import annotations

import json
import os
import struct
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Wiring baselines (expected dims per arch)
# ---------------------------------------------------------------------------
try:
    from core.models.components.wiring import (
        ComponentWiringSpec,
        SD15_WIRING, SDXL_WIRING, ZIMAGE_WIRING, ANIMA_WIRING, LENS_WIRING,
        IDEOGRAM4_WIRING, MINIT2I_WIRING, KREA2_WIRING, FLUX2_WIRING, LTX2_WIRING,
    )
    _WIRING_BY_ARCH: Dict[str, ComponentWiringSpec] = {
        "sd15": SD15_WIRING,
        "sdxl": SDXL_WIRING,
        "zimage": ZIMAGE_WIRING,
        "anima": ANIMA_WIRING,
        "lens": LENS_WIRING,
        "ideogram4": IDEOGRAM4_WIRING,
        "minit2i": MINIT2I_WIRING,
        "krea2": KREA2_WIRING,
        "flux2": FLUX2_WIRING,
        "ltx2": LTX2_WIRING,
    }
except Exception as _e:  # pragma: no cover - wiring is a hard dependency
    _WIRING_BY_ARCH = {}
    print(f"[ComponentRegistry] WARNING: could not import wiring specs: {_e}")


# archs whose backbone is a U-Net; everything else is a transformer/DiT
_UNET_ARCHS = {"sd15", "sdxl"}


# ---------------------------------------------------------------------------
# Cheap safetensors header reader (header bytes only, no tensor data)
# ---------------------------------------------------------------------------

def _read_safetensors_header(path: str) -> Dict[str, Any]:
    """Read ONLY the JSON header of a .safetensors file.

    The safetensors format is: 8-byte little-endian u64 header length, then that
    many bytes of JSON. The JSON maps ``tensor_name -> {dtype, shape, data_offsets}``
    plus an optional ``__metadata__`` string dict. This reads at most a few hundred
    KB regardless of file size — no tensor data is touched.
    """
    with open(path, "rb") as f:
        n_bytes = f.read(8)
        if len(n_bytes) < 8:
            raise ValueError("file too small to be safetensors")
        (header_len,) = struct.unpack("<Q", n_bytes)
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length {header_len}")
        raw = f.read(header_len)
    return json.loads(raw.decode("utf-8"))


def _header_shape(header: Dict[str, Any], key: str) -> Optional[List[int]]:
    entry = header.get(key)
    if isinstance(entry, dict):
        shp = entry.get("shape")
        if isinstance(shp, list):
            return shp
    return None


def _first_shape(header: Dict[str, Any], suffixes: List[str]) -> Optional[List[int]]:
    """Return the shape of the first tensor whose name endswith any suffix."""
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        for suf in suffixes:
            if key.endswith(suf):
                shp = entry.get("shape")
                if isinstance(shp, list):
                    return shp
    return None


# ---------------------------------------------------------------------------
# Source-type + constituent-file enumeration (for cheap invalidation)
# ---------------------------------------------------------------------------

def _infer_source_type(path: str) -> str:
    if isinstance(path, str) and path.endswith(".safetensors.index.json"):
        return "sharded"
    if isinstance(path, str) and path.endswith(".safetensors") and os.path.isfile(path):
        return "safetensors"
    if os.path.isdir(path):
        return "diffusers"
    # sentinel / unknown
    return "safetensors"


def _normalize_source_type(source_type: Optional[str], path: str) -> str:
    st = (source_type or "").strip().lower()
    if st in ("sharded", "safetensors", "diffusers"):
        # honor the explicit choice, but repair the common "safetensors" label on
        # a sharded index or a directory
        if st == "safetensors":
            return _infer_source_type(path)
        return st
    return _infer_source_type(path)


def _constituent_files(path: str, source_type: str) -> List[str]:
    """Files whose (size, mtime) collectively identify the model content.

    NEVER opens a directory as a file. Enumerates cheaply via listdir/index JSON.
    """
    files: List[str] = []
    try:
        if source_type == "safetensors":
            if os.path.isfile(path):
                files.append(path)
        elif source_type == "sharded":
            if os.path.isfile(path):
                files.append(path)
                base = os.path.dirname(path)
                try:
                    with open(path, encoding="utf-8") as f:
                        weight_map = json.load(f).get("weight_map", {}) or {}
                    for shard in sorted(set(weight_map.values())):
                        sp = os.path.join(base, shard)
                        if os.path.isfile(sp):
                            files.append(sp)
                except Exception:
                    pass
        elif source_type == "diffusers":
            # model_index.json + every component config.json + every weight file
            # (safetensors / bin / index json) one level into each subfolder.
            mi = os.path.join(path, "model_index.json")
            if os.path.isfile(mi):
                files.append(mi)
            try:
                for name in os.listdir(path):
                    sub = os.path.join(path, name)
                    if os.path.isdir(sub):
                        for fn in os.listdir(sub):
                            if fn.endswith((".json", ".safetensors", ".bin")):
                                fp = os.path.join(sub, fn)
                                if os.path.isfile(fp):
                                    files.append(fp)
                    elif os.path.isfile(sub) and name.endswith(".json"):
                        files.append(sub)
            except OSError:
                pass
    except Exception as e:
        print(f"[ComponentRegistry] _constituent_files failed for {path}: {e}")
    return files


def _stat_signature(path: str, source_type: str) -> Tuple[int, float]:
    """Cheap aggregate (total_size, max_mtime) over constituent files.

    Used purely as an invalidation guard (no header reads).
    """
    total_size = 0
    max_mtime = 0.0
    for fp in _constituent_files(path, source_type):
        try:
            st = os.stat(fp)
            total_size += st.st_size
            if st.st_mtime > max_mtime:
                max_mtime = st.st_mtime
        except OSError:
            continue
    return total_size, max_mtime


# ---------------------------------------------------------------------------
# Content hash (dir-aware; NOT a full-file SHA256)
# ---------------------------------------------------------------------------

def _hash_parts(parts: List[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _header_identity(header: Dict[str, Any]) -> str:
    """Stable {key:[shape,dtype]} projection of a safetensors header (sorted)."""
    items = []
    for key in sorted(header.keys()):
        if key == "__metadata__":
            continue
        entry = header.get(key)
        if isinstance(entry, dict):
            items.append(f"{key}:{entry.get('shape')}:{entry.get('dtype')}")
    return "|".join(items)


def compute_content_hash(path: str, source_type: str) -> str:
    """Dir-aware cheap content identity.

    single-file : sha256(header {key:[shape,dtype]} + file size)
    sharded     : sha256(index weight_map + each shard header + total size)
    diffusers   : sha256(model_index.json + each component config.json
                          + shard-index header JSONs + per-file (size, mtime))

    This is a valid stable identity for diffusers directories too, so it doubles
    as a ``model_hash`` for the pipeline (which cannot open() a directory).
    """
    parts: List[str] = [f"src={source_type}"]
    try:
        if source_type == "safetensors":
            header = _read_safetensors_header(path)
            parts.append(_header_identity(header))
            try:
                parts.append(f"size={os.path.getsize(path)}")
            except OSError:
                pass

        elif source_type == "sharded":
            base = os.path.dirname(path)
            with open(path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {}) or {}
            parts.append("wm=" + "|".join(f"{k}={v}" for k, v in sorted(weight_map.items())))
            parts.append("meta=" + json.dumps(index.get("metadata", {}) or {}, sort_keys=True))
            total = 0
            for shard in sorted(set(weight_map.values())):
                sp = os.path.join(base, shard)
                try:
                    hdr = _read_safetensors_header(sp)
                    parts.append(f"{shard}:{_header_identity(hdr)}")
                    total += os.path.getsize(sp)
                except Exception:
                    parts.append(f"{shard}:ERR")
            parts.append(f"total={total}")

        elif source_type == "diffusers":
            mi = os.path.join(path, "model_index.json")
            if os.path.isfile(mi):
                try:
                    with open(mi, encoding="utf-8") as f:
                        parts.append("model_index=" + json.dumps(json.load(f), sort_keys=True))
                except Exception:
                    parts.append("model_index=ERR")
            for fp in sorted(_constituent_files(path, source_type)):
                rel = os.path.relpath(fp, path).replace("\\", "/")
                if fp.endswith("config.json") or fp.endswith(".index.json"):
                    try:
                        with open(fp, encoding="utf-8") as f:
                            parts.append(f"{rel}=" + json.dumps(json.load(f), sort_keys=True))
                        continue
                    except Exception:
                        pass
                # weight file: use a content-derived identity (NOT mtime, so the
                # hash is stable across copy/re-download). safetensors -> header
                # identity (keys/shapes/dtypes); other weights -> size only.
                try:
                    if fp.endswith(".safetensors"):
                        hdr = _read_safetensors_header(fp)
                        parts.append(f"{rel}={_header_identity(hdr)}")
                    else:
                        parts.append(f"{rel}={os.path.getsize(fp)}")
                except Exception:
                    parts.append(f"{rel}=ERR")
    except Exception as e:
        print(f"[ComponentRegistry] compute_content_hash failed for {path}: {e}")
        parts.append(f"fallback={path}")
    return _hash_parts(parts)


# ---------------------------------------------------------------------------
# Component extraction per format
# ---------------------------------------------------------------------------

# VAE decoder "conv_in" style tensor: shape[1] == latent_channels.
_VAE_DECODER_CONVIN_SUFFIXES = [
    "decoder.conv_in.weight",  # covers first_stage_model.decoder / vae.decoder / decoder
]
# Backbone entry conv (U-Net): shape[1] == in_channels. Suffixes are kept
# specific (prefixed) so they never collide with the VAE's own conv_in/conv_out.
_UNET_CONVIN_SUFFIXES = [
    "diffusion_model.input_blocks.0.0.weight",
    "unet.conv_in.weight",
]
# Backbone exit conv (U-Net): shape[0] == out_channels.
_UNET_CONVOUT_SUFFIXES = [
    "diffusion_model.out.2.weight",
    "unet.conv_out.weight",
]

# VAE class -> default latent_channels, used ONLY as a fallback when a diffusers
# vae/config.json omits the ``latent_channels`` key (some VAE classes bake it in
# as a class default rather than serializing it). Verified on-disk:
#   * AutoencoderKLQwenImage (krea2): config lacks latent_channels -> class default 16.
_VAE_CLASS_DEFAULT_LATENT_CHANNELS = {
    "AutoencoderKLQwenImage": 16,
}


def _empty_components() -> Dict[str, Any]:
    return {
        "text_encoder": {"present": False, "out_dim": None, "pooled_dim": None,
                         "te_type": None, "embedded": False},
        "backbone": {"kind": None, "in_channels": None, "out_channels": None,
                     "cond_dim": None},
        "vae": {"present": False, "latent_channels": None, "scale_spatial": None,
                "scale_temporal": None, "embedded": False},
        "audio_vae": {"present": False},
    }


def _apply_component_hints(components: Dict[str, Any], metadata: Optional[dict]) -> None:
    """Fold single_file_format component.* hints in (best-effort, no override of
    a concretely-observed number with None)."""
    try:
        from core.models.common.single_file_format import parse_component_metadata
        hints = parse_component_metadata(metadata)
    except Exception:
        return
    vae_h = hints.get("vae") or {}
    if vae_h:
        if vae_h.get("channels") is not None and components["vae"]["latent_channels"] is None:
            try:
                components["vae"]["latent_channels"] = int(vae_h["channels"])
            except (TypeError, ValueError):
                pass
        if vae_h.get("type") and not components["vae"].get("vae_type"):
            components["vae"]["vae_type"] = vae_h["type"]
        if vae_h.get("embedded") is not None:
            components["vae"]["embedded"] = bool(vae_h["embedded"])
            if vae_h["embedded"]:
                components["vae"]["present"] = True
    te_h = hints.get("te") or {}
    if te_h:
        if te_h.get("dim") is not None and components["text_encoder"]["out_dim"] is None:
            try:
                components["text_encoder"]["out_dim"] = int(te_h["dim"])
            except (TypeError, ValueError):
                pass
        if te_h.get("type") and not components["text_encoder"].get("te_type"):
            components["text_encoder"]["te_type"] = te_h["type"]
        if te_h.get("embedded") is not None:
            components["text_encoder"]["embedded"] = bool(te_h["embedded"])
            if te_h["embedded"]:
                components["text_encoder"]["present"] = True


def _extract_from_header(components: Dict[str, Any], header: Dict[str, Any],
                         arch: str) -> None:
    """Extract component dims from a merged safetensors header (single-file /
    all-shards). Presence keyed on transformer./text_encoder./vae. prefixes and
    the classic single-file first_stage_model./conditioner./model. prefixes."""
    keys = [k for k in header.keys() if k != "__metadata__"]

    has_te = any(k.startswith(("text_encoder", "conditioner.", "te.")) for k in keys)
    has_backbone = any(k.startswith(("transformer.", "model.diffusion_model.",
                                     "unet.", "diffusion_model.", "net.",
                                     # FLUX-style top-level DiT keys (lens/krea2 sharded)
                                     "img_in.", "transformer_blocks.",
                                     # ideogram4 dual-transformer second backbone
                                     "unconditional_transformer.")) for k in keys)

    has_vae = any(("vae." in k) or k.startswith("first_stage_model.") for k in keys)
    if not has_vae and not has_backbone and not has_te:
        # Bare original/LDM-format AutoencoderKL VAE checkpoint: no "vae."/
        # "first_stage_model." prefix at all (e.g. a plain `sdxl_vae.safetensors`
        # with top-level `encoder.*`/`decoder.*`/`quant_conv.*` keys, no
        # config.json). Recognized ONLY when the header is VAE-only (no
        # diffusion-backbone or text-encoder keys), so a full model checkpoint's
        # own (prefixed) VAE sub-weights are never mis-flagged as a standalone
        # VAE by this branch.
        has_vae = any(
            k.endswith(("decoder.conv_in.weight", "encoder.conv_in.weight",
                        "post_quant_conv.weight", "quant_conv.weight"))
            for k in keys
        )

    # VAE latent channels from decoder conv-in (shape[1])
    if has_vae:
        components["vae"]["present"] = True
        components["vae"]["embedded"] = True
        shp = _first_shape(header, _VAE_DECODER_CONVIN_SUFFIXES)
        if shp and len(shp) >= 2:
            components["vae"]["latent_channels"] = int(shp[1])

    # Backbone in/out (U-Net conv style)
    if has_backbone:
        components["backbone"]["kind"] = "unet" if arch in _UNET_ARCHS else "transformer"
        cin = _first_shape(header, _UNET_CONVIN_SUFFIXES)
        if cin and len(cin) >= 2:
            components["backbone"]["in_channels"] = int(cin[1])
        cout = _first_shape(header, _UNET_CONVOUT_SUFFIXES)
        if cout and len(cout) >= 1:
            components["backbone"]["out_channels"] = int(cout[0])

    if has_te:
        components["text_encoder"]["present"] = True
        components["text_encoder"]["embedded"] = True


def _scan_single_file(path: str, arch: str, components: Dict[str, Any]) -> None:
    try:
        header = _read_safetensors_header(path)
    except Exception as e:
        print(f"[ComponentRegistry] header read failed for {path}: {e}")
        return
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else None
    try:
        _extract_from_header(components, header, arch)
    except Exception as e:
        print(f"[ComponentRegistry] header extract failed for {path}: {e}")
    _apply_component_hints(components, metadata)


def _scan_sharded(path: str, arch: str, components: Dict[str, Any]) -> None:
    base = os.path.dirname(path)
    merged: Dict[str, Any] = {}
    metadata = None
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
        metadata = index.get("metadata", {}) or {}
        weight_map = index.get("weight_map", {}) or {}
        for shard in sorted(set(weight_map.values())):
            sp = os.path.join(base, shard)
            try:
                hdr = _read_safetensors_header(sp)
                for k, v in hdr.items():
                    if k != "__metadata__":
                        merged[k] = v
            except Exception as e:
                print(f"[ComponentRegistry] shard header failed {sp}: {e}")
    except Exception as e:
        print(f"[ComponentRegistry] sharded index read failed {path}: {e}")
        return
    try:
        _extract_from_header(components, merged, arch)
    except Exception as e:
        print(f"[ComponentRegistry] sharded extract failed {path}: {e}")
    _apply_component_hints(components, metadata)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _scan_diffusers(path: str, arch: str, components: Dict[str, Any]) -> None:
    """Diffusers dir: pure JSON config reads per subfolder (no weight load)."""
    model_index = _read_json(os.path.join(path, "model_index.json")) or {}

    # --- VAE ---
    vae_cfg = _read_json(os.path.join(path, "vae", "config.json"))
    if vae_cfg is not None:
        components["vae"]["present"] = True
        components["vae"]["embedded"] = True
        lc = vae_cfg.get("latent_channels")
        if lc is None:
            # some VAE classes omit latent_channels (baked as a class default)
            cls_name = vae_cfg.get("_class_name")
            lc = _VAE_CLASS_DEFAULT_LATENT_CHANNELS.get(cls_name)
        if lc is not None:
            components["vae"]["latent_channels"] = int(lc)
        ss = vae_cfg.get("spatial_compression_ratio")
        if ss is not None:
            components["vae"]["scale_spatial"] = int(ss)
        ts = vae_cfg.get("temporal_compression_ratio")
        if ts is not None:
            components["vae"]["scale_temporal"] = int(ts)

    # --- Backbone (transformer or unet) ---
    tf_cfg = _read_json(os.path.join(path, "transformer", "config.json"))
    unet_cfg = _read_json(os.path.join(path, "unet", "config.json"))
    bb_cfg = tf_cfg if tf_cfg is not None else unet_cfg
    if bb_cfg is not None:
        components["backbone"]["kind"] = (
            "unet" if unet_cfg is not None and tf_cfg is None else "transformer"
        )
        if arch in _UNET_ARCHS:
            components["backbone"]["kind"] = "unet"
        ic = bb_cfg.get("in_channels")
        if ic is not None:
            components["backbone"]["in_channels"] = int(ic)
        oc = bb_cfg.get("out_channels")
        if oc is not None:
            components["backbone"]["out_channels"] = int(oc)
        cond = (bb_cfg.get("caption_channels")
                or bb_cfg.get("cross_attention_dim")
                or bb_cfg.get("joint_attention_dim"))
        if cond is not None:
            components["backbone"]["cond_dim"] = int(cond)

    # --- Text encoder ---
    te_cfg = _read_json(os.path.join(path, "text_encoder", "config.json"))
    if te_cfg is not None:
        components["text_encoder"]["present"] = True
        components["text_encoder"]["embedded"] = True
        # transformers configs nest text dims under text_config for multimodal LMs
        tc = te_cfg.get("text_config") if isinstance(te_cfg.get("text_config"), dict) else te_cfg
        hs = tc.get("hidden_size") or te_cfg.get("hidden_size") or te_cfg.get("d_model")
        if hs is not None:
            components["text_encoder"]["out_dim"] = int(hs)
        mt = te_cfg.get("model_type") or (te_cfg.get("architectures") or [None])[0]
        if mt:
            components["text_encoder"]["te_type"] = str(mt)

    # --- Audio VAE (video models) ---
    if os.path.isdir(os.path.join(path, "audio_vae")) or "audio_vae" in model_index:
        components["audio_vae"]["present"] = True


# ---------------------------------------------------------------------------
# Baseline + mismatch folding
# ---------------------------------------------------------------------------

def _fold_baseline(record: Dict[str, Any], arch: str) -> None:
    spec = _WIRING_BY_ARCH.get(arch)
    if spec is None:
        return
    components = record["components"]
    mismatches: List[str] = []

    # expected baseline (stored for RP2 compat checks)
    record["expected"] = {
        "te_out_dim": spec.te_out_dim,
        "te_pooled_dim": spec.te_pooled_dim,
        "latent_channels": spec.latent_channels,
        "latent_ndim": spec.latent_ndim,
        "vae_scale_factor": spec.vae_scale_factor,
    }

    # fill VAE latent_channels from baseline if not observed
    obs_lc = components["vae"]["latent_channels"]
    if obs_lc is None and spec.latent_channels:
        components["vae"]["latent_channels"] = spec.latent_channels
    elif obs_lc is not None and spec.latent_channels and obs_lc != spec.latent_channels:
        mismatches.append(
            f"vae.latent_channels observed={obs_lc} expected={spec.latent_channels}")

    # fill TE out_dim from baseline if not observed
    obs_te = components["text_encoder"]["out_dim"]
    if obs_te is None and spec.te_out_dim:
        components["text_encoder"]["out_dim"] = spec.te_out_dim
    elif obs_te is not None and spec.te_out_dim and obs_te != spec.te_out_dim:
        mismatches.append(
            f"te.out_dim observed={obs_te} expected={spec.te_out_dim}")

    if components["text_encoder"]["pooled_dim"] is None and spec.te_pooled_dim:
        components["text_encoder"]["pooled_dim"] = spec.te_pooled_dim

    # backbone cond_dim baseline
    if components["backbone"]["cond_dim"] is None and spec.te_out_dim:
        components["backbone"]["cond_dim"] = spec.te_out_dim

    record["mismatches"] = mismatches


# ---------------------------------------------------------------------------
# Public: scan_model
# ---------------------------------------------------------------------------

def scan_model(path: str, source_type: Optional[str] = None) -> Dict[str, Any]:
    """Produce a ModelComponentRecord (dict) for the model at ``path``.

    CHEAP: reads safetensors headers + JSON configs only. NEVER loads weights,
    NEVER open()'s a directory. Never raises out — a total failure yields a minimal
    record ``{arch, content_hash, source_type, path, ...}``.

    Args:
        path: model path (single-file .safetensors, <stem>.safetensors.index.json,
              or a diffusers directory).
        source_type: optional "safetensors"|"sharded"|"diffusers" (inferred if omitted).

    Returns:
        ModelComponentRecord dict per the shared contract.
    """
    import time

    st = _normalize_source_type(source_type, path)

    # A scanned model_dir often contains non-model files (e.g. sample .png images in
    # a training run's folder). A safetensors header read / content hash on those
    # fails noisily per-file (and slows a full scan); skip any file that isn't a
    # recognizable model file with a minimal record, so no header read is attempted
    # and nothing is logged. Directories still fall through to the diffusers scan.
    if (
        isinstance(path, str)
        and os.path.isfile(path)
        and not path.endswith(".safetensors.index.json")
        and not path.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin"))
    ):
        return {
            "content_hash": _hash_parts([f"nonmodel={path}"]),
            "path": path,
            "source_type": st,
            "size_bytes": 0,
            "mtime": 0,
            "arch": "unknown",
            "is_video": False,
            "latent_ndim": 4,
            "components": _empty_components(),
            "mismatches": [],
            "scanned_at": time.time(),
        }

    # arch detection (its own cheap reads; defensive)
    try:
        from core.model_loader import ModelLoader
        arch = ModelLoader.detect_model_type(path)
    except Exception as e:
        print(f"[ComponentRegistry] detect_model_type failed for {path}: {e}")
        arch = "unknown"

    try:
        content_hash = compute_content_hash(path, st)
    except Exception as e:
        print(f"[ComponentRegistry] content_hash failed for {path}: {e}")
        content_hash = _hash_parts([f"fallback={path}"])

    size_bytes, mtime = _stat_signature(path, st)

    record: Dict[str, Any] = {
        "content_hash": content_hash,
        "path": path,
        "source_type": st,
        "size_bytes": size_bytes,
        "mtime": mtime,
        "arch": arch,
        "is_video": False,
        "latent_ndim": 4,
        "components": _empty_components(),
        "mismatches": [],
        "scanned_at": time.time(),
    }

    # component extraction (per-format, defensive)
    try:
        if st == "safetensors":
            _scan_single_file(path, arch, record["components"])
        elif st == "sharded":
            _scan_sharded(path, arch, record["components"])
        elif st == "diffusers":
            _scan_diffusers(path, arch, record["components"])
    except Exception as e:
        print(f"[ComponentRegistry] extraction failed for {path}: {e}")

    # baseline + mismatches
    try:
        _fold_baseline(record, arch)
    except Exception as e:
        print(f"[ComponentRegistry] baseline fold failed for {path}: {e}")

    # latent_ndim / is_video from wiring baseline (or audio_vae presence)
    spec = _WIRING_BY_ARCH.get(arch)
    if spec is not None:
        record["latent_ndim"] = spec.latent_ndim
    if record["components"]["audio_vae"]["present"]:
        record["latent_ndim"] = 5
    record["is_video"] = record["latent_ndim"] == 5

    return record


# ---------------------------------------------------------------------------
# Persistent cache + get_or_scan
# ---------------------------------------------------------------------------

# Bump when the record shape or the content-hash formula changes, so cached
# entries from an older format are recomputed instead of served stale.
# v2: diffusers content_hash uses safetensors header identity (mtime-free).
# v3: backbone detection extended (img_in./transformer_blocks./unconditional_transformer.);
#     VAE latent_channels class-default fallback; lens/ideogram4/flux2 wiring latent=32.
# v4: bare original/LDM-format AutoencoderKL VAE (un-prefixed decoder/encoder/
#     quant_conv keys, no config.json) is now recognized as a standalone VAE.
_REGISTRY_SCHEMA_VERSION = 4


class ComponentRegistryCache:
    """Persistent path -> {v, content_hash, size, mtime, record} store.

    Mirrors ``backend/utils/hash_cache.py``'s (mtime + size) invalidation, but is
    dir-aware (aggregate stat over constituent files) and stores the full record.
    """

    def __init__(self, cache_file: Optional[str] = None):
        if cache_file is None:
            try:
                from config.settings import settings
                cache_dir = settings.cache_dir
            except Exception:
                cache_dir = os.path.join(os.getcwd(), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "model_component_registry.json")
        self.cache_path = Path(cache_file)
        self._lock = threading.Lock()
        self.cache: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ComponentRegistry] Failed to load cache: {e}")
        return {}

    def _save(self):
        try:
            tmp = self.cache_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            os.replace(tmp, self.cache_path)
        except Exception as e:
            print(f"[ComponentRegistry] Failed to save cache: {e}")

    def get_or_scan(self, path: str, source_type: Optional[str] = None) -> Dict[str, Any]:
        st = _normalize_source_type(source_type, path)
        size_now, mtime_now = _stat_signature(path, st)

        with self._lock:
            entry = self.cache.get(path)
            if entry is not None:
                if (entry.get("v") == _REGISTRY_SCHEMA_VERSION
                        and entry.get("size") == size_now
                        and entry.get("mtime") == mtime_now
                        and isinstance(entry.get("record"), dict)):
                    return entry["record"]

        # cache miss / stale -> rescan (outside lock; scan is pure)
        record = scan_model(path, st)

        with self._lock:
            self.cache[path] = {
                "v": _REGISTRY_SCHEMA_VERSION,
                "content_hash": record.get("content_hash"),
                "size": size_now,
                "mtime": mtime_now,
                "record": record,
            }
            self._save()
        return record

    def invalidate(self, path: str):
        with self._lock:
            if path in self.cache:
                del self.cache[path]
                self._save()

    def clear(self):
        with self._lock:
            self.cache = {}
            self._save()


# module-level singleton (mirrors hash_cache._hash_cache / get_cached_file_hash)
_registry_cache: Optional[ComponentRegistryCache] = None
_singleton_lock = threading.Lock()


def get_registry_cache() -> ComponentRegistryCache:
    global _registry_cache
    if _registry_cache is None:
        with _singleton_lock:
            if _registry_cache is None:
                _registry_cache = ComponentRegistryCache()
    return _registry_cache


def get_or_scan(path: str, source_type: Optional[str] = None) -> Dict[str, Any]:
    """Convenience: cached record for ``path`` (scan + persist on miss/stale)."""
    return get_registry_cache().get_or_scan(path, source_type)


def get_content_hash(path: str, source_type: Optional[str] = None) -> str:
    """Dir-aware content hash for ``path`` (usable as a diffusers-dir model_hash)."""
    return get_or_scan(path, source_type).get("content_hash", "")
