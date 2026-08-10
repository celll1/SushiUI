"""Component loader for MiniMax-H3 (joint video + audio DiT + Qwen3-VL + two VAEs).

Phase 1: make the model LOADABLE into SushiUI's single-in-memory model slot and
switchable out of it again. Sampling (``h3_pipeline_ops``), the block-loop
wrapper and the quantization registries are Phase 2/4 and are deliberately NOT
here.

DISTRIBUTION FORMAT
-------------------
The primary tree is the ComfyUI-style flat layout (confirmed locally), whose
weights carry no diffusers ``config.json`` anywhere::

    <root>/diffusion_models/minimax_h3_{fl2va,ref2va}_pruned_fp8_scaled.safetensors
    <root>/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
    <root>/vae/minimax_h3_video_vae_fp16.safetensors
    <root>/vae/minimax_h3_audio_vae_fp32.safetensors
    <root>/official/                 <- MiniMax's own config-only tree
        model_index.json  (_class_name: "MiniMaxH3ModularPipeline")
        {transformer,vae,audio_vae,text_encoder,scheduler,audio_scheduler}/config.json
        {tokenizer,processor}/

``official/`` holds the CONFIGS and the tokenizer/processor; it holds no weights
(``official/vae/`` is a lone ``config.json``). So the two halves are always used
together: geometry and normalization vectors come from ``official/``, tensors
from the flat tree. A bare ``official/`` path resolves upward to its parent root
so the model still loads when the user points at the diffusers-shaped directory.

WHAT THIS LOADER HAS TO GET RIGHT (all measured in Phase 0 / K0; see
``scratchpad/minimax_h3_phase0_findings.md`` and ``minimax_h3_k0_results.md``)
------------------------------------------------------------------------------
1. **The released checkpoints are the "pruned" / AdaLN-curve variant.** They
   carry no ``time_embedder.*`` at all: the timestep MLP is replaced by an
   ``adaln_t_table`` ``[1025, 8]`` buffer plus per-block ``adaln_proj.linear``
   projections of width 8. ``official/transformer/config.json`` describes the
   FULL-modulation variant (``time_embed_dim`` 2688) and must NOT be applied to
   a pruned file -- so the variant-dependent geometry is synthesised from the
   file's own header (``_synthesize_transformer_config``).
2. **TWO OPPOSITE QKV CONVENTIONS IN ONE DISTRIBUTION.** The DiT single file is
   already ``[q_all | k_all | v_all]`` CONTIGUOUS -- it needs a plain split and
   MUST NOT be de-interleaved (this contradicts the upstream conversion
   script's premise, which is written for the original MiniMax shards). The
   video VAE decoder's ``to_qkv`` IS per-head interleaved and MUST be
   de-interleaved. Both were discriminated by measured RoPE row-norm signatures
   (rotated q/k rows vs flat v rows) and corroborated against ComfyUI's own
   ``split`` / ``view(...).chunk(3)`` call sites. Getting either one backwards
   produces a model that loads perfectly and generates noise.
3. **SwiGLU halves are swapped.** Comfy/reference ``fc1`` is ``[gate; up]``
   (``silu(gate) * up``); the diffusers ``SwiGLU`` chunks to ``[hidden; gate]``
   (``hidden * silu(gate)``). Swap at load, in the DiT and in the video VAE.
4. **The audio VAE ships weight-norm PRE-FOLDED.** The vendored class carries
   ``weight_g``/``weight_v``; ``remove_weight_norm`` must be folded out of 172
   modules before the load, or it is 268 missing / 134 unexpected keys.
5. **fp8 scale sidecars are SCALARS.** ``weight_scale`` is a per-tensor
   ``F32 []``, not the ``(out_features,)`` vector ``Fp8Linear`` expects, so it
   is broadcast at load. 150 of the 200 quantized Linears also carry an
   ``input_scale``, and the 50 that do not are exactly the 50 ``mlp.fc2`` layers
   marked ``"full_precision_matrix_mult": true`` -- see
   ``_dit_quantization_policy`` for why every one of the 200 is pinned to the
   dequant path here.
6. **The text encoder is truncated to 50 decoder layers** and keeps its full
   27-block vision tower. A three-rule prefix rewrite maps it onto transformers'
   ``Qwen3VLForConditionalGeneration`` with ``num_hidden_layers=50``, leaving
   exactly two missing keys (``lm_head.weight``,
   ``model.language_model.norm.weight``), neither of which the
   "unnormalised hidden state after layer 50" read uses.
7. **The TE's CPU weights must stay MEMORY-MAPPED.** ``load_state_dict(assign=True)``
   installs the ``safe_open`` tensors directly and nothing here writes them back
   or casts them. K0.7 measured the alternative: moving each layer with
   ``layer.to(cuda, fp32)`` / ``layer.to("cpu", bf16)`` detaches every parameter
   from the file mapping and costs 73.08 GB peak RSS + pagefile growth against
   49.82 GB flat for the ``torch.func.functional_call`` shape Phase 2 must use.
   A second concurrent ``safe_open`` of the 48 GiB file in one process killed a
   K0.7 run with Windows ``os error 1455``; this loader opens exactly one at a
   time and closes it.
8. **The video VAE's normalization vectors come from the fp32 config**, not from
   the fp16 tensors in the file (max abs diff 8.4e-4 -- pure rounding), and its
   pixel convention is ImageNet-normalised RGB over a ``[0, 1]`` base, NOT
   ``[-1, 1]`` like every other VAE in this repo. The conversion is owned by the
   consumer (``vae_encode_clip`` / the sampler), not by this loader; the
   constants are exported here so there is one copy.
9. **Spatial tiling is load-bearing, not a memory nicety.** Flipping the shipped
   flags changed the latents by rel-RMS 0.355 and the decode by 0.212 on the
   same input (K0.5 supplementary). The policy is therefore PINNED here and
   reported in the component dict so the training-latent cache and generation
   cannot silently disagree about it.

QUANTIZED-SEMANTICS GUARD
-------------------------
Every one of the four component loads runs the state dict through
``core.models.common.quantized_checkpoint_guard`` BEFORE any tensor is
installed: the DiT through ``quantized_state_dict_report`` +
``scaled_quantization_report`` + ``verify_quantized_swap`` (it supports the
scaled layout), and the TE and both VAEs through ``refuse_quantized_state_dict``
(they have no swap path, so a quantized file must be refused rather than cast).
The DiT additionally accepts the exact released ``int8_tensorwise`` ConvRot
contract (groupsize 256) and executes its online activation rotation through
Comfy-Kitchen; other ConvRot declarations remain refused. Packed
``asym_w4a8_int8`` DiTs are handled separately from file-level metadata.
``_assert_guard_reached`` pins the remaining guard property in code.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Distribution layout
# ---------------------------------------------------------------------------

# Preferred filenames, in probe order. The glob fallback in ``_find_first``
# covers the variants that are not downloaded here (``*_pruned_bf16``,
# ``*_video_vae_fp32``) as well as a re-export under a user-chosen stem.
MINIMAX_H3_DIT_PATTERNS: List[str] = [
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_bf16.safetensors",
    "minimax_h3_ref2va_pruned_bf16.safetensors",
]
MINIMAX_H3_VIDEO_VAE_PATTERNS: List[str] = [
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_video_vae_fp32.safetensors",
]
MINIMAX_H3_AUDIO_VAE_PATTERNS: List[str] = [
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_audio_vae_fp16.safetensors",
]
MINIMAX_H3_TE_PATTERNS: List[str] = [
    "qwen3vl_32b_minimax_h3_bf16.safetensors",
]

# Sibling directory names probed for MiniMax's config-only tree (configs +
# tokenizer + processor). ``official`` is what the download script produces.
_OFFICIAL_DIR_NAMES = ("official", "MiniMax-H3", "minimax_h3_official")

# The diffusers ``model_index.json`` class name of MiniMax's own modular
# pipeline. Unique to this architecture, so it disambiguates a directory that
# carries one.
MINIMAX_H3_PIPELINE_CLASS = "MiniMaxH3ModularPipeline"


# ---------------------------------------------------------------------------
# Measured geometry (Phase 0). Exported so nothing downstream re-derives them.
# ---------------------------------------------------------------------------

# VAE spatial / temporal compression. READ from the video VAE file's own
# embedded ``source_config`` (``vae_ratio`` 16, ``vae_ratio_t`` 4) and confirmed
# against the loaded module. NOTE: the *token* downsample seen by the DiT is 32x
# spatially, because the transformer additionally patchifies 2x2 -- that is a
# transformer property (``patch_size``), not a VAE scale factor, and the two are
# deliberately kept apart here.
MINIMAX_H3_VAE_SPATIAL_COMPRESSION = 16
MINIMAX_H3_VAE_TEMPORAL_COMPRESSION = 4
MINIMAX_H3_LATENT_CHANNELS = 24
MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
# 32000 / hop_length 800. Exact: ``T_aud = round(T / 24 * 40)``.
MINIMAX_H3_AUDIO_LATENT_RATE = 40.0
MINIMAX_H3_FPS = 24.0

# Pixel convention of the video VAE: ImageNet-normalised RGB over a [0, 1] base
# (``"pixel_norm_type": "imagenet"`` in the file's own source_config, verified
# bitwise against MiniMax's reference implementation in K0.5). Every other VAE in
# this repo takes [-1, 1].
MINIMAX_H3_PIXEL_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# PINNED spatial tiling policy for the video VAE (K0.5 supplementary: flipping
# these changes the latents by rel-RMS 0.355 and the decode by 0.212). These are
# the values MiniMax's own wrapper ships and the vendored class defaults to; they
# are restated explicitly so the policy is a decision in one place rather than a
# class default that a future vendor refresh could move. Reported in the
# component dict as ``vae_tiling_policy`` so the training-latent cache key can
# carry it (Phase 6a).
#
# NOT the ``vae_tiling`` GENERATION parameter in
# ``param_defaults.GENERATION_DEFAULTS``, and it must never be wired to it: that
# one is a user-facing memory knob for the image archs' decode, where turning
# tiling off costs peak VRAM and nothing else. Turning THIS one off changes what
# the model produces. The names are kept distinct on purpose.
MINIMAX_H3_VAE_TILING_POLICY: Dict[str, Any] = {
    "enabled": True,
    "tile_sample_min_height": 256,
    "tile_sample_min_width": 256,
    "tile_sample_min_overlap_height": 64,
    "tile_sample_min_overlap_width": 64,
}


def minimax_h3_latent_frames(num_frames: int) -> int:
    """Latent frame count for ``num_frames`` pixel frames. MEASURED.

    ``1`` at ``T == 1`` (the spatial-only image-conditioning path), else
    ``ceil(T / 17) * 5 - 3``. ComfyUI's own ``2 if T<=5 else ((T-5)//17)*5+2``
    agrees only on the ``17n+5`` grid and disagrees off it (T=18: Comfy 2,
    measured 7), so this form is the one to use.

    Note the decode floor this implies: ``_decode`` needs at least 7 latent
    frames, so the shortest decodable clip is 22 pixel frames (0.917 s) -- T = 5
    is on the grid and cannot be decoded.
    """
    if num_frames <= 1:
        return 1
    return math.ceil(num_frames / 17) * 5 - 3


def _find_first(directory: Path, patterns: List[str],
                accept=None) -> Optional[Path]:
    """The first file in ``directory`` matching ``patterns``, then a glob, then any.

    ``accept`` is an optional predicate a candidate must satisfy. It exists for
    the DiT slot: detection identifies a tree by HEADER-PROBING its
    ``diffusion_models/`` files, so without the same probe here a directory
    holding a renamed H3 DiT beside any other ``.safetensors`` could detect as
    MiniMax-H3 and then load the other file. The failure is loud (the key map
    would not match), but it names the wrong file, and the two functions
    disagreeing about which file IS the model is the kind of thing that costs a
    session.
    """
    if not directory.is_dir():
        return None
    ok = accept or (lambda _p: True)
    for pat in patterns:
        candidate = directory / pat
        if candidate.is_file() and ok(candidate):
            return candidate
    for group in (sorted(directory.glob("*minimax_h3*.safetensors")),
                  sorted(directory.glob("*.safetensors"))):
        for candidate in group:
            if ok(candidate):
                return candidate
    return None


def _resolve_official_dir(root: Path) -> Optional[str]:
    """MiniMax's config-only tree under ``root`` (or ``root`` itself)."""
    for name in _OFFICIAL_DIR_NAMES:
        candidate = root / name
        if (candidate / "model_index.json").is_file():
            return str(candidate)
    if (root / "model_index.json").is_file():
        return str(root)
    return None


def _is_h3_model_index(directory: Path) -> bool:
    index = directory / "model_index.json"
    if not index.is_file():
        return False
    try:
        with open(index, encoding="utf-8") as fh:
            return json.load(fh).get("_class_name") == MINIMAX_H3_PIPELINE_CLASS
    except Exception:
        return False


def detect_minimax_h3_layout(path: str) -> Optional[Dict[str, Optional[str]]]:
    """``{dit, vae, audio_vae, text_encoder, official, root, variant}`` or ``None``.

    Accepts three spellings of the same tree:

    * the flat root (``<root>/diffusion_models/`` + ``vae/`` + ``text_encoders/``);
    * a DiT ``.safetensors`` inside ``<root>/diffusion_models/`` (walks up);
    * MiniMax's config-only ``official/`` directory, i.e. one carrying a
      ``model_index.json`` whose ``_class_name`` is ``MiniMaxH3ModularPipeline``
      -- resolved to its parent when that parent holds the weights, because
      ``official/`` alone has none.
    """
    if not path:
        return None
    p = Path(path)

    root: Optional[Path] = None
    if p.is_file() and p.suffix == ".safetensors":
        for parent in p.parents:
            if (parent / "diffusion_models").is_dir():
                return _layout_from_root(parent, dit_override=p)
        return None
    if not p.is_dir():
        return None

    if (p / "diffusion_models").is_dir():
        root = p
    elif _is_h3_model_index(p) and (p.parent / "diffusion_models").is_dir():
        # A bare ``official/``: the configs are here, the weights one level up.
        root = p.parent
    elif _is_h3_model_index(p):
        # A model_index.json with no reachable weight tree. Returned anyway (so
        # detection is honest about WHAT this is) with every weight slot None;
        # the loader turns that into a message naming the missing files.
        return {"dit": None, "vae": None, "audio_vae": None, "text_encoder": None,
                "official": str(p), "root": str(p), "variant": None}
    if root is None:
        return None
    return _layout_from_root(root)


def _layout_from_root(root: Path, dit_override: Optional[Path] = None) -> Optional[Dict[str, Optional[str]]]:
    # The DiT slot is filtered by the SAME key-name signature detection uses, so
    # the file this resolves and the file that made the tree detect as MiniMax-H3
    # are always the same one.
    dit = dit_override if dit_override is not None else _find_first(
        root / "diffusion_models", MINIMAX_H3_DIT_PATTERNS,
        accept=lambda p: is_minimax_h3_safetensors(str(p)))
    if dit is None:
        return None
    vae = _find_first(root / "vae", MINIMAX_H3_VIDEO_VAE_PATTERNS)
    audio_vae = _find_first(root / "vae", MINIMAX_H3_AUDIO_VAE_PATTERNS)
    # ``_find_first``'s glob fallback can hand back the same file for both slots
    # when only one of the two is present; a video VAE is not an audio VAE.
    if vae is not None and audio_vae is not None and vae == audio_vae:
        audio_vae = None
    te = _find_first(root / "text_encoders", MINIMAX_H3_TE_PATTERNS)
    name = dit.name.lower()
    variant = "ref2va" if "ref2va" in name else ("fl2va" if "fl2va" in name else None)
    return {
        "dit": str(dit),
        "vae": str(vae) if vae else None,
        "audio_vae": str(audio_vae) if audio_vae else None,
        "text_encoder": str(te) if te else None,
        "official": _resolve_official_dir(root),
        "root": str(root),
        "variant": variant,
    }


# ---------------------------------------------------------------------------
# Header reading + single-file key signature
# ---------------------------------------------------------------------------

def read_safetensors_header(path: str) -> Dict[str, Any]:
    """The JSON header of a safetensors file. ZERO tensor bytes are read."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length {header_len} in {path}")
        return json.loads(fh.read(header_len).decode("utf-8"))


# Key prefixes carried by NO other architecture in this repo. MEASURED over every
# ``.safetensors`` under ``M:/model/**`` (77 files, 11 archs): each of these
# appears in the MiniMax-H3 files and in nothing else.
#
# The diffusers spellings are deliberately ABSENT from this tuple. LTX-2.3's
# transformer shards carry ``proj_in.``, ``audio_proj_in.`` AND ``audio_proj_out.``
# (measured on
# ``M:/model/LTX2.3/distilled/transformer/diffusion_pytorch_model-00001-of-00008.safetensors``),
# so "a second, audio-specific projection on the DiT" is NOT unique to MiniMax-H3
# -- an earlier version of this probe claimed it was, and was wrong.
_MINIMAX_H3_ONLY_KEYS = (
    "adaln_t_table",            # the pruned/AdaLN-curve table (top-level tensor)
    "condition_proj.",          # Comfy's name for the text projection
    "audio_patch_proj.",
    "video_patch_proj.",
    "final_layer.audio_out.",
    "final_layer.video_out.",
)


def keys_look_minimax_h3(keys) -> bool:
    """MiniMax-H3 DiT single-file signature. Key NAMES only, no tensor reads.

    TWO requirements, both needed, because this probe runs FIRST among the
    single-file probes in ``detect_model_type`` and therefore has to be the
    narrow one:

    * ``token_refiner.`` -- the 2-layer text-stream refiner. Measured absent from
      every other arch on this box, but it is an ordinary DiT component and a
      future LTX-2.x could grow one, so it is not sufficient on its own;
    * at least one key from ``_MINIMAX_H3_ONLY_KEYS`` -- the released
      single-file (Comfy) naming plus the pruned variant's ``adaln_t_table``,
      none of which any other arch here carries.

    LIMIT, stated rather than papered over: a hypothetical FULL-modulation H3
    checkpoint re-exported under DIFFUSERS names would match neither clause's
    second half (``context_embedder.`` / ``proj_out.`` / ``audio_proj_out.`` are
    all shared with LTX-2.3, and it has no ``adaln_t_table``) and would not be
    detected here. No such file is distributed -- MiniMax ships the diffusers
    form as a DIRECTORY, which is matched by ``model_index.json`` instead -- and
    a probe that accepted it would be one that can capture LTX-2.3.
    """
    keys = list(keys)
    if not any(k.startswith("token_refiner.") for k in keys):
        return False
    return any(k in _MINIMAX_H3_ONLY_KEYS or k.startswith(_MINIMAX_H3_ONLY_KEYS)
               for k in keys)


def is_minimax_h3_safetensors(path: str) -> bool:
    """``keys_look_minimax_h3`` against a file's header. Never raises."""
    try:
        header = read_safetensors_header(path)
        header.pop("__metadata__", None)
        return keys_look_minimax_h3(header.keys())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DiT: config synthesis from the header census
# ---------------------------------------------------------------------------

# safetensors header dtype names -> torch dtypes. The names are part of the
# on-disk format, so this cannot drift with the library's API.
_HEADER_DTYPES: Dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    _dt = getattr(torch, _attr, None)
    if _dt is not None:
        _HEADER_DTYPES[_name] = _dt

_W4A8_SUFFIXES = (
    ".weight", ".weight_s_rel", ".weight_s_channel", ".weight_codebook",
    ".weight_correction",
)

# Bitwise equal to ``1 / theta ** (arange(0, 2d, 2) / 2d)`` (MEASURED), and a
# non-persistent buffer in the vendored class, so it is dropped rather than
# mapped.
_DIT_DROPPED_KEYS = ("rope.inv_freq",)


def _header_shape(header: Dict[str, Any], key: str) -> Optional[List[int]]:
    entry = header.get(key)
    if isinstance(entry, dict) and isinstance(entry.get("shape"), list):
        return entry["shape"]
    return None


def _synthesize_transformer_config(
    header: Dict[str, Any], official_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """The vendored transformer's config, DERIVED from the checkpoint header.

    ``official/transformer/config.json`` describes the FULL-modulation variant
    (``time_embed_dim`` 2688, a ``time_embedder`` MLP) and every released
    single-file checkpoint is the PRUNED one (``time_embed_dim`` 8 plus an
    ``adaln_t_table``), so applying that config to these weights builds 50
    blocks' worth of module paths that match no tensor. Every field that differs
    between the two variants is therefore taken from the header; the official
    config contributes only the fields the header cannot express
    (``patch_size``, the head split, the epsilons, the rope base), and the
    vendored class defaults -- which are the release's own values -- fill in
    when it is absent.

    The head split is the one place a wrong answer would be silent: the header
    gives ``num_attention_heads * attention_head_dim`` (7168) but not its
    factors, so whichever source supplies them is CHECKED against that product
    and a mismatch raises here rather than at the first attention forward.
    """
    config: Dict[str, Any] = {}
    if official_dir:
        candidate = os.path.join(official_dir, "transformer", "config.json")
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as fh:
                    config = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
            except Exception as exc:
                print(f"[MiniMaxH3Loader] could not read {candidate}: {exc}; using class defaults")
                config = {}
    # Only the variant-INDEPENDENT fields survive from the official config; the
    # rest are derived below. Dropping them explicitly (rather than letting the
    # override happen to cover them) keeps a future config key from sneaking in.
    for variant_dependent in ("time_embed_dim", "time_embed_hidden_dim", "freq_dim",
                              "num_layers", "num_refiner_layers", "hidden_size",
                              "ffn_dim", "in_channels", "audio_in_channels", "text_dim",
                              "adaln_curve_grid"):
        config.pop(variant_dependent, None)

    def shape(key: str) -> List[int]:
        shp = _header_shape(header, key)
        if shp is None:
            raise ValueError(
                f"the MiniMax-H3 transformer checkpoint has no '{key}' tensor, so its geometry "
                f"cannot be derived from the header. This loader reads the Comfy-style "
                f"single-file layout (video_patch_proj / audio_patch_proj / condition_proj / "
                f"blocks.N.* / final_layer.*).")
        return shp

    hidden_size = int(shape("video_patch_proj.weight")[0])
    video_patch_dim = int(shape("video_patch_proj.weight")[1])
    audio_in_channels = int(shape("audio_patch_proj.weight")[1])
    text_dim = int(shape("condition_proj.weight")[1])
    ffn_dim = int(shape("blocks.0.mlp.fc1.weight")[0]) // 2  # fc1 is [gate; up]
    qkv_rows = int(shape("blocks.0.attn.qkv_proj.weight")[0])

    num_layers = 1 + max(
        (int(k.split(".")[1]) for k in header if k.startswith("blocks.") and k.count(".") > 1),
        default=-1)
    num_refiner_layers = 1 + max(
        (int(k.split(".")[2]) for k in header if k.startswith("token_refiner.blocks.")),
        default=-1)
    if num_layers <= 0 or num_refiner_layers <= 0:
        raise ValueError(
            f"the MiniMax-H3 transformer checkpoint declares {num_layers} block(s) and "
            f"{num_refiner_layers} refiner block(s); both must be positive.")

    config["hidden_size"] = hidden_size
    config["num_layers"] = num_layers
    config["num_refiner_layers"] = num_refiner_layers
    config["ffn_dim"] = ffn_dim
    config["audio_in_channels"] = audio_in_channels
    config["text_dim"] = text_dim
    config.setdefault("num_attention_heads", 56)
    config.setdefault("attention_head_dim", 128)
    config.setdefault("patch_size", (1, 2, 2))

    patch_size = tuple(int(x) for x in config["patch_size"])
    patch_elems = patch_size[0] * patch_size[1] * patch_size[2]
    if video_patch_dim % patch_elems:
        raise ValueError(
            f"the MiniMax-H3 video patch projection takes {video_patch_dim} channels, which is "
            f"not divisible by the patch volume {patch_elems} implied by patch_size="
            f"{patch_size}. The transformer config and the checkpoint disagree about the patch "
            f"geometry; latent channels cannot be derived.")
    config["in_channels"] = video_patch_dim // patch_elems
    config["patch_size"] = patch_size

    inner = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    if qkv_rows != 3 * inner:
        raise ValueError(
            f"the MiniMax-H3 checkpoint's fused qkv projection has {qkv_rows} rows, but the "
            f"configured head split ({config['num_attention_heads']} heads x "
            f"{config['attention_head_dim']} channels) implies {3 * inner}. The split cannot be "
            f"read off the header, so a wrong one would only surface as garbage inside "
            f"attention; refusing here instead.")

    # The pruned / AdaLN-curve variant, detected from the file rather than
    # assumed: an ``adaln_t_table`` (grid rows x time_embed_dim) and NO
    # ``time_embedder.*``.
    table = _header_shape(header, "adaln_t_table")
    has_time_embedder = any(k.startswith("time_embedder.") for k in header)
    if table is not None:
        if has_time_embedder:
            raise ValueError(
                "the MiniMax-H3 checkpoint carries BOTH an 'adaln_t_table' and 'time_embedder.*' "
                "keys. The AdaLN-curve ('pruned') variant and the full-modulation variant are "
                "mutually exclusive; this file matches neither.")
        if len(table) != 2:
            raise ValueError(f"'adaln_t_table' must be 2-D (grid, time_embed_dim), got {table}")
        config["adaln_curve_grid"] = int(table[0])
        config["time_embed_dim"] = int(table[1])
    else:
        config["adaln_curve_grid"] = None
        adaln_weight = shape("blocks.0.adaln_proj.linear.weight")
        config["time_embed_dim"] = int(adaln_weight[1])
        config.setdefault("time_embed_hidden_dim", hidden_size)
        config.setdefault("freq_dim", 256)

    adaln_in = int(shape("blocks.0.adaln_proj.linear.weight")[1])
    if adaln_in != int(config["time_embed_dim"]):
        raise ValueError(
            f"the MiniMax-H3 AdaLN projection takes {adaln_in} inputs but the derived "
            f"time_embed_dim is {config['time_embed_dim']}.")
    return config


# ---------------------------------------------------------------------------
# DiT: key mapping
# ---------------------------------------------------------------------------

def _rename_dit_key(key: str) -> str:
    """Comfy single-file key -> vendored ``MiniMaxH3Transformer3DModel`` key.

    The full table (MEASURED: 1082 source tensors -> 1 dropped + 550 sidecars +
    635 module keys, which is exactly the vendored model's key count, strict-load
    clean with no missing / unexpected / shape mismatch).
    """
    out = key
    if out.startswith("token_refiner.blocks."):
        out = out.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif out.startswith("blocks."):
        out = out.replace("blocks.", "transformer_blocks.", 1)
    out = out.replace("video_patch_proj.", "proj_in.")
    out = out.replace("audio_patch_proj.", "audio_proj_in.")
    out = out.replace("condition_proj.", "context_embedder.")
    out = out.replace("final_layer.norm.", "norm_out.norm.")
    out = out.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    out = out.replace("final_layer.video_out.", "proj_out.")
    out = out.replace("final_layer.audio_out.", "audio_proj_out.")
    out = out.replace(".attn.q_norm.", ".attn.norm_q.")
    out = out.replace(".attn.k_norm.", ".attn.norm_k.")
    out = out.replace(".attn.out_proj.", ".attn.to_out.0.")
    out = out.replace(".mlp.fc1.", ".ff.net.0.proj.")
    out = out.replace(".mlp.fc2.", ".ff.net.2.")
    return out


# Module-name prefixes/suffixes the checkpoint stores in float32 and the vendored
# class lists in ``_keep_in_fp32_modules``. Matched as substrings of the MAPPED
# key, so the audio heads are covered by the same two entries.
_DIT_FP32_MODULES = ("proj_in.", "audio_proj_in.", "proj_out.", "audio_proj_out.",
                     "time_embedder.")

# The AdaLN projections run in float32 in curve mode -- ComfyUI sets
# ``adaln_dtype = torch.float32`` there, and the F16-stored weights are upcast.
# The modulation vectors are cast back down to the block stack's dtype inside the
# block (mirroring ComfyUI's ``_mod_scale_shift``), so this does NOT promote the
# residual stream.
_DIT_ADALN_KEYS = (".adaln_proj.linear.", "norm_out.linear.")


def _dit_target_dtype(mapped_key: str, compute_dtype: torch.dtype,
                      curve_variant: bool) -> Optional[torch.dtype]:
    """The dtype a mapped DiT tensor is installed at, or ``None`` to leave it.

    ``None`` means "do not touch": the fp8 codes and their float32 scales, which
    ``Fp8Linear`` owns.
    """
    if mapped_key.endswith((".weight_scale", ".weight_s_rel", ".weight_s_channel",
                            ".weight_codebook", ".weight_correction")):
        return None
    if mapped_key == "adaln_t_table":
        return torch.float32
    if any(part in mapped_key for part in _DIT_FP32_MODULES):
        return torch.float32
    if curve_variant and any(part in mapped_key for part in _DIT_ADALN_KEYS):
        return torch.float32
    return compute_dtype


def _map_dit_state_dict(
    handle, header: Dict[str, Any], config: Dict[str, Any], compute_dtype: torch.dtype,
    w4a8_layers: Optional[Dict[str, Dict[str, Any]]] = None,
    int8_convrot_layers: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Read the Comfy single file into the vendored model's key space.

    ``handle`` is an OPEN ``safe_open``; tensors come back memory-mapped, so the
    entries that need no value transform cost no resident memory here. The two
    that do -- the qkv split and the SwiGLU half swap -- are handled as follows:

    * **qkv**: ``[q_all | k_all | v_all]`` CONTIGUOUS (measured; see the module
      docstring). Each third is a leading row slice of a contiguous tensor, so
      the three views are contiguous and cost nothing. **No de-interleave**: that
      is the video VAE's convention, not this one.
    * **fc1**: ``[gate; up]`` -> ``[up; gate]``, which is a real copy (the only
      unavoidable one).

    A per-tensor ``weight_scale`` scalar is broadcast to the ``(out_features,)``
    vector ``Fp8Linear`` registers, and a split qkv's scalar is broadcast into
    each of the three parts -- exact, because the scale is per TENSOR. K0.1
    verified this adapter against an ``F.linear(x, codes.float() * scalar, bias)``
    oracle on all 200 quantized Linears (fp32 bitwise; bf16 rel-RMS 4.7e-3).
    A PER-ROW scale is indexed by output row, so it goes through the SAME row
    permutation as its weight. This is live for the released INT8 ConvRot files:
    fused QKV scales split three ways and the SwiGLU scale halves swap.

    ``input_scale`` is DROPPED, and that is safe only because the W8A8 path is
    switched off for every layer of this arch (``_dit_quantization_policy``):
    the repo's ``Fp8Linear`` quantizes activations dynamically and never reads a
    checkpoint ``input_scale``, so keeping it would be decoration and dropping it
    while a scaled GEMM could run would be silently wrong. The count is returned
    and printed rather than discarded.
    """
    inner = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    curve = config.get("adaln_curve_grid") is not None
    mapped: Dict[str, torch.Tensor] = {}
    w4a8_layers = w4a8_layers or {}
    int8_convrot_layers = int8_convrot_layers or {}
    stats = {"dropped": 0, "input_scale_dropped": 0, "qkv_split": 0, "swiglu_swapped": 0,
             "markers": 0, "scales_broadcast": 0, "swiglu_scale_swapped": 0}

    for key in header:
        if key == "__metadata__":
            continue
        if key in _DIT_DROPPED_KEYS:
            stats["dropped"] += 1
            continue
        if key.endswith(".input_scale"):
            stats["input_scale_dropped"] += 1
            continue

        if key.endswith(".comfy_quant"):
            source = key[: -len(".comfy_quant")]
            marker = handle.get_tensor(key)
            if source in int8_convrot_layers:
                if source.endswith(".attn.qkv_proj"):
                    target = _rename_dit_key(source + ".weight")
                    stem = target.split(".attn.qkv_proj.")[0] + ".attn."
                    for name in ("to_q", "to_k", "to_v"):
                        mapped[stem + name + ".comfy_quant"] = marker
                else:
                    target = _rename_dit_key(source + ".weight")[:-len(".weight")]
                    mapped[target + ".comfy_quant"] = marker
            else:
                mapped[_rename_dit_key(key)] = marker
            stats["markers"] += 1
            continue

        w4_suffix = next((suffix for suffix in _W4A8_SUFFIXES if key.endswith(suffix)), None)
        w4_module = key[:-len(w4_suffix)] if w4_suffix else None
        if w4_module in w4a8_layers:
            tensor = handle.get_tensor(key)
            if ".attn.qkv_proj" in w4_module:
                stem = _rename_dit_key(w4_module + ".weight").split(".attn.qkv_proj.")[0] + ".attn."
                names = [stem + "to_q", stem + "to_k", stem + "to_v"]
                if w4_suffix == ".weight_codebook":
                    if tuple(tensor.shape) != (16,):
                        raise ValueError(f"{key}: W4A8 codebook must have shape (16,)")
                    for name in names:
                        mapped[name + w4_suffix] = tensor
                elif w4_suffix == ".weight_correction":
                    if tensor.ndim != 2 or tensor.shape[1] != 3 * inner:
                        raise ValueError(
                            f"{key}: W4A8 qkv correction has shape {tuple(tensor.shape)}, "
                            f"expected (*, {3 * inner})")
                    for i, name in enumerate(names):
                        mapped[name + w4_suffix] = tensor[:, i * inner:(i + 1) * inner].contiguous()
                else:
                    if tensor.shape[0] != 3 * inner:
                        raise ValueError(
                            f"{key}: fused W4A8 qkv has {tensor.shape[0]} rows, "
                            f"expected {3 * inner}")
                    for i, name in enumerate(names):
                        mapped[name + w4_suffix] = tensor[i * inner:(i + 1) * inner]
                if w4_suffix == ".weight":
                    stats["qkv_split"] += 1
                continue

            target = _rename_dit_key(key)
            if target.endswith(".ff.net.0.proj" + w4_suffix):
                if w4_suffix == ".weight_codebook":
                    pass
                elif w4_suffix == ".weight_correction":
                    gate, up = tensor.chunk(2, dim=1)
                    tensor = torch.cat([up, gate], dim=1).contiguous()
                else:
                    gate, up = tensor.chunk(2, dim=0)
                    tensor = torch.cat([up, gate], dim=0).contiguous()
                if w4_suffix == ".weight":
                    stats["swiglu_swapped"] += 1
            mapped[target] = tensor
            continue

        target = _rename_dit_key(key)
        is_scale = key.endswith(".weight_scale")
        tensor = handle.get_tensor(key)

        if ".attn.qkv_proj." in key:
            # Everything up to (and including) ".attn." -- the fused name AND the
            # parameter suffix come off, since the three parts get their own.
            stem = target.split(".attn.qkv_proj.")[0] + ".attn."
            names = [stem + "to_q", stem + "to_k", stem + "to_v"]
            if is_scale:
                flat_scale = tensor.to(torch.float32).reshape(-1)
                if flat_scale.numel() == 3 * inner:
                    scales = [
                        flat_scale[i * inner:(i + 1) * inner].contiguous()
                        for i in range(3)
                    ]
                else:
                    scales = [_broadcast_scale(tensor, inner)] * 3
                for name, scale in zip(names, scales):
                    mapped[name + ".weight_scale"] = scale
                    stats["scales_broadcast"] += 1
            else:
                suffix = key.rsplit(".", 1)[1]  # "weight" (qkv is bias-free here)
                if tensor.shape[0] != 3 * inner:
                    raise ValueError(
                        f"{key}: fused qkv has {tensor.shape[0]} rows, expected {3 * inner}")
                for i, name in enumerate(names):
                    part = tensor[i * inner:(i + 1) * inner]
                    mapped[f"{name}.{suffix}"] = _maybe_cast(
                        part, _dit_target_dtype(f"{name}.{suffix}", compute_dtype, curve))
                stats["qkv_split"] += 1
            continue

        if is_scale:
            out_features = _header_shape(header, key[: -len(".weight_scale")] + ".weight")
            if out_features is None:
                raise ValueError(f"{key}: a weight_scale with no matching .weight tensor")
            # Keyed on the SOURCE tensor, not on the broadcast result: a
            # per-tensor scalar becomes a constant vector, for which the swap is
            # a numerical no-op but a misleading log line and pointless work.
            was_per_row = tensor.numel() > 1
            scale = _broadcast_scale(tensor, int(out_features[0]))
            if target.endswith(".ff.net.0.proj.weight_scale") and was_per_row:
                # THE SCALE IS INDEXED BY OUTPUT ROW, AND THIS LAYER'S ROWS MOVE.
                # `fc1` is stored `[gate; up]` and installed as `[up; gate]` (the
                # SwiGLU swap below), so a PER-ROW scale has to be swapped with
                # them or every row of every FFN in all 50 blocks is dequantized
                # with the other half's scale -- a load that succeeds, a
                # `verify_quantized_swap` that agrees, and a wrong model.
                # Dormant on the shipped checkpoint (its scales are per-tensor
                # scalars, so `_broadcast_scale` returns a constant vector and the
                # swap is a no-op), live the moment anyone re-quantizes this arch
                # per row.
                gate_scale, up_scale = scale.chunk(2, dim=0)
                scale = torch.cat([up_scale, gate_scale], dim=0).contiguous()
                stats["swiglu_scale_swapped"] += 1
            mapped[target] = scale
            stats["scales_broadcast"] += 1
            continue

        if target.endswith(".ff.net.0.proj.weight"):
            # Comfy [gate; up] -> diffusers SwiGLU [hidden; gate]. See the
            # per-row-scale swap above: any transform that reorders OUTPUT ROWS
            # has to be applied to `.weight_scale` as well.
            gate, up = tensor.chunk(2, dim=0)
            tensor = torch.cat([up, gate], dim=0)
            stats["swiglu_swapped"] += 1

        mapped[target] = _maybe_cast(tensor, _dit_target_dtype(target, compute_dtype, curve))

    return mapped, stats


def _broadcast_scale(scale: torch.Tensor, out_features: int) -> torch.Tensor:
    """A per-TENSOR ``weight_scale`` scalar as the ``(out_features,)`` vector.

    ``Fp8Linear`` registers ``weight_scale`` as ``(out_features,)`` and
    dequantizes with ``weight.to(dt) * weight_scale.to(dt).unsqueeze(1)``; the
    MiniMax-H3 distribution stores a scalar ``F32 []`` per tensor. Broadcasting
    it is exact (every row shares the scale) and it is what K0.1 measured.

    A scale that is ALREADY per-row is passed through UNCHANGED, which is correct
    only for a layer whose output rows this loader does not move. It is the
    CALLER's job to apply the same row permutation to the scale as to the weight:

    * ``ff.net.0.proj`` (``fc1``) exchanges its two row-halves, and the caller
      exchanges the scale's halves with them. Getting that wrong is SILENT --
      the shapes match, the swap count matches, the load is clean.
    * the fused ``qkv`` splits its rows three ways. That one cannot be silent:
      the per-row scale would arrive here with ``3 * inner`` elements against an
      ``inner`` target and the check below raises. Left as a loud failure rather
      than a split, because no such file exists to test a split against.
    """
    scale = scale.to(torch.float32).reshape(-1)
    if scale.numel() == out_features:
        return scale.contiguous()
    if scale.numel() != 1:
        raise ValueError(
            f"weight_scale has {scale.numel()} element(s); expected a per-tensor scalar or "
            f"one per output row ({out_features})")
    return scale.expand(out_features).contiguous()


def _maybe_cast(tensor: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if dtype is None or tensor.dtype == dtype or not tensor.is_floating_point():
        return tensor
    if tensor.dtype in (getattr(torch, "float8_e4m3fn", None),
                        getattr(torch, "float8_e5m2", None)):
        # An fp8 CODE. Only ``Fp8Linear`` may interpret it, and it does so with
        # the scale; casting it here would drop the scale silently.
        return tensor
    return tensor.to(dtype)


def _assert_guard_reached(state_dict: Dict[str, torch.Tensor], *, label: str, path: str) -> None:
    """Run the DECLARED-SEMANTICS refusal, and prove it ran.

    Design requirement (Phase 1): every H3 component load must reach the
    ``quantized_checkpoint_guard`` before a tensor is installed -- it is not
    enough that the guard exists. This is the single choke point, so it is called
    by all four component loaders and it fails loudly if the guard module cannot
    be imported, rather than degrading into a no-op the way a ``try/except``
    around the import would.
    """
    from core.models.common.quantized_checkpoint_guard import (
        refuse_unsupported_quant_semantics,
    )

    refuse_unsupported_quant_semantics(
        state_dict, arch="MiniMax-H3", path=path, label=label)


def _supported_int8_convrot_marker(
    key: str,
    marker: torch.Tensor,
    header: Dict[str, Any],
    *,
    path: str,
) -> Optional[Dict[str, int]]:
    """Validate the one ConvRot contract implemented by the H3 DiT loader."""
    from core.models.common.quantized_checkpoint_guard import decode_comfy_quant_marker

    parsed = decode_comfy_quant_marker(marker)
    if parsed != {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    }:
        return None
    layer = key[: -len(".comfy_quant")]
    weight = header.get(layer + ".weight")
    scale = header.get(layer + ".weight_scale")
    if not isinstance(weight, dict) or not isinstance(scale, dict):
        raise ValueError(f"{path}: ConvRot INT8 layer '{layer}' is missing weight or weight_scale")
    shape = weight.get("shape", [])
    if weight.get("dtype") != "I8" or not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{path}: ConvRot INT8 layer '{layer}' weight must be 2-D I8")
    out_features, in_features = (int(x) for x in shape)
    if in_features % 256:
        raise ValueError(
            f"{path}: ConvRot INT8 layer '{layer}' K={in_features} is not divisible by 256"
        )
    scale_shape = list(scale.get("shape", []))
    if scale.get("dtype") != "F32" or scale_shape not in ([out_features], [out_features, 1]):
        raise ValueError(
            f"{path}: ConvRot INT8 layer '{layer}' weight_scale must be F32 "
            f"[{out_features}] or [{out_features}, 1], got {scale.get('dtype')} {scale_shape}"
        )
    return {"convrot_groupsize": 256, "marker_numel": int(marker.numel())}


def _guard_component_file(
    path: str,
    *,
    label: str,
    allow_h3_int8_convrot: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """THE FIRST STATEMENT of every component builder.

    Returns ``(header without __metadata__, __metadata__)``.

    Runs the declared-semantics refusal on the MARKERS ALONE, before the builder
    does anything else at all -- before it reads a config, derives a geometry, or
    opens the weights for real.

    That ordering is the guard's own contract ("ahead of the census and ahead of
    any shape adaptation"), and it is not academic: a file this refuses is
    LAYOUT-compatible, so every other check passes it, and any check that runs
    FIRST will report its own unrelated complaint instead. Three concrete cases
    this fixes: an ``int8_convrot`` DiT with an unfamiliar head split was refused
    with a geometry message; a convrot VAE in a tree with no ``official/`` was
    refused with "MiniMax-H3 needs vae/config.json"; a convrot text encoder was
    refused by the double-mapping check. All three are true statements about the
    file and none of them is the reason that matters.

    Cost: one header read (a few hundred KB), and a ``safe_open`` only when the
    header actually declares ``.comfy_quant`` markers -- so the 48 GiB text
    encoder is not mapped at all on the ordinary path. ``.pre_quant_scale`` needs
    no bytes; a zero-element dtype proxy is enough for the guard to see it.
    """
    header = read_safetensors_header(path)
    metadata = header.pop("__metadata__", None) or {}

    probe: Dict[str, torch.Tensor] = {}
    marker_keys = [k for k in header if k.endswith(".comfy_quant")]
    for key, entry in header.items():
        if key.endswith(".pre_quant_scale"):
            dtype = _HEADER_DTYPES.get((entry or {}).get("dtype"), torch.float32) \
                if isinstance(entry, dict) else torch.float32
            probe[key] = torch.empty(0, dtype=dtype)
    if marker_keys:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in marker_keys:
                marker = handle.get_tensor(key)
                if (
                    allow_h3_int8_convrot
                    and _supported_int8_convrot_marker(
                        key, marker, header, path=path
                    ) is not None
                ):
                    continue
                probe[key] = marker
    if probe:
        _assert_guard_reached(probe, label=label, path=path)
    return header, metadata


def _int8_convrot_layers_from_markers(
    handle,
    header: Dict[str, Any],
    *,
    path: str,
) -> Dict[str, Dict[str, int]]:
    """Return source-layer configs for validated H3 ConvRot marker tensors."""
    layers: Dict[str, Dict[str, int]] = {}
    for key in header:
        if not key.endswith(".comfy_quant"):
            continue
        config = _supported_int8_convrot_marker(
            key, handle.get_tensor(key), header, path=path
        )
        if config is not None:
            layers[key[: -len(".comfy_quant")]] = config
    return layers


def _w4a8_layers_from_metadata(
    metadata: Dict[str, Any], header: Dict[str, Any], *, path: str,
) -> Dict[str, Dict[str, Any]]:
    """Validate Comfy's file-level quantization metadata without reading weights."""
    raw = metadata.get("_quantization_metadata")
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: malformed _quantization_metadata JSON") from exc
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(layers, dict):
        raise ValueError(f"{path}: _quantization_metadata must contain a 'layers' object")

    supported_passthrough = {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"}
    from core.models.common.quantized_checkpoint_guard import KNOWN_COMFY_QUANT_FIELDS

    w4a8: Dict[str, Dict[str, Any]] = {}
    for layer, value in layers.items():
        if not isinstance(layer, str) or not isinstance(value, dict):
            raise ValueError(f"{path}: every quantized layer entry must be an object")
        quant_format = value.get("format")
        if quant_format != "asym_w4a8_int8":
            unknown = sorted(set(value) - KNOWN_COMFY_QUANT_FIELDS)
            if unknown:
                raise ValueError(
                    f"{path}: layer '{layer}' has unknown quantization field(s): {unknown}")
            if value.get("convrot"):
                raise ValueError(
                    f"{path}: layer '{layer}' declares unsupported {quant_format!r} ConvRot "
                    "semantics; only asym_w4a8_int8 is supported by the MiniMax-H3 loader")
            if quant_format not in supported_passthrough:
                raise ValueError(f"{path}: layer '{layer}' has unknown quantization format {quant_format!r}")
            continue

        allowed = {"format", "convrot", "convrot_groupsize", "group_size"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"{path}: W4A8 layer '{layer}' has unknown field(s): {unknown}")
        group_size = int(value.get("group_size", 16))
        convrot_groupsize = int(value.get("convrot_groupsize", 256))
        if value.get("convrot") is not True:
            raise ValueError(f"{path}: W4A8 layer '{layer}' must declare convrot=true")

        weight = header.get(layer + ".weight")
        s_rel = header.get(layer + ".weight_s_rel")
        s_channel = header.get(layer + ".weight_s_channel")
        if not all(isinstance(item, dict) for item in (weight, s_rel, s_channel)):
            raise ValueError(f"{path}: W4A8 layer '{layer}' is missing weight/s_rel/s_channel")
        if weight.get("dtype") != "I8" or len(weight.get("shape", [])) != 2:
            raise ValueError(f"{path}: W4A8 layer '{layer}' weight must be packed 2-D I8")
        out_features, packed_k = (int(x) for x in weight["shape"])
        logical_k = packed_k * 2
        if (
            group_size < 4
            or convrot_groupsize <= 0
            or logical_k % 16
            or logical_k % group_size
            or logical_k % convrot_groupsize
            or (16 % group_size != 0 and group_size % 16 != 0)
        ):
            raise ValueError(
                f"{path}: W4A8 layer '{layer}' logical K={logical_k} is incompatible with "
                f"group_size={group_size}, convrot_groupsize={convrot_groupsize}")
        if list(s_rel.get("shape", [])) != [out_features, logical_k // group_size]:
            raise ValueError(f"{path}: W4A8 layer '{layer}' has an invalid weight_s_rel shape")
        if s_rel.get("dtype") not in {"F8_E4M3", "F32"}:
            raise ValueError(
                f"{path}: W4A8 layer '{layer}' weight_s_rel must be F8_E4M3 or F32")
        if list(s_channel.get("shape", [])) != [out_features]:
            raise ValueError(f"{path}: W4A8 layer '{layer}' has an invalid weight_s_channel shape")
        if s_channel.get("dtype") != "F32":
            raise ValueError(f"{path}: W4A8 layer '{layer}' weight_s_channel must be F32")
        codebook = header.get(layer + ".weight_codebook")
        if codebook is not None and list(codebook.get("shape", [])) != [16]:
            raise ValueError(f"{path}: W4A8 layer '{layer}' codebook must have shape [16]")
        if codebook is not None and codebook.get("dtype") != "F32":
            raise ValueError(f"{path}: W4A8 layer '{layer}' codebook must be F32")
        correction = header.get(layer + ".weight_correction")
        if correction is not None and list(correction.get("shape", [])) != [
            logical_k // group_size, out_features
        ]:
            raise ValueError(f"{path}: W4A8 layer '{layer}' has an invalid correction shape")
        if correction is not None and correction.get("dtype") not in {"F16", "BF16", "F32"}:
            raise ValueError(
                f"{path}: W4A8 layer '{layer}' correction must be F16, BF16 or F32")
        w4a8[layer] = {
            "group_size": group_size,
            "convrot_groupsize": convrot_groupsize,
        }

    sidecar_modules = {
        key[: -len(suffix)]
        for key in header
        for suffix in _W4A8_SUFFIXES[1:]
        if key.endswith(suffix)
    }
    undeclared = sorted(sidecar_modules - set(w4a8))
    if undeclared:
        raise ValueError(
            f"{path}: W4A8 sidecars exist without matching metadata (first 5: {undeclared[:5]})")
    return w4a8


def _mapped_w4a8_layer_configs(
    source_layers: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for source, config in source_layers.items():
        if source.endswith(".attn.qkv_proj"):
            target = _rename_dit_key(source + ".weight")
            stem = target.split(".attn.qkv_proj.")[0] + ".attn."
            for suffix in ("to_q", "to_k", "to_v"):
                mapped[stem + suffix] = dict(config)
        else:
            mapped[_rename_dit_key(source + ".weight")[:-len(".weight")]] = dict(config)
    return mapped


def _mapped_int8_convrot_layer_configs(
    source_layers: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Expand fused source QKV declarations to the three live projections."""
    mapped: Dict[str, Dict[str, int]] = {}
    for source, config in source_layers.items():
        if source.endswith(".attn.qkv_proj"):
            target = _rename_dit_key(source + ".weight")
            stem = target.split(".attn.qkv_proj.")[0] + ".attn."
            for suffix in ("to_q", "to_k", "to_v"):
                mapped[stem + suffix] = dict(config)
        else:
            mapped[_rename_dit_key(source + ".weight")[:-len(".weight")]] = dict(config)
    return mapped


def _dit_quantization_policy(model: nn.Module) -> int:
    """Pin every ``Fp8Linear`` of the DiT to the DEQUANT path. Returns the count.

    Two reasons, both measured, and the second is not optional:

    * 50 of the 200 quantized Linears -- exactly the 50 ``mlp.fc2`` -- carry
      ``{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}``, i.e.
      the writer declares that their product must NOT be computed in fp8. They
      are also exactly the 50 that carry no ``input_scale``.
    * The remaining 150 do carry an ``input_scale`` that this repo's
      ``Fp8Linear`` does not read (it quantizes activations dynamically instead),
      so running them through the scaled GEMM would apply a different
      activation-scaling contract than the file declares.

    ``disable_scaled_mm`` is the AUTHORITATIVE per-module gate (it outranks the
    ``SUSHI_FP8_SCALED_MM`` env flag and grad mode), so calling it on the whole
    DiT makes the prohibition hold for both, whatever a future
    ``ARCH_QUANT_POLICY`` entry says. Phase 4 registers the policy in the
    quantization tables; this is the load-time half, and it must not be removed
    when that lands.
    """
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm

    return disable_scaled_mm(model, label="MiniMax-H3 transformer")


def _swap_minimax_h3_quantized_linears(model: nn.Module, state_dict: Dict[str, torch.Tensor],
                                       dtype: torch.dtype) -> int:
    """Replace the DiT's ``nn.Linear``s that have a quantized saved weight. Count.

    INT8 and e4m3 are detected INDEPENDENTLY and both swaps run, for the reason
    every sibling helper does it (``ltx2/loader._swap_ltx2_quantized_linears``,
    ``model_loader._swap_flux2_quantized_linears``,
    ``anima_loader._swap_quantized_linears``): the offline ``--format int8``
    tool emits a MIXED file on purpose, and each detector gates on the weight
    DTYPE as well as the shared ``.weight_scale`` suffix, so neither can claim
    the other's layers and the call order does not matter. The released
    MiniMax-H3 checkpoints are pure e4m3, so today only the second swap fires --
    the int8 half is here because a SushiUI-exported artifact of this arch may
    not be.

    MiniMax-H3 needs no prefix argument: by the time this runs, ``_map_dit_state_dict``
    has already rewritten every key to a module path.

    The returned count is NOT decorative -- the caller compares it against the
    header census (``verify_quantized_swap``) and refuses the load when they
    disagree, because a quantized layer this helper did not take is a layer whose
    fp8 CODES ``load_state_dict(assign=True)`` would install into a plain
    parameter without a word.
    """
    from core.models.ideogram4.vendor.fp8_linear import is_fp8_state_dict, swap_linears_to_fp8
    from core.models.ideogram4.vendor.int8_linear import is_int8_state_dict, swap_linears_to_int8

    swapped = 0
    if is_int8_state_dict(state_dict):
        swapped += swap_linears_to_int8(model, state_dict, compute_dtype=dtype)
    if is_fp8_state_dict(state_dict):
        swapped += swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
    return swapped


def _build_transformer(dit_path: str, torch_dtype: torch.dtype,
                       official_dir: Optional[str]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate the vendored transformer and load ``dit_path`` into it."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from core.models.common.quantized_checkpoint_guard import (
        quantized_state_dict_report, scaled_quantization_report, verify_quantized_swap,
    )
    from .vendor import MiniMaxH3Transformer3DModel

    # The guard must precede geometry synthesis so an unsupported quantized
    # contract reports its actual incompatibility.
    header, metadata = _guard_component_file(
        dit_path, label="transformer", allow_h3_int8_convrot=True
    )
    w4a8_source_layers = _w4a8_layers_from_metadata(metadata, header, path=dit_path)
    w4a8_layer_configs = _mapped_w4a8_layer_configs(w4a8_source_layers)
    if w4a8_layer_configs:
        from core.models.common.w4a8_linear import require_w4a8_runtime

        require_w4a8_runtime()

    config = _synthesize_transformer_config(header, official_dir)
    curve = config.get("adaln_curve_grid") is not None
    print(f"[MiniMaxH3Loader] transformer geometry (synthesised from the header): "
          f"{config['num_layers']} blocks, hidden {config['hidden_size']}, ffn {config['ffn_dim']}, "
          f"text_dim {config['text_dim']}, in_channels {config['in_channels']}, "
          f"variant={'AdaLN-curve (pruned)' if curve else 'full modulation'}"
          + (f", grid {config['adaln_curve_grid']} x {config['time_embed_dim']}" if curve else ""))

    with safe_open(dit_path, framework="pt", device="cpu") as handle:
        int8_convrot_source_layers = _int8_convrot_layers_from_markers(
            handle, header, path=dit_path
        )
        int8_convrot_layer_configs = _mapped_int8_convrot_layer_configs(
            int8_convrot_source_layers
        )
        if int8_convrot_layer_configs:
            from core.models.common.convrot_int8_linear import require_convrot_int8_runtime

            require_convrot_int8_runtime()
        state_dict, stats = _map_dit_state_dict(
            handle,
            header,
            config,
            torch_dtype,
            w4a8_layers=w4a8_source_layers,
            int8_convrot_layers=int8_convrot_source_layers,
        )

        # The early header guard validated every supported ConvRot marker. Keep
        # those markers as live module state, while every other declaration
        # still passes through the generic refusal before any tensor is installed.
        guard_state_dict = {
            key: value for key, value in state_dict.items()
            if not (
                key.endswith(".comfy_quant")
                and key[: -len(".comfy_quant")] in int8_convrot_layer_configs
            )
        }
        _assert_guard_reached(guard_state_dict, label="transformer", path=dit_path)

        w4a8_prefixes = tuple(name + "." for name in w4a8_layer_configs)
        int8_convrot_prefixes = tuple(name + "." for name in int8_convrot_layer_configs)
        scaled_state_dict = {
            key: value for key, value in state_dict.items()
            if (not w4a8_prefixes or not key.startswith(w4a8_prefixes))
            and (not int8_convrot_prefixes or not key.startswith(int8_convrot_prefixes))
        }
        census = quantized_state_dict_report(
            scaled_state_dict, arch="MiniMax-H3", path=dit_path, label="transformer")
        report = scaled_quantization_report(
            census, arch="MiniMax-H3", path=dit_path, label="transformer")

        # Plain provenance markers have served their purpose. ConvRot modules
        # retain theirs so a state_dict/export cannot lose the rotation contract.
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.endswith(".comfy_quant")
            or key[: -len(".comfy_quant")] in int8_convrot_layer_configs
        }

        with init_empty_weights():
            model = MiniMaxH3Transformer3DModel(**config)

        swapped = 0
        if int8_convrot_layer_configs:
            from core.models.common.convrot_int8_linear import swap_linears_to_convrot_int8

            convrot_swapped = swap_linears_to_convrot_int8(
                model, state_dict, int8_convrot_layer_configs, torch_dtype
            )
            if convrot_swapped != len(int8_convrot_layer_configs):
                raise RuntimeError(
                    f"MiniMax-H3 ConvRot metadata mapped "
                    f"{len(int8_convrot_layer_configs)} Linear(s), but only "
                    f"{convrot_swapped} module(s) were replaced"
                )
            swapped += convrot_swapped
        if w4a8_layer_configs:
            from core.models.common.w4a8_linear import swap_linears_to_w4a8

            w4a8_swapped = swap_linears_to_w4a8(
                model, state_dict, w4a8_layer_configs, torch_dtype)
            if w4a8_swapped != len(w4a8_layer_configs):
                raise RuntimeError(
                    f"MiniMax-H3 W4A8 metadata mapped {len(w4a8_layer_configs)} Linear(s), "
                    f"but only {w4a8_swapped} module(s) were replaced")
            swapped += w4a8_swapped
        if report is not None:
            scaled_swapped = _swap_minimax_h3_quantized_linears(model, state_dict, torch_dtype)
            verify_quantized_swap(report, scaled_swapped, arch="MiniMax-H3", path=dit_path,
                                  label="transformer")
            swapped += scaled_swapped

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    if unexpected:
        raise RuntimeError(
            f"the MiniMax-H3 transformer checkpoint ({dit_path}) produced {len(unexpected)} "
            f"unexpected key(s) after mapping (first 5: {sorted(unexpected)[:5]}). The rename "
            f"table and the checkpoint disagree; refusing rather than loading a partial model.")
    if missing:
        # The model was built on the META device, so ``assign=True`` is the only
        # thing that gives a parameter a real tensor: a missing key leaves a meta
        # tensor in a live model that detonates at the first forward, minutes
        # later and somewhere else.
        raise RuntimeError(
            f"the MiniMax-H3 transformer checkpoint ({dit_path}) is missing {len(missing)} key(s) "
            f"required by the model built from the synthesised config (first 5: "
            f"{sorted(missing)[:5]}).")
    stranded = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)]
    if stranded:
        raise RuntimeError(
            f"the MiniMax-H3 transformer from {dit_path} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}).")

    print(f"[MiniMaxH3Loader] transformer: {stats['qkv_split']} fused qkv split (contiguous, NOT "
          f"de-interleaved), {stats['swiglu_swapped']} SwiGLU half-swap(s), "
          f"{stats['scales_broadcast']} weight_scale(s) broadcast "
          f"({stats['swiglu_scale_swapped']} per-row scale(s) half-swapped with their weight), "
          f"{stats['markers']} quant marker(s) validated "
          f"(ConvRot declarations retained as module state), "
          f"{stats['input_scale_dropped']} input_scale(s) dropped (W8A8 is off for this arch), "
          f"{stats['dropped']} recomputable buffer(s) dropped")
    if swapped:
        pinned = _dit_quantization_policy(model)
        print(f"[MiniMaxH3Loader] {swapped} weight-only quantized Linear(s) kept quantized; "
              f"{pinned} FP8 Linear(s) pinned to the dequant path")

    model.eval().requires_grad_(False)
    return model, config


# ---------------------------------------------------------------------------
# Video VAE
# ---------------------------------------------------------------------------

_VIDEO_VAE_DROPPED = ("decoder.mask_token", "latents_mean", "latents_std")


def _reorder_interleaved_qkv(weight: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """Per-head-interleaved ``[h0 q k v | h1 q k v | ...]`` -> ``[q_all|k_all|v_all]``.

    ONLY for the video VAE decoder. The DiT's fused qkv in the same distribution
    is already contiguous and must NOT go through this (module docstring, point
    2).
    """
    grouped = weight.reshape(heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat([t.reshape(heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0)


def _rename_video_vae_key(key: str) -> str:
    out = key
    if out.startswith("encoder.down."):
        level, rest = out[len("encoder.down."):].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        out = f"encoder.down_blocks.{level}.{rest}"
    out = out.replace("decoder.x_embedder.", "decoder.proj_in.")
    out = out.replace(".attn.to_out.", ".attn.to_out.0.")
    out = out.replace(".ff.w1.", ".ff.net.0.proj.")
    out = out.replace(".ff.w2.", ".ff.net.2.")
    return out


def _build_video_vae(vae_path: str, official_dir: Optional[str], torch_dtype: torch.dtype):
    """The 24-channel causal video VAE with the ViT decoder.

    fp16 is the design point (66.72 dB against a full-fp32 decode, MEASURED),
    which halves 10.4 GB to 5.2 GB resident; upstream's ``_keep_in_fp32_modules``
    would refuse that through ``from_pretrained``, so the cast is explicit here.
    ``torch_dtype`` is a parameter and not a constant precisely because Phase 2
    owes an fp16-vs-fp32 decode A/B on a real clip.

    The module is built on the META device and the mapped tensors are ASSIGNED
    into it. Constructing it materialised would allocate 10.4 GB of float32
    parameters, cast the fp16 file up into them, then cast the whole module back
    down -- numerically identical, and the entire peak of this loader's VAE step.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict

    from .vendor import AutoencoderKLMiniMaxH3

    # THE GUARD FIRST -- ahead of the config read, which raises its own
    # "needs vae/config.json" on a tree with no official/ and would otherwise
    # answer a convrot file with that instead of with the reason that matters.
    _guard_component_file(vae_path, label="video VAE")

    config = _read_component_config(official_dir, "vae", vae_path)
    heads = int(config["decoder_num_attention_heads"])
    head_dim = int(config["decoder_attention_head_dim"])
    inner = heads * head_dim

    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(vae_path, framework="pt", device="cpu") as handle:
        raw = {k: handle.get_tensor(k) for k in handle.keys()}
    # The guard, on the RAW dict, before anything is transformed or installed.
    refuse_quantized_state_dict(raw, arch="MiniMax-H3", path=vae_path, label="video VAE")
    _assert_guard_reached(raw, label="video VAE", path=vae_path)

    for key, tensor in raw.items():
        if key in _VIDEO_VAE_DROPPED:
            continue
        if ".attn.to_qkv." in key:
            # TRAP: this one IS per-head interleaved (the opposite of the DiT).
            tensor = _reorder_interleaved_qkv(tensor.to(torch_dtype), heads, head_dim)
            prefix, suffix = key.split(".attn.to_qkv.")
            q, k, v = tensor.split(inner, dim=0)
            state_dict[f"{prefix}.attn.to_q.{suffix}"] = q.contiguous()
            state_dict[f"{prefix}.attn.to_k.{suffix}"] = k.contiguous()
            state_dict[f"{prefix}.attn.to_v.{suffix}"] = v.contiguous()
            continue
        target = _rename_video_vae_key(key)
        tensor = tensor.to(torch_dtype) if tensor.is_floating_point() else tensor
        if ".ff.net.0.proj." in target:
            gate, up = tensor.chunk(2, dim=0)
            tensor = torch.cat([up, gate], dim=0).contiguous()
        state_dict[target] = tensor
    del raw

    with init_empty_weights():
        vae = AutoencoderKLMiniMaxH3(**config)
    missing, unexpected = vae.load_state_dict(state_dict, strict=True, assign=True)
    if missing or unexpected:  # pragma: no cover - strict=True raises first
        raise RuntimeError(
            f"MiniMax-H3 video VAE state_dict mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}")
    # The parameters are already at ``torch_dtype`` (the state dict was cast on
    # the way in); this catches the real, constructor-built BUFFERS, exactly as
    # the materialised build did.
    vae = vae.to(dtype=torch_dtype).eval().requires_grad_(False)
    stranded = [n for n, t in list(vae.named_parameters()) + list(vae.named_buffers())
                if getattr(t, "is_meta", False)]
    if stranded:
        raise RuntimeError(
            f"the MiniMax-H3 video VAE from {vae_path} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}); it would fail at the first "
            f"encode, not here.")

    # PINNED tiling policy (K0.5 supplementary: the flags change the output, they
    # do not just change the memory profile).
    #
    # PHASE 2 NOTE: K0.5's bitwise-vs-reference result was obtained with tiling
    # OFF on both sides. The SHIPPED (tiled) path therefore has no numerical
    # oracle behind it, and the fp16-vs-fp32 A/B must hold this policy fixed or
    # it measures tiling instead of precision.
    if MINIMAX_H3_VAE_TILING_POLICY["enabled"]:
        vae.enable_tiling(
            tile_sample_min_height=MINIMAX_H3_VAE_TILING_POLICY["tile_sample_min_height"],
            tile_sample_min_width=MINIMAX_H3_VAE_TILING_POLICY["tile_sample_min_width"],
            tile_sample_min_overlap_height=MINIMAX_H3_VAE_TILING_POLICY["tile_sample_min_overlap_height"],
            tile_sample_min_overlap_width=MINIMAX_H3_VAE_TILING_POLICY["tile_sample_min_overlap_width"],
        )
    else:
        vae.disable_tiling()
    return vae, config


# ---------------------------------------------------------------------------
# Audio VAE
# ---------------------------------------------------------------------------

def _build_audio_vae(vae_path: str, official_dir: Optional[str], torch_dtype: torch.dtype):
    """The 32-channel mono audio autoencoder (32 kHz, 40 Hz latent rate).

    TRAP: the original MiniMax checkpoint carries ``weight_norm``'s
    ``weight_g``/``weight_v`` (which the vendored class reproduces) while the
    Comfy file ships the FOLDED plain ``weight``. Without folding the
    parameterization out first this is 268 missing / 134 unexpected keys.

    Kept in float32: the file is 0.6 GB and this decoder runs once per
    generation, so there is nothing to buy by halving it.
    """
    from safetensors.torch import load_file

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict

    from .vendor import AutoencoderKLMiniMaxH3Audio

    # THE GUARD FIRST, for the same reason as the video VAE above.
    _guard_component_file(vae_path, label="audio VAE")

    config = _read_component_config(official_dir, "audio_vae", vae_path)
    state_dict = load_file(vae_path)
    refuse_quantized_state_dict(state_dict, arch="MiniMax-H3", path=vae_path, label="audio VAE")
    _assert_guard_reached(state_dict, label="audio VAE", path=vae_path)
    state_dict.pop("latents_mean", None)
    state_dict.pop("latents_std", None)

    vae = AutoencoderKLMiniMaxH3Audio(**config)
    folded = 0
    for module in vae.modules():
        hooks = getattr(module, "_forward_pre_hooks", {}) or {}
        if any(h.__class__.__name__ == "WeightNorm" for h in hooks.values()):
            torch.nn.utils.remove_weight_norm(module)
            folded += 1
    print(f"[MiniMaxH3Loader] audio VAE: weight-norm folded out of {folded} module(s) "
          f"(the Comfy file ships the pre-folded plain `weight`)")

    missing, unexpected = vae.load_state_dict(state_dict, strict=True)
    if missing or unexpected:  # pragma: no cover - strict=True raises first
        raise RuntimeError(
            f"MiniMax-H3 audio VAE state_dict mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}")
    return vae.to(dtype=torch_dtype).eval().requires_grad_(False), config


def _read_component_config(official_dir: Optional[str], component: str,
                           weight_path: str) -> Dict[str, Any]:
    """``official/<component>/config.json``, minus its ``_``-prefixed entries.

    Both VAE configs are variant-independent (unlike the transformer's), so they
    are read rather than synthesised -- and their ``latents_mean``/``latents_std``
    are the fp32 originals, which is why they must be preferred over the fp16
    copies inside the weight file (max abs diff 8.4e-4).
    """
    if official_dir:
        candidate = os.path.join(official_dir, component, "config.json")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as fh:
                return {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    raise FileNotFoundError(
        f"MiniMax-H3 needs {component}/config.json to build its {component} for "
        f"{weight_path}, and no config-only tree was found next to the weights. Expected one of "
        f"{'/'.join(_OFFICIAL_DIR_NAMES)}/ (a directory whose model_index.json declares "
        f"{MINIMAX_H3_PIPELINE_CLASS}) beside diffusion_models/.")


# ---------------------------------------------------------------------------
# Text encoder (Qwen3-VL 32B, truncated to 50 decoder layers)
# ---------------------------------------------------------------------------

def _rewrite_te_key(key: str) -> str:
    """The whole text-encoder adapter: three prefix rules.

    The file uses the older flat Qwen3-VL naming; the installed transformers
    expects the language model and the vision tower under ``model.``.
    """
    if key.startswith("model.layers."):
        return "model.language_model.layers." + key[len("model.layers."):]
    if key.startswith("model.embed_tokens."):
        return "model.language_model.embed_tokens." + key[len("model.embed_tokens."):]
    if key.startswith("visual."):
        return "model.visual." + key[len("visual."):]
    return key


# The two keys the truncated read genuinely does not have. ``lm_head`` is a
# generation head this never runs, and ``norm`` is the FINAL norm the file
# deliberately omits -- its declared output is the UNNORMALISED hidden state
# after layer 50.
_TE_EXPECTED_MISSING = frozenset({"lm_head.weight", "model.language_model.norm.weight"})

# Weak reference to the text encoder this process built last, keyed by file path.
# Weak so it never keeps one alive; see `_refuse_double_mapping` for what it is
# for. Not thread-safe by design -- a model load is already serialised by
# `PipelineManager._load_model_lock`, and a stale entry only costs one
# `gc.collect()`.
_LIVE_TEXT_ENCODER: Dict[str, Any] = {}


def _refuse_double_mapping(te_path: str) -> None:
    """Refuse to map the 48 GiB text encoder twice in one process.

    MEASURED, twice, deterministically: reloading MiniMax-H3 while ANYTHING still
    holds a reference to the previous component dict maps the same 48 GiB file a
    second time and the process dies -- a hard crash (Windows), not an exception,
    so there is no traceback and no message. K0.7 hit the same wall from the other
    direction with ``os error 1455`` ("the paging file is too small").

    In production the previous dict is dropped by ``PipelineManager``'s teardown
    branch before the new load starts, so this never fires. It exists because the
    failure it converts is unattributable: a future keep-hot entry, a cached
    pipeline handle or a debugger reference would take the backend down with no
    evidence of why. A ``gc.collect()`` first, because the ordinary case is a
    dead-but-uncollected cycle rather than a live holder.
    """
    import gc

    ref = _LIVE_TEXT_ENCODER.get(te_path)
    if ref is None or ref() is None:
        return
    gc.collect()
    if ref() is None:
        return
    raise RuntimeError(
        f"a MiniMax-H3 text encoder built from {te_path} is STILL ALIVE in this process. Its "
        f"48 GiB of weights are memory-mapped from that file, and mapping it a second time "
        f"terminates the process outright (measured; Windows reports 'the paging file is too "
        f"small' when it reports anything at all). Something is holding a reference to the "
        f"previous minimax_h3 component dict across this load -- the pipeline's own teardown "
        f"branch drops it, so look for a cache, a keep-hot resident set, or a debugger handle. "
        f"Refusing here so the cause is visible instead of the backend simply vanishing.")


def _build_text_encoder(te_path: str, official_dir: Optional[str]):
    """Build the truncated Qwen3-VL and install the file's tensors BY REFERENCE.

    Three properties are load-bearing and each is measured (K0.7):

    * ``load_state_dict(assign=True)`` installs the ``safe_open`` tensors, which
      are MEMORY-MAPPED. Nothing here copies or casts them, so a 48 GiB encoder
      costs ~no resident memory until something reads it -- and Phase 2's
      ``torch.func.functional_call`` streaming keeps it that way. Writing the CPU
      weights back (``layer.to("cpu", bf16)``) detaches every parameter from the
      mapping: 73.08 GB peak RSS and pagefile growth, against 49.82 GB flat.
    * Exactly ONE ``safe_open`` of this file is alive at a time; a second
      concurrent mapping aborted a K0.7 process with Windows ``os error 1455``.
    * The two missing keys are asserted, not tolerated: anything else missing
      would leave a META tensor in a live model.

    The dtype stays the file's bf16. It is NOT cast here, because a cast is a
    copy and a copy is the failure above.
    """
    import weakref

    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict

    # THE GUARD FIRST -- ahead of the double-mapping check and both config reads.
    # The co-distributed `qwen3vl_32b_minimax_h3_int8_convrot` / `_nvfp4_awq`
    # text encoders are exactly what it exists for, and a reload that also
    # tripped `_refuse_double_mapping` would otherwise answer with THAT.
    header, metadata = _guard_component_file(te_path, label="text encoder")

    _refuse_double_mapping(te_path)

    if not official_dir:
        raise FileNotFoundError(
            f"MiniMax-H3 needs text_encoder/config.json to build the Qwen3-VL text encoder for "
            f"{te_path}; no config-only tree was found beside the weights.")
    cfg_path = os.path.join(official_dir, "text_encoder", "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        raw_config = {k: v for k, v in json.load(fh).items() if k != "architectures"}

    num_layers = None
    try:
        declared = json.loads(metadata.get("minimax_h3_te", "{}"))
        num_layers = int(declared.get("num_hidden_layers")) if declared.get("num_hidden_layers") else None
    except Exception:
        num_layers = None
    if num_layers is None:
        # Derive it from the file rather than assuming the published 50: the
        # truncation is a property of THIS file, and a differently-truncated one
        # must not be built at the wrong depth.
        num_layers = 1 + max(
            (int(k.split(".")[2]) for k in header if k.startswith("model.layers.")), default=-1)
    if num_layers <= 0:
        raise ValueError(f"could not determine the decoder depth of {te_path}")

    config = Qwen3VLConfig(**raw_config)
    config.text_config.num_hidden_layers = num_layers
    print(f"[MiniMaxH3Loader] text encoder: Qwen3-VL truncated to {num_layers} decoder layer(s) "
          f"(the file's own declared output is the unnormalised hidden state after the last one)")

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    with safe_open(te_path, framework="pt", device="cpu") as handle:
        state_dict = {_rewrite_te_key(k): handle.get_tensor(k) for k in header}
        refuse_quantized_state_dict(
            state_dict, arch="MiniMax-H3", path=te_path, label="text encoder")
        _assert_guard_reached(state_dict, label="text encoder", path=te_path)
        result = model.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict

    unexpected = sorted(result.unexpected_keys)
    missing = sorted(result.missing_keys)
    if unexpected:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder ({te_path}) produced {len(unexpected)} unexpected "
            f"key(s) (first 5: {unexpected[:5]}); the prefix rewrite and the file disagree.")
    if set(missing) - _TE_EXPECTED_MISSING:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder ({te_path}) is missing key(s) beyond the two the "
            f"truncated read expects ({sorted(_TE_EXPECTED_MISSING)}): "
            f"{sorted(set(missing) - _TE_EXPECTED_MISSING)[:5]}. Those parameters were built on "
            f"the meta device and would detonate at the first forward.")

    # The two absent modules are REPLACED rather than left holding meta tensors:
    # neither is used by the layer-N hidden-state read, and a meta parameter in a
    # live model fails far from here.
    model.lm_head = None
    model.model.language_model.norm = nn.Identity()

    stranded = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)]
    if stranded:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder from {te_path} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}).")

    model.eval().requires_grad_(False)
    _LIVE_TEXT_ENCODER[te_path] = weakref.ref(model)
    return model, config


def _load_tokenizer_and_processor(official_dir: Optional[str]):
    """``(tokenizer, processor)``; either may be ``None`` with a printed reason.

    The processor is what feeds the vision tower for ref2va / fl2va reference
    images (K0.7 verified its placeholder count against the tower's output rows).
    Neither is fatal at load time.
    """
    tokenizer = processor = None
    if not official_dir:
        return None, None
    try:
        from transformers import AutoTokenizer

        tok_dir = os.path.join(official_dir, "tokenizer")
        if os.path.isdir(tok_dir):
            tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    except Exception as exc:
        print(f"[MiniMaxH3Loader] WARNING: tokenizer load failed: {exc}")
    try:
        from transformers import AutoProcessor

        proc_dir = os.path.join(official_dir, "processor")
        if os.path.isdir(proc_dir):
            processor = AutoProcessor.from_pretrained(proc_dir)
    except Exception as exc:
        print(f"[MiniMaxH3Loader] WARNING: processor load failed: {exc}")
    return tokenizer, processor


def _load_schedulers(official_dir: Optional[str]):
    """``(video scheduler, audio scheduler)`` -- shift 12.0 / 3.0.

    Two configs of the SAME class. The sampler runs ONE Euler integrator on the
    video sigma and scales the audio velocity by ``d(sigma_a)/d(sigma_v)``
    (Phase 2), so both grids are needed even though there is a single loop.
    """
    from .vendor import MiniMaxH3Scheduler

    def one(name: str, fallback_shift: float):
        if official_dir:
            path = os.path.join(official_dir, name, "scheduler_config.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    cfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
                return MiniMaxH3Scheduler(**cfg)
        print(f"[MiniMaxH3Loader] no {name}/scheduler_config.json; using shift={fallback_shift}")
        return MiniMaxH3Scheduler(shift=fallback_shift)

    return one("scheduler", 12.0), one("audio_scheduler", 3.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# The video VAE's shipped compute dtype. fp16 is the design point -- 66.72 dB
# against a full-fp32 decode (MEASURED, on random pixels) for half the resident
# footprint, 5.2 GB instead of 10.4 GB, which matters against a 48 GB envelope
# holding a 36-layer ViT decoder. Named here rather than inlined so the fp32 path
# is reachable: design sec.5 step 3 and Phase 2 both owe an fp16-vs-fp32 A/B on a
# REAL clip before that number is repeated anywhere user-visible.
MINIMAX_H3_VIDEO_VAE_DTYPE = torch.float16


def load_minimax_h3_from_path(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *,
    load_text_encoder: bool = True,
    video_vae_dtype: Optional[torch.dtype] = None,
) -> dict:
    """Load MiniMax-H3 from its ComfyUI-style flat tree (or MiniMax's own dir).

    Returns the component dict ``PipelineManager.load_model()`` consumes. Every
    component stays on the CPU; nothing is staged to GPU here (Phase 2 owns the
    strictly sequential text-encode -> denoise -> decode phases, because no two
    of these components fit in 48 GB together).

    ``load_text_encoder=False`` skips the 48 GiB encoder -- for a caller that
    only needs the DiT/VAE geometry (a probe, a test). It is NOT a memory
    optimization for generation: the encoder is memory-mapped and costs almost
    nothing until it is read.

    ``video_vae_dtype`` overrides ``MINIMAX_H3_VIDEO_VAE_DTYPE`` (fp16). Pass
    ``torch.float32`` for the fp16-vs-fp32 decode A/B design sec.5 step 3 owes --
    holding the tiling policy fixed, or the A/B measures tiling instead of
    precision. The dtype actually used is reported back as ``video_vae_dtype`` in
    the component dict so a measurement cannot mislabel itself.
    """
    layout = detect_minimax_h3_layout(model_path)
    if layout is None:
        raise ValueError(
            f"MiniMax-H3 model layout not found at {model_path!r}. Expected a directory holding "
            f"diffusion_models/ + vae/ + text_encoders/ (plus MiniMax's config-only official/ "
            f"tree), a DiT .safetensors inside such a diffusion_models/, or a directory whose "
            f"model_index.json declares {MINIMAX_H3_PIPELINE_CLASS}.")

    missing = [name for name in ("dit", "vae", "audio_vae") if not layout.get(name)]
    if load_text_encoder and not layout.get("text_encoder"):
        missing.append("text_encoder")
    if missing:
        raise ValueError(
            f"MiniMax-H3 at {layout['root']!r} is missing the following component file(s): "
            f"{', '.join(missing)}. Expected diffusion_models/{MINIMAX_H3_DIT_PATTERNS[0]}, "
            f"vae/{MINIMAX_H3_VIDEO_VAE_PATTERNS[0]}, vae/{MINIMAX_H3_AUDIO_VAE_PATTERNS[0]} and "
            f"text_encoders/{MINIMAX_H3_TE_PATTERNS[0]}.")

    official = layout["official"]
    # Checked UP FRONT, not where it is first needed. The transformer is the only
    # component that can be built without the config tree (it synthesises its
    # geometry from the checkpoint header), so without this the first hard
    # failure -- "MiniMax-H3 needs vae/config.json" -- arrives AFTER 21 GB of DiT
    # has been mapped, swapped and strict-loaded, which is a minute of work and a
    # peak of memory spent to reach a message that was knowable immediately.
    # Both VAEs need their config for geometry AND for the fp32
    # latents_mean/latents_std, and the text encoder needs it for the Qwen3-VL
    # config, the tokenizer and the processor; none of them is optional.
    needed = [c for c in ("vae", "audio_vae") if c] + (["text_encoder"] if load_text_encoder else [])
    if official is None:
        raise FileNotFoundError(
            f"MiniMax-H3 at {layout['root']!r} has the weight files but no config-only tree, and "
            f"{'/'.join(needed)} cannot be built without one. Expected a directory named one of "
            f"{'/'.join(_OFFICIAL_DIR_NAMES)} beside diffusion_models/, whose model_index.json "
            f"declares {MINIMAX_H3_PIPELINE_CLASS} and which holds vae/config.json, "
            f"audio_vae/config.json, text_encoder/config.json, tokenizer/ and processor/.")
    missing_cfg = [f"{c}/config.json" for c in needed
                   if not os.path.isfile(os.path.join(official, c, "config.json"))]
    if missing_cfg:
        raise FileNotFoundError(
            f"MiniMax-H3's config tree at {official!r} is missing {', '.join(missing_cfg)}. "
            f"Those carry the component geometry and the fp32 latents_mean/latents_std vectors; "
            f"they are not optional and are not derivable from the weight files.")
    print(f"[MiniMaxH3Loader] root:         {layout['root']}")
    print(f"[MiniMaxH3Loader] DiT:          {layout['dit']} (variant={layout['variant']})")
    print(f"[MiniMaxH3Loader] video VAE:    {layout['vae']}")
    print(f"[MiniMaxH3Loader] audio VAE:    {layout['audio_vae']}")
    print(f"[MiniMaxH3Loader] text encoder: {layout['text_encoder']}")
    print(f"[MiniMaxH3Loader] configs:      {official}")

    transformer, transformer_config = _build_transformer(layout["dit"], torch_dtype, official)
    # fp16 for the video VAE (see MINIMAX_H3_VIDEO_VAE_DTYPE), float32 for the
    # small audio one -- 0.6 GB, decoded once per generation, nothing to buy.
    vae_dtype = video_vae_dtype or MINIMAX_H3_VIDEO_VAE_DTYPE
    vae, vae_config = _build_video_vae(layout["vae"], official, vae_dtype)
    audio_vae, audio_vae_config = _build_audio_vae(layout["audio_vae"], official, torch.float32)

    text_encoder = text_encoder_config = None
    if load_text_encoder:
        text_encoder, text_encoder_config = _build_text_encoder(layout["text_encoder"], official)
    tokenizer, processor = _load_tokenizer_and_processor(official)
    scheduler, audio_scheduler = _load_schedulers(official)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[MiniMaxH3Loader] Loaded MiniMax-H3 components (CPU-resident; no sampler wired yet).")

    return {
        "type": "minimax_h3",
        "is_video": True,
        "variant": layout["variant"],
        "transformer": transformer,
        "transformer_config": transformer_config,
        "vae": vae,
        "vae_config": vae_config,
        "audio_vae": audio_vae,
        "audio_vae_config": audio_vae_config,
        "text_encoder": text_encoder,
        "text_encoder_config": text_encoder_config,
        "tokenizer": tokenizer,
        "processor": processor,
        "scheduler": scheduler,
        "audio_scheduler": audio_scheduler,
        # Geometry, so nothing downstream re-derives it.
        "latent_channels": MINIMAX_H3_LATENT_CHANNELS,
        "audio_latent_channels": MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        "vae_scale_factor_spatial": MINIMAX_H3_VAE_SPATIAL_COMPRESSION,
        "vae_scale_factor_temporal": MINIMAX_H3_VAE_TEMPORAL_COMPRESSION,
        "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        "audio_latent_rate": MINIMAX_H3_AUDIO_LATENT_RATE,
        "fps": MINIMAX_H3_FPS,
        # fp32 normalization vectors from the config (NOT the fp16 copies in the
        # weight file) and the pinned tiling policy, which changes the latents.
        "latents_mean": vae_config.get("latents_mean"),
        "latents_std": vae_config.get("latents_std"),
        "audio_latents_mean": audio_vae_config.get("latents_mean"),
        "audio_latents_std": audio_vae_config.get("latents_std"),
        "pixel_mean": MINIMAX_H3_PIXEL_MEAN,
        "pixel_std": MINIMAX_H3_PIXEL_STD,
        "vae_tiling_policy": dict(MINIMAX_H3_VAE_TILING_POLICY),
        "video_vae_dtype": str(vae_dtype),
        # Identity, for the gallery's VAE record and for reloads.
        "vae_source": os.path.basename(layout["vae"]),
        "vae_path": layout["vae"],
        "dit_path": layout["dit"],
        "audio_vae_path": layout["audio_vae"],
        "text_encoder_path": layout["text_encoder"],
        "official_dir": official,
    }
