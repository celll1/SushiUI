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
   (rotated q/k rows vs flat v rows) and corroborated against a second,
   independently written reader of the same files. Getting either one backwards
   produces a model that loads perfectly and generates noise.
3. **SwiGLU halves are swapped.** The single-file/reference ``fc1`` is ``[gate; up]``
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
   "unnormalised hidden state after layer 50" read uses. A file converted from
   a smaller Qwen3-VL (``te_gguf_convert``) declares its own dims and is built
   from THEM, is text-only, and is refused unless a projection trained for that
   exact (width, tap) pair resolves -- see ``te_projection.py``. Such a file is
   never auto-selected; a load-time override or a component switch reaches it,
   both through ``build_minimax_h3_text_encoder_bundle``.
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
scaled layout), the TE through the same census/verify pattern restricted to
its non-ConvRot layers (see below), and both VAEs through
``refuse_quantized_state_dict`` (they have no swap path at all, so a
quantized file must be refused rather than cast).

Both the DiT and the TE additionally accept the exact released
``int8_tensorwise`` ConvRot contract (groupsize 256) and execute its online
activation rotation through Comfy-Kitchen (``ConvRotInt8Linear``); other
ConvRot declarations remain refused. The TE additionally accepts the exact
released ``nvfp4`` / ``full_precision_matrix_mult`` AWQ contract on its
co-distributed ``nvfp4_awq`` file: the AWQ smoothing already folded into
``input_layernorm`` / ``post_attention_layernorm`` is loaded as-is (no
un-smoothing needed -- the file is self-consistent), and the ``.pre_quant_scale``
vectors that exist only on ``self_attn.o_proj`` / ``mlp.down_proj`` (the two
Linears per decoder layer with nowhere upstream to fold the smoothing into) are
installed on a dedicated ``Nvfp4Linear`` and applied to the ACTIVATION at
inference (see ``core.models.common.nvfp4_linear`` and
``scratchpad/minimax_h3_te_nvfp4_verification.md``). That file's
``model.embed_tokens`` carries a THIRD, unrelated ``int8_tensorwise`` (no
rotation) contract on an ``nn.Embedding`` rather than an ``nn.Linear``, handled
by a dedicated gather-then-scale ``Int8Embedding``
(``core.models.common.int8_embedding``). Every other quantization declaration
on the TE remains refused. Packed ``asym_w4a8_int8`` DiTs are handled
separately from file-level metadata. ``_assert_guard_reached`` pins the
remaining guard property in code.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_bf16.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
]

# Text-encoder file spellings this loader can build from. A ``.gguf`` is
# reachable ONLY as an explicit override: nothing globs for it and
# ``_te_capability_accept`` refuses it, for the same reason a converted file is
# refused there -- a small stand-in encoder must never be auto-selected.
_TE_SUFFIXES = frozenset({".safetensors", ".gguf"})

# ``.comfy_quant`` "format" strings (see ``quantized_checkpoint_guard.py``)
# whose TEXT ENCODER weights ``_build_text_encoder`` can actually install.
# ``int8_tensorwise`` (the co-distributed ConvRot file, groupsize 256) is
# installable: ``_build_text_encoder`` accepts the same exact contract the DiT
# builder does (``_supported_int8_convrot_marker``) and swaps in
# ``ConvRotInt8Linear`` before ``load_state_dict``. ``nvfp4`` (the
# co-distributed ``nvfp4_awq`` file) is installable too: ``_build_text_encoder``
# validates the exact released contract (``_supported_h3_nvfp4_marker``) and
# swaps in ``Nvfp4Linear``; its ``model.embed_tokens`` layer is a separate,
# always-installable ``int8_tensorwise`` (no rotation) ``nn.Embedding`` contract
# handled by ``Int8Embedding`` regardless of which TE file is selected. This
# frozenset only gates ``_te_capability_accept``'s header-only "is a quantized
# file loadable at all" predicate (it does not itself discriminate by format
# string -- see that function); a later TE quant contract this loader cannot
# yet install still needs its own ``_supported_h3_*_marker`` validator and swap
# path before adding its format string here would be honest.
MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS: FrozenSet[str] = frozenset({"int8_tensorwise", "nvfp4"})

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
    ``ceil(T / 17) * 5 - 3``. The closed form that circulates for this model,
    ``2 if T<=5 else ((T-5)//17)*5+2``, agrees only on the ``17n+5`` grid and
    disagrees off it (T=18: it says 2, measured 7), so this form is the one to
    use.

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


def detect_minimax_h3_layout(
    path: str, *, te_override: Optional[str] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """``{dit, vae, audio_vae, text_encoder, official, root, variant, text_encoder_reason}`` or ``None``.

    Accepts three spellings of the same tree:

    * the flat root (``<root>/diffusion_models/`` + ``vae/`` + ``text_encoders/``);
    * a DiT ``.safetensors`` inside ``<root>/diffusion_models/`` (walks up);
    * MiniMax's config-only ``official/`` directory, i.e. one carrying a
      ``model_index.json`` whose ``_class_name`` is ``MiniMaxH3ModularPipeline``
      -- resolved to its parent when that parent holds the weights, because
      ``official/`` alone has none.

    ``te_override``, when given, names the exact text encoder file to use,
    symmetric with the DiT ``.safetensors``-path spelling of ``path`` above: an
    explicit request bypasses ``MINIMAX_H3_TE_PATTERNS`` and the loadability
    predicate entirely, because naming a file IS the caller's decision, not
    ours to second-guess. It is still validated here (existence + extension)
    so a typo fails with a message naming the bad path, rather than as a
    downstream "missing text_encoder" that does not mention it.
    """
    te_override_path: Optional[Path] = None
    if te_override is not None:
        te_override_path = Path(te_override)
        if not te_override_path.is_file() or te_override_path.suffix.lower() not in _TE_SUFFIXES:
            raise FileNotFoundError(
                f"MiniMax-H3 text_encoder override {te_override!r} is not an existing "
                f"{' or '.join(sorted(_TE_SUFFIXES))} file")

    if not path:
        return None
    p = Path(path)

    root: Optional[Path] = None
    if p.is_file() and p.suffix == ".safetensors":
        for parent in p.parents:
            if (parent / "diffusion_models").is_dir():
                return _layout_from_root(parent, dit_override=p, te_override=te_override_path)
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
                "text_encoder_reason": None,
                "official": str(p), "root": str(p), "variant": None}
    if root is None:
        return None
    return _layout_from_root(root, te_override=te_override_path)


# The dims a CONVERTED small encoder (``te_gguf_convert``) declares in its own
# ``minimax_h3_te`` metadata. The shipped 32B files declare only
# ``num_hidden_layers`` and ``output``, so "all of these present" is what
# discriminates a file whose geometry comes from ITSELF from one whose geometry
# comes from ``official/text_encoder/config.json``.
_TE_DECLARED_DIM_KEYS = ("hidden_size", "num_attention_heads", "num_key_value_heads",
                         "head_dim", "intermediate_size", "rms_norm_eps", "rope_theta",
                         "mrope_section", "vocab_size")


def _te_declaration(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A text encoder file's own ``minimax_h3_te`` block, or ``{}``."""
    try:
        declared = json.loads((metadata or {}).get("minimax_h3_te", "{}"))
    except Exception:
        return {}
    return declared if isinstance(declared, dict) else {}


def _te_declared_dims(declared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The declared text-config dims, or ``None`` when the file declares none."""
    if all(key in declared for key in _TE_DECLARED_DIM_KEYS):
        return {key: declared[key] for key in _TE_DECLARED_DIM_KEYS}
    return None


def _te_file_declaration(te_path: str) -> Dict[str, Any]:
    """Header-only ``minimax_h3_te`` block of ``te_path``; ``{}`` if unreadable.

    A raw GGUF has no such block; its KV metadata is read into the same shape
    (``te_gguf_native.read_gguf_te_declaration``), minus ``num_hidden_layers``
    -- an unconverted file carries every block, so the depth is the trained
    projection's ``tap`` rather than the file's.

    Unreadable is not fatal here: every caller goes on to build the encoder from
    the same file and fails there with its own message.
    """
    from core.models.minimax_h3.te_gguf_native import is_gguf_path, read_gguf_te_declaration

    if is_gguf_path(te_path):
        try:
            return read_gguf_te_declaration(te_path)
        except Exception:
            return {}
    try:
        header = read_safetensors_header(te_path)
    except Exception:
        return {}
    return _te_declaration(header.get("__metadata__"))


def _te_capability_accept(path: Path) -> bool:
    """HEADER-ONLY: can ``_build_text_encoder`` actually load ``path`` today?

    A converted small encoder is rejected outright, however loadable it is: it
    is a DIFFERENT model that only approximates the shipped 32B, and
    ``_find_first``'s glob fallbacks would otherwise auto-select one the moment
    no shipped file is present. Reaching it must take an explicit
    ``te_override``.

    Zero tensor bytes are read (the JSON a ``.comfy_quant`` marker carries lives
    in the tensor BODY, and deciding "is this file quantized" needs only the
    header's key names). Any of the three markers a quantized Comfy-Org text
    encoder distribution carries -- ``.comfy_quant``, ``.pre_quant_scale``, or a
    ``.weight_scale`` beside its ``.weight`` -- is treated as positive evidence
    of quantization; ``_build_text_encoder`` calls
    ``refuse_quantized_state_dict``/``refuse_unsupported_quant_semantics``
    unconditionally, so a file with any of that evidence would be refused at
    load time regardless of which specific format it declares. The only escape
    is a non-empty ``MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS`` -- consulted here,
    the single named place -- which is empty until a TE quant decode path
    exists.

    A ``.gguf`` is refused on the same terms as a converted file, belt and
    braces: nothing globs for that suffix either.
    """
    from core.models.minimax_h3.te_gguf_native import is_gguf_path

    if is_gguf_path(str(path)):
        return False
    header = read_safetensors_header(str(path))
    metadata = header.pop("__metadata__", None)
    if _te_declared_dims(_te_declaration(metadata)) is not None:
        return False
    quantized = any(
        key.endswith(".comfy_quant") or key.endswith(".pre_quant_scale")
        or key.endswith(".weight_scale")
        for key in header
    )
    if not quantized:
        return True
    return bool(MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS)


def _inspect_converted_te_candidate(
    result: Dict[str, Any], declared: Dict[str, Any], dims: Dict[str, Any],
    layers: Set[int], embed: Optional[List[int]],
) -> Dict[str, Any]:
    """Listing entry for a converted small encoder: loadable, never the default.

    Kept out of the 32B geometry branch because the whole point of these files
    is that their geometry differs; what is checked instead is that the file
    agrees with its OWN declaration.
    """
    tap = declared.get("num_hidden_layers")
    hidden = dims["hidden_size"]
    vocab = dims["vocab_size"]
    if not isinstance(tap, int) or set(layers) != set(range(tap)):
        result["reason"] = (
            f"Declares num_hidden_layers={tap} but carries decoder layers "
            f"{sorted(layers)[:3]}...{sorted(layers)[-1:] or []}.")
        return result
    if embed != [vocab, hidden]:
        result["reason"] = (
            f"Declares vocab_size={vocab}/hidden_size={hidden} but its embedding is {embed}.")
        return result

    size = declared.get("source_size_label") or "small"
    result["compatible"] = True
    result["variant"] = "converted_small"
    result["hidden_size"] = int(hidden)
    result["num_hidden_layers"] = int(tap)
    result["reason"] = (
        f"Converted {size} Qwen3-VL ({tap} blocks, hidden {hidden}), text-only. Never the "
        f"architecture default; usable only with the trained d_in={hidden} projection from "
        f"clip_projections/, which is paired with it whenever it is selected.")
    return result


def _inspect_gguf_te_candidate(result: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Listing entry for a raw Qwen3-VL GGUF: KV metadata only, no tensor bytes.

    ``num_hidden_layers`` stays ``None`` -- the file carries every block and the
    trained projection's ``tap`` decides how many are mapped -- so the depth is
    reported as ``block_count`` instead.
    """
    from core.models.minimax_h3.te_gguf_native import (
        SUPPORTED_GGML_TYPES, read_gguf_te_declaration,
    )

    try:
        declared = read_gguf_te_declaration(path)
    except Exception as exc:
        result["reason"] = f"GGUF metadata inspection failed: {exc}"
        return result

    unsupported = sorted(set(declared.get("ggml_types") or ()) - set(SUPPORTED_GGML_TYPES))
    if unsupported:
        result["variant"] = "gguf_unsupported"
        result["reason"] = (
            f"Carries GGML type(s) {', '.join(unsupported)}; native loading implements "
            f"{', '.join(SUPPORTED_GGML_TYPES)} only. Convert the file with "
            f"core.models.minimax_h3.te_gguf_convert first.")
        return result

    hidden = int(declared["hidden_size"])
    blocks = int(declared["block_count"])
    size = declared.get("source_size_label") or "small"
    result["compatible"] = True
    result["variant"] = "gguf_q8_0"
    result["hidden_size"] = hidden
    result["block_count"] = blocks
    result["reason"] = (
        f"Q8_0 GGUF, {size} Qwen3-VL ({blocks} blocks, hidden {hidden}), text-only, loaded "
        f"without conversion. Never the architecture default; usable only with a trained "
        f"d_in={hidden} projection from clip_projections/, whose tap selects how many of those "
        f"blocks are run.")
    return result


def inspect_minimax_h3_text_encoder_candidate(path: str) -> Dict[str, Any]:
    """Header-only compatibility result for the dedicated H3 TE loader.

    Candidate discovery must never open a second tensor mapping while the live
    encoder is mapped. Only the three formats this loader explicitly implements
    are accepted; renamed or structurally ambiguous files remain visible but
    disabled.
    """
    result: Dict[str, Any] = {
        "path": path,
        "compatible": False,
        "variant": "unknown",
        "reason": "Not an implemented MiniMax-H3 text-encoder format.",
        "size_bytes": None,
        # The file's OWN declaration; the released 32B files declare neither and
        # take their geometry from official/text_encoder/config.json.
        "hidden_size": None,
        "num_hidden_layers": None,
        # Only a raw GGUF sets this: it carries every block, and the projection
        # picks the depth (see ``_inspect_gguf_te_candidate``).
        "block_count": None,
    }
    try:
        result["size_bytes"] = os.path.getsize(path)
    except Exception as exc:
        result["reason"] = f"Header inspection failed: {exc}"
        return result

    from core.models.minimax_h3.te_gguf_native import is_gguf_path

    if is_gguf_path(path):
        return _inspect_gguf_te_candidate(result, path)

    try:
        header = read_safetensors_header(path)
    except Exception as exc:
        result["reason"] = f"Header inspection failed: {exc}"
        return result

    keys = [key for key in header if key != "__metadata__"]
    layers = {
        int(parts[2])
        for key in keys
        if key.startswith("model.layers.")
        for parts in (key.split("."),)
        if len(parts) > 3 and parts[2].isdigit()
    }
    embed = _header_shape(header, "model.embed_tokens.weight")
    q_proj = _header_shape(header, "model.layers.0.self_attn.q_proj.weight")

    declared = _te_declaration(header.get("__metadata__"))
    dims = _te_declared_dims(declared)
    if dims is not None:
        return _inspect_converted_te_candidate(result, declared, dims, layers, embed)

    if layers != set(range(50)) or embed != [151936, 5120]:
        result["reason"] = "H3 requires the released 50-layer Qwen3-VL-32B encoder geometry."
        return result
    q_entry = header.get("model.layers.0.self_attn.q_proj.weight") or {}
    marker_count = sum(key.endswith(".comfy_quant") for key in keys)
    scale_count = sum(key.endswith(".weight_scale") for key in keys)
    pre_quant_count = sum(key.endswith(".pre_quant_scale") for key in keys)
    q_dtype = q_entry.get("dtype")
    dense_shapes = {
        "self_attn.q_proj": [8192, 5120],
        "self_attn.k_proj": [1024, 5120],
        "self_attn.v_proj": [1024, 5120],
        "self_attn.o_proj": [5120, 8192],
        "mlp.gate_proj": [25600, 5120],
        "mlp.up_proj": [25600, 5120],
        "mlp.down_proj": [5120, 25600],
    }

    def geometry_matches(dtype: str, packed: bool = False) -> bool:
        for layer in range(50):
            for suffix, expected in dense_shapes.items():
                entry = header.get(f"model.layers.{layer}.{suffix}.weight")
                shape = list(expected)
                if packed:
                    shape[1] //= 2
                if not isinstance(entry, dict) or entry.get("dtype") != dtype or entry.get("shape") != shape:
                    return False
        return True

    if (q_proj == [8192, 5120] and q_dtype == "BF16"
            and not (marker_count or scale_count or pre_quant_count)
            and geometry_matches("BF16")):
        variant = "bf16"
    elif (q_proj == [8192, 5120] and q_dtype == "I8"
          and marker_count == 350 and scale_count == 350 and pre_quant_count == 0
          and geometry_matches("I8")):
        variant = "int8_convrot"
    elif (q_proj == [8192, 2560] and q_dtype == "U8"
          and marker_count == 351 and scale_count == 351 and pre_quant_count == 100
          and geometry_matches("U8", packed=True)):
        variant = "nvfp4_awq"
    else:
        result["reason"] = (
            "Q projection dtype/shape and quantization marker coverage do not match "
            "an implemented H3 TE format."
        )
        return result

    if variant != "bf16":
        try:
            markers = _read_comfy_quant_markers(path, header)
            if variant == "int8_convrot":
                valid = len(markers) == 350 and all(
                    _supported_int8_convrot_marker(key, marker, header, path=path) is not None
                    for key, marker in markers.items()
                )
            else:
                valid = len(markers) == 351 and all(
                    (
                        _supported_h3_int8_embedding_marker(key, marker, header, path=path)
                        if key == "model.embed_tokens.comfy_quant"
                        else _supported_h3_nvfp4_marker(key, marker, header, path=path)
                    ) is not None
                    for key, marker in markers.items()
                )
            if not valid:
                result["reason"] = "Quantization marker payloads do not match the implemented H3 loader contract."
                return result
        except Exception as exc:
            result["reason"] = f"Quantization marker inspection failed: {exc}"
            return result

    result["variant"] = variant
    result["compatible"] = True
    result["reason"] = f"Header matches the H3 {variant} loader contract."
    return result


def _read_comfy_quant_markers(path: str, header: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Read only tiny U8 marker payloads without mapping checkpoint weights."""
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("truncated safetensors header")
        (header_length,) = struct.unpack("<Q", raw_length)
        data_start = 8 + header_length
        markers: Dict[str, torch.Tensor] = {}
        for key, entry in header.items():
            if not key.endswith(".comfy_quant"):
                continue
            if not isinstance(entry, dict) or entry.get("dtype") != "U8":
                raise ValueError(f"{key} is not a U8 marker")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if (not isinstance(shape, list) or len(shape) != 1
                    or not isinstance(offsets, list) or len(offsets) != 2):
                raise ValueError(f"{key} has invalid marker metadata")
            start, end = (int(value) for value in offsets)
            if start < 0 or end < start or end - start != int(shape[0]) or end - start > 4096:
                raise ValueError(f"{key} marker range is invalid")
            handle.seek(data_start + start)
            payload = handle.read(end - start)
            if len(payload) != end - start:
                raise ValueError(f"{key} marker is truncated")
            markers[key] = torch.tensor(list(payload), dtype=torch.uint8)
        return markers


def list_minimax_h3_text_encoder_candidates(model_path: str) -> List[Dict[str, Any]]:
    """List every H3-tree TE file, including disabled unknown ones.

    ``.gguf`` files are listed (they are selectable as an explicit override)
    even though nothing auto-selects one.
    """
    layout = detect_minimax_h3_layout(model_path)
    if layout is None or not layout.get("root"):
        return []
    directory = Path(str(layout["root"])) / "text_encoders"
    if not directory.is_dir():
        return []
    found = [path for suffix in sorted(_TE_SUFFIXES) for path in directory.glob(f"*{suffix}")]
    return [
        inspect_minimax_h3_text_encoder_candidate(str(path))
        for path in sorted(found, key=lambda item: item.name.lower())
    ]


def list_minimax_h3_te_projection_candidates(model_path: str) -> List[Dict[str, Any]]:
    """Every parseable ``clip_projections/`` spec in this tree, header-only.

    A file that fails to parse is skipped, matching ``discover_te_projections``:
    listing is not the place a malformed projection is reported, pairing is.
    """
    from core.models.minimax_h3.te_projection import projection_dir, read_te_projection_spec

    layout = detect_minimax_h3_layout(model_path)
    if layout is None or not layout.get("root"):
        return []
    directory = projection_dir(str(layout["root"]))
    if not directory.is_dir():
        return []
    found: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.safetensors"), key=lambda item: item.name.lower()):
        try:
            spec = read_te_projection_spec(str(path))
        except Exception as exc:
            print(f"[MiniMaxH3Loader] skipping projection {path.name}: {exc}")
            continue
        try:
            spec["size_bytes"] = os.path.getsize(path)
        except OSError:
            spec["size_bytes"] = None
        found.append(spec)
    return found


def _te_projection_candidates(
    root: str, entry: Dict[str, Any], text_dim: int,
) -> List[Dict[str, Any]]:
    """Every projection declaring this encoder's width, each gated on its own.

    ``usable`` comes from ``resolve_te_projection`` with the file NAMED, i.e.
    from the gates the load path runs, so a candidate listed as usable cannot be
    one a load would then refuse. One that matches ``d_in`` but fails another
    gate is listed with its reason rather than dropped: a client offering the
    set must be able to say why an entry is unavailable.
    """
    from core.models.minimax_h3.te_projection import (
        discover_te_projections, resolve_te_projection,
    )

    hidden = int(entry["hidden_size"])
    tap = int(entry.get("num_hidden_layers") or 0)
    blocks = entry.get("block_count")
    candidates: List[Dict[str, Any]] = []
    for spec in discover_te_projections(root, d_in=hidden):
        reason = None
        try:
            resolve_te_projection(
                root=root, te_path=entry["path"], hidden_size=hidden,
                num_hidden_layers=tap, text_dim=text_dim, override=spec["path"],
                available_blocks=int(blocks) if blocks else None)
        except Exception as exc:
            reason = str(exc)
        candidates.append({
            "path": spec["path"],
            "name": os.path.splitext(os.path.basename(spec["path"]))[0],
            "d_in": spec["d_in"], "d_out": spec["d_out"], "tap": spec["tap"],
            "usable": reason is None, "reason": reason,
        })
    return candidates


def describe_minimax_h3_text_encoder_choices(model_path: str) -> Dict[str, Any]:
    """The load-time text-encoder choices for this tree, for the API.

    ``requires_projection`` is decided the same way ``load_minimax_h3_from_path``
    decides it -- the file's declared width against the DiT's ``condition_proj``
    -- rather than from "declares its own dims", so the listing cannot say a
    file needs a projection that the loader would then not ask for.

    ``projection`` is the file a load would actually pair, resolved through the
    same ``resolve_te_projection`` gates (header-only, no tensor bytes), so this
    listing cannot offer a pairing the load path would then refuse;
    ``projection_reason`` carries that refusal when none resolves.
    ``projection_candidates`` is the set to choose FROM when auto-resolution
    refuses because several files declare this encoder's width.

    ``agreement`` is the measurement recorded for that resolved pairing; its
    ``source`` is ``"local"`` (measured on this installation) or ``"published"``
    (shipped, measured elsewhere).
    """
    from core.models.minimax_h3.te_projection import (
        measured_te_substitution, resolve_te_projection,
    )

    layout = detect_minimax_h3_layout(model_path)
    if layout is None:
        raise ValueError(f"{model_path!r} does not resolve to a MiniMax-H3 model tree.")

    text_dim = dit_text_dim(layout["dit"]) if layout.get("dit") else None
    projections = list_minimax_h3_te_projection_candidates(model_path)
    encoders = list_minimax_h3_text_encoder_candidates(model_path)
    for entry in encoders:
        hidden = entry.get("hidden_size")
        entry["requires_projection"] = bool(
            hidden is not None and text_dim is not None and hidden != text_dim)
        entry["projection"] = None
        entry["projection_reason"] = None
        entry["projection_candidates"] = []
        entry["agreement"] = None
        if entry["requires_projection"]:
            entry["projection_candidates"] = _te_projection_candidates(
                str(layout["root"]), entry, int(text_dim))
            try:
                spec = resolve_te_projection(
                    root=str(layout["root"]), te_path=entry["path"], hidden_size=int(hidden),
                    num_hidden_layers=int(entry.get("num_hidden_layers") or 0),
                    text_dim=int(text_dim),
                    available_blocks=(int(entry["block_count"])
                                      if entry.get("block_count") else None),
                )
            except Exception as exc:
                entry["projection_reason"] = str(exc)
                continue
            entry["projection"] = spec["path"]
            measured = measured_te_substitution(entry["path"], spec["path"])
            if measured is not None:
                entry["agreement"] = dict(
                    measured, projection=os.path.basename(spec["path"]))
    return {
        "selected": layout.get("text_encoder"),
        "selected_reason": layout.get("text_encoder_reason"),
        "text_encoders": encoders,
        "clip_projections": projections,
    }


def assert_no_live_text_encoder() -> None:
    """Assert that all prior H3 TE owners and their mapped storages are gone."""
    import gc

    gc.collect()
    live = sorted(path for path, ref in _LIVE_TEXT_ENCODER.items() if ref() is not None)
    live_tensors = {
        path: sum(ref() is not None for ref in refs)
        for path, refs in _LIVE_TEXT_ENCODER_TENSORS.items()
        if any(ref() is not None for ref in refs)
    }
    if live or live_tensors:
        raise RuntimeError(
            "MiniMax-H3 text encoder detach left a live owner; refusing to map another "
            f"50 GB-class encoder (models: {live}, mapped tensors: {live_tensors})."
        )


def _projection_tap(projection: Optional[Dict[str, Any]]) -> Optional[int]:
    """The depth a resolved projection was trained on, for the GGUF builder."""
    tap = ((projection or {}).get("spec") or {}).get("tap")
    return int(tap) if tap else None


def _build_text_encoder_for(te_path: str, official_dir: Optional[str],
                            projection: Optional[Dict[str, Any]]):
    """``_build_text_encoder``, with the ``tap`` only the GGUF path needs.

    Any other file is built through the ORIGINAL two-argument call, so a caller
    (or a test) that replaces ``_build_text_encoder`` with a two-parameter stub
    keeps working -- the same reason ``load_minimax_h3_from_path`` calls
    ``detect_minimax_h3_layout`` both ways.
    """
    from core.models.minimax_h3.te_gguf_native import is_gguf_path

    if is_gguf_path(te_path):
        return _build_text_encoder(te_path, official_dir, tap=_projection_tap(projection))
    return _build_text_encoder(te_path, official_dir)


def build_minimax_h3_text_encoder(te_path: str, official_dir: Optional[str]):
    """Projection-free TE entry point; intentionally performs no device move.

    A small stand-in encoder (converted, or a raw GGUF) is refused here because
    ``(model, config)`` has nowhere to put the projection its hidden state is
    only valid through; such a file goes through
    ``build_minimax_h3_text_encoder_bundle``.
    """
    if _te_declared_dims(_te_file_declaration(te_path)) is not None:
        raise ValueError(
            f"{os.path.basename(te_path)} is a small stand-in text encoder and is usable only "
            f"with its trained projection, which this two-value entry point cannot carry. Build "
            f"it through build_minimax_h3_text_encoder_bundle, which resolves the pairing.")
    bundle = build_minimax_h3_text_encoder_bundle(
        te_path, official_dir, root=None, dit_path=None)
    return bundle["text_encoder"], bundle["text_encoder_config"]


def build_minimax_h3_text_encoder_bundle(
    te_path: str,
    official_dir: Optional[str],
    *,
    root: Optional[str],
    dit_path: Optional[str],
    projection_override: Optional[str] = None,
) -> Dict[str, Any]:
    """The encoder plus everything its conditioning is only valid together with.

    Returns the four component-dict entries a caller must install as one unit:
    ``text_encoder``, ``text_encoder_config``, ``te_projection`` (``None`` for a
    released 32B encoder) and ``te_text_only``.

    The pairing runs through the same ``resolve_minimax_h3_te_projection`` the
    load path uses, against the DiT at ``dit_path``, and it runs BEFORE the
    encoder is mapped: a pairing a load would refuse cannot be installed by a
    component switch, and a failed resolve cannot leave an unprojected encoder
    behind. No device move happens here.
    """
    inspected = inspect_minimax_h3_text_encoder_candidate(te_path)
    if not inspected["compatible"]:
        raise ValueError(inspected["reason"])

    declaration = _te_file_declaration(te_path)
    projection = None
    if _te_declared_dims(declaration) is not None or projection_override is not None:
        if not dit_path:
            raise ValueError(
                f"{os.path.basename(te_path)} needs a trained projection to the DiT's "
                f"conditioning, but no DiT was named to check its width against.")
        projection = resolve_minimax_h3_te_projection(
            te_path=te_path, declared=declaration, root=root,
            text_dim=dit_text_dim(dit_path), override=projection_override,
        )

    assert_no_live_text_encoder()
    model, config = _build_text_encoder_for(te_path, official_dir, projection)
    return {
        "text_encoder": model,
        "text_encoder_config": config,
        "te_projection": projection,
        "te_text_only": str(declaration.get("modalities") or "") == "text",
    }


def minimax_h3_te_model_info_fields(components: Dict[str, Any]) -> Dict[str, Any]:
    """The encoder/projection identity ``current_model_info`` reports.

    One place, so a full load, a DiT-only reload and a component switch cannot
    describe the same component dict differently.
    """
    return {
        "text_encoder_file": os.path.basename(
            str(components.get("text_encoder_path") or "")) or None,
        "clip_projection_file": os.path.basename(
            str((components.get("te_projection") or {}).get("path") or "")) or None,
        "te_text_only": bool(components.get("te_text_only")),
    }


def _te_selection_reason(directory: Path, selected: Optional[Path]) -> Optional[str]:
    """Why ``selected`` (or nothing) is the resolved text encoder, for the log.

    Distinguishes "the most-preferred candidate was used" from "a
    less-preferred one was used because a more-preferred file exists on disk
    but ``_te_capability_accept`` rejected it" -- so a user expecting int8 and
    silently getting bf16 can see why from the loader's own log line, instead
    of having to diff directory listings against ``MINIMAX_H3_TE_PATTERNS``.
    """
    if selected is None:
        stand_ins = sorted(
            path.name
            for suffix in _TE_SUFFIXES
            for path in (directory.glob(f"*{suffix}") if directory.is_dir() else ())
            if _te_declared_dims(_te_file_declaration(str(path))) is not None
        )
        if stand_ins:
            return (f"no text encoder file found ({', '.join(stand_ins)} are small stand-in "
                    f"encoders, reachable only as an explicit override)")
        return "no text encoder file found"
    for idx, pattern in enumerate(MINIMAX_H3_TE_PATTERNS):
        if directory / pattern == selected:
            if idx == 0:
                return "preferred"
            skipped_present = [
                other for other in MINIMAX_H3_TE_PATTERNS[:idx]
                if (directory / other).is_file()
            ]
            if skipped_present:
                return (
                    f"fell back past {', '.join(skipped_present)} (present but not "
                    f"loadable by this build -- see MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS)"
                )
            return "preferred candidate(s) not present"
    return "resolved via glob fallback, no listed filename matched"


def _layout_from_root(
    root: Path,
    dit_override: Optional[Path] = None,
    te_override: Optional[Path] = None,
) -> Optional[Dict[str, Optional[str]]]:
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
    te_dir = root / "text_encoders"
    if te_override is not None:
        te = te_override
        te_reason: Optional[str] = "explicit override"
    else:
        te = _find_first(te_dir, MINIMAX_H3_TE_PATTERNS, accept=_te_capability_accept)
        te_reason = _te_selection_reason(te_dir, te)
    name = dit.name.lower()
    variant = "ref2va" if "ref2va" in name else ("fl2va" if "fl2va" in name else None)
    return {
        "dit": str(dit),
        "vae": str(vae) if vae else None,
        "audio_vae": str(audio_vae) if audio_vae else None,
        "text_encoder": str(te) if te else None,
        "text_encoder_reason": te_reason,
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

# The AdaLN projections run in float32 in curve mode: their input is the 8-dim
# curve coordinate read from a float32 table, and the F16-stored weights are
# upcast to match. The modulation vectors are cast back down to the block stack's
# dtype inside the block, so this does NOT promote the residual stream.
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


# Decoder layers whose input does NOT come from a layernorm, so the co-
# distributed NVFP4/AWQ file's smoothing scale has nowhere upstream to fold
# into and is stored explicitly as ``.pre_quant_scale`` instead (see
# ``scratchpad/minimax_h3_te_nvfp4_verification.md``, section E). Named here,
# not inlined, because both the marker validator and its test must agree on
# EXACTLY these two suffixes.
_H3_NVFP4_PRE_QUANT_SCALE_LAYER_SUFFIXES = (".self_attn.o_proj", ".mlp.down_proj")


def _supported_h3_nvfp4_marker(
    key: str,
    marker: torch.Tensor,
    header: Dict[str, Any],
    *,
    path: str,
) -> Optional[Dict[str, Any]]:
    """Validate the one NVFP4/AWQ contract implemented by the H3 TE loader.

    Verified against the real file at the noise floor of 4-bit block-scaled
    quantization (cos ~0.9955-0.9957, relFrob ~0.0925-0.0949) -- see
    ``scratchpad/minimax_h3_te_nvfp4_verification.md``. Refuses (returns
    ``None``, which the caller treats as an unrecognized marker, still
    refused by the generic guard) a ``.pre_quant_scale`` on any layer other
    than ``self_attn.o_proj`` / ``mlp.down_proj``, and a MISSING one on those
    two -- not a blanket bypass of the AWQ contract, the exact one measured.
    """
    from core.models.common.quantized_checkpoint_guard import decode_comfy_quant_marker

    parsed = decode_comfy_quant_marker(marker)
    if parsed != {"format": "nvfp4", "full_precision_matrix_mult": True}:
        return None
    layer = key[: -len(".comfy_quant")]
    weight = header.get(layer + ".weight")
    scale = header.get(layer + ".weight_scale")
    scale_2 = header.get(layer + ".weight_scale_2")
    if not isinstance(weight, dict) or not isinstance(scale, dict) or not isinstance(scale_2, dict):
        raise ValueError(
            f"{path}: NVFP4 layer '{layer}' is missing weight, weight_scale or weight_scale_2"
        )
    shape = weight.get("shape", [])
    if weight.get("dtype") != "U8" or not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{path}: NVFP4 layer '{layer}' weight must be 2-D U8")
    out_features, packed_k = (int(x) for x in shape)
    # ``packed_k`` is K/2 (two E2M1 codes per byte). Requiring it divisible by
    # 8 is exactly requiring K divisible by 16, the block size -- the "K/2 and
    # K/16 divisibility" the task specifies collapse to this one check.
    if packed_k % 8:
        raise ValueError(
            f"{path}: NVFP4 layer '{layer}' packed K/2={packed_k} is not divisible by 8 "
            f"(K={packed_k * 2} would not be divisible by the block size 16)"
        )
    in_features = packed_k * 2
    scale_shape = list(scale.get("shape", []))
    if scale.get("dtype") != "F8_E4M3" or scale_shape != [out_features, in_features // 16]:
        raise ValueError(
            f"{path}: NVFP4 layer '{layer}' weight_scale must be F8_E4M3 "
            f"[{out_features}, {in_features // 16}], got {scale.get('dtype')} {scale_shape}"
        )
    scale_2_shape = list(scale_2.get("shape", []))
    if scale_2.get("dtype") != "F32" or scale_2_shape not in ([], [1]):
        raise ValueError(
            f"{path}: NVFP4 layer '{layer}' weight_scale_2 must be a scalar F32, "
            f"got {scale_2.get('dtype')} {scale_2_shape}"
        )
    pqs_key = layer + ".pre_quant_scale"
    pqs = header.get(pqs_key)
    is_smoothing_source_layer = layer.endswith(_H3_NVFP4_PRE_QUANT_SCALE_LAYER_SUFFIXES)
    if pqs is not None:
        if not is_smoothing_source_layer:
            raise ValueError(
                f"{path}: NVFP4 layer '{layer}' carries '{pqs_key}', but only "
                f"{_H3_NVFP4_PRE_QUANT_SCALE_LAYER_SUFFIXES} layers are validated to have "
                f"one -- refusing rather than silently ignoring an AWQ scale on a layer "
                f"this build has never seen it on"
            )
        pqs_shape = list(pqs.get("shape", []))
        if pqs.get("dtype") != "BF16" or pqs_shape != [in_features]:
            raise ValueError(
                f"{path}: NVFP4 layer '{layer}' pre_quant_scale must be BF16 "
                f"[{in_features}], got {pqs.get('dtype')} {pqs_shape}"
            )
    elif is_smoothing_source_layer:
        raise ValueError(
            f"{path}: NVFP4 layer '{layer}' matches {_H3_NVFP4_PRE_QUANT_SCALE_LAYER_SUFFIXES} "
            f"but carries no '{pqs_key}' -- its input does not come from a layernorm, so "
            f"the AWQ smoothing has nowhere to have been folded and must be present"
        )
    return {
        "in_features": in_features,
        "out_features": out_features,
        "has_pre_quant_scale": pqs is not None,
        "marker_numel": int(marker.numel()),
    }


def _supported_h3_int8_embedding_marker(
    key: str,
    marker: torch.Tensor,
    header: Dict[str, Any],
    *,
    path: str,
) -> Optional[Dict[str, Any]]:
    """Validate the plain (non-rotated) int8 ``nn.Embedding`` contract.

    The co-distributed NVFP4/AWQ file's ``model.embed_tokens`` carries this --
    a SEPARATE contract from the ``nvfp4`` one above, on an ``nn.Embedding``
    rather than an ``nn.Linear`` (see ``scratchpad/minimax_h3_te_nvfp4_verification.md``,
    section A). Its marker declares only ``{"format": "int8_tensorwise"}``
    (no ``"convrot"`` key), which the GENERIC guard already treats as an
    ordinary per-row-scaled tensor and does not refuse -- this validator exists
    to identify the layer for ``Int8Embedding``'s gather-then-scale swap, not
    to waive anything the generic guard would otherwise block.
    """
    from core.models.common.quantized_checkpoint_guard import decode_comfy_quant_marker

    parsed = decode_comfy_quant_marker(marker)
    if parsed != {"format": "int8_tensorwise"}:
        return None
    layer = key[: -len(".comfy_quant")]
    weight = header.get(layer + ".weight")
    scale = header.get(layer + ".weight_scale")
    if not isinstance(weight, dict) or not isinstance(scale, dict):
        raise ValueError(f"{path}: INT8 embedding '{layer}' is missing weight or weight_scale")
    shape = weight.get("shape", [])
    if weight.get("dtype") != "I8" or not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"{path}: INT8 embedding '{layer}' weight must be 2-D I8")
    num_embeddings, embedding_dim = (int(x) for x in shape)
    scale_shape = list(scale.get("shape", []))
    if scale.get("dtype") != "F32" or scale_shape not in ([num_embeddings], [num_embeddings, 1]):
        raise ValueError(
            f"{path}: INT8 embedding '{layer}' weight_scale must be F32 "
            f"[{num_embeddings}] or [{num_embeddings}, 1], got {scale.get('dtype')} {scale_shape}"
        )
    return {
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "marker_numel": int(marker.numel()),
    }


def _guard_component_file(
    path: str,
    *,
    label: str,
    allow_h3_int8_convrot: bool = False,
    allow_h3_nvfp4: bool = False,
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

    ``allow_h3_nvfp4`` waives the ONE exact NVFP4/AWQ contract
    ``_supported_h3_nvfp4_marker`` validates: a marker exactly matching it is
    excluded from the probe, and so is its ``.pre_quant_scale`` sidecar (but
    ONLY when that marker's own layer is one the validator confirmed is
    allowed to carry one, ``self_attn.o_proj`` / ``mlp.down_proj`` --
    ``.pre_quant_scale`` on any other layer, or one the marker validator
    otherwise rejects, still hits the probe and refuses exactly as before).
    ``model.embed_tokens``'s separate ``int8_tensorwise`` (no rotation) marker
    needs no flag here: the generic guard already treats a known, unrotated
    format as an ordinary scaled tensor and does not refuse it.
    """
    header = read_safetensors_header(path)
    metadata = header.pop("__metadata__", None) or {}

    probe: Dict[str, torch.Tensor] = {}
    marker_keys = [k for k in header if k.endswith(".comfy_quant")]
    validated_nvfp4_layers: Dict[str, Dict[str, Any]] = {}
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
                if allow_h3_nvfp4:
                    nvfp4_config = _supported_h3_nvfp4_marker(key, marker, header, path=path)
                    if nvfp4_config is not None:
                        validated_nvfp4_layers[key[: -len(".comfy_quant")]] = nvfp4_config
                        continue
                probe[key] = marker
    for key, entry in header.items():
        if not key.endswith(".pre_quant_scale"):
            continue
        layer = key[: -len(".pre_quant_scale")]
        validated_config = validated_nvfp4_layers.get(layer)
        if validated_config is not None and validated_config.get("has_pre_quant_scale"):
            continue
        dtype = _HEADER_DTYPES.get((entry or {}).get("dtype"), torch.float32) \
            if isinstance(entry, dict) else torch.float32
        probe[key] = torch.empty(0, dtype=dtype)
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


def _h3_nvfp4_layers_from_markers(
    handle,
    header: Dict[str, Any],
    *,
    path: str,
) -> Dict[str, Dict[str, Any]]:
    """Return source-layer configs for validated H3 NVFP4/AWQ marker tensors."""
    layers: Dict[str, Dict[str, Any]] = {}
    for key in header:
        if not key.endswith(".comfy_quant"):
            continue
        config = _supported_h3_nvfp4_marker(key, handle.get_tensor(key), header, path=path)
        if config is not None:
            layers[key[: -len(".comfy_quant")]] = config
    return layers


def _h3_int8_embedding_layers_from_markers(
    handle,
    header: Dict[str, Any],
    *,
    path: str,
) -> Dict[str, Dict[str, Any]]:
    """Return source-layer configs for validated H3 plain-int8 embedding markers."""
    layers: Dict[str, Dict[str, Any]] = {}
    for key in header:
        if not key.endswith(".comfy_quant"):
            continue
        config = _supported_h3_int8_embedding_marker(
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

    ``key == "model.embed_tokens"`` (no trailing dot) is handled explicitly,
    not only ``"model.embed_tokens."``: the INT8-embedding marker layer
    configs are keyed by the LAYER STEM (``.comfy_quant``/``.weight`` stripped
    off), which for this one module IS the whole tensor-name prefix with
    nothing after it -- unlike every ``model.layers.N....`` stem, which always
    has more path after the prefix. Without this branch the stem would fail
    BOTH ``startswith`` checks and pass through unmodified, mapping the
    embedding swap's config key to the file's flat name instead of the built
    model's ``model.language_model.embed_tokens`` path.
    """
    if key == "model.embed_tokens":
        return "model.language_model.embed_tokens"
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

# Where `_rewrite_te_key` puts the vision tower. A converted (`modalities:
# "text"`) file has none of it.
_TE_VISION_PREFIX = "model.visual."


def _te_text_config_from_dims(official_text_config: Optional[Dict[str, Any]],
                              dims: Dict[str, Any]) -> Dict[str, Any]:
    """The official 32B text config with the FILE's declared dims substituted.

    Everything the file does not declare (activation, attention_bias, the token
    ids, ``mrope_interleaved``) is kept from ``official/``. ``mrope_interleaved``
    is immaterial on the text-only path: all three mrope position streams carry
    the same positions there, so the sectioning cannot change the rotation.
    """
    text_config = dict(official_text_config or {})
    for key in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "intermediate_size", "rms_norm_eps", "rope_theta", "vocab_size"):
        text_config[key] = dims[key]
    rope = dict(text_config.get("rope_scaling") or {})
    rope["mrope_section"] = list(dims["mrope_section"])
    rope.setdefault("rope_type", "default")
    text_config["rope_scaling"] = rope
    return text_config


# Weak reference to the text encoder this process built last, keyed by file path.
# Weak so it never keeps one alive; see `_refuse_double_mapping` for what it is
# for. Not thread-safe by design -- a model load is already serialised by
# `PipelineManager._load_model_lock`, and a stale entry only costs one
# `gc.collect()`.
_LIVE_TEXT_ENCODER: Dict[str, Any] = {}
_LIVE_TEXT_ENCODER_TENSORS: Dict[str, Tuple[Any, ...]] = {}


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
    tensor_refs = _LIVE_TEXT_ENCODER_TENSORS.get(te_path, ())
    if (ref is None or ref() is None) and not any(item() is not None for item in tensor_refs):
        return
    gc.collect()
    if (ref is None or ref() is None) and not any(item() is not None for item in tensor_refs):
        return
    raise RuntimeError(
        f"a MiniMax-H3 text encoder built from {te_path} is STILL ALIVE in this process. Its "
        f"48 GiB of weights are memory-mapped from that file, and mapping it a second time "
        f"terminates the process outright (measured; Windows reports 'the paging file is too "
        f"small' when it reports anything at all). Something is holding a reference to the "
        f"previous minimax_h3 component dict across this load -- the pipeline's own teardown "
        f"branch drops it, so look for a cache, a keep-hot resident set, or a debugger handle. "
        f"Refusing here so the cause is visible instead of the backend simply vanishing.")


def _te_guard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    swappable_layer_configs: Dict[str, Dict[str, Any]],
    nvfp4_layer_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """The text encoder's view of its own tensors for the semantics refusal.

    Hides exactly what the builder goes on to implement, and nothing else:

    * the ``.comfy_quant`` marker of a validated ConvRot/NVFP4 Linear, which
      stays as live module state on the swapped module;
    * the ``.pre_quant_scale`` of a validated NVFP4 layer that
      ``_supported_h3_nvfp4_marker`` confirmed carries one -- ``Nvfp4Linear``
      applies it, and the generic guard refuses that suffix outright, so
      leaving it visible refused a file this builder fully reads.

    ``model.embed_tokens``'s plain ``int8_tensorwise`` marker stays visible
    (the generic guard does not refuse it), as does a ``.pre_quant_scale`` on
    any other layer -- no module here would apply one.

    Same waiver as the header-side probe in ``_guard_component_file``; the two
    views of one file must not disagree about which keys are hidden.
    """
    waived = {
        layer + ".pre_quant_scale"
        for layer, cfg in nvfp4_layer_configs.items()
        if cfg.get("has_pre_quant_scale")
    }
    return {
        key: value for key, value in state_dict.items()
        if key not in waived
        and not (
            key.endswith(".comfy_quant")
            and key[: -len(".comfy_quant")] in swappable_layer_configs
        )
    }


def _build_text_encoder(te_path: str, official_dir: Optional[str], *, tap: Optional[int] = None):
    """Build the truncated Qwen3-VL and install the file's tensors BY REFERENCE.

    ``tap`` is used by the ``.gguf`` path only, where the file carries every
    block and the trained projection's tap decides the depth; a safetensors file
    declares its own and ignores it.

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

    ConvRot-swapped layers keep this property. The int8 ``.weight`` and the
    ``.comfy_quant`` marker are installed exactly as ``safe_open`` returns
    them; the only transform is ``.weight_scale.reshape(-1)`` on a `[out, 1]`
    tensor, which PyTorch resolves as a stride-only view (an ``[out, 1]``
    C-contiguous tensor is always reshapeable to ``[out]`` without a copy), so
    no new storage is allocated even for that one. Nothing here reads the int8
    codes into a dense fp32/bf16 dequantized weight -- the point of ConvRot at
    inference is that comfy-kitchen's kernel consumes the codes and the scale
    directly (see ``ConvRotInt8Linear.forward``).
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from core.models.common.quantized_checkpoint_guard import (
        quantized_state_dict_report, scaled_quantization_report, verify_quantized_swap,
    )
    from core.models.minimax_h3.te_gguf_native import is_gguf_path

    if is_gguf_path(te_path):
        return _build_text_encoder_from_gguf(te_path, official_dir, tap)

    # THE GUARD FIRST -- ahead of the double-mapping check and both config reads.
    # The co-distributed `qwen3vl_32b_minimax_h3_int8_convrot` / `_nvfp4_awq`
    # text encoders are exactly what it exists for. `allow_h3_int8_convrot`
    # waives only the one exact ConvRot contract this builder goes on to
    # install (`_supported_int8_convrot_marker`, same validation the DiT
    # builder runs); `allow_h3_nvfp4` waives the one exact NVFP4/AWQ contract
    # (`_supported_h3_nvfp4_marker`) below. `model.embed_tokens`'s separate
    # plain int8 marker needs no flag (see `_guard_component_file`). A reload
    # that also tripped `_refuse_double_mapping` would otherwise answer with
    # THAT.
    header, metadata = _guard_component_file(
        te_path, label="text encoder", allow_h3_int8_convrot=True, allow_h3_nvfp4=True,
    )

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

    declared = _te_declaration(metadata)
    num_layers = None
    try:
        num_layers = int(declared["num_hidden_layers"]) if declared.get("num_hidden_layers") else None
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

    dims = _te_declared_dims(declared)
    text_only = str(declared.get("modalities") or "") == "text"
    if dims is not None:
        raw_config["text_config"] = _te_text_config_from_dims(raw_config.get("text_config"), dims)
        print(f"[MiniMaxH3Loader] text encoder: geometry from the file's own minimax_h3_te "
              f"metadata (hidden {dims['hidden_size']}, {dims['num_attention_heads']} heads / "
              f"{dims['num_key_value_heads']} kv, head_dim {dims['head_dim']}, ffn "
              f"{dims['intermediate_size']}, vocab {dims['vocab_size']}) -- the 32B "
              f"text_encoder/config.json is NOT applied")

    config = Qwen3VLConfig(**raw_config)
    config.text_config.num_hidden_layers = num_layers
    print(f"[MiniMaxH3Loader] text encoder: Qwen3-VL truncated to {num_layers} decoder layer(s) "
          f"(the file's own declared output is the unnormalised hidden state after the last one)")

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    # torch 2.10's `load_state_dict(assign=True)` DOES shape-check (measured;
    # `Module._load_from_state_dict`, "Shape checks are already done above"),
    # but it blames the checkpoint for disagreeing with a skeleton this loader
    # built from the file's own declaration. Checking here fires before
    # anything is installed and names the declaration as the suspect. Only the
    # declared-dims path needs it: the official-config path's module shapes are
    # the file's by construction.
    skeleton_shapes = None
    if dims is not None:
        skeleton_shapes = {
            name: tuple(tensor.shape)
            for name, tensor in (list(model.named_parameters()) + list(model.named_buffers()))
        }

    with safe_open(te_path, framework="pt", device="cpu") as handle:
        int8_convrot_source_layers = _int8_convrot_layers_from_markers(
            handle, header, path=te_path
        )
        nvfp4_source_layers = _h3_nvfp4_layers_from_markers(handle, header, path=te_path)
        int8_embedding_source_layers = _h3_int8_embedding_layers_from_markers(
            handle, header, path=te_path
        )
        # Unlike the DiT's fused `qkv_proj`, every quantized TE Linear is
        # already a single Linear -- marker coverage is exactly
        # self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj on all 50
        # decoder layers, no fused projection among them (measured; see
        # scratchpad/minimax_h3_te_convrot_verification.md /
        # scratchpad/minimax_h3_te_nvfp4_verification.md) -- so the source
        # key rewritten through `_rewrite_te_key` IS the target module path;
        # no fan-out helper like `_mapped_int8_convrot_layer_configs` is
        # needed. Same for `model.embed_tokens` (a single fixed path).
        int8_convrot_layer_configs = {
            _rewrite_te_key(source): dict(cfg)
            for source, cfg in int8_convrot_source_layers.items()
        }
        nvfp4_layer_configs = {
            _rewrite_te_key(source): dict(cfg)
            for source, cfg in nvfp4_source_layers.items()
        }
        int8_embedding_layer_configs = {
            _rewrite_te_key(source): dict(cfg)
            for source, cfg in int8_embedding_source_layers.items()
        }
        if int8_convrot_layer_configs:
            from core.models.common.convrot_int8_linear import require_convrot_int8_runtime

            require_convrot_int8_runtime()
        if nvfp4_layer_configs:
            from core.models.common.nvfp4_linear import require_nvfp4_runtime

            require_nvfp4_runtime()

        state_dict = {_rewrite_te_key(k): handle.get_tensor(k) for k in header}

        # `Int8Linear.weight_scale` (the base class `ConvRotInt8Linear` swaps
        # in) registers `(out_features,)`; the marker-validated file stores
        # `[out, 1]` (`_supported_int8_convrot_marker` accepts both, the
        # narrower of which this file uses). Reshaped here, on the very
        # tensors both the swap below and the `load_state_dict` after it read,
        # so the two cannot disagree about which shape is expected -- squeezing
        # a copy elsewhere is exactly the "obvious fix" the module docstring
        # above warns turns every guard green on a rotated model. `Int8Embedding`
        # has the identical `(num_embeddings,)` vs `[num_embeddings, 1]` shape
        # gap; `Nvfp4Linear.weight_scale` needs NO reshape, it already stores
        # `[out, K/16]`, matching the module's buffer shape exactly.
        for layer in int8_convrot_layer_configs:
            scale_key = layer + ".weight_scale"
            scale = state_dict.get(scale_key)
            if scale is not None:
                state_dict[scale_key] = scale.reshape(-1)
        for layer in int8_embedding_layer_configs:
            scale_key = layer + ".weight_scale"
            scale = state_dict.get(scale_key)
            if scale is not None:
                state_dict[scale_key] = scale.reshape(-1)

        swappable_layer_configs = {
            **int8_convrot_layer_configs, **nvfp4_layer_configs,
        }

        guard_state_dict = _te_guard_state_dict(
            state_dict,
            swappable_layer_configs=swappable_layer_configs,
            nvfp4_layer_configs=nvfp4_layer_configs,
        )
        _assert_guard_reached(guard_state_dict, label="text encoder", path=te_path)

        # This builder swaps ONLY the validated ConvRot/NVFP4 Linears and the
        # validated plain-int8 embedding -- there is no Int8Linear/Fp8Linear
        # swap for anything else on this component, unlike the DiT. Excluding
        # those layers and running the DiT's own census+verify pattern on the
        # remainder (with an always-0 swap count, since nothing else here is
        # swappable) still catches a scale-only or unscaled quantized tensor
        # the header guard above did not recognize, instead of letting
        # `load_state_dict` cast its codes into a bf16 parameter -- the exact
        # silent failure this module exists to prevent.
        excluded_prefixes = tuple(
            name + "." for name in (*swappable_layer_configs, *int8_embedding_layer_configs)
        )
        scaled_state_dict = {
            key: value for key, value in state_dict.items()
            if not excluded_prefixes or not key.startswith(excluded_prefixes)
        }
        census = quantized_state_dict_report(
            scaled_state_dict, arch="MiniMax-H3", path=te_path, label="text encoder")
        report = scaled_quantization_report(
            census, arch="MiniMax-H3", path=te_path, label="text encoder")
        verify_quantized_swap(
            report, 0, arch="MiniMax-H3", path=te_path, label="text encoder")

        # Plain provenance markers have served their purpose. ConvRot/NVFP4
        # modules retain theirs so a state_dict/export cannot lose the
        # rotation/AWQ contract. `Int8Embedding` retains its own for the same
        # provenance reason.
        retained_marker_layers = {
            **swappable_layer_configs, **int8_embedding_layer_configs,
        }
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.endswith(".comfy_quant")
            or key[: -len(".comfy_quant")] in retained_marker_layers
        }

        if int8_convrot_layer_configs:
            from core.models.common.convrot_int8_linear import swap_linears_to_convrot_int8

            # The file's own dtype (bf16, never cast -- see the docstring
            # above); `compute_dtype` only feeds a bias buffer this arch's
            # quantized layers do not have (Qwen3-VL's q/k/v/o_proj,
            # gate/up/down_proj all carry `attention_bias=False` / no MLP
            # bias here).
            convrot_swapped = swap_linears_to_convrot_int8(
                model, state_dict, int8_convrot_layer_configs, torch.bfloat16
            )
            if convrot_swapped != len(int8_convrot_layer_configs):
                raise RuntimeError(
                    f"the MiniMax-H3 text encoder ({te_path}) ConvRot metadata mapped "
                    f"{len(int8_convrot_layer_configs)} Linear(s), but only "
                    f"{convrot_swapped} module(s) were replaced -- the marker's module "
                    f"paths and the built Qwen3-VL model disagree")
            print(f"[MiniMaxH3Loader] text encoder: {convrot_swapped} ConvRot INT8 "
                  f"Linear(s) kept quantized (comfy-kitchen online activation rotation)")

        if nvfp4_layer_configs:
            from core.models.common.nvfp4_linear import swap_linears_to_nvfp4

            nvfp4_swapped = swap_linears_to_nvfp4(
                model, state_dict, nvfp4_layer_configs, torch.bfloat16
            )
            if nvfp4_swapped != len(nvfp4_layer_configs):
                raise RuntimeError(
                    f"the MiniMax-H3 text encoder ({te_path}) NVFP4 metadata mapped "
                    f"{len(nvfp4_layer_configs)} Linear(s), but only "
                    f"{nvfp4_swapped} module(s) were replaced -- the marker's module "
                    f"paths and the built Qwen3-VL model disagree")
            with_pqs = sum(1 for cfg in nvfp4_layer_configs.values() if cfg.get("has_pre_quant_scale"))
            print(f"[MiniMaxH3Loader] text encoder: {nvfp4_swapped} NVFP4 Linear(s) kept "
                  f"quantized ({with_pqs} with AWQ pre_quant_scale on the activation, "
                  f"comfy-kitchen dequant-on-device)")

        if int8_embedding_layer_configs:
            from core.models.common.int8_embedding import swap_embedding_to_int8

            embedding_swapped = swap_embedding_to_int8(
                model, state_dict, int8_embedding_layer_configs, torch.bfloat16
            )
            if embedding_swapped != len(int8_embedding_layer_configs):
                raise RuntimeError(
                    f"the MiniMax-H3 text encoder ({te_path}) INT8 embedding metadata mapped "
                    f"{len(int8_embedding_layer_configs)} nn.Embedding(s), but only "
                    f"{embedding_swapped} module(s) were replaced -- the marker's module "
                    f"paths and the built Qwen3-VL model disagree")
            print(f"[MiniMaxH3Loader] text encoder: {embedding_swapped} INT8 "
                  f"nn.Embedding(s) kept quantized (gather-then-scale)")

        if skeleton_shapes is not None:
            mismatched = {
                key: f"file {tuple(value.shape)} != model {skeleton_shapes[key]}"
                for key, value in state_dict.items()
                if key in skeleton_shapes and tuple(value.shape) != skeleton_shapes[key]
            }
            if mismatched:
                raise RuntimeError(
                    f"the MiniMax-H3 text encoder ({te_path}) has {len(mismatched)} tensor(s) "
                    f"whose shape contradicts the dims its own minimax_h3_te metadata declares "
                    f"(first 5: {dict(list(mismatched.items())[:5])}).")

        result = model.load_state_dict(state_dict, strict=False, assign=True)
        del state_dict

    _finalize_text_encoder(model, result, te_path, text_only)
    return model, config


def _build_text_encoder_from_gguf(te_path: str, official_dir: Optional[str], tap: Optional[int]):
    """Build the truncated Qwen3-VL over the GGUF's own memory-mapped blocks.

    Same mapping discipline as ``_build_text_encoder``: nothing here copies or
    casts a CPU weight, the Q8_0 blocks stay packed as module buffers and are
    dequantized on the GPU inside ``functional_call``.

    The depth is the projection's ``tap``, because an unconverted file carries
    every block; tensors of blocks at or beyond it are never mapped, so their
    bytes are never touched. There is no ``_guard_component_file`` call: a GGUF
    declares its quantization in its own type tags, and ``plan_gguf_text_encoder``
    refuses every type this loader has no dequantizer for.
    """
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from core.models.minimax_h3.te_gguf_native import (
        open_gguf, plan_gguf_text_encoder, read_gguf_te_declaration, swap_modules_to_gguf_q8,
    )

    if not tap:
        raise ValueError(
            f"{os.path.basename(te_path)} is an unconverted GGUF and carries every decoder "
            f"block, so the depth to read comes from the trained projection's tap -- and none "
            f"was resolved for this load. Build it through "
            f"build_minimax_h3_text_encoder_bundle or load_minimax_h3_from_path, which pair the "
            f"projection first.")
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

    declared = read_gguf_te_declaration(te_path)
    dims = _te_declared_dims(declared)
    if dims is None:
        raise ValueError(f"{te_path} does not declare a complete Qwen3-VL text geometry")
    raw_config["text_config"] = _te_text_config_from_dims(raw_config.get("text_config"), dims)
    print(f"[MiniMaxH3Loader] text encoder: geometry from the GGUF's own KV metadata (hidden "
          f"{dims['hidden_size']}, {dims['num_attention_heads']} heads / "
          f"{dims['num_key_value_heads']} kv, head_dim {dims['head_dim']}, ffn "
          f"{dims['intermediate_size']}, vocab {dims['vocab_size']}) -- the 32B "
          f"text_encoder/config.json is NOT applied")

    config = Qwen3VLConfig(**raw_config)
    config.text_config.num_hidden_layers = int(tap)
    print(f"[MiniMaxH3Loader] text encoder: Qwen3-VL read to decoder layer {tap} of the GGUF's "
          f"{declared['block_count']} (the projection's tap; deeper blocks are never mapped)")

    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    reader = open_gguf(te_path)
    try:
        plan = plan_gguf_text_encoder(reader, int(tap), te_path, _rewrite_te_key)
        linears, embeddings = swap_modules_to_gguf_q8(
            model, plan["linear_configs"], plan["embedding_configs"], torch.bfloat16)
        if (linears, embeddings) != (len(plan["linear_configs"]), len(plan["embedding_configs"])):
            raise RuntimeError(
                f"the MiniMax-H3 text encoder ({te_path}) maps {len(plan['linear_configs'])} "
                f"Q8_0 Linear(s) and {len(plan['embedding_configs'])} embedding(s), but "
                f"{linears}/{embeddings} module(s) were replaced -- the GGUF's tensor names and "
                f"the built Qwen3-VL model disagree")
        result = model.load_state_dict(plan["state_dict"], strict=False, assign=True)
        print(f"[MiniMaxH3Loader] text encoder: {linears} Q8_0 Linear(s) + {embeddings} Q8_0 "
              f"embedding(s) kept packed (dequantized per layer on the GPU), "
              f"{plan['skipped']} tensor(s) never mapped")
        del plan
    finally:
        # The reader itself must not outlive this call: the tensors installed
        # above hold their own views of the same mmap, and that is what the
        # weakref assertion tracks.
        del reader

    _finalize_text_encoder(model, result, te_path, text_only=True)
    return model, config


def _finalize_text_encoder(model, result, te_path: str, text_only: bool) -> None:
    """The post-load assertions and the two deliberately-absent modules.

    Shared by both builders so a GGUF-backed encoder is held to the same
    contract -- including the weakref registry ``assert_no_live_text_encoder``
    reads, which is what proves the previous file's mapping was released.
    """
    import weakref

    unexpected = sorted(result.unexpected_keys)
    missing = sorted(result.missing_keys)
    if unexpected:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder ({te_path}) produced {len(unexpected)} unexpected "
            f"key(s) (first 5: {unexpected[:5]}); the prefix rewrite and the file disagree.")
    unexplained = [key for key in missing
                   if key not in _TE_EXPECTED_MISSING
                   and not (text_only and key.startswith(_TE_VISION_PREFIX))]
    if unexplained:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder ({te_path}) is missing key(s) beyond what the "
            f"truncated read expects ({sorted(_TE_EXPECTED_MISSING)}"
            f"{' plus the vision tower, this file declaring modalities=text' if text_only else ''}): "
            f"{sorted(unexplained)[:5]}. Those parameters were built on "
            f"the meta device and would detonate at the first forward.")

    # The two absent modules are REPLACED rather than left holding meta tensors:
    # neither is used by the layer-N hidden-state read, and a meta parameter in a
    # live model fails far from here.
    model.lm_head = None
    model.model.language_model.norm = nn.Identity()

    # A text-only file's vision tower KEEPS its meta tensors. `encode_presentation`
    # reaches `model.visual` only through `vision_inputs`, which the t2va path
    # never sets; P3 refuses ref2va/fl2va for such an encoder via `te_text_only`.
    stranded = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)
                and not (text_only and n.startswith(_TE_VISION_PREFIX))]
    if stranded:
        raise RuntimeError(
            f"the MiniMax-H3 text encoder from {te_path} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}).")
    if text_only:
        vision_meta = sum(
            1 for _n, t in list(model.named_parameters()) + list(model.named_buffers())
            if getattr(t, "is_meta", False))
        print(f"[MiniMaxH3Loader] text encoder: text-only file; {vision_meta} vision-tower "
              f"tensor(s) left on the meta device (never reached by the t2va path)")

    model.eval().requires_grad_(False)
    _LIVE_TEXT_ENCODER[te_path] = weakref.ref(model)
    # Buffers are included, which is what makes this cover a GGUF-backed
    # encoder: its mapped bytes are Q8_0 buffers, not parameters.
    _LIVE_TEXT_ENCODER_TENSORS[te_path] = tuple(
        weakref.ref(tensor)
        for tensor in list(model.parameters()) + list(model.buffers())
    )


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


def dit_text_dim(dit_path: str) -> int:
    """The DiT's conditioning width, from its header alone (``condition_proj``).

    Read WITHOUT building the transformer so a projection that would not fit is
    refused before the encode and VAE phases, not inside ``context_embedder``
    at the end of them.
    """
    shape = _header_shape(read_safetensors_header(dit_path), "condition_proj.weight")
    if not shape or len(shape) != 2:
        raise ValueError(
            f"the MiniMax-H3 DiT {dit_path} has no 2-D 'condition_proj.weight'; its text "
            f"conditioning width cannot be read from the header.")
    return int(shape[1])


def resolve_minimax_h3_te_projection(
    *,
    te_path: str,
    declared: Dict[str, Any],
    root: Optional[str],
    text_dim: int,
    override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The loaded projection this text encoder needs, or ``None`` if it needs none.

    A file that declares its own dims (a converted small encoder, or a raw
    GGUF) and whose ``hidden_size`` is not the DiT's ``text_dim`` MUST be
    paired: there is no fallback to unprojected conditioning, because that is a
    wrong encode rather than a cheaper one.

    For a raw GGUF the declaration carries ``block_count`` instead of
    ``num_hidden_layers``, and the resolved projection's ``tap`` is what the
    builder then maps -- see ``resolve_te_projection``'s ``available_blocks``.
    """
    from core.models.minimax_h3.te_projection import (
        describe_te_substitution, load_te_projection, resolve_te_projection,
    )

    dims = _te_declared_dims(declared)
    hidden_size = int(dims["hidden_size"]) if dims else text_dim
    if hidden_size == text_dim:
        if override is None:
            return None
        # Refused HERE rather than left to the d_in check below, which would
        # report a width mismatch for what is really "this encoder takes no
        # projection at all".
        raise ValueError(
            f"a text-encoder projection ({os.path.basename(override)}) was named for "
            f"{os.path.basename(te_path)}, whose hidden state is already the DiT's "
            f"{text_dim}-wide conditioning. That encoder is used directly; only a narrower "
            f"converted encoder is projected.")

    blocks = declared.get("block_count")
    spec = resolve_te_projection(
        root=root, te_path=te_path, hidden_size=hidden_size,
        num_hidden_layers=int(declared.get("num_hidden_layers") or 0),
        text_dim=text_dim, override=override,
        available_blocks=int(blocks) if blocks else None,
    )
    projection = load_te_projection(spec)
    print(f"[MiniMaxH3Loader] TE projection: {spec['path']} "
          f"(d_in {spec['d_in']} -> d_out {spec['d_out']}, tap {spec['tap']})")
    # The selection line above names the encoder; this names what it is only
    # usable through, and what is measured about the pair.
    print(f"[MiniMaxH3Loader] {describe_te_substitution(te_path, spec['path'])}")
    return projection


def load_minimax_h3_from_path(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *,
    load_text_encoder: bool = True,
    video_vae_dtype: Optional[torch.dtype] = None,
    te_override: Optional[str] = None,
    te_projection_override: Optional[str] = None,
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

    ``te_override`` names an exact text encoder file, bypassing
    ``MINIMAX_H3_TE_PATTERNS`` and its loadability predicate -- see
    ``detect_minimax_h3_layout``. Whatever this build's ``_build_text_encoder``
    then does with it (load, or refuse with its own quantization-semantics
    error) is unchanged by naming the file explicitly here.

    ``te_projection_override`` names the trained projection to pair with it,
    skipping ``clip_projections/`` discovery. Every pairing check still runs.
    """
    # ``te_override is None`` calls with the ORIGINAL one-argument signature,
    # not with an explicit ``te_override=None`` -- callers (including existing
    # tests) that monkeypatch ``detect_minimax_h3_layout`` with a single-arg
    # stub must keep working when they never asked for an override.
    layout = (detect_minimax_h3_layout(model_path, te_override=te_override)
              if te_override is not None else detect_minimax_h3_layout(model_path))
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
            f"text_encoders/ holding one of {MINIMAX_H3_TE_PATTERNS} (searched in that "
            f"preference order, with a glob fallback).")

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
    print(f"[MiniMaxH3Loader] text encoder: {layout['text_encoder']} "
          f"({layout.get('text_encoder_reason') or 'n/a'})")
    print(f"[MiniMaxH3Loader] configs:      {official}")

    # The pairing gates run BEFORE the encoder is mapped: a mismatch is knowable
    # from two headers, and answering it after 5-48 GiB has been installed is a
    # minute of work spent to reach a message that was available immediately.
    te_declaration: Dict[str, Any] = {}
    te_projection = None
    if load_text_encoder:
        te_declaration = _te_file_declaration(layout["text_encoder"])
        # A file that declares no dims of its own is the shipped 32B: same width
        # as the DiT, nothing to pair, and the DiT header is not read at all.
        if _te_declared_dims(te_declaration) is not None or te_projection_override is not None:
            te_projection = resolve_minimax_h3_te_projection(
                te_path=layout["text_encoder"], declared=te_declaration, root=layout["root"],
                text_dim=dit_text_dim(layout["dit"]), override=te_projection_override,
            )

    # Map the 48 GiB encoder before the smaller component files.  On Windows,
    # doing this last can access-violate inside safetensors/torch storage.
    text_encoder = text_encoder_config = None
    if load_text_encoder:
        text_encoder, text_encoder_config = _build_text_encoder_for(
            layout["text_encoder"], official, te_projection)

    transformer, transformer_config = _build_transformer(layout["dit"], torch_dtype, official)
    # fp16 for the video VAE (see MINIMAX_H3_VIDEO_VAE_DTYPE), float32 for the
    # small audio one -- 0.6 GB, decoded once per generation, nothing to buy.
    vae_dtype = video_vae_dtype or MINIMAX_H3_VIDEO_VAE_DTYPE
    vae, vae_config = _build_video_vae(layout["vae"], official, vae_dtype)
    audio_vae, audio_vae_config = _build_audio_vae(layout["audio_vae"], official, torch.float32)

    tokenizer, processor = _load_tokenizer_and_processor(official)
    scheduler, audio_scheduler = _load_schedulers(official)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[MiniMaxH3Loader] Loaded MiniMax-H3 components (CPU-resident; no sampler wired yet).")

    components = {
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
        "text_encoder_origin": "selected_external" if te_override is not None else "architecture_default",
        # A converted small encoder: no vision tower (P3 refuses ref2va/fl2va on
        # it) and its hidden state is only usable through `te_projection`.
        "te_text_only": str(te_declaration.get("modalities") or "") == "text",
        "te_projection": te_projection,
        "official_dir": official,
    }

    # A substituted encoder's agreement with the released one, measured here the
    # first time this pairing is loaded and stored per installation. Costs a
    # directory listing unless a reference bank exists AND this pairing has no
    # record yet; the measurement itself is seconds (gate G0c: 4.3 s at 4B).
    # It cannot raise -- see `maybe_measure_substitution`.
    from core.models.minimax_h3.te_agreement import maybe_measure_substitution

    maybe_measure_substitution(components)
    return components
