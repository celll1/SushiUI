"""MiniMax Music 3 loader: detection + component build (design doc phases 2, 9 + 10).

The released snapshot ships the model twice: ``official/`` (MiniMax's own
config-and-weight tree, loads into the vendored classes key-for-key, no
remap) and a flat ComfyUI-style repack under ``diffusion_models/`` +
``text_encoders/`` + ``vae/`` (different key names and a fused-QKV /
folded-in-condition-encoder DiT; a merged LM+depth-decoder text encoder with
a pruned-vocabulary variant on top). ``official/`` is the SHIPPED DEFAULT --
every load that does not name a flat file explicitly goes through it
unchanged, key-for-key. See ``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Which
tree the loader reads" / "GGUF weights" for the full investigation this
module and ``core.models.minimax_music3.flat_remap`` were written against.

What this loader reads from the flat tree, briefly:

* a NON-quantized flat DiT file (FP32 or FP16) with a reachable ``official/``
  beside it now LOADS: transformer + condition encoder come from the flat
  file via ``flat_remap``, every other component still comes from
  ``official/`` -- pointing at a flat DiT file selects only the DiT's
  SOURCE, not a different model;
* the flat NON-pruned text encoder is readable by
  ``build_language_model_and_depth_decoder_from_flat_text_encoder`` below,
  and the flat PRUNED-vocabulary text encoder is readable by
  ``build_language_model_and_depth_decoder_from_pruned_flat_text_encoder``
  (design doc phase 10, "The pruned vocabulary" -- see that function's
  docstring for the representation choice: a real ``Qwen3ForCausalLM``
  patched with two extra leaf modules, ``lm_head_pruned`` and
  ``model.embed_tokens_audio``, and its default ``lm_head`` removed). Neither
  is wired into ``load_minimax_music3_from_path``'s directory-detection
  dispatch -- nothing in this loader's detection points AT a text-encoder
  file at all (detection keys off the DiT's tensor signature only);
* ``int8_convrot`` (either the flat DiT or either text encoder) is refused,
  HEADER-ONLY, naming design doc phase 13. The pruned-vocabulary text encoder
  is NO LONGER refused (design doc phase 10 landed) -- only its own dedicated
  builder reads it; handing a pruned file to
  ``build_language_model_and_depth_decoder_from_flat_text_encoder`` (the
  NON-pruned builder) still raises ``PrunedTextEncoderNotSupported``, because
  that specific function's remap genuinely cannot read the pruned layout
  (see ``flat_remap.PrunedTextEncoderNotSupported``'s docstring);
* GGUF containers (design doc phase 11) ARE read here now, via
  ``core.models.common.gguf_container`` (a native reader, no ``gguf`` pip
  dependency -- see that module's docstring). A GGUF DiT file
  (``general.architecture = "minimax_music3"`` metadata plus the flat DiT's
  own tensor-name signature) is accepted at directory-detection time exactly
  where a flat safetensors DiT file already is, and
  ``build_transformer_and_condition_encoder_from_gguf_dit`` routes its
  tensors through ``flat_remap.apply_flat_dit_state_dict`` UNCHANGED -- the
  GGUF DiT's tensor names are identical to the flat safetensors' own. The
  staged ``minimax_music3_dit_BF16.gguf`` (F32 + F16 on disk, no Q8_0) loads;
  any Q8_0 (or other GGML type this reader does not materialize) is refused
  HEADER-ONLY, naming design doc phase 12/13. A GGUF PRUNED text encoder is
  readable by ``build_language_model_and_depth_decoder_from_pruned_gguf_text_
  encoder`` (mirroring the safetensors pruned builder, same "not wired into
  directory-detection dispatch" status as every other text-encoder builder in
  this module) -- the staged
  ``minimax_music3_text_encoder_pruned_Q8_0.gguf`` carries 169 Q8_0 tensors
  and is therefore ALWAYS refused today by that same header-only gate.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.models.common import gguf_container
from core.models.minimax_music3.defaults import (
    EXPECTED_LANGUAGE_MODEL_ROPE_THETA,
    FALLBACK_FRAME_RATE,
    FALLBACK_NUM_CHANNELS_LATENTS,
    FALLBACK_SAMPLING_RATE,
)

# The class MiniMax's own config-only tree declares in modular_model_index.json.
# NOTE the filename: `modular_model_index.json`, NOT `model_index.json` --
# MiniMax-H3's directory-detection probe (`_is_h3_model_index` in
# minimax_h3/loader.py) reads `model_index.json` and therefore cannot fire on
# this tree; the two probes are filename-disjoint by construction, not by luck.
MINIMAX_MUSIC3_PIPELINE_CLASS = "MiniMaxMusic3ModularPipeline"

# Directory names `_resolve_official_dir` looks for beside the flat tree,
# mirroring `minimax_h3.loader._OFFICIAL_DIR_NAMES`'s reasoning: the released
# snapshot names it `official/`, but a user's own re-export might not.
_OFFICIAL_DIR_NAMES = ("official", "MiniMax-Music3", "minimax_music3_official")

# `official/qwen_7B/` is a PERMANENT exclusion (design doc, "Dependency gate"
# and "Quantization"): its `auto_map` targets `modeling_abab.py` /
# `configuration_abab.py`, neither of which is in the snapshot, and its
# 48-shard index is missing a shard. Nothing in this module ever constructs a
# path through it; the language model is always read from
# `official/language_model/`. Named here only so a future maintainer sees the
# exclusion is deliberate rather than an oversight if they go looking for it.
_QWEN_7B_EXCLUDED_SUBDIR = "qwen_7B"

# Every diffusers-class component this loader builds directly from
# `official/<subdir>/`: (subdir, expected `_class_name`, weight-file basename).
# `language_model` and `tokenizer`/`scheduler` are handled separately (they are
# transformers/diffusers-native classes with their own `from_pretrained`).
_DIFFUSERS_COMPONENTS = (
    ("transformer", "MiniMaxMusic3Transformer1DModel"),
    ("condition_encoder", "MiniMaxMusic3ConditionEncoder"),
    ("rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder"),
    ("vocoder", "MiniMaxMusic3Vocoder"),
)

_WEIGHT_BASENAME = "diffusion_pytorch_model"

# Shared by the pre-load JSON gate (`_build_language_model`) and the post-load
# gate (`_assert_language_model_rope_theta`) -- one constant, not two literals
# that can drift apart. 10.0 separates the fp32 inv_freq round-trip artifact
# (measured 999997.4 against a stored 1e6) from an actually different rope
# base (real alternates differ by orders of magnitude, e.g. 10000.0).
_ROPE_THETA_TOLERANCE = 10.0


# ---------------------------------------------------------------------------
# Cheap header reading (no tensor bytes) -- for the single-file key-signature
# probe only. Component loads below read real tensors via
# `core.models.common.single_file_format`, which this loader reuses rather
# than reimplementing.
# ---------------------------------------------------------------------------

def read_safetensors_header(path: str) -> Dict[str, Any]:
    """The JSON header of a safetensors file. ZERO tensor bytes are read."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length {header_len} in {path}")
        return json.loads(fh.read(header_len).decode("utf-8"))


# Key-signature of the FLAT (ComfyUI-repack) DiT, per the design doc's
# "Quantization" section census: 370 `diffusion_transformer.*` tensors plus
# `latent_conditioners.0.{weight,bias}`, `cond_layer_logits`,
# `cond_layer_scale` (374 total) -- the condition encoder folded into the
# same file. Verified against
# `M:/model/minimax-music3/diffusion_models/minimax_music3_dit_fp16.safetensors`.
def keys_look_like_flat_minimax_music3_dit(keys) -> bool:
    keys = list(keys)
    if not any(k.startswith("diffusion_transformer.") for k in keys):
        return False
    return "cond_layer_logits" in keys and any(k.startswith("latent_conditioners.") for k in keys)


def is_minimax_music3_safetensors(path: str) -> bool:
    """``keys_look_like_flat_minimax_music3_dit`` against a file's header. Never raises."""
    try:
        header = read_safetensors_header(path)
        header.pop("__metadata__", None)
        return keys_look_like_flat_minimax_music3_dit(header.keys())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GGUF header reading (design doc phase 11) -- also header-only, via
# ``core.models.common.gguf_container``. ``general.architecture`` is the
# STRONG signal (a GGUF for another model must be refused, not mis-claimed);
# the tensor-name signature additionally tells the DiT file apart from the
# (also ``minimax_music3``-declaring) text-encoder GGUF.
# ---------------------------------------------------------------------------

GGUF_ARCHITECTURE_METADATA_KEY = "general.architecture"
GGUF_EXPECTED_ARCHITECTURE = "minimax_music3"


def is_minimax_music3_gguf_dit(path: str) -> bool:
    """True iff ``path`` is a GGUF container declaring
    ``general.architecture = "minimax_music3"`` AND carrying the flat DiT's
    own tensor-name signature (``keys_look_like_flat_minimax_music3_dit``,
    reused UNCHANGED -- the GGUF DiT's tensor names are identical to the flat
    safetensors DiT's, see ``flat_remap``'s module docstring and the design
    doc's "GGUF weights" section). Never raises -- mirrors
    ``is_minimax_music3_safetensors``'s "probe, don't raise" contract for
    this same detection call site.
    """
    try:
        header = gguf_container.parse_gguf_header(path)
        if header.metadata.get(GGUF_ARCHITECTURE_METADATA_KEY) != GGUF_EXPECTED_ARCHITECTURE:
            return False
        return keys_look_like_flat_minimax_music3_dit(header.tensor_names())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_music3_official_dir(directory: Path) -> bool:
    index = directory / "modular_model_index.json"
    if not index.is_file():
        return False
    try:
        with open(index, encoding="utf-8") as fh:
            return json.load(fh).get("_class_name") == MINIMAX_MUSIC3_PIPELINE_CLASS
    except Exception:
        return False


def _resolve_official_dir(root: Path) -> Optional[str]:
    """MiniMax's config-and-weight tree under ``root`` (or ``root`` itself)."""
    for name in _OFFICIAL_DIR_NAMES:
        candidate = root / name
        if _is_music3_official_dir(candidate):
            return str(candidate)
    if _is_music3_official_dir(root):
        return str(root)
    return None


def detect_minimax_music3_layout(path: str) -> Optional[Dict[str, Optional[str]]]:
    """``{root, official, flat_dit}`` or ``None``.

    Accepts four spellings of the same model, matching the design doc's
    phase-plan item 2 ("directory detection... flat-tree completion by
    sibling-probe into official/") plus item 11's GGUF extension:

    * the flat root (``<root>/diffusion_models/`` + ``vae/`` +
      ``text_encoders/``, with ``official/`` beside it);
    * a DiT ``.safetensors`` inside such a ``diffusion_models/`` (walks up to
      find the root, then the sibling ``official/``);
    * a DiT ``.gguf`` in the same place, detected by
      ``is_minimax_music3_gguf_dit`` instead of
      ``is_minimax_music3_safetensors`` -- same walk-up, same downstream
      handling (``load_minimax_music3_from_path`` picks the GGUF builder over
      the safetensors one by the path's own suffix);
    * MiniMax's config-and-weight ``official/`` directory itself, i.e. one
      whose ``modular_model_index.json`` declares
      ``MiniMaxMusic3ModularPipeline``.

    ``official`` is required either way -- CONFIGS always come from it, and
    it is the only source for every component except a flat DiT file named
    explicitly (see the module docstring). ``flat_dit`` is populated -- to a
    non-``None`` path -- exactly when the caller pointed at a lone DiT file
    rather than at the root or at ``official/`` directly:
    `load_minimax_music3_from_path` uses its presence to source the
    transformer + condition encoder from that file (via ``flat_remap``) when
    ``official`` is also reachable, and to raise a distinct "no reachable
    official/" message when it is not.
    """
    if not path:
        return None
    p = Path(path)

    p_suffix_lower = p.suffix.lower()
    if p.is_file() and (p.suffix == ".safetensors" or p_suffix_lower == ".gguf"):
        is_dit = (
            is_minimax_music3_safetensors(str(p)) if p.suffix == ".safetensors"
            else is_minimax_music3_gguf_dit(str(p))
        )
        if not is_dit:
            return None
        for parent in p.parents:
            if (parent / "diffusion_models").is_dir():
                return {
                    "root": str(parent),
                    "official": _resolve_official_dir(parent),
                    "flat_dit": str(p),
                }
        # A lone DiT file with no reachable `diffusion_models/` root at all.
        return {"root": None, "official": None, "flat_dit": str(p)}

    if not p.is_dir():
        return None

    if _is_music3_official_dir(p):
        # A bare `official/`: return it directly, root is its parent (which may
        # or may not also hold the flat tree -- irrelevant to this loader).
        return {"root": str(p.parent), "official": str(p), "flat_dit": None}

    official = _resolve_official_dir(p)
    if official is not None:
        return {"root": str(p), "official": official, "flat_dit": None}

    return None


# ---------------------------------------------------------------------------
# Component build (official/ tree only -- see module docstring)
# ---------------------------------------------------------------------------

def _read_component_config(official: str, subdir: str, expected_class: str) -> Dict[str, Any]:
    config_path = os.path.join(official, subdir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing {subdir}/config.json. "
            f"That file carries this component's geometry and is not optional."
        )
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    if config.get("_class_name") != expected_class:
        raise ValueError(
            f"MiniMax Music 3's {subdir}/config.json declares _class_name="
            f"{config.get('_class_name')!r}, expected {expected_class!r}. Refusing to build "
            f"an unrelated module from this component's weights."
        )
    return config


def _component_weight_present(official: str, subdir: str) -> bool:
    comp_dir = os.path.join(official, subdir)
    return (
        os.path.isfile(os.path.join(comp_dir, f"{_WEIGHT_BASENAME}.safetensors"))
        or os.path.isfile(os.path.join(comp_dir, f"{_WEIGHT_BASENAME}.safetensors.index.json"))
    )


def _build_diffusers_component(
    official: str,
    subdir: str,
    cls,
    expected_class: str,
    torch_dtype: torch.dtype,
) -> tuple:
    """Meta-device construction + strict ``assign=True`` load for one vendored component.

    Mirrors ``minimax_h3.loader``'s pattern (see its build helpers, and the
    stranded-meta-tensor sweep at the end of each): ``init_empty_weights()`` +
    ``cls.from_config(config)`` + ``load_state_dict(..., assign=True)``, dtype
    cast PER KEY on the way in rather than after the module is built.
    """
    from accelerate import init_empty_weights

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    from core.models.common.single_file_format import load_component_state_dict

    config = _read_component_config(official, subdir, expected_class)
    comp_dir = os.path.join(official, subdir)

    state_dict = load_component_state_dict(comp_dir, _WEIGHT_BASENAME)
    # Defensive: see module docstring -- official/ files are never observed
    # quantized, but the check is cheap and matches the design doc's
    # BF16/FP16-only Phase 1 rule unconditionally rather than by omission.
    refuse_quantized_state_dict(state_dict, arch="MiniMax Music 3", path=comp_dir, label=subdir)
    # Only floating tensors are cast (matches minimax_h3.loader's guard): an
    # int64/bool persistent buffer cast to torch_dtype would silently corrupt
    # it instead of erroring. No such tensor exists in these components today,
    # but the guard is what makes that "today" instead of an invariant.
    state_dict = {k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v)
                  for k, v in state_dict.items()}

    with init_empty_weights():
        model = cls.from_config(config)
    model.load_state_dict(state_dict, strict=True, assign=True)

    stranded = _stranded_meta_tensors(model)
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's {subdir} from {comp_dir} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}); it would fail at the first "
            f"forward."
        )
    model.eval()
    model.requires_grad_(False)
    return model, config


# ---------------------------------------------------------------------------
# Flat (ComfyUI-repack) safetensors -- design doc phase 9. Configs still come
# from `official/`; only the WEIGHTS for the components named below come from
# the flat file. See `core.models.minimax_music3.flat_remap` for the key
# remap itself and the module docstring above for what is and is not wired.
# ---------------------------------------------------------------------------

def _stranded_meta_tensors(model) -> List[str]:
    return [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if getattr(t, "is_meta", False)
    ]


def _build_module_from_remapped_state_dict(cls, config: Dict[str, Any], state_dict, torch_dtype: torch.dtype, *, label: str):
    """``init_empty_weights()`` + ``from_config`` + strict ``assign=True`` load.

    Shared by the flat DiT and flat text-encoder builders below; identical in
    shape to ``_build_diffusers_component``'s own pattern, just parameterized
    over an already-remapped state dict instead of one read straight off
    disk.
    """
    from accelerate import init_empty_weights

    from core.models.minimax_music3.flat_remap import (
        assert_state_dict_matches_module_keys,
        expected_module_state_dict_keys,
    )

    with init_empty_weights():
        model = cls.from_config(config)

    cast_state_dict = {
        k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v)
        for k, v in state_dict.items()
    }
    assert_state_dict_matches_module_keys(
        cast_state_dict.keys(), expected_module_state_dict_keys(model), component=label,
    )
    model.load_state_dict(cast_state_dict, strict=True, assign=True)

    stranded = _stranded_meta_tensors(model)
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's {label} (flat safetensors source) still holds "
            f"{len(stranded)} meta tensor(s) after loading (first 5: {stranded[:5]}); it "
            f"would fail at the first forward."
        )
    model.eval()
    model.requires_grad_(False)
    return model


def _header_looks_quantized(header_keys) -> bool:
    """HEADER-ONLY pre-check: a key suffix alone declaring weight-only
    quantization or AWQ input smoothing. Not a replacement for
    ``refuse_quantized_state_dict`` (still run afterward, unconditionally,
    on the file's real state dict) -- a fast path so an obviously-quantized
    multi-GB file is never read just to be refused.
    """
    from core.models.common.quantized_checkpoint_guard import (
        COMFY_QUANT_MARKER_SUFFIX,
        PRE_QUANT_SCALE_SUFFIX,
        QUANT_SCALE_SUFFIX,
    )

    return any(
        k.endswith(QUANT_SCALE_SUFFIX) or k.endswith(COMFY_QUANT_MARKER_SUFFIX)
        or k.endswith(PRE_QUANT_SCALE_SUFFIX)
        for k in header_keys
    )


def build_transformer_and_condition_encoder_from_flat_dit(
    flat_dit_path: str,
    official: str,
    torch_dtype: torch.dtype,
) -> tuple:
    """The flow-matching transformer + condition encoder from a flat DiT file.

    ``flat_dit_path`` is a NON-quantized flat DiT safetensors file
    (``minimax_music3_dit_{fp32,fp16}.safetensors``); the ``int8_convrot``
    variant is refused here by ``refuse_quantized_state_dict``, the same
    guard every ``official/`` component load runs (design doc phase 13 is
    what would replace this refusal with a swap-in quantized Linear). Configs
    for BOTH components come from ``official/`` -- the flat file carries only
    weights, no config.json -- matching every other component this loader
    builds.

    Returns ``(transformer, transformer_config, condition_encoder,
    condition_encoder_config)``. The condition encoder is built in float32
    regardless of ``torch_dtype``, matching ``_build_diffusers_component``'s
    judgment for the ``official/`` path (tiny, 4 tensors, sits directly on the
    conditioning precision path).
    """
    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    from core.models.common.single_file_format import read_state_dict
    from core.models.minimax_music3.flat_remap import apply_flat_dit_state_dict
    from core.models.minimax_music3.vendor import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3Transformer1DModel,
    )

    # Header-only fast path: refuse an obviously-quantized file (e.g.
    # minimax_music3_dit_int8_convrot.safetensors) before reading a single
    # tensor byte of it.
    if _header_looks_quantized(read_safetensors_header(flat_dit_path).keys()):
        raise RuntimeError(
            f"the MiniMax Music 3 flat DiT checkpoint ({flat_dit_path}) declares weight-only "
            f"quantization in its header (a '.weight_scale' or '.comfy_quant' key), and the "
            f"MiniMax Music 3 loader does not support quantized flat checkpoints (design doc "
            f"phase 13, 'INT8 ConvRot'). Load an unquantized flat DiT "
            f"(minimax_music3_dit_{{fp32,fp16}}.safetensors) instead."
        )

    flat_state_dict, _metadata = read_state_dict(flat_dit_path)
    refuse_quantized_state_dict(
        flat_state_dict, arch="MiniMax Music 3", path=flat_dit_path, label="flat DiT",
    )

    remapped = apply_flat_dit_state_dict(flat_state_dict)
    del flat_state_dict

    transformer_config = _read_component_config(official, "transformer", "MiniMaxMusic3Transformer1DModel")
    condition_encoder_config = _read_component_config(official, "condition_encoder", "MiniMaxMusic3ConditionEncoder")

    transformer = _build_module_from_remapped_state_dict(
        MiniMaxMusic3Transformer1DModel, transformer_config, remapped["transformer"], torch_dtype,
        label="transformer",
    )
    condition_encoder = _build_module_from_remapped_state_dict(
        MiniMaxMusic3ConditionEncoder, condition_encoder_config, remapped["condition_encoder"], torch.float32,
        label="condition_encoder",
    )
    return transformer, transformer_config, condition_encoder, condition_encoder_config


def build_transformer_and_condition_encoder_from_gguf_dit(
    gguf_dit_path: str,
    official: str,
    torch_dtype: torch.dtype,
) -> tuple:
    """The flow-matching transformer + condition encoder from a GGUF DiT file
    (design doc phase 11).

    Mirrors ``build_transformer_and_condition_encoder_from_flat_dit`` -- same
    remap (``flat_remap.apply_flat_dit_state_dict``, UNCHANGED: the GGUF
    DiT's tensor names are identical to the flat safetensors' own), same
    "configs from ``official/``, weights from this file" contract, same
    return shape. What differs is the state-dict SOURCE (a lazy
    ``gguf_container.GGUFStateDict`` instead of an eagerly-read safetensors
    dict) and the quantization gate (this file's own declared GGML tensor
    TYPES, via ``gguf_container.refuse_unsupported_tensor_types``, HEADER-ONLY
    before any tensor byte is read -- GGUF has no ``.weight_scale`` sibling
    convention). The staged ``minimax_music3_dit_BF16.gguf`` is F32 + F16 on
    disk (no Q8_0) and loads.

    What the "BF16" label actually means for THIS file's precision -- and why
    it is, per-tensor, NOT the same rounding as the flat "fp16" safetensors
    DiT -- is investigated and stated once, in
    ``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "GGUF weights" (item 11's
    status entry); this docstring does not repeat that argument.

    Returns ``(transformer, transformer_config, condition_encoder,
    condition_encoder_config)``, same as the safetensors builder.
    """
    from core.models.minimax_music3.flat_remap import apply_flat_dit_state_dict
    from core.models.minimax_music3.vendor import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3Transformer1DModel,
    )

    header = gguf_container.parse_gguf_header(gguf_dit_path)
    # Header-only fast path, same ordering rule as the safetensors builder's
    # `_header_looks_quantized` gate: refuse before opening the data section.
    gguf_container.refuse_unsupported_tensor_types(header, arch="MiniMax Music 3", label="flat DiT")

    state = gguf_container.GGUFStateDict(header, arch="MiniMax Music 3", label="flat DiT")
    try:
        remapped = apply_flat_dit_state_dict(state)
    finally:
        state.close()

    transformer_config = _read_component_config(official, "transformer", "MiniMaxMusic3Transformer1DModel")
    condition_encoder_config = _read_component_config(official, "condition_encoder", "MiniMaxMusic3ConditionEncoder")

    transformer = _build_module_from_remapped_state_dict(
        MiniMaxMusic3Transformer1DModel, transformer_config, remapped["transformer"], torch_dtype,
        label="transformer",
    )
    condition_encoder = _build_module_from_remapped_state_dict(
        MiniMaxMusic3ConditionEncoder, condition_encoder_config, remapped["condition_encoder"], torch.float32,
        label="condition_encoder",
    )
    return transformer, transformer_config, condition_encoder, condition_encoder_config


def build_language_model_and_depth_decoder_from_flat_text_encoder(
    flat_text_encoder_path: str,
    official: str,
    torch_dtype: torch.dtype,
):
    """The ``Qwen3ForCausalLM`` + RVQ depth decoder from a flat, NON-pruned text encoder file.

    Not wired into ``load_minimax_music3_from_path``'s dispatch -- see the
    module docstring for why. Tested with a tiny real Qwen3 + RVQ depth
    decoder round-trip in ``backend/tests/minimax_music3_loader_test.py``
    (``test_flat_text_encoder_builder_round_trip``).

    Raises ``core.models.minimax_music3.flat_remap.PrunedTextEncoderNotSupported``
    for the pruned-vocabulary variant (design doc phase 10) and
    ``RuntimeError`` (via ``refuse_quantized_state_dict``) for the
    ``int8_convrot`` variant (design doc phase 13).

    ``Qwen3ForCausalLM`` is built via ``AutoModelForCausalLM.from_config`` +
    meta + strict ``assign=True`` load, mirroring the diffusers-class
    components rather than ``_build_language_model``'s ``from_pretrained``
    path -- there is no sharded file on disk for THIS source; the state dict
    is already in memory, remapped from the flat file.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    from core.models.common.single_file_format import read_state_dict
    from core.models.minimax_music3.flat_remap import (
        apply_flat_text_encoder_state_dict,
        raise_if_pruned_flat_text_encoder,
    )
    from core.models.minimax_music3.vendor import MiniMaxMusic3RVQDepthDecoder

    # Header-only fast path, in the order a caller would want to know about
    # them: pruned-vocabulary first (the more informative refusal -- it names
    # the phase that would fix it), then quantization. Neither reads a single
    # tensor byte of what can be an 18 GB file.
    header_keys = list(read_safetensors_header(flat_text_encoder_path).keys())
    raise_if_pruned_flat_text_encoder(header_keys)
    if _header_looks_quantized(header_keys):
        raise RuntimeError(
            f"the MiniMax Music 3 flat text encoder checkpoint ({flat_text_encoder_path}) "
            f"declares weight-only quantization in its header (a '.weight_scale' or "
            f"'.comfy_quant' key), and the MiniMax Music 3 loader does not support quantized "
            f"flat checkpoints (design doc phase 13, 'INT8 ConvRot'). Load an unquantized, "
            f"non-pruned flat text encoder (minimax_music3_text_encoder_bf16.safetensors) "
            f"instead."
        )

    # Cheap config reads + the rope-theta gate BEFORE the heavy read -- same
    # ordering rule `_build_language_model` follows (fail on a JSON read, not
    # after an 18 GB load): `config.rope_theta` is None on transformers 5.1
    # for this config form, so `rope_parameters['rope_theta']` is what must be
    # checked -- see `_assert_language_model_rope_theta`'s docstring.
    lm_config = AutoConfig.from_pretrained(os.path.join(official, "language_model"))
    rope_parameters = getattr(lm_config, "rope_parameters", None)
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model config.rope_parameters['rope_theta'] is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}. Checked from "
            f"official/language_model/config.json BEFORE reading the flat text encoder's "
            f"multi-GB weights."
        )
    depth_config = _read_component_config(official, "rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder")

    flat_state_dict, _metadata = read_state_dict(flat_text_encoder_path)
    refuse_quantized_state_dict(
        flat_state_dict, arch="MiniMax Music 3", path=flat_text_encoder_path, label="flat text encoder",
    )

    remapped = apply_flat_text_encoder_state_dict(flat_state_dict)  # raises PrunedTextEncoderNotSupported
    del flat_state_dict

    language_model = _build_module_from_remapped_state_dict(
        AutoModelForCausalLM, lm_config, remapped["language_model"], torch_dtype,
        label="language_model",
    )
    # Defense in depth, same as `_build_language_model`'s own post-load
    # re-assert: against the LOADED model's own config object, not just the
    # pre-load JSON read above.
    _assert_language_model_rope_theta(language_model)

    rvq_depth_decoder = _build_module_from_remapped_state_dict(
        MiniMaxMusic3RVQDepthDecoder, depth_config, remapped["rvq_depth_decoder"], torch_dtype,
        label="rvq_depth_decoder",
    )
    return language_model, rvq_depth_decoder, depth_config


def build_language_model_and_depth_decoder_from_pruned_flat_text_encoder(
    flat_text_encoder_path: str,
    official: str,
    torch_dtype: torch.dtype,
):
    """The PRUNED-vocabulary flat text encoder (design doc phase 10) -- a real
    ``Qwen3ForCausalLM`` with its default ``lm_head`` removed and two extra leaf modules
    (``lm_head_pruned``, ``model.embed_tokens_audio``) attached, ``config.vocab_size`` set to
    the checkpoint's own text-row count; not a subclass or hand-rolled wrapper, because every
    transformer layer here is bit-identical to ``official/language_model``'s -- see the design
    doc's phase-10 section for the full justification and the numeric proof.
    ``core.models.minimax_music3.vocab_view.resolve_vocab_view`` detects this patching
    (``hasattr(language_model, "lm_head_pruned")``) at generation time and routes accordingly.

    TRAP: ``language_model.save_pretrained()`` succeeds on this patched model (writes
    ``vocab_size: 151675`` and the two patched keys) and a later plain ``from_pretrained()``
    would silently rebuild a random 200,000-wide ``lm_head`` and only WARN about the two
    unexpected keys -- do not round-trip a pruned-loaded language model through
    ``save_pretrained``/``from_pretrained``.

    Mirrors ``build_language_model_and_depth_decoder_from_flat_text_encoder``'s shape; NOT
    wired into ``load_minimax_music3_from_path``'s directory-detection dispatch, same status
    as that function -- see the module docstring.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    from core.models.common.single_file_format import read_state_dict
    from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET
    from core.models.minimax_music3.flat_remap import (
        _PRUNED_TELLS,
        assert_state_dict_matches_module_keys,
        expected_module_state_dict_keys,
        is_pruned_flat_text_encoder,
    )
    from core.models.minimax_music3.pruned_text_encoder_remap import (
        AUDIO_HEAD_VOCAB_SIZE,
        SEMANTIC_VOCAB_SIZE,
        apply_pruned_text_encoder_state_dict,
    )
    from core.models.minimax_music3.vendor import MiniMaxMusic3RVQDepthDecoder

    # Header-only gates, in the order a caller would want to know about them: "is this even the
    # pruned layout" first (a caller that reaches this function with the NON-pruned file gets a
    # message naming the OTHER builder, not a confusing remap failure), then quantization.
    # Neither reads a single tensor byte of what can be a 16.7 GB file.
    header_keys = list(read_safetensors_header(flat_text_encoder_path).keys())
    if not is_pruned_flat_text_encoder(header_keys):
        raise ValueError(
            f"{flat_text_encoder_path!r} does not carry any of the pruned-vocabulary tells "
            f"({sorted(_PRUNED_TELLS)}) in its header -- it looks like the NON-pruned flat text "
            f"encoder. Use build_language_model_and_depth_decoder_from_flat_text_encoder for that "
            f"file instead."
        )
    if _header_looks_quantized(header_keys):
        raise RuntimeError(
            f"the MiniMax Music 3 pruned flat text encoder checkpoint ({flat_text_encoder_path}) "
            f"declares weight-only quantization in its header (a '.weight_scale' or "
            f"'.comfy_quant' key), and the MiniMax Music 3 loader does not support quantized "
            f"flat checkpoints (design doc phase 13, 'INT8 ConvRot'). Load the unquantized pruned "
            f"flat text encoder (minimax_music3_text_encoder_pruned_bf16.safetensors) instead."
        )

    # Cheap config reads + the rope-theta gate BEFORE the heavy read -- same ordering rule
    # `_build_language_model` and the non-pruned builder above follow.
    lm_config_path = os.path.join(official, "language_model", "config.json")
    if not os.path.isfile(lm_config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing language_model/config.json."
        )
    with open(lm_config_path, encoding="utf-8") as fh:
        lm_config_dict = json.load(fh)
    rope_parameters = lm_config_dict.get("rope_parameters")
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model config.rope_parameters['rope_theta'] is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}. Checked from "
            f"official/language_model/config.json BEFORE reading the pruned flat text encoder's "
            f"multi-GB weights."
        )
    depth_config = _read_component_config(official, "rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder")

    flat_state_dict, _metadata = read_state_dict(flat_text_encoder_path)
    refuse_quantized_state_dict(
        flat_state_dict, arch="MiniMax Music 3", path=flat_text_encoder_path, label="pruned flat text encoder",
    )

    # Shape census BEFORE remapping -- measured from the checkpoint's own tensors, not assumed
    # (see the module-level docstring's "verify from evidence" convention). `prefill_rows` is
    # cross-checked against `AUDIO_CODE_OFFSET`: that constant is DEFINED as "where audio codes
    # begin in the merged (full-vocab) embedding table", which is only meaningful if it equals
    # this checkpoint's own count of text rows -- a mismatch here means either constant is stale
    # for whatever produced this file, and every other AUDIO_CODE_OFFSET-derived assumption in
    # this codebase (SEMANTIC_VOCAB_SIZE's placement, etc.) would be silently wrong too.
    hidden_size = int(lm_config_dict["hidden_size"])
    prefill_rows = int(flat_state_dict["model.embed_tokens_prefill.weight"].shape[0])
    if prefill_rows != AUDIO_CODE_OFFSET:
        raise ValueError(
            f"MiniMax Music 3 pruned text encoder: model.embed_tokens_prefill has {prefill_rows} "
            f"rows, expected AUDIO_CODE_OFFSET ({AUDIO_CODE_OFFSET}). These must be the same "
            f"number by construction (AUDIO_CODE_OFFSET is defined as the text-vocabulary size); "
            f"a mismatch means this file was built against a different checkpoint revision than "
            f"the one docs/guides/MINIMAX_MUSIC3_DESIGN.md was written against."
        )
    audio_rows = int(flat_state_dict["model.embed_tokens_audio.weight"].shape[0])
    if audio_rows != SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"MiniMax Music 3 pruned text encoder: model.embed_tokens_audio has {audio_rows} "
            f"rows, expected SEMANTIC_VOCAB_SIZE ({SEMANTIC_VOCAB_SIZE})."
        )
    head_rows = int(flat_state_dict["model.lm_head_pruned.weight"].shape[0])
    if head_rows != AUDIO_HEAD_VOCAB_SIZE:
        raise ValueError(
            f"MiniMax Music 3 pruned text encoder: model.lm_head_pruned has {head_rows} rows, "
            f"expected SEMANTIC_VOCAB_SIZE + 1 ({AUDIO_HEAD_VOCAB_SIZE})."
        )

    remapped = apply_pruned_text_encoder_state_dict(flat_state_dict, lm_config_dict)
    del flat_state_dict

    lm_hf_config = AutoConfig.from_pretrained(os.path.join(official, "language_model"))
    # See this function's docstring: the checkpoint's OWN embed_tokens size, not official/'s
    # merged 200,000 -- `lm_hf_config.vocab_size` would otherwise size `model.embed_tokens` (and
    # the default `lm_head` this function immediately deletes) to a value the pruned file never
    # populates 48,325 rows of.
    lm_hf_config.vocab_size = prefill_rows

    with init_empty_weights():
        language_model = AutoModelForCausalLM.from_config(lm_hf_config)
        del language_model.lm_head
        language_model.lm_head_pruned = torch.nn.Linear(hidden_size, AUDIO_HEAD_VOCAB_SIZE, bias=False)
        language_model.model.embed_tokens_audio = torch.nn.Embedding(SEMANTIC_VOCAB_SIZE, hidden_size)

    lm_cast_state_dict = {
        k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v)
        for k, v in remapped["language_model"].items()
    }
    assert_state_dict_matches_module_keys(
        lm_cast_state_dict.keys(), expected_module_state_dict_keys(language_model),
        component="language_model (pruned)",
    )
    language_model.load_state_dict(lm_cast_state_dict, strict=True, assign=True)
    stranded = _stranded_meta_tensors(language_model)
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's language_model (pruned flat text encoder source) still holds "
            f"{len(stranded)} meta tensor(s) after loading (first 5: {stranded[:5]}); it would "
            f"fail at the first forward."
        )
    language_model.eval()
    language_model.requires_grad_(False)
    # Defense in depth, same as `_build_language_model`'s / the non-pruned builder's own
    # post-load re-assert: against the LOADED model's own config object, not just the pre-load
    # JSON read above.
    _assert_language_model_rope_theta(language_model)

    rvq_depth_decoder = _build_module_from_remapped_state_dict(
        MiniMaxMusic3RVQDepthDecoder, depth_config, remapped["rvq_depth_decoder"], torch_dtype,
        label="rvq_depth_decoder (pruned)",
    )
    return language_model, rvq_depth_decoder, depth_config


def build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(
    gguf_text_encoder_path: str,
    official: str,
    torch_dtype: torch.dtype,
):
    """The PRUNED-vocabulary flat text encoder (design doc phase 10's remap),
    read from a GGUF container (design doc phase 11) instead of safetensors.

    Mirrors ``build_language_model_and_depth_decoder_from_pruned_flat_text_
    encoder``'s shape, gate ordering and representation choice (a real
    ``Qwen3ForCausalLM``, ``lm_head`` removed, ``lm_head_pruned`` /
    ``model.embed_tokens_audio`` attached) exactly; NOT wired into
    ``load_minimax_music3_from_path``'s directory-detection dispatch, same
    status as every other text-encoder builder in this module -- see the
    module docstring.

    Refuses HEADER-ONLY (``gguf_container.refuse_unsupported_tensor_types``,
    no tensor byte read) for any GGML type this reader does not materialize,
    checked BEFORE the rope-theta config gate and before the data section is
    opened at all. The staged ``minimax_music3_text_encoder_pruned_Q8_0.gguf``
    carries 169 Q8_0 tensors (of 328 total; plus 155 F32 + 4 BF16) and is
    therefore ALWAYS refused today by this gate -- Q8_0 residency is design
    doc phase 12. A future all-F32/F16/BF16 pruned GGUF text encoder would
    proceed past this gate and load through the exact same
    ``pruned_text_encoder_remap.apply_pruned_text_encoder_state_dict`` the
    safetensors path uses, unchanged -- kept implemented and tested here
    (a tiny all-unquantized fixture) rather than stopping at the refusal, so
    item 12 needs no new code in THIS function, only Q8_0 support in
    ``gguf_container``.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET
    from core.models.minimax_music3.flat_remap import (
        _PRUNED_TELLS,
        assert_state_dict_matches_module_keys,
        expected_module_state_dict_keys,
        is_pruned_flat_text_encoder,
    )
    from core.models.minimax_music3.pruned_text_encoder_remap import (
        AUDIO_HEAD_VOCAB_SIZE,
        SEMANTIC_VOCAB_SIZE,
        apply_pruned_text_encoder_state_dict,
    )
    from core.models.minimax_music3.vendor import MiniMaxMusic3RVQDepthDecoder

    header = gguf_container.parse_gguf_header(gguf_text_encoder_path)
    if header.metadata.get(GGUF_ARCHITECTURE_METADATA_KEY) != GGUF_EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"{gguf_text_encoder_path!r} does not declare "
            f"{GGUF_ARCHITECTURE_METADATA_KEY}={GGUF_EXPECTED_ARCHITECTURE!r} in its GGUF "
            f"metadata (found {header.metadata.get(GGUF_ARCHITECTURE_METADATA_KEY)!r}) -- "
            f"refusing to guess this is a MiniMax Music 3 checkpoint."
        )
    header_keys = header.tensor_names()
    if not is_pruned_flat_text_encoder(header_keys):
        raise ValueError(
            f"{gguf_text_encoder_path!r} does not carry any of the pruned-vocabulary tells "
            f"({sorted(_PRUNED_TELLS)}) in its tensor names -- this builder reads only the "
            f"pruned-vocabulary GGUF layout; no non-pruned GGUF text-encoder builder exists "
            f"(design doc phase 11 covers the pruned distribution's own tensor names only)."
        )
    # HEADER-ONLY refusal, before the rope-theta config read below and before
    # opening the data section at all -- proves the file's 9.59 GB / 169
    # Q8_0-of-328 tensors are never touched.
    gguf_container.refuse_unsupported_tensor_types(
        header, arch="MiniMax Music 3", label="pruned text encoder",
    )

    # Cheap config read + the rope-theta gate BEFORE the heavy read -- same
    # ordering rule the safetensors pruned builder follows.
    lm_config_path = os.path.join(official, "language_model", "config.json")
    if not os.path.isfile(lm_config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing language_model/config.json."
        )
    with open(lm_config_path, encoding="utf-8") as fh:
        lm_config_dict = json.load(fh)
    rope_parameters = lm_config_dict.get("rope_parameters")
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model config.rope_parameters['rope_theta'] is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}. Checked from "
            f"official/language_model/config.json BEFORE reading the pruned GGUF text "
            f"encoder's tensor data."
        )
    depth_config = _read_component_config(official, "rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder")

    state = gguf_container.GGUFStateDict(header, arch="MiniMax Music 3", label="pruned text encoder")
    try:
        # Shape census BEFORE remapping, same cross-check the safetensors
        # pruned builder runs against `AUDIO_CODE_OFFSET` -- see that
        # function's comment for why a mismatch here means a stale constant,
        # not a checkpoint bug.
        hidden_size = int(lm_config_dict["hidden_size"])
        prefill_rows = int(state["model.embed_tokens_prefill.weight"].shape[0])
        if prefill_rows != AUDIO_CODE_OFFSET:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF text encoder: model.embed_tokens_prefill has "
                f"{prefill_rows} rows, expected AUDIO_CODE_OFFSET ({AUDIO_CODE_OFFSET})."
            )
        audio_rows = int(state["model.embed_tokens_audio.weight"].shape[0])
        if audio_rows != SEMANTIC_VOCAB_SIZE:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF text encoder: model.embed_tokens_audio has "
                f"{audio_rows} rows, expected SEMANTIC_VOCAB_SIZE ({SEMANTIC_VOCAB_SIZE})."
            )
        head_rows = int(state["model.lm_head_pruned.weight"].shape[0])
        if head_rows != AUDIO_HEAD_VOCAB_SIZE:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF text encoder: model.lm_head_pruned has "
                f"{head_rows} rows, expected SEMANTIC_VOCAB_SIZE + 1 ({AUDIO_HEAD_VOCAB_SIZE})."
            )

        remapped = apply_pruned_text_encoder_state_dict(state, lm_config_dict)
    finally:
        state.close()

    lm_hf_config = AutoConfig.from_pretrained(os.path.join(official, "language_model"))
    lm_hf_config.vocab_size = prefill_rows

    with init_empty_weights():
        language_model = AutoModelForCausalLM.from_config(lm_hf_config)
        del language_model.lm_head
        language_model.lm_head_pruned = torch.nn.Linear(hidden_size, AUDIO_HEAD_VOCAB_SIZE, bias=False)
        language_model.model.embed_tokens_audio = torch.nn.Embedding(SEMANTIC_VOCAB_SIZE, hidden_size)

    lm_cast_state_dict = {
        k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v)
        for k, v in remapped["language_model"].items()
    }
    assert_state_dict_matches_module_keys(
        lm_cast_state_dict.keys(), expected_module_state_dict_keys(language_model),
        component="language_model (pruned GGUF)",
    )
    language_model.load_state_dict(lm_cast_state_dict, strict=True, assign=True)
    stranded = _stranded_meta_tensors(language_model)
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's language_model (pruned GGUF text encoder source) still holds "
            f"{len(stranded)} meta tensor(s) after loading (first 5: {stranded[:5]}); it would "
            f"fail at the first forward."
        )
    language_model.eval()
    language_model.requires_grad_(False)
    _assert_language_model_rope_theta(language_model)

    rvq_depth_decoder = _build_module_from_remapped_state_dict(
        MiniMaxMusic3RVQDepthDecoder, depth_config, remapped["rvq_depth_decoder"], torch_dtype,
        label="rvq_depth_decoder (pruned GGUF)",
    )
    return language_model, rvq_depth_decoder, depth_config


def build_language_model_and_depth_decoder_from_pruned_gguf_q8_0_text_encoder(
    gguf_text_encoder_path: str,
    official: str,
    torch_dtype: torch.dtype,
):
    """The PRUNED-vocabulary GGUF text encoder's Q8_0 tensors, kept PACKED
    (design doc phase 12) instead of being refused the way
    ``build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder``
    (above) refuses them. Mirrors that function's shape and gate ordering
    exactly except for the one difference this docstring exists to explain.

    Every Q8_0-typed tensor (169 in the real staged
    ``minimax_music3_text_encoder_pruned_Q8_0.gguf`` -- every quantized
    Linear weight in both the language model and the RVQ depth decoder,
    INCLUDING ``lm_head_pruned``; see
    ``pruned_text_encoder_q8_0_remap``'s module docstring for the full
    census) is read PACKED via ``pruned_text_encoder_q8_0_remap.
    apply_pruned_text_encoder_state_dict_packed`` and installed as a
    ``core.models.common.gguf_q8_0_linear.GGUFQ8_0Linear`` -- weight-only
    quantized, dequantized ONCE PER DEVICE MOVE rather than once per forward
    (see that module's docstring for why: the AR loop that owns this text
    encoder calls it up to ~9,000 times per generation, so a per-forward
    dequant of an 8B-parameter stack is not viable). Every F32/BF16 tensor
    (norms, the three vocabulary tables) loads exactly as the dense pruned
    GGUF builder loads them, through the SAME ``load_state_dict`` call.

    ANY OTHER unsupported GGML type (this checkpoint carries none; a
    hypothetical Q4_0 sibling would) is still refused HEADER-ONLY -- Q8_0 is
    popped out of the refusal set FIRST, so this function's tolerance is
    exactly one type wider than the dense builder's, not "anything goes".

    NOT wired into ``load_minimax_music3_from_path``'s directory-detection
    dispatch, same status as every other text-encoder builder in this
    module -- see the module docstring.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from core.models.common.gguf_q8_0_linear import install_packed_q8_0_linears
    from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET
    from core.models.minimax_music3.flat_remap import (
        _PRUNED_TELLS,
        assert_state_dict_matches_module_keys,
        expected_module_state_dict_keys,
        is_pruned_flat_text_encoder,
    )
    from core.models.minimax_music3.pruned_text_encoder_q8_0_remap import (
        PackedQ8_0Weight,
        apply_pruned_text_encoder_state_dict_packed,
    )
    from core.models.minimax_music3.pruned_text_encoder_remap import (
        AUDIO_HEAD_VOCAB_SIZE,
        SEMANTIC_VOCAB_SIZE,
    )
    from core.models.minimax_music3.vendor import MiniMaxMusic3RVQDepthDecoder

    header = gguf_container.parse_gguf_header(gguf_text_encoder_path)
    if header.metadata.get(GGUF_ARCHITECTURE_METADATA_KEY) != GGUF_EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"{gguf_text_encoder_path!r} does not declare "
            f"{GGUF_ARCHITECTURE_METADATA_KEY}={GGUF_EXPECTED_ARCHITECTURE!r} in its GGUF "
            f"metadata (found {header.metadata.get(GGUF_ARCHITECTURE_METADATA_KEY)!r}) -- "
            f"refusing to guess this is a MiniMax Music 3 checkpoint."
        )
    header_keys = header.tensor_names()
    if not is_pruned_flat_text_encoder(header_keys):
        raise ValueError(
            f"{gguf_text_encoder_path!r} does not carry any of the pruned-vocabulary tells "
            f"({sorted(_PRUNED_TELLS)}) in its tensor names -- this builder reads only the "
            f"pruned-vocabulary GGUF layout."
        )
    # HEADER-ONLY: refuse any GGML type other than Q8_0/F32/F16/BF16, with
    # Q8_0 popped out of the refusal set first -- see the docstring above.
    unsupported = gguf_container.unsupported_tensor_types(header)
    unsupported.pop("Q8_0", None)
    if unsupported:
        raise gguf_container.GGUFUnsupportedTensorTypeError(
            f"the MiniMax Music 3 pruned text encoder GGUF checkpoint ({gguf_text_encoder_path}) "
            f"declares tensor type(s) neither this reader's dense path (F32/F16/BF16) nor its "
            f"Q8_0 packed path (design doc phase 12) materializes: "
            f"{ {k: len(v) for k, v in unsupported.items()} }. Header-only refusal -- no tensor "
            f"byte of this {header.file_size}-byte file was read."
        )

    lm_config_path = os.path.join(official, "language_model", "config.json")
    if not os.path.isfile(lm_config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing language_model/config.json."
        )
    with open(lm_config_path, encoding="utf-8") as fh:
        lm_config_dict = json.load(fh)
    rope_parameters = lm_config_dict.get("rope_parameters")
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model config.rope_parameters['rope_theta'] is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}. Checked from "
            f"official/language_model/config.json BEFORE reading the pruned GGUF text "
            f"encoder's tensor data."
        )
    depth_config = _read_component_config(official, "rvq_depth_decoder", "MiniMaxMusic3RVQDepthDecoder")

    state = gguf_container.GGUFStateDict(
        header, arch="MiniMax Music 3", label="pruned text encoder (Q8_0 packed)",
    )
    try:
        hidden_size = int(lm_config_dict["hidden_size"])
        prefill_rows = int(state["model.embed_tokens_prefill.weight"].shape[0])
        if prefill_rows != AUDIO_CODE_OFFSET:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF (Q8_0) text encoder: model.embed_tokens_prefill has "
                f"{prefill_rows} rows, expected AUDIO_CODE_OFFSET ({AUDIO_CODE_OFFSET})."
            )
        audio_rows = int(state["model.embed_tokens_audio.weight"].shape[0])
        if audio_rows != SEMANTIC_VOCAB_SIZE:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF (Q8_0) text encoder: model.embed_tokens_audio has "
                f"{audio_rows} rows, expected SEMANTIC_VOCAB_SIZE ({SEMANTIC_VOCAB_SIZE})."
            )
        # `model.lm_head_pruned.weight` is Q8_0-typed on the real checkpoint (unlike the two
        # vocab-table reads above, which are BF16), so it cannot go through `state[...]` --
        # that refuses Q8_0 by design (see gguf_container's module docstring). Its row count
        # is already in the header's own tensor descriptor (`torch_shape`), which needs no
        # tensor byte read at all -- cheaper than the other two checks, not just Q8_0-safe.
        header_tensors_by_name = {t.name: t for t in header.tensors}
        lm_head_pruned_info = header_tensors_by_name.get("model.lm_head_pruned.weight")
        if lm_head_pruned_info is None:
            # `is_pruned_flat_text_encoder` above only requires ANY ONE of
            # its three tells to be present, so a file could pass that gate
            # via `embed_tokens_prefill`/`embed_tokens_audio` alone and still
            # be missing this specific tensor -- name that explicitly rather
            # than a bare `KeyError`, matching every neighbouring gate here.
            raise ValueError(
                f"{gguf_text_encoder_path!r} is a pruned-vocabulary GGUF text encoder (matched a "
                f"pruned-vocabulary tell) but declares no 'model.lm_head_pruned.weight' tensor -- "
                f"this builder cannot construct the patched Qwen3ForCausalLM without it."
            )
        head_rows = lm_head_pruned_info.torch_shape[0]
        if head_rows != AUDIO_HEAD_VOCAB_SIZE:
            raise ValueError(
                f"MiniMax Music 3 pruned GGUF (Q8_0) text encoder: model.lm_head_pruned has "
                f"{head_rows} rows, expected SEMANTIC_VOCAB_SIZE + 1 ({AUDIO_HEAD_VOCAB_SIZE})."
            )

        remapped = apply_pruned_text_encoder_state_dict_packed(state, lm_config_dict)
    finally:
        state.close()

    lm_hf_config = AutoConfig.from_pretrained(os.path.join(official, "language_model"))
    lm_hf_config.vocab_size = prefill_rows

    with init_empty_weights():
        language_model = AutoModelForCausalLM.from_config(lm_hf_config)
        del language_model.lm_head
        language_model.lm_head_pruned = torch.nn.Linear(hidden_size, AUDIO_HEAD_VOCAB_SIZE, bias=False)
        language_model.model.embed_tokens_audio = torch.nn.Embedding(SEMANTIC_VOCAB_SIZE, hidden_size)

    lm_remapped = remapped["language_model"]
    lm_packed = {k: v for k, v in lm_remapped.items() if isinstance(v, PackedQ8_0Weight)}
    lm_dense = {k: v for k, v in lm_remapped.items() if not isinstance(v, PackedQ8_0Weight)}
    assert_state_dict_matches_module_keys(
        set(lm_packed.keys()) | set(lm_dense.keys()), expected_module_state_dict_keys(language_model),
        component="language_model (pruned GGUF, Q8_0 packed)",
    )
    lm_dense_cast = {
        k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v) for k, v in lm_dense.items()
    }
    language_model.load_state_dict(lm_dense_cast, strict=False, assign=True)
    installed = install_packed_q8_0_linears(
        language_model, {k: (v.codes, v.scale) for k, v in lm_packed.items()}, torch_dtype,
    )
    if installed != len(lm_packed):
        raise RuntimeError(
            f"MiniMax Music 3 pruned GGUF (Q8_0) text encoder: installed {installed} packed "
            f"Linear(s), expected {len(lm_packed)} -- a destination key produced no swap, which "
            f"is a bug in install_packed_q8_0_linears or in the remap plan."
        )
    stranded = _stranded_meta_tensors(language_model)
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's language_model (pruned GGUF Q8_0 text encoder source) still "
            f"holds {len(stranded)} meta tensor(s) after loading and packed-swap (first 5: "
            f"{stranded[:5]}); it would fail at the first forward."
        )
    language_model.eval()
    language_model.requires_grad_(False)
    _assert_language_model_rope_theta(language_model)

    with init_empty_weights():
        rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_config(depth_config)

    depth_remapped = remapped["rvq_depth_decoder"]
    depth_packed = {k: v for k, v in depth_remapped.items() if isinstance(v, PackedQ8_0Weight)}
    depth_dense = {k: v for k, v in depth_remapped.items() if not isinstance(v, PackedQ8_0Weight)}
    assert_state_dict_matches_module_keys(
        set(depth_packed.keys()) | set(depth_dense.keys()),
        expected_module_state_dict_keys(rvq_depth_decoder),
        component="rvq_depth_decoder (pruned GGUF, Q8_0 packed)",
    )
    depth_dense_cast = {
        k: (v.to(dtype=torch_dtype) if v.is_floating_point() else v) for k, v in depth_dense.items()
    }
    rvq_depth_decoder.load_state_dict(depth_dense_cast, strict=False, assign=True)
    depth_installed = install_packed_q8_0_linears(
        rvq_depth_decoder, {k: (v.codes, v.scale) for k, v in depth_packed.items()}, torch_dtype,
    )
    if depth_installed != len(depth_packed):
        raise RuntimeError(
            f"MiniMax Music 3 pruned GGUF (Q8_0) text encoder: installed {depth_installed} packed "
            f"Linear(s) in rvq_depth_decoder, expected {len(depth_packed)}."
        )
    stranded_depth = _stranded_meta_tensors(rvq_depth_decoder)
    if stranded_depth:
        raise RuntimeError(
            f"MiniMax Music 3's rvq_depth_decoder (pruned GGUF Q8_0 text encoder source) still "
            f"holds {len(stranded_depth)} meta tensor(s) after loading and packed-swap (first 5: "
            f"{stranded_depth[:5]}); it would fail at the first forward."
        )
    rvq_depth_decoder.eval()
    rvq_depth_decoder.requires_grad_(False)

    return language_model, rvq_depth_decoder, depth_config


def _assert_language_model_rope_theta(language_model) -> None:
    """Load-time gate: ``config.rope_parameters["rope_theta"] == 1e6``.

    NOT ``config.rope_theta``, which is ``None`` on the installed
    transformers 5.1 for a config written by transformers 5.13's
    ``rope_parameters`` form -- see the design doc's "Dependency gate". A
    gate written against the old field would misfire (silently pass with
    ``None`` compared against nothing) on a HEALTHY load; this one reads the
    field the config actually populates.
    """
    rope_parameters = getattr(language_model.config, "rope_parameters", None)
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model config.rope_parameters['rope_theta'] is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}. This is the load-time "
            f"gate the design doc requires (see 'Dependency gate'): a silently wrong rope base "
            f"degrades every AR-stage output without erroring anywhere else."
        )


def _build_language_model(official: str, torch_dtype: torch.dtype):
    """The 8B ``Qwen3ForCausalLM`` from ``official/language_model/``.

    Uses transformers' own ``from_pretrained`` directly rather than the
    meta+``load_state_dict`` pattern the vendored diffusers-class components
    above use: ``Qwen3ForCausalLM`` is a REGISTERED transformers class, so its
    native loader already assembles the sharded state dict this loader would
    otherwise have to reimplement. (``low_cpu_mem_usage`` is NOT passed: the
    installed transformers 5.1.0 pops and discards that kwarg, so passing it
    would credit this call with a memory behavior it does not have.)
    """
    from transformers import AutoModelForCausalLM

    lm_dir = os.path.join(official, "language_model")
    config_path = os.path.join(lm_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing "
            f"language_model/config.json."
        )
    with open(config_path, encoding="utf-8") as fh:
        lm_config = json.load(fh)
    # Cheap gate BEFORE the ~17 GiB load, matching the design doc's "read
    # rope_parameters, not rope_theta" requirement and the general ordering
    # rule (fail on a JSON read, not after a multi-GB load).
    rope_parameters = lm_config.get("rope_parameters")
    theta = rope_parameters.get("rope_theta") if isinstance(rope_parameters, dict) else None
    if theta is None or abs(float(theta) - EXPECTED_LANGUAGE_MODEL_ROPE_THETA) > _ROPE_THETA_TOLERANCE:
        raise ValueError(
            f"MiniMax Music 3's language_model/config.json rope_parameters.rope_theta is "
            f"{theta!r}, expected {EXPECTED_LANGUAGE_MODEL_ROPE_THETA}."
        )
    language_model = AutoModelForCausalLM.from_pretrained(lm_dir, torch_dtype=torch_dtype)
    # Re-assert against the LOADED model's own config object -- defense in
    # depth against transformers doing anything different with the field
    # during from_pretrained.
    _assert_language_model_rope_theta(language_model)
    language_model.eval()
    language_model.requires_grad_(False)
    return language_model


def _load_tokenizer(official: str):
    from transformers import AutoTokenizer

    tok_dir = os.path.join(official, "tokenizer")
    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing tokenizer/."
        )
    return AutoTokenizer.from_pretrained(tok_dir)


def _load_scheduler(official: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched_dir = os.path.join(official, "scheduler")
    config_path = os.path.join(sched_dir, "scheduler_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"MiniMax Music 3's config tree at {official!r} is missing "
            f"scheduler/scheduler_config.json."
        )
    return FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)


def load_minimax_music3_from_path(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *,
    load_language_model: bool = True,
) -> dict:
    """Load MiniMax Music 3, reading ``official/`` and (if pointed at a flat DiT
    file) the transformer + condition encoder from it instead -- see the
    module docstring. ``load_language_model=False`` skips the ~17 GiB
    Qwen3-8B load -- for a probe or a test that only needs the flow-matching
    side's geometry; it is NOT a memory optimization for generation (the AR
    stage needs the language model resident).

    Returns the component dict ``PipelineManager.load_model()`` consumes,
    with ``type == "minimax_music3"``. Every component stays CPU-resident;
    GPU staging is a later commit's (pipeline-backend) concern.
    """
    layout = detect_minimax_music3_layout(model_path)
    if layout is None:
        raise ValueError(
            f"MiniMax Music 3 model layout not found at {model_path!r}. Expected the model's "
            f"root directory (holding official/ and optionally diffusion_models/ + vae/ + "
            f"text_encoders/), a DiT .safetensors inside such a diffusion_models/, or "
            f"MiniMax's own official/ directory (modular_model_index.json declaring "
            f"{MINIMAX_MUSIC3_PIPELINE_CLASS})."
        )

    if layout.get("flat_dit") is not None and layout.get("official") is None:
        # A lone flat DiT file with no official/ tree reachable beside it:
        # `core.models.minimax_music3.flat_remap` CAN remap its weights, but
        # this loader still reads every component's CONFIG from official/
        # (the flat file carries no config.json of its own), so there is
        # nothing to build against without it.
        raise NotImplementedError(
            f"MiniMax Music 3's flat repacked DiT ({layout['flat_dit']}) has no reachable "
            f"official/ config-and-weight tree beside it. Its weights ARE remappable "
            f"(core.models.minimax_music3.flat_remap), but this loader still reads every "
            f"component's config -- including the transformer's and condition encoder's -- "
            f"from official/, which the flat file does not carry. Point the model path at "
            f"the model's root directory, which must contain an official/ tree."
        )
    # official/ IS reachable and a flat DiT file was named explicitly: read the
    # transformer + condition encoder from THAT file (design doc phase 9),
    # via `core.models.minimax_music3.flat_remap`. `int8_convrot` and any
    # other quantized flat DiT is still refused, inside the builder itself
    # (`refuse_quantized_state_dict`, same guard every official/ component
    # load runs) -- design doc phase 13. Configs still come from official/;
    # only the transformer's WEIGHTS are sourced from the flat file.
    use_flat_dit = layout.get("flat_dit") is not None
    # Which builder: the file's own suffix decides (design doc phase 11) --
    # `detect_minimax_music3_layout` already proved it is a MiniMax Music 3
    # DiT of ONE of these two formats before `flat_dit` was ever populated.
    use_gguf_dit = use_flat_dit and str(layout["flat_dit"]).lower().endswith(".gguf")

    official = layout["official"]
    if official is None:
        raise FileNotFoundError(
            f"MiniMax Music 3 at {layout.get('root')!r} has no official/ config-and-weight "
            f"tree (modular_model_index.json declaring {MINIMAX_MUSIC3_PIPELINE_CLASS}). Every "
            f"component's CONFIG comes from that tree regardless of which file the WEIGHTS come "
            f"from (see the module docstring); without it, nothing can be built."
        )

    # List ALL missing weight/config slots at once, before anything is built --
    # a caller should learn every gap in one message, not one per attempt.
    missing: List[str] = []
    for subdir, _expected_class in _DIFFUSERS_COMPONENTS:
        if not os.path.isfile(os.path.join(official, subdir, "config.json")):
            missing.append(f"{subdir}/config.json")
        if subdir in ("transformer", "condition_encoder") and use_flat_dit:
            # Both components' WEIGHTS come from the flat file instead (see
            # `build_transformer_and_condition_encoder_from_flat_dit`); only
            # the CONFIG (checked above) is still read from official/<subdir>/.
            continue
        if not _component_weight_present(official, subdir):
            missing.append(f"{subdir}/{_WEIGHT_BASENAME}.safetensors")
    if load_language_model:
        if not os.path.isfile(os.path.join(official, "language_model", "config.json")):
            missing.append("language_model/config.json")
        lm_dir = os.path.join(official, "language_model")
        if not (
            os.path.isfile(os.path.join(lm_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(lm_dir, "model.safetensors.index.json"))
        ):
            missing.append("language_model weights")
    if not os.path.isdir(os.path.join(official, "tokenizer")):
        missing.append("tokenizer/")
    if not os.path.isfile(os.path.join(official, "scheduler", "scheduler_config.json")):
        missing.append("scheduler/scheduler_config.json")
    if missing:
        raise FileNotFoundError(
            f"MiniMax Music 3's config-and-weight tree at {official!r} is missing the "
            f"following: {', '.join(missing)}."
        )

    # Validate the config tree UP FRONT (class-name census), before any
    # multi-GB weight is mapped -- a class mismatch is knowable from four
    # small JSON reads and answering it after gigabytes of weight have been
    # loaded is work spent to reach a message that was available immediately.
    for subdir, expected_class in _DIFFUSERS_COMPONENTS:
        _read_component_config(official, subdir, expected_class)

    print(f"[MiniMaxMusic3Loader] root:            {layout.get('root')}")
    print(f"[MiniMaxMusic3Loader] official tree:    {official}")
    for subdir, _ in _DIFFUSERS_COMPONENTS:
        if subdir in ("transformer", "condition_encoder") and use_flat_dit:
            source_label = "GGUF, remapped" if use_gguf_dit else "flat, remapped"
            print(f"[MiniMaxMusic3Loader] {subdir}:{' ' * max(1, 17 - len(subdir))}{layout['flat_dit']} ({source_label})")
            continue
        print(f"[MiniMaxMusic3Loader] {subdir}:{' ' * max(1, 17 - len(subdir))}{os.path.join(official, subdir)}")
    if use_gguf_dit and torch_dtype == torch.bfloat16:
        # Design doc "GGUF weights": at this dtype, the GGUF DiT's own
        # GGML-F16 tensors (~40% of the file) are `official.half()` cast
        # AGAIN to bf16 -- a double rounding, up to 2**-8 max abs diff from
        # `official.bfloat16()` -- while the flat fp16 safetensors DiT's
        # equivalent residual is ~2.98e-08 (four orders of magnitude
        # closer). Stated here, at load time, not only in the design doc.
        print(f"[MiniMaxMusic3Loader] note: this GGUF DiT's GGML-F16 tensors are "
              f"official.half() cast again to bf16 here -- up to 2**-8 EXTRA rounding "
              f"vs. the flat fp16/fp32 safetensors DiT on those tensors, not a wash.")
    if load_language_model:
        print(f"[MiniMaxMusic3Loader] language_model:  {os.path.join(official, 'language_model')}")
    print(f"[MiniMaxMusic3Loader] tokenizer:       {os.path.join(official, 'tokenizer')}")
    print(f"[MiniMaxMusic3Loader] scheduler:       {os.path.join(official, 'scheduler')}")

    from core.models.minimax_music3.vendor import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    # Build the LARGEST component FIRST (language model, ~17 GiB bf16), then
    # the fp32-source transformer (~9.7 GiB before the cast to torch_dtype).
    # On Windows, doing the largest last can access-violate inside
    # safetensors/torch storage -- same ordering rule
    # `minimax_h3.loader.load_minimax_h3_from_path` documents for its own
    # (much larger) text encoder.
    language_model = _build_language_model(official, torch_dtype) if load_language_model else None

    if use_gguf_dit:
        transformer, transformer_config, condition_encoder, condition_encoder_config = (
            build_transformer_and_condition_encoder_from_gguf_dit(
                layout["flat_dit"], official, torch_dtype,
            )
        )
    elif use_flat_dit:
        transformer, transformer_config, condition_encoder, condition_encoder_config = (
            build_transformer_and_condition_encoder_from_flat_dit(
                layout["flat_dit"], official, torch_dtype,
            )
        )
    else:
        transformer, transformer_config = _build_diffusers_component(
            official, "transformer", MiniMaxMusic3Transformer1DModel,
            "MiniMaxMusic3Transformer1DModel", torch_dtype,
        )
        # condition_encoder and vocoder are tiny (4 and 121 tensors respectively)
        # and both sit directly on the conditioning/waveform precision path, so
        # they are kept at float32 regardless of `torch_dtype` -- mirroring
        # `minimax_h3.loader`'s judgment for its own small audio VAE ("0.6 GB,
        # decoded once per generation, nothing to buy" by quantizing it).
        condition_encoder, condition_encoder_config = _build_diffusers_component(
            official, "condition_encoder", MiniMaxMusic3ConditionEncoder,
            "MiniMaxMusic3ConditionEncoder", torch.float32,
        )
    rvq_depth_decoder, rvq_depth_decoder_config = _build_diffusers_component(
        official, "rvq_depth_decoder", MiniMaxMusic3RVQDepthDecoder,
        "MiniMaxMusic3RVQDepthDecoder", torch_dtype,
    )
    vocoder, vocoder_config = _build_diffusers_component(
        official, "vocoder", MiniMaxMusic3Vocoder,
        "MiniMaxMusic3Vocoder", torch.float32,
    )

    tokenizer = _load_tokenizer(official)
    scheduler = _load_scheduler(official)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[MiniMaxMusic3Loader] Loaded MiniMax Music 3 components (CPU-resident; no "
          "pipeline-backend wiring yet -- design doc phase plan item 3).")

    return {
        "type": "minimax_music3",
        "is_audio": True,
        "transformer": transformer,
        "transformer_config": transformer_config,
        "condition_encoder": condition_encoder,
        "condition_encoder_config": condition_encoder_config,
        "rvq_depth_decoder": rvq_depth_decoder,
        "rvq_depth_decoder_config": rvq_depth_decoder_config,
        "language_model": language_model,
        "vocoder": vocoder,
        "vocoder_config": vocoder_config,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        # Geometry, so nothing downstream re-derives it. `sample_rate` is read
        # from the vocoder's own config, not the model card's 32 kHz (SGLang
        # serving path only). `frame_rate` is likewise DERIVED, from the
        # condition encoder's own config (input_sampling_rate /
        # input_hop_length = 24000 / 960 = 25.0) -- the FALLBACK_FRAME_RATE
        # constant is the fallback for a missing config, not the primary value.
        "sample_rate": int(vocoder_config.get("sampling_rate", FALLBACK_SAMPLING_RATE)),
        "frame_rate": (
            condition_encoder_config["input_sampling_rate"] / condition_encoder_config["input_hop_length"]
            if condition_encoder_config.get("input_sampling_rate") and condition_encoder_config.get("input_hop_length")
            else FALLBACK_FRAME_RATE
        ),
        "latent_channels": int(transformer_config.get("in_channels", FALLBACK_NUM_CHANNELS_LATENTS)),
        # Identity, for the component catalog and for reloads. `dit_path` is
        # what `component_catalog._configured_path` looks for on "backbone".
        # `text_encoder_path`/`vae_path` alias language_model/vocoder onto the
        # catalog's generic slots -- see `_component_object`'s arch alias for
        # why those two slot NAMES, not the components' own dict keys, are
        # what the catalog reads by default.
        "dit_path": layout["flat_dit"] if use_flat_dit else os.path.join(official, "transformer"),
        "text_encoder_path": os.path.join(official, "language_model"),
        "vae_path": os.path.join(official, "vocoder"),
        "official_dir": official,
        "root": layout.get("root"),
    }
