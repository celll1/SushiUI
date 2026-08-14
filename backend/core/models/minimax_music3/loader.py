"""MiniMax Music 3 loader: detection + component build (design doc phase 2).

Supported load path -- decided, not deferred by omission
==========================================================

The released snapshot ships the SAME model twice, in two different shapes:

* MiniMax's own config-and-weight tree, ``official/`` (``transformer/``,
  ``condition_encoder/``, ``rvq_depth_decoder/``, ``vocoder/``,
  ``language_model/``, ``tokenizer/``, ``scheduler/``), where every
  component's ``config.json`` names the exact vendored class
  (``core.models.minimax_music3.vendor``) and every weight key matches that
  class's ``state_dict`` VERBATIM -- verified against the real files (see
  ``docs/guides/MINIMAX_MUSIC3_DESIGN.md`` for the audit this loader was
  written against). No remap of any kind is needed;
* the flat, ComfyUI-style repack under ``diffusion_models/`` +
  ``text_encoders/`` + ``vae/``, which is NOT a re-export of the same
  tensors under different names: the flat DiT folds the condition encoder
  in (``latent_conditioners.{0,1}``, ``cond_layer_logits``,
  ``cond_layer_scale``), fuses QKV into one ``to_qkv`` per layer, and uses
  ``.gamma``/``.beta`` norm names instead of ``nn.LayerNorm``'s
  ``.weight``/``.bias``; the flat text encoder merges the language model AND
  the RVQ depth decoder into one file with a pruned-vocabulary variant on
  top. Reading either correctly means writing and proving a key-remap, which
  this commit does not do.

THIS LOADER SUPPORTS ONLY THE FIRST SHAPE. Detection (below) still
recognizes all three spellings the design doc's phase-plan item 2 asks
for -- the flat root, a lone DiT ``.safetensors`` inside
``diffusion_models/``, and the ``official/`` directory itself -- because
telling MiniMax Music 3 apart from every other architecture must not
depend on which of the two shapes a user happened to point at. But
``load_minimax_music3_from_path`` refuses the flat shape outright, with a
message naming exactly what would have to be built, rather than attempting
a partial remap that could silently produce a wrong model. A future commit
that proves the remap against the real flat files may add it; this one
does not half-wire it.

The flat tree's quantized (``int8_convrot``) files are consequently never
read by this loader at all -- they live under the shape this loader
refuses -- but ``refuse_quantized_state_dict`` is still called on every
component state dict this loader DOES read, immediately after it is read
and before that component is built, as a defensive load-time gate matching
the design doc's "Phase 1 is BF16/FP16 only" rule (see
"Dependency gate" / "Quantization" in the design doc) in case a future
snapshot ever ships a quantized variant under ``official/``.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

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
# `latent_conditioners.{0,1}`, `cond_layer_logits`, `cond_layer_scale` (374
# total) -- the condition encoder folded into the same file. Verified against
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

    Accepts three spellings of the same model, matching the design doc's
    phase-plan item 2 ("directory detection... flat-tree completion by
    sibling-probe into official/"):

    * the flat root (``<root>/diffusion_models/`` + ``vae/`` +
      ``text_encoders/``, with ``official/`` beside it);
    * a DiT ``.safetensors`` inside such a ``diffusion_models/`` (walks up to
      find the root, then the sibling ``official/``);
    * MiniMax's config-and-weight ``official/`` directory itself, i.e. one
      whose ``modular_model_index.json`` declares
      ``MiniMaxMusic3ModularPipeline``.

    ``official`` is the ONLY component tree this module's loader reads (see
    the module docstring). ``flat_dit`` is populated -- to a non-``None``
    path -- exactly when the caller pointed at a lone DiT file rather than at
    the root or at ``official/`` directly: `load_minimax_music3_from_path`
    uses its presence to refuse with a message that says what was pointed at,
    distinct from "the official/ tree is missing" when there is no
    ``official/`` at all.
    """
    if not path:
        return None
    p = Path(path)

    if p.is_file() and p.suffix == ".safetensors":
        if not is_minimax_music3_safetensors(str(p)):
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

    stranded = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if getattr(t, "is_meta", False)
    ]
    if stranded:
        raise RuntimeError(
            f"MiniMax Music 3's {subdir} from {comp_dir} still holds {len(stranded)} meta "
            f"tensor(s) after loading (first 5: {stranded[:5]}); it would fail at the first "
            f"forward."
        )
    model.eval()
    model.requires_grad_(False)
    return model, config


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
    """Load MiniMax Music 3 from its ``official/`` config-and-weight tree.

    See the module docstring for why the flat (ComfyUI-repack) tree is
    refused rather than half-remapped. ``load_language_model=False`` skips
    the ~17 GiB Qwen3-8B load -- for a probe or a test that only needs the
    flow-matching side's geometry; it is NOT a memory optimization for
    generation (the AR stage needs the language model resident).

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
        # A lone flat DiT file with no official/ tree reachable beside it: there
        # is nothing this loader can build from (see module docstring), and
        # unlike the "official present but flat file named explicitly" case
        # below, there is no config tree to point the user at either.
        raise NotImplementedError(
            f"MiniMax Music 3's flat repacked DiT ({layout['flat_dit']}) has no reachable "
            f"official/ config-and-weight tree beside it, and this loader does not read the "
            f"flat tree directly (see docs/guides/MINIMAX_MUSIC3_DESIGN.md, 'Quantization': "
            f"the flat DiT folds the condition encoder in, fuses QKV, and uses .gamma/.beta "
            f"norm names, none of which is remapped in this commit). Point the model path at "
            f"the model's root directory, which must contain an official/ tree."
        )
    if layout.get("flat_dit") is not None:
        # official/ IS reachable -- the user pointed at the flat file specifically
        # (e.g. to select a precision/quantization variant this loader does not
        # support choosing). Refuse rather than silently substituting official/'s
        # transformer for the file the caller named.
        raise NotImplementedError(
            f"MiniMax Music 3's flat repacked DiT ({layout['flat_dit']}) is not loadable by "
            f"this loader (see docs/guides/MINIMAX_MUSIC3_DESIGN.md, 'Quantization': fused "
            f"QKV, folded-in condition encoder, .gamma/.beta norm names -- none remapped in "
            f"this commit). Point the model path at the root directory or at "
            f"{layout['official']!r} directly; both load the full-precision official/ "
            f"transformer instead."
        )

    official = layout["official"]
    if official is None:
        raise FileNotFoundError(
            f"MiniMax Music 3 at {layout.get('root')!r} has no official/ config-and-weight "
            f"tree (modular_model_index.json declaring {MINIMAX_MUSIC3_PIPELINE_CLASS}). This "
            f"loader reads every component from that tree; see the module docstring for why "
            f"the flat repacked tree is not an alternative in this commit."
        )

    # List ALL missing weight/config slots at once, before anything is built --
    # a caller should learn every gap in one message, not one per attempt.
    missing: List[str] = []
    for subdir, _expected_class in _DIFFUSERS_COMPONENTS:
        if not os.path.isfile(os.path.join(official, subdir, "config.json")):
            missing.append(f"{subdir}/config.json")
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
        print(f"[MiniMaxMusic3Loader] {subdir}:{' ' * max(1, 17 - len(subdir))}{os.path.join(official, subdir)}")
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
        "dit_path": os.path.join(official, "transformer"),
        "text_encoder_path": os.path.join(official, "language_model"),
        "vae_path": os.path.join(official, "vocoder"),
        "official_dir": official,
        "root": layout.get("root"),
    }
