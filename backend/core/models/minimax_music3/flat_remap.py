"""Flat (ComfyUI-repack) key remap for MiniMax Music 3 -- design doc phase 9.

Serves the flat safetensors (this commit's caller) AND design doc phase 11's
future GGUF reader with the SAME remap: the two distributions' tensor names
are identical, so the remap logic here never assumes a safetensors origin.
Scope and the investigation behind these mappings (census counts, the real
files each shape was verified against) are in
``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Which tree the loader reads" /
"GGUF weights" -- this docstring covers only what a maintainer needs at the
call site.

Handled here: the flat DiT (``diffusion_transformer.*`` ->
``MiniMaxMusic3Transformer1DModel``, plus ``latent_conditioners.0`` /
``cond_layer_logits`` / ``cond_layer_scale`` -> ``MiniMaxMusic3ConditionEncoder``)
and the flat NON-PRUNED text encoder (``model.layers.*`` /
``model.embed_tokens`` / ``model.norm`` / ``model.lm_head`` ->
``Qwen3ForCausalLM``, plus ``model.audio_decoder.*`` /
``model.audio_extra_embedding`` -> ``MiniMaxMusic3RVQDepthDecoder``).

Refused, not half-remapped, BY THIS MODULE: the pruned text encoder's
vocabulary split (design doc phase 10 -- an AR-loop offset/mask change, not a
rename; handled by the dedicated ``pruned_text_encoder_remap`` module
instead) and the GGUF container format itself (phase 11; handled by
``core.models.common.gguf_container``). ``int8_convrot`` on either file is
now READABLE (design doc phase 13), but not by this module directly: this
module's ``plan_*``/``apply_*`` functions still operate on DENSE keys only
(a ``.weight_scale``/``.comfy_quant`` sidecar handed to them lands in
``unrecognized`` and refuses the remap) -- ``core.models.minimax_music3.
convrot_remap`` calls them UNCHANGED for every dense tensor (including a
quantized layer's own int8 ``.weight``, whose row-wise split this module
already performs is exact for ConvRot codes too) and places the ConvRot
sidecars at the destinations this module's plan already computed. Guarded at
the loader call site by ``core.models.common.quantized_checkpoint_guard``,
not duplicated here.

Three findings that are not obvious from either file alone:

* ``diffusion_transformer.transformer.layers.N.self_attn.to_qkv.weight`` is
  ``[q | k | v]`` ROW-CONTIGUOUS, not per-head interleaved -- there is no
  per-head reshape between this Linear and the attention processor's
  ``.view(batch, seq, heads, head_dim)``.
* ``...rotary_pos_emb.inv_freq`` has NO destination:
  ``MiniMaxMusic3RotaryEmbedding`` computes it inside ``forward()`` via
  ``lru_cache_unless_export`` and never assigns it to ``self``, so the
  vendored module's ``state_dict()`` has no key for it at all.
* ``model.audio_extra_embedding.weight`` sits at the LANGUAGE MODEL's top
  level in the flat file but IS the RVQ depth decoder's ``audio_embeddings``
  table (same shape, ``audio_vocab_size * (num_codebooks - 1)`` rows) -- a
  CROSS-COMPONENT rename, not a drop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

__all__ = [
    "FlatRemapPlan",
    "PrunedTextEncoderNotSupported",
    "plan_flat_dit_keys",
    "apply_flat_dit_state_dict",
    "plan_flat_text_encoder_keys",
    "apply_flat_text_encoder_state_dict",
    "is_pruned_flat_text_encoder",
    "raise_if_pruned_flat_text_encoder",
    "assert_state_dict_matches_module_keys",
    "expected_module_state_dict_keys",
]


class PrunedTextEncoderNotSupported(NotImplementedError):
    """Raised by THIS module's remap functions when handed the pruned-vocabulary layout --
    that layout is supported, but by the dedicated ``pruned_text_encoder_remap`` module
    instead (see ``raise_if_pruned_flat_text_encoder``'s message)."""


@dataclass
class FlatRemapPlan:
    """The result of matching a flat key set against this module's rename rules.

    ``renames`` and ``splits`` are keyed by DESTINATION COMPONENT (e.g.
    ``"transformer"``, ``"condition_encoder"``, ``"language_model"``,
    ``"rvq_depth_decoder"``) so ``apply_*`` can build one output state dict
    per component directly from this plan without re-deriving component
    membership.

    * ``renames[component]``: ``{flat_key: dest_key}``, one tensor in, one out,
      unchanged.
    * ``splits[component]``: ``{flat_key: (dest_key, ...)}``, one tensor in,
      split along dim 0 into ``len(dest_keys)`` equal pieces, in order.
    * ``dropped``: ``{flat_key: reason}`` -- keys with no destination at all.
    * ``unrecognized``: flat keys this plan's rules did not match. A
      non-empty list here means the remap is INCOMPLETE for this input and
      ``apply_*`` refuses rather than silently dropping them.
    """

    renames: Dict[str, Dict[str, str]] = field(default_factory=dict)
    splits: Dict[str, Dict[str, Tuple[str, ...]]] = field(default_factory=dict)
    dropped: Dict[str, str] = field(default_factory=dict)
    unrecognized: List[str] = field(default_factory=list)

    def produced_keys(self, component: str) -> set:
        keys = set(self.renames.get(component, {}).values())
        for dest_keys in self.splits.get(component, {}).values():
            keys.update(dest_keys)
        return keys


# ---------------------------------------------------------------------------
# DiT: diffusion_transformer.* (+ latent_conditioners.0 / cond_layer_*)
#   -> MiniMaxMusic3Transformer1DModel + MiniMaxMusic3ConditionEncoder
# ---------------------------------------------------------------------------

_DIT_LAYER_RE = re.compile(r"^diffusion_transformer\.transformer\.layers\.(\d+)\.(.+)$")

# Per-layer suffix, flat -> vendored `MiniMaxMusic3TransformerBlock`. Verified
# against BOTH sides' shapes (fp16 flat file / official fp32 file): every
# entry here is a pure rename, same shape on both sides.
_DIT_LAYER_SUFFIX_RENAME: Dict[str, str] = {
    "pre_norm.gamma": "norm1.weight",
    "pre_norm.beta": "norm1.bias",
    "ff_norm.gamma": "norm2.weight",
    "ff_norm.beta": "norm2.bias",
    # The GEGLU naming ("ff.ff.0.proj" / "ff.ff.2") is the flat repack's; the
    # vendored block spells the same two Linears "ff_in" / "ff_out" directly
    # (see MiniMaxMusic3TransformerBlock in vendor/transformer_minimax_music3.py).
    "ff.ff.0.proj.weight": "ff_in.weight",
    "ff.ff.0.proj.bias": "ff_in.bias",
    "ff.ff.2.weight": "ff_out.weight",
    "ff.ff.2.bias": "ff_out.bias",
    # attn.to_out is `nn.ModuleList([Linear, Dropout])` in the vendored module
    # (`MiniMaxMusic3Attention.to_out`); the flat file has no dropout entry to
    # begin with, so only index 0 (the Linear) needs a destination.
    "self_attn.to_out.weight": "attn.to_out.0.weight",
}

# The one per-layer SPLIT: fused QKV [3 * inner_dim, dim] -> three
# [inner_dim, dim] Linears, [q | k | v] ROW-CONTIGUOUS (not per-head
# interleaved -- see the module docstring's first finding; confirmed
# numerically against the real checkpoint, design doc "GGUF weights").
_DIT_LAYER_QKV_SUFFIX = "self_attn.to_qkv.weight"
_DIT_QKV_SPLIT_SUFFIXES = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight")

# Non-layer DiT keys. Every destination here is a direct attribute of
# `MiniMaxMusic3Transformer1DModel` (see its `__init__`).
_DIT_NON_LAYER_RENAME: Dict[str, str] = {
    "diffusion_transformer.preprocess_conv.weight": "preprocess_conv.weight",
    "diffusion_transformer.postprocess_conv.weight": "postprocess_conv.weight",
    # `MiniMaxMusic3FourierEmbedding.weight` is `time_proj.weight` on the
    # vendored side; the flat repack calls the same tensor "timestep_features".
    "diffusion_transformer.timestep_features.weight": "time_proj.weight",
    # `TimestepEmbedding` (diffusers) names its two Linears `linear_1` /
    # `linear_2`; the flat repack stores them as a `nn.Sequential`-style
    # `to_timestep_embed.0` / `.2` (index 1 is the activation, weightless).
    "diffusion_transformer.to_timestep_embed.0.weight": "time_embed.linear_1.weight",
    "diffusion_transformer.to_timestep_embed.0.bias": "time_embed.linear_1.bias",
    "diffusion_transformer.to_timestep_embed.2.weight": "time_embed.linear_2.weight",
    "diffusion_transformer.to_timestep_embed.2.bias": "time_embed.linear_2.bias",
    "diffusion_transformer.transformer.project_in.weight": "proj_in.weight",
    "diffusion_transformer.transformer.project_out.weight": "proj_out.weight",
}

# See the module docstring's first "finding": no destination exists for this
# key in the vendored module's state_dict() at all.
_DIT_NON_LAYER_DROP: Dict[str, str] = {
    "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": (
        "MiniMaxMusic3RotaryEmbedding (vendor/transformer_minimax_music3.py) computes "
        "inv_freq inside forward() via lru_cache_unless_export and never assigns it to "
        "self; the vendored module registers no parameter or buffer for it, so there is "
        "no destination key. The official/ tree's own transformer weights (441 tensors, "
        "loaded key-for-key with no remap) confirm this: they carry no rotary_emb key "
        "either."
    ),
}

# `MiniMaxMusic3ConditionEncoder` (4 tensors: layer_weight_logits, layer_scale,
# proj.weight, proj.bias). The flat file folds it into the DiT file under
# these top-level (unprefixed) keys.
_CONDITION_ENCODER_RENAME: Dict[str, str] = {
    "cond_layer_logits": "layer_weight_logits",
    "cond_layer_scale": "layer_scale",
    "latent_conditioners.0.weight": "proj.weight",
    "latent_conditioners.0.bias": "proj.bias",
}

TRANSFORMER_COMPONENT = "transformer"
CONDITION_ENCODER_COMPONENT = "condition_encoder"


def plan_flat_dit_keys(flat_keys: Iterable[str]) -> FlatRemapPlan:
    """Match every flat DiT-file key against the rules above.

    Pure key-set inspection -- no tensor is touched, so this is also what the
    header-only census (this phase's verification) runs directly against a
    safetensors header's key list.
    """
    plan = FlatRemapPlan()
    transformer_renames: Dict[str, str] = {}
    transformer_splits: Dict[str, Tuple[str, ...]] = {}
    condition_encoder_renames: Dict[str, str] = {}

    for key in flat_keys:
        layer_match = _DIT_LAYER_RE.match(key)
        if layer_match is not None:
            index, suffix = layer_match.group(1), layer_match.group(2)
            if suffix == _DIT_LAYER_QKV_SUFFIX:
                transformer_splits[key] = tuple(
                    f"transformer_blocks.{index}.{dest}" for dest in _DIT_QKV_SPLIT_SUFFIXES
                )
                continue
            dest_suffix = _DIT_LAYER_SUFFIX_RENAME.get(suffix)
            if dest_suffix is not None:
                transformer_renames[key] = f"transformer_blocks.{index}.{dest_suffix}"
                continue
            plan.unrecognized.append(key)
            continue

        if key in _DIT_NON_LAYER_RENAME:
            transformer_renames[key] = _DIT_NON_LAYER_RENAME[key]
            continue
        if key in _DIT_NON_LAYER_DROP:
            plan.dropped[key] = _DIT_NON_LAYER_DROP[key]
            continue
        if key in _CONDITION_ENCODER_RENAME:
            condition_encoder_renames[key] = _CONDITION_ENCODER_RENAME[key]
            continue

        plan.unrecognized.append(key)

    plan.renames[TRANSFORMER_COMPONENT] = transformer_renames
    plan.splits[TRANSFORMER_COMPONENT] = transformer_splits
    plan.renames[CONDITION_ENCODER_COMPONENT] = condition_encoder_renames
    return plan


def apply_flat_dit_state_dict(
    flat_state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """``{flat key: tensor}`` -> ``{"transformer": {...}, "condition_encoder": {...}}``.

    Raises ``ValueError`` if any input key is unrecognized -- see
    ``FlatRemapPlan.unrecognized``. Never partial: either every key in
    ``flat_state_dict`` lands in a produced state dict or is a documented
    drop, or this call raises.
    """
    plan = plan_flat_dit_keys(flat_state_dict.keys())
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 flat DiT remap: {len(plan.unrecognized)} key(s) matched no "
            f"known rule (first 10: {plan.unrecognized[:10]}). Refusing a partial remap "
            f"rather than silently dropping them -- see flat_remap.py's module docstring."
        )

    transformer_out: Dict[str, torch.Tensor] = {}
    for flat_key, dest_key in plan.renames[TRANSFORMER_COMPONENT].items():
        transformer_out[dest_key] = flat_state_dict[flat_key]
    for flat_key, dest_keys in plan.splits[TRANSFORMER_COMPONENT].items():
        tensor = flat_state_dict[flat_key]
        n = len(dest_keys)
        if tensor.shape[0] % n != 0:
            raise ValueError(
                f"MiniMax Music 3 flat DiT remap: {flat_key!r} has {tensor.shape[0]} rows, "
                f"not divisible by {n} (expected a fused [q|k|v] projection)."
            )
        chunks = torch.chunk(tensor, n, dim=0)
        for dest_key, chunk in zip(dest_keys, chunks):
            # `torch.chunk` returns VIEWS into `tensor`'s storage. Materialize
            # each split into its own storage: otherwise all `n` destination
            # Parameters alias one buffer, which (a) means `del
            # flat_state_dict` at the call site frees nothing, and (b) makes
            # `safetensors.save_file` refuse the resulting state dict with
            # "tensors share memory" on any future single-file export or
            # LoRA-merge save of a flat-loaded DiT.
            transformer_out[dest_key] = chunk.contiguous().clone()

    condition_encoder_out: Dict[str, torch.Tensor] = {
        dest_key: flat_state_dict[flat_key]
        for flat_key, dest_key in plan.renames[CONDITION_ENCODER_COMPONENT].items()
    }

    return {
        TRANSFORMER_COMPONENT: transformer_out,
        CONDITION_ENCODER_COMPONENT: condition_encoder_out,
    }


# ---------------------------------------------------------------------------
# Text encoder: model.layers.* / model.embed_tokens / model.norm / model.lm_head
#   (+ model.audio_decoder.* / model.audio_extra_embedding / tokenizer_json)
#   -> Qwen3ForCausalLM + MiniMaxMusic3RVQDepthDecoder
# ---------------------------------------------------------------------------

_LM_LAYER_RE = re.compile(r"^model\.layers\.\d+\.(.+)$")
_DEPTH_LAYER_RE = re.compile(r"^model\.audio_decoder\.layers\.(\d+)\.(.+)$")
_DEPTH_AUDIO_HEAD_RE = re.compile(r"^model\.audio_decoder\.audio_heads\.(\d+)\.weight$")

# Pruned-variant tells: any of these present means THIS module's remap
# functions refuse (see `is_pruned_flat_text_encoder` /
# `raise_if_pruned_flat_text_encoder`) -- the pruned layout is handled by the
# dedicated `pruned_text_encoder_remap` module instead, see
# `PrunedTextEncoderNotSupported`'s docstring. Verified against
# text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors's header
# (328 tensors: the vocab split plus fused `qkv_proj` / `gate_up_proj` per
# layer, replacing the non-pruned file's separate q/k/v/gate/up projections).
_PRUNED_TELLS = (
    "model.embed_tokens_prefill.weight",
    "model.embed_tokens_audio.weight",
    "model.lm_head_pruned.weight",
)

# Qwen3's own per-layer key spelling is IDENTICAL between the flat file and
# `official/language_model`'s own weights, so `model.layers.*` keys pass
# through unchanged -- but WHICH suffixes, exactly, is a WHITELIST, not "any
# suffix passes": the pruned variant's fused `self_attn.qkv_proj.weight` /
# `mlp.gate_up_proj.weight` (see `_PRUNED_TELLS`'s comment) are real
# `model.layers.N.*` keys too, and a bare `_LM_LAYER_RE.match()` with no
# suffix check would rename them straight through -- silently producing a
# state dict `Qwen3ForCausalLM` cannot load correctly (no `q_proj`/`k_proj`/
# `v_proj` at all), rather than refusing. This whitelist is what makes
# `plan_flat_text_encoder_keys` catch that (and any other future foreign
# key) as `unrecognized`, the same way `_DEPTH_LAYER_SUFFIX_RENAME` already
# does for the depth decoder.
_LM_LAYER_SUFFIX_WHITELIST = frozenset({
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
})

# Only `model.lm_head.weight` moves -- see `_LM_NON_LAYER_RENAME` -- because
# the flat repack nests it under `model.` while `Qwen3ForCausalLM.state_dict()`
# keeps `lm_head` at the TOP level (a sibling of `.model`, not inside it).
_LM_NON_LAYER_RENAME: Dict[str, str] = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.norm.weight": "model.norm.weight",
    "model.lm_head.weight": "lm_head.weight",
}

# See the module docstring's second "finding": this LM-namespaced tensor is
# actually the depth decoder's embedding table.
_LM_TO_DEPTH_DECODER_RENAME: Dict[str, str] = {
    "model.audio_extra_embedding.weight": "audio_embeddings.weight",
}

_TEXT_ENCODER_DROP: Dict[str, str] = {
    "tokenizer_json": (
        "an informational UTF-8 JSON byte blob (dtype U8), not a model weight -- no "
        "vendored module has a parameter or buffer for it. The tokenizer this loader uses "
        "is read from official/tokenizer/, not from this file."
    ),
}

# Depth-decoder per-layer suffix, flat -> vendored `MiniMaxMusic3DepthDecoderBlock`.
_DEPTH_LAYER_SUFFIX_RENAME: Dict[str, str] = {
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.down_proj.weight": "down_proj.weight",
    "mlp.gate_proj.weight": "gate_proj.weight",
    "mlp.up_proj.weight": "up_proj.weight",
    "self_attn.q_proj.weight": "attn.to_q.weight",
    "self_attn.k_proj.weight": "attn.to_k.weight",
    "self_attn.v_proj.weight": "attn.to_v.weight",
    "self_attn.o_proj.weight": "attn.to_out.weight",
}

_DEPTH_NON_LAYER_RENAME: Dict[str, str] = {
    "model.audio_decoder.norm.weight": "norm.weight",
    "model.audio_decoder.pos_embedding.weight": "pos_embedding.weight",
    "model.audio_decoder.projection.weight": "projection.weight",
}

LANGUAGE_MODEL_COMPONENT = "language_model"
RVQ_DEPTH_DECODER_COMPONENT = "rvq_depth_decoder"


def is_pruned_flat_text_encoder(flat_keys: Iterable[str]) -> bool:
    """``True`` iff ``flat_keys`` is the pruned-vocabulary text encoder.

    Positive evidence only (presence of any pruned-only tensor name) --
    matches the repo-wide convention of detecting a layout from what IS
    there, not from what is absent.
    """
    keys = set(flat_keys)
    return any(tell in keys for tell in _PRUNED_TELLS)


def raise_if_pruned_flat_text_encoder(flat_keys: Iterable[str]) -> None:
    """Raise ``PrunedTextEncoderNotSupported`` iff ``flat_keys`` is the pruned variant.
    HEADER-ONLY (no tensor bytes read). ``plan_flat_text_encoder_keys`` calls this first."""
    keys = list(flat_keys)
    if not is_pruned_flat_text_encoder(keys):
        return
    present = sorted(t for t in _PRUNED_TELLS if t in keys)
    raise PrunedTextEncoderNotSupported(
        f"this text encoder is the PRUNED-vocabulary flat layout (found {present}); use "
        f"core.models.minimax_music3.pruned_text_encoder_remap via "
        f"core.models.minimax_music3.loader."
        f"build_language_model_and_depth_decoder_from_pruned_flat_text_encoder instead."
    )


def plan_flat_text_encoder_keys(flat_keys: Iterable[str]) -> FlatRemapPlan:
    """Match every flat text-encoder key against the rules above.

    Raises ``PrunedTextEncoderNotSupported`` immediately if ``flat_keys``
    looks like the pruned variant -- see that exception's docstring and the
    design doc's phase 10.
    """
    keys = list(flat_keys)
    raise_if_pruned_flat_text_encoder(keys)

    plan = FlatRemapPlan()
    lm_renames: Dict[str, str] = {}
    depth_renames: Dict[str, str] = {}

    for key in keys:
        lm_layer_match = _LM_LAYER_RE.match(key)
        if lm_layer_match is not None:
            # Identity rename, but ONLY for a whitelisted suffix -- see
            # `_LM_LAYER_SUFFIX_WHITELIST`'s comment for why a bare regex
            # match here would be wrong.
            if lm_layer_match.group(1) in _LM_LAYER_SUFFIX_WHITELIST:
                lm_renames[key] = key
                continue
            plan.unrecognized.append(key)
            continue

        depth_layer_match = _DEPTH_LAYER_RE.match(key)
        if depth_layer_match is not None:
            index, suffix = depth_layer_match.group(1), depth_layer_match.group(2)
            dest_suffix = _DEPTH_LAYER_SUFFIX_RENAME.get(suffix)
            if dest_suffix is not None:
                depth_renames[key] = f"layers.{index}.{dest_suffix}"
                continue
            plan.unrecognized.append(key)
            continue

        audio_head_match = _DEPTH_AUDIO_HEAD_RE.match(key)
        if audio_head_match is not None:
            depth_renames[key] = f"audio_heads.{audio_head_match.group(1)}.weight"
            continue

        if key in _LM_NON_LAYER_RENAME:
            lm_renames[key] = _LM_NON_LAYER_RENAME[key]
            continue
        if key in _LM_TO_DEPTH_DECODER_RENAME:
            depth_renames[key] = _LM_TO_DEPTH_DECODER_RENAME[key]
            continue
        if key in _DEPTH_NON_LAYER_RENAME:
            depth_renames[key] = _DEPTH_NON_LAYER_RENAME[key]
            continue
        if key in _TEXT_ENCODER_DROP:
            plan.dropped[key] = _TEXT_ENCODER_DROP[key]
            continue

        plan.unrecognized.append(key)

    plan.renames[LANGUAGE_MODEL_COMPONENT] = lm_renames
    plan.renames[RVQ_DEPTH_DECODER_COMPONENT] = depth_renames
    return plan


def apply_flat_text_encoder_state_dict(
    flat_state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """``{flat key: tensor}`` -> ``{"language_model": {...}, "rvq_depth_decoder": {...}}``.

    Raises ``PrunedTextEncoderNotSupported`` for the pruned variant and
    ``ValueError`` for any unrecognized key, same totality guarantee as
    ``apply_flat_dit_state_dict``.
    """
    plan = plan_flat_text_encoder_keys(flat_state_dict.keys())
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 flat text encoder remap: {len(plan.unrecognized)} key(s) "
            f"matched no known rule (first 10: {plan.unrecognized[:10]}). Refusing a "
            f"partial remap rather than silently dropping them -- see flat_remap.py's "
            f"module docstring."
        )

    lm_out = {
        dest_key: flat_state_dict[flat_key]
        for flat_key, dest_key in plan.renames[LANGUAGE_MODEL_COMPONENT].items()
    }
    depth_out = {
        dest_key: flat_state_dict[flat_key]
        for flat_key, dest_key in plan.renames[RVQ_DEPTH_DECODER_COMPONENT].items()
    }
    return {
        LANGUAGE_MODEL_COMPONENT: lm_out,
        RVQ_DEPTH_DECODER_COMPONENT: depth_out,
    }


# ---------------------------------------------------------------------------
# Totality check -- shared by both remaps above and by the verification script.
# ---------------------------------------------------------------------------

def expected_module_state_dict_keys(module: torch.nn.Module) -> set:
    """``set(module.state_dict().keys())`` -- named here so callers do not repeat it."""
    return set(module.state_dict().keys())


def assert_state_dict_matches_module_keys(
    produced_keys: Iterable[str],
    expected_keys: Iterable[str],
    *,
    component: str,
) -> None:
    """Raise unless ``produced_keys`` and ``expected_keys`` are the SAME set.

    This is the "total and checked" gate the design doc's phase-9 task
    requires: every produced key must be consumed by the real vendored
    module, and every key the real vendored module's own ``state_dict()``
    declares must have been produced. Either direction failing means the
    remap is silently partial for this input, which is exactly the failure
    this phase must not ship.
    """
    produced = set(produced_keys)
    expected = set(expected_keys)
    missing = expected - produced
    extra = produced - expected
    if not missing and not extra:
        return
    parts: List[str] = []
    if missing:
        parts.append(
            f"{len(missing)} key(s) the {component} module expects were NOT produced "
            f"(first 10: {sorted(missing)[:10]})"
        )
    if extra:
        parts.append(
            f"{len(extra)} produced key(s) the {component} module does not expect "
            f"(first 10: {sorted(extra)[:10]})"
        )
    raise ValueError(
        f"MiniMax Music 3 flat remap for {component!r} is not total: " + "; ".join(parts) + ". "
        f"A partial remap that loads with strict=False would run and sound wrong -- this "
        f"module refuses instead."
    )
