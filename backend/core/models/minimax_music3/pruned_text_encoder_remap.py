"""Pruned-vocabulary flat text-encoder remap for MiniMax Music 3 -- design doc phase 10.

Serves the flat pruned safetensors (this commit's caller,
``text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors``) AND
design doc phase 11's future GGUF reader with the SAME remap: the pruned
GGUF text encoder's tensor names are identical to the pruned flat safetensors'
(design doc, "GGUF weights" -- "the text encoder is the same 328 tensors as
the flat pruned repack"), so the remap logic here never assumes a safetensors
origin, the same discipline ``flat_remap.py`` follows for the DiT and the
non-pruned text encoder.

Two things distinguish this file from the pruned checkpoint, verified against
the real snapshot (see this module's own docstrings below for exact numbers,
not asserted from memory):

1. **The vocabulary is split three ways**, not merged into one table:
   ``model.embed_tokens_prefill`` (text, 151,675 rows) + ``model.embed_tokens_audio``
   (semantic codes, 16,384 rows, INDEXED WITHOUT ``AUDIO_CODE_OFFSET``) +
   ``model.lm_head_pruned`` (16,385 rows: row 0 is end-of-audio, rows 1..16384
   are semantic codes 0..16383 -- see ``EOA_LM_HEAD_ROW``'s docstring for how
   that row assignment was determined, not guessed).
2. **Every layer's attention and MLP are pre-fused**: ``self_attn.qkv_proj``
   (``[q | k | v]`` row-contiguous, GQA-uneven: ``q_dim = num_attention_heads *
   head_dim``, ``kv_dim = num_key_value_heads * head_dim``, NOT an equal
   three-way split) and ``mlp.gate_up_proj`` (``[gate | up]``, an EVEN
   two-way split). This applies to BOTH the language model's 36 layers and
   the RVQ depth decoder's 4 layers (the depth decoder is plain MHA --
   ``num_key_value_heads`` is absent from its config, so its qkv_proj splits
   evenly in three, unlike the language model's).

Numerically verified against the real snapshot (``M:/model/minimax-music3``):
every per-layer split tensor (q/k/v/gate/up, both the language model and the
depth decoder) and both vocab-table rows (``embed_tokens_prefill`` /
``embed_tokens_audio`` against ``official/language_model``'s single
``embed_tokens`` at ``[0:151675)`` / ``[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET +
16384)``, and ``lm_head_pruned`` row 0 / rows ``1:16385`` against
``official/language_model``'s ``lm_head`` at ``AUDIO_END_TOKEN_ID`` /
``[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + 16384)``) matches the corresponding
``official/`` weight to the LAST BIT of its bf16 representation (max abs diff
0.0, not "close"). The pruned checkpoint's body is the SAME weights as
``official/``, repacked -- not a retrained or approximated variant.

What this module does NOT do: decide which vocab table an AR-loop caller
should read from at generation time (that dispatch -- "is this language model
the full-vocabulary or the pruned-vocabulary layout" -- lives in
``core.models.minimax_music3.vocab_view``, which reads back the
``lm_head_pruned`` / ``embed_tokens_audio`` attributes this module's loader
caller attaches). This module only turns the pruned file's tensors into a
state dict a REAL ``Qwen3ForCausalLM`` (patched with those two extra
attributes -- see ``core.models.minimax_music3.loader.
build_language_model_and_depth_decoder_from_pruned_flat_text_encoder``) can
load ``strict=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple

import torch

from core.models.minimax_music3 import flat_remap as _flat_remap

__all__ = [
    "PrunedRemapPlan",
    "SEMANTIC_VOCAB_SIZE",
    "AUDIO_HEAD_VOCAB_SIZE",
    "EOA_LM_HEAD_ROW",
    "LANGUAGE_MODEL_COMPONENT",
    "RVQ_DEPTH_DECODER_COMPONENT",
    "lm_qkv_split_sizes",
    "plan_pruned_text_encoder_keys",
    "apply_pruned_text_encoder_state_dict",
]

LANGUAGE_MODEL_COMPONENT = _flat_remap.LANGUAGE_MODEL_COMPONENT  # "language_model"
RVQ_DEPTH_DECODER_COMPONENT = _flat_remap.RVQ_DEPTH_DECODER_COMPONENT  # "rvq_depth_decoder"

# ``design doc defaults.py`` already declares ``SEMANTIC_VOCAB_SIZE = 16384``
# for the semantic-code alphabet; re-imported (not re-declared) so the two
# numbers cannot drift.
from core.models.minimax_music3.defaults import SEMANTIC_VOCAB_SIZE  # noqa: E402

# `model.lm_head_pruned.weight` is `[SEMANTIC_VOCAB_SIZE + 1, hidden]` -- the
# 16,384 semantic-code rows plus one end-of-audio row. See `EOA_LM_HEAD_ROW`.
AUDIO_HEAD_VOCAB_SIZE = SEMANTIC_VOCAB_SIZE + 1

# ---------------------------------------------------------------------------
# Which `lm_head_pruned` row is end-of-audio -- DETERMINED, not assumed.
#
# Measured directly against the real snapshot (both files bf16, no cast):
#   pruned "model.lm_head_pruned.weight"[0]      == official lm_head.weight[AUDIO_END_TOKEN_ID]        (max abs diff 0.0)
#   pruned "model.lm_head_pruned.weight"[1:16385] == official lm_head.weight[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET+16384] (max abs diff 0.0)
# i.e. row 0 is the end-of-audio classifier weight, and semantic code `c`
# lives at row `c + 1` -- the opposite ordering would have put end-of-audio
# LAST; it is FIRST. `core.models.minimax_music3.vocab_view.PrunedVocabView`
# is the sole reader of this constant at generation time.
# ---------------------------------------------------------------------------
EOA_LM_HEAD_ROW = 0


def lm_qkv_split_sizes(lm_config: Mapping[str, object]) -> Tuple[int, int, int]:
    """``(q_dim, k_dim, v_dim)`` for the language model's fused ``qkv_proj``, from its own config.

    NOT an equal three-way split (unlike the depth decoder's, and unlike the
    DiT's fused ``to_qkv`` in ``flat_remap.py``): the language model is
    Qwen3's grouped-query attention, so `k_dim == v_dim == num_key_value_heads
    * head_dim`, which is smaller than `q_dim == num_attention_heads *
    head_dim` on the real checkpoint (32 vs 8 heads). Falls back to
    `hidden_size // num_attention_heads` for `head_dim` and to
    `num_attention_heads` for `num_key_value_heads` (plain MHA) only if either
    is absent from the config -- matching how `transformers`' own `Qwen3Config`
    resolves the same two fields when they are omitted.
    """
    num_attention_heads = int(lm_config["num_attention_heads"])
    hidden_size = int(lm_config["hidden_size"])
    head_dim = int(lm_config["head_dim"]) if lm_config.get("head_dim") is not None else hidden_size // num_attention_heads
    num_key_value_heads = (
        int(lm_config["num_key_value_heads"]) if lm_config.get("num_key_value_heads") is not None else num_attention_heads
    )
    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    return q_dim, kv_dim, kv_dim


@dataclass
class PrunedRemapPlan:
    """Like ``flat_remap.FlatRemapPlan``, but a fused-projection split needs a
    SIZE alongside each destination key (the language model's qkv split is
    NOT equal thirds -- see ``lm_qkv_split_sizes``), which
    ``flat_remap.FlatRemapPlan.splits`` (``Tuple[str, ...]``, implicitly
    equal-sized) cannot express. Kept as a separate dataclass rather than
    widening the shared one -- the DiT and non-pruned text-encoder remaps
    never need per-split sizes and should not carry a field only this remap
    populates.

    * ``renames[component]``: ``{flat_key: dest_key}``, same as ``FlatRemapPlan``.
    * ``splits[component]``: ``{flat_key: ((dest_key, size), ...)}`` -- one
      fused tensor in, split along dim 0 into the given SIZES, in order.
    * ``dropped`` / ``unrecognized``: same meaning as ``FlatRemapPlan``.
    """

    renames: Dict[str, Dict[str, str]] = field(default_factory=dict)
    splits: Dict[str, Dict[str, Tuple[Tuple[str, int], ...]]] = field(default_factory=dict)
    dropped: Dict[str, str] = field(default_factory=dict)
    unrecognized: List[str] = field(default_factory=list)


_LM_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_LM_FUSED_QKV_SUFFIX = "self_attn.qkv_proj.weight"
_LM_FUSED_GATE_UP_SUFFIX = "mlp.gate_up_proj.weight"

_DEPTH_FUSED_QKV_SUFFIX = "self_attn.qkv_proj.weight"
_DEPTH_FUSED_GATE_UP_SUFFIX = "mlp.gate_up_proj.weight"

# Non-layer vocab-table keys -- the substance of this module. All three stay
# in the SAME "language_model" component: `embed_tokens_audio` and
# `lm_head_pruned` are attached as EXTRA attributes on the same
# `Qwen3ForCausalLM` instance the loader builds (see the loader's docstring),
# not separate modules of their own.
_LM_PREFILL_EMBED_RENAME = {"model.embed_tokens_prefill.weight": "model.embed_tokens.weight"}
_LM_AUDIO_EMBED_RENAME = {"model.embed_tokens_audio.weight": "model.embed_tokens_audio.weight"}
_LM_HEAD_PRUNED_RENAME = {"model.lm_head_pruned.weight": "lm_head_pruned.weight"}
_LM_NORM_RENAME = {"model.norm.weight": "model.norm.weight"}


def plan_pruned_text_encoder_keys(
    flat_keys: Iterable[str],
    lm_config: Mapping[str, object],
) -> PrunedRemapPlan:
    """Match every pruned flat text-encoder key against this module's rules.

    ``lm_config`` supplies the language model's ``qkv_proj`` split sizes
    (``lm_qkv_split_sizes``) -- the ONLY config-dependent decision in this
    plan; every other split (the depth decoder's qkv, both models' gate_up)
    is architecturally fixed (equal halves/thirds) and needs no config.

    Pure key-set inspection plus the one config read above -- no tensor byte
    is touched, matching ``flat_remap.plan_flat_dit_keys``'s header-only
    contract.
    """
    plan = PrunedRemapPlan()
    lm_renames: Dict[str, str] = {}
    lm_splits: Dict[str, Tuple[Tuple[str, int], ...]] = {}
    depth_renames: Dict[str, str] = {}
    depth_splits: Dict[str, Tuple[Tuple[str, int], ...]] = {}

    q_dim, k_dim, v_dim = lm_qkv_split_sizes(lm_config)

    for key in flat_keys:
        lm_layer_match = _LM_LAYER_RE.match(key)
        if lm_layer_match is not None:
            index, suffix = lm_layer_match.group(1), lm_layer_match.group(2)
            if suffix == _LM_FUSED_QKV_SUFFIX:
                lm_splits[key] = (
                    (f"model.layers.{index}.self_attn.q_proj.weight", q_dim),
                    (f"model.layers.{index}.self_attn.k_proj.weight", k_dim),
                    (f"model.layers.{index}.self_attn.v_proj.weight", v_dim),
                )
                continue
            if suffix == _LM_FUSED_GATE_UP_SUFFIX:
                # Equal halves -- Qwen3's own `gate_up_proj` fusion convention
                # (verified against the real checkpoint: `intermediate_size *
                # 2` rows, both halves bit-identical to `official/`'s separate
                # `gate_proj` / `up_proj`). No config lookup needed: the size
                # is derived from the TENSOR itself at apply time (see
                # `apply_pruned_text_encoder_state_dict`), matching this
                # plan's own general policy of deferring tensor-shaped facts
                # to apply time and using config only for the one split
                # (qkv) that genuinely cannot be inferred from a row count.
                lm_splits[key] = (
                    (f"model.layers.{index}.mlp.gate_proj.weight", -1),
                    (f"model.layers.{index}.mlp.up_proj.weight", -1),
                )
                continue
            if suffix in _flat_remap._LM_LAYER_SUFFIX_WHITELIST:
                lm_renames[key] = key
                continue
            plan.unrecognized.append(key)
            continue

        depth_layer_match = _flat_remap._DEPTH_LAYER_RE.match(key)
        if depth_layer_match is not None:
            index, suffix = depth_layer_match.group(1), depth_layer_match.group(2)
            if suffix == _DEPTH_FUSED_QKV_SUFFIX:
                # Equal thirds -- the RVQ depth decoder is plain MHA (no
                # `num_key_value_heads` in its config; verified against the
                # real checkpoint: 12288 rows = 3 * 4096, each third
                # bit-identical to `official/`'s separate `attn.to_q` /
                # `attn.to_k` / `attn.to_v`).
                dest_suffixes = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
                depth_splits[key] = tuple(
                    (f"layers.{index}.{_flat_remap._DEPTH_LAYER_SUFFIX_RENAME[s]}", -1) for s in dest_suffixes
                )
                continue
            if suffix == _DEPTH_FUSED_GATE_UP_SUFFIX:
                dest_suffixes = ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                depth_splits[key] = tuple(
                    (f"layers.{index}.{_flat_remap._DEPTH_LAYER_SUFFIX_RENAME[s]}", -1) for s in dest_suffixes
                )
                continue
            dest_suffix = _flat_remap._DEPTH_LAYER_SUFFIX_RENAME.get(suffix)
            if dest_suffix is not None:
                depth_renames[key] = f"layers.{index}.{dest_suffix}"
                continue
            plan.unrecognized.append(key)
            continue

        audio_head_match = _flat_remap._DEPTH_AUDIO_HEAD_RE.match(key)
        if audio_head_match is not None:
            depth_renames[key] = f"audio_heads.{audio_head_match.group(1)}.weight"
            continue

        if key in _LM_PREFILL_EMBED_RENAME:
            lm_renames[key] = _LM_PREFILL_EMBED_RENAME[key]
            continue
        if key in _LM_AUDIO_EMBED_RENAME:
            lm_renames[key] = _LM_AUDIO_EMBED_RENAME[key]
            continue
        if key in _LM_HEAD_PRUNED_RENAME:
            lm_renames[key] = _LM_HEAD_PRUNED_RENAME[key]
            continue
        if key in _LM_NORM_RENAME:
            lm_renames[key] = _LM_NORM_RENAME[key]
            continue
        if key in _flat_remap._LM_TO_DEPTH_DECODER_RENAME:
            depth_renames[key] = _flat_remap._LM_TO_DEPTH_DECODER_RENAME[key]
            continue
        if key in _flat_remap._DEPTH_NON_LAYER_RENAME:
            depth_renames[key] = _flat_remap._DEPTH_NON_LAYER_RENAME[key]
            continue
        if key in _flat_remap._TEXT_ENCODER_DROP:
            plan.dropped[key] = _flat_remap._TEXT_ENCODER_DROP[key]
            continue

        plan.unrecognized.append(key)

    plan.renames[LANGUAGE_MODEL_COMPONENT] = lm_renames
    plan.splits[LANGUAGE_MODEL_COMPONENT] = lm_splits
    plan.renames[RVQ_DEPTH_DECODER_COMPONENT] = depth_renames
    plan.splits[RVQ_DEPTH_DECODER_COMPONENT] = depth_splits
    return plan


def _apply_splits(
    flat_state_dict: Mapping[str, torch.Tensor],
    splits: Dict[str, Tuple[Tuple[str, int], ...]],
    out: Dict[str, torch.Tensor],
) -> None:
    """Split every fused tensor in ``splits`` into ``out``, by explicit SIZE where given (>=0) or by an
    EQUAL n-way split (all sizes `-1`, matching the plan's own convention above for the architecturally-fixed
    splits). Mirrors `flat_remap.apply_flat_dit_state_dict`'s `.contiguous().clone()` treatment of `torch.split`
    results: a plain `torch.split`/`torch.chunk` output is a VIEW sharing the fused tensor's storage, which
    would otherwise (a) keep the whole fused tensor alive in memory for as long as any one split piece is
    referenced, and (b) make a future `safetensors.save_file` of a pruned-loaded language model refuse with
    "tensors share memory".
    """
    for flat_key, dest_sizes in splits.items():
        tensor = flat_state_dict[flat_key]
        if any(size < 0 for _dest, size in dest_sizes):
            if not all(size < 0 for _dest, size in dest_sizes):
                raise ValueError(
                    f"MiniMax Music 3 pruned text encoder remap: {flat_key!r} mixes explicit and "
                    f"equal-split sizes in the same split plan -- this is a bug in "
                    f"plan_pruned_text_encoder_keys, not a checkpoint problem."
                )
            n = len(dest_sizes)
            if tensor.shape[0] % n != 0:
                raise ValueError(
                    f"MiniMax Music 3 pruned text encoder remap: {flat_key!r} has {tensor.shape[0]} rows, "
                    f"not divisible by {n} (expected an equally-fused projection)."
                )
            chunks = torch.chunk(tensor, n, dim=0)
        else:
            total = sum(size for _dest, size in dest_sizes)
            if tensor.shape[0] != total:
                raise ValueError(
                    f"MiniMax Music 3 pruned text encoder remap: {flat_key!r} has {tensor.shape[0]} rows, "
                    f"expected {total} ({[size for _dest, size in dest_sizes]}, from the language model's "
                    f"own config -- see lm_qkv_split_sizes)."
                )
            chunks = torch.split(tensor, [size for _dest, size in dest_sizes], dim=0)
        for (dest_key, _size), chunk in zip(dest_sizes, chunks):
            out[dest_key] = chunk.contiguous().clone()


def apply_pruned_text_encoder_state_dict(
    flat_state_dict: Mapping[str, torch.Tensor],
    lm_config: Mapping[str, object],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """``{flat key: tensor}`` -> ``{"language_model": {...}, "rvq_depth_decoder": {...}}``.

    Raises ``ValueError`` for any unrecognized key or a fused tensor whose row
    count does not match its expected split -- never a partial remap, same
    totality guarantee as ``flat_remap.apply_flat_text_encoder_state_dict``.
    The caller (``core.models.minimax_music3.loader.
    build_language_model_and_depth_decoder_from_pruned_flat_text_encoder``)
    is expected to run ``flat_remap.assert_state_dict_matches_module_keys``
    against the PATCHED ``Qwen3ForCausalLM`` (with ``lm_head`` removed and
    ``lm_head_pruned`` / ``model.embed_tokens_audio`` attached) -- this
    function does not know about that patching and cannot itself confirm
    totality against a real module.
    """
    plan = plan_pruned_text_encoder_keys(flat_state_dict.keys(), lm_config)
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 pruned text encoder remap: {len(plan.unrecognized)} key(s) matched no "
            f"known rule (first 10: {plan.unrecognized[:10]}). Refusing a partial remap rather than "
            f"silently dropping them -- see pruned_text_encoder_remap.py's module docstring."
        )

    lm_out: Dict[str, torch.Tensor] = {
        dest_key: flat_state_dict[flat_key] for flat_key, dest_key in plan.renames[LANGUAGE_MODEL_COMPONENT].items()
    }
    _apply_splits(flat_state_dict, plan.splits[LANGUAGE_MODEL_COMPONENT], lm_out)

    depth_out: Dict[str, torch.Tensor] = {
        dest_key: flat_state_dict[flat_key] for flat_key, dest_key in plan.renames[RVQ_DEPTH_DECODER_COMPONENT].items()
    }
    _apply_splits(flat_state_dict, plan.splits[RVQ_DEPTH_DECODER_COMPONENT], depth_out)

    return {LANGUAGE_MODEL_COMPONENT: lm_out, RVQ_DEPTH_DECODER_COMPONENT: depth_out}
