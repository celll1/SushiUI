"""Packed Q8_0 remap for MiniMax Music 3's pruned GGUF text encoder -- design
doc phase 12 ("Q8_0 residency").

Sibling of ``pruned_text_encoder_remap.py`` (phase 10): SAME key plan
(``plan_pruned_text_encoder_keys``), SAME fused-projection splits (the
language model's GQA-uneven ``qkv_proj``, the depth decoder's equal-thirds
``qkv_proj``, both models' equal-halves ``gate_up_proj``) -- but a
Q8_0-typed source tensor is kept PACKED (``PackedQ8_0Weight``: int8 codes +
per-block float16 scale, from ``gguf_container.GGUFStateDict.
get_q8_0_packed``) instead of being read as a dense tensor and split with
``torch.split``/``torch.chunk``.

WHY THE SAME ROW SPLIT WORKS ON PACKED DATA WITHOUT DEQUANTIZING FIRST. Q8_0
blocks run along ``in_features`` (dim 1, 32 values per block -- see
``gguf_container``'s block-layout docstring), never across ``out_features``
(dim 0). The plan's fused-projection splits are ALL along dim 0 (splitting a
fused ``[q | k | v]`` or ``[gate | up]`` weight into its pieces by ROW), so
splitting ``codes`` and ``scale`` with the identical row ranges the dense
plan already computes is exact: each destination row keeps its OWN,
untouched set of blocks. No block is ever cut in half by a dim-0 split, and
this module never needs to know a block's contents to perform one -- see
``_split_packed`` below, which is a thin dim-0-only counterpart of
``pruned_text_encoder_remap._apply_splits``.

WHAT STAYS DENSE. F32/BF16 tensors (every norm, the three vocabulary tables,
and ``tokenizer_json``, which is dropped exactly as the dense pruned builders
drop it) are read through ``GGUFStateDict.__getitem__`` unchanged -- this
module changes nothing about how those are handled, only how Q8_0 tensors
are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch

from core.models.common import gguf_container
from core.models.minimax_music3.pruned_text_encoder_remap import (
    LANGUAGE_MODEL_COMPONENT,
    RVQ_DEPTH_DECODER_COMPONENT,
    plan_pruned_text_encoder_keys,
)

__all__ = [
    "PackedQ8_0Weight",
    "apply_pruned_text_encoder_state_dict_packed",
]


@dataclass(frozen=True)
class PackedQ8_0Weight:
    """One Linear weight kept PACKED: Q8_0 ``codes`` (int8) + ``scale``
    (float16). See ``core.models.common.gguf_q8_0_linear.dequantize_q8_0``
    for how -- and when -- the two are ever combined into a dense tensor.
    """

    codes: torch.Tensor
    scale: torch.Tensor


def _split_packed(
    codes: torch.Tensor,
    scale: torch.Tensor,
    dest_sizes: Tuple[Tuple[str, int], ...],
) -> Dict[str, "PackedQ8_0Weight"]:
    """dim-0 (``out_features``) split of one packed tensor into its pieces,
    by explicit SIZE (all entries >= 0) or an EQUAL n-way split (all entries
    ``-1``) -- the same two conventions
    ``pruned_text_encoder_remap._apply_splits`` uses for the dense path, kept
    in lockstep deliberately: a plan built by
    ``plan_pruned_text_encoder_keys`` must be interpretable identically by
    both appliers, or the dense and packed builders would silently diverge
    on which rows land in which destination.
    """
    if any(size < 0 for _dest, size in dest_sizes):
        if not all(size < 0 for _dest, size in dest_sizes):
            raise ValueError(
                "MiniMax Music 3 packed Q8_0 remap: a split plan mixes explicit and equal-split "
                "sizes -- this is a bug in plan_pruned_text_encoder_keys, not a checkpoint problem."
            )
        n = len(dest_sizes)
        if codes.shape[0] % n != 0:
            raise ValueError(
                f"MiniMax Music 3 packed Q8_0 remap: {codes.shape[0]} rows not divisible by {n} "
                f"(expected an equally-fused projection)."
            )
        sizes = [codes.shape[0] // n] * n
    else:
        sizes = [size for _dest, size in dest_sizes]
        total = sum(sizes)
        if codes.shape[0] != total:
            raise ValueError(
                f"MiniMax Music 3 packed Q8_0 remap: {codes.shape[0]} rows, expected {total} "
                f"({sizes}, from the language model's own config -- see lm_qkv_split_sizes)."
            )
    code_chunks = torch.split(codes, sizes, dim=0)
    scale_chunks = torch.split(scale, sizes, dim=0)
    out: Dict[str, PackedQ8_0Weight] = {}
    for (dest_key, _size), c, s in zip(dest_sizes, code_chunks, scale_chunks):
        # `.contiguous().clone()`, matching every other splitter in this
        # package (`flat_remap.apply_flat_dit_state_dict`,
        # `pruned_text_encoder_remap._apply_splits`): a plain `torch.split`
        # result is a VIEW into the fused tensor's storage, which would keep
        # the whole fused (codes, scale) pair alive for as long as ANY one
        # split piece is referenced, and would make a future
        # `torch.save`/state-dict export of a packed-loaded module refuse
        # with "tensors share memory".
        out[dest_key] = PackedQ8_0Weight(c.contiguous().clone(), s.contiguous().clone())
    return out


def apply_pruned_text_encoder_state_dict_packed(
    state: "gguf_container.GGUFStateDict",
    lm_config: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    """``{"language_model": {dest_key: tensor | PackedQ8_0Weight}, "rvq_depth_decoder": {...}}``.

    Every dest key that ``pruned_text_encoder_remap.
    apply_pruned_text_encoder_state_dict`` would have produced is produced
    here too -- SAME totality guarantee (``plan.unrecognized`` still raises)
    -- but a key whose SOURCE tensor is Q8_0-typed carries a
    ``PackedQ8_0Weight`` instead of a dense ``torch.Tensor``. A caller
    (``core.models.minimax_music3.loader.
    build_language_model_and_depth_decoder_from_pruned_gguf_q8_0_text_encoder``)
    is expected to route ``PackedQ8_0Weight`` entries through
    ``core.models.common.gguf_q8_0_linear.GGUFQ8_0Linear`` and everything
    else through the ordinary ``load_state_dict`` path -- this function does
    not know about that split and cannot itself confirm it happened.
    """
    header = state.header
    ggml_type_by_name = {t.name: t.ggml_type_name for t in header.tensors}
    plan = plan_pruned_text_encoder_keys(header.tensor_names(), lm_config)
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 packed Q8_0 text encoder remap: {len(plan.unrecognized)} key(s) "
            f"matched no known rule (first 10: {plan.unrecognized[:10]}). Refusing a partial "
            f"remap rather than silently dropping them -- see pruned_text_encoder_remap.py's "
            f"module docstring, whose plan this reuses unchanged."
        )

    def _is_q8_0(flat_key: str) -> bool:
        return ggml_type_by_name.get(flat_key) == "Q8_0"

    def _build(component: str) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for flat_key, dest_key in plan.renames[component].items():
            if _is_q8_0(flat_key):
                codes, scale = state.get_q8_0_packed(flat_key)
                out[dest_key] = PackedQ8_0Weight(codes, scale)
            else:
                out[dest_key] = state[flat_key]
        for flat_key, dest_sizes in plan.splits[component].items():
            if not _is_q8_0(flat_key):
                # Every fused projection this plan splits is Q8_0-typed on
                # the real staged checkpoint (169/169; see the design doc
                # phase 12 census) -- a non-Q8_0 fused source is refused
                # rather than served through an untested, unreachable second
                # code path. If a future GGUF file mixes formats, add and
                # test a dense-split branch here deliberately rather than
                # reviving one that was never exercised.
                raise NotImplementedError(
                    f"MiniMax Music 3 packed Q8_0 text encoder remap: {flat_key!r} is a fused "
                    f"projection this plan splits, but its source tensor is "
                    f"{ggml_type_by_name.get(flat_key)!r}, not Q8_0 -- this builder only handles "
                    f"a Q8_0-typed fused source (every one on the real staged checkpoint is)."
                )
            codes, scale = state.get_q8_0_packed(flat_key)
            out.update(_split_packed(codes, scale, dest_sizes))
        return out

    lm_out = _build(LANGUAGE_MODEL_COMPONENT)
    depth_out = _build(RVQ_DEPTH_DECODER_COMPONENT)
    return {LANGUAGE_MODEL_COMPONENT: lm_out, RVQ_DEPTH_DECODER_COMPONENT: depth_out}
