"""INT8 ConvRot remap for MiniMax Music 3's flat DiT and pruned text encoder --
design doc phase 13.

Item 9 (``flat_remap.py``) and item 10 (``pruned_text_encoder_remap.py``) already
turn the flat DiT's and the pruned text encoder's DENSE tensors into the
vendored modules' key space. This module does not reimplement that: it reuses
``flat_remap.plan_flat_dit_keys`` / ``apply_flat_dit_state_dict`` and
``pruned_text_encoder_remap.plan_pruned_text_encoder_keys`` /
``apply_pruned_text_encoder_state_dict`` UNCHANGED for every tensor those
functions already know how to place -- including the quantized ``.weight``
tensors themselves, whose INT8 codes ride through the identical rename/split
rules a dense ``.weight`` would (a row-wise split of int8 codes is exact; see
below). What this module adds is placing the two ConvRot SIDECARS
(``.weight_scale``, ``.comfy_quant``) a quantized Linear's ``.weight`` does not
carry, at the same destination the corresponding ``.weight`` was already
plotted to.

WHY THE SAME ROW SPLIT WORKS ON A CONVROT WEIGHT WITHOUT DEQUANTIZING FIRST:
ConvRot groups run along K (``in_features``, dim 1), never across
``out_features`` (dim 0), and every fused projection this module splits is
split by OUTPUT ROW (dim 0) -- so no K-group is ever cut. Full argument (and
the same one ``pruned_text_encoder_q8_0_remap.py`` makes for Q8_0) in
``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Quantization" / the phase-13
status entry -- not repeated here.

Marker validation (``supported_int8_convrot_marker``) is a DELIBERATE
duplicate of ``core.models.minimax_h3.loader._supported_int8_convrot_marker``,
not an import of it: this repo's existing discipline is that each
architecture's loader owns its own quantization CONTRACT validator (H3's DiT
and TE loaders both keep their own copies too), so a future contract change
for one architecture cannot silently change another's.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

__all__ = [
    "supported_int8_convrot_marker",
    "apply_flat_dit_state_dict_with_convrot",
    "apply_pruned_text_encoder_state_dict_with_convrot",
]


def supported_int8_convrot_marker(
    key: str,
    marker: torch.Tensor,
    header: Dict[str, Any],
    *,
    path: str,
) -> Optional[Dict[str, int]]:
    """Validate the ONE ConvRot contract this loader implements.

    Returns ``{"convrot_groupsize": 256, "marker_numel": N}`` for a marker
    that decodes to exactly ``{"format": "int8_tensorwise", "convrot": True,
    "convrot_groupsize": 256}`` AND whose sibling ``.weight`` / ``.weight_scale``
    header entries have the shapes that contract requires; ``None`` for any
    other marker (the caller then routes it to the generic declared-semantics
    refusal, same as an unrecognized H3 marker does).
    """
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


def _split_convrot_sidecar(
    marker: torch.Tensor,
    scale: torch.Tensor,
    dest_sizes: Tuple[Tuple[str, int], ...],
    config: Dict[str, int],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]]:
    """dim-0 (``out_features``) split of one fused ConvRot layer's
    ``(weight_scale, comfy_quant)`` pair into its per-destination pieces, by
    explicit SIZE (all entries ``>= 0``, the language model's GQA-uneven qkv)
    or an EQUAL n-way split (all entries ``-1``, every other fused projection
    this module splits) -- the same two conventions
    ``pruned_text_encoder_remap._apply_splits`` /
    ``pruned_text_encoder_q8_0_remap._split_packed`` use, so a plan built by
    ``plan_flat_dit_keys`` / ``plan_pruned_text_encoder_keys`` is
    interpretable identically by every applier that consumes it.

    Returns ``{dest_base: (scale_chunk, marker_copy, layer_config)}``, each
    entry its OWN clone (never a view/reuse across destinations -- see the
    inline comment below for why).
    """
    flat_scale = scale.to(torch.float32).reshape(-1)
    if any(size < 0 for _dest, size in dest_sizes):
        if not all(size < 0 for _dest, size in dest_sizes):
            raise ValueError(
                "MiniMax Music 3 ConvRot remap: a split plan mixes explicit and equal-split "
                "sizes -- this is a bug in the plan, not a checkpoint problem."
            )
        n = len(dest_sizes)
        if flat_scale.numel() % n:
            raise ValueError(
                f"MiniMax Music 3 ConvRot remap: weight_scale has {flat_scale.numel()} "
                f"value(s), not divisible by {n} (expected an equally-fused projection)."
            )
        sizes = [flat_scale.numel() // n] * n
    else:
        sizes = [size for _dest, size in dest_sizes]
        total = sum(sizes)
        if flat_scale.numel() != total:
            raise ValueError(
                f"MiniMax Music 3 ConvRot remap: weight_scale has {flat_scale.numel()} "
                f"value(s), expected {total} ({sizes})."
            )
    chunks = torch.split(flat_scale, sizes, dim=0)
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]] = {}
    for (dest_key, _size), chunk in zip(dest_sizes, chunks):
        dest_base = dest_key[: -len(".weight")]
        # `.clone()` the marker too, not just the scale chunk: the sibling
        # `pruned_text_encoder_remap._apply_splits` clones every split piece
        # for exactly this reason (a plain view/reuse would keep every
        # destination's buffer aliased to the SAME storage, and a future
        # `safetensors.save_file` of a ConvRot-loaded module would refuse
        # with "tensors share memory"). No music3 export path reads this
        # today -- latent, not live -- but the sibling already decided the
        # question and there is no reason to answer it differently here.
        out[dest_base] = (chunk.contiguous().clone(), marker.clone(), dict(config))
    return out


def _pop_convrot_sidecars(
    flat_state_dict: Mapping[str, torch.Tensor],
    convrot_source_layers: Mapping[str, Dict[str, int]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split ``flat_state_dict`` into ``(dense_state, sidecar_state)``:
    ``sidecar_state`` holds exactly the ``.comfy_quant`` / ``.weight_scale``
    entries belonging to a layer in ``convrot_source_layers``; everything else
    (including the quantized ``.weight`` tensors themselves) stays in
    ``dense_state`` -- see the module docstring for why the ``.weight`` tensors
    do not need to move.
    """
    dense_state: Dict[str, torch.Tensor] = {}
    sidecar_state: Dict[str, torch.Tensor] = {}
    for key, tensor in flat_state_dict.items():
        base = None
        if key.endswith(".comfy_quant"):
            base = key[: -len(".comfy_quant")]
        elif key.endswith(".weight_scale"):
            base = key[: -len(".weight_scale")]
        if base is not None and base in convrot_source_layers:
            sidecar_state[key] = tensor
        else:
            dense_state[key] = tensor
    return dense_state, sidecar_state


def apply_flat_dit_state_dict_with_convrot(
    flat_state_dict: Mapping[str, torch.Tensor],
    convrot_source_layers: Mapping[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, int]]]:
    """``flat_remap.apply_flat_dit_state_dict``, extended to place ConvRot
    sidecars.

    ``convrot_source_layers``: ``{source_base: {"convrot_groupsize": 256,
    "marker_numel": N}}`` for every VALIDATED ConvRot layer in the source file
    (from ``supported_int8_convrot_marker`` via the loader's header-only
    census). Empty means a plain (unquantized) flat DiT -- this function then
    degrades exactly to ``apply_flat_dit_state_dict``.

    Returns ``(component_state_dicts, dest_layer_configs)`` where
    ``dest_layer_configs`` is ``{dest_module_path: {"convrot_groupsize": 256,
    "marker_numel": N}}``, in the SHAPE
    ``core.models.common.convrot_int8_linear.swap_linears_to_convrot_int8``
    expects -- a fused source (``self_attn.to_qkv``) expands to its three
    destination Linears (``attn.to_q`` / ``attn.to_k`` / ``attn.to_v``), each
    carrying the SAME config (the contract, not the geometry, is what the
    config records).
    """
    from core.models.minimax_music3.flat_remap import (
        CONDITION_ENCODER_COMPONENT,
        TRANSFORMER_COMPONENT,
        apply_flat_dit_state_dict,
        plan_flat_dit_keys,
    )

    if not convrot_source_layers:
        return apply_flat_dit_state_dict(flat_state_dict), {}

    dense_state, sidecar_state = _pop_convrot_sidecars(flat_state_dict, convrot_source_layers)

    plan = plan_flat_dit_keys(dense_state.keys())
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 ConvRot DiT remap: {len(plan.unrecognized)} key(s) matched no "
            f"known rule (first 10: {plan.unrecognized[:10]}) after the ConvRot sidecars were "
            f"set aside. Refusing a partial remap rather than silently dropping them."
        )
    remapped = apply_flat_dit_state_dict(dense_state)

    dest_layer_configs: Dict[str, Dict[str, int]] = {}
    for source_base, config in convrot_source_layers.items():
        marker = sidecar_state.get(source_base + ".comfy_quant")
        scale = sidecar_state.get(source_base + ".weight_scale")
        if marker is None or scale is None:
            raise ValueError(
                f"MiniMax Music 3 ConvRot DiT layer {source_base!r} is missing its "
                f".comfy_quant/.weight_scale sidecar -- this is a bug in the header census, "
                f"not a checkpoint problem (the census already validated both are present)."
            )
        source_weight_key = source_base + ".weight"

        if source_weight_key in plan.splits.get(TRANSFORMER_COMPONENT, {}):
            dest_weight_keys = plan.splits[TRANSFORMER_COMPONENT][source_weight_key]
            component_out = remapped[TRANSFORMER_COMPONENT]
        elif source_weight_key in plan.renames.get(TRANSFORMER_COMPONENT, {}):
            dest_weight_keys = (plan.renames[TRANSFORMER_COMPONENT][source_weight_key],)
            component_out = remapped[TRANSFORMER_COMPONENT]
        elif source_weight_key in plan.renames.get(CONDITION_ENCODER_COMPONENT, {}):
            dest_weight_keys = (plan.renames[CONDITION_ENCODER_COMPONENT][source_weight_key],)
            component_out = remapped[CONDITION_ENCODER_COMPONENT]
        else:
            raise ValueError(
                f"MiniMax Music 3 ConvRot DiT layer {source_base!r}.weight was not found in "
                f"the flat DiT remap plan (renamed or split) -- this is a bug in the remap "
                f"tables, not a checkpoint problem."
            )

        dest_sizes = tuple((dest_key, -1) for dest_key in dest_weight_keys)
        for dest_base, (scale_chunk, marker_value, cfg) in _split_convrot_sidecar(
            marker, scale, dest_sizes, config
        ).items():
            component_out[dest_base + ".weight_scale"] = scale_chunk
            component_out[dest_base + ".comfy_quant"] = marker_value
            dest_layer_configs[dest_base] = cfg

    return remapped, dest_layer_configs


def apply_pruned_text_encoder_state_dict_with_convrot(
    flat_state_dict: Mapping[str, torch.Tensor],
    lm_config: Mapping[str, object],
    convrot_source_layers: Mapping[str, Dict[str, int]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, Dict[str, int]]]]:
    """``pruned_text_encoder_remap.apply_pruned_text_encoder_state_dict``,
    extended to place ConvRot sidecars for BOTH the language model's and the
    RVQ depth decoder's quantized Linears -- the pruned text encoder file
    composes two independent transformations (pruned vocabulary AND ConvRot);
    this function is the point where both are applied, in that order (the
    vocabulary split/dense remap first, via the unchanged
    ``apply_pruned_text_encoder_state_dict``, then the sidecar placement).

    Returns ``(component_state_dicts, dest_layer_configs_by_component)`` where
    ``dest_layer_configs_by_component`` is ``{"language_model": {...},
    "rvq_depth_decoder": {...}}``, each in the shape
    ``convrot_int8_linear.swap_linears_to_convrot_int8`` expects for that
    component's module tree.
    """
    from core.models.minimax_music3.pruned_text_encoder_remap import (
        LANGUAGE_MODEL_COMPONENT,
        RVQ_DEPTH_DECODER_COMPONENT,
        apply_pruned_text_encoder_state_dict,
        plan_pruned_text_encoder_keys,
    )

    if not convrot_source_layers:
        return apply_pruned_text_encoder_state_dict(flat_state_dict, lm_config), {
            LANGUAGE_MODEL_COMPONENT: {},
            RVQ_DEPTH_DECODER_COMPONENT: {},
        }

    dense_state, sidecar_state = _pop_convrot_sidecars(flat_state_dict, convrot_source_layers)

    plan = plan_pruned_text_encoder_keys(dense_state.keys(), lm_config)
    if plan.unrecognized:
        raise ValueError(
            f"MiniMax Music 3 ConvRot pruned text encoder remap: {len(plan.unrecognized)} "
            f"key(s) matched no known rule (first 10: {plan.unrecognized[:10]}) after the "
            f"ConvRot sidecars were set aside. Refusing a partial remap rather than silently "
            f"dropping them."
        )
    remapped = apply_pruned_text_encoder_state_dict(dense_state, lm_config)

    dest_layer_configs: Dict[str, Dict[str, Dict[str, int]]] = {
        LANGUAGE_MODEL_COMPONENT: {},
        RVQ_DEPTH_DECODER_COMPONENT: {},
    }
    for source_base, config in convrot_source_layers.items():
        marker = sidecar_state.get(source_base + ".comfy_quant")
        scale = sidecar_state.get(source_base + ".weight_scale")
        if marker is None or scale is None:
            raise ValueError(
                f"MiniMax Music 3 ConvRot pruned text encoder layer {source_base!r} is "
                f"missing its .comfy_quant/.weight_scale sidecar -- this is a bug in the "
                f"header census, not a checkpoint problem."
            )
        source_weight_key = source_base + ".weight"

        component = None
        dest_sizes: Optional[Tuple[Tuple[str, int], ...]] = None
        for candidate in (LANGUAGE_MODEL_COMPONENT, RVQ_DEPTH_DECODER_COMPONENT):
            if source_weight_key in plan.splits.get(candidate, {}):
                dest_sizes = plan.splits[candidate][source_weight_key]
                component = candidate
                break
            if source_weight_key in plan.renames.get(candidate, {}):
                dest_sizes = ((plan.renames[candidate][source_weight_key], -1),)
                component = candidate
                break
        if component is None:
            raise ValueError(
                f"MiniMax Music 3 ConvRot pruned text encoder layer {source_base!r}.weight "
                f"was not found in the remap plan (renamed or split) -- this is a bug in the "
                f"remap tables, not a checkpoint problem."
            )

        component_out = remapped[component]
        for dest_base, (scale_chunk, marker_value, cfg) in _split_convrot_sidecar(
            marker, scale, dest_sizes, config
        ).items():
            component_out[dest_base + ".weight_scale"] = scale_chunk
            component_out[dest_base + ".comfy_quant"] = marker_value
            dest_layer_configs[component][dest_base] = cfg

    return remapped, dest_layer_configs
