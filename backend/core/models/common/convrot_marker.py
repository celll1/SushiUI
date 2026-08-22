"""Shared ConvRot INT8 ``.comfy_quant`` marker validation.

Factored out of ``core.models.minimax_h3.loader`` (the original, and still the
only caller that streams a lazy header/handle pair rather than a fully
materialized state dict) so SenseNova can validate the identical contract
without importing an H3-specific module. See
``core.models.common.quantized_checkpoint_guard`` for why a ``convrot: true``
marker must be validated before any tensor it names is installed.

MiniMax-Music3 (``core.models.minimax_music3.convrot_remap``) keeps its own,
behaviorally identical copy of this validator on purpose -- a deliberate
per-arch isolation decision, not an oversight; see that module's docstring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def supported_int8_convrot_marker(
    key: str,
    marker: torch.Tensor,
    header: Dict[str, Any],
    *,
    path: str,
) -> Optional[Dict[str, int]]:
    """Validate the one ConvRot contract implemented by this repo's loaders."""
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


def int8_convrot_layers_from_markers(
    handle,
    header: Dict[str, Any],
    *,
    path: str,
) -> Dict[str, Dict[str, int]]:
    """Return source-layer configs for validated ConvRot marker tensors.

    ``handle`` need only implement ``get_tensor(key)``; the H3 loader passes
    the same raw ``safe_open``/hybrid reader it maps the state dict through, so
    a marker always comes from the file its weight comes from (doc section
    4.3 of that loader).
    """
    layers: Dict[str, Dict[str, int]] = {}
    for key in header:
        if not key.endswith(".comfy_quant"):
            continue
        config = supported_int8_convrot_marker(
            key, handle.get_tensor(key), header, path=path
        )
        if config is not None:
            layers[key[: -len(".comfy_quant")]] = config
    return layers
