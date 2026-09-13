"""Canonical YuE2 adapter targets and stage metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch.nn as nn

from core.adapters import is_adapter_covered, is_lora_wrappable_linear
from core.adapters.groups import TensorGroup, group_adapter_tensors


YUE2_STAGES = frozenset({"abc", "semantic", "ar", "nar"})
_ATTENTION_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_LEAVES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class YuE2LoRATarget:
    path: str
    parent: nn.Module
    attr: str
    module: nn.Module
    half: str


def _is_target(module: nn.Module | None) -> bool:
    return bool(module is not None and (
        is_lora_wrappable_linear(module) or is_adapter_covered(module)
    ))


def normalize_yue2_stages(value: str) -> tuple[str, ...]:
    stages = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    unknown = set(stages) - YUE2_STAGES
    if unknown:
        raise ValueError(f"Unknown YuE2 adapter stage(s): {sorted(unknown)}")
    if not stages:
        raise ValueError("YuE2 adapters require at least one apply stage")
    if "ar" in stages and ({"abc", "semantic"} & set(stages)):
        raise ValueError("YuE2 stage 'ar' already includes abc and semantic")
    return stages


def iter_yue2_lora_targets(
    transformer: nn.Module,
    *,
    half: str = "ar",
    scope: Dict[str, bool] | None = None,
) -> Iterator[YuE2LoRATarget]:
    """Yield the exact MoT projection surface shared by training and inference."""
    if half not in {"ar", "nar"}:
        raise ValueError("YuE2 LoRA half must be 'ar' or 'nar'")
    scope = {"attention": True, "mlp": False, **(scope or {})}
    backbone = getattr(transformer, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return
    attn_name = "self_attn" if half == "ar" else "nar_self_attn"
    mlp_name = "mlp" if half == "ar" else "nar_mlp"
    for index, layer in enumerate(layers):
        if scope["attention"]:
            parent = getattr(layer, attn_name, None)
            if parent is not None:
                for attr in _ATTENTION_LEAVES:
                    module = getattr(parent, attr, None)
                    if _is_target(module):
                        yield YuE2LoRATarget(
                            f"model.layers.{index}.{attn_name}.{attr}", parent, attr, module, half
                        )
        if scope["mlp"]:
            parent = getattr(layer, mlp_name, None)
            if parent is not None:
                for attr in _MLP_LEAVES:
                    module = getattr(parent, attr, None)
                    if _is_target(module):
                        yield YuE2LoRATarget(
                            f"model.layers.{index}.{mlp_name}.{attr}", parent, attr, module, half
                        )


def stage_is_active(adapter_stages: tuple[str, ...], stage: str) -> bool:
    if stage not in YUE2_STAGES:
        raise ValueError(f"Unknown YuE2 execution stage: {stage}")
    return stage in adapter_stages or ("ar" in adapter_stages and stage in {"abc", "semantic"})


def _checkpoint_stem(raw_stem: str) -> str | None:
    prefix = "lora_unet_"
    if raw_stem.startswith(prefix):
        return raw_stem[len(prefix):]
    if raw_stem.startswith("transformer."):
        return raw_stem[len("transformer."):].replace(".", "_")
    return None


def normalise_yue2_lora_state_dict(raw) -> Dict[str, TensorGroup]:
    """Map SushiUI sd-scripts keys to live-target flattened stems."""
    return group_adapter_tensors(raw, _checkpoint_stem).groups
