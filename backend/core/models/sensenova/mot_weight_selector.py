"""Persistent SenseNova MoT weight selection shared by generation and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from torch import nn


@dataclass(frozen=True)
class MotWeightSelection:
    gen_modules: Tuple[nn.Module, ...]
    und_modules: Tuple[nn.Module, ...]


def _owns_persistent_tensor(module: nn.Module) -> bool:
    if any(parameter is not None for parameter in module._parameters.values()):
        return True
    return any(
        buffer is not None and name not in module._non_persistent_buffers_set
        for name, buffer in module._buffers.items()
    )


def _base_signature(path: str, module: nn.Module) -> tuple[str, tuple[tuple, ...]]:
    normalized = path.replace("_mot_gen", "").replace(".original_module", "")
    tensors = tuple(
        sorted(
            [
                (name, tuple(value.shape), str(value.dtype))
                for name, value in module._parameters.items()
                if value is not None
            ]
            + [
                (name, tuple(value.shape), str(value.dtype))
                for name, value in module._buffers.items()
                if value is not None and name not in module._non_persistent_buffers_set
            ]
        )
    )
    return normalized, tensors


def _select_layer_modules(
    layer: nn.Module,
) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, nn.Module]]]:
    gen: list[tuple[str, nn.Module]] = []
    und: list[tuple[str, nn.Module]] = []
    for path, module in layer.named_modules():
        if not path or "rotary_emb" in path or not _owns_persistent_tensor(module):
            continue
        (gen if "_mot_gen" in path else und).append((path, module))
    return gen, und


def _strip_adapters(entries):
    return [
        (path, module)
        for path, module in entries
        if ".lora_down" not in path and ".lora_up" not in path
    ]


def select_mot_weight_modules(
    transformer: nn.Module,
    *,
    require_exact_symmetry: bool = False,
    allow_understanding_adapters: bool = False,
) -> MotWeightSelection:
    """Select owned Parameters and persistent buffers from both decoder halves.

    ``require_exact_symmetry`` (the TRAINING evictor) excludes LoRA children on
    the generation side only, so a run that also carries understanding LoRA
    fails this check rather than passing silently. That is the intended
    outcome for the THREE-state evictor: it stages the understanding half to CPU
    for the denoise phase, which cannot host a branch that must survive to
    backward. ``train_runner._apply_sensenova_training_contract`` refuses that
    combination up front with a readable message; this is the backstop.

    ``allow_understanding_adapters`` is the four-phase evictor (8.3.2), which
    brings the understanding half BACK for its own backward, so an understanding
    adapter is no longer a reason to fail. It excludes adapters from both sides
    of the signature comparison rather than from the generation side only; the
    adapter modules themselves stay in the returned lists and travel with their
    half. NOTE that on the only route which can set it today there are no adapter
    modules to exclude -- four-phase is full-fine-tune only, and full fine-tuning
    wraps nothing. It is the symmetry rule that generalizes here, not a
    configuration that exists yet; the installer additionally gates the flag on
    the training method so a bypassed front-line check cannot reach it from LoRA.

    The INFERENCE evictor leaves symmetry unchecked and classifies purely by
    path, so understanding wrappers travel with the understanding half.
    """
    layers: Iterable[nn.Module] = transformer.language_model.model.layers
    all_gen: list[nn.Module] = []
    all_und: list[nn.Module] = []
    layer_count = 0
    for layer_index, layer in enumerate(layers):
        layer_count += 1
        gen, und = _select_layer_modules(layer)
        if require_exact_symmetry:
            gen_base = {
                _base_signature(path, module) for path, module in _strip_adapters(gen)
            }
            und_entries = _strip_adapters(und) if allow_understanding_adapters else und
            und_base = {_base_signature(path, module) for path, module in und_entries}
            missing = sorted(und_base - gen_base)
            extra = sorted(gen_base - und_base)
            if not gen or not und or missing or extra:
                raise RuntimeError(
                    "SenseNova MoT weight halves are missing or asymmetric at "
                    f"layer {layer_index} (missing_gen={missing[:3]}, extra_gen={extra[:3]})"
                )
        all_gen.extend(module for _, module in gen)
        all_und.extend(module for _, module in und)

    if require_exact_symmetry and (layer_count != 42 or not all_gen or not all_und):
        raise RuntimeError(
            "SenseNova MoT eviction requires exactly 42 non-empty decoder layers "
            f"(found {layer_count})"
        )
    return MotWeightSelection(tuple(all_gen), tuple(all_und))
