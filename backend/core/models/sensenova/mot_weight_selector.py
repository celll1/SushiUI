"""Persistent SenseNova MoT weight selection shared by generation and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from torch import nn


@dataclass(frozen=True)
class MotWeightSelection:
    gen_modules: Tuple[nn.Module, ...]
    und_modules: Tuple[nn.Module, ...]
    #: (generation, understanding) twins, populated only under
    #: ``require_exact_symmetry`` -- see ``_pair_layer_modules``.
    pairs: Tuple[Tuple[nn.Module, nn.Module], ...] = ()
    gen_unpaired: Tuple[nn.Module, ...] = ()
    und_unpaired: Tuple[nn.Module, ...] = ()


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


_ADAPTER_MARKERS = (".lora_down", ".lora_up")


def _is_adapter(path: str) -> bool:
    return any(marker in path for marker in _ADAPTER_MARKERS)


def _strip_adapters(entries):
    return [(path, module) for path, module in entries if not _is_adapter(path)]


def _pair_layer_modules(layer_index: int, gen, und):
    """Pair twins by signature, rejecting duplicates; return adapters as extras."""
    gen_index: dict = {}
    for path, module in _strip_adapters(gen):
        key = _base_signature(path, module)
        if key in gen_index:
            raise RuntimeError(
                f"SenseNova MoT weight halves are not pairable at layer {layer_index}: "
                f"two generation modules share the base signature {key[0]!r}"
            )
        gen_index[key] = module
    pairs = []
    for path, module in _strip_adapters(und):
        key = _base_signature(path, module)
        peer = gen_index.pop(key, None)
        if peer is None:
            raise RuntimeError(
                f"SenseNova MoT weight halves are not pairable at layer {layer_index}: "
                f"understanding module {path!r} has no unused generation twin"
            )
        pairs.append((peer, module))
    if gen_index:
        leftover = sorted(key[0] for key in gen_index)
        raise RuntimeError(
            f"SenseNova MoT weight halves are not pairable at layer {layer_index}: "
            f"generation modules {leftover[:3]} have no understanding twin"
        )
    return (
        pairs,
        [module for path, module in gen if _is_adapter(path)],
        [module for path, module in und if _is_adapter(path)],
    )


def select_mot_weight_modules(
    transformer: nn.Module,
    *,
    require_exact_symmetry: bool = False,
    allow_understanding_adapters: bool = False,
) -> MotWeightSelection:
    """Select owned Parameters and persistent buffers from both decoder halves.

    Training requires exact base symmetry and returns pairs for interleaved
    eviction. Four-phase may exclude adapters from that comparison because it
    brings the understanding half back for backward; adapters still travel with
    their own half. Inference only classifies modules by path.
    """
    layers: Iterable[nn.Module] = transformer.language_model.model.layers
    all_gen: list[nn.Module] = []
    all_und: list[nn.Module] = []
    all_pairs: list[tuple[nn.Module, nn.Module]] = []
    gen_unpaired: list[nn.Module] = []
    und_unpaired: list[nn.Module] = []
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
            pairs, gen_extras, und_extras = _pair_layer_modules(layer_index, gen, und)
            all_pairs.extend(pairs)
            gen_unpaired.extend(gen_extras)
            und_unpaired.extend(und_extras)
        all_gen.extend(module for _, module in gen)
        all_und.extend(module for _, module in und)

    if require_exact_symmetry and (layer_count != 42 or not all_gen or not all_und):
        raise RuntimeError(
            "SenseNova MoT eviction requires exactly 42 non-empty decoder layers "
            f"(found {layer_count})"
        )
    return MotWeightSelection(
        tuple(all_gen),
        tuple(all_und),
        tuple(all_pairs),
        tuple(gen_unpaired),
        tuple(und_unpaired),
    )
