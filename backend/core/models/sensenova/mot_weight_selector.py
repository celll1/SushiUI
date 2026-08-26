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
    """Pair each generation module with its understanding twin, by base signature.

    The symmetry check compares SETS of base signatures, which is strictly
    weaker than pairability: duplicate signatures collapse in a set, so a layer
    holding both ``dup.leaf_mot_gen`` and ``dup_mot_gen.leaf`` against a single
    ``dup.leaf`` passes symmetry with 2 generation modules against 1
    understanding module. A positional zip over the two returned lists would
    mis-pair such a tree silently, so the index below rejects a duplicate key
    and a leftover on either side instead.

    Adapters are the one documented exception: the symmetry check excludes them
    from BOTH signature sets, so nothing guarantees a counterpart. They are
    returned as per-side extras rather than treated as a pairing failure.
    """
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

    ``pairs``/``gen_unpaired``/``und_unpaired`` are populated only under
    ``require_exact_symmetry``, because only there is there a guarantee to rest
    a pairing on. The training evictor's interleaved transition consumes them;
    ``_pair_layer_modules`` documents why the symmetry check alone is not enough.
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
