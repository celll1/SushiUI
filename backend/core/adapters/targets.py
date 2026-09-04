"""Target-side policy for adapter injection: what can be wrapped, in what dtype,
and how much of the base is weight-only quantized.

Plus ``AdapterTarget``, the target-topology record of design-doc section 3
(``docs/guides/LYCORIS_ADAPTER_DESIGN.md``).

The three predicates were moved verbatim from
``core.training.adapters.base_adapter``, which now imports
``count_quantized_linears`` from here like every other caller.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn

from .composite import is_adapter_covered

Slot = Union[str, int]

_QUANTIZED_KIND_BY_CLASS = {
    "Int8Linear": "int8",
    "Fp8Linear": "fp8",
    "W4A8Linear": "w4a8",
}


def _quantized_linear_classes() -> Tuple[type, ...]:
    """The weight-only quantized Linear classes, imported lazily.

    Lazy because they live under ``core.models.*``: hoisting the import to
    module scope would give this package an import-time edge into the model
    loaders. Raises when they are unavailable -- each caller decides what that
    means for it.
    """
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.common.w4a8_linear import W4A8Linear

    return (Int8Linear, Fp8Linear, W4A8Linear)


def count_quantized_linears(module: Optional[nn.Module]) -> int:
    """Number of weight-only quantized Linear modules under ``module``.

    These hold ``weight`` and its scale sidecars as BUFFERS, not
    ``nn.Parameter``s, so they are invisible to both ``requires_grad_(True)``
    and ``named_parameters()``. First half of ``reject_quantized_base``
    (``core.training.adapters.base_adapter``); a cheap scan that returns 0 for
    an architecture whose loader never produces these classes.
    """
    if module is None:
        return 0
    try:
        quantized = _quantized_linear_classes()
    except Exception as e:
        print(f"[quantized-base-guard] weight-only quant classes unavailable "
              f"({e}); assuming an unquantized base")
        return 0
    return sum(1 for m in module.modules() if isinstance(m, quantized))


def is_lora_wrappable_linear(module: Optional[nn.Module]) -> bool:
    """True for a module a LoRA can wrap: ``nn.Linear`` or a weight-only
    quantized Linear (``Int8Linear`` / ``Fp8Linear`` / ``W4A8Linear``).

    The quantized classes are ``nn.Module``s, NOT ``nn.Linear`` subclasses, so
    an ``isinstance(x, nn.Linear)`` target scan drops every quantized layer
    SILENTLY -- measured on Anima, where it dropped 75% of the intended targets.

    Deliberately excludes ``LoRALinearLayer``: callers use this to decide
    whether to WRAP, and an already-wrapped module must not be wrapped twice.
    """
    if module is None:
        return False
    if isinstance(module, nn.Linear):
        return True
    try:
        quantized = _quantized_linear_classes()
    except Exception:
        return False
    return isinstance(module, quantized)


def lora_branch_dtype(module: nn.Module,
                      default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """The dtype a LoRA branch attached to ``module`` should compute in.

    The base weight's own dtype when that is a real float, else ``default`` --
    which is also what a quantized base's own forward produces from a bf16
    activation. Copying the branch at ``base.weight.dtype`` instead would
    quantize the adapter to int8's 8 uniform levels, or to e4m3 and lose most
    of its precision.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        return default
    dtype = weight.dtype
    if dtype.is_floating_point and "float8" not in str(dtype):
        return dtype
    return default


def quantization_kind(module: nn.Module) -> Optional[str]:
    """``"int8"`` / ``"fp8"`` / ``"w4a8"`` for a weight-only quantized Linear,
    else ``None``.

    The kind, not the count ``count_quantized_linears`` gives: the capability
    gates differ per kind.
    """
    try:
        quantized = _quantized_linear_classes()
    except Exception:
        return None
    for cls in quantized:
        if isinstance(module, cls):
            return _QUANTIZED_KIND_BY_CLASS[cls.__name__]
    return None


@dataclass(frozen=True)
class AdapterTarget:
    """One module an adapter may cover, with everything the engine needs to
    cover it: its address, its geometry, its dtype, and what its base can do.

    ``parent`` + ``slot`` is the ADDRESS, not a convenience: installing a
    wrapper is ``set_module_slot(parent, slot, wrapper)`` and restoring is the
    same call with ``module``, and an integer ``slot`` (an ``nn.Sequential``
    index) has no attribute name to fall back on. See
    ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` section 3.
    """

    module_path: str
    parent: nn.Module
    slot: Slot
    module: nn.Module
    component: Optional[str] = None
    scope: Optional[str] = None
    block: Optional[str] = None
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    base_dtype: Optional[torch.dtype] = None
    branch_dtype: torch.dtype = torch.bfloat16
    quantization: Optional[str] = None
    mergeable: bool = False

    @classmethod
    def describe(
        cls,
        module_path: str,
        parent: nn.Module,
        slot: Slot,
        module: nn.Module,
        *,
        component: Optional[str] = None,
        scope: Optional[str] = None,
        block: Optional[str] = None,
        branch_dtype_default: torch.dtype = torch.bfloat16,
    ) -> "AdapterTarget":
        """Fill the derived fields from ``module`` itself."""
        weight = getattr(module, "weight", None)
        base_dtype = None if weight is None else weight.dtype
        quantization = quantization_kind(module)
        return cls(
            module_path=module_path,
            parent=parent,
            slot=slot,
            module=module,
            component=component,
            scope=scope,
            block=block,
            in_features=getattr(module, "in_features", None),
            out_features=getattr(module, "out_features", None),
            base_dtype=base_dtype,
            branch_dtype=lora_branch_dtype(module, default=branch_dtype_default),
            quantization=quantization,
            # A merge writes the delta into the base weight, so it needs a real
            # float weight to write into: a weight-only quantized base would
            # have to be dequantized and requantized first.
            mergeable=(quantization is None and base_dtype is not None
                       and base_dtype.is_floating_point
                       and "float8" not in str(base_dtype)),
        )


def enumerate_adapter_targets(
    root: nn.Module,
    *,
    predicate: Callable[[nn.Module], bool] = is_lora_wrappable_linear,
    component: Optional[str] = None,
    scope_of: Optional[Callable[[str], Optional[str]]] = None,
    block_of: Optional[Callable[[str], Optional[str]]] = None,
    branch_dtype_default: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> Iterator[AdapterTarget]:
    """Yield an ``AdapterTarget`` per module under ``root`` that ``predicate``
    accepts, carrying the parent/slot pair ``set_module_slot`` needs.

    Does not descend into an adapter-covered slot, for the reason
    ``named_modules_outside_adapters`` does not: a wrapper's own branch Linears
    and its hidden base would otherwise be offered as fresh targets.
    """
    for name, child in root.named_children():
        path = f"{prefix}.{name}" if prefix else name
        slot: Slot = int(name) if isinstance(root, (nn.Sequential, nn.ModuleList)) else name
        if is_adapter_covered(child):
            continue
        if predicate(child):
            yield AdapterTarget.describe(
                path, root, slot, child,
                component=component,
                scope=scope_of(path) if scope_of else None,
                block=block_of(path) if block_of else None,
                branch_dtype_default=branch_dtype_default,
            )
            continue
        yield from enumerate_adapter_targets(
            child,
            predicate=predicate,
            component=component,
            scope_of=scope_of,
            block_of=block_of,
            branch_dtype_default=branch_dtype_default,
            prefix=path,
        )
