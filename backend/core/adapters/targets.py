"""Target-side policy for adapter injection: what can be wrapped, in what dtype,
and how much of the base is weight-only quantized.

The quantized-Linear classes are imported LAZILY inside each function. They live
under ``core.models.*``, and hoisting them to module scope would give this
package an import-time edge into the model loaders -- the same shape of back-edge
that put this package outside ``core.training`` in the first place.

Moved verbatim from ``core.training.adapters.base_adapter``, which now imports
``count_quantized_linears`` from here like every other caller.
"""

from typing import Optional

import torch
import torch.nn as nn


def count_quantized_linears(module: Optional[nn.Module]) -> int:
    """Number of weight-only quantized Linear modules under ``module``.

    Called from EVERY full-parameter adapter's ``prepare_models_for_training``
    / ``setup_trainable_parameters``, not just the three architectures whose
    loaders can currently produce these classes (Anima, Ideogram 4, Krea 2).
    Quantized Linears hold ``weight`` and scale sidecars as
    buffers, not ``nn.Parameter``s, so they are invisible to both
    ``requires_grad_(True)`` and ``named_parameters()``. Detecting them is
    the first half of ``reject_quantized_base``
    (``core.training.adapters.base_adapter``).

    For an architecture whose loader never swaps in these classes, this is a
    guaranteed no-op (returns 0, ``reject_quantized_base`` returns without
    raising) -- it costs one cheap module scan and exists so the same silent
    failure cannot reappear unnoticed if that architecture later gains a
    weight-only quantized load path, the way Anima/Krea2/Ideogram4 already
    have.
    """
    if module is None:
        return 0
    try:
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.common.w4a8_linear import W4A8Linear
    except Exception as e:
        print(f"[quantized-base-guard] weight-only quant classes unavailable "
              f"({e}); assuming an unquantized base")
        return 0
    return sum(1 for m in module.modules() if isinstance(m, (Int8Linear, Fp8Linear, W4A8Linear)))


def is_lora_wrappable_linear(module: Optional[nn.Module]) -> bool:
    """True for a module a LoRA can wrap: ``nn.Linear`` or EITHER weight-only
    quantized Linear (``Int8Linear`` / ``Fp8Linear`` / ``W4A8Linear``).

    THE reason this exists: the quantized Linear classes are ``nn.Module``s,
    NOT ``nn.Linear`` subclasses. Every ``isinstance(x, nn.Linear)`` site that
    selects LoRA targets therefore skips every quantized layer SILENTLY -- no
    error, just a smaller ``applied`` count that looks like a LoRA which happens
    to touch fewer modules. Measured on Anima, where the naive predicate dropped
    75% of the intended targets.

    Deliberately does NOT include ``LoRALinearLayer``: the call sites this
    replaces use the predicate to decide whether to WRAP, and an already-wrapped
    module must not be wrapped twice. A caller that wants "wrappable or already
    wrapped" (re-application, target enumeration) tests for that class itself --
    ``core.models.krea2.krea2_lora._is_target`` is the example.
    """
    if module is None:
        return False
    if isinstance(module, nn.Linear):
        return True
    try:
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
        from core.models.common.w4a8_linear import W4A8Linear
    except Exception:
        return False
    return isinstance(module, (Int8Linear, Fp8Linear, W4A8Linear))


def lora_branch_dtype(module: nn.Module,
                      default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """The dtype a LoRA branch attached to ``module`` should compute in.

    The base weight's own dtype when that is a real float, else ``default``.
    Weight-only quantized bases take the default branch -- e4m3 by the "float8" test, int8
    because an integer dtype is not floating point at all -- which is also the
    dtype their own forward produces from a bf16 activation. Without this a
    caller that copies the LoRA weights with ``dtype=base.weight.dtype`` would
    cast them to int8 and quantize the adapter to 8 uniform levels, or to e4m3
    and lose most of its precision.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        return default
    dtype = weight.dtype
    if dtype.is_floating_point and "float8" not in str(dtype):
        return dtype
    return default
