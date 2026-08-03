"""Refuse a weight-only quantized checkpoint at a loader that cannot read one.

WHY THIS EXISTS
---------------
Several DiT loaders call ``load_state_dict(..., strict=False)`` because their
checkpoints legitimately carry extra or missing sections (embedded VAE/TE
blocks, tied weights, a base module that is then overridden). That tolerance is
correct for those cases and catastrophic for one specific input: hand such a
loader a checkpoint produced by ``subapps/fp8_quantize/quantize_transformer_fp8.py``
or by ``POST /models/export-quantized`` and

  * every ``.weight_scale`` lands in ``unexpected_keys`` (the module has no such
    buffer), and
  * every quantized ``.weight`` is an int8 / float8 tensor being copied into a
    bf16 parameter -- which ``load_state_dict`` performs as a DTYPE CAST, so the
    int8 CODES (-127..127) or the e4m3 values are written as if they were the
    real weights, without their per-row scale.

The load then "succeeds", one warning line scrolls past, and a silently wrong
model generates noise. The archs that DO support these files
(``int8_runtime_quantize.QUANTIZED_LINEAR_ARCHS``) detect the layout first and
swap in ``Int8Linear`` / ``Fp8Linear`` before loading; every other arch must
fail loudly instead.

This is deliberately a check on the STATE DICT, not on the file name or its
metadata: the offline tool's output is named by the user, and a shard index
carries no format flag the reader must trust.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = [
    "QUANT_SCALE_SUFFIX",
    "quantized_state_dict_report",
    "refuse_quantized_state_dict",
]

# Both weight-only formats in this repo use the same scale suffix; only the
# weight dtype tells them apart (see ``int8_linear.INT8_SCALE_SUFFIX`` and
# ``fp8_linear.FP8_SCALE_SUFFIX``, which are both ".weight_scale").
QUANT_SCALE_SUFFIX = ".weight_scale"

_QUANT_WEIGHT_DTYPES = tuple(
    d for d in (
        torch.int8,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
        getattr(torch, "uint8", None),
    ) if d is not None
)


def quantized_state_dict_report(state_dict: Dict[str, "torch.Tensor"]) -> Optional[Dict[str, object]]:
    """``None`` for an ordinary checkpoint; a report dict for a quantized one.

    Two independent pieces of evidence, either of which is sufficient:

    * a ``.weight_scale`` key -- the sibling that gates both swap helpers;
    * a ``.weight`` whose dtype is int8 / float8 / uint8. Kept as a separate
      test so a file whose scales were stripped, or a future layout that names
      them differently, is still caught. Restricted to ``.weight`` keys so an
      ordinary integer BUFFER (a mask, an index table) cannot trip it.

    The report carries ``scale_keys`` / ``quantized_weight_keys`` counts and a
    few example key names, for the message.
    """
    if not state_dict:
        return None
    scale_keys: List[str] = []
    quant_weights: List[str] = []
    for key, value in state_dict.items():
        if key.endswith(QUANT_SCALE_SUFFIX):
            scale_keys.append(key)
            continue
        if key.endswith(".weight") and getattr(value, "dtype", None) in _QUANT_WEIGHT_DTYPES:
            quant_weights.append(key)
    if not scale_keys and not quant_weights:
        return None
    return {
        "scale_keys": len(scale_keys),
        "quantized_weight_keys": len(quant_weights),
        "examples": (scale_keys[:3] + quant_weights[:3])[:5],
    }


def refuse_quantized_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    *,
    arch: str,
    path: Optional[str] = None,
    label: str = "transformer",
) -> None:
    """Raise ``RuntimeError`` when ``state_dict`` is weight-only quantized.

    Call it immediately before a ``strict=False`` load on an architecture that
    has no quantized-Linear swap. Silent on every ordinary checkpoint, so it is
    safe to call unconditionally.
    """
    report = quantized_state_dict_report(state_dict)
    if report is None:
        return
    try:
        from core.models.common.int8_runtime_quantize import (
            QUANTIZED_LINEAR_ARCHS, arch_names,
        )
        supported = arch_names(QUANTIZED_LINEAR_ARCHS)
    except Exception:  # pragma: no cover - defensive
        supported = "the architectures whose loaders swap in quantized Linear layers"
    where = f" ({path})" if path else ""
    raise RuntimeError(
        f"the {arch} {label} checkpoint{where} is weight-only QUANTIZED "
        f"({report['scale_keys']} '{QUANT_SCALE_SUFFIX}' key(s), "
        f"{report['quantized_weight_keys']} int8/float8 weight(s); e.g. "
        f"{', '.join(report['examples'])}), and the {arch} loader does not support "
        f"quantized checkpoints. Loading it would drop every scale as an unexpected "
        f"key and cast the quantized codes into bf16 parameters, producing a "
        f"silently wrong model. Weight-only quantized checkpoints are readable only "
        f"on {supported}; load an unquantized {arch} checkpoint instead."
    )
