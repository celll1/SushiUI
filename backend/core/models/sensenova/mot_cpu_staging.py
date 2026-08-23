"""Pinned-CPU staging shared by SenseNova MoT generation and training eviction.

Both evictors move the phase-inactive MoT half to pinned CPU RAM with the same
rules, so the staging loop lives here once. Host RAM: pinned CPU tensors are
never explicitly un-pinned -- torch's caching host allocator pools freed pinned
blocks rather than returning them to the OS, so a later
``stage_modules_to_pinned_cpu`` call reuses the same pool for free, and a
tensor that is already pinned is returned untouched (zero copies).
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

DEFAULT_PIN_FAILURE_MESSAGE = (
    "[SenseNova] MoT phase eviction: pin_memory() failed ({exc}); "
    "continuing with unpinned CPU staging (slower transfer, same result)."
)


def _stage_tensor(
    tensor: torch.Tensor, warn_once: Dict[str, bool], warn_message: str
) -> torch.Tensor:
    """Return ``tensor``'s contents on pinned CPU memory using ONE host copy:
    the pinned destination is allocated first and written directly, instead of
    ``.to("cpu")`` (copy 1, pageable) followed by ``.pin_memory()`` (copy 2).
    ``copy_`` is blocking, which the denoise-phase ordering relies on."""
    tensor = tensor.detach()
    if tensor.device.type == "cpu" and tensor.is_pinned():
        return tensor
    try:
        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
    except Exception as exc:
        # Reached without CUDA, and when the int8 weight buffers (Int8Linear)
        # exhaust pinned host memory.
        if "pin_failed" not in warn_once:
            warn_once["pin_failed"] = True
            print(warn_message.format(exc=exc))
        return tensor.to("cpu")
    pinned.copy_(tensor)
    return pinned


def stage_modules_to_pinned_cpu(
    modules: Iterable[nn.Module],
    *,
    warn_once: Dict[str, bool],
    warn_message: str = DEFAULT_PIN_FAILURE_MESSAGE,
) -> None:
    """Move every parameter/PERSISTENT buffer OWNED (recurse=False) by each of
    ``modules`` to a pinned CPU tensor. Touches ``_parameters``/``_buffers``
    directly (not ``.to()``) because pinning requires an explicit pinned
    allocation, which ``nn.Module.to()`` has no option for. Non-persistent
    buffers (e.g. a rotary embedding's cached ``inv_freq``) are skipped even if
    a caller passes such a module in -- they are derived tensors, not weights,
    and moving them is never the point of this call. Parameter objects are
    updated in place (``.data``), never replaced.

    ``warn_once`` is shared by the caller across all of its modules so a pinned
    allocation failure prints exactly once per evictor."""
    for module in modules:
        for key, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            parameter.data = _stage_tensor(parameter.data, warn_once, warn_message)
        for key, buffer in list(module._buffers.items()):
            if buffer is None or key in module._non_persistent_buffers_set:
                continue
            module._buffers[key] = _stage_tensor(buffer, warn_once, warn_message)
