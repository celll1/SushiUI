"""Pinned-CPU staging shared by SenseNova MoT generation and training eviction.

Both evictors move the phase-inactive MoT half to pinned CPU RAM with the same
rules, so the staging loop lives here once. Host RAM: pinned CPU tensors are
never explicitly un-pinned -- torch's caching host allocator pools freed pinned
blocks rather than returning them to the OS, so a later
``stage_modules_to_pinned_cpu`` call reuses the same pool for free, and a
tensor that is already pinned is returned untouched (zero copies).

PAGEABLE ESCAPE HATCH. ``pageable=True`` (``sensenova_mot_pageable_staging``)
never attempts a pinned allocation, trading transfer speed for host memory the
OS can reclaim -- unlike a pinned block, which this module's caching allocator
never returns once allocated. Distinct from the failure-path fallback below
(``tensor.to("cpu")`` after a caught pin exception): that is an unplanned,
warned-once degradation; ``pageable=True`` is a deliberate, silent, every-call
choice with no pin attempt to fail in the first place. It is refused together
with ``non_blocking=True`` (the training evictor's overlap path): an async copy
out of pageable host memory goes through a driver bounce buffer and is
effectively host-synchronous, so the two do not compose.
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
    tensor: torch.Tensor,
    warn_once: Dict[str, bool],
    warn_message: str,
    *,
    pageable: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Return ``tensor``'s contents on CPU memory using ONE host copy: the
    pinned destination is allocated first and written directly, instead of
    ``.to("cpu")`` (copy 1, pageable) followed by ``.pin_memory()`` (copy 2).
    ``copy_`` is blocking unless ``non_blocking=True``, which only the training
    evictor's overlap path passes and which requires the caller to keep the
    CUDA source alive (or ``record_stream`` it) until the copy has landed.

    ``pageable=True`` skips the pinned allocation outright (see this module's
    PAGEABLE ESCAPE HATCH note) rather than attempting one and falling back:
    an already-CPU tensor short-circuits regardless of its pin state, so a
    pageable-staged tensor is never re-pinned on a later call, and a tensor
    that arrived already pinned from outside this module is left pinned (this
    call does not retroactively un-pin it -- every tensor this evictor stages
    is freshly materialized from the checkpoint loader, never pre-pinned, so
    that case does not arise on the route that sets this flag)."""
    if non_blocking and pageable:
        raise ValueError(
            "MoT staging cannot combine pageable=True with non_blocking=True: a "
            "cudaMemcpyAsync out of pageable host memory is staged through a "
            "driver bounce buffer and is effectively host-synchronous"
        )
    tensor = tensor.detach()
    if pageable:
        if tensor.device.type == "cpu":
            return tensor
        return tensor.to("cpu")
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
    pinned.copy_(tensor, non_blocking=non_blocking)
    return pinned


def stage_modules_to_pinned_cpu(
    modules: Iterable[nn.Module],
    *,
    warn_once: Dict[str, bool],
    warn_message: str = DEFAULT_PIN_FAILURE_MESSAGE,
    pageable: bool = False,
    non_blocking: bool = False,
    sources: list | None = None,
) -> None:
    """Move every parameter/PERSISTENT buffer OWNED (recurse=False) by each of
    ``modules`` to a CPU tensor, pinned unless ``pageable=True`` (see this
    module's PAGEABLE ESCAPE HATCH note). Touches ``_parameters``/``_buffers``
    directly (not ``.to()``) because pinning requires an explicit pinned
    allocation, which ``nn.Module.to()`` has no option for. Non-persistent
    buffers (e.g. a rotary embedding's cached ``inv_freq``) are skipped even if
    a caller passes such a module in -- they are derived tensors, not weights,
    and moving them is never the point of this call. Parameter objects are
    updated in place (``.data``), never replaced.

    ``warn_once`` is shared by the caller across all of its modules so a pinned
    allocation failure prints exactly once per evictor. Unused when
    ``pageable=True``, which has no pin attempt to fail.

    ``non_blocking=True`` leaves the copies in flight on the current stream; the
    device sources are appended to ``sources``, which is what keeps them alive
    once the reassignment below drops the MODEL's reference, so the caller can
    still ``record_stream`` a block the allocator would otherwise be free to hand
    to a concurrent copy."""
    def _stage(tensor):
        staged = _stage_tensor(
            tensor, warn_once, warn_message, pageable=pageable,
            non_blocking=non_blocking,
        )
        if sources is not None and tensor.device.type != "cpu":
            sources.append(tensor)
        return staged

    for module in modules:
        for key, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            parameter.data = _stage(parameter.data)
        for key, buffer in list(module._buffers.items()):
            if buffer is None or key in module._non_persistent_buffers_set:
                continue
            module._buffers[key] = _stage(buffer)
