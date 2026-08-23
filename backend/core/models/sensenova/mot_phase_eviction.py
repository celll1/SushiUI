"""MoT phase-exclusive half-weight eviction for SenseNova U1.5.

Every one of the 42 Qwen3-MoT decoder layers carries TWO branch halves: the
"understanding" weights (plain names) and their ``_mot_gen``-suffixed twins.
The prefix phase passes ``image_gen_indicators=None`` (every layer takes
``forward_und`` exclusively -- the gen half is dead weight); the denoise
phase passes an all-ones ``image_gen_indicators`` (every layer takes
``forward_gen`` exclusively -- the understanding half is dead weight).
``self_attn.rotary_emb`` / ``rotary_emb_hw`` are the exception: un-suffixed,
shared, called from ``forward_gen`` too (``vendor/modeling_qwen3.py``) --
they must stay GPU-resident always. Measured (safetensors header read):
386,221,056 bytes/layer, exactly 50/50, x42 layers = 15.11 GiB of layer
weights, 7.55 GiB/half (see MODEL_FACTS.md's sensenova entry).

Also outside the swapped set (never touched, always GPU-resident):
``language_model.model.norm_mot_gen`` and ``vision_model_mot_gen``.

This module registers a callback on ``NEOChatModel._layer_offload_phase_callback``.
Callers place only the non-generation half on GPU up front
(``move_non_gen_to_device``), so a generation performs two half-transfers:

  * "prefix"  notification -> gen half pinned in place on CPU (already there).
  * "denoise" notification -> understanding half GPU -> pinned CPU FIRST
    (blocking), THEN gen half pinned CPU -> GPU. Order matters: reversing it
    would co-reside both halves on GPU for one window, which is exactly the
    peak this feature exists to avoid.

Target is pinned CPU RAM, not lazy disk loading -- both halves are always
needed within one generation. Weights are only ever MOVED, never modified.

Host RAM: pinned CPU tensors are never explicitly un-pinned/freed after a
generation (see ``teardown()``) -- torch's caching host allocator pools freed
pinned blocks rather than returning them to the OS, so an explicit "unpin"
would only add a pageable clone on top of the still-reserved pool (measured:
a net increase, not a release). Leaving tensors pinned lets the next
generation's ``_pin_module_cpu_`` reuse the same pool for free.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

LABEL = "SenseNova"


def _module_nbytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=False):
        total += p.numel() * p.element_size()
    for key, b in module._buffers.items():
        if b is None or key in module._non_persistent_buffers_set:
            continue
        total += b.numel() * b.element_size()
    return total


def _pin_module_cpu_(module: nn.Module, *, warn_once: Dict[str, bool]) -> None:
    """Move every parameter/PERSISTENT buffer OWNED (recurse=False) by
    ``module`` to a pinned CPU tensor. Touches ``_parameters``/``_buffers``
    directly (not ``.to()``) because pinning requires an explicit
    ``.pin_memory()`` copy, which ``nn.Module.to()`` has no option for.
    Non-persistent buffers (e.g. a rotary embedding's cached ``inv_freq``) are
    skipped even if a caller passes such a module in -- they are derived
    tensors, not weights, and moving them is never the point of this call."""
    def _warn_pin_failed(exc: Exception) -> None:
        if "pin_failed" not in warn_once:
            warn_once["pin_failed"] = True
            print(f"[{LABEL}] MoT phase eviction: pin_memory() failed ({exc}); "
                  f"continuing with unpinned CPU staging (slower transfer, same result).")

    for key, p in list(module._parameters.items()):
        if p is None:
            continue
        cpu = p.data.detach().to("cpu")
        if not cpu.is_pinned():
            try:
                cpu = cpu.pin_memory()
            except Exception as exc:
                _warn_pin_failed(exc)
        module._parameters[key].data = cpu
    for key, b in list(module._buffers.items()):
        if b is None or key in module._non_persistent_buffers_set:
            continue
        cpu = b.detach().to("cpu")
        if not cpu.is_pinned():
            try:
                cpu = cpu.pin_memory()
            except Exception as exc:
                # The int8 weight buffers (Int8Linear) are the tensors large
                # enough to actually exhaust pinned host memory -- this branch
                # must warn too, not just the parameter loop above.
                _warn_pin_failed(exc)
        module._buffers[key] = cpu


class MotPhaseEvictor:
    """One instance per generation. Built with the transformer's CURRENT
    module tree (after LoRA is applied, if any -- LoRA wraps the gen-branch
    Linears in place via ``setattr``, so building the gen/understanding split
    beforehand would collect stale pre-wrap module references)."""

    def __init__(self, transformer: nn.Module, device: Any):
        self.device = device
        self.transformer = transformer
        from .mot_weight_selector import select_mot_weight_modules

        layers = list(transformer.language_model.model.layers)
        selection = select_mot_weight_modules(transformer)
        self._gen_modules = list(selection.gen_modules)
        self._und_modules = list(selection.und_modules)
        self._warn_once: Dict[str, bool] = {}
        self._phase: Optional[str] = None

        gen_bytes = sum(_module_nbytes(m) for m in self._gen_modules)
        und_bytes = sum(_module_nbytes(m) for m in self._und_modules)
        self.gen_bytes = gen_bytes
        self.und_bytes = und_bytes
        # Host RAM: pinned tensors are pooled by torch's caching host allocator and
        # reused across generations, never returned to the OS.
        print(f"[{LABEL}] MoT phase eviction ENABLED: {len(self._gen_modules)} generation-branch "
              f"module(s) across {len(layers)} layers, {gen_bytes / 1024 ** 3:.2f} GiB.")
        self._sanity_check_selection(len(layers))

    def _sanity_check_selection(self, num_layers: int) -> None:
        """Self-check against the two prior builds that shipped a classifier
        silently selecting almost nothing (~0.21 GiB, buffer-only rule --
        see the NOTE in ``__init__``) with no code-level signal; only a full
        GPU measurement gate caught it, twice. Thresholds: each half should be
        ~7.55 GiB (42 layers x ~0.18 GiB); flag if either half is under 1 GiB
        (an order of magnitude below expected) or the halves differ by more
        than 2x (they are structurally symmetric and should be near-equal)."""
        min_bytes = 1 * 1024 ** 3
        broken = self.gen_bytes < min_bytes or self.und_bytes < min_bytes
        if not broken and self.gen_bytes > 0 and self.und_bytes > 0:
            ratio = max(self.gen_bytes, self.und_bytes) / min(self.gen_bytes, self.und_bytes)
            broken = ratio > 2.0
        if broken:
            msg = (f"[{LABEL}] MoT phase eviction: selected {self.gen_bytes / 1024 ** 3:.2f} GiB "
                   f"generation-branch / {self.und_bytes / 1024 ** 3:.2f} GiB understanding-branch "
                   f"across {num_layers} layers -- expected ~7.55 GiB each. The classifier likely "
                   f"failed to select the weights; this feature is probably inert (no VRAM savings).")
            print(msg)
            try:
                from api.generation_status import add_warning
                add_warning(msg, code="sensenova_mot_phase_eviction_selection_suspect")
            except Exception:
                pass

    def move_non_gen_to_device(self) -> None:
        """Initial placement for everything OUTSIDE the generation branch,
        used INSTEAD OF a blanket ``transformer.to(device)``: a whole-model
        move followed by eviction still marks both halves GPU-resident on
        ``max_memory_allocated()``'s high-water mark, however brief the
        overlap.

        Own params/buffers only -- ``.to()`` on a decoder layer container
        would recurse into its gen-branch children, which are classified as
        leaves."""
        gen_ids = {id(m) for m in self._gen_modules}
        device = self.device
        for module in self.transformer.modules():
            if id(module) in gen_ids:
                continue
            for key, p in list(module._parameters.items()):
                if p is not None:
                    module._parameters[key].data = p.data.to(device)
            for key, b in list(module._buffers.items()):
                if b is not None:
                    module._buffers[key] = b.to(device)

    def on_phase(self, phase: str) -> None:
        """Bound to ``transformer._layer_offload_phase_callback`` -- called
        from ``_notify_layer_offload_phase`` at exactly 2 points per
        generation (see this module's docstring)."""
        if phase == self._phase:
            return  # idempotent: a repeat notification of the same phase is a no-op.
        if phase == "prefix":
            for m in self._gen_modules:
                _pin_module_cpu_(m, warn_once=self._warn_once)
        elif phase == "denoise":
            # Evict the understanding half FIRST -- `.to("cpu")` is a blocking
            # copy, so this loop fully completes (freeing the GPU blocks)
            # before the gen half's non_blocking H2D load below is enqueued.
            # Reversing this order co-resides both halves on GPU for one
            # window, defeating the peak-VRAM reduction this feature exists for.
            for m in self._und_modules:
                _pin_module_cpu_(m, warn_once=self._warn_once)
            for m in self._gen_modules:
                m.to(self.device, non_blocking=True)
        self._phase = phase

    def teardown(self) -> None:
        """End-of-generation hook. Call AFTER the caller's own full-model
        ``.to("cpu")`` (which already moves whatever is on GPU back,
        regardless of how far the phase sequence got, including on an
        exception/cancellation). Deliberately does NOT un-pin: torch's caching
        host allocator does not return pinned blocks to the OS on free, so an
        explicit unpin only adds a pageable clone on top of the still-reserved
        pool (measured net increase, not a release) -- see this module's
        docstring. Left pinned, the next generation's pin is a no-op reuse."""
        return


def install(transformer: nn.Module, device: Any) -> Optional[MotPhaseEvictor]:
    """Build a ``MotPhaseEvictor`` for this generation and register it as
    ``transformer._layer_offload_phase_callback``. Returns None (feature
    silently inert, warned once) without CUDA -- there is nothing to evict
    when the model never leaves the CPU."""
    if not torch.cuda.is_available():
        try:
            from api.generation_status import add_warning
            add_warning(
                f"[{LABEL}] sensenova_mot_phase_eviction requested but CUDA is unavailable; "
                f"the mechanism is a GPU<->pinned-CPU swap, so it has nothing to do here.",
                code="sensenova_mot_phase_eviction_no_cuda",
            )
        except Exception:
            pass
        return None
    try:
        evictor = MotPhaseEvictor(transformer, device)
    except Exception as exc:
        # Never take a generation down over an optional VRAM-saving feature --
        # fall back to the eviction-off path (both halves stay GPU-resident).
        print(f"[{LABEL}] MoT phase eviction: failed to install ({exc}); continuing without it.")
        return None
    transformer._layer_offload_phase_callback = evictor.on_phase
    # stdout is unobservable to the HTTP test path; surface engagement + the
    # measured GiB figures in the response so a gate can assert on them directly.
    try:
        from api.generation_status import add_warning
        add_warning(
            f"[{LABEL}] MoT phase eviction active: {len(evictor._gen_modules)} generation-branch "
            f"module(s) ({evictor.gen_bytes / 1024 ** 3:.2f} GiB) / "
            f"{len(evictor._und_modules)} understanding-branch module(s) "
            f"({evictor.und_bytes / 1024 ** 3:.2f} GiB) staged to pinned CPU per phase.",
            code="sensenova_mot_phase_eviction_active",
        )
    except Exception:
        pass
    return evictor


def uninstall(transformer: nn.Module, evictor: Optional[MotPhaseEvictor]) -> None:
    """Tear down and unregister. Safe to call even if ``install`` returned
    None (nothing to do) or was never called (idempotent attribute clear) --
    always call this from the generation's ``finally``, after the whole-model
    CPU restore, so a leftover callback never fires on a later generation
    that did not request eviction."""
    if evictor is not None:
        try:
            evictor.teardown()
        except Exception as exc:
            print(f"[{LABEL}] MoT phase eviction teardown raised (non-fatal): {exc}")
    if getattr(transformer, "_layer_offload_phase_callback", None) is not None:
        transformer._layer_offload_phase_callback = None
