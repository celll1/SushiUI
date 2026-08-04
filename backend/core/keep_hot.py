"""Keep-models-hot: opt-in queue optimization for SD1.5/SDXL (and future arch)
generation.

When a caller runs several generations back-to-back on the SAME loaded model
(a queue), the reference pipeline normally re-stages every component between
generations: text encoder(s) -> GPU -> encode -> CPU, U-Net -> GPU -> denoise
-> CPU, VAE -> GPU -> decode -> CPU. If ``keep_models_hot`` is requested, this
module lets the caller SKIP the CPU offload at the end of a successful
generation (and the corresponding GPU stage at the start of the next one) for
components it is safe to keep resident, bounded by a VRAM guard.

This module is arch-agnostic scaffolding: SD1.5/SDXL (``core/pipeline.py``)
wire it in for Phase A; DiT archs (Phase B) reuse the exact same functions —
see the call-site contract documented on each function below.

Safety invariants (do not relax without re-reading the design doc):
  - A component is NEVER kept resident under block-swap streaming (the whole
    point of block swap is to NOT be fully GPU-resident) or when it runs on
    CPU by request (``cpu_text_encoding``): there is nothing to "keep".
  - The resident set is only valid for an exact ``model_key`` (loaded
    checkpoint + LoRA set + unet_quantization + weight dtype). Any mismatch
    invalidates ALL residents immediately (forces a normal offload), because
    reusing a component from a different LoRA/quantization/checkpoint context
    is a correctness bug, not just a missed optimization.
  - On an EXCEPTION, the caller must ALWAYS force a full offload + clear the
    resident state — the "keep hot" fast path is trust-based (skips checks
    the normal path implicitly re-does every time), so an error means we no
    longer trust the pipeline state going into the next generation.
"""

from typing import Any, Callable, Dict, Optional, Set

import torch

# Fixed VRAM headroom subtracted from the CUDA free-memory reading before the
# guard is evaluated. Conservative on purpose: allocator fragmentation and
# transient allocations (activations, decode buffers) between the guard check
# and actual use are not accounted for individually, so a flat safety margin
# protects against the guard passing right at the edge of available memory.
KEEP_HOT_VRAM_HEADROOM_BYTES = int(1.5 * 1024 ** 3)  # 1.5 GB


def _add_generation_warning(message: str, code: Optional[str] = None) -> None:
    """Best-effort: record a feature-degradation warning for the current
    generation. Lazily imported so this module never hard-depends on the api
    package at import time (mirrors vram_optimization._add_generation_warning).
    """
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


def _ensure_state(manager) -> Dict[str, Any]:
    """Lazily initialize the manager-side keep-hot state.

    ``manager`` is the DiffusionPipelineManager instance (or any object the
    caller uses to persist state across generations — DiT arch mixins share
    the same ``self``).
    """
    state = getattr(manager, "_keep_hot", None)
    if state is None:
        state = {"model_key": None, "resident": set()}
        manager._keep_hot = state
    return state


def keep_hot_requested(params: Dict[str, Any]) -> bool:
    """Whether the current request opts into keep-models-hot."""
    return bool(params.get("keep_models_hot", False))


def _lora_fingerprint(lora_configs) -> tuple:
    """Sorted (path, weight) fingerprint of the applied LoRA set.

    Order-independent (a queue that reorders the same LoRA list should not
    invalidate residency) but exact on path + weight (any change is a
    correctness-relevant difference for a hot denoiser — see module docstring).
    """
    if not lora_configs:
        return ()
    items = []
    for cfg in lora_configs:
        path = cfg.get("path") or cfg.get("name") or cfg.get("lora_path")
        weight = cfg.get("weight", cfg.get("strength", 1.0))
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = str(weight)
        items.append((str(path), weight))
    return tuple(sorted(items, key=lambda t: t[0]))


def _quantization_fingerprint(manager, params: Dict[str, Any]) -> str:
    """The ``unet_quantization`` component of the model key.

    Normalised to ``"int8"`` once the loaded transformer has been converted in
    place (``vram_optimization.apply_runtime_int8_quantization``). That
    conversion is ONE-WAY until the model is reloaded, so the raw request value
    stops describing the resident components: a follow-up generation that omits
    the parameter would otherwise compute a different key and evict a component
    set that is in fact still exactly what was staged.

    A conversion that FAILED part-way gets its own value: the module is neither
    the checkpoint's nor fully int8, and it stops being "int8_partial" the moment
    a later request converts the remainder -- which is exactly when the resident
    set must be re-keyed.

    ``_runtime_int8_from_checkpoint`` (the transformer arrived quantized, so an
    int8 request was a no-op) normalises to the SAME value. The two latches
    differ only in provenance -- which decides whether the user is told the
    conversion is one-way -- and provenance is not a property of the resident
    component set. Keying them apart would evict a component set that is in fact
    still exactly what was staged.
    """
    if getattr(manager, "_runtime_int8_converted", False) \
            or getattr(manager, "_runtime_int8_from_checkpoint", False):
        return "int8"
    if getattr(manager, "_runtime_int8_partial", False):
        return "int8_partial"
    return str(params.get("unet_quantization"))


def compute_model_key(manager, params: Dict[str, Any]) -> str:
    """Identity string the resident component set is valid for.

    MUST change whenever keeping a component resident across generations
    would be unsafe: a different checkpoint, a different LoRA set (path or
    weight), a different U-Net quantization, or a different weight dtype.
    """
    lora_configs = params.get("loras", []) or []
    key_parts = (
        str(getattr(manager, "current_model", None)),
        _lora_fingerprint(lora_configs),
        _quantization_fingerprint(manager, params),
        # A kept-hot TE is only valid for the exact TE placement/precision it was
        # staged under: a different text_encoder_quantization re-quantizes it, and
        # cpu_text_encoding moves it off-GPU entirely. Either change must
        # invalidate a resident TE (else it would be skip-staged stale).
        str(params.get("text_encoder_quantization")),
        bool(params.get("cpu_text_encoding", False)),
        str(params.get("weight_dtype", params.get("torch_dtype"))),
    )
    return repr(key_parts)


def is_resident(manager, component_name: str, model_key: str) -> bool:
    """True if ``component_name`` is currently GPU-resident and valid for
    ``model_key`` (i.e. the caller may SKIP the ->GPU staging call entirely).
    """
    state = _ensure_state(manager)
    return state["model_key"] == model_key and component_name in state["resident"]


def mark_resident(manager, component_name: str, model_key: str) -> None:
    """Record ``component_name`` as kept GPU-resident for ``model_key``.

    Call only after a SUCCESSFUL generation whose offload step was skipped
    for this component. If ``model_key`` differs from the currently tracked
    key, the resident set is reset first (a stale set must never carry over
    to a new model_key — mark_resident is not itself responsible for
    invalidation; callers must invoke ``invalidate_if_model_changed`` first).
    """
    state = _ensure_state(manager)
    if state["model_key"] != model_key:
        state["resident"] = set()
        state["model_key"] = model_key
    state["resident"].add(component_name)


def discard_resident(manager, component_name: str) -> None:
    """Drop a SINGLE component from the resident set (it was just offloaded to
    CPU while others may remain hot).

    Call this whenever a previously-resident component is offloaded on a
    successful generation because it is no longer eligible to stay hot
    (keep_models_hot turned off, the VRAM guard failed this time, or the
    component became block-swapped / CPU-inference) BUT the model_key is
    unchanged so ``clear_resident`` would wrongly drop the still-hot siblings.
    Keeps ``state["resident"]`` in sync with physical device placement, so
    ``is_resident`` never reports a component that is actually on CPU (which
    would make the next generation skip its ->GPU stage -> device mismatch).
    """
    state = _ensure_state(manager)
    state["resident"].discard(component_name)


def clear_resident(manager) -> None:
    """Drop all tracked residency (e.g. on exception, on keep_models_hot=False
    queue-end cleanup, or on model load/unload).
    """
    state = _ensure_state(manager)
    state["model_key"] = None
    state["resident"] = set()


def invalidate_if_model_changed(manager, params: Dict[str, Any], offload_fn: Callable[[], None]) -> None:
    """At the START of a generation: if there is a resident set from a
    PREVIOUS generation but it is no longer valid for the current request's
    model_key (checkpoint/LoRA/quantization/dtype changed), force a full
    offload via ``offload_fn`` (arch-provided: moves all previously-resident
    components to CPU) and clear the state.

    No-op if there is no resident state, or if the model_key is unchanged.
    """
    state = _ensure_state(manager)
    if state["model_key"] is None or not state["resident"]:
        return
    new_key = compute_model_key(manager, params)
    if new_key != state["model_key"]:
        offload_fn()
        clear_resident(manager)


def should_keep_resident(
    manager,
    component_name: str,
    params: Dict[str, Any],
    *,
    is_block_swapped: bool = False,
    is_cpu_inference: bool = False,
    component_bytes: int = 0,
    free_vram_bytes: Optional[int] = None,
) -> bool:
    """Whether ``component_name`` may be left GPU-resident at the end of THIS
    generation (i.e. the caller should SKIP the ->CPU offload call).

    Returns False (offload as normal) whenever keep-hot is not requested, the
    component runs under block-swap streaming, the component is a
    CPU-inference component (nothing GPU-resident to keep), or the VRAM guard
    fails. On a VRAM-guard failure, records a warnings[] entry (via
    ``api.generation_status.add_warning``) and falls back to normal offload —
    it never raises.
    """
    if not keep_hot_requested(params):
        return False
    if is_block_swapped:
        return False
    if is_cpu_inference:
        return False

    if free_vram_bytes is None:
        if not torch.cuda.is_available():
            _add_generation_warning(
                "keep_models_hot requested but CUDA is unavailable; falling back to normal offload.",
                code="keep_hot_no_cuda",
            )
            return False
        free_vram_bytes, _total = torch.cuda.mem_get_info()

    available = free_vram_bytes - KEEP_HOT_VRAM_HEADROOM_BYTES
    if available < component_bytes:
        _add_generation_warning(
            f"keep_models_hot: insufficient free VRAM to keep '{component_name}' resident "
            f"({component_bytes / 1024 ** 3:.2f} GB needed, {available / 1024 ** 3:.2f} GB "
            f"available after headroom) - falling back to normal offload for this component.",
            code="keep_hot_vram_guard",
        )
        return False

    return True


def component_nbytes(component) -> int:
    """Best-effort total parameter+buffer byte size of an nn.Module-like
    component. Returns 0 (never blocks the guard) if the component is None or
    byte-counting fails for any reason.
    """
    if component is None:
        return 0
    try:
        total = 0
        for p in component.parameters():
            total += p.numel() * p.element_size()
        for b in component.buffers():
            total += b.numel() * b.element_size()
        return total
    except Exception:
        return 0
