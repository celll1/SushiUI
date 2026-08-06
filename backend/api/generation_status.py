"""In-memory image-generation status store.

This is a polling-friendly alternative to the WebSocket ``/api/v1/ws/progress``
channel (see ``backend/api/WS_PROTOCOL.md``). The WS channel is a global,
unfiltered broadcast with no ``complete``/``error`` message type — a client
can only infer completion/failure from the REST POST response returning.
This module keeps a small snapshot of the latest known state so a
frontend-less script can instead poll ``GET /generation/status`` without
holding a WebSocket connection open.

Single-process, single FastAPI worker assumption — same as
``ConnectionManager`` (the WS manager this mirrors). Not persisted across
backend restarts, and not multi-worker safe (this project runs one worker).

Hooked from (additive only, no change to WS behavior or generation logic):
  - ``create_progress_callback_factory()`` in ``backend/api/generation_utils.py``
    (per-step callback) -> ``update_progress()``
  - ``generate_txt2img`` / ``generate_img2img`` / ``generate_inpaint`` in
    ``backend/api/routes.py`` -> ``start_generation()`` at the top of the
    ``try:`` block, ``complete_generation()`` right before the success
    ``return``, and ``fail_generation()`` in the ``except`` blocks.
"""
import contextvars
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_lock = threading.Lock()

# --- Warning buckets -------------------------------------------------------
# ``warnings`` used to be a single list on ``_state``, cleared by every
# ``start_generation()``. That made the accumulator a module GLOBAL with no
# per-request identity: a second request entering ``start_generation()`` wiped
# the first request's warnings and mis-attributed every later one. Warnings now
# live in per-generation buckets keyed by a monotonic id that
# ``start_generation()`` returns and every route reads back with.
#
# ATTRIBUTION — what is actually guaranteed:
#
# Requests do NOT serialize at the handler. ``start_generation()`` runs at the
# top of the handler; the GPU slot is not acquired until the executor call much
# further down. A queued second request therefore sits INSIDE its own
# ``start_generation()``..``complete_generation()`` window, blocked on the slot,
# for the whole of the first request's denoise. "Newest started" is thus the
# WRONG generation for the entire time that matters — under a queue (the normal
# path, not a race) it would file the running generation's
# ``attention_kernel_fallback`` onto the request that is still waiting.
#
# So the emitter, not the reader, carries the identity: ``start_generation()``
# sets ``_current_generation`` (a ``contextvars.ContextVar``), which follows the
# handler's own task, and the routes wrap their ``run_in_executor`` call in
# ``contextvars.copy_context().run`` so the sampling thread inherits it. Bare
# ``run_in_executor`` does NOT propagate context (unlike ``asyncio.to_thread``),
# which is why that wrapping is deliberate and load-bearing.
#
# Guarantees:
#   * Anything raised on the handler's task or in its context-wrapped executor
#     work is attributed EXACTLY, regardless of overlap.
#   * A warning raised on a thread the emitter spawned itself (no inherited
#     context) falls back to the OLDEST still-active generation — the one
#     holding the GPU slot — which is the best available guess, not a guarantee.
#   * Nothing is attributed to a generation that has finished, and no
#     generation's bucket is ever cleared by another's.
_MAX_BUCKETS = 8
_buckets: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()
_active_ids: List[int] = []   # ids of generations that started and have not finished
_next_id: int = 0

# Set by start_generation() on the handler's context; read by add_warning().
_current_generation: "contextvars.ContextVar[int]" = contextvars.ContextVar(
    "sushiui_current_generation", default=0
)

_state: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "error"
    "generation_type": None,  # "txt2img" | "img2img" | "inpaint"
    "current_step": None,
    "total_steps": None,
    "phase": None,  # human-readable status text, e.g. "Step 3/28"
    "started_at": None,
    "updated_at": None,
    "last_error": None,
    "last_result": None,  # {"image_id", "filename", "seed"} of the last successful generation
    "warnings": [],  # [{"code", "message"}] feature-degradation notices for the current generation
    "generation_id": None,  # id of the newest generation (see the bucket note above)
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_generation(generation_type: str) -> int:
    """Call at the very start of a generation request (top of the try: block).

    Returns the generation id, which the route must hand back to
    ``get_warnings()``, ``complete_generation()`` and ``fail_generation()``.
    Also binds the id to the calling context so emitters deep in the call stack
    need no id — see the attribution note at the top of this module.
    """
    global _next_id
    with _lock:
        now = _now()
        _next_id += 1
        gen_id = _next_id
        _buckets[gen_id] = []
        _active_ids.append(gen_id)
        _evict_locked()
        _state["status"] = "running"
        _state["generation_type"] = generation_type
        _state["current_step"] = None
        _state["total_steps"] = None
        _state["phase"] = None
        _state["started_at"] = now
        _state["updated_at"] = now
        _state["last_error"] = None
        _state["generation_id"] = gen_id
        # Same list object as the bucket, so ``get_snapshot()`` keeps reporting
        # the live warnings of the newest generation.
        _state["warnings"] = _buckets[gen_id]
    _current_generation.set(gen_id)
    # Start recording which attention backend(s) this generation actually runs
    # (see core/attention/observed.py). Best-effort and lazily imported: the
    # status store must not hard-depend on the attention conduit, and a build
    # without it still generates.
    try:
        from core.attention.observed import begin_generation as _begin_attention
        _begin_attention(gen_id)
    except Exception:
        pass
    return gen_id


def _evict_locked() -> None:
    """Trim ``_buckets`` to ``_MAX_BUCKETS``, never dropping a LIVE bucket.

    ``popitem(last=False)`` alone took the strictly-oldest entry whether or not
    it was still being written to, which silently emptied a running
    generation's warnings — the exact failure mode this module exists to
    remove. Only INACTIVE (finished) buckets are evictable; when every retained
    bucket is still active the cap is allowed to overshoot, loudly.
    """
    while len(_buckets) > _MAX_BUCKETS:
        victim = next((gid for gid in _buckets if gid not in _active_ids), None)
        if victim is None:
            print(f"[GenerationStatus] {len(_buckets)} generations still active "
                  f"(> retention cap {_MAX_BUCKETS}); keeping every live warning "
                  f"bucket rather than dropping one.")
            return
        _buckets.pop(victim, None)


def _bucket_for_emitter() -> Optional[list]:
    """Bucket a context-less ``add_warning()`` should target (holds ``_lock``).

    The context var is authoritative; this is only reached when an emitter runs
    on a thread that inherited no context. It picks the OLDEST active
    generation — the one that owns the GPU slot and is actually computing —
    rather than the newest, which under a queue is a request still blocked
    before its first denoise step.
    """
    for gen_id in _active_ids:
        bucket = _buckets.get(gen_id)
        if bucket is not None:
            return bucket
    return None


def current_generation_id() -> int:
    """Id of the generation the CALLER belongs to, or 0 when there is none.

    Exposed for hot-path emitters that dedup their own warnings and must
    re-arm that dedup for each generation (see
    ``core/attention/backends._warn_kernel_fallback``). Resolves through the
    same context var / oldest-active fallback that ``add_warning()`` uses, so a
    dedup key built from it can never name a different generation than the one
    the warning lands on.
    """
    ctx_id = _current_generation.get()
    with _lock:
        if ctx_id:
            return ctx_id if ctx_id in _active_ids else 0
        for candidate in _active_ids:
            if candidate in _buckets:
                return candidate
        return 0


def add_warning(message: str, code: Optional[str] = None) -> None:
    """Record a feature-degradation notice for the calling generation.

    Only appends while that generation is running so stray warnings from
    background work don't leak into an idle/next snapshot. Pure dict ops under
    the lock — cannot raise for normal string/None inputs.
    """
    ctx_id = _current_generation.get()
    with _lock:
        if ctx_id:
            # The caller's identity is known: file it there, or nowhere. Never
            # spill a finished generation's late warning onto a live one.
            bucket = _buckets.get(ctx_id) if ctx_id in _active_ids else None
        else:
            bucket = _bucket_for_emitter()
        if bucket is None:
            return
        # Warnings are idempotent per generation; attention downgrades fire per
        # attention call (hundreds-thousands per run), so dedup here bounds the
        # list for every emitter.
        entry = {"code": code, "message": message}
        if entry in bucket:
            return
        bucket.append(entry)


def get_warnings(generation_id: Optional[int] = None) -> list:
    """Return a copy of the warnings recorded for a generation.

    ``generation_id`` is the value returned by ``start_generation()`` and every
    caller that has one MUST pass it. Without it the calling context is
    consulted, then the oldest active generation, then the most recent bucket —
    a best effort for callers with no id in hand, not an attribution guarantee.
    """
    with _lock:
        if generation_id is None:
            ctx_id = _current_generation.get()
            if ctx_id and ctx_id in _buckets:
                generation_id = ctx_id
        if generation_id is not None:
            return list(_buckets.get(generation_id) or [])
        bucket = _bucket_for_emitter()
        if bucket is None:
            bucket = next(reversed(_buckets.values())) if _buckets else []
        return list(bucket)


def update_progress(current_step: int, total_steps: int, phase: Optional[str] = None) -> None:
    """Call from the per-step progress callback (same callback that feeds the WS broadcast)."""
    with _lock:
        if _state["status"] != "running":
            # A stray callback firing after completion/failure should not
            # resurrect a "running" snapshot.
            return
        _state["current_step"] = current_step
        _state["total_steps"] = total_steps
        if phase is not None:
            _state["phase"] = phase
        _state["updated_at"] = _now()


def _finish(generation_id: Optional[int]) -> Optional[list]:
    """Retire a generation's bucket and return it (caller holds ``_lock``).

    The bucket itself is KEPT (bounded by ``_MAX_BUCKETS``, and never evicted
    while active) so a route can still read its own warnings after calling
    ``complete_generation()``.

    ``generation_id`` should always be the caller's own id. When it is omitted
    the CALLING CONTEXT is consulted before any global guess, so a finishing
    route can never retire an overlapping generation that is still running.
    """
    if generation_id is None:
        ctx_id = _current_generation.get()
        if ctx_id and ctx_id in _buckets:
            generation_id = ctx_id
        else:
            # No identity at all: retire the OLDEST active generation (the one
            # holding the GPU slot), matching _bucket_for_emitter().
            generation_id = _active_ids[0] if _active_ids else None
    if generation_id is not None and generation_id in _active_ids:
        _active_ids.remove(generation_id)
        _evict_locked()
    if generation_id is None:
        return None
    return _buckets.get(generation_id)


def complete_generation(last_result: Optional[Dict[str, Any]] = None,
                        generation_id: Optional[int] = None) -> None:
    """Call right before the success return of a generation endpoint."""
    with _lock:
        bucket = _finish(generation_id)
        if not _active_ids:
            _state["status"] = "idle"
            _state["current_step"] = None
            _state["total_steps"] = None
            _state["phase"] = None
            _state["last_error"] = None
        _state["updated_at"] = _now()
        if last_result is not None:
            result = dict(last_result)
            result["warnings"] = list(bucket or [])
            _state["last_result"] = result


def fail_generation(error_message: str, generation_id: Optional[int] = None) -> None:
    """Call from a generation endpoint's exception handler."""
    with _lock:
        _finish(generation_id)
        # Only claim the global lifecycle state once nothing is running:
        # flipping to "error" while an overlapping generation is still
        # sampling would report a failure the poller cannot attribute (and
        # would silence update_progress for the survivor).
        if not _active_ids:
            _state["status"] = "error"
            _state["current_step"] = None
            _state["total_steps"] = None
            _state["phase"] = None
        _state["updated_at"] = _now()
        _state["last_error"] = error_message


def get_snapshot() -> Dict[str, Any]:
    """Return a shallow copy of the current status snapshot."""
    with _lock:
        return dict(_state)
