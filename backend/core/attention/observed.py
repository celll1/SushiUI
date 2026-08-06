"""Which attention backend(s) a generation ACTUALLY ran.

The conduit resolves the effective backend per attention call: a requested
backend can be downgraded by a capability guard (``config.resolve_backend``) or
by a kernel that failed at runtime (``dispatch_attention``'s ``None`` fallback),
and an architecture that does not route through the conduit at all runs none of
them. The REQUESTED string is therefore not evidence of what executed, and a
gallery row that records only the request can claim a backend that never ran.

This module is the witness. ``note_backend`` is called from the conduit with the
backend that is about to run; ``observed_backends`` reads back the set for one
generation.

ATTRIBUTION, and why it is not a single global set. Requests do not serialize at
the handler: a queued second request sits inside its own
``start_generation()``..``complete_generation()`` window for the whole of the
first request's denoise (see the long note in ``api/generation_status.py``). A
single "current generation" set would therefore hand the running generation's
observations to the waiting one, and a guard against that would make the running
generation record NOTHING -- which is what a first draft of this module did, and
it cost a real row its record. So the identity travels with the EMITTER, exactly
as the warning channel's does: the generation-status ``ContextVar`` is read per
call, and each generation gets its own bucket. Only a thread that inherited no
context at all falls back to the newest started generation.

COST. ``note_backend`` is on the per-attention-call path: one ``ContextVar.get``
(no lock) plus a dict lookup and a set add on an at-most-4-element set.
"""

from collections import OrderedDict
from typing import Optional, Tuple

# generation id -> the backends observed for it. Bounded like the warning
# buckets; a generation whose bucket has been evicted reports nothing rather
# than something belonging to another run.
_MAX_BUCKETS = 8
_observed: "OrderedDict[int, set]" = OrderedDict()

# Newest generation announced by ``begin_generation``; the fallback identity for
# an emitter running on a thread that inherited no context.
_latest_id: int = 0

# Resolved once: the generation-status ContextVar. Held directly (not via
# ``current_generation_id()``) because that helper takes a module lock, and this
# runs per attention call.
_ctx_var = None
_ctx_resolved = False


def _current_id() -> int:
    """Generation id of the CALLER, or the newest started one."""
    global _ctx_var, _ctx_resolved
    if not _ctx_resolved:
        _ctx_resolved = True
        try:
            from api.generation_status import _current_generation

            _ctx_var = _current_generation
        except Exception:
            _ctx_var = None
    if _ctx_var is not None:
        try:
            gen_id = _ctx_var.get()
        except Exception:
            gen_id = 0
        if gen_id:
            return gen_id
    return _latest_id


def begin_generation(generation_id: int) -> None:
    """Open a bucket for ``generation_id`` (called by ``start_generation``)."""
    global _latest_id
    generation_id = int(generation_id)
    _latest_id = generation_id
    _observed[generation_id] = set()
    while len(_observed) > _MAX_BUCKETS:
        _observed.popitem(last=False)


def note_backend(backend: str) -> None:
    """Record that ``backend`` ran an attention call. Hot path; never raises."""
    try:
        bucket = _observed.get(_current_id())
        if bucket is None:
            return
        if backend not in bucket:
            bucket.add(backend)
    except Exception:  # pragma: no cover - must never break a forward
        pass


def observed_backends(generation_id: Optional[int] = None) -> Tuple[str, ...]:
    """Backends observed for ``generation_id``, sorted; empty when unknown.

    Empty means "no evidence", which is the honest answer in three real cases:
    the architecture does not route attention through the conduit (LTX-2.3, the
    diffusers-dispatch paths), no attention ran at all, or the bucket is gone.
    """
    if generation_id is None:
        generation_id = _current_id()
    return tuple(sorted(_observed.get(int(generation_id), ())))


def reset() -> None:
    """Drop all recording (tests)."""
    global _latest_id
    _observed.clear()
    _latest_id = 0
