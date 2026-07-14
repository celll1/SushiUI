"""In-RAM, thread-safe, LRU-bounded cache for loop-generation latent passthrough.

Used by the per-request ``loop_decode`` param (see custom_sampling.py's three
Stage-3 decode sites and routes.py's txt2img/img2img/inpaint endpoints).

When ``loop_decode="none"``, a loop step skips the VAE/PiD decode entirely and
hands back an opaque ``latent_id`` instead of an image. The stored tensor is
the model's clean denoised latent, PRE-unscale -- i.e. the SAME frame img2img's
own initial-image encode produces (``(vae.encode(img) - shift_factor) *
scaling_factor``). The NEXT loop step (img2img/inpaint ``input_latent_id``)
retrieves it and feeds it directly as the denoise loop's starting latent, with
no re-scaling and no VAE round-trip.

In-RAM only (cleared on process restart), matching the WS preview-frame cache
in routes.py -- this is a short-lived hand-off between two HTTP requests of the
same client-driven loop, not a persisted artifact. Bounded to a small LRU so an
abandoned/failed loop can't leak memory indefinitely; entries are tiny (a
1024px SDXL latent is [1,4,128,128] fp32 ~256KB), so the bound is about hygiene,
not capacity.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

# Small on purpose: entries are hand-offs between consecutive loop steps, not a
# durable store. 32 covers many concurrent/overlapping loop chains with margin.
_MAX_ENTRIES = 32

_lock = threading.Lock()
_cache: "OrderedDict[str, Tuple[torch.Tensor, Dict[str, Any]]]" = OrderedDict()


def store_latent(latent: torch.Tensor, meta: Optional[Dict[str, Any]] = None) -> str:
    """Store a latent tensor under a new opaque id and return that id.

    The tensor is detached and moved to CPU before storage (never keeps a GPU
    reference alive between requests). Evicts the least-recently-used entry
    when the cache is at capacity.

    Args:
        latent: The latent tensor to cache (any shape/dtype; caller's concern).
        meta: Optional bookkeeping dict (e.g. width/height/seed/arch) that the
            consumer of ``get_latent`` may use, but the cache itself never
            inspects.

    Returns:
        A uuid4 hex string identifying this entry.
    """
    latent_id = uuid.uuid4().hex
    cpu_latent = latent.detach().to("cpu")
    with _lock:
        _cache[latent_id] = (cpu_latent, dict(meta or {}))
        _cache.move_to_end(latent_id)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return latent_id


def get_latent(latent_id: str) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
    """Retrieve a cached (latent, meta) pair, or None if missing/expired/evicted.

    A hit promotes the entry to most-recently-used, so an actively-chained
    loop's latent is never evicted ahead of an abandoned one under LRU pressure.
    """
    with _lock:
        entry = _cache.get(latent_id)
        if entry is None:
            return None
        _cache.move_to_end(latent_id)
        latent, meta = entry
        return latent, dict(meta)


def clear() -> None:
    """Drop all cached latents. Test-only; production code never needs this."""
    with _lock:
        _cache.clear()


def size() -> int:
    """Current entry count. Test-only introspection."""
    with _lock:
        return len(_cache)
