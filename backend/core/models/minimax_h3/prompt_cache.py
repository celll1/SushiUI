"""Bounded LRU cache for MiniMax-H3's post-projection prompt embedding.

MiniMax-H3 has no CFG and no negative prompt -- guidance is distilled into the
weights -- so a prompt's encoded conditioning is fully determined by the
prompt string, which text encoder is loaded, which TE projection (if any) is
paired with it, and the loaded DiT's conditioning width. Re-encoding an
IDENTICAL prompt against the SAME loaded model currently re-pays the full
Qwen3-VL-32B streamed encode (measured ~21s) every call. This cache makes
that identity-repeat free.

SCOPE: only the plain (no vision references) ``ops.encode_prompt`` path --
the ref2va vision-conditioned encode also depends on reference image/video
content, which this cache's key does not capture, so that branch stays on
its direct, uncached call. Deliberate, not an oversight.

CACHE KEY: ``(text_encoder_path, te_projection_path, prompt, text_dim)`` --
path/string-based, never a Python object ``id()`` (an id can be reused after
garbage collection across a reload). ``text_dim`` guards a DiT-only reload,
which deliberately does NOT clear this cache (encoder/projection identity is
unchanged) but DOES replace ``transformer_config`` -- see ``_cache_key``.

``clear()`` runs on every FULL MiniMax-H3 model (re)load and on unload
(``core/pipeline.py``), not on a DiT-only reload (``reload.py``'s
``build_dit_only_reload`` keeps ``text_encoder_path``/``te_projection``
unchanged, so a DiT swap followed by the same prompt should keep hitting). A
live TE/projection component switch needs no explicit hook: it writes a new
path into the component dict, so the key misses on its own.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch

# Small and fixed: this is not meant to be an unbounded prompt history, just
# enough to make a same-prompt sweep (or a user regenerating the same prompt a
# few times) fast. Each entry is a CPU tensor of num_text_tokens x text_dim --
# a few hundred KB to a few MB -- so this cap is a trivial memory cost.
_MAX_ENTRIES = 8

_CacheKey = Tuple[str, Optional[str], str, int]
_CacheValue = Tuple[torch.Tensor, int]

_cache: "OrderedDict[_CacheKey, _CacheValue]" = OrderedDict()


def clear() -> None:
    """Drop every cached embedding.

    Call on every full MiniMax-H3 model load and on unload -- see the module
    docstring's "WHY A FULL MODEL LOAD CLEARS IT" for why a DiT-only reload
    must NOT call this.
    """
    _cache.clear()


def _cache_key(
    text_encoder_path: str, te_projection_path: Optional[str], prompt: str, text_dim: int,
) -> _CacheKey:
    # `text_dim` (the loaded DiT's conditioning width) is in the key, not just
    # (encoder, projection, prompt): a DiT-only reload deliberately does not
    # clear this cache (the encoder/projection identity is unchanged), but it
    # DOES replace `transformer_config`. Two checkpoints in one H3 tree that
    # differed in `text_dim` would otherwise let a hit hand the old-width
    # embedding to the new DiT -- the miss path's own `d_in`-mismatch guard
    # (`h3_pipeline_ops.project_prompt_embeds`) would never run to catch it.
    return (str(text_encoder_path or ""), te_projection_path, prompt, int(text_dim))


def get_or_encode_prompt(
    text_encoder_path: str,
    te_projection_path: Optional[str],
    prompt: str,
    text_dim: int,
    encode_fn: Callable[[], _CacheValue],
) -> Tuple[_CacheValue, bool]:
    """The cached ``(prompt_embeds_cpu, num_text_tokens)`` pair, or a fresh one.

    ``encode_fn`` takes no arguments and, on a miss, is called exactly once to
    produce the SAME post-projection ``(prompt_embeds_cpu, num_text_tokens)``
    pair the caller would otherwise compute directly -- this function only
    remembers that result, it does not know how to encode or project on its
    own. Returns ``(result, cache_hit)`` so the caller can log whether the
    encode was skipped and can still run any per-generation bookkeeping
    (provenance fields, warnings) that a skipped encode would also have run.

    Every tensor that crosses this cache's boundary is cloned -- once on
    store, once on a hit -- so the cache owns independent CPU memory that
    neither a caller's later in-place op nor a fresh encode overwriting the
    same variable name can corrupt a value another generation is still
    holding a reference to. The clone cost is the same few-hundred-KB-to-few-MB
    copy either way; trivial next to the encode it is avoiding.
    """
    key = _cache_key(text_encoder_path, te_projection_path, prompt, text_dim)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        embeds, num_tokens = cached
        return (embeds.clone(), num_tokens), True

    embeds, num_tokens = encode_fn()
    stored: _CacheValue = (embeds.clone(), num_tokens)
    _cache[key] = stored
    _cache.move_to_end(key)
    while len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)
    return (embeds, num_tokens), False


def cache_size() -> int:
    """Current entry count -- test/observability hook, not used on the hot path."""
    return len(_cache)
