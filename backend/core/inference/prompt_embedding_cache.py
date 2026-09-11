"""Small CPU LRU for generation prompt-conditioning tensors."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any, Hashable
import weakref

import torch


def _copy_tree(value: Any, device: str | torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _copy_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_copy_tree(item, device) for item in value]
    return value


class PromptEmbeddingCache:
    """Bounded cache isolated by the live text-encoder object.

    The weak reference prevents cached tensors from extending a model's
    lifetime. Callers still provide a complete, architecture-specific key for
    every conditioning input that can change an embedding.
    """

    def __init__(self, max_entries: int = 16):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            tuple[int, Hashable], tuple[weakref.ReferenceType, Any]
        ] = OrderedDict()
        self._lock = RLock()

    def get(
        self,
        encoder: object,
        key: Hashable,
        device: str | torch.device,
    ) -> tuple[Any, bool]:
        entry_key = (id(encoder), key)
        with self._lock:
            entry = self._entries.get(entry_key)
            if entry is None:
                return None, False
            encoder_ref, stored = entry
            if encoder_ref() is not encoder:
                del self._entries[entry_key]
                return None, False
            self._entries.move_to_end(entry_key)
            return _copy_tree(stored, device), True

    def put(self, encoder: object, key: Hashable, value: Any) -> None:
        entry_key = (id(encoder), key)
        stored = _copy_tree(value, "cpu")
        with self._lock:
            self._entries[entry_key] = (weakref.ref(encoder), stored)
            self._entries.move_to_end(entry_key)
            dead = [item_key for item_key, (ref, _value) in self._entries.items()
                    if ref() is None]
            for item_key in dead:
                self._entries.pop(item_key, None)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


generation_prompt_cache = PromptEmbeddingCache(max_entries=8)


def tokenizer_cache_key(tokenizer: object) -> tuple:
    """Stable tokenizer settings that affect token IDs and prompt templates."""
    try:
        tokenizer_length = len(tokenizer)
    except (TypeError, AttributeError):
        tokenizer_length = None
    return (
        type(tokenizer).__module__,
        type(tokenizer).__qualname__,
        str(getattr(tokenizer, "name_or_path", "")),
        getattr(tokenizer, "vocab_size", None),
        tokenizer_length,
        getattr(tokenizer, "model_max_length", None),
        str(getattr(tokenizer, "chat_template", "")),
    )
