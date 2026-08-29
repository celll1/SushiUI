"""Which open checkpoint a raw MiniMax-H3 DiT key is read from.

See ``docs/guides/MINIMAX_H3_HYBRID_LOADER_DESIGN.md``. The reader is a
PASSIVE DISPATCHER: it never enumerates keys and never holds a state
dict. Key traversal stays where it already is -- ``_map_dit_state_dict`` walks
the BASE header (insertion order, not sorted) and asks for one key at a time --
so a reader is substitutable for a bare ``safe_open`` handle at every call site
that consumes one.

There are TWO such call sites in ``loader.py``, and both go through a reader:
``_map_dit_state_dict`` for the weights and ``_int8_convrot_layers_from_markers``
for the ``.comfy_quant`` markers. A marker read straight from the base handle
would pin a ConvRot layer's provenance to the base while its weight came from
the overlay -- a load that succeeds and infers garbage.

MMAP LIFETIME
--------------------------------------------------------------------
``safe_open`` hands back memory-mapped tensors and ``_build_transformer``
installs them with ``load_state_dict(assign=True)``, so the live model's CPU
weights ARE the file mappings. A hybrid model therefore keeps TWO files mapped
for its whole CPU lifetime (12-21 GB each), and on Windows both stay
undeletable/unreplaceable until the model is dropped. That is the cost of not
materialising two state dicts, and nothing here tries to undo it.

The CONCURRENT count is two, not the four the design doc feared:
``_guard_component_file`` maps one file at a time (and only when the header
declares ``.comfy_quant`` markers at all), and both guard passes are finished
before this module opens anything. This module adds no mapping beyond the two it
is asked for and copies no tensor -- ``get_tensor`` returns the handle's own
result unchanged, same storage, same ``data_ptr``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Set

import torch


class SingleTensorReader:
    """One checkpoint, read through the same interface a hybrid uses.

    The base-only load path uses this rather than the raw handle so that the two
    paths are one piece of code: a difference between "with a reader" and
    "without" cannot exist if there is no "without".
    """

    __slots__ = ("_base",)

    is_hybrid = False

    def __init__(self, base_handle: Any):
        self._base = base_handle

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._base.get_tensor(key)


class HybridTensorReader:
    """Two open checkpoints; the selector decides per key, and it decides totally.

    ``selector.source_for(key)`` is answered for every key that is asked for, and
    an answer that is neither ``"base"`` nor ``"overlay"`` raises rather than
    defaulting -- a silent default is the one failure mode of this class that
    produces a clean load and a wrong model.

    Keys actually taken from the overlay are recorded (``overlay_keys_read``) so
    the caller can check the realised selection against the preflight's expected
    set, the same way the quantized-Linear swap counts are checked.
    """

    __slots__ = ("_base", "_overlay", "_selector", "_overlay_keys_read",
                 "_base_token", "_overlay_token")

    is_hybrid = True

    def __init__(self, base_handle: Any, overlay_handle: Any, selector: Any):
        # Imported here, not at module scope: ``hybrid_spec`` imports ``loader``,
        # which is what constructs this class.
        from .hybrid_spec import BASE, OVERLAY

        if selector is None:
            raise ValueError("a hybrid reader needs a selector; without one no key has a source")
        self._base = base_handle
        self._overlay = overlay_handle
        self._selector = selector
        self._base_token = BASE
        self._overlay_token = OVERLAY
        self._overlay_keys_read: Set[str] = set()

    def get_tensor(self, key: str) -> torch.Tensor:
        source = self._selector.source_for(key)
        if source == self._overlay_token:
            self._overlay_keys_read.add(key)
            return self._overlay.get_tensor(key)
        if source != self._base_token:
            raise ValueError(
                f"{type(self._selector).__name__}.source_for({key!r}) returned {source!r}; a "
                f"source must be {self._base_token!r} or {self._overlay_token!r}. Refusing "
                f"rather than guessing a file.")
        return self._base.get_tensor(key)

    @property
    def overlay_keys_read(self) -> Set[str]:
        return set(self._overlay_keys_read)


@contextmanager
def open_dit_reader(
    base_path: str,
    *,
    overlay_path: Optional[str] = None,
    selector: Any = None,
) -> Iterator[Any]:
    """Open the one or two checkpoints a DiT load reads, and close them together.

    ``overlay_path=None`` yields a ``SingleTensorReader`` over exactly the one
    handle the base-only path always opened. The mappings live until the block
    exits, and the tensors installed from them live longer (see the module
    docstring).
    """
    from safetensors import safe_open

    with safe_open(base_path, framework="pt", device="cpu") as base_handle:
        if overlay_path is None:
            yield SingleTensorReader(base_handle)
            return
        with safe_open(overlay_path, framework="pt", device="cpu") as overlay_handle:
            yield HybridTensorReader(base_handle, overlay_handle, selector)
