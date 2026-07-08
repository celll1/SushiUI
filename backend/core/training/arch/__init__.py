"""
Arch handler registry + factory (plan A.2, requirement #2).

``get_arch_handler(trainer)`` reads the trainer's already-computed ``is_<arch>``
flags ONCE and returns the registered handler, replacing the 111 ``if
self.is_<arch>`` branches with a single dispatch point. Adding an arch = one
registry entry.

Registry keys MUST equal the arch strings produced by
``BaseTrainer._build_cache_namespace`` (base_trainer.py:9821-9838) — cache
stability invariant (plan R6). The module-level assertion enforces this.

Flag-priority order below MIRRORS ``_build_cache_namespace`` exactly
(zimage, flux2, anima, lens, ideogram4, minit2i, krea2, sdxl, else sd15) so the
handler chosen here and the cache namespace can never diverge.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from core.training.arch.base_arch import ArchHandler
from core.training.arch.sd15 import SD15ArchHandler
from core.training.arch.sdxl import SDXLArchHandler
from core.training.arch.zimage import ZImageArchHandler
from core.training.arch.anima import AnimaArchHandler
from core.training.arch.lens import LensArchHandler
from core.training.arch.ideogram4 import Ideogram4ArchHandler
from core.training.arch.minit2i import MiniT2IArchHandler
from core.training.arch.krea2 import Krea2ArchHandler
from core.training.arch.flux2 import Flux2ArchHandler
from core.training.arch.ltx2 import Ltx2ArchHandler

ARCH_REGISTRY: Dict[str, Type[ArchHandler]] = {
    "sd15": SD15ArchHandler,
    "sdxl": SDXLArchHandler,
    "zimage": ZImageArchHandler,
    "anima": AnimaArchHandler,
    "lens": LensArchHandler,
    "ideogram4": Ideogram4ArchHandler,
    "minit2i": MiniT2IArchHandler,
    "krea2": Krea2ArchHandler,
    "flux2": Flux2ArchHandler,
    "ltx2": Ltx2ArchHandler,
}

# R6 invariant: registry keys == _build_cache_namespace arch strings.
_EXPECTED_ARCH_KEYS = {
    "sd15", "sdxl", "zimage", "anima", "lens",
    "ideogram4", "minit2i", "krea2", "flux2", "ltx2",
}
assert set(ARCH_REGISTRY) == _EXPECTED_ARCH_KEYS, (
    f"ARCH_REGISTRY keys {set(ARCH_REGISTRY)} != expected {_EXPECTED_ARCH_KEYS} "
    f"(must match _build_cache_namespace, plan R6)"
)


def resolve_arch_name(trainer: Any) -> str:
    """Return the arch-registry key for ``trainer`` by reading its ``is_<arch>``
    flags in the SAME priority order as ``_build_cache_namespace``.

    Falls through to ``"sd15"`` (today's ``else`` behavior — SD1.5/SDXL share the
    flag set; ``is_sdxl`` distinguishes SDXL, everything else is SD1.5)."""
    if getattr(trainer, "is_ltx2", False):
        return "ltx2"
    if getattr(trainer, "is_zimage", False):
        return "zimage"
    if getattr(trainer, "is_flux2", False):
        return "flux2"
    if getattr(trainer, "is_anima", False):
        return "anima"
    if getattr(trainer, "is_lens", False):
        return "lens"
    if getattr(trainer, "is_ideogram4", False):
        return "ideogram4"
    if getattr(trainer, "is_minit2i", False):
        return "minit2i"
    if getattr(trainer, "is_krea2", False):
        return "krea2"
    if getattr(trainer, "is_sdxl", False):
        return "sdxl"
    return "sd15"


def get_arch_handler(trainer: Any) -> ArchHandler:
    """Construct the ArchHandler bound to ``trainer`` (plan A.2).

    Reads the ``is_<arch>`` flags once via :func:`resolve_arch_name` and returns
    the registered handler instance. In P1 the handlers are stubs; base_trainer
    still uses its if-chains and never calls handler methods yet.
    """
    name = resolve_arch_name(trainer)
    handler_cls = ARCH_REGISTRY[name]
    return handler_cls(trainer)


__all__ = [
    "ARCH_REGISTRY",
    "ArchHandler",
    "get_arch_handler",
    "resolve_arch_name",
]
