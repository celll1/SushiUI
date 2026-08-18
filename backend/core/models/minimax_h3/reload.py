"""DiT-only reload support for MiniMax-H3 checkpoints in one model tree."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .loader import _build_transformer, detect_minimax_h3_layout


_SHARED_LAYOUT_KEYS = ("root", "official", "vae", "audio_vae", "image_vae", "text_encoder")


def canonical_path(path: str) -> str:
    """The folded form ``same_path`` compares.

    Its own function because the hybrid's model-identity token folds paths too
    (``hybrid_spec.hybrid_model_identity``), and a token folded differently from
    the equality checks it exists to match would drift silently.
    """
    return os.path.normcase(os.path.realpath(path))


def same_path(left: Optional[str], right: Optional[str]) -> bool:
    """Whether two optional paths name the same file (case- and link-folded)."""
    if left is None or right is None:
        return left is right
    return canonical_path(left) == canonical_path(right)


def build_dit_only_reload(
    current_components: Dict[str, Any],
    current_source: str,
    new_source: str,
    *,
    hybrid: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Build a replacement DiT while retaining an unchanged H3 shared bundle.

    ``None`` means the two sources do not resolve to the same component tree and
    the caller must use the ordinary full-model load. The current component dict
    is never mutated, including when the replacement transformer fails to load.

    ``hybrid`` is a validated ``MiniMaxH3HybridPreflight`` whose base is
    ``new_source``; the replacement DiT is then the merged one and reports
    ``variant="hybrid"``. It changes nothing about the atomicity above: the
    merged transformer is built before anything is swapped, so a refused
    preflight or a failed merge leaves the live model, TE, VAEs and schedulers
    exactly as they were.
    """
    if current_components.get("type") != "minimax_h3":
        return None

    current_layout = detect_minimax_h3_layout(current_source)
    new_layout = detect_minimax_h3_layout(new_source)
    if current_layout is None or new_layout is None:
        return None
    if not all(same_path(current_layout.get(key), new_layout.get(key))
               for key in _SHARED_LAYOUT_KEYS):
        return None

    if hybrid is not None and not same_path(hybrid.spec.base_dit_path, new_layout["dit"]):
        # Not ``None``: falling back to the full loader here would serve a hybrid
        # request as a base-only load and report success.
        raise ValueError(
            f"the hybrid preflight validated {hybrid.spec.base_dit_path!r} as the base, but "
            f"{new_source!r} resolves to {new_layout['dit']!r}.")

    # Base-only keeps the exact three-argument call it always had.
    transformer, transformer_config = _build_transformer(
        new_layout["dit"], torch.bfloat16, new_layout["official"],
        **({} if hybrid is None else {"hybrid": hybrid}))
    replacement = dict(current_components)
    # The copy carries the PREVIOUS DiT's hybrid provenance. A base-only reload
    # after a hybrid one must not keep reporting a recipe it no longer has.
    from .hybrid_spec import HYBRID_COMPONENT_KEYS  # imports this module; import late.

    for key in HYBRID_COMPONENT_KEYS:
        replacement.pop(key, None)
    replacement["transformer"] = transformer
    replacement["transformer_config"] = transformer_config
    replacement["variant"] = new_layout["variant"]
    if hybrid is not None:
        from .hybrid_spec import hybrid_component_fields

        replacement.update(hybrid_component_fields(hybrid))
    return replacement
