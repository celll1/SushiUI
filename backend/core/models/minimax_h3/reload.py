"""DiT-only reload support for MiniMax-H3 checkpoints in one model tree."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .loader import _build_transformer, detect_minimax_h3_layout


_SHARED_LAYOUT_KEYS = ("root", "official", "vae", "audio_vae", "text_encoder")


def same_path(left: Optional[str], right: Optional[str]) -> bool:
    """Whether two optional paths name the same file (case- and link-folded)."""
    if left is None or right is None:
        return left is right
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


def build_dit_only_reload(
    current_components: Dict[str, Any],
    current_source: str,
    new_source: str,
) -> Optional[Dict[str, Any]]:
    """Build a replacement DiT while retaining an unchanged H3 shared bundle.

    ``None`` means the two sources do not resolve to the same component tree and
    the caller must use the ordinary full-model load. The current component dict
    is never mutated, including when the replacement transformer fails to load.
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

    transformer, transformer_config = _build_transformer(
        new_layout["dit"], torch.bfloat16, new_layout["official"])
    replacement = dict(current_components)
    replacement["transformer"] = transformer
    replacement["transformer_config"] = transformer_config
    replacement["variant"] = new_layout["variant"]
    return replacement
