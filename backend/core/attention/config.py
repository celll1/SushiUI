"""
Backend-string normalization, capability resolution, and diffusers mapping.

This module owns the string vocabulary shared by inference (``attention_type``)
and training (``attention_backend``) selectors:

    * ``normalize_backend`` collapses UI/alias spellings ("normal", "none",
      "sdpa", ``None``) to the canonical registry keys, while letting
      non-fungible backends owned by other subsystems (``sla``) pass through
      verbatim.
    * ``resolve_backend`` applies MODE + capability guards and downgrades to
      native (with a one-time reason log) when the requested backend cannot
      satisfy the call.
    * ``to_diffusers_backend`` maps our canonical string to the string the
      diffusers ``set_attention_backend`` registry expects, so both registries
      are driven from ONE source string.
"""

from typing import Optional

import torch

from .registry import BACKENDS

# UI / diffusers alias spellings -> canonical registry key. ``None`` (no
# selection) resolves to native. This is what kills the historical
# "normal" vs "native" divergence: both map to "native".
_ALIASES = {
    "normal": "native",
    "none": "native",
    "sdpa": "native",
    None: "native",
}

# Backends owned by OTHER subsystems that are non-fungible and must never be
# rewritten to native by ``normalize_backend``. SLA-trained models are
# structurally incompatible with normal attention (extra ``proj_l`` layer), so
# swallowing the string would silently corrupt them. These are short-circuited
# at the top of ``dispatch_attention`` before registry resolution (R2).
_PASSTHROUGH = {"sla"}

# One-time dedup for unknown-backend warnings and downgrade-reason logs, keyed
# by a stable string so we don't spam per-attention-call.
_normalize_warned = set()
_downgrade_logged = set()


def normalize_backend(backend: Optional[str]) -> str:
    """Normalize a backend string to a canonical key.

    Rules:
        * ``None`` -> ``"native"``.
        * Case-insensitive; leading/trailing whitespace stripped.
        * Passthrough backends (``sla``) are returned verbatim (lowercased).
        * Alias spellings ("normal"/"none"/"sdpa") map to "native".
        * Known registry keys map to themselves.
        * Anything else -> "native" with a one-time warning.
    """
    if backend is None:
        return "native"

    key = backend.strip().lower()

    # Non-fungible backends owned elsewhere -- never rewrite.
    if key in _PASSTHROUGH:
        return key

    if key in _ALIASES:
        return _ALIASES[key]

    if key in BACKENDS:
        return key

    if key not in _normalize_warned:
        print(f"[Attention] unknown backend '{backend}'; using native")
        _normalize_warned.add(key)
    return "native"


def _log_downgrade(backend: str, reason: str) -> None:
    """Emit a downgrade reason once per (backend, reason) pair."""
    dedup_key = f"{backend}:{reason}"
    if dedup_key not in _downgrade_logged:
        print(f"[Attention] {reason}")
        _downgrade_logged.add(dedup_key)


def _heads(tensor: torch.Tensor, layout: str) -> int:
    """Return the number of heads given the tensor layout.

    Both layouts are 4D with head_dim last:
        * BSHD == [B, S, H, D]  -> heads at dim 2
        * BHSD == [B, H, S, D]  -> heads at dim 1
    """
    return tensor.shape[1] if layout == "BHSD" else tensor.shape[2]


def resolve_backend(
    backend: str,
    mode,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    layout: str = "BSHD",
) -> str:
    """Apply MODE + capability guards, downgrading to native when needed.

    Args:
        backend: Canonical backend key (already ``normalize_backend``-d, and NOT
            a passthrough backend -- those are short-circuited earlier).
        mode: ``AttentionMode`` (a ``str`` enum). Compared to ``"training"``.
        query/key: BSHD or BHSD tensors (head_dim is always the last dim).
        attn_mask: Optional mask; presence forces native for mask-less backends.
        layout: "BSHD" (canonical) or "BHSD" (arch-local) -- used to locate the
            head axis for the GQA guard.

    Returns:
        The backend key to actually use ("native" on any downgrade).
    """
    b = BACKENDS.get(backend)
    if b is None:
        # Defensive: should not happen post-normalize (passthrough handled
        # earlier), but never dispatch to a missing backend.
        return "native"

    if b.name == "native":
        return "native"

    head_dim = query.shape[-1]

    # MODE guard: no backward kernel for training.
    # AttentionMode is a str Enum, so ``mode == "training"`` matches without an
    # import (avoids a config<->dispatch circular import).
    is_training = (mode == "training") or (getattr(mode, "value", None) == "training")
    if is_training and not b.trainable:
        _log_downgrade(b.name, f"{b.name} has no backward; using native for training")
        return "native"

    # Mask guard.
    if attn_mask is not None and not b.supports_mask:
        _log_downgrade(b.name, f"{b.name} ignores masks; using native (mask present)")
        return "native"

    # head_dim max guard.
    if b.max_head_dim is not None and head_dim > b.max_head_dim:
        _log_downgrade(b.name, f"{b.name} unsupported head_dim={head_dim} (>max {b.max_head_dim})")
        return "native"

    # head_dim allowed-set guard.
    if b.allowed_head_dims is not None and head_dim not in b.allowed_head_dims:
        _log_downgrade(b.name, f"{b.name} unsupported head_dim={head_dim}")
        return "native"

    # GQA guard.
    if not b.supports_gqa:
        h_q = _heads(query, layout)
        h_kv = _heads(key, layout)
        if h_kv != h_q:
            _log_downgrade(b.name, f"{b.name} requires equal q/kv heads (got {h_q} vs {h_kv})")
            return "native"

    return b.name


# Canonical string -> diffusers AttentionBackendName string. They are identical
# for the backends we expose, but the explicit map documents the contract and
# guards against a future divergence. Unknown inputs collapse to native.
_DIFFUSERS_MAP = {
    "native": "native",
    "flash": "flash",
    "sage": "sage",
}


def to_diffusers_backend(backend: Optional[str]) -> str:
    """Map our canonical backend string to the diffusers registry string.

    Used by the FLUX.2 / SDXL diffusers ``set_attention_backend`` path so the
    diffusers registry and our conduit share ONE source string.

    Conduit-only backends (e.g. ``tq``) have no diffusers registry equivalent, so
    they collapse to ``native`` here with a one-time warning -- the diffusers path
    (FLUX.2 default processors, SDXL/FLUX.2 training) cannot run them. Such
    backends are only effective on conduit-routed paths.
    """
    norm = normalize_backend(backend)
    mapped = _DIFFUSERS_MAP.get(norm, "native")
    if mapped == "native" and norm != "native":
        key = f"diffusers:{norm}"
        if key not in _normalize_warned:
            print(
                f"[Attention] backend '{norm}' has no diffusers equivalent; "
                f"using native on the diffusers path (conduit-routed paths still use '{norm}')"
            )
            _normalize_warned.add(key)
    return mapped
