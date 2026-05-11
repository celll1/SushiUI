"""Migrate per-parameter optimizer state when the head's tag axis changes.

When a tagger run resumes with a vocabulary that differs from the saved
checkpoint (e.g. alias additions caused a few tags to merge), the head
parameter changes shape on the new model.  PyTorch's
``optimizer.load_state_dict`` does NOT reshape state tensors, so the
loaded ``exp_avg`` / ``state1`` etc. end up mismatched with the new
parameter, corrupting the momentum and producing a large loss spike.

This module rewrites the ``state[param_idx]`` entries for the head's
weight and bias in the saved state dict so they line up with the new
shape via tag-name alignment (mirroring ``_inherit_head``).

Supported optimizers:
  - AdamW (FP32)        — exp_avg, exp_avg_sq           : tag-name aligned
  - Lion (FP32)         — exp_avg                       : tag-name aligned
  - AdamW8bit (bnb)     — state1/2, absmax1/2, qmap1/2  : RESET (see below)
  - Lion8bit (bnb)      — state1, absmax1, qmap1        : RESET (see below)

8-bit optimizer handling
------------------------
8-bit ``state1``/``state2`` are uint8-quantised values whose dequantised
form is ``qmap[state[i]] * absmax[block_idx]``.  When we re-arrange rows
by tag name, the per-block ``absmax`` is no longer aligned with the new
block boundaries — and bitsandbytes' ``init_state`` (which would compute
fresh ``absmax``) is only invoked when ``"step"`` is absent from the
state.  Mixing migrated ``state1`` with stale or missing ``absmax``
causes ``KeyError: 'absmax1'`` in ``update_step`` (or, if we kept the
old ``absmax``, silently corrupts the head momentum).

We therefore **clear the entire param state for 8-bit head params** so
bnb re-initialises everything from scratch on the next ``step()``.  This
loses the head's accumulated momentum (a 1-step perturbation, comparable
in size to the original vocab-mismatch spike we are trying to prevent),
but keeps training stable and bit-correct.

Properly re-quantising the migrated ``state1`` (dequantise via old
``absmax``, reorder by tag name, re-quantise with new block-wise
``absmax``) is feasible but adds significant complexity and depends on
qmap monotonicity; left as a future enhancement.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


# Per-parameter state keys whose first dim matches the parameter shape
# (these are migrated by tag-name alignment).
_TAG_AXIS_KEYS = ("exp_avg", "exp_avg_sq", "state1", "state2")

# Per-block metadata (block_size=256). Block boundaries depend on
# parameter numel, so we cannot reshape — reset to let bnb recompute.
_BLOCK_WISE_KEYS = (
    "absmax1", "absmax2",
    "max1", "max2", "new_max1", "new_max2",
)

# Keys that are global / shared and need no migration.
_PRESERVE_KEYS = ("step", "qmap1", "qmap2", "gnorm_vec")


def _find_param_index(
    optimizer: torch.optim.Optimizer,
    target: torch.nn.Parameter,
) -> Optional[int]:
    """Return the global parameter index used by ``optimizer.state_dict()``.

    PyTorch flattens param_groups and assigns each parameter a sequential
    integer key.  We mirror that ordering to look up state entries.
    """
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is target:
                return idx
            idx += 1
    return None


def _is_8bit_state(state: Dict[str, Any]) -> bool:
    """Detect bitsandbytes 8-bit optimizer state by characteristic keys."""
    return any(k in state for k in ("state1", "state2", "absmax1", "absmax2"))


def _migrate_one_param_state(
    state: Dict[str, Any],
    new_shape: Tuple[int, ...],
    old_tag_to_idx: Dict[str, int],
    new_tag_to_idx: Dict[str, int],
) -> Dict[str, int]:
    """In-place migration of one parameter's optimizer state.

    Returns
    -------
    dict
        ``{"copied": int, "reset": int, "preserved": int, "mode": str}``
        ``mode`` is ``"tag_aligned"`` for FP32 optimisers and
        ``"reset_8bit"`` when the param's state was cleared to let bnb
        re-initialise (see module docstring).
    """
    stats: Dict[str, int] = {"copied": 0, "reset": 0, "preserved": 0,
                             "mode": "tag_aligned"}
    if not new_shape:
        return stats
    new_num_tags = int(new_shape[0])

    # 8-bit optimisers: clear the entire param state so that bnb's
    # ``init_state`` fires on the next ``step()``.  Required because:
    #   • migrating ``state1`` (uint8) by tag name shifts block boundaries,
    #     so the stale ``absmax`` no longer dequantises correctly;
    #   • bnb's init_state only fires when ``"step"`` is absent — keeping
    #     ``state1`` + dropping ``absmax`` results in ``KeyError`` instead
    #     of a re-initialisation.
    # Same vocab size with identical tag→idx mapping won't reach this
    # function (the caller guards on dict equality), but if state1 already
    # matches the new shape we still skip the reset to avoid needless
    # momentum loss on edge-case calls.
    if _is_8bit_state(state):
        s1 = state.get("state1")
        if torch.is_tensor(s1) and tuple(s1.shape) == tuple(new_shape):
            stats["mode"] = "no_op_8bit"
            return stats
        n_keys = sum(1 for k, v in state.items() if torch.is_tensor(v))
        state.clear()
        stats["reset"] = n_keys
        stats["mode"] = "reset_8bit"
        return stats

    # FP32 optimisers (AdamW, Lion): tag-name aligned migration.
    for key in list(state.keys()):
        val = state[key]

        if key in _PRESERVE_KEYS:
            stats["preserved"] += 1
            continue

        if not torch.is_tensor(val):
            continue

        if key in _BLOCK_WISE_KEYS:
            # FP32 path: should never see block-wise keys, but defensive.
            del state[key]
            stats["reset"] += 1
            continue

        if key in _TAG_AXIS_KEYS:
            if tuple(val.shape) == tuple(new_shape):
                continue   # same vocab — no migration needed
            new_t = torch.zeros(
                new_shape, dtype=val.dtype, device=val.device,
            )
            n_copied = 0
            old_first_dim = val.shape[0]
            for tag, new_idx in new_tag_to_idx.items():
                if new_idx >= new_num_tags:
                    continue
                old_idx = old_tag_to_idx.get(tag)
                if old_idx is None or old_idx >= old_first_dim:
                    continue
                new_t[new_idx] = val[old_idx]
                n_copied += 1
            state[key] = new_t
            stats["copied"] = max(stats["copied"], n_copied)
        # else: unknown key — leave untouched

    return stats


def migrate_head_optimizer_state(
    saved_state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    head_weight: torch.nn.Parameter,
    head_bias: Optional[torch.nn.Parameter],
    old_tag_to_idx: Dict[str, int],
    new_tag_to_idx: Dict[str, int],
) -> Dict[str, Any]:
    """Rewrite the head's per-parameter state in ``saved_state`` so it
    matches the new optimizer's parameter shape.

    Parameters
    ----------
    saved_state
        The dict returned by ``torch.load(<name>_optimizer.pt, ...)``.
        Mutated in place.
    optimizer
        The current optimizer (built from the resized model).  Used to
        find the parameter indices for ``head_weight`` and ``head_bias``.
    head_weight, head_bias
        Parameters of the new (resized) head.
    old_tag_to_idx, new_tag_to_idx
        Tag-to-index mappings from the saved checkpoint and the current
        run, respectively.  Used for tag-name aligned row migration.

    Returns
    -------
    dict
        Diagnostics: ``{"weight": {...}, "bias": {...},
        "head_weight_shape_old": tuple|None,
        "head_weight_shape_new": tuple}``.
    """
    summary: Dict[str, Any] = {
        "weight": None,
        "bias":   None,
        "head_weight_shape_old": None,
        "head_weight_shape_new": tuple(head_weight.shape),
    }

    weight_idx = _find_param_index(optimizer, head_weight)
    bias_idx = (
        _find_param_index(optimizer, head_bias) if head_bias is not None else None
    )

    state_dict = saved_state.get("state", {})

    if weight_idx is not None and weight_idx in state_dict:
        # Record the old tag-axis size for diagnostics
        for k in _TAG_AXIS_KEYS:
            t = state_dict[weight_idx].get(k)
            if torch.is_tensor(t):
                summary["head_weight_shape_old"] = tuple(t.shape)
                break
        summary["weight"] = _migrate_one_param_state(
            state_dict[weight_idx],
            tuple(head_weight.shape),
            old_tag_to_idx,
            new_tag_to_idx,
        )

    if (
        bias_idx is not None
        and bias_idx in state_dict
        and head_bias is not None
    ):
        summary["bias"] = _migrate_one_param_state(
            state_dict[bias_idx],
            tuple(head_bias.shape),
            old_tag_to_idx,
            new_tag_to_idx,
        )

    return summary
