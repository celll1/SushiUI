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
  - AdamW (FP32)        — exp_avg, exp_avg_sq
  - AdamW8bit (bnb)     — state1, state2, absmax1, absmax2, qmap1, qmap2, ...
  - Lion (FP32)         — exp_avg
  - Lion8bit (bnb)      — state1, absmax1, qmap1, ...

8-bit ``state1``/``state2`` are uint8-quantised values whose dequantised
form is ``qmap[state[i]] * absmax[block_idx]``.  When we re-arrange rows
by tag name, the per-block ``absmax`` is no longer aligned with the new
block boundaries, so we delete it and let bnb recompute on the next
``step()``.  This produces a brief, small magnitude perturbation but
preserves the directional information stored in ``state1``/``state2``.
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
        ``{"copied": int, "reset": int, "preserved": int}`` summary.
    """
    stats: Dict[str, int] = {"copied": 0, "reset": 0, "preserved": 0}
    if not new_shape:
        return stats
    new_num_tags = int(new_shape[0])

    for key in list(state.keys()):
        val = state[key]

        if key in _PRESERVE_KEYS:
            stats["preserved"] += 1
            continue

        if not torch.is_tensor(val):
            continue

        if key in _BLOCK_WISE_KEYS:
            # Block boundaries change with param numel; let bnb recompute.
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
