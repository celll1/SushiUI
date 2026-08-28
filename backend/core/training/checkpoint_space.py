"""Free-space policy for periodic checkpoint saves.

Run 121 (SenseNova both-branch full FT) died at step 39672 with 32.7 GiB free
and a 60.85 GiB checkpoint set (30.19 GiB of weight shards + a 30.66 GiB
optimizer ``.pt``). ``max_step_saves_to_keep`` was 2, and the trainer saves
BEFORE it prunes, so keep=N transiently needs N+1 sets of room -- ~182 GiB for
that run. The write failed mid-file and left a truncated optimizer state behind.

This module owns the arithmetic and the error classification; the filesystem
walk and the deletions stay in ``base_trainer`` next to the entry helpers they
share with resume detection.
"""

from __future__ import annotations

import errno
import math
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Sequence

GIB = 1024 ** 3

# A save needs more than the tensor bytes: safetensors writes a header, torch's
# zip writer pads records, and the volume must not be driven to zero free.
SPACE_HEADROOM_FRACTION = 0.05

# Retained-set floors, counting the set about to be written.
# After the write, one complete set may legitimately be all that is left.
KEEP_FLOOR_AFTER_WRITE = 1
# Before it, the newest set on disk is what the run would resume from, and the
# save that is about to replace it may itself fail. Never trade it for room.
KEEP_FLOOR_BEFORE_WRITE = 2

# State bytes per trainable scalar, used only for the FIRST save of a run (no
# previous save to measure). Both ring-buffer optimizers and bitsandbytes keep
# uint8 moments; Adafactor's factored second moment is O(rows+cols).
_OPTIMIZER_STATE_BYTES_PER_PARAM = {
    "adafactor": 1,
    "adafactor8bit": 1,
    "adamw8bit": 2,
    "adamw8bit_ringbuffer": 2,
    "lion8bit": 1,
    "lion8bit_ringbuffer": 1,
    "lion": 4,
}
# AdamW fp32 exp_avg + exp_avg_sq. Overestimating only prunes harder; the
# underestimate is the one that reproduces the incident.
_DEFAULT_OPTIMIZER_STATE_BYTES_PER_PARAM = 8

# ENOSPC reaches us through three unrelated writers, only one of which raises an
# OSError we can read an errno off:
#   * safetensors -> SafetensorError("... I/O error: <localized OS text> (os error 112)")
#   * torch.save  -> RuntimeError("[enforce fail at inline_container.cc:668] .
#                    unexpected pos X vs Y") -- the zip writer's short-write path
#   * plain open()/write -> OSError(errno 28 / winerror 112)
# The localized text is not matchable, so the numeric tail and the zip writer's
# signature are.
_DISK_FULL_MARKERS = (
    "os error 112",
    "os error 28",
    "no space left on device",
    "not enough space on the disk",
    "enforce fail at inline_container",
)


def is_disk_full_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a volume-out-of-space failure from any save writer."""
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, errno.EFBIG):
            return True
        if getattr(exc, "winerror", None) == 112:
            return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _DISK_FULL_MARKERS)


def free_bytes(path: Any) -> Optional[int]:
    """Free bytes on the volume holding ``path``; None if it cannot be read."""
    try:
        return int(shutil.disk_usage(str(path)).free)
    except Exception:
        return None


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    return f"{value / GIB:.2f} GiB"


def estimate_set_bytes(
    num_params: int,
    weight_bytes_per_param: int,
    optimizer_type: Optional[str],
) -> int:
    """Bytes one checkpoint set needs, from the parameter count.

    First-save fallback only -- a measured previous save is strictly better.
    Assumes the saved weights cover the same scalars the optimizer holds state
    for, which under-counts a full fine-tune that freezes part of the model.
    """
    per_param = _OPTIMIZER_STATE_BYTES_PER_PARAM.get(
        (optimizer_type or "").lower(), _DEFAULT_OPTIMIZER_STATE_BYTES_PER_PARAM
    )
    return int(num_params) * (int(weight_bytes_per_param) + per_param)


@dataclass(frozen=True)
class RetentionPlan:
    """What retention the next save can afford. ``effective_keep`` counts the
    set about to be written, so ``effective_keep - 1`` old sets survive it."""

    effective_keep: int
    requested_keep: int
    required_bytes: int
    free_bytes: int
    reclaim_bytes: int
    fits: bool
    # True when the save fits with the retention pass left where it has always
    # been (after the write). Pruning first is a small durability regression --
    # one complete set fewer is held while the new one is written -- so it is
    # done only when this is False.
    fits_as_is: bool = True

    @property
    def reduced(self) -> bool:
        return self.effective_keep != self.requested_keep

    @property
    def prune_first(self) -> bool:
        return (not self.fits_as_is) and self.reclaim_bytes > 0

    def describe(self, volume: str) -> str:
        return (
            f"free={format_bytes(self.free_bytes)}, "
            f"required={format_bytes(self.required_bytes)} "
            f"(+{int(SPACE_HEADROOM_FRACTION * 100)}% headroom) on {volume}"
        )


def plan_retention(
    free: Optional[int],
    required: int,
    set_sizes_newest_first: Sequence[int],
    requested_keep: int,
    floor: int = KEEP_FLOOR_BEFORE_WRITE,
) -> RetentionPlan:
    """Largest affordable keep count, never below ``floor``.

    ``requested_keep <= 0`` means "keep everything"; it is returned unchanged
    when everything still fits, and becomes a concrete count when it does not.
    A free-space reading of None (unreadable volume) keeps the request as-is:
    guessing is worse than the behaviour that shipped.
    """
    total = len(set_sizes_newest_first)
    keep_all = requested_keep is None or requested_keep <= 0
    ceiling = total + 1 if keep_all else int(requested_keep)
    requested_repr = 0 if keep_all else int(requested_keep)

    if free is None:
        return RetentionPlan(requested_repr, requested_repr, required, -1, 0, True)

    need = int(math.ceil(required * (1.0 + SPACE_HEADROOM_FRACTION)))
    fits_as_is = free >= need
    low = max(1, min(int(floor), ceiling))

    for keep in range(ceiling, low - 1, -1):
        survivors = min(max(keep - 1, 0), total)
        reclaim = int(sum(set_sizes_newest_first[survivors:]))
        if free + reclaim >= need:
            effective = requested_repr if (keep_all and keep == ceiling) else keep
            return RetentionPlan(
                effective, requested_repr, required, free,
                0 if fits_as_is else reclaim, True, fits_as_is,
            )

    survivors = min(max(low - 1, 0), total)
    reclaim = int(sum(set_sizes_newest_first[survivors:]))
    return RetentionPlan(low, requested_repr, required, free, reclaim, False, fits_as_is)


class CheckpointSaveSpaceError(RuntimeError):
    """A save that could not be made to fit, naming the numbers the raw OS
    error does not (it surfaces as a localized string inside SafetensorError)."""

    def __init__(self, step: int, volume: str, free: Optional[int], required: int, detail: str = ""):
        self.step = step
        self.volume = volume
        self.free_bytes = free
        self.required_bytes = required
        message = (
            f"Checkpoint save at step {step} ran out of disk space: "
            f"{format_bytes(free)} free on {volume}, "
            f"~{format_bytes(required)} required. Old checkpoints were pruned to "
            f"the retention floor and the save was retried once."
        )
        if detail:
            message += f" Last error: {detail}"
        super().__init__(message)
