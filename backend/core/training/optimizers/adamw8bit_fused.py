"""AdamW8bit with a per-parameter update, for Block Swap's fused backward pass.

``register_post_accumulate_grad_hook`` calls ``step_param(p, group)`` while the
parameter is still on the GPU, before Block Swap moves it back to the CPU.

The update itself delegates to bitsandbytes' own
``Optimizer8bit.init_state`` / ``update_step``, which is exactly what its
``step()`` drives one parameter at a time. That keeps the real blockwise 8-bit
state (uint8 ``state1``/``state2`` + fp32 ``absmax``, measured 2.031 B/param) in
BOTH paths, so Block Swap no longer silently doubles optimizer state and a
checkpoint is portable between Block Swap on and off. The previous hand-written
AdamW here allocated ``zeros_like(p)`` moments: 4 B/param, and a state format
``step()`` cannot read.

Requirements: PyTorch 2.1+ (for the hook), bitsandbytes (for the update).
"""

import torch

from .stochastic_rounding import (
    NATIVE_ATTR,
    Fp32ScratchPool,
    fp32_master_update,
    should_use_stochastic_rounding,
)
from .update_census import record_param_update

_INDEX_ATTR = "_sushiui_bnb_param_index"


def _param_index(self, p):
    """The ``(gindex, pindex)`` bitsandbytes' ``step()`` would have passed.

    They are used for one thing only -- ``get_config`` looks up
    ``GlobalOptimManager.index2config[(gindex, pindex)]`` (optimizer.py:316) --
    so the hook has to reproduce ``step()``'s enumeration order. ``(-1, -1)``
    for a parameter that is in no group can never match an override.
    """
    index = getattr(self, _INDEX_ATTR, None)
    if index is None or id(p) not in index:
        index = {
            id(param): (gindex, pindex)
            for gindex, group in enumerate(self.param_groups)
            for pindex, param in enumerate(group["params"])
        }
        setattr(self, _INDEX_ATTR, index)
    return index.get(id(p), (-1, -1))


def _bnb_update(self, p, group):
    gindex, pindex = _param_index(self, p)
    state = self.state[p]

    if len(state) == 0:
        self.init_state(group, p, gindex, pindex)
    else:
        # Per-parameter form of Optimizer8bit.to_gpu(): a resumed state can be on
        # another device, and the global version would follow block-swapped
        # parameters onto the CPU, where the 8-bit kernels cannot run.
        stale = [
            key for key, value in state.items()
            if isinstance(value, torch.Tensor) and value.device != p.device
            and not getattr(value, "is_paged", False)
        ]
        for key in stale:
            state[key] = state[key].to(p.device)

    self.prefetch_state(p)
    # self.update_step, not the class method: any interposition rebinds the
    # instance attribute (see adafactor_fused.adafactor_step).
    self.update_step(group, p, gindex, pindex)

    if self.is_paged:
        from bitsandbytes.utils import sync_gpu
        sync_gpu(p)


@torch.no_grad()
def adamw8bit_step_param(self, p, group):
    """Update one parameter immediately after its gradient is ready."""
    if p.grad is None:
        return

    if not getattr(self, "initialized", False):
        # step()'s to_gpu() is deliberately skipped here; _bnb_update aligns the
        # state device per parameter instead.
        self.check_overrides()
        self.initialized = True

    if should_use_stochastic_rounding(getattr(self, "stochastic_rounding", False), p):
        pool = getattr(self, "_sr_pool", None)
        if pool is None:
            pool = Fp32ScratchPool()
            self._sr_pool = pool
        # The 8-bit kernels dispatch on the gradient dtype and read the parameter
        # through a pointer of that type, so both become FP32 for the call and the
        # result is rounded back stochastically. The state stays uint8: init_state
        # allocates it with an explicit dtype, not from the parameter's.
        with fp32_master_update(p, pool):
            _bnb_update(self, p, group)
    else:
        _bnb_update(self, p, group)

    # See adafactor_fused: setup_update_census arms the census for every
    # fused-backward optimizer, so every one of them has to report.
    record_param_update(self, p)


# Tells attach_stochastic_rounding() not to interpose on this function: it
# applies stochastic rounding itself, above.
setattr(adamw8bit_step_param, NATIVE_ATTR, True)


@torch.no_grad()
def adamw8bit_step(self, closure=None):
    """Batch update built from the per-parameter seam (fallback; not installed)."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            # self.step_param, not the module-level function: see the same note
            # in adafactor_fused.adafactor_step. Any interposition on the
            # per-parameter update rebinds the instance attribute, and calling
            # the module-level function bypasses it.
            self.step_param(p, group)

    return loss


def patch_adamw8bit_fused(optimizer, stochastic_rounding: bool = False):
    """Give a bitsandbytes AdamW8bit a ``step_param`` for the fused backward pass.

    Args:
        optimizer: ``bnb.optim.AdamW8bit`` instance (any ``Optimizer8bit`` with
            ``init_state``/``update_step`` works).
        stochastic_rounding: Write BF16 parameters with stochastic rounding
            instead of round-to-nearest. Only affects BF16 parameters.
    """
    missing = [
        name for name in ("init_state", "update_step", "get_config", "prefetch_state")
        if not callable(getattr(optimizer, name, None))
    ]
    if missing:
        raise TypeError(
            f"patch_adamw8bit_fused expects a bitsandbytes Optimizer8bit "
            f"(bnb.optim.AdamW8bit); {type(optimizer).__name__} is missing "
            f"{', '.join(missing)}. The per-parameter update delegates to "
            f"bitsandbytes so that Block Swap keeps 8-bit optimizer state."
        )

    optimizer.stochastic_rounding = bool(stochastic_rounding)
    optimizer.step_param = adamw8bit_step_param.__get__(optimizer)
    # Don't replace step() - keep bitsandbytes implementation for fallback
    print(f"[AdamW8bitFused] Optimizer patched with fused backward support "
          f"(delegates to bitsandbytes 8-bit state)"
          f"{' (stochastic rounding)' if optimizer.stochastic_rounding else ''}")
