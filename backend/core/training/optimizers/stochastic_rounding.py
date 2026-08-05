"""Stochastic rounding for BF16 parameter updates.

WHY THIS EXISTS
---------------
BF16 keeps an 8-bit significand, so the gap between representable neighbours
(one ULP) is between 2^-8 and 2^-7 of the value's magnitude. Round-to-nearest
discards any update smaller than half a ULP, and it does so deterministically:
a weight whose per-step update never reaches half a ULP is frozen at its
initial bit pattern for the whole run, no matter how many steps are taken.

For an Adam-family optimizer the per-element step is on the order of the
learning rate, so with round-to-nearest an element only ever moves when
``|w| <= 2^9 * lr``. At lr 1e-5 that is |w| <= 5.12e-3, which excludes most of
a DiT's weights.

Stochastic rounding replaces the tie-break with a coin flip weighted by the
fractional part, so ``E[round_stochastic(x)] == x``. Sub-ULP updates then
survive in expectation: an update of 0.01 ULP moves the element by a full ULP
on 1% of steps instead of never.

MEMORY
------
No persistent FP32 master weight is kept. Each optimizer step materialises an
FP32 image of the parameter in a scratch buffer that is reused by every
parameter of the optimizer, so the extra device memory is

    4 bytes * (numel of the LARGEST single parameter) * (number of slots)

not ``4 bytes * (all trainable parameters)``. A persistent master would cost
4 bytes per trained element (about 51 GB for a 12.8 B model) and would also
have to be written into every optimizer checkpoint.

Concretely, for Krea 2 the largest single tensor is ``time_mod_proj.weight``,
[36864, 6144] = 226.5 M elements, so its FP32 image is 864 MiB and the two slots
(master + gradient) come to **1.69 GiB** of extra device memory. Note that the
pool is per OPTIMIZER OBJECT, not global: under ``fused_optimizer_groups`` there
is one optimizer -- and therefore one pool -- per group, so multiply by the
group count. The gradient slot is only allocated for gradients that are not
already FP32 and contiguous.

The same pool supplies the FP32 gradient image, and it must: the 8-bit kernels
dispatch on the GRADIENT dtype and then read the parameter through a pointer of
that same type. Replacing only one of the pair silently reinterprets memory --
in the installed bitsandbytes build the dtype ``torch._check``s are commented
out (``backends/cuda/ops.py:698-717``), so nothing raises. This repo's own
extensions do check (``TORCH_CHECK`` in ``adamw8bit_cuda.cpp``, and
``Tensor::data_ptr<T>()`` in ``lion8bit_cuda.cpp``). Autograd produces gradients
in the parameter's own dtype, i.e. BF16, so ``prepare_master_and_grad`` is what
keeps the pair consistent.

REACHING OPTIMIZERS THAT DO NOT IMPLEMENT ANY OF THIS
-----------------------------------------------------
The two ring-buffer optimizers call the primitives above from inside their own
update. Every other optimizer a full fine-tune can select -- including
``adamw8bit``, the shipped default -- is third-party code that writes the
parameter itself. ``attach_stochastic_rounding`` covers those without touching
them, by interposing on whatever *per-parameter* seam they expose:

* ``bitsandbytes`` (``adamw8bit``, ``lion8bit``, every ``paged_*``) drives one
  parameter per call of ``Optimizer8bit.update_step(group, p, gindex, pindex)``.
* the fused-backward patches (``adafactor_fused``, ``adamw8bit_fused``) expose
  ``step_param(p, group)``, which Block Swap's post-accumulate-grad hooks call.

For the duration of one such call the parameter is made to *be* FP32:
``p.data`` is pointed at a pooled FP32 image and ``p.grad`` at an FP32 image of
the gradient, so the optimizer -- kernel or Python -- reads and writes FP32 and
never sees BF16 storage. The result is then written back with stochastic
rounding. This is a per-parameter interposition on purpose: making every
parameter FP32 for a whole ``step()`` would be the 4-bytes-per-element
persistent master that the MEMORY note above rules out.

An optimizer that updates all of its parameters inside one opaque call, with no
per-parameter entry point, cannot be covered this way -- ``torch.optim.AdamW``
(``optimizer: adamw``) is the one such optimizer selectable in the UI.
"""

import contextlib
from typing import Iterator, Optional, Tuple

import torch


def copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor) -> None:
    """Round FP32 ``source`` into BF16 ``target`` stochastically, in place.

    BF16 is the top 16 bits of an FP32 word. Adding a uniform random value to
    the discarded low 16 bits before truncating makes the carry into bit 16
    happen with probability equal to the fractional part, which is exactly an
    unbiased rounding: ``E[target] == source``.

    Reference: https://github.com/pytorch/pytorch/issues/120376

    Args:
        target: BF16 tensor, modified in place.
        source: FP32 tensor with the same shape.
    """
    if target.dtype != torch.bfloat16:
        raise ValueError(f"Target must be BF16, got {target.dtype}")
    if source.dtype != torch.float32:
        raise ValueError(f"Source must be FP32, got {source.dtype}")

    # Uniform random in the 16 discarded mantissa bits.
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # Add it to the FP32 bit pattern, then truncate to the top 16 bits.
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)  # 0xFFFF0000

    target.copy_(result.view(dtype=torch.float32))


class Fp32ScratchPool:
    """Reusable FP32 scratch buffers, one per (slot, device).

    A slot's buffer grows to the largest request it has seen and is then reused,
    so a training step allocates nothing after the first parameter of that size.
    See the module docstring for the memory bound.
    """

    __slots__ = ("_buffers", "in_use")

    def __init__(self) -> None:
        self._buffers = {}
        # Guards against two overlapping users of the same slot: the second
        # would be handed the same storage and the first's FP32 image would be
        # overwritten mid-update. Not reachable today (every interposition is
        # one parameter at a time and does not nest), but it corrupts silently
        # rather than failing, so it is checked rather than assumed.
        self.in_use = False

    def buffer(self, slot: str, like: torch.Tensor) -> torch.Tensor:
        """Return an uninitialised FP32 tensor shaped like ``like``, on its device."""
        key = (slot, like.device)
        n = like.numel()
        buf = self._buffers.get(key)
        if buf is None or buf.numel() < n:
            buf = torch.empty(n, dtype=torch.float32, device=like.device)
            self._buffers[key] = buf
        return buf[:n].view(like.shape)

    def copy_of(self, slot: str, source: torch.Tensor) -> torch.Tensor:
        """Return an FP32 copy of ``source`` held in this pool's ``slot`` buffer."""
        out = self.buffer(slot, source)
        out.copy_(source)
        return out

    def release(self) -> None:
        """Drop every buffer (frees the scratch memory)."""
        self._buffers.clear()


def should_use_stochastic_rounding(enabled: bool, param: torch.Tensor) -> bool:
    """Stochastic rounding only applies when writing into a BF16 parameter."""
    return bool(enabled) and param.dtype == torch.bfloat16


def prepare_master_and_grad(
    param: torch.Tensor,
    grad: torch.Tensor,
    pool: Fp32ScratchPool,
    master_slot: str = "master",
    grad_slot: str = "grad",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the FP32 (param, grad) pair the 8-bit kernels need for one step.

    The kernels dispatch on the parameter dtype and read the gradient as the
    same type, so both tensors must be FP32 here. The gradient is copied only
    when it is not already FP32 and contiguous.

    Args:
        param: The BF16 parameter being updated (not modified).
        grad: Its gradient, typically BF16.
        pool: Scratch pool that owns the buffers.

    Returns:
        (master, grad_fp32) -- ``master`` is the FP32 image of ``param`` that the
        kernel updates in place; write it back with ``copy_stochastic_bf16``.
    """
    master = pool.copy_of(master_slot, param)

    if grad.dtype == torch.float32 and grad.is_contiguous():
        grad_fp32 = grad
    else:
        grad_fp32 = pool.copy_of(grad_slot, grad)

    return master, grad_fp32


def stochastic_round_(param: torch.Tensor, master: Optional[torch.Tensor]) -> None:
    """Write ``master`` back into BF16 ``param`` with stochastic rounding (no-op if None)."""
    if master is None:
        return
    copy_stochastic_bf16(param, master)


# ---------------------------------------------------------------------------
# Interposing on optimizers that write BF16 parameters themselves
# ---------------------------------------------------------------------------

# Set on a per-parameter update function that already applies stochastic
# rounding internally, so ``attach_stochastic_rounding`` leaves it alone instead
# of rounding twice (see adamw8bit_fused.adamw8bit_step_param).
NATIVE_ATTR = "_sushiui_stochastic_rounding_native"

# Set on a function this module has already wrapped, so attaching twice (the
# trainer attaches after Block Swap may have re-patched ``step_param``) does not
# nest two interpositions.
WRAPPED_ATTR = "_sushiui_stochastic_rounding_wrapped"

_POOL_ATTR = "_sushiui_stochastic_rounding_pool"

# Reported by attach_stochastic_rounding() when the optimizer's own step_param
# applies stochastic rounding, so "covered natively" is distinguishable from
# "no seam" (an empty result).
NATIVE_STEP_PARAM = "step_param (native)"


@contextlib.contextmanager
def fp32_master_update(param: torch.Tensor, pool: Fp32ScratchPool) -> Iterator[bool]:
    """Make ``param`` an FP32 tensor for the body, then stochastically round back.

    Inside the ``with`` block ``param.data`` and ``param.grad`` are pooled FP32
    images, so an optimizer that updates ``param`` in place -- a CUDA kernel that
    dispatches on the gradient dtype, or Python that calls ``p.addcdiv_`` --
    accumulates in FP32. On exit the FP32 result is written into the original
    BF16 storage with stochastic rounding, and the original gradient is put back.

    Yields ``True`` when the interposition happened, ``False`` when it was
    skipped (non-BF16 parameter, or no gradient) and the body runs unchanged.

    The assignment order is not cosmetic: ``Tensor.grad``'s setter rejects a
    gradient whose dtype differs from the tensor's, so ``param.data`` has to
    become FP32 before ``param.grad`` does, and has to go back to BF16 only
    after ``param.grad`` has been released.
    """
    grad = param.grad
    if param.dtype != torch.bfloat16 or grad is None:
        yield False
        return

    if pool.in_use:
        raise RuntimeError(
            "fp32_master_update is already active on this scratch pool. The pool "
            "hands out one buffer per slot, so a nested update would overwrite the "
            "outer one's FP32 image of its parameter and lose that update silently. "
            "Give the inner update its own Fp32ScratchPool."
        )

    original = param.data
    master, grad_fp32 = prepare_master_and_grad(original, grad, pool)

    param.data = master
    param.grad = grad_fp32
    pool.in_use = True
    try:
        yield True
    finally:
        pool.in_use = False
        # The optimizer may have rebound param.data (bitsandbytes assigns
        # ``p.data = p.data.contiguous()``), so read it back rather than
        # assuming it is still ``master``.
        updated = param.data
        param.grad = None
        param.data = original
        copy_stochastic_bf16(original, updated)
        param.grad = grad


def _scratch_pool(optimizer) -> Fp32ScratchPool:
    """One scratch pool per optimizer, shared by every parameter it owns."""
    pool = getattr(optimizer, _POOL_ATTR, None)
    if pool is None:
        pool = Fp32ScratchPool()
        setattr(optimizer, _POOL_ATTR, pool)
    return pool


def _already_handled(fn) -> bool:
    return bool(getattr(fn, NATIVE_ATTR, False) or getattr(fn, WRAPPED_ATTR, False))


def attach_stochastic_rounding(optimizer) -> Tuple[str, ...]:
    """Interpose stochastic rounding on every per-parameter seam ``optimizer`` has.

    Returns the names of the methods that were wrapped. An empty tuple means the
    optimizer exposes no per-parameter entry point, so its BF16 updates are still
    round-to-nearest and the caller should say so rather than imply coverage.

    Idempotent: attaching again wraps nothing that is already wrapped.
    """
    pool = _scratch_pool(optimizer)
    wrapped = []

    step_param = getattr(optimizer, "step_param", None)
    native_step_param = callable(step_param) and getattr(step_param, NATIVE_ATTR, False)

    # bitsandbytes Optimizer8bit.update_step(group, p, gindex, pindex).
    # Skipped when a native step_param is installed: that is the block-swap
    # configuration, where every update goes through the hook and update_step is
    # never called, so wrapping it would only be dead weight to reason about.
    update_step = getattr(optimizer, "update_step", None)
    if callable(update_step) and not _already_handled(update_step) and not native_step_param:

        def sr_update_step(group, p, gindex, pindex, _inner=update_step):
            with fp32_master_update(p, pool):
                return _inner(group, p, gindex, pindex)

        setattr(sr_update_step, WRAPPED_ATTR, True)
        optimizer.update_step = sr_update_step
        wrapped.append("update_step")

    if native_step_param:
        # Reported so the caller can distinguish "covered by the optimizer
        # itself" from "no seam at all", which is what an empty result means.
        wrapped.append(NATIVE_STEP_PARAM)

    # Fused backward pass: step_param(p, group)
    if callable(step_param) and not _already_handled(step_param):

        def sr_step_param(p, group, _inner=step_param):
            with fp32_master_update(p, pool):
                return _inner(p, group)

        setattr(sr_step_param, WRAPPED_ATTR, True)
        optimizer.step_param = sr_step_param
        wrapped.append("step_param")

    return tuple(wrapped)
