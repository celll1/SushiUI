"""Stochastic rounding for BF16 parameter updates (shared by the ring-buffer optimizers).

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

The same pool supplies the FP32 gradient image. The 8-bit CUDA kernels require
``param.dtype == grad.dtype`` (``TORCH_CHECK`` in ``adamw8bit_cuda.cpp``, and
``Tensor::data_ptr<T>()`` in ``lion8bit_cuda.cpp``), while autograd produces
gradients in the parameter's own dtype, i.e. BF16. Handing an FP32 master and a
BF16 gradient to the kernel raises at the first step; ``prepare_master_and_grad``
is what keeps the pair consistent.
"""

from typing import Optional, Tuple

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

    __slots__ = ("_buffers",)

    def __init__(self) -> None:
        self._buffers = {}

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
