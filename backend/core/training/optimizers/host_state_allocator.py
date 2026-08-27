"""Pinned host buffers for the ring-buffer optimizers' ``get_state_buffer``.

The two ring-buffer optimizers allocate their 8-bit state through an injected
``get_state_buffer(p, dtype=...)`` callable, falling back to GPU allocation when
none is supplied. Nothing ever supplied one (since 190c876e), so their host-state
mode -- the whole point of the name -- was unreachable in production. This module
is the supplier, and ``BaseTrainer._ringbuffer_optimizer_kwargs`` passes it.

Why NOT ``core.memory_management.RingBufferAllocator``, which
SENSENOVA_TRAINING_DESIGN.md 6.5 names as the intended base: that allocator hands
out *views into a recycled* byte buffer and exposes ``free_layer`` /
``start_offset`` / ``end_offset`` wrap-around, because it was written for layer
parameters that live for one forward/backward. Optimizer state lives for the
whole run and must never be recycled: two parameters whose state aliased the same
bytes would silently corrupt each other's moments. Optimizer state needs
persistent, non-overlapping, per-parameter buffers, which is what this allocator
gives.

Pinning happens HERE rather than in the optimizer. The optimizers do
``state[k] = self.get_state_buffer(...)`` and then ``state[k].pin_memory()`` for
CPU buffers -- and ``Tensor.pin_memory()`` on an unpinned tensor returns a NEW
pinned copy. An allocator that keeps a reference to the buffer it returned would
therefore hold the unpinned original alive alongside the pinned copy and double
the host RAM this route is budgeted for (G-RB2). Returning an already-pinned
buffer makes that ``pin_memory()`` a no-op that returns the same tensor, and the
accounting below deliberately stores byte counts, never the tensors.

Pinned memory is also what makes the mode work at all: the update kernels are
handed ``state['exp_avg']`` directly, and a pinned host allocation is addressable
from the device through UVA (measured at PCIe line rate, 8c13c493).
"""

from typing import Dict

import torch


class HostOptimizerStateAllocator:
    """Persistent, pinned, per-parameter host buffers for optimizer state.

    Call signature matches the two call sites in ``_init_param_state``:
    ``allocator(param, dtype=torch.uint8)`` -> flat buffer of ``param.numel()``.
    """

    def __init__(self, pin: bool = True) -> None:
        self.pin = bool(pin)
        self.bytes = 0
        self.tensors = 0
        self.pinned_bytes = 0

    def __call__(self, p: torch.Tensor, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        buffer = torch.zeros(p.numel(), dtype=dtype, device="cpu")
        if self.pin:
            # Allocated unpinned then pinned (one copy of a zero buffer at
            # allocation time only) because torch has no "allocate pinned"
            # constructor; the transient original is dropped on return.
            buffer = buffer.pin_memory()
            self.pinned_bytes += buffer.numel() * buffer.element_size()
        self.bytes += buffer.numel() * buffer.element_size()
        self.tensors += 1
        return buffer

    def summary(self) -> Dict[str, float]:
        return {
            "tensors": self.tensors,
            "bytes": self.bytes,
            "pinned_bytes": self.pinned_bytes,
            "gib": self.bytes / (1024 ** 3),
        }


def copy_containers_only(obj):
    """``deepcopy``'s container semantics without duplicating tensors.

    ``_load_state_dict_uint8`` deepcopies the incoming state dict so the caller's
    tensors are never aliased. Under host residency the loaded tensors are only
    ever ``copy_``d into buffers this optimizer owns, and the state is tens of
    GiB, so the deepcopy is a pure host-RAM doubling.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        return {k: copy_containers_only(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(copy_containers_only(v) for v in obj)
    return obj


def place_loaded_state_tensor(optimizer, param, key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Where a loaded optimizer-state tensor must land.

    ``absmax*`` are GPU-only (the update kernels index them there). Everything
    else, under host residency, goes INTO the pinned buffer this parameter
    already has: sending it to ``param.device`` puts the whole host budget on the
    GPU, and allocating a fresh host tensor doubles the pinned budget instead.
    """
    if key.startswith("absmax"):
        device = param.device if param.device.type == "cuda" else torch.device("cuda:0")
        return tensor.to(device)

    get_buffer = getattr(optimizer, "get_state_buffer", None)
    if get_buffer is None or tensor.dtype != torch.uint8:
        return tensor.to(param.device)

    existing = optimizer.state.get(param)
    buffer = existing.get(key) if isinstance(existing, dict) else None
    if not (isinstance(buffer, torch.Tensor)
            and buffer.dtype == tensor.dtype
            and buffer.numel() == tensor.numel()):
        buffer = get_buffer(param, dtype=tensor.dtype)
        if buffer.is_cpu and not buffer.is_pinned():
            buffer = buffer.pin_memory()
    buffer.copy_(tensor.reshape(buffer.shape))
    return buffer


def state_device_census(optimizer) -> Dict[str, Dict[str, int]]:
    """Where the optimizer's state tensors actually live, by key.

    The proof that host-state mode is ON is this census, not the presence of a
    ``get_state_buffer`` attribute: an allocator that returned CUDA tensors, or a
    key the optimizer allocates itself (absmax is deliberately GPU-resident),
    would leave the flag true and the bytes on the GPU.

    Returns ``{state_key: {"cpu": bytes, "cpu_pinned": bytes, "cuda": bytes}}``.
    """
    out: Dict[str, Dict[str, int]] = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            bucket = out.setdefault(key, {"cpu": 0, "cpu_pinned": 0, "cuda": 0})
            size = value.numel() * value.element_size()
            if value.is_cuda:
                bucket["cuda"] += size
            else:
                bucket["cpu"] += size
                if value.is_pinned():
                    bucket["cpu_pinned"] += size
    return out


def assert_state_host_resident(optimizer, keys=("exp_avg", "exp_avg_sq", "z")) -> Dict[str, Dict[str, int]]:
    """Raise unless the bulk state keys present are host-resident and pinned.

    ``absmax*`` are excluded by omission: the optimizers keep them on the GPU on
    purpose (adamw8bit_ringbuffer.py's "ALWAYS keep on GPU even if param moves to
    CPU"), and they are the 0.031250 B/param remainder in 6.5's table.
    """
    census = state_device_census(optimizer)
    problems = []
    for key in keys:
        bucket = census.get(key)
        if bucket is None:
            continue
        if bucket["cuda"]:
            problems.append(f"{key}: {bucket['cuda']} bytes on CUDA")
        if bucket["cpu"] and bucket["cpu_pinned"] != bucket["cpu"]:
            problems.append(
                f"{key}: {bucket['cpu'] - bucket['cpu_pinned']} of {bucket['cpu']} "
                f"host bytes are not pinned"
            )
    if problems:
        raise AssertionError(
            "Ring-buffer optimizer state is not host-resident as configured: "
            + "; ".join(problems)
            + f". Full census: {census}"
        )
    return census
