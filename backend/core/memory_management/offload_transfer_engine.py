"""Bounded asynchronous transfers for immutable host-resident tensor bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence

import torch


Plane = Hashable


@dataclass(frozen=True)
class TransferStats:
    h2d_bytes: int
    h2d_submissions: int
    acquire_misses: int
    consumer_waits: int


class FrozenSequentialTransferEngine:
    """Stream immutable CPU masters through a fixed set of GPU slots.

    ``masters[key]`` is a mapping of dtype/format plane to a flat CPU tensor.
    A plane is copied once per load, allowing quantized weights and their
    sidecars to retain their native dtypes without per-tensor DMA submissions.
    The caller owns tensor layouts and repoints them in ``point_bundle``.
    """

    def __init__(
        self,
        *,
        keys: Sequence[int],
        masters: Mapping[int, Mapping[Plane, torch.Tensor]],
        ring_size: int,
        device: torch.device,
        point_bundle: Callable[[int, Mapping[Plane, torch.Tensor]], None],
        stream=None,
    ) -> None:
        if not keys:
            raise ValueError("keys must not be empty")
        self.keys = tuple(keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys must be unique")
        self.position = {key: index for index, key in enumerate(self.keys)}
        self.masters = {key: dict(masters[key]) for key in self.keys}
        self.ring_size = max(1, min(int(ring_size), len(self.keys)))
        self.device = torch.device(device)
        self.cuda_available = self.device.type == "cuda"
        self.point_bundle = point_bundle
        self.stream = stream
        if self.cuda_available and self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device)

        planes = set()
        for bundle in self.masters.values():
            planes.update(bundle)
        if not planes:
            raise ValueError("masters must contain at least one tensor plane")

        self.slots: list[Dict[Plane, torch.Tensor]] = []
        for _ in range(self.ring_size):
            slot = {}
            for plane in planes:
                tensors = [bundle[plane] for bundle in self.masters.values() if plane in bundle]
                dtype = tensors[0].dtype
                if any(t.device.type != "cpu" for t in tensors):
                    raise ValueError("master tensors must be on CPU")
                if any(t.dtype != dtype for t in tensors):
                    raise ValueError(f"plane {plane!r} contains mixed dtypes")
                slot[plane] = torch.empty(
                    max(t.numel() for t in tensors), dtype=dtype, device=self.device
                )
            self.slots.append(slot)

        self.loaded_key: list[Optional[int]] = [None] * self.ring_size
        self.ready_event = [None] * self.ring_size
        self._h2d_bytes = 0
        self._h2d_submissions = 0
        self._acquire_misses = 0
        self._consumer_waits = 0

    def _slot_for(self, key: int) -> int:
        try:
            return self.position[key] % self.ring_size
        except KeyError as exc:
            raise KeyError(f"unregistered offload key: {key}") from exc

    def _views(self, key: int, slot: int) -> Dict[Plane, torch.Tensor]:
        return {
            plane: self.slots[slot][plane][:master.numel()]
            for plane, master in self.masters[key].items()
        }

    def submit_load(self, key: int, slot: Optional[int] = None) -> None:
        """Enqueue one logical bundle after work already on the compute stream."""
        slot = self._slot_for(key) if slot is None else slot
        if not 0 <= slot < self.ring_size:
            raise IndexError(f"slot {slot} outside ring of size {self.ring_size}")

        bundle = self.masters[key]
        if self.cuda_available:
            compute_done = torch.cuda.current_stream(self.device).record_event()
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(compute_done)
                for plane, master in bundle.items():
                    self.slots[slot][plane][:master.numel()].copy_(master, non_blocking=True)
                ready = self.stream.record_event()
        else:
            for plane, master in bundle.items():
                self.slots[slot][plane][:master.numel()].copy_(master)
            ready = None

        self.loaded_key[slot] = key
        self.ready_event[slot] = ready
        self._h2d_bytes += sum(t.numel() * t.element_size() for t in bundle.values())
        self._h2d_submissions += len(bundle)

    def prime(self) -> None:
        for slot, key in enumerate(self.keys[: self.ring_size]):
            self.submit_load(key, slot)

    def acquire(self, key: int) -> None:
        """Make ``key`` visible to the current stream without a host synchronize."""
        slot = self._slot_for(key)
        if self.loaded_key[slot] != key:
            self._acquire_misses += 1
            self.submit_load(key, slot)
        ready = self.ready_event[slot]
        if self.cuda_available and ready is not None:
            torch.cuda.current_stream(self.device).wait_event(ready)
            self._consumer_waits += 1
        self.point_bundle(key, self._views(key, slot))

    def release(self, key: int) -> None:
        """Release a slot and prefetch its next key, wrapping across iterations."""
        position = self.position[key]
        slot = position % self.ring_size
        self.point_bundle(key, self.masters[key])

        next_position = position + self.ring_size
        if next_position >= len(self.keys):
            next_position = slot
        self.submit_load(self.keys[next_position], slot)

    def synchronize(self) -> None:
        if self.cuda_available:
            self.stream.synchronize()

    def close(self) -> None:
        self.synchronize()
        for key in self.keys:
            self.point_bundle(key, self.masters[key])
        self.ready_event = [None] * self.ring_size
        self.loaded_key = [None] * self.ring_size
        self.slots.clear()

    def stats(self) -> TransferStats:
        return TransferStats(
            h2d_bytes=self._h2d_bytes,
            h2d_submissions=self._h2d_submissions,
            acquire_misses=self._acquire_misses,
            consumer_waits=self._consumer_waits,
        )


class FrozenLruTransferEngine(FrozenSequentialTransferEngine):
    """Order-agnostic immutable residency for checkpoint recomputation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.block_slot: Dict[int, int] = {}
        self.slot_block: list[Optional[int]] = [None] * self.ring_size
        self.lru = list(range(self.ring_size))

    def prime(self) -> None:
        # Training access order can differ between forward and recomputation.
        return

    def _touch(self, slot: int) -> None:
        self.lru.remove(slot)
        self.lru.append(slot)

    def acquire(self, key: int) -> None:
        try:
            slot = self.block_slot[key]
        except KeyError:
            self._acquire_misses += 1
            slot = self.lru[0]
            victim = self.slot_block[slot]
            if victim is not None:
                self.point_bundle(victim, self.masters[victim])
                del self.block_slot[victim]
            self.submit_load(key, slot)
            self.slot_block[slot] = key
            self.block_slot[key] = slot

        ready = self.ready_event[slot]
        if self.cuda_available and ready is not None:
            torch.cuda.current_stream(self.device).wait_event(ready)
            self._consumer_waits += 1
        self.point_bundle(key, self._views(key, slot))
        self._touch(slot)

    def release(self, key: int) -> None:
        # Immutable weights remain resident until selected as an LRU victim.
        return

    def close(self) -> None:
        super().close()
        self.block_slot.clear()
        self.slot_block = [None] * self.ring_size
        self.lru = list(range(self.ring_size))
