"""Hook-driven immutable module streaming for generation graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Mapping, Sequence

import torch
from torch import nn

from .offload_transfer_engine import FrozenSequentialTransferEngine


Key = Hashable
Layout = tuple[nn.Module, str, torch.dtype, int, int, tuple[int, ...]]


def _owned_tensors(module: nn.Module):
    for owner in module.modules():
        for name, parameter in owner.named_parameters(recurse=False):
            if parameter is not None:
                yield owner, name, parameter
        for name, buffer in owner.named_buffers(recurse=False):
            if buffer is not None:
                yield owner, name, buffer


class _FrozenBundleStore:
    def __init__(self, *, device: torch.device, use_pinned_memory: bool) -> None:
        self.device = torch.device(device)
        self.use_pinned_memory = bool(use_pinned_memory)
        self.masters: Dict[Key, Dict[torch.dtype, torch.Tensor]] = {}
        self.layouts: Dict[Key, list[Layout]] = {}
        self.managed_tensor_ids: set[int] = set()

    def _empty_cpu(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(numel, dtype=dtype, device="cpu")
        if self.use_pinned_memory:
            try:
                tensor = tensor.pin_memory(device=self.device)
            except (RuntimeError, NotImplementedError):
                pass
        return tensor

    def add(self, key: Key, tensors: Sequence[tuple[nn.Module, str, torch.Tensor]]) -> None:
        plane_sizes: Dict[torch.dtype, int] = {}
        positioned = []
        local_ids: set[int] = set()
        for owner, name, tensor in tensors:
            if id(tensor) in local_ids:
                continue
            if id(tensor) in self.managed_tensor_ids:
                raise ValueError(f"generation offload tensor is shared by multiple units: {key!r}")
            local_ids.add(id(tensor))
            offset = plane_sizes.get(tensor.dtype, 0)
            positioned.append((owner, name, tensor, tensor.dtype, offset))
            plane_sizes[tensor.dtype] = offset + tensor.numel()
        if not plane_sizes:
            return

        planes = {
            dtype: self._empty_cpu(numel, dtype)
            for dtype, numel in plane_sizes.items()
        }
        layout: list[Layout] = []
        for owner, name, tensor, dtype, offset in positioned:
            numel = tensor.numel()
            shape = tuple(tensor.shape)
            master = planes[dtype][offset:offset + numel].view(shape)
            master.copy_(tensor.detach(), non_blocking=False)
            tensor.data = master
            self.managed_tensor_ids.add(id(tensor))
            layout.append((owner, name, dtype, offset, numel, shape))
        self.masters[key] = planes
        self.layouts[key] = layout

    def point(self, key: Key, bundle: Mapping[torch.dtype, torch.Tensor]) -> None:
        for owner, name, dtype, offset, numel, shape in self.layouts[key]:
            getattr(owner, name).data = bundle[dtype][offset:offset + numel].view(shape)

    def stage_unmanaged(self, root: nn.Module) -> None:
        seen: set[int] = set()
        for owner, name, tensor in _owned_tensors(root):
            if id(tensor) in seen or id(tensor) in self.managed_tensor_ids:
                continue
            seen.add(id(tensor))
            if tensor.device != self.device:
                getattr(owner, name).data = tensor.data.to(self.device)


class FrozenModuleOffloadConductor:
    """Stream the trailing execution units of a forward-only module graph."""

    def __init__(
        self,
        *,
        root: nn.Module,
        modules: Sequence[nn.Module],
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        ring_size: int = 2,
    ) -> None:
        self.root = root
        self.modules = tuple(modules)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("generation block offload requires a CUDA device")
        if len(self.modules) < 2:
            raise ValueError("generation block offload requires at least two execution units")
        self.blocks_to_swap = max(0, min(int(blocks_to_swap), len(self.modules) - 1))
        if self.blocks_to_swap == 0:
            raise ValueError("blocks_to_swap must select at least one execution unit")
        first = len(self.modules) - self.blocks_to_swap
        self.keys = tuple(range(first, len(self.modules)))
        self.store = _FrozenBundleStore(
            device=self.device, use_pinned_memory=use_pinned_memory
        )
        for key in self.keys:
            self.store.add(key, list(_owned_tensors(self.modules[key])))
        self.keys = tuple(key for key in self.keys if key in self.store.masters)
        if not self.keys:
            raise ValueError("selected generation units contain no persistent tensors")
        self.store.stage_unmanaged(root)
        self.engine = FrozenSequentialTransferEngine(
            keys=self.keys,
            masters=self.store.masters,
            ring_size=max(1, int(ring_size)),
            device=self.device,
            point_bundle=self.store.point,
        )
        self.hook_handles: list[Any] = []
        self._depth: Dict[Key, int] = defaultdict(int)
        self._closed = False

    def register_hooks(self) -> None:
        if self.hook_handles:
            return
        for key in self.keys:
            module = self.modules[key]

            def pre_hook(_module, _args, unit=key):
                if self._depth[unit] == 0:
                    self.engine.acquire(unit)
                self._depth[unit] += 1

            def post_hook(_module, _args, output, unit=key):
                self._depth[unit] -= 1
                if self._depth[unit] == 0:
                    self.engine.release(unit)
                return output

            self.hook_handles.append(module.register_forward_pre_hook(pre_hook))
            self.hook_handles.append(
                module.register_forward_hook(post_hook, always_call=True)
            )
        self.engine.prime()

    def cleanup(self) -> None:
        if self._closed:
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.engine.close()
        self._depth.clear()
        self._closed = True

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "units": len(self.modules),
            "blocks_to_swap": self.blocks_to_swap,
            "ring_size": self.engine.ring_size,
            "transfers": self.engine.stats(),
        }


class FrozenBranchedLayerOffloadConductor:
    """Stream branch-specific bundles selected by one physical layer call."""

    def __init__(
        self,
        *,
        root: nn.Module,
        layers: Sequence[nn.Module],
        branches: Sequence[Hashable],
        classify_module: Callable[[str, nn.Module], Hashable | None],
        resolve_call_branches: Callable[[tuple, Mapping[str, Any]], Sequence[Hashable]],
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
    ) -> None:
        self.root = root
        self.layers = tuple(layers)
        self.branches = tuple(branches)
        self.resolve_call_branches = resolve_call_branches
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("generation branch offload requires a CUDA device")
        blocks = max(0, min(int(blocks_to_swap), len(self.layers) - 1))
        if blocks == 0:
            raise ValueError("blocks_to_swap must select at least one layer")
        self.blocks_to_swap = blocks
        self.layer_indices = tuple(range(len(self.layers) - blocks, len(self.layers)))
        self.store = _FrozenBundleStore(
            device=self.device, use_pinned_memory=use_pinned_memory
        )
        for layer_idx in self.layer_indices:
            records: Dict[Hashable, list[tuple[nn.Module, str, torch.Tensor]]] = defaultdict(list)
            seen: set[int] = set()
            for path, module in self.layers[layer_idx].named_modules():
                branch = classify_module(path, module)
                if branch is None:
                    continue
                if branch not in self.branches:
                    raise ValueError(f"unknown generation branch {branch!r}")
                for owner, name, tensor in _owned_tensors(module):
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    records[branch].append((owner, name, tensor))
            for branch in self.branches:
                self.store.add((branch, layer_idx), records.get(branch, ()))

        # Interleaving makes the two branches of one layer use distinct slots;
        # ring=2 also prefetches the next layer of the same branch.
        self.keys = tuple(
            (branch, layer_idx)
            for layer_idx in self.layer_indices
            for branch in self.branches
            if (branch, layer_idx) in self.store.masters
        )
        if not self.keys:
            raise ValueError("selected branch layers contain no persistent tensors")
        self.store.stage_unmanaged(root)
        self.engine = FrozenSequentialTransferEngine(
            keys=self.keys,
            masters=self.store.masters,
            ring_size=2,
            device=self.device,
            point_bundle=self.store.point,
        )
        self.hook_handles: list[Any] = []
        self._calls: Dict[int, list[tuple[Key, ...]]] = defaultdict(list)
        self._closed = False

    def register_hooks(self) -> None:
        if self.hook_handles:
            return
        for layer_idx in self.layer_indices:
            layer = self.layers[layer_idx]

            def pre_hook(_module, args, kwargs, idx=layer_idx):
                branches = tuple(dict.fromkeys(self.resolve_call_branches(args, kwargs)))
                keys = tuple((branch, idx) for branch in branches)
                if any(key not in self.store.masters for key in keys):
                    raise RuntimeError(f"unregistered generation branch at layer {idx}: {keys}")
                for key in keys:
                    self.engine.acquire(key)
                self._calls[idx].append(keys)

            def post_hook(_module, _args, _kwargs, output, idx=layer_idx):
                if not self._calls[idx]:
                    raise RuntimeError(f"generation branch offload lost layer {idx} call state")
                for key in self._calls[idx].pop():
                    self.engine.release(key)
                return output

            self.hook_handles.append(
                layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            self.hook_handles.append(
                layer.register_forward_hook(post_hook, with_kwargs=True, always_call=True)
            )
        self.engine.prime()

    def cleanup(self) -> None:
        if self._closed:
            return
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.engine.close()
        self._calls.clear()
        self._closed = True

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "layers": len(self.layers),
            "blocks_to_swap": self.blocks_to_swap,
            "ring_size": 2,
            "branches": tuple(str(branch) for branch in self.branches),
            "transfers": self.engine.stats(),
        }
