"""Checkpoint-aware mutable offload for several logical branches per layer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Mapping, Sequence

import torch
from torch import nn

from .layer_offload_strategy import LayerOffloadStrategy
from .offload_transfer_engine import MutableLruTransferEngine


Branch = Hashable
Key = tuple[Branch, int]


class BranchedLayerOffloadConductor:
    """Stream branch-specific bundles from shared physical decoder layers.

    ``classify_module`` maps a module path inside one physical layer to its
    logical branch. ``resolve_call_branches`` maps one layer invocation to the
    branch or branches that invocation reads. The transfer/storage lifecycle is
    otherwise the same one used by :class:`LayerOffloadConductor`.
    """

    def __init__(
        self,
        *,
        root: nn.Module,
        layers: nn.ModuleList,
        branches: Sequence[Branch],
        classify_module: Callable[[str, nn.Module], Branch | None],
        resolve_call_branches: Callable[[tuple, Mapping[str, Any]], Sequence[Branch]],
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = True,
        enable_prefetch: bool = True,
        ring_size: int = 2,
        frozen_base_only: bool = False,
    ) -> None:
        self.root = root
        self.layers = layers
        self.num_layers = len(layers)
        self.branches = tuple(branches)
        self.classify_module = classify_module
        self.resolve_call_branches = resolve_call_branches
        self.blocks_to_swap = int(blocks_to_swap)
        self.device = torch.device(device)
        self.use_pinned_memory = bool(use_pinned_memory)
        self.enable_prefetch = bool(enable_prefetch)
        self.frozen_base_only = bool(frozen_base_only)
        self.enable_activation_offload = False
        if self.device.type != "cuda":
            raise ValueError("branched mutable block swap requires a CUDA device")

        self.strategy = LayerOffloadStrategy(
            num_layers=self.num_layers,
            blocks_to_swap=self.blocks_to_swap,
            device=self.device,
        )
        self.swappable_layer_indices = tuple(
            idx for idx in range(self.num_layers) if self.strategy.is_offloadable(idx)
        )
        if not self.swappable_layer_indices:
            raise ValueError("blocks_to_swap must select at least one layer")

        self._masters: Dict[Key, Dict[torch.dtype, torch.Tensor]] = {}
        self._layouts: Dict[
            Key, list[tuple[Any, str, torch.dtype, int, int, tuple[int, ...]]]
        ] = {}
        self._trainable: Dict[Key, tuple[nn.Parameter, ...]] = {}
        self._managed_tensor_ids: set[int] = set()
        self._calls: Dict[int, list[tuple[tuple[Key, ...], bool]]] = defaultdict(list)
        self._recompute_calls: Dict[int, list[tuple[Key, ...]]] = defaultdict(list)
        self._backward_seen: set[Key] = set()
        self._optimizer_hooks_ready = False
        self._closed = False
        self.hook_handles: list[Any] = []
        self.optimizer_hook_handles: list[Any] = []

        self._initialize_bundles()
        self.keys = tuple(
            (branch, idx)
            for branch in self.branches
            for idx in self.swappable_layer_indices
            if (branch, idx) in self._masters
        )
        if not self.keys:
            raise ValueError("branched block swap selected no persistent tensors")
        self.ring_size = max(1, min(int(ring_size), len(self.keys)))
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.engine = MutableLruTransferEngine(
            keys=self.keys,
            masters=self._masters,
            ring_size=self.ring_size,
            device=self.device,
            point_bundle=self._point_bundle,
            stream=self.transfer_stream,
        )
        self._stage_unmanaged_tensors()

    @staticmethod
    def _owned_tensors(module: nn.Module):
        for name, parameter in module.named_parameters(recurse=False):
            if parameter is not None:
                yield module, name, parameter
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                yield module, name, buffer

    def _pinned_empty(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(numel, dtype=dtype, device="cpu")
        if self.use_pinned_memory:
            try:
                tensor = tensor.pin_memory(device=self.device)
            except (RuntimeError, NotImplementedError):
                pass
        return tensor

    def _initialize_bundles(self) -> None:
        for layer_idx in self.swappable_layer_indices:
            layer = self.layers[layer_idx]
            records: Dict[Branch, list[tuple[Any, str, torch.Tensor]]] = defaultdict(list)
            seen: set[int] = set()
            for path, module in layer.named_modules():
                branch = self.classify_module(path, module)
                if branch is None:
                    continue
                if branch not in self.branches:
                    raise ValueError(f"unknown branch {branch!r} at layer {layer_idx}")
                if self.frozen_base_only and any(
                    parameter is not None and parameter.requires_grad
                    for parameter in module._parameters.values()
                ):
                    continue
                for owner, name, tensor in self._owned_tensors(module):
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    if self.frozen_base_only and isinstance(tensor, nn.Parameter) \
                            and tensor.requires_grad:
                        continue
                    records[branch].append((owner, name, tensor))

            for branch, tensors in records.items():
                plane_sizes: Dict[torch.dtype, int] = {}
                positioned = []
                for owner, name, tensor in tensors:
                    plane = tensor.dtype
                    offset = plane_sizes.get(plane, 0)
                    positioned.append((owner, name, tensor, plane, offset))
                    plane_sizes[plane] = offset + tensor.numel()
                if not plane_sizes:
                    continue
                planes = {
                    plane: self._pinned_empty(total, plane)
                    for plane, total in plane_sizes.items()
                }
                layout = []
                trainable = []
                for owner, name, tensor, plane, offset in positioned:
                    numel = tensor.numel()
                    shape = tuple(tensor.shape)
                    master = planes[plane][offset:offset + numel].view(shape)
                    master.copy_(tensor.detach(), non_blocking=False)
                    tensor.data = master
                    layout.append((owner, name, plane, offset, numel, shape))
                    self._managed_tensor_ids.add(id(tensor))
                    if isinstance(tensor, nn.Parameter) and tensor.requires_grad:
                        trainable.append(tensor)
                key = (branch, layer_idx)
                self._masters[key] = planes
                self._layouts[key] = layout
                self._trainable[key] = tuple(trainable)

    def _stage_unmanaged_tensors(self) -> None:
        """Move everything outside CPU masters without replacing Parameters."""
        seen: set[int] = set()
        for module in self.root.modules():
            for _owner, _name, tensor in self._owned_tensors(module):
                if id(tensor) in seen or id(tensor) in self._managed_tensor_ids:
                    continue
                seen.add(id(tensor))
                if tensor.device != self.device:
                    tensor.data = tensor.data.to(self.device)

    def _point_bundle(
        self, key: Key, bundle: Mapping[torch.dtype, torch.Tensor]
    ) -> None:
        for owner, name, plane, offset, numel, shape in self._layouts[key]:
            getattr(owner, name).data = bundle[plane][offset:offset + numel].view(shape)

    @staticmethod
    def _inside_backward() -> bool:
        getter = getattr(torch._C, "_current_graph_task_id", None)
        return bool(getter is not None and getter() >= 0)

    def _prefetch_neighbor(self, key: Key, *, backward: bool) -> None:
        if not self.enable_prefetch or self.ring_size < 2:
            return
        branch, layer_idx = key
        position = self.swappable_layer_indices.index(layer_idx)
        position += -1 if backward else 1
        if 0 <= position < len(self.swappable_layer_indices):
            neighbor = (branch, self.swappable_layer_indices[position])
            if neighbor in self._masters:
                self.engine.prefetch(neighbor)

    def _acquire(self, key: Key) -> None:
        self.engine.acquire(key)

    def _release_if_ready(self, key: Key) -> bool:
        if key not in self._backward_seen:
            return False
        params = self._trainable[key]
        if params and not all(parameter.grad is None for parameter in params):
            return False
        self.engine.release(key, dirty=bool(params))
        self._backward_seen.discard(key)
        return True

    def _try_release_ready(self) -> None:
        for key in tuple(self._backward_seen):
            self._release_if_ready(key)

    def register_hooks(self) -> None:
        if self.hook_handles:
            return
        for layer_idx in self.swappable_layer_indices:
            layer = self.layers[layer_idx]

            def pre_hook(module, args, kwargs, idx=layer_idx):
                del module
                branches = tuple(dict.fromkeys(self.resolve_call_branches(args, kwargs)))
                keys = tuple((branch, idx) for branch in branches)
                missing = [key for key in keys if key not in self._masters]
                if missing:
                    raise RuntimeError(
                        f"branched block swap call at layer {idx} requested "
                        f"unregistered bundle(s): {missing}"
                    )
                backward = self._inside_backward()
                for key in keys:
                    self._acquire(key)
                if len(keys) == 1:
                    self._prefetch_neighbor(keys[0], backward=backward)
                self._calls[idx].append((keys, backward))

            def post_hook(module, args, kwargs, output, idx=layer_idx):
                del module, args, kwargs
                if not self._calls[idx]:
                    raise RuntimeError(f"branched block swap lost layer {idx} call state")
                keys, backward = self._calls[idx].pop()
                if backward:
                    self._recompute_calls[idx].append(keys)
                else:
                    for key in keys:
                        self.engine.release(key, dirty=False)
                return output

            def backward_hook(module, grad_input, grad_output, idx=layer_idx):
                del module, grad_input, grad_output
                if not self._recompute_calls[idx]:
                    raise RuntimeError(
                        "branched block swap requires non-reentrant gradient "
                        f"checkpoint recomputation; layer {idx} reached backward without it"
                    )
                keys = self._recompute_calls[idx].pop()
                self._backward_seen.update(keys)
                for key in keys:
                    self._release_if_ready(key)

            self.hook_handles.append(
                layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            self.hook_handles.append(
                layer.register_forward_hook(post_hook, with_kwargs=True, always_call=True)
            )
            self.hook_handles.append(layer.register_full_backward_hook(backward_hook))

    def register_optimizer_hooks(self) -> None:
        if self._optimizer_hooks_ready:
            return
        for key in self.keys:
            for parameter in self._trainable[key]:
                def update_observer(tensor, bundle_key=key):
                    if tensor.grad is None:
                        self._release_if_ready(bundle_key)

                self.optimizer_hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(update_observer)
                )
        self._optimizer_hooks_ready = True

    def finish_backward(self) -> None:
        self._try_release_ready()
        if self._backward_seen:
            raise RuntimeError(
                "branched block swap has bundles awaiting optimizer update: "
                f"{sorted(self._backward_seen, key=str)}"
            )

    def clear_activations(self) -> None:
        self.finish_backward()

    def flush(self) -> None:
        self._try_release_ready()
        if self.engine.active:
            raise RuntimeError(
                "cannot flush branched offload during active bundles: "
                f"{sorted(self.engine.active, key=str)}"
            )
        self.engine.synchronize()

    def abort_step(self) -> None:
        for key in tuple(self.engine.active):
            self.engine.release(key, dirty=bool(self._trainable[key]))
        self.engine.synchronize()
        self._calls.clear()
        self._recompute_calls.clear()
        self._backward_seen.clear()

    def remove_hooks(self) -> None:
        for handle in (*self.hook_handles, *self.optimizer_hook_handles):
            handle.remove()
        self.hook_handles.clear()
        self.optimizer_hook_handles.clear()

    def cleanup(self) -> None:
        if self._closed:
            return
        self.flush()
        self.remove_hooks()
        self.engine.close()
        self._closed = True

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "blocks_to_swap": self.blocks_to_swap,
            "ring_size": self.ring_size,
            "branches": tuple(str(branch) for branch in self.branches),
            "transfers": self.engine.stats(),
            "gpu_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
        }
