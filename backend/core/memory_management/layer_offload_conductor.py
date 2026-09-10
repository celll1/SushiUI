"""Checkpoint-aware mutable layer offload backed by the shared transfer engine."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
import torch.nn as nn

from .layer_offload_strategy import LayerOffloadStrategy
from .offload_transfer_engine import MutableLruTransferEngine


class LayerOffloadConductor:
    """Keep mutable block weights in persistent CPU masters and fixed GPU slots.

    The initial checkpointed forward releases a clean slot immediately. During
    checkpoint recomputation the slot remains active until the layer's optimizer
    hooks have applied its update, then the updated bundle is written back.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = True,
        cpu_buffer_size_mb: int = 8192,
        activation_buffer_size_mb: int = 4096,
        enable_prefetch: bool = True,
        enable_activation_offload: bool = False,
        ring_size: int = 2,
    ) -> None:
        del cpu_buffer_size_mb, activation_buffer_size_mb
        self.layers = layers
        self.num_layers = len(layers)
        self.blocks_to_swap = int(blocks_to_swap)
        self.device = torch.device(device)
        self.use_pinned_memory = bool(use_pinned_memory)
        self.enable_prefetch = bool(enable_prefetch)
        self.enable_activation_offload = False
        if enable_activation_offload:
            raise ValueError(
                "LayerOffloadConductor activation offload was unreachable; use "
                "activation_dispatch_enable for saved-tensor offload"
            )
        if self.device.type != "cuda":
            raise ValueError("mutable block swap requires a CUDA device")

        self.strategy = LayerOffloadStrategy(
            num_layers=self.num_layers,
            blocks_to_swap=self.blocks_to_swap,
            device=self.device,
        )
        self.swappable = tuple(
            idx for idx in range(self.num_layers) if self.strategy.is_offloadable(idx)
        )
        if not self.swappable:
            raise ValueError("blocks_to_swap must select at least one layer")
        self.ring_size = max(1, min(int(ring_size), len(self.swappable)))
        self.transfer_stream = torch.cuda.Stream(device=self.device)

        self._masters: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        self._layouts: Dict[int, list[tuple[Any, str, torch.dtype, int, int, tuple[int, ...]]]] = {}
        self._trainable: Dict[int, tuple[nn.Parameter, ...]] = {}
        self._backward_seen: set[int] = set()
        self._recompute_seen: set[int] = set()
        self._optimizer_hooks_ready = False
        self._closed = False
        self.hook_handles: List[Any] = []
        self.optimizer_hook_handles: List[Any] = []
        self.saved_activations: Dict[int, Any] = {}
        self.layer_cpu_copies: Dict[int, Dict[str, torch.Tensor]] = {}

        self._initialize_layers()
        self.engine = MutableLruTransferEngine(
            keys=self.swappable,
            masters=self._masters,
            ring_size=self.ring_size,
            device=self.device,
            point_bundle=self._point_bundle,
            stream=self.transfer_stream,
        )
        self.layer_states: Dict[int, str] = {
            idx: ("cpu" if idx in self.swappable else "gpu")
            for idx in range(self.num_layers)
        }

    @staticmethod
    def _owned_tensors(layer: nn.Module):
        seen = set()
        for module in layer.modules():
            for name, tensor in module.named_parameters(recurse=False):
                if id(tensor) not in seen:
                    seen.add(id(tensor))
                    yield module, name, tensor
            for name, tensor in module.named_buffers(recurse=False):
                if tensor is not None and id(tensor) not in seen:
                    seen.add(id(tensor))
                    yield module, name, tensor

    def _pinned_empty(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(numel, dtype=dtype, device="cpu")
        if self.use_pinned_memory:
            try:
                tensor = tensor.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass
        return tensor

    def _initialize_layers(self) -> None:
        print("[LayerOffloadConductor] Initializing shared mutable transfer engine...")
        for idx, layer in enumerate(self.layers):
            if idx not in self.swappable:
                layer.to(self.device)
                continue

            tensors = list(self._owned_tensors(layer))
            plane_sizes: Dict[torch.dtype, int] = {}
            records = []
            for owner, name, tensor in tensors:
                plane = tensor.dtype
                offset = plane_sizes.get(plane, 0)
                records.append((owner, name, tensor, plane, offset))
                plane_sizes[plane] = offset + tensor.numel()
            if not plane_sizes:
                raise ValueError(f"offloaded layer {idx} contains no parameters or buffers")

            planes = {
                plane: self._pinned_empty(total, plane)
                for plane, total in plane_sizes.items()
            }
            layout = []
            cpu_named = {}
            parameter_names = {id(p): name for name, p in layer.named_parameters()}
            for owner, name, tensor, plane, offset in records:
                numel = tensor.numel()
                shape = tuple(tensor.shape)
                master = planes[plane][offset:offset + numel].view(shape)
                master.copy_(tensor.detach(), non_blocking=False)
                tensor.data = master
                layout.append((owner, name, plane, offset, numel, shape))
                if id(tensor) in parameter_names:
                    cpu_named[parameter_names[id(tensor)]] = master

            self._masters[idx] = planes
            self._layouts[idx] = layout
            self._trainable[idx] = tuple(p for p in layer.parameters() if p.requires_grad)
            self.layer_cpu_copies[idx] = cpu_named

        self.strategy.print_strategy()
        print(
            f"[LayerOffloadConductor] Shared mutable engine ready: "
            f"{len(self.swappable)} layers, {self.ring_size} GPU slots"
        )

    def _point_bundle(
        self, layer_idx: int, bundle: Mapping[torch.dtype, torch.Tensor]
    ) -> None:
        for owner, name, plane, offset, numel, shape in self._layouts[layer_idx]:
            getattr(owner, name).data = bundle[plane][offset:offset + numel].view(shape)

    @staticmethod
    def _inside_backward() -> bool:
        getter = getattr(torch._C, "_current_graph_task_id", None)
        return bool(getter is not None and getter() >= 0)

    def _prefetch_neighbor(self, layer_idx: int, *, backward: bool) -> None:
        if not self.enable_prefetch or self.ring_size < 2:
            return
        position = self.swappable.index(layer_idx)
        position += -1 if backward else 1
        if 0 <= position < len(self.swappable):
            self.engine.prefetch(self.swappable[position])

    def _acquire(self, layer_idx: int) -> None:
        backward = self._inside_backward()
        self.engine.acquire(layer_idx)
        self.layer_states[layer_idx] = "gpu"
        if backward:
            self._recompute_seen.add(layer_idx)
        self._prefetch_neighbor(layer_idx, backward=backward)

    def _after_forward(self, layer_idx: int) -> None:
        if self._inside_backward():
            return
        self.engine.release(layer_idx, dirty=False)
        self.layer_states[layer_idx] = "cpu"

    def _all_updates_applied(self, layer_idx: int) -> bool:
        params = self._trainable[layer_idx]
        return not params or all(param.grad is None for param in params)

    def _release_if_ready(self, layer_idx: int) -> bool:
        if layer_idx not in self._backward_seen or not self._all_updates_applied(layer_idx):
            return False
        self.engine.release(layer_idx, dirty=bool(self._trainable[layer_idx]))
        self.layer_states[layer_idx] = "cpu"
        self._backward_seen.discard(layer_idx)
        self._recompute_seen.discard(layer_idx)
        return True

    def _try_release_ready_layers(self) -> None:
        for layer_idx in tuple(self._backward_seen):
            self._release_if_ready(layer_idx)

    def register_hooks(self) -> None:
        if self.hook_handles:
            return
        for layer_idx in self.swappable:
            layer = self.layers[layer_idx]

            def pre_hook(module, inputs, idx=layer_idx):
                self._acquire(idx)

            def post_hook(module, inputs, output, idx=layer_idx):
                self._after_forward(idx)

            def backward_hook(module, grad_input, grad_output, idx=layer_idx):
                if idx not in self._recompute_seen:
                    raise RuntimeError(
                        "mutable block swap requires non-reentrant gradient "
                        f"checkpoint recomputation; layer {idx} reached backward without it"
                    )
                self._backward_seen.add(idx)
                self._release_if_ready(idx)

            self.hook_handles.append(layer.register_forward_pre_hook(pre_hook))
            self.hook_handles.append(layer.register_forward_hook(post_hook, always_call=True))
            self.hook_handles.append(layer.register_full_backward_hook(backward_hook))
        print(f"[LayerOffloadConductor] Registered {len(self.hook_handles)} lifecycle hooks")

    def register_optimizer_hooks(self) -> None:
        """Register after fused optimizer hooks so cleared grads mean updated weights."""
        if self._optimizer_hooks_ready:
            return
        for layer_idx in self.swappable:
            for parameter in self._trainable[layer_idx]:
                def update_observer(tensor, idx=layer_idx):
                    if tensor.grad is None:
                        self._try_release_ready_layers()

                self.optimizer_hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(update_observer)
                )
        self._optimizer_hooks_ready = True

    def finish_backward(self) -> None:
        """Release groups stepped after autograd and reject stale GPU layers."""
        self._try_release_ready_layers()
        if self._backward_seen:
            pending = sorted(self._backward_seen)
            raise RuntimeError(
                f"mutable block swap has layers awaiting optimizer update: {pending}"
            )

    def flush(self) -> None:
        """Make all CPU masters current before save or teardown."""
        self._try_release_ready_layers()
        if self.engine.active:
            raise RuntimeError(
                f"cannot flush mutable offload during active layers: {sorted(self.engine.active)}"
            )
        self.engine.synchronize()

    def abort_step(self) -> None:
        """Recover slot ownership after an abandoned backward on a live context."""
        for layer_idx in tuple(self.engine.active):
            self.engine.release(layer_idx, dirty=bool(self._trainable[layer_idx]))
            self.layer_states[layer_idx] = "cpu"
        self.engine.synchronize()
        self._backward_seen.clear()
        self._recompute_seen.clear()

    def load_layer_to_gpu(self, layer_idx: int, async_transfer: bool = True) -> None:
        del async_transfer
        if layer_idx in self.swappable:
            self.engine.acquire(layer_idx)
            self.layer_states[layer_idx] = "gpu"

    def offload_layer_to_cpu(self, layer_idx: int, async_transfer: bool = True) -> None:
        del async_transfer
        if layer_idx in self.swappable and layer_idx in self.engine.active:
            self.engine.release(layer_idx, dirty=bool(self._trainable[layer_idx]))
            self.layer_states[layer_idx] = "cpu"

    def sync_layer(self, layer_idx: int) -> None:
        del layer_idx

    def clear_activations(self) -> None:
        self.saved_activations.clear()
        self.finish_backward()

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
        state_counts = {"cpu": 0, "gpu": 0, "loading": 0, "offloading": 0}
        for state in self.layer_states.values():
            state_counts[state] += 1
        stats = {
            "num_layers": self.num_layers,
            "blocks_to_swap": self.blocks_to_swap,
            "ring_size": self.ring_size,
            "layer_states": dict(self.layer_states),
            "state_counts": state_counts,
            "transfers": self.engine.stats(),
            "gpu_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "gpu_reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
        }
        return stats

    def print_memory_stats(self) -> None:
        stats = self.get_memory_stats()
        print("=" * 60)
        print("[LayerOffloadConductor] Memory Statistics")
        print(
            f"  Layer States: CPU={stats['state_counts']['cpu']}, "
            f"GPU={stats['state_counts']['gpu']}"
        )
        print(f"  GPU Allocated: {stats['gpu_allocated_mb']:.2f} MB")
        print(f"  GPU Reserved:  {stats['gpu_reserved_mb']:.2f} MB")
        print(f"  Transfers: {stats['transfers']}")
        print("=" * 60)
