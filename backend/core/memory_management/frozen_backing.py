"""Retain immutable CPU parameter storage across temporary GPU residency."""

import os

import torch


def trim_mapped_pages(tensors):
    """Release only contiguous file-mapped CPU ranges from the Windows working set."""
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    class MemoryInfo(ctypes.Structure):
        _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                    ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD),
                    ("RegionSize", ctypes.c_size_t), ("State", wintypes.DWORD),
                    ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]

    library = ctypes.WinDLL("kernel32", use_last_error=True)
    query = library.VirtualQuery
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryInfo), ctypes.c_size_t]
    query.restype = ctypes.c_size_t
    unlock = library.VirtualUnlock
    unlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    unlock.restype = wintypes.BOOL
    seen, trimmed = set(), 0
    for tensor in tensors:
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or not tensor.numel():
            continue
        pointer, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        if (pointer, size) in seen:
            continue
        seen.add((pointer, size))
        info = MemoryInfo()
        if query(pointer, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
            continue
        if info.Type != 0x40000 or pointer + size > info.BaseAddress + info.RegionSize:
            continue
        # ERROR_NOT_LOCKED still removes mapped pages; bytes remain addressable.
        ctypes.set_last_error(0)
        if unlock(pointer, size) or ctypes.get_last_error() == 158:
            trimmed += size
    return trimmed


class FrozenCPUBacking:
    def __init__(self, modules):
        self.entries = []
        for module in modules:
            if module is None:
                continue
            parameters = dict(module.named_parameters())
            if any(p.device.type != "cpu" or p.requires_grad or p.grad is not None
                   for p in parameters.values()):
                raise ValueError("Frozen backing requires frozen CPU parameters without gradients")
            self.entries.append((module, parameters, {name: p.detach() for name, p in parameters.items()}))

    def _validate(self):
        for module, parameters, backing in self.entries:
            live = dict(module.named_parameters())
            if live.keys() != parameters.keys():
                raise RuntimeError("Frozen component parameter structure changed")
            for name, parameter in live.items():
                if (parameter is not parameters[name] or parameter.requires_grad or parameter.grad is not None
                        or parameter.dtype != backing[name].dtype or parameter.shape != backing[name].shape):
                    raise RuntimeError(f"Frozen component parameter changed: {name}")

    def stage(self, device):
        if torch.device(device).type == "cpu":
            return self.park()
        self._validate()
        for module, _, _ in self.entries:
            module.to(device)

    def park(self):
        self._validate()
        devices = {p.device for _, parameters, _ in self.entries for p in parameters.values()
                   if p.device.type == "cuda"}
        for device in devices:
            torch.cuda.synchronize(device)
        with torch.no_grad():
            for module, parameters, backing in self.entries:
                for name, parameter in parameters.items():
                    parameter.data = backing[name]
                # Runtime rotary buffers may change; preserve their current values.
                for buffer in module.buffers():
                    if buffer.device.type != "cpu":
                        buffer.data = buffer.to("cpu")
        return trim_mapped_pages(value for _, _, backing in self.entries for value in backing.values())


def install_frozen_cpu_offload(pipeline, component_names):
    """Preserve frozen storage when Diffusers rebuilds its model-offload hooks."""
    from functools import wraps
    from types import MethodType
    from accelerate.hooks import CpuOffload

    names = tuple(component_names)
    existing = getattr(pipeline, "_sushi_frozen_offload_names", None)
    if existing is not None:
        if existing != names:
            raise ValueError("Frozen offload component selection changed")
        return
    original = pipeline.enable_model_cpu_offload.__func__

    def park_hook(hook, module):
        hook._sushi_cpu_backing.park()
        return module

    @wraps(original)
    def enable(owner, *args, **kwargs):
        backings = owner._sushi_frozen_offload_backings
        for name in names:
            module = getattr(owner, name, None)
            if module is None:
                continue
            if name not in backings:
                module.eval().requires_grad_(False)
                backings[name] = FrozenCPUBacking((module,))
            elif backings[name].entries[0][0] is not module:
                raise RuntimeError(f"Frozen offload component replaced: {name}")
            backings[name].park()
        result = original(owner, *args, **kwargs)
        for name in getattr(owner, "_exclude_from_cpu_offload", ()):
            offloader = getattr(getattr(owner, name, None), "_block_offloader", None)
            if offloader is not None and getattr(offloader, "h2d_only", False):
                offloader.prepare_block_devices_before_forward()
        attached = set()
        for user_hook in owner._all_hooks:
            for name, backing in backings.items():
                if user_hook.model is backing.entries[0][0]:
                    hook = user_hook.hook
                    if not isinstance(hook, CpuOffload):
                        raise RuntimeError(f"Unsupported frozen offload hook: {name}")
                    hook._sushi_cpu_backing = backing
                    hook.init_hook = MethodType(park_hook, hook)
                    attached.add(name)
        if attached != backings.keys():
            raise RuntimeError("Frozen components are missing their CPU offload hooks")
        return result

    pipeline._sushi_frozen_offload_names = names
    pipeline._sushi_frozen_offload_backings = {}
    pipeline.enable_model_cpu_offload = MethodType(enable, pipeline)
