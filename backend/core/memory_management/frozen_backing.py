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
