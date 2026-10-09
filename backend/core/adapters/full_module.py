"""Complete trainable module payloads accompanying low-rank branches."""
from collections.abc import Mapping

import torch
from torch import nn


class FullModulePayload(nn.Module):
    CHECKPOINT_WRITES_ALPHA = False

    def __init__(self, module):
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("Full payload requires a module")
        self.module = module
        self._shapes = {key: value.shape for key, value in module.state_dict().items()}
        if not self._shapes:
            raise ValueError("Full payload cannot be empty")
        if any(not value.is_floating_point() for value in module.state_dict().values()):
            raise ValueError("Full payload requires floating tensors")

    def branch_tensors(self):
        tensors = self.module.state_dict(keep_vars=True)
        if (set(tensors) != set(self._shapes)
                or any(value.shape != self._shapes[key] or not value.is_floating_point()
                       for key, value in tensors.items())):
            raise ValueError("Full payload module layout changed")
        return tensors

    def trainable_parameters(self):
        return (value for value in self.module.parameters() if value.requires_grad)

    def spec_constants(self):
        return ()

    def export_tensors(self):
        tensors = self.branch_tensors()
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("Full payload contains non-finite tensors")
        return {key: value.detach().to("cpu").contiguous().clone()
                for key, value in tensors.items()}

    def load_tensors(self, values):
        targets = self.branch_tensors()
        if not isinstance(values, Mapping) or set(values) != set(targets):
            raise ValueError("Full payload requires exactly its declared tensor keys")
        prepared = {}
        for key, target in targets.items():
            value = values[key]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid full payload tensor {key!r}")
            converted = value.detach().to(device=target.device, dtype=target.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError(f"Full payload tensor {key!r} overflows the target dtype")
            prepared[key] = converted.clone()
        snapshots = {key: value.detach().clone() for key, value in targets.items()}
        with torch.no_grad():
            try:
                for key, target in targets.items():
                    target.copy_(prepared[key])
            except Exception:
                for key, target in targets.items():
                    target.copy_(snapshots[key])
                raise
