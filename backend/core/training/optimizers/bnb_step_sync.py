"""Coalesce ordinary bitsandbytes updates without changing their kernels."""

from types import MethodType

import torch


@torch.no_grad()
def _step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    if not self.initialized:
        self.check_overrides()
        self.to_gpu()
        self.initialized = True
    devices = set()
    try:
        for gindex, group in enumerate(self.param_groups):
            for pindex, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue
                devices.add(parameter.device)
                if not self.state[parameter]:
                    self.init_state(group, parameter, gindex, pindex)
                self.prefetch_state(parameter)
                self.update_step(group, parameter, gindex, pindex)
    finally:
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "xpu":
                torch.xpu.synchronize(device)
    return loss


def coalesce_adamw8bit_step(optimizer):
    """Keep paged and per-parameter/fused update paths on their existing owners."""
    if type(optimizer).__name__ != "AdamW8bit" or optimizer.is_paged:
        return optimizer
    optimizer.step = MethodType(torch.optim.Optimizer.profile_hook_step(_step), optimizer)
    return optimizer
