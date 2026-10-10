"""Private channelwise control of retained detail at restored boundaries."""
from contextlib import contextmanager
from threading import Lock

import torch
from torch import nn
from torch.nn import functional as F

from core.rewiring.hybrid_modules import HybridExecution, HybridModules

_probe_lock = Lock()


class DetailGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("Detail gate requires a positive channel width")
        self.gain = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, saved, low):
        if (saved.ndim != 4 or low.ndim != 4 or saved.shape[:2] != low.shape[:2]
                or saved.shape[1] != self.gain.shape[1] or saved.device != low.device
                or not all(low.shape) or any(a < b for a,b in zip(saved.shape[-2:],low.shape[-2:]))):
            raise ValueError("Detail gate requires matching fine/coarse channel grids")
        detail = saved.float() - F.interpolate(low.float(), size=saved.shape[-2:], mode='bilinear', align_corners=False)
        return self.gain.float()*detail


def build_gates(modules):
    if not isinstance(modules, HybridModules):
        raise ValueError("Detail probe requires hybrid modules")
    regions = {item['id']:item for item in modules.declaration['regions']}
    return nn.ModuleDict({item['id']:DetailGate(modules.declaration['stream_widths'][
        regions[item['region']]['stream']]) for item in modules.declaration['boundaries']}).to(next(modules.parameters()))


@contextmanager
def boundary_detail_probe(modules, gates):
    """Temporary state only; public declarations do not describe these gains."""
    if not isinstance(modules,HybridModules) or not isinstance(gates,nn.ModuleDict):
        raise ValueError("Detail probe requires hybrid modules and gates")
    regions = {item['id']:item for item in modules.declaration['regions']}
    items = modules.declaration['boundaries']
    if set(gates) != {item['id'] for item in items} or hasattr(modules,'_boundary_detail_probe'):
        raise ValueError("Detail probe must own the installed boundary IDs once")
    for item in items:
        gate=gates[item['id']]
        width=modules.declaration['stream_widths'][regions[item['region']]['stream']]
        if not isinstance(gate,DetailGate) or gate.gain.shape != (1,width,1,1):
            raise ValueError("Detail probe width differs from its boundary")
    if not _probe_lock.acquire(blocking=False):
        raise RuntimeError("Only one detail probe may execute per process")
    existed = hasattr(HybridExecution,'boundary_fusion')
    original = getattr(HybridExecution,'boundary_fusion',None)
    registered=False
    installed=False

    def fusion(execution,region,saved,low_in,update,height,width):
        if execution.modules is not modules:
            return original(execution,region,saved,low_in,update,height,width) if original is not None else None
        name=execution.boundaries.get(region['id'])
        if name is None:
            return None
        saved_grid=execution.grid(saved,0,height,width)
        low_grid=execution.grid(low_in,0,height//region['down_factor'][0],width//region['down_factor'][1])
        gate=gates[name]
        if not execution.checkpoint_modules or not torch.is_grad_enabled():
            return gate(saved_grid,low_grid)
        from torch.utils.checkpoint import checkpoint
        if execution.checkpoint_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=saved_grid.is_cuda):
                return checkpoint(gate,saved_grid,low_grid,use_reentrant=False)
        return checkpoint(gate,saved_grid,low_grid,use_reentrant=False)

    try:
        gates.to(next(modules.parameters()))
        modules.add_module('_boundary_detail_probe',gates)
        registered=True
        HybridExecution.boundary_fusion=fusion
        installed=True
        yield gates
    finally:
        if installed:
            if existed:
                HybridExecution.boundary_fusion=original
            else:
                del HybridExecution.boundary_fusion
        if registered:
            del modules._boundary_detail_probe
        _probe_lock.release()
