"""Private full-rank fusion of saved and coarse input/output features."""
from contextlib import contextmanager
from threading import Lock

import torch
from torch import nn
from torch.nn import functional as F

from core.rewiring.hybrid_modules import HybridModules, HybridExecution

_probe_lock = Lock()


class FeatureFusion(nn.Module):
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError('Feature fusion requires a positive channel width')
        self.weight = nn.Parameter(torch.zeros(width, 3*width, 1, 1))

    def forward(self, saved, low_in, low_out):
        if (saved.ndim != 4 or low_in.ndim != 4 or low_out.shape != low_in.shape
                or saved.shape[:2] != low_in.shape[:2] or saved.shape[1] != self.weight.shape[0]
                or saved.device != low_in.device or low_out.device != saved.device
                or not all(low_in.shape) or any(a < b for a,b in zip(saved.shape[-2:],low_in.shape[-2:]))):
            raise ValueError('Feature fusion requires matching fine/coarse channel grids')
        fields = [saved, F.interpolate(low_in, size=saved.shape[-2:], mode='bilinear', align_corners=False),
                  F.interpolate(low_out, size=saved.shape[-2:], mode='bilinear', align_corners=False)]
        return F.conv2d(torch.cat(fields,dim=1).to(self.weight.dtype), self.weight)


def build_fusions(modules):
    if not isinstance(modules, HybridModules):
        raise ValueError('Feature fusion requires hybrid modules')
    regions = {item['id']:item for item in modules.declaration['regions']}
    with torch.device('cpu'):
        result = nn.ModuleDict({item['id']:FeatureFusion(modules.declaration['stream_widths'][
            regions[item['region']]['stream']]) for item in modules.declaration['boundaries']})
    return result.to(next(modules.parameters()))


@contextmanager
def boundary_features_probe(modules, fusions):
    if not isinstance(modules, HybridModules) or not isinstance(fusions,nn.ModuleDict):
        raise ValueError('Feature fusion requires hybrid modules and named fusions')
    regions = {item['id']:item for item in modules.declaration['regions']}
    items = modules.declaration['boundaries']
    if (set(fusions) != {item['id'] for item in items}
            or hasattr(modules,'_boundary_features_probe')
            or getattr(HybridExecution,'boundary_features',None) is not None):
        raise ValueError('Feature fusion requires exclusive hook ownership')
    for item in items:
        width = modules.declaration['stream_widths'][regions[item['region']]['stream']]
        if not isinstance(fusions[item['id']],FeatureFusion) or fusions[item['id']].weight.shape != (width,3*width,1,1):
            raise ValueError('Feature fusion width differs from its boundary')
    if not _probe_lock.acquire(blocking=False):
        raise RuntimeError('Only one feature fusion probe may execute per process')
    existed = hasattr(HybridExecution,'boundary_features')
    original = getattr(HybridExecution,'boundary_features',None)
    registered = installed = False

    def fusion(execution,region,saved,low_in,low_out,update,height,width):
        if execution.modules is not modules:
            return None
        name = execution.boundaries.get(region['id'])
        if name is None:
            return None
        fine = execution.grid(saved,0,height,width)
        h,w = height//region['down_factor'][0],width//region['down_factor'][1]
        incoming,outgoing = [execution.grid(value,0,h,w) for value in (low_in,low_out)]
        module = fusions[name]
        if not execution.checkpoint_modules or not torch.is_grad_enabled():
            return module(fine,incoming,outgoing)
        from torch.utils.checkpoint import checkpoint
        if execution.checkpoint_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=fine.is_cuda):
                return checkpoint(module,fine,incoming,outgoing,use_reentrant=False)
        return checkpoint(module,fine,incoming,outgoing,use_reentrant=False)

    try:
        fusions.to(next(modules.parameters()))
        modules.add_module('_boundary_features_probe',fusions)
        registered = True
        HybridExecution.boundary_features = fusion
        installed = True
        yield fusions
    finally:
        if installed:
            if existed:
                HybridExecution.boundary_features = original
            else:
                del HybridExecution.boundary_features
        if registered:
            del modules._boundary_features_probe
        _probe_lock.release()
