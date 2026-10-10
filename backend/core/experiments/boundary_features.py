"""Private full-rank fusion of saved and coarse input/output features."""
from contextlib import contextmanager
from threading import Lock

import torch
from torch import nn
from core.rewiring.hybrid_modules import HybridModules, HybridExecution, FullFeatureFusion

_probe_lock = Lock()
_public_hook = HybridExecution.boundary_features


FeatureFusion = FullFeatureFusion


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
            or modules.features
            or getattr(HybridExecution,'boundary_features',None) not in (None,_public_hook)):
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
            return original(execution,region,saved,low_in,low_out,update,height,width) if original is not None else None
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
