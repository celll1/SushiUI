"""Private phase-separated boundary branches; public declarations omit their weights."""
from contextlib import contextmanager
from threading import Lock

import torch
from torch import nn
from torch.nn import functional as F

from core.rewiring.hybrid_modules import HybridExecution, HybridModules

_probe_lock = Lock()


class PhaseDown(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if type(width) is not int or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError("Phase boundary requires positive width and bounded rank")
        self.norm = nn.LayerNorm(width)
        self.reduce = nn.Conv2d(4*width, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, width, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.norm.normalized_shape[0] or any(x < 2 or x % 2 for x in value.shape[-2:]):
            raise ValueError("Phase down requires an even image grid with matching channels")
        value = value.to(self.reduce.weight.dtype)
        normalized = self.norm(value.movedim(1, -1)).movedim(-1, 1)
        return self.expand(F.silu(self.reduce(F.pixel_unshuffle(normalized, 2))))


class PhaseUp(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if type(width) is not int or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError("Phase boundary requires positive width and bounded rank")
        self.reduce = nn.Conv2d(width, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, 4*width, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.reduce.in_channels or not all(value.shape):
            raise ValueError("Phase up requires a nonempty coarse grid with matching channels")
        return F.pixel_shuffle(self.expand(F.silu(self.reduce(value.to(self.reduce.weight.dtype)))), 2)


class _DownBranch(nn.Module):
    def __init__(self, base, phase):
        super().__init__()
        self.base, self.phase = base, phase

    def forward(self, value):
        return self.base(value) + self.phase(value)


def build_branches(modules, rank):
    if not isinstance(modules, HybridModules) or hasattr(modules, '_phase_up_probe'):
        raise ValueError("Phase probe requires hybrid modules")
    reference = next(modules.parameters())
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        down = nn.ModuleDict({name: PhaseDown(module.spatial.in_channels, rank) for name,module in modules.down.items()})
        up = nn.ModuleDict({name: PhaseUp(module.spatial.in_channels, rank) for name,module in modules.up.items()})
    return down.to(reference), up.to(reference)


@contextmanager
def phase_boundary_probe(modules, down, up):
    """Temporary root-specific branches, including checkpoint replay ownership."""
    if (not isinstance(modules, HybridModules) or not isinstance(down, nn.ModuleDict)
            or not isinstance(up, nn.ModuleDict) or set(down) != set(modules.down)
            or set(up) != set(modules.up) or hasattr(modules, '_phase_up_probe')):
        raise ValueError("Phase probe must own the installed boundaries once")
    originals = dict(modules.down.items())
    for name, base in originals.items():
        width = base.spatial.in_channels
        if (not isinstance(down[name], PhaseDown) or not isinstance(up[name], PhaseUp)
                or down[name].norm.normalized_shape != (width,) or up[name].reduce.in_channels != width):
            raise ValueError("Phase branch width differs from the declared boundary")
    if not _probe_lock.acquire(blocking=False):
        raise RuntimeError("Only one phase probe may execute per process")
    existed = hasattr(HybridExecution, 'up_from_coarse')
    original_hook = getattr(HybridExecution, 'up_from_coarse', None)
    installed = False
    registered = False
    region_names = {item['region']:item['id'] for item in modules.declaration['boundaries']}
    try:
        reference = next(modules.parameters())
        down.to(reference); up.to(reference)
        modules.add_module('_phase_up_probe', up); registered = True
        for name, base in originals.items():
            modules.down[name] = _DownBranch(base, down[name])
        def phase_up(execution, region, coarse, fine):
            if execution.modules is not modules:
                return original_hook(execution, region, coarse, fine) if original_hook is not None else None
            name = region_names.get(region['id'])
            return None if name is None else execution.call(up[name], coarse).to(fine)
        HybridExecution.up_from_coarse = phase_up; installed = True
        yield down, up
    finally:
        for name, base in originals.items():
            modules.down[name] = base
        if registered:
            del modules._phase_up_probe
        if installed:
            if existed:
                HybridExecution.up_from_coarse = original_hook
            else:
                del HybridExecution.up_from_coarse
        _probe_lock.release()
