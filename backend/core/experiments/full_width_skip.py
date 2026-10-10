"""Private full-channel skip control; public declarations omit its activation cost."""
from contextlib import contextmanager

import torch
from torch import nn

from core.rewiring.hybrid_modules import HybridModules


class FullWidthSkip(nn.Module):
    def __init__(self, base, correction):
        super().__init__()
        self.base = base
        self.correction = correction
        self.rank = base.capture.out_channels

    def capture_grid(self, value):
        compressed = self.base.capture_grid(value)
        return torch.cat((compressed, value.to(compressed)), dim=1)

    def restore_grid(self, value):
        compressed, full = value[:, :self.rank], value[:, self.rank:]
        return self.base.restore_grid(compressed) + self.correction(full.to(self.correction.weight.dtype))


def build_corrections(modules):
    if not isinstance(modules, HybridModules):
        raise ValueError("Full-width probe requires hybrid modules")
    reference = next(modules.parameters())
    with torch.random.fork_rng(devices=[]):
        corrections = nn.ModuleDict()
        for name, module in modules.skips.items():
            width = module.capture.in_channels
            correction = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
            nn.init.zeros_(correction.weight)
            corrections[name] = correction
    return corrections.to(reference)


@contextmanager
def full_width_probe(modules, corrections):
    """Retain all source channels temporarily; never export as a public checkpoint."""
    if not isinstance(modules, HybridModules) or not isinstance(corrections, nn.ModuleDict):
        raise ValueError("Full-width probe requires hybrid modules and corrections")
    if set(corrections) != set(modules.skips) or any(isinstance(m, FullWidthSkip) for m in modules.skips.values()):
        raise ValueError("Full-width probe must own the installed skip IDs once")
    originals = dict(modules.skips.items())
    for name, base in originals.items():
        correction = corrections[name]
        width = base.capture.in_channels
        if (not isinstance(correction, nn.Conv2d) or correction.in_channels != width
                or correction.out_channels != width or correction.groups != width
                or correction.kernel_size != (3, 3) or correction.padding != (1, 1)
                or correction.stride != (1, 1) or correction.dilation != (1, 1)
                or correction.bias is not None or correction.padding_mode != 'zeros'):
            raise ValueError("Full-width probe requires channelwise 3x3 corrections")
    corrections.to(next(modules.parameters()))
    try:
        for name, base in originals.items():
            modules.skips[name] = FullWidthSkip(base, corrections[name])
        yield corrections
    finally:
        for name, base in originals.items():
            modules.skips[name] = base
