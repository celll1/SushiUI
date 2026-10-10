"""Exclusive, temporary concat correction integration for standalone probes."""
from contextlib import contextmanager
from threading import Lock

import torch
from torch import nn

from core.rewiring.hybrid_modules import HybridModules, HybridExecution
from core.rewiring.hybrid_spec import skip_storage
from .concat_fusion import CompressedConcatCorrection

_probe_lock = Lock()


def build_corrections(modules):
    reference = next(modules.parameters())
    with torch.random.fork_rng(devices=[]):
        return nn.ModuleDict({item['id']: CompressedConcatCorrection(
            modules.declaration['stream_widths'][item['stream']], item['rank'])
            for item in modules.declaration['skips']}).to(reference)


@contextmanager
def concat_probe(modules, corrections):
    """Private integration only; its extra weights have no public topology declaration."""
    if not isinstance(modules, HybridModules) or not isinstance(corrections, nn.ModuleDict):
        raise ValueError("Concat probe requires hybrid modules and declared corrections")
    items = modules.declaration['skips']
    if set(corrections) != {item['id'] for item in items} or hasattr(modules, '_concat_probe_corrections'):
        raise ValueError("Concat probe must own exactly the installed skip IDs")
    for item in items:
        correction = corrections[item['id']]
        if (not isinstance(correction, CompressedConcatCorrection)
                or correction.width != modules.declaration['stream_widths'][item['stream']]
                or correction.rank != item['rank']):
            raise ValueError("Concat probe dimensions differ from the compressed skips")
    if not _probe_lock.acquire(blocking=False):
        raise RuntimeError("Only one concat probe may execute per process")
    original = HybridExecution.before

    def before(execution, index, hidden, prefix, height, width):
        if execution.modules is not modules:
            return original(execution, index, hidden, prefix, height, width)
        destinations = [item for item in items if item['stream'] == execution.stream
                        and item['destination_before'] == index]
        if not destinations:
            return hidden
        grid = execution.grid(hidden, prefix, height, width)
        for item in destinations:
            module, correction = modules.skips[item['id']], corrections[item['id']]
            captured = (execution.replay(item) if execution.replay is not None and skip_storage(item) == 'recompute'
                        else execution.saved.pop(item['id']))
            device = grid.device

            def fuse(current, compressed, module=module, correction=correction, device=device):
                compressed = compressed.to(device)
                return module.restore_grid(compressed).to(current) + correction(current, compressed)

            if not execution.checkpoint_modules or not torch.is_grad_enabled():
                delta = fuse(grid, captured)
            else:
                from torch.utils.checkpoint import checkpoint
                if execution.checkpoint_offload:
                    with torch.autograd.graph.save_on_cpu(pin_memory=grid.is_cuda):
                        delta = checkpoint(fuse, grid, captured, use_reentrant=False)
                else:
                    delta = checkpoint(fuse, grid, captured, use_reentrant=False)
            grid = grid + delta.to(grid)
        return execution.merge(hidden, prefix, grid)

    registered = False
    try:
        corrections.to(next(modules.parameters()))
        modules.add_module('_concat_probe_corrections', corrections)
        registered = True
        HybridExecution.before = before
        yield corrections
    finally:
        HybridExecution.before = original
        if registered:
            del modules._concat_probe_corrections
        _probe_lock.release()
