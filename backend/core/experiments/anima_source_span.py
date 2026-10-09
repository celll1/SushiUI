"""Experimental full-image RoPE coordinates spanning a declared source grid."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def source_span_angles(generate_embeddings, shape, source_grid, *, fps=None):
    """Average native linear angles at cell centers; do not pool sine/cosine."""
    if len(shape) != 5 or shape[1] != 1:
        raise ValueError("Source-span Anima coordinates require one image frame")
    if len(source_grid) != 2 or any(type(n) is not int or n < 1 for n in source_grid):
        raise ValueError("Source-span coordinates require a positive source grid")
    batch, frames, height, width, channels = shape
    sh, sw = source_grid
    if height < 1 or width < 1 or sh % height or sw % width:
        raise ValueError("Source grid must be divisible by the current image grid")
    native = generate_embeddings((batch, frames, sh, sw, channels), fps=fps)
    if native.ndim != 4 or native.shape[:3] != (sh * sw, 1, 1):
        raise ValueError("Source-span coordinates require native Anima angle layout")
    if not torch.isfinite(native).all():
        raise ValueError("Source-span native angles must be finite")
    grid = native.reshape(1, sh, sw, -1).permute(0, 3, 1, 2)
    reduced = F.avg_pool2d(grid, (sh // height, sw // width))
    return reduced.permute(0, 2, 3, 1).reshape(height * width, 1, 1, -1)
