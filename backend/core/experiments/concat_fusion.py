"""Zero-output compressed concat correction for decoder-fusion probes."""
import torch
from torch import nn
from torch.nn import functional as F


class CompressedConcatCorrection(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if type(width) is not int or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError("Concat fusion requires integer width/rank with 1 <= rank <= width")
        self.width, self.rank = width, rank
        self.current_norm = nn.LayerNorm(width)
        self.skip_norm = nn.RMSNorm(rank)
        self.mix = nn.Conv2d(width + rank, rank, 1, bias=False)
        self.output = nn.Conv2d(rank, width, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, current, compressed):
        if (current.ndim != 4 or compressed.ndim != 4
                or current.shape[1] != self.width or compressed.shape[1] != self.rank
                or current.shape[0] != compressed.shape[0] or current.shape[2:] != compressed.shape[2:]):
            raise ValueError("Concat fusion requires matching BCHW grids and declared channels")
        dtype = self.mix.weight.dtype
        high = self.current_norm(current.to(dtype).movedim(1, -1)).movedim(-1, 1)
        skip = self.skip_norm(compressed.to(dtype).movedim(1, -1)).movedim(-1, 1)
        output = self.output(F.silu(self.mix(torch.cat((high, skip), dim=1))))
        # Restore per-token units without introducing canvas-wide partition statistics.
        scale = torch.linalg.vector_norm(current.float(), dim=1, keepdim=True) / self.width**.5
        return (output.float() * scale).to(current.dtype)


def parameter_count(width, rank):
    return 2 * width * rank + rank * rank + 2 * width + rank
