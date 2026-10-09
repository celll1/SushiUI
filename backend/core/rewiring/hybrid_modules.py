"""Zero-output residual modules for fixed-width hybrid image streams."""
import torch
from torch import nn
from torch.nn import functional as F

from .hybrid_spec import validate_hybrid_spec, legacy_topology, hybrid_preview
from .runtime import run_rewired_stream


class LocalCorrection(nn.Module):
    def __init__(self, width, rank, kernel, *, stride, normalize):
        super().__init__()
        self.norm = nn.LayerNorm(width) if normalize else None
        self.spatial = nn.Conv2d(width, width, kernel, stride=stride, padding=kernel//2,
                                 groups=width, bias=False)
        self.reduce = nn.Conv2d(width, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, width, 1, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, value):
        value = value.to(self.spatial.weight.dtype)
        if self.norm is not None:
            value = self.norm(value.movedim(1, -1)).movedim(-1, 1)
        return self.expand(F.silu(self.reduce(self.spatial(value))))


class CompressedSkip(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.capture = nn.Conv2d(width, rank, 1, bias=False)
        self.restore = nn.Conv2d(rank, width, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def capture_grid(self, value):
        return self.capture(value.to(self.capture.weight.dtype))

    def restore_grid(self, value):
        return self.restore(value.to(self.restore.weight.dtype))


class HybridModules(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        self.declaration = validate_hybrid_spec(declaration)
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        self.local = nn.ModuleDict()
        self.skips = nn.ModuleDict()
        regions = {region["id"]: region for region in self.declaration["regions"]}
        widths = self.declaration["stream_widths"]
        for item in self.declaration["boundaries"]:
            width = widths[regions[item["region"]]["stream"]]
            self.down[item["id"]] = LocalCorrection(width, item["rank"], item["kernel"], stride=2, normalize=True)
            self.up[item["id"]] = LocalCorrection(width, item["rank"], item["kernel"], stride=1, normalize=False)
        for item in self.declaration["refiners"]:
            self.local[item["id"]] = LocalCorrection(widths[item["stream"]], item["rank"], item["kernel"],
                                                       stride=1, normalize=True)
        for item in self.declaration["skips"]:
            self.skips[item["id"]] = CompressedSkip(widths[item["stream"]], item["rank"])

    def forward_stream(self, hidden, stream, height, width, prefix, context, execute_block, coarse_context):
        if hidden.shape[-1] != self.declaration["stream_widths"].get(stream):
            raise ValueError("Hybrid hidden width differs from declaration")
        hybrid_preview(self.declaration, height, width, batch_size=hidden.shape[0], element_bytes=hidden.element_size())
        execution = HybridExecution(self, stream)
        return run_rewired_stream(hidden, legacy_topology(self.declaration), stream, height, width, prefix,
            context, execute_block, coarse_context, extensions=execution)


class HybridExecution:
    """Per-forward activation ownership; module state never retains a skip graph."""
    def __init__(self, modules, stream):
        self.modules = modules
        self.stream = stream
        self.saved = {}
        self.boundaries = {item["region"]: item["id"] for item in modules.declaration["boundaries"]}

    def down(self, region, grid, pooled):
        name = self.boundaries.get(region["id"])
        return pooled if name is None else pooled + self.modules.down[name](grid).to(pooled)

    def up(self, region, update):
        name = self.boundaries.get(region["id"])
        return update if name is None else update + self.modules.up[name](update).to(update)

    @staticmethod
    def grid(hidden, prefix, height, width):
        return hidden[:, prefix:].transpose(1, 2).reshape(hidden.shape[0], hidden.shape[-1], height, width)

    @staticmethod
    def merge(hidden, prefix, grid):
        return torch.cat((hidden[:, :prefix], grid.flatten(2).transpose(1, 2).to(hidden)), dim=1)

    def before(self, index, hidden, prefix, height, width):
        items = [item for item in self.modules.declaration["skips"]
                 if item["stream"] == self.stream and item["destination_before"] == index]
        if not items:
            return hidden
        grid = self.grid(hidden, prefix, height, width)
        for item in items:
            grid = grid + self.modules.skips[item["id"]].restore_grid(self.saved.pop(item["id"])).to(grid)
        return self.merge(hidden, prefix, grid)

    def after(self, index, hidden, prefix, height, width):
        refiners = [item for item in self.modules.declaration["refiners"]
                    if item["stream"] == self.stream and item["after_block"] == index]
        sources = [item for item in self.modules.declaration["skips"]
                   if item["stream"] == self.stream and item["source_after"] == index]
        if not refiners and not sources:
            return hidden
        grid = self.grid(hidden, prefix, height, width)
        for item in refiners:
            grid = grid + self.modules.local[item["id"]](grid).to(grid)
        for item in sources:
            self.saved[item["id"]] = self.modules.skips[item["id"]].capture_grid(grid)
        return self.merge(hidden, prefix, grid)
