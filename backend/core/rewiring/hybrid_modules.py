"""Zero-output residual modules for fixed-width hybrid image streams."""
import torch
from torch import nn
from torch.nn import functional as F

from .hybrid_spec import validate_hybrid_spec, legacy_topology, hybrid_preview, skip_storage
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
        device = torch.empty(0).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Adding a topology must not advance the run's exposure RNG.
        with torch.random.fork_rng(devices=devices):
            self._build_modules()

    def _build_modules(self):
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

    def forward_stream(self, hidden, stream, height, width, prefix, context, execute_block, coarse_context, *,
                       grid_groups=1, checkpoint_modules=False, checkpoint_offload=False, allow_encoder_replay=True):
        if hidden.shape[-1] != self.declaration["stream_widths"].get(stream):
            raise ValueError("Hybrid hidden width differs from declaration")
        if type(grid_groups) is not int or grid_groups < 1 or height % grid_groups:
            raise ValueError("Hybrid grid groups must evenly divide token height")
        hybrid_preview(self.declaration, height // grid_groups, width,
                       batch_size=hidden.shape[0] * grid_groups, element_bytes=hidden.element_size())
        execution = HybridExecution(self, stream, grid_groups, checkpoint_modules, checkpoint_offload)
        if torch.is_grad_enabled() and any(item["stream"] == stream and skip_storage(item) == "recompute"
                                           for item in self.declaration["skips"]):
            if not allow_encoder_replay:
                raise ValueError("Hybrid encoder replay is incompatible with Block Swap")
            execution.configure_replay(hidden, height, width, prefix, context, execute_block, coarse_context)
        return run_rewired_stream(hidden, legacy_topology(self.declaration), stream, height, width, prefix,
            context, execute_block, coarse_context, grid_groups=grid_groups, extensions=execution)


class HybridExecution:
    """Per-forward activation ownership; module state never retains a skip graph."""
    def __init__(self, modules, stream, grid_groups=1, checkpoint_modules=False, checkpoint_offload=False,
                 capture_skips=True, capture_target=None):
        self.modules = modules
        self.stream = stream
        self.grid_groups = grid_groups
        self.checkpoint_modules = checkpoint_modules
        self.checkpoint_offload = checkpoint_offload
        self.saved = {}
        self.capture_skips = capture_skips
        self.capture_target = capture_target
        self.captured = None
        self.replay = None
        self.replay_input = None
        self.boundaries = {item["region"]: item["id"] for item in modules.declaration["boundaries"]}

    def configure_replay(self, hidden, height, width, prefix, context, execute_block, coarse_context):
        from torch.utils.checkpoint import checkpoint
        device = hidden.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if hidden.is_cuda else []
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if hidden.is_cuda else None
        topology = legacy_topology(self.modules.declaration)
        modules, stream, groups = self.modules, self.stream, self.grid_groups
        self.replay_input = hidden.to("cpu")
        cpu_input = self.replay_input

        def replay(item):
            def prefix_forward(value):
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                    # Each replay owns its dictionaries; checkpoint never mutates the live execution.
                    execution = HybridExecution(modules, stream, groups, capture_skips=False, capture_target=item)
                    run_rewired_stream(value.to(device), topology, stream, height, width,
                        prefix, context, execute_block, coarse_context, grid_groups=groups,
                        extensions=execution, stop_after=item["source_after"])
                    return execution.captured
            return checkpoint(prefix_forward, cpu_input, use_reentrant=False)
        self.replay = replay

    def call(self, function, value):
        if not self.checkpoint_modules or not torch.is_grad_enabled():
            return function(value)
        from torch.utils.checkpoint import checkpoint
        if self.checkpoint_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=value.is_cuda):
                return checkpoint(function, value, use_reentrant=False)
        return checkpoint(function, value, use_reentrant=False)

    def down(self, region, grid, pooled):
        name = self.boundaries.get(region["id"])
        return pooled if name is None else pooled + self.call(self.modules.down[name], grid).to(pooled)

    def up(self, region, update):
        name = self.boundaries.get(region["id"])
        return update if name is None else update + self.call(self.modules.up[name], update).to(update)

    def grid(self, hidden, prefix, height, width):
        batch, channels = hidden.shape[0], hidden.shape[-1]
        grid = hidden[:, prefix:].transpose(1, 2).reshape(batch, channels, self.grid_groups, height // self.grid_groups, width)
        return grid.permute(0, 2, 1, 3, 4).reshape(batch * self.grid_groups, channels, height // self.grid_groups, width)

    def merge(self, hidden, prefix, grid):
        grid = grid.reshape(hidden.shape[0], self.grid_groups, *grid.shape[1:])
        tokens = grid.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
        return torch.cat((hidden[:, :prefix], tokens.to(hidden)), dim=1)

    def before(self, index, hidden, prefix, height, width):
        items = [item for item in self.modules.declaration["skips"]
                 if item["stream"] == self.stream and item["destination_before"] == index]
        if not items:
            return hidden
        grid = self.grid(hidden, prefix, height, width)
        for item in items:
            module = self.modules.skips[item["id"]]
            device = grid.device
            # Transfer inside the checkpoint so replay does not retain a GPU copy.
            def restore(value, module=module, device=device):
                return module.restore_grid(value.to(device))
            value = self.replay(item) if self.replay is not None and skip_storage(item) == "recompute" else self.saved.pop(item["id"])
            grid = grid + self.call(restore, value).to(grid)
        return self.merge(hidden, prefix, grid)

    def after(self, index, hidden, prefix, height, width):
        refiners = [item for item in self.modules.declaration["refiners"]
                    if item["stream"] == self.stream and item["after_block"] == index]
        sources = [item for item in self.modules.declaration["skips"]
                   if self.capture_skips and item["stream"] == self.stream and item["source_after"] == index]
        target = self.capture_target if self.capture_target is not None and self.capture_target["source_after"] == index else None
        if not refiners and not sources and target is None:
            return hidden
        grid = self.grid(hidden, prefix, height, width)
        for item in refiners:
            grid = grid + self.call(self.modules.local[item["id"]], grid).to(grid)
        if target is not None:
            self.captured = self.modules.skips[target["id"]].capture_grid(grid)
        for item in sources:
            if self.replay is not None and skip_storage(item) == "recompute":
                continue
            value = self.call(self.modules.skips[item["id"]].capture_grid, grid)
            if torch.is_grad_enabled() and skip_storage(item) == "cpu":
                value = value.to("cpu")
            self.saved[item["id"]] = value
        return self.merge(hidden, prefix, grid)
