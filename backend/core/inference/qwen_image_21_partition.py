"""Qwen partition geometry and generation-scoped velocity prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import math

import torch

from api.param_defaults import GENERATION_DEFAULTS

MAX_REGIONS = 64
MAX_LAYOUT_DEPTH = 8


def normalize_partition_config(value, *, defaults=None):
    defaults = GENERATION_DEFAULTS["qwen_image_21_partition"] if defaults is None else defaults
    if value is None or value == "":
        return copy.deepcopy(defaults)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("qwen_image_21_partition must be a JSON object") from exc
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("qwen_image_21_partition has unsupported fields")
    settings = {**defaults, **value}
    for key in ("version", "max_target_tokens", "halo_tokens"):
        if type(settings[key]) is not int:
            raise ValueError(f"qwen_image_21_partition.{key} must be an integer")
    if (settings["version"] != 1 or not isinstance(settings["mode"], str)
            or settings["mode"] not in {"off", "auto", "manual"}):
        raise ValueError("Unsupported Qwen partition version or mode")
    if settings["max_target_tokens"] < 1:
        raise ValueError("Qwen partition token ceiling must be positive")
    if settings["halo_tokens"] < 0 or settings["halo_tokens"] % 2:
        raise ValueError("Qwen partition halo must be nonnegative and even")
    nodes = 0

    def tree(node, depth):
        nonlocal nodes
        if node is None:
            return None
        nodes += 1
        if depth > MAX_LAYOUT_DEPTH or nodes >= MAX_REGIONS:
            raise ValueError("Qwen partition layout exceeds the region/depth limit")
        if not isinstance(node, dict) or set(node) != {"axis", "ratio", "first", "second"}:
            raise ValueError("Qwen partition split requires axis, ratio, first and second")
        ratio = node["ratio"]
        if (not isinstance(node["axis"], str) or node["axis"] not in {"x", "y"} or type(ratio) not in (int, float)
                or not math.isfinite(ratio) or not 0 < ratio < 1):
            raise ValueError("Qwen partition split needs a finite ratio strictly between 0 and 1")
        return {"axis": node["axis"], "ratio": ratio,
                "first": tree(node["first"], depth + 1),
                "second": tree(node["second"], depth + 1)}

    settings["layout"] = tree(settings["layout"], 0)
    return settings


@dataclass(frozen=True)
class Bounds:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def width(self):
        return self.right - self.left

    @property
    def tokens(self):
        return self.height * self.width

    def expanded(self, halo, height, width):
        return Bounds(max(0, self.top - halo), max(0, self.left - halo),
                      min(height, self.bottom + halo), min(width, self.right + halo))


@dataclass(frozen=True)
class GenerationPlan:
    height: int
    width: int
    cores: tuple[Bounds, ...]
    inputs: tuple[Bounds, ...]
    mode: str

    def metadata(self):
        return {"version": 1, "mode": self.mode, "latent_height": self.height,
                "latent_width": self.width,
                "regions": [{"core": asdict(core), "input": asdict(box)}
                            for core, box in zip(self.cores, self.inputs)]}


def resolve_partition_plan(value, height, width, *, defaults=None):
    settings = normalize_partition_config(value, defaults=defaults)
    if settings["mode"] == "off":
        return None
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("Qwen partition canvas requires positive even latent sides")
    full = Bounds(0, 0, height, width)
    halo = settings["halo_tokens"]

    def split(box, axis, ratio):
        length = box.width if axis == "x" else box.height
        cut = int(math.floor(length * ratio / 2 + 0.5)) * 2
        if cut < 2 or cut > length - 2:
            raise ValueError("Qwen partition boundary collapses after 32-pixel snapping")
        if axis == "x":
            cut += box.left
            return (Bounds(box.top, box.left, box.bottom, cut),
                    Bounds(box.top, cut, box.bottom, box.right))
        cut += box.top
        return (Bounds(box.top, box.left, cut, box.right),
                Bounds(cut, box.left, box.bottom, box.right))

    if settings["mode"] == "manual":
        cores = []

        def visit(box, node):
            if node is None:
                cores.append(box)
                return
            first, second = split(box, node["axis"], node["ratio"])
            visit(first, node["first"])
            visit(second, node["second"])

        visit(full, settings["layout"])
    else:
        cores = [full]
        limit = settings["max_target_tokens"]
        while True:
            index = max(range(len(cores)), key=lambda i: cores[i].expanded(halo, height, width).tokens)
            box = cores[index]
            if box.expanded(halo, height, width).tokens <= limit:
                break
            if len(cores) >= MAX_REGIONS:
                raise ValueError("Qwen Auto cannot satisfy the token ceiling within 64 regions")
            axes = ("x", "y") if box.width >= box.height else ("y", "x")
            for axis in axes:
                try:
                    pair = split(box, axis, 0.5)
                except ValueError:
                    continue
                cores[index:index + 1] = pair
                break
            else:
                raise ValueError("Qwen Auto token ceiling is too small for the halo/minimum region")
    if len(cores) == 1:
        return None
    return GenerationPlan(height, width, tuple(cores),
                          tuple(box.expanded(halo, height, width) for box in cores), settings["mode"])


class PartitionExecutor:
    """No transformer hooks or module-tree changes; release CUDA copies in close()."""

    def __init__(self, model, plan, payloads=(), *, step_sources=(), use_cache=True, cancelled=lambda: False):
        self.model, self.plan = model, plan
        self.payloads = list(payloads)
        self.use_cache = bool(use_cache and model.config.causal_condition)
        self.cancelled = cancelled
        self.modules = {}
        self.step_sources = list(step_sources)
        self.caches = {}
        self.cache_inputs = {}
        self.step, self.steps = 0, 1
        self.signature = None

    def set_step(self, index, total):
        self.step, self.steps = index, total
        signature = tuple(self._strength(source) != 0 for source in self.step_sources)
        if signature != self.signature:
            self.caches.clear()
            self.cache_inputs.clear()
            self.signature = signature

    def _strength(self, source):
        if source is None:
            return 1.0
        if not source.apply_to_unet:
            return 0.0
        if source.step_range is not None:
            start, end = (int(value / 1000 * self.steps) for value in source.step_range)
            if not start <= self.step <= end:
                return 0.0
        return float(source.strength)

    def _prepare_modules(self, grid):
        from core.models.qwen_image_21.partition_global import QwenPartitionGlobalAdapter
        for index, (payload, source) in enumerate(self.payloads):
            if index in self.modules or not self._strength(source):
                continue
            metadata = payload["metadata"]
            with torch.device("meta"):
                module = QwenPartitionGlobalAdapter(
                    int(self.model.config.in_channels), int(self.model.inner_dim),
                    rank=int(metadata["qwen_partition_global_rank"]),
                    summary_tokens=int(metadata["qwen_partition_global_tokens"]), dtype=grid.dtype,
                )
            module.to_empty(device=grid.device).eval().requires_grad_(False)
            module.load_tensors(payload["state"])
            self.modules[index] = (module, source)

    def predict(self, *, branch, hidden_states, img_shapes, img_mask, kv_cache=None,
                kv_cache_mode=None, **kwargs):
        plan = self.plan
        target_count = plan.height * plan.width
        prefix = hidden_states[:, :-target_count]
        grid = hidden_states[:, -target_count:].reshape(hidden_states.shape[0], plan.height, plan.width, -1)
        self._prepare_modules(grid)
        prediction = torch.empty_like(grid)
        def identity(tensor):
            if tensor is None:
                return None
            try:
                version = tensor._version
            except RuntimeError:
                version = None
            return (tensor.data_ptr(), version, tuple(tensor.shape), tensor.dtype, tensor.device)

        conditioning = (identity(kwargs.get("encoder_hidden_states")),
                        identity(kwargs.get("encoder_hidden_states_mask")))
        if self.cache_inputs.get(branch) != conditioning:
            self.caches.pop(branch, None)
            self.cache_inputs[branch] = conditioning
        cache = self.caches.get(branch)
        filled = cache is not None
        if self.use_cache and cache is None:
            from core.models.qwen_image_21.vendor.transformer import QwenImage21KVCache
            cache = QwenImage21KVCache(len(self.model.transformer_blocks))
            self.caches[branch] = cache
        for core, box in zip(plan.cores, plan.inputs):
            if self.cancelled():
                raise RuntimeError("Generation cancelled between Qwen partitions")
            tile = grid[:, box.top:box.bottom, box.left:box.right].reshape(grid.shape[0], box.tokens, -1)
            states = torch.cat([prefix, tile], dim=1) if prefix.shape[1] else tile
            residual = None
            for module, source in self.modules.values():
                strength = self._strength(source)
                if strength:
                    value = module(grid, box) * strength
                    residual = value if residual is None else residual + value
            if residual is not None and prefix.shape[1]:
                residual = torch.cat([residual.new_zeros(grid.shape[0], prefix.shape[1], residual.shape[-1]), residual], dim=1)
            y = torch.arange(box.top, box.bottom, device=grid.device) - (plan.height - plan.height // 2)
            x = torch.arange(box.left, box.right, device=grid.device) - (plan.width - plan.width // 2)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            positions = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
            tile_mask = torch.cat([img_mask[:, :-target_count // 4],
                                   img_mask.new_ones(img_mask.shape[0], box.tokens // 4)], dim=1)
            result = self.model(
                hidden_states=states, img_shapes=[[*shapes[:-1], (1, box.height, box.width)] for shapes in img_shapes],
                img_mask=tile_mask, target_spatial_position_ids=positions,
                target_input_residual=residual, kv_cache=cache,
                kv_cache_mode=("cached" if filled else "extract") if cache is not None else None,
                **kwargs,
            )[0][:, -box.tokens:].reshape(grid.shape[0], box.height, box.width, -1)
            prediction[:, core.top:core.bottom, core.left:core.right] = result[
                :, core.top - box.top:core.bottom - box.top, core.left - box.left:core.right - box.left]
            filled = True
        return prediction.reshape(grid.shape[0], target_count, -1)

    def close(self):
        self.caches.clear()
        self.cache_inputs.clear()
        self.modules.clear()
        self.payloads.clear()
        self.step_sources.clear()
