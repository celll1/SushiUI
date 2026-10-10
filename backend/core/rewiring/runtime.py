"""Residual stream execution with architecture-owned block and position callbacks."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import preview_plan, validate_spec


def run_rewired_stream(hidden, spec, stream, height, width, prefix, context,
                       execute_block, coarse_context, *, grid_groups=1, extensions=None, stop_after=None):
    """Callbacks own native attention, checkpointing, and block-swap execution."""
    spec = validate_spec(spec)
    if spec["method"] != "residual_delta_unet":
        raise ValueError("Hybrid topology must execute through its installed learned modules")
    if type(height) is not int or type(grid_groups) is not int or grid_groups < 1 or height % grid_groups:
        raise ValueError("Rewired grid groups must evenly divide the token height")
    preview_plan(spec, height // grid_groups, width)
    declaration = next((s for s in spec["streams"] if s["name"] == stream), None)
    if declaration is None:
        raise ValueError("Unknown rewired stream")
    if stop_after is not None and (type(stop_after) is not int or not 0 <= stop_after < declaration["block_count"]):
        raise ValueError("Invalid encoder replay stop block")
    if (hidden.ndim != 3 or type(prefix) is not int or prefix < 0
            or hidden.shape[1] != prefix + height * width):
        raise ValueError("Rewired stream requires a contiguous token-grid tail")
    batch, channels = hidden.shape[0], hidden.shape[2]
    regions = [r for r in spec["regions"] if r["stream"] == stream]
    starts = {r["start_block"]: r for r in regions}
    stack = []
    for index in range(declaration["block_count"]):
        while stack and index == stack[-1]["end"]:
            entry = stack.pop()
            saved, low_in = entry["saved"], entry["low_in"]
            delta = (hidden[:, prefix:].float() - low_in.float()).transpose(1, 2)
            delta = delta.reshape(batch, channels, grid_groups, height // grid_groups, width)
            delta = delta.permute(0, 2, 1, 3, 4).reshape(batch * grid_groups, channels, height // grid_groups, width)
            height, width = entry["grid"]
            update = F.interpolate(delta, size=(height // grid_groups, width), mode="bilinear", align_corners=False)
            if extensions is not None:
                update = extensions.up(entry["region"], update)
                phase_up = getattr(extensions, "up_from_coarse", None)
                if phase_up is not None:
                    correction = phase_up(entry["region"], delta, update)
                    if correction is not None:
                        if correction.shape != update.shape:
                            raise ValueError("Coarse up correction changed the restored grid layout")
                        update = update + correction.to(update)
                fusion = getattr(extensions, "boundary_fusion", None)
                if fusion is not None:
                    correction = fusion(entry["region"], saved, low_in, update, height, width)
                    if correction is not None:
                        if correction.shape != update.shape:
                            raise ValueError("Boundary fusion changed the restored grid layout")
                        update = update + correction.to(update)
                feature_fusion = getattr(extensions, "boundary_features", None)
                if feature_fusion is not None:
                    correction = feature_fusion(entry["region"], saved, low_in,
                        hidden[:, prefix:], update, height, width)
                    if correction is not None:
                        if correction.shape != update.shape:
                            raise ValueError("Boundary feature fusion changed the restored grid layout")
                        update = update + correction.to(update)
            update = update.reshape(batch, grid_groups, channels, height // grid_groups, width)
            update = update.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :prefix],
                                (saved.float() + update).to(saved.dtype)), dim=1)
            context = entry["context"]
        region = starts.get(index)
        if region is not None:
            saved = hidden[:, prefix:]
            grid = saved.transpose(1, 2).reshape(batch, channels, grid_groups, height // grid_groups, width)
            grid = grid.permute(0, 2, 1, 3, 4).reshape(batch * grid_groups, channels, height // grid_groups, width)
            factor = tuple(region["down_factor"])
            pooled = F.avg_pool2d(grid, factor)
            if region["down"] == "rms_pool":
                source_energy = F.avg_pool2d(grid.float().square(), factor).mean(1, keepdim=True)
                pooled_energy = pooled.float().square().mean(1, keepdim=True)
                gain = (source_energy.clamp_min(1e-12) / pooled_energy.clamp_min(1e-12)).sqrt()
                gain = torch.where(pooled_energy > 1e-12, gain, torch.ones_like(gain))
                pooled = (pooled.float() * gain).to(pooled.dtype)
            if extensions is not None:
                pooled = extensions.down(region, grid, pooled)
            low_in = pooled.reshape(batch, grid_groups, channels, height // grid_groups // factor[0], width // factor[1])
            low_in = low_in.permute(0, 2, 1, 3, 4).flatten(2).transpose(1, 2)
            stack.append(dict(end=region["end_block"], saved=saved, low_in=low_in,
                              grid=(height, width), context=context, region=region))
            height, width = height // factor[0], width // factor[1]
            hidden = torch.cat((hidden[:, :prefix], low_in), dim=1)
            context = coarse_context(context, hidden, height, width, prefix)
        if extensions is not None:
            hidden = extensions.before(index, hidden, prefix, height, width)
        hidden = execute_block(index, hidden, context)
        if hidden.shape != (batch, prefix + height * width, channels):
            raise ValueError("Block changed the declared stream token layout")
        if extensions is not None:
            hidden = extensions.after(index, hidden, prefix, height, width)
        if index == stop_after:
            return hidden
    if stack:
        raise RuntimeError("Rewired stream did not restore its original grid")
    return hidden
