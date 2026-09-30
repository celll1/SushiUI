"""Residual stream execution with architecture-owned block and position callbacks."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .spec import preview_plan, validate_spec


def run_rewired_stream(hidden, spec, stream, height, width, prefix, context,
                       execute_block, coarse_context):
    """Callbacks own native attention, checkpointing, and block-swap execution."""
    spec = validate_spec(spec)
    preview_plan(spec, height, width)
    declaration = next((s for s in spec["streams"] if s["name"] == stream), None)
    if declaration is None:
        raise ValueError("Unknown rewired stream")
    if (hidden.ndim != 3 or type(prefix) is not int or prefix < 0
            or hidden.shape[1] != prefix + height * width):
        raise ValueError("Rewired stream requires a contiguous 2D image-token tail")
    batch, channels = hidden.shape[0], hidden.shape[2]
    regions = [r for r in spec["regions"] if r["stream"] == stream]
    starts = {r["start_block"]: r for r in regions}
    stack = []
    for index in range(declaration["block_count"]):
        while stack and index == stack[-1]["end"]:
            entry = stack.pop()
            saved, low_in = entry["saved"], entry["low_in"]
            delta = (hidden[:, prefix:].float() - low_in.float()).transpose(1, 2)
            delta = delta.reshape(saved.shape[0], saved.shape[2], height, width)
            height, width = entry["grid"]
            update = F.interpolate(delta, size=(height, width), mode="bilinear",
                                   align_corners=False).flatten(2).transpose(1, 2)
            hidden = torch.cat((hidden[:, :prefix],
                                (saved.float() + update).to(saved.dtype)), dim=1)
            context = entry["context"]
        region = starts.get(index)
        if region is not None:
            saved = hidden[:, prefix:]
            grid = saved.transpose(1, 2).reshape(saved.shape[0], saved.shape[2], height, width)
            factor = tuple(region["down_factor"])
            low_in = F.avg_pool2d(grid, factor).flatten(2).transpose(1, 2)
            stack.append(dict(end=region["end_block"], saved=saved, low_in=low_in,
                              grid=(height, width), context=context))
            height, width = height // factor[0], width // factor[1]
            hidden = torch.cat((hidden[:, :prefix], low_in), dim=1)
            context = coarse_context(context, hidden, height, width, prefix)
        hidden = execute_block(index, hidden, context)
        if hidden.shape != (batch, prefix + height * width, channels):
            raise ValueError("Block changed the declared stream token layout")
    if stack:
        raise RuntimeError("Rewired stream did not restore its original grid")
    return hidden
