"""Packed-row transport for MiniMax-H3's frame-local residual video stream."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedVideoLayout:
    order: torch.Tensor
    inverse: torch.Tensor
    prefix: int
    frames: int
    height: int
    width: int
    context: dict

    def pack(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] != self.order.numel():
            raise ValueError("MiniMax-H3 hidden state disagrees with packed layout")
        return hidden.index_select(1, self.order)

    def unpack(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] != self.order.numel():
            raise ValueError("MiniMax-H3 residual stream has not restored its native rows")
        return hidden.index_select(1, self.inverse)


def prepare_layout(position_ids, token_tags, timestep_indices, target_indices, *, frames, height, width):
    """The caller supplies target rows in frame-major order, excluding reference banks."""
    if any(type(n) is not int or n < 1 for n in (frames, height, width)):
        raise ValueError("MiniMax-H3 target geometry must contain positive integers")
    if position_ids.ndim != 2 or position_ids.shape[1] != 3 or not torch.isfinite(position_ids).all():
        raise ValueError("MiniMax-H3 positions require finite native (time, height, width) coordinates")
    total = position_ids.shape[0]
    if any(value.shape != (total,) or value.dtype != torch.long
           for value in (token_tags, timestep_indices)):
        raise ValueError("MiniMax-H3 modality and timestep labels must address every packed row")
    if (target_indices.ndim != 1 or target_indices.dtype != torch.long
            or target_indices.numel() != frames * height * width
            or target_indices.unique().numel() != target_indices.numel()
            or (target_indices < 0).any() or (target_indices >= total).any()):
        raise ValueError("MiniMax-H3 target indices must identify one complete frame-major video grid")
    if any(value.device != position_ids.device for value in (token_tags, timestep_indices, target_indices)):
        raise ValueError("MiniMax-H3 layout tensors must share a device")
    if (timestep_indices < 0).any() or ((token_tags < 0) | (token_tags > 2)).any():
        raise ValueError("MiniMax-H3 packed AdaLN labels are invalid")
    if (token_tags.index_select(0, target_indices) != 0).any():
        raise ValueError("MiniMax-H3 target rows must carry the native video modality")
    grid = position_ids.index_select(0, target_indices).reshape(frames, height, width, 3)
    time = grid[:, 0, 0, 0]
    ys, xs = grid[0, :, 0, 1], grid[0, 0, :, 2]
    if (not torch.equal(grid[..., 0], time[:, None, None].expand(frames, height, width))
            or not torch.equal(grid[..., 1], ys[None, :, None].expand(frames, height, width))
            or not torch.equal(grid[..., 2], xs[None, None, :].expand(frames, height, width))
            or (frames > 1 and not (time[1:] > time[:-1]).all())
            or (height > 1 and not (ys[1:] > ys[:-1]).all())
            or (width > 1 and not (xs[1:] > xs[:-1]).all())):
        raise ValueError("MiniMax-H3 target positions must form a chronological rectangular lattice")
    mask = torch.ones(total, dtype=torch.bool, device=position_ids.device)
    mask[target_indices] = False
    order = torch.cat((torch.arange(total, device=position_ids.device)[mask], target_indices))
    context = dict(position_ids=position_ids.index_select(0, order),
                   token_tags=token_tags.index_select(0, order),
                   timestep_indices=timestep_indices.index_select(0, order),
                   frames=frames, height=height, width=width)
    return PackedVideoLayout(order, order.argsort(), total - target_indices.numel(),
                             frames, height, width, context)


def coarse_context(context, hidden, flat_height, width, prefix):
    """Keep non-target rows native and refuse pooling across different noise labels."""
    frames, old_h, old_w = context['frames'], context['height'], context['width']
    if flat_height % frames:
        raise ValueError("MiniMax-H3 residual transport cannot shrink time")
    height = flat_height // frames
    if height < 1 or width < 1 or old_h % height or old_w % width:
        raise ValueError("MiniMax-H3 residual transport requires integral spatial factors")
    if hidden.ndim != 3 or hidden.shape[1] != prefix + frames * height * width:
        raise ValueError("MiniMax-H3 coarse hidden state disagrees with frame geometry")
    fh, fw = old_h // height, old_w // width
    result = dict(frames=frames, height=height, width=width)
    for name in ('token_tags', 'timestep_indices'):
        value = context[name]
        cells = value[prefix:].reshape(frames, height, fh, width, fw).permute(0, 1, 3, 2, 4)
        label = cells[..., 0, 0]
        if not torch.equal(cells, label[..., None, None].expand_as(cells)):
            raise ValueError("MiniMax-H3 spatial pooling cannot mix pinned and denoised AdaLN labels")
        result[name] = torch.cat((value[:prefix], label.flatten()))
    positions = context['position_ids']
    grid = positions[prefix:].reshape(frames, old_h, old_w, 3).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, (fh, fw)).permute(0, 2, 3, 1)
    # Pool only the spatial axes; keep the native media clock exactly.
    pooled[..., 0] = grid[:, 0, 0, 0][:, None, None]
    result['position_ids'] = torch.cat((positions[:prefix], pooled.reshape(-1, 3)))
    return result
