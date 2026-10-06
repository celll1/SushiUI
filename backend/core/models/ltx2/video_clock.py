"""Physical video time coordinates for target-native LTX codec grids."""

import torch


def video_pixel_frames(vae, latent_frames):
    contract = getattr(vae, "contract", None)
    if contract is not None:
        return contract.pixel_frames(latent_frames)
    from core.models.components.wiring import LTX2_WIRING
    temporal = getattr(vae, "temporal_compression_ratio", LTX2_WIRING.vae_scale_temporal)
    return (latent_frames - 1) * int(temporal) + 1


def prepare_video_coords(rope, contract, batch, frames, height, width, device, *, fps):
    coords = rope.prepare_video_coords(batch, frames, height, width, device, fps=1.0)
    if contract is not None:
        spans = contract.frame_spans(frames)
        patch_t, patch = rope.patch_size_t, rope.patch_size
        if frames % patch_t or height % patch or width % patch:
            raise ValueError("LTX codec grid must contain whole spatiotemporal patches")
        bounds = [(spans[i][0], spans[i+patch_t-1][1]) for i in range(0, frames, patch_t)]
        temporal = torch.tensor(bounds, device=device, dtype=coords.dtype)
        temporal = temporal.repeat_interleave((height//patch)*(width//patch), dim=0)
        coords[:, 0] = temporal
    rates = torch.as_tensor(fps, device=device, dtype=coords.dtype)
    if rates.ndim == 0:
        rates = rates.expand(batch)
    if rates.shape != (batch,) or not bool(torch.isfinite(rates).all()) or bool((rates <= 0).any()):
        raise ValueError("LTX video coordinates require positive finite per-item FPS")
    coords[:, 0] /= rates[:, None, None]
    return coords


def calibration_clip_frames(source_temporal, target):
    """Choose the shortest moving clip accepted by both frame lattices."""
    from math import lcm
    if type(source_temporal) is not int or source_temporal < 1:
        raise ValueError("Source video temporal compression is missing")
    period = 17 if target.temporal_policy == "minimax_h3_chunks_v1" else target.temporal_scale
    floor = 22 if target.temporal_policy == "minimax_h3_chunks_v1" else 2
    for frames in range(floor, floor+lcm(source_temporal, period)+1):
        if (frames-1) % source_temporal:
            continue
        try:
            target.latent_frames(frames)
        except ValueError:
            continue
        return frames
    raise ValueError("Source and target video frame lattices have no common moving clip")


def condition_clip_frames(contract, source_frames, available_frames):
    limit = min(source_frames, available_frames)
    if limit < 1:
        return 0
    if contract.temporal_policy == "minimax_h3_chunks_v1":
        if source_frames > 1 and limit < 22:
            raise ValueError("The remaining target timeline cannot hold an H3 moving condition clip")
        return 1 if limit < 22 else ((limit-5)//17)*17+5
    return ((limit-1)//contract.temporal_scale)*contract.temporal_scale+1
