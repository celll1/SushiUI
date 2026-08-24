"""Shared ``debug_latents`` dump construction for the video / audio archs.

The image archs each build their own dump inline (``anima_ops``, ``zimage_ops``,
...) because a 4-D latent needs no shaping. LTX-2.3, MiniMax-H3 and ACE-Step do:
a 5-D video latent and a 2-D audio latent both have to be reduced to something
``routes.visualize_debug_latent`` can false-colour, and all three would otherwise
copy the same reduction. That reduction, the file naming and the key names live
here so they stay one decision.

The dump never decodes: ``.pt`` files carry latents, decoding is the endpoint's
job (the split every other arch already uses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

#: Keys ``routes.visualize_debug_latent`` renders. Audio streams that sit
#: alongside a video stream in one dump are the same names with this prefix.
AUDIO_KEY_PREFIX = "audio_"


def leading_window_frames(spec, t_lat: int) -> int:
    """Latent frames in the leading window this arch's VAE can decode.

    A video VAE is causal in time and compresses it 4-8x with the first latent
    frame special-cased, so no single latent frame decodes on its own: the dump
    must be a CONTIGUOUS window from the clip head, not sampled frames. Taking
    only the minimum decodable window keeps the dump's cost independent of clip
    length.
    """
    if spec is None:
        return 1
    need = int(getattr(spec, "min_decodable_frames", 1))
    pattern = tuple(getattr(spec, "latent_chunk_pattern", ()) or ())
    if pattern:
        covered = 0
        n = 0
        while covered < need:
            covered += int(pattern[n % len(pattern)])
            n += 1
    else:
        n = int(spec.latent_frames(need))
    return max(1, min(n, int(t_lat)))


def video_filmstrip(window_5d: torch.Tensor) -> torch.Tensor:
    """``[B, C, n, H, W]`` -> ``[1, C, H, n*W]`` for sample 0.

    Tiling time along width keeps the saved tensor in the ``[1, C, H, W]`` shape
    the visualize endpoint already understands, so the window needs no
    shape-aware branch there.
    """
    if window_5d.dim() != 5:
        raise ValueError(f"expected 5D [B, C, T, H, W], got {tuple(window_5d.shape)}")
    x = window_5d[0:1].detach().float().cpu()
    b, c, n, h, w = x.shape
    return x.permute(0, 1, 3, 2, 4).reshape(b, c, h, n * w).contiguous()


def audio_strip(rows_3d: torch.Tensor) -> torch.Tensor:
    """``[B, T, C]`` -> ``[1, 1, C, T]`` for sample 0.

    Latent channels become image rows and time the width, i.e. the audio latent
    is false-coloured as a spectrogram-shaped map. No vocoder is involved.
    """
    if rows_3d.dim() != 3:
        raise ValueError(f"expected 3D [B, T, C], got {tuple(rows_3d.shape)}")
    x = rows_3d[0:1].detach().float().cpu()
    return x.permute(0, 2, 1).unsqueeze(1).contiguous()


def channel_stats(x: torch.Tensor, channel_dim: int) -> Dict[str, List[float]]:
    """Per-channel mean/std — the arch-independent half of the diagnosis.

    Collapsing toward the channel mean (std -> 0) is the most common training
    failure and is visible here without looking at any image.
    """
    xf = x.detach().float()
    dims = [d for d in range(xf.dim()) if d != channel_dim]
    mean = xf.mean(dim=dims)
    std = xf.std(dim=dims) if xf.numel() > xf.shape[channel_dim] else torch.zeros_like(mean)
    return {"mean": [float(v) for v in mean.flatten()],
            "std": [float(v) for v in std.flatten()]}


def save_dump(
    debug_save_path: Path,
    *,
    timestep: float,
    model_type: str,
    video: Optional[Dict[str, torch.Tensor]] = None,
    audio: Optional[Dict[str, torch.Tensor]] = None,
    scalars: Optional[Dict[str, Any]] = None,
    captions: Optional[List[str]] = None,
    reference_image_paths: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one ``latents_t<sigma>.pt``.

    ``video`` values are ``[1, C, H, W]`` (channel dim 1, from
    :func:`video_filmstrip`); ``audio`` values are ``[1, 1, C, T]`` (channel dim
    2, from :func:`audio_strip`). Keys are the endpoint's own names.
    """
    debug_save_path.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "timestep": float(timestep),
        "model_type": model_type,
    }
    stats: Dict[str, Dict[str, List[float]]] = {}

    for key, tensor in (video or {}).items():
        data[key] = tensor
        stats[key] = channel_stats(tensor, channel_dim=1)
    for key, tensor in (audio or {}).items():
        data[key] = tensor
        stats[key] = channel_stats(tensor, channel_dim=2)

    data["channel_stats"] = stats
    data.update(scalars or {})
    data.update(extra or {})

    if captions:
        data["caption"] = captions[0]
        data["all_captions"] = list(captions)
    if reference_image_paths:
        first_ref = next((p for p in reference_image_paths if p), None)
        if first_ref:
            data["reference_image_path"] = first_ref

    out = debug_save_path / f"latents_t{float(timestep):.4f}.pt"
    torch.save(data, out)
    return out
